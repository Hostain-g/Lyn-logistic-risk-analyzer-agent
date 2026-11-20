# %% [markdown]
# # Agentic Risk Analyzer: LangGraph + RAG over construction meeting-minute PDFs
# 10 nodes: intake -> PDF extraction/OCR -> metadata parsing -> chunk+index -> RAG retrieval ->
# risk extraction (LLM) -> classification -> historical matching -> quality check (loops back) -> report.
# Runs end-to-end with LLM_PROVIDER="mock" (rule-based, no API key). Switch to "anthropic"/"openai"/"ollama" for the real LLM.

# %%
# %pip install -q langgraph pydantic pdfplumber pypdfium2 pillow pytesseract scikit-learn pandas matplotlib reportlab numpy
# Pick ONE LLM SDK if you use a real model:  %pip install -q anthropic   |   %pip install -q openai
# OCR also needs the Tesseract binary: Windows https://github.com/UB-Mannheim/tesseract/wiki | macOS: brew install tesseract | Colab/Linux: apt-get install tesseract-ocr

# %%
import os, re, json, glob, time, hashlib, difflib, operator, copy
from pathlib import Path
from datetime import datetime
from typing import TypedDict, List, Dict, Optional, Annotated, Literal
import numpy as np
import pandas as pd
from pydantic import BaseModel
from langgraph.graph import StateGraph, START, END

CFG = dict(
    PDF_DIR="synthetic_minutes/pdfs",          # folder with the PDFs (searched recursively)
    GT_DIR="synthetic_minutes/ground_truth",   # optional: ground-truth JSON for evaluate()
    OUT_DIR="agent_output",
    MAX_DOCS=30,                               # set None to process everything
    LLM_PROVIDER="mock",                       # "mock" | "anthropic" | "openai" | "ollama"
    LLM_MODEL="claude-sonnet-5-5",             # e.g. "gpt-4o-mini" for openai, "llama3.1:8b" for ollama
    EMBEDDER="tfidf",                          # "tfidf" (offline) | "sentence-transformers"
    ST_MODEL="all-MiniLM-L6-v2",
    MATCH_THRESHOLD=0.07,                      # historical-match similarity; lower = merges more (use ~0.55 with sentence-transformers)
    CONF_MIN=0.6, MAX_RETRIES=1, MIN_TEXT_CHARS=50, MAX_FILE_MB=50, TOP_K=4,
    USE_EXISTING_REGISTER=False,               # True = continue from OUT_DIR/risk_register.json
)
# For a real LLM set the key first, e.g.  os.environ["ANTHROPIC_API_KEY"] = "sk-..."   (or OPENAI_API_KEY)
Path(CFG["OUT_DIR"]).mkdir(exist_ok=True)

CATEGORIES = ["Cost", "Schedule", "Supply Chain", "Technical", "Resource", "Regulatory",
              "Contractual", "Quality", "Safety", "Environmental", "Stakeholder"]
STAGES = ["raised", "escalated", "mitigating", "closed", "open"]
RUBRIC = """Likelihood (1-5): 1 rare; 2 unlikely; 3 possible; 4 likely / already showing signs; 5 almost certain or occurring now.
Impact (1-5): 1 negligible; 2 minor (<0.5% contract value or <2 days); 3 moderate (0.5-1.5% or 2-5 days);
4 major (1.5-3% or 5-10 days, or regulatory/quality consequences); 5 severe (>3%, >10 days, safety/legal consequences).
Score = Likelihood x Impact. Red >= 15, Amber 8-14, Green < 8. Closed risks get Likelihood 1."""

def rag_of(score): return "Red" if score >= 15 else "Amber" if score >= 8 else "Green"

# %% [markdown]
# ## LLM wrapper (provider-agnostic) + schema

# %%
class RiskCandidate(BaseModel):
    title: str
    category: str = "Schedule"
    description: str = ""
    signals: List[str] = []
    cause: str = ""
    consequence: str = ""
    owner: Optional[str] = None
    due_date: Optional[str] = None
    actions: List[str] = []
    stage: str = "open"
    evidence_quote: str = ""
    source_section: str = ""
    page: Optional[int] = None
    confidence: float = 0.7

_client = {}
def _raw_llm(system, user, max_tokens):
    p, model = CFG["LLM_PROVIDER"], CFG["LLM_MODEL"]
    if p == "anthropic":
        import anthropic
        c = _client.setdefault("a", anthropic.Anthropic())
        r = c.messages.create(model=model, max_tokens=max_tokens, system=system, messages=[{"role": "user", "content": user}])
        return "".join(b.text for b in r.content if b.type == "text")
    if p == "openai":
        from openai import OpenAI
        c = _client.setdefault("o", OpenAI())
        r = c.chat.completions.create(model=model, response_format={"type": "json_object"},
                                      messages=[{"role": "system", "content": system}, {"role": "user", "content": user}])
        return r.choices[0].message.content
    if p == "ollama":
        import requests
        r = requests.post("http://localhost:11434/api/chat", timeout=300, json=dict(
            model=model, stream=False, format="json",
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}]))
        return r.json()["message"]["content"]
    raise ValueError(p)

def _parse_json(txt):
    txt = re.sub(r"^```(?:json)?|```$", "", txt.strip(), flags=re.M).strip()
    a, b = txt.find("{"), txt.rfind("}")
    return json.loads(txt[a:b + 1])

def llm_json(system, user, max_tokens=4000):
    """Returns a parsed dict, or None (mock provider or repeated failure) so callers use the rule-based fallback."""
    if CFG["LLM_PROVIDER"] == "mock": return None
    err = None
    for _ in range(2):
        try: return _parse_json(_raw_llm(system, user, max_tokens))
        except Exception as e: err = e; time.sleep(1)
    print("LLM call failed, using rule-based fallback:", err)
    return None

def llm_text(system, user, max_tokens=1200):
    if CFG["LLM_PROVIDER"] == "mock": return None
    try: return _raw_llm(system, user, max_tokens)
    except Exception as e: print("LLM error:", e); return None

# %% [markdown]
# ## Vector store (node 4/5): chunk metadata + embeddings, with metadata filtering

# %%
class VectorStore:
    def __init__(self, kind="tfidf"):
        self.kind, self.chunks, self.ids, self.matrix, self.vec, self.st = kind, [], set(), None, None, None
    def _embed_st(self, texts):
        if self.st is None:
            from sentence_transformers import SentenceTransformer
            self.st = SentenceTransformer(CFG["ST_MODEL"])
        return self.st.encode(texts, normalize_embeddings=True, show_progress_bar=False)
    def add(self, chunks):
        new = [c for c in chunks if c["id"] not in self.ids]
        self.chunks += new; self.ids |= {c["id"] for c in new}
        texts = [c["text"] for c in self.chunks]
        if self.kind == "tfidf":
            from sklearn.feature_extraction.text import TfidfVectorizer
            self.vec = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, stop_words="english")
            self.matrix = self.vec.fit_transform(texts)
        else:
            self.matrix = self._embed_st(texts)
    def embed(self, texts):
        return self.vec.transform(texts) if self.kind == "tfidf" else self._embed_st(texts)
    def sims(self, a, b):
        """cosine similarity matrix between texts a and b"""
        A, B = self.embed(a), self.embed(b)
        M = A @ B.T
        return M.toarray() if hasattr(M, "toarray") else np.asarray(M)
    def search(self, query, k=5, where=None):
        idx = [i for i, c in enumerate(self.chunks) if where is None or where(c["meta"])]
        if not idx: return []
        q = self.embed([query])
        s = self.matrix[idx] @ q.T
        s = (s.toarray() if hasattr(s, "toarray") else np.asarray(s)).ravel()
        top = np.argsort(-s)[:k]
        return [dict(self.chunks[idx[i]], score=float(s[i])) for i in top if s[i] > 0]

STORE = VectorStore(CFG["EMBEDDER"])

# %% [markdown]
# ## Helpers: section parsing, risk-bullet parsing, numeric checks

# %%
SECTION_MAP = [("ATTENDEES", "attendees"), ("PREVIOUS ACTION", "previous_actions"), ("PROGRESS", "progress"),
               ("MATERIAL", "materials"), ("CHANGE ORDER", "change_orders"), ("PAYMENT", "payments"),
               ("ACTION ITEMS", "new_actions"), ("KEY RISK", "key_risks"), ("NEXT MEETING", "next_meeting")]
HEADING = re.compile(r"^\s*(\d{1,2})\.\s+([A-Z][A-Z0-9 &/\-()]{3,})\s*$")
BULLET = re.compile(r"^\s*(?:\(cid:\d+\)|[^\w\s(]|[eo](?=\s))?\s*([A-Z][A-Za-z ]{2,22}?)(?:\s+Risk)?\s*(?:\((Red|Amber|Green|Closed)\))?\s*:\s+\S")
LABELS = {c.lower(): c for c in CATEGORIES}; LABELS["schedule status"] = "Schedule Status"

def canon_section(name):
    for key, c in SECTION_MAP:
        if key in name.upper(): return c
    return "other"

def parse_date(s):
    for f in ("%d %B %Y", "%d %b %Y"):
        try: return datetime.strptime(s.strip().replace("  ", " "), f).date().isoformat()
        except Exception: pass
    return None

def split_risk_bullets(text):
    out, cur = [], None
    for line in text.splitlines():
        m = BULLET.match(line)
        if m and m.group(1).strip().lower() in LABELS:
            if cur: out.append(cur)
            cur = dict(label=LABELS[m.group(1).strip().lower()], rag=m.group(2), body=re.sub(r"^[^:]*:\s*", "", line, count=1).strip())
        elif cur and line.strip():
            cur["body"] += " " + line.strip()
    if cur: out.append(cur)
    return out

def norm(s): return re.sub(r"[^a-z0-9% ]+", " ", re.sub(r"\s+", " ", s.lower())).strip()

def cover(a, b):
    """fraction of text a that appears as one contiguous block in text b (0..1)"""
    a, b = norm(a), norm(b)
    if not a or not b: return 0.0
    m = difflib.SequenceMatcher(None, a, b, autojunk=False).find_longest_match(0, len(a), 0, len(b))
    return m.size / len(a)

def _num(s): return float(s.replace(",", ""))

def numeric_checks(rec):
    """Cross-section consistency checks. Returns a list of issue dicts."""
    T, S, issues = rec["full_text"], rec["sections"], []
    sec = lambda c: "\n".join(s["text"] for s in S if s["canonical"] == c)
    # 1) days behind must be the same everywhere
    vals = sorted({int(v) for v in re.findall(r"(\d+)\s+(?:working\s+)?days?\s+behind", T, re.I)})
    if len(vals) > 1:
        issues.append(dict(kind="days_behind", detail=f"Schedule delay stated as different values: {vals}"))
    # 2) cumulative change % vs change-order amounts
    m = re.search(r"([\d.]+)%\s+of\s+(?:the\s+)?original contract value\s*\(\$([\d,.]+)\s*(M|million)?\)", T, re.I)
    co = sec("change_orders")
    if m and co:
        contract = _num(m.group(2)) * (1e6 if m.group(3) else 1)
        parts = re.split(r"(?=\b(?:P?CO)-\d+)", co)
        amts = [_num(a) for p in parts if p.startswith(("CO-", "PCO-")) and "rejected" not in p.lower()
                for a in re.findall(r"\+\$([\d,]+)", p)[:1]]
        if amts and contract:
            computed, stated = 100 * sum(amts) / contract, float(m.group(1))
            if abs(computed - stated) > 0.3:
                issues.append(dict(kind="cumulative_pct", detail=f"Stated {stated:.1f}% of contract but change orders sum to {computed:.1f}%"))
    # 3) stated remaining material vs materials table
    mat = sec("materials")
    pats = [r"approximately\s+([\d,]+)\s+(\S+)\s+of\s+(.+?)\s+remaining", r"remaining\s+([\d,]+)\s+(?:tonnes|t|units|m)\s+of\s+([A-Za-z ]+?)(?:\s+from|\.|,|$)"]
    for i, pat in enumerate(pats):
        for g in re.finditer(pat, T, re.I):
            stated, name = _num(g.group(1)), (g.group(3) if i == 0 else g.group(2)).strip().lower()
            for line in mat.splitlines():
                l = re.sub(r"\bm(?:[³²]|[23](?![\d,]))", "m", line)
                key = name if i == 0 else name.split()[0]
                if key in l.lower():
                    rest = l[l.lower().find(key) + len(key):]
                    nums = [_num(x) for x in re.findall(r"(\d[\d,]*)\s*(?:tonnes|units|t|m)\b", re.sub(r"\([^)]*\)", "", rest))]  # quantity columns only
                    if len(nums) >= 3 and abs(nums[2] - stated) / max(nums[2], 1) > 0.01:
                        issues.append(dict(kind="material_qty", detail=f"Text says {stated:,.0f} remaining of '{name}', table says {nums[2]:,.0f}"))
                    break
    if rec.get("ocr"):
        for i in issues: i["detail"] += " [from OCR text - verify against the PDF]"
    return issues

# %% [markdown]
# ## State + Nodes 1-4: intake, PDF extraction/OCR, metadata parsing, chunking/indexing

# %%
class S(TypedDict, total=False):
    files: List[str]; docs: List[dict]; rejected: List[dict]
    records: Dict[str, dict]; evidence: Dict[str, List[dict]]; history: Dict[str, List[dict]]
    todo: List[str]; redo: List[str]; feedback: Dict[str, str]; retries: Dict[str, int]
    candidates: Dict[str, List[dict]]; assessed: Dict[str, List[dict]]
    register_base: Dict[str, list]; register: Dict[str, list]
    validated: Dict[str, List[dict]]; doc_issues: Dict[str, List[dict]]
    report: dict; log: Annotated[List[str], operator.add]

PID = r"([A-Z]{2,4}-\d{4}-\d{3})"

def node_intake(state: S):
    docs, rejected, seen = [], [], {}
    for f in state["files"]:
        p, why = Path(f), None
        if p.suffix.lower() != ".pdf": why = "not a PDF"
        elif not p.exists() or p.stat().st_size == 0: why = "missing or empty"
        elif p.stat().st_size > CFG["MAX_FILE_MB"] * 1e6: why = "file too large"
        else:
            h = hashlib.md5(p.read_bytes()).hexdigest()
            if h in seen: why = f"duplicate of {seen[h]}"
            else: seen[h] = p.name
        if why: rejected.append(dict(file=str(p), reason=why)); continue
        m = re.search(PID, p.stem) or re.search(PID, p.parent.name)
        docs.append(dict(doc_id=p.stem, file=str(p), project_hint=m.group(1) if m else None))
    return {"docs": docs, "rejected": rejected, "log": [f"[1 intake] {len(docs)} valid, {len(rejected)} rejected"]}

def route_after_intake(state: S): return "pdf_extraction" if state["docs"] else END

def _ocr_pdf(path):
    import pypdfium2 as pdfium, pytesseract
    pdf, out = pdfium.PdfDocument(path), []
    for i in range(len(pdf)):
        out.append(pytesseract.image_to_string(pdf[i].render(scale=2.5).to_pil().convert("L")))
    pdf.close(); return out

def node_pdf_extraction(state: S):
    import pdfplumber
    docs, log, n_ocr = [], [], 0
    for d in state["docs"]:
        try:
            with pdfplumber.open(d["file"]) as pdf:
                pages = [dict(page=i + 1, text=pg.extract_text() or "", method="text") for i, pg in enumerate(pdf.pages)]
        except Exception as e:
            log.append(f"  ! cannot open {d['doc_id']}: {e}"); continue
        if any(len(p["text"].strip()) < CFG["MIN_TEXT_CHARS"] for p in pages):          # scanned -> OCR route
            try:
                for p, o in zip(pages, _ocr_pdf(d["file"])):
                    if len(p["text"].strip()) < CFG["MIN_TEXT_CHARS"]: p["text"], p["method"] = o, "ocr"
                n_ocr += 1
            except Exception as e:
                log.append(f"  ! OCR unavailable for {d['doc_id']} ({type(e).__name__}); install Tesseract + pytesseract")
        if any(len(p["text"].strip()) >= CFG["MIN_TEXT_CHARS"] for p in pages): docs.append({**d, "pages": pages})
        else: log.append(f"  ! no text extracted from {d['doc_id']}; skipped")
    return {"docs": docs, "log": [f"[2 extraction] {len(docs)} docs with text, {n_ocr} via OCR"] + log}

def node_metadata(state: S):
    records = {}
    for d in state["docs"]:
        sections, cur, header = [], dict(name="HEADER", canonical="header", text="", page=1), []
        for p in d["pages"]:
            for line in p["text"].splitlines():
                m = HEADING.match(line)
                if m:
                    sections.append(cur); name = f"{m.group(1)}. {m.group(2).strip()}"
                    cur = dict(name=name, canonical=canon_section(name), text="", page=p["page"])
                else: cur["text"] += line + "\n"
        sections.append(cur)
        sections = [s for s in sections if s["text"].strip()]
        full = "\n".join(p["text"] for p in d["pages"])
        head = next((s["text"] for s in sections if s["canonical"] == "header"), full[:800])
        pn = re.search(PID, head) or re.search(PID, full)
        name = re.search(r"Project Name:\s*(.+?)(?:\s+Project No|\n|$)", head)
        if not name:
            name = re.search(r"^(.*?)\s+Project No:", head, re.M)
        mt = re.search(r"Meeting\s*#\s*(\d+)", head)
        dt = re.search(r"\bDate:\s*(\d{1,2}\s+[A-Za-z]+\s+\d{4})", head)
        att = [a.strip() for a in re.findall(r"^([A-Z][a-z]+ [A-Z][A-Za-z'\-]+)\s", "\n".join(s["text"] for s in sections if s["canonical"] == "attendees"), re.M)]
        records[d["doc_id"]] = dict(
            doc_id=d["doc_id"], file=d["file"], ocr=any(p["method"] == "ocr" for p in d["pages"]),
            project_no=pn.group(1) if pn else d["project_hint"], project_name=name.group(1).strip() if name else None,
            meeting_no=int(mt.group(1)) if mt else None, meeting_date=parse_date(dt.group(1)) if dt else None,
            attendees=att, sections=sections, full_text=full, n_pages=len(d["pages"]))
    miss = [k for k, r in records.items() if not r["project_no"] or not r["meeting_date"]]
    return {"records": records, "log": [f"[3 metadata] parsed {len(records)} records" + (f"; incomplete metadata: {miss[:5]}" if miss else "")]}

def node_chunk_index(state: S):
    chunks = []
    for rid, r in state["records"].items():
        order = 0
        for s in r["sections"]:
            if s["canonical"] == "header": continue
            if s["canonical"] == "key_risks":
                pieces = [(f"{b['label']} Risk ({b['rag']}): {b['body']}" if b["rag"] else f"{b['label']}: {b['body']}") for b in split_risk_bullets(s["text"])] or [s["text"]]
            else:
                pieces, buf = [], ""
                for line in s["text"].splitlines():
                    if len(buf) + len(line) > 800 and buf: pieces.append(buf); buf = buf.splitlines()[-1] + "\n" if buf else ""
                    buf += line + "\n"
                if buf.strip(): pieces.append(buf)
            for t in pieces:
                meta = dict(doc_id=rid, project_no=r["project_no"], meeting_no=r["meeting_no"], date=r["meeting_date"],
                            section=s["canonical"], page=s["page"], file=r["file"], order=order)
                chunks.append(dict(id=f"{rid}::{order}", text=t.strip(), meta=meta)); order += 1
    STORE.add(chunks)
    return {"log": [f"[4 chunk+index] {len(chunks)} chunks added; store size {len(STORE.chunks)} ({CFG['EMBEDDER']})"]}

# %% [markdown]
# ## Node 5: evidence retrieval (RAG)

# %%
QUERIES = ["risk exposure concern problem", "cost increase price escalation change order", "schedule behind delay late delivery",
           "overdue open action item", "supply shortage supplier dependency", "safety quality environmental permit regulator"]
MUST = {"key_risks", "change_orders", "progress"}

def node_retrieval(state: S):
    evidence, history = {}, {}
    for rid, r in state["records"].items():
        own = lambda m, rid=rid: m["doc_id"] == rid
        got = {c["id"]: c for c in STORE.chunks if c["meta"]["doc_id"] == rid and c["meta"]["section"] in MUST}
        for q in QUERIES:
            for c in STORE.search(q, CFG["TOP_K"], own): got.setdefault(c["id"], c)
        evidence[rid] = sorted(got.values(), key=lambda c: c["meta"]["order"])[:20]
        prev = lambda m, r=r: (m["project_no"] == r["project_no"] and m["section"] == "key_risks"
                               and (m["date"] or "") < (r["meeting_date"] or ""))
        history[rid] = STORE.search("risk status update delay cost supplier", 6, prev)
    return {"evidence": evidence, "history": history, "todo": list(state["records"]), "feedback": {}, "retries": {},
            "candidates": {}, "assessed": {}, "log": [f"[5 retrieval] evidence passages for {len(evidence)} docs (+ prior-meeting context)"]}

# %% [markdown]
# ## Node 6: risk extraction (LLM + structured schema; rule-based fallback)

# %%
EXTRACT_SYS = f"""You are a construction project risk analyst. From meeting-minute passages, extract ONLY risks actually discussed in this meeting.
Return JSON: {{"risks":[{{"title":short noun phrase,"category":one of {CATEGORIES},"description":str,"signals":[str],"cause":str,"consequence":str,
"owner":name from the action items or null,"due_date":ISO date or null,"actions":[str],"stage":one of {STAGES},"evidence_quote":text copied VERBATIM from the passages (max 250 chars),
"source_section":str,"page":int,"confidence":0-1}}]}}.
Stages: raised=newly flagged; escalated=getting worse/now impacting work; mitigating=being resolved; closed=resolved; open=ongoing, no change.
Rules: never invent owners, dates, or numbers. Skip administrative items. Overall schedule status counts as a Schedule risk only if a cause or threat is described."""

def _stage_rule(body, rag):
    b = body.lower()
    if rag == "Closed" or "risk closed" in b or re.search(r"\bclosed\b", b): return "closed"
    if re.search(r"resolved|agreed|mobiliz|recover|improving|reduced|implemented|installed|delivered|completed|under way|underway|secured|resuming|being rebuilt|are in place|is under way", b): return "mitigating"
    if re.search(r"\bnow\b|second|persist|unresolved|not yet|on hold|confirmed|required|increased|formally|exposure|constrain|delaying|holding up|has missed|issued an order", b): return "escalated"
    return "raised"

def _find_owner(rec, title):
    words = {w for w in norm(title).split() if len(w) > 4}
    for s in rec["sections"]:
        if s["canonical"] in ("new_actions", "previous_actions"):
            for line in s["text"].splitlines():
                if words & set(norm(line).split()):
                    o = re.search(r"([A-Z]\. [A-Z][a-z]+ \([A-Za-z]+\))", line)
                    d = re.search(r"(\d{2} [A-Z][a-z]{2} \d{4})", line)
                    if o: return o.group(1), parse_date(d.group(1)) if d else None
    return None, None

def _rule_extract(rec, ev):
    out = []
    for e in ev:
        if e["meta"]["section"] != "key_risks": continue
        for b in split_risk_bullets(e["text"]):
            if b["label"] == "Schedule Status": continue
            first = re.split(r"(?<=[a-z0-9\)])\.\s", b["body"], maxsplit=1)
            title = first[0][:120]; rest = first[1] if len(first) > 1 else ""
            owner, due = _find_owner(rec, title)
            out.append(dict(title=title, category=b["label"], description=rest or b["body"], owner=owner, due_date=due,
                            stage=_stage_rule(b["body"], b["rag"]), evidence_quote=b["body"][:250],
                            source_section="key_risks", page=e["meta"]["page"], confidence=0.75 if not rec["ocr"] else 0.6))
    return out

def _fix_category(c):
    for k in CATEGORIES:
        if str(c).strip().lower() == k.lower(): return k
    best = difflib.get_close_matches(str(c).title(), CATEGORIES, n=1, cutoff=0.5)
    return best[0] if best else "Schedule"

def node_extract_risks(state: S):
    cands, feedback = dict(state.get("candidates", {})), state.get("feedback", {})
    n_llm = 0
    for rid in state["todo"]:
        rec, ev = state["records"][rid], state["evidence"][rid]
        raw = None
        passages = "\n\n".join(f"[{e['meta']['section']} | page {e['meta']['page']}]\n{e['text']}" for e in ev)
        hist = "\n".join(f"- (meeting {h['meta']['meeting_no']}) {h['text'][:200]}" for h in state["history"][rid]) or "none"
        user = (f"Project {rec['project_no']} meeting #{rec['meeting_no']} on {rec['meeting_date']}.\n"
                f"Risks from earlier meetings (context only):\n{hist}\n\nPASSAGES FROM THIS MEETING:\n{passages}\n"
                + (f"\nREVIEWER FEEDBACK (fix these problems): {feedback[rid]}\n" if rid in feedback else ""))
        res = llm_json(EXTRACT_SYS, user)
        if res and isinstance(res.get("risks"), list): raw, n_llm = res["risks"], n_llm + 1
        if raw is None: raw = _rule_extract(rec, ev)
        good = []
        for r in raw:
            try:
                c = RiskCandidate(**{k: v for k, v in r.items() if k in RiskCandidate.model_fields and v is not None}).model_dump()
                c["category"] = _fix_category(c["category"]); c["stage"] = c["stage"] if c["stage"] in STAGES else "open"
                good.append(c)
            except Exception: pass
        cands[rid] = good
    return {"candidates": cands, "redo": list(state["todo"]), "todo": [], "log": [f"[6 extraction] {sum(len(v) for v in cands.values())} candidate risks ({n_llm} docs via LLM)"]}

# %% [markdown]
# ## Node 7: classification & assessment

# %%
CLASSIFY_SYS = f"""You assess construction project risks using this rubric:\n{RUBRIC}
Return JSON {{"assessments":[{{"index":int,"category":one of {CATEGORIES},"likelihood":1-5,"impact":1-5,"rationale":"one sentence citing the evidence"}}]}}."""
IMPACT_BASE = dict(Safety=5, Contractual=4, Regulatory=4, Cost=4, Technical=4, **{"Supply Chain": 4}, Environmental=4,
                   Schedule=3, Quality=3, Resource=3, Stakeholder=2)

def node_classify(state: S):
    assessed = dict(state.get("assessed", {}))
    for rid in state["redo"]:
        cs = state["candidates"][rid]
        res = llm_json(CLASSIFY_SYS, json.dumps([dict(index=i, title=c["title"], category=c["category"], stage=c["stage"],
                       description=c["description"], evidence=c["evidence_quote"]) for i, c in enumerate(cs)]), 3000) if cs else None
        by = {a.get("index"): a for a in (res or {}).get("assessments", [])}
        out = []
        for i, c in enumerate(cs):
            c = dict(c); a = by.get(i)
            if a:
                c["category"] = _fix_category(a.get("category", c["category"]))
                L, I, why = int(a.get("likelihood", 3)), int(a.get("impact", 3)), a.get("rationale", "")
            else:
                L = dict(raised=3, escalated=4, mitigating=2, closed=1, open=3)[c["stage"]]
                I = IMPACT_BASE.get(c["category"], 3) - (1 if c["stage"] == "mitigating" else 0)
                I = 1 if c["stage"] == "closed" else I
                why = f"Rule-based: stage '{c['stage']}' and category '{c['category']}'."
            L, I = max(1, min(5, L)), max(1, min(5, I))
            c.update(likelihood=L, impact=I, score=L * I, rag="Closed" if c["stage"] == "closed" else rag_of(L * I), rationale=why)
            out.append(c)
        assessed[rid] = out
    return {"assessed": assessed, "log": [f"[7 classification] assessed {sum(len(v) for v in assessed.values())} risks (L x I rubric)"]}

# %% [markdown]
# ## Node 8: historical matching (project register)

# %%
_match_cache = {}
def _llm_match(c, entries):
    key = (c["title"], tuple(e["reg_id"] for e in entries))
    if key in _match_cache: return _match_cache[key]
    opts = "\n".join(f"{e['reg_id']}: {e['title']} -- {e['text'][-200:]}" for e in entries)
    res = llm_json("Decide whether a newly reported construction risk is the SAME underlying risk as one in the register. Return JSON {\"match\": \"<reg_id or null>\"}.",
                   f"New risk: {c['title']} -- {c['description']} {c['evidence_quote']}\n\nRegister:\n{opts}", 300)
    m = (res or {}).get("match"); _match_cache[key] = m if m in {e["reg_id"] for e in entries} else None
    return _match_cache[key]

def node_match(state: S):
    reg, matches = copy.deepcopy(state.get("register_base", {})), {}
    recs = state["records"]
    order = sorted(state["assessed"], key=lambda d: (recs[d]["project_no"] or "", recs[d]["meeting_date"] or "", recs[d]["meeting_no"] or 0))
    assessed = {}
    for rid in order:
        r, entries, used = recs[rid], reg.setdefault(recs[rid]["project_no"] or "UNKNOWN", []), set()
        assessed[rid] = []
        for c in state["assessed"][rid]:
            c = dict(c); text = f"{c['category']} {c['title']} {c['description']} {c['evidence_quote']}"
            pool = [e for e in entries if e["category"] == c["category"] and e["status"] != "closed" and e["reg_id"] not in used]
            dup = [e for e in entries if e["reg_id"] in used and e["category"] == c["category"]]
            best, sim, is_dup = None, 0.0, False
            if dup:
                ds = STORE.sims([text], [e["text"] for e in dup])[0]
                if ds.max() >= 0.6: best, sim, is_dup = dup[int(ds.argmax())], float(ds.max()), True
            if pool and not is_dup:
                sims = STORE.sims([text], [e["text"] for e in pool])[0]
                j = int(np.argmax(sims)); best, sim = pool[j], float(sims[j])
                if sim < CFG["MATCH_THRESHOLD"]:
                    near = [pool[k] for k in np.argsort(-sims)[:3] if sims[k] > 0.05]
                    m = _llm_match(c, near) if near else None
                    best = next((e for e in near if e["reg_id"] == m), None)
            c["match_similarity"] = round(sim, 3)
            if best is None:
                e = dict(reg_id=f"{(r['project_no'] or 'UNK')}-R{len(entries)+1:02d}", title=c["title"], category=c["category"],
                         first_meeting=r["meeting_no"], first_date=r["meeting_date"], last_meeting=r["meeting_no"], last_date=r["meeting_date"],
                         status=c["stage"], score=c["score"], text=text, history=[])
                entries.append(e); c.update(reg_id=e["reg_id"], is_new=True, status_change=None, duplicate=False); best = e
            else:
                c.update(reg_id=best["reg_id"], is_new=False, duplicate=is_dup,
                         status_change=f"{best['status']} -> {c['stage']}" if best["status"] != c["stage"] else None)
            used.add(best["reg_id"])
            if not c["duplicate"] and not any(h["doc_id"] == rid for h in best["history"]):
                best["history"].append(dict(doc_id=rid, meeting_no=r["meeting_no"], date=r["meeting_date"], stage=c["stage"],
                                            score=c["score"], rag=c["rag"], title_as_written=c["title"]))
                best.update(last_meeting=r["meeting_no"], last_date=r["meeting_date"], status=c["stage"], score=c["score"],
                            text=(best["text"] + " " + text)[-1500:])
            assessed[rid].append(c)
    n = sum(len(v) for v in reg.values())
    return {"assessed": assessed, "register": reg, "log": [f"[8 matching] register holds {n} distinct risks across {len(reg)} projects"]}

# %% [markdown]
# ## Node 9: quality / evidence check (can send documents back to node 6)

# %%
REQUIRED = ["title", "category", "description", "evidence_quote"]

def node_quality(state: S):
    recs, validated, doc_issues, todo, fb = state["records"], {}, {}, [], {}
    retries = dict(state.get("retries", {}))
    for rid, cs in state["assessed"].items():
        rec, bad, out = recs[rid], 0, []
        for c in cs:
            c, iss = dict(c), []
            if [f for f in REQUIRED if not c.get(f)]: iss.append("missing_required_field")
            if c.get("evidence_quote") and cover(c["evidence_quote"], rec["full_text"]) < 0.7: iss.append("unsupported_evidence")
            if c.get("confidence", 1) < CFG["CONF_MIN"]: iss.append("low_confidence")
            if c.get("duplicate"): iss.append("duplicate_in_document")
            if iss: bad += 1
            c["issues"], c["needs_review"] = iss, bool(iss); out.append(c)
        validated[rid] = out
        doc_issues[rid] = numeric_checks(rec)
        hard = sum(1 for c in out if {"unsupported_evidence", "missing_required_field"} & set(c["issues"]))
        if cs and hard / len(cs) > 0.3 and retries.get(rid, 0) < CFG["MAX_RETRIES"]:
            todo.append(rid); retries[rid] = retries.get(rid, 0) + 1
            fb[rid] = "Some evidence_quote values were not verbatim or required fields were empty. Copy quotes exactly from the passages and fill every required field."
    flagged = sum(1 for v in validated.values() for c in v if c["needs_review"])
    nums = sum(len(v) for v in doc_issues.values())
    return {"validated": validated, "doc_issues": doc_issues, "todo": todo, "feedback": fb, "retries": retries,
            "log": [f"[9 quality] {flagged} risks flagged for review, {nums} cross-section inconsistencies, {len(todo)} docs sent back for re-extraction"]}

def route_after_quality(state: S): return "risk_extraction" if state.get("todo") else "report"

# %% [markdown]
# ## Node 10: report & interface (CSV/PDF export, trends chart, filters)

# %%
def build_frames(state):
    recs, rows = state["records"], []
    for rid, cs in state["validated"].items():
        r = recs[rid]
        for c in cs:
            rows.append(dict(doc_id=rid, project_no=r["project_no"], meeting_no=r["meeting_no"], date=r["meeting_date"], reg_id=c.get("reg_id"),
                             title=c["title"], category=c["category"], stage=c["stage"], likelihood=c["likelihood"], impact=c["impact"],
                             score=c["score"], rag=c["rag"], owner=c.get("owner"), due_date=c.get("due_date"), status_change=c.get("status_change"),
                             needs_review=c["needs_review"], issues=",".join(c["issues"]), rationale=c["rationale"], evidence=c["evidence_quote"], ocr=r["ocr"]))
    findings = pd.DataFrame(rows)
    reg = [dict(project_no=p, **{k: v for k, v in e.items() if k not in ("text", "history")}, n_mentions=len(e["history"]))
           for p, es in state["register"].items() for e in es]
    inc = [dict(doc_id=d, project_no=recs[d]["project_no"], meeting_no=recs[d]["meeting_no"], **i) for d, v in state["doc_issues"].items() for i in v]
    return findings, pd.DataFrame(reg), pd.DataFrame(inc)

def make_pdf(path, findings, register, inc):
    from reportlab.lib.pagesizes import letter, landscape
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.lib import colors
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
    st = getSampleStyleSheet(); c = st["BodyText"]; c.fontSize = 8; c.leading = 9.5
    def tbl(df, cols, widths):
        data = [cols] + [[Paragraph(str(v)[:140], c) for v in row] for row in df[cols].values.tolist()]
        t = Table(data, colWidths=widths, repeatRows=1)
        t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#16324f")), ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                               ("GRID", (0, 0), (-1, -1), 0.3, colors.grey), ("FONTSIZE", (0, 0), (-1, -1), 8), ("VALIGN", (0, 0), (-1, -1), "TOP")]))
        return t
    doc = SimpleDocTemplate(path, pagesize=landscape(letter), leftMargin=36, rightMargin=36, topMargin=36, bottomMargin=36)
    s = [Paragraph("Project Risk Analysis Report", st["Title"]),
         Paragraph(f"{len(findings)} risk mentions across {findings['doc_id'].nunique()} meetings and {register['project_no'].nunique()} projects; "
                   f"{len(register)} distinct risks; {int(findings['needs_review'].sum())} flagged for review; {len(inc)} inconsistencies.", st["BodyText"]), Spacer(1, 10)]
    top = register[register["status"] != "closed"].sort_values("score", ascending=False).head(25)
    s += [Paragraph("Top open risks", st["Heading2"]), tbl(top, ["project_no", "reg_id", "title", "category", "status", "score", "last_meeting"], [80, 90, 280, 80, 70, 40, 70]), Spacer(1, 10)]
    rv = findings[findings["needs_review"]].head(30)
    if len(rv): s += [Paragraph("Flagged for human review", st["Heading2"]), tbl(rv, ["project_no", "meeting_no", "title", "issues", "evidence"], [80, 50, 180, 110, 290])]
    if len(inc): s += [Spacer(1, 10), Paragraph("Cross-section inconsistencies", st["Heading2"]), tbl(inc.head(40), ["project_no", "meeting_no", "kind", "detail"], [80, 50, 100, 450])]
    doc.build(s)

def node_report(state: S):
    out = Path(CFG["OUT_DIR"]); findings, register, inc = build_frames(state)
    findings.to_csv(out / "risk_findings.csv", index=False); register.to_csv(out / "risk_register.csv", index=False); inc.to_csv(out / "inconsistencies.csv", index=False)
    json.dump({p: [{k: v for k, v in e.items()} for e in es] for p, es in state["register"].items()}, open(out / "risk_register.json", "w"), indent=1)
    try: make_pdf(str(out / "risk_report.pdf"), findings, register, inc)
    except Exception as e: print("PDF report skipped:", e)
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(8, 4))
        for p, g in findings.groupby("project_no"):
            t = g.groupby("meeting_no")["score"].mean(); ax.plot(t.index, t.values, marker="o", label=p)
        ax.set_xlabel("Meeting #"); ax.set_ylabel("Mean risk score"); ax.set_title("Risk score trend by project"); ax.legend(fontsize=7)
        fig.tight_layout(); fig.savefig(out / "risk_trends.png", dpi=130); plt.close(fig)
    except Exception as e: print("Trend chart skipped:", e)
    return {"report": dict(findings=findings, register=register, inconsistencies=inc, out_dir=str(out)),
            "log": [f"[10 report] wrote CSV/JSON/PDF/PNG to {out}/"]}

# %% [markdown]
# ## Build the LangGraph

# %%
g = StateGraph(S)
for name, fn in [("intake", node_intake), ("pdf_extraction", node_pdf_extraction), ("metadata", node_metadata), ("chunk_index", node_chunk_index),
                 ("retrieval", node_retrieval), ("risk_extraction", node_extract_risks), ("classification", node_classify),
                 ("historical_matching", node_match), ("quality_check", node_quality), ("report", node_report)]:
    g.add_node(name, fn)
g.add_edge(START, "intake")
g.add_conditional_edges("intake", route_after_intake, {"pdf_extraction": "pdf_extraction", END: END})
for a, b in [("pdf_extraction", "metadata"), ("metadata", "chunk_index"), ("chunk_index", "retrieval"), ("retrieval", "risk_extraction"),
             ("risk_extraction", "classification"), ("classification", "historical_matching"), ("historical_matching", "quality_check")]:
    g.add_edge(a, b)
g.add_conditional_edges("quality_check", route_after_quality, {"risk_extraction": "risk_extraction", "report": "report"})
g.add_edge("report", END)
app = g.compile()
print(app.get_graph().draw_mermaid())      # paste into https://mermaid.live to view the graph

# %% [markdown]
# ## Run

# %%
files = sorted(glob.glob(os.path.join(CFG["PDF_DIR"], "**", "*.pdf"), recursive=True))
# process meetings in order within each project; MAX_DOCS keeps the first N overall (set None for all)
if CFG["MAX_DOCS"]: files = files[:CFG["MAX_DOCS"]]
base = {}
if CFG["USE_EXISTING_REGISTER"] and Path(CFG["OUT_DIR"], "risk_register.json").exists():
    base = json.load(open(Path(CFG["OUT_DIR"]) / "risk_register.json"))
t0 = time.time()
state = app.invoke({"files": files, "register_base": base, "log": []}, config={"recursion_limit": 80})
print("\n".join(state["log"])); print(f"done in {time.time()-t0:.0f}s; rejected files: {len(state['rejected'])}")
findings, register, inconsistencies = (state["report"][k] for k in ("findings", "register", "inconsistencies"))
findings.head(10)

# %% [markdown]
# ## Interface: filters, Q&A, evaluation against ground truth

# %%
def dashboard():
    """Interactive filters (needs ipywidgets); falls back to a static table."""
    try:
        import ipywidgets as w; from IPython.display import display
    except ImportError:
        return findings
    pj = w.Dropdown(options=["All"] + sorted(findings.project_no.dropna().unique()), description="Project")
    ct = w.Dropdown(options=["All"] + CATEGORIES, description="Category")
    rg = w.Dropdown(options=["All", "Red", "Amber", "Green", "Closed"], description="RAG")
    rv = w.Checkbox(description="Needs review only")
    out = w.Output()
    def refresh(*_):
        d = findings
        if pj.value != "All": d = d[d.project_no == pj.value]
        if ct.value != "All": d = d[d.category == ct.value]
        if rg.value != "All": d = d[d.rag == rg.value]
        if rv.value: d = d[d.needs_review]
        with out: out.clear_output(); display(d[["project_no", "meeting_no", "reg_id", "title", "category", "stage", "score", "rag", "status_change", "issues"]].head(50))
    for x in (pj, ct, rg, rv): x.observe(refresh, "value")
    display(w.VBox([w.HBox([pj, ct]), w.HBox([rg, rv]), out])); refresh()
# dashboard()

def ask(question, project=None, k=6):
    """Ask a question over the indexed minutes (RAG). Uses the LLM if configured, otherwise shows the retrieved passages."""
    hits = STORE.search(question, k, (lambda m: m["project_no"] == project) if project else None)
    ctx = "\n\n".join(f"[{h['meta']['project_no']} meeting {h['meta']['meeting_no']} {h['meta']['date']} {h['meta']['section']}]\n{h['text']}" for h in hits)
    ans = llm_text("Answer using ONLY the meeting-minute excerpts. Cite project and meeting number. Say if the excerpts do not contain the answer.", f"Question: {question}\n\nExcerpts:\n{ctx}")
    print(ans or ctx)
# ask("What is the status of cement supply delays?", project=None)

def evaluate(state, gt_dir=None):
    """Score the system against ground-truth JSON produced by the dataset generator."""
    gt_dir = Path(gt_dir or CFG["GT_DIR"]); tp = fp = fn = 0; cat_ok = stage_ok = 0; pairs = []
    flag_tp = flag_fp = flag_fn = 0; kinds = {}
    for rid, cs in state["validated"].items():
        p = gt_dir / f"{rid}.json"
        if not p.exists(): continue
        gt, used = json.load(open(p)), set()
        for gr in gt["risks"]:
            sc = [(cover(gr["evidence"], c["evidence_quote"] + " " + c["description"]), i) for i, c in enumerate(cs) if i not in used]
            s, i = max(sc, default=(0, None))
            if s >= 0.6: tp += 1; used.add(i); c = cs[i]; cat_ok += c["category"] == gr["category"]; stage_ok += c["stage"] == gr["stage"]; pairs.append((c.get("reg_id"), gr["risk_id"], rid[:11]))
            else: fn += 1
        fp += len(cs) - len(used)
        injected = {x["kind"] for x in gt["injected_inconsistencies"]}; found = {x["kind"] for x in state["doc_issues"].get(rid, [])}
        flag_tp += bool(injected and found); flag_fn += bool(injected and not found); flag_fp += bool(found and not injected)
        for k in injected: kinds.setdefault(k, [0, 0]); kinds[k][0] += 1; kinds[k][1] += k in found
    df = pd.DataFrame(pairs, columns=["reg", "gt", "proj"])
    purity = (df.groupby("reg")["gt"].agg(lambda s: s.value_counts().iloc[0]).sum() / len(df)) if len(df) else 0
    f = lambda a, b: round(a / b, 3) if b else None
    res = dict(risk_precision=f(tp, tp + fp), risk_recall=f(tp, tp + fn), category_accuracy=f(cat_ok, tp), stage_accuracy=f(stage_ok, tp),
               register_linkage_purity=round(float(purity), 3), inconsistency_recall=f(flag_tp, flag_tp + flag_fn),
               inconsistency_precision=f(flag_tp, flag_tp + flag_fp), inconsistency_recall_by_kind={k: f(b, a) for k, (a, b) in kinds.items()})
    print(json.dumps(res, indent=1)); return res
# evaluate(state)

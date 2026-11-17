#!/usr/bin/env python3
"""
Synthetic construction meeting-minutes dataset generator (fully offline).

Creates hundreds of realistic weekly progress-meeting PDFs across several projects,
with evolving risks (raised -> escalated -> mitigating -> closed), action items,
change orders, payments, and deliberately planted inconsistencies. Also writes
ground-truth JSON, per-project risk registers, a manifest CSV and a ZIP archive.

Setup:   pip install reportlab
Optional (image-only "scanned" PDFs to test OCR):  pip install pypdfium2 pillow
Run:     python generate_minutes_dataset.py
         python generate_minutes_dataset.py --projects 15 --meetings 30 --scan-ratio 0.3 --seed 7
"""
import argparse, csv, json, os, re, shutil, sys, zipfile
from datetime import date, timedelta
import random

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from xml.sax.saxutils import escape as esc

# ----------------------------------------------------------------------------- pools
CITIES = ["Lakeshore", "Maple Ridge", "Northgate", "Cedar Falls", "Brookhaven", "Riverton",
          "Stonebridge", "Harborview", "Pinecrest", "Westfield", "Oakhurst", "Fairmont"]
STREETS = ["Riverside Ave", "Maple St", "Industrial Rd", "Lakeview Blvd", "King St",
           "Harbour Way", "Elm Dr", "Quarry Rd", "Station Rd", "Mill St"]
FIRST = ["Sarah", "James", "Mei", "Robert", "Lisa", "David", "Priya", "Michael", "Aisha", "Daniel",
         "Elena", "Omar", "Grace", "Kevin", "Nadia", "Thomas", "Hannah", "Victor", "Amara", "Peter"]
LAST = ["Patel", "Okonkwo", "Chen", "Singh", "Tran", "Morales", "Nguyen", "Fraser", "Adeyemi", "Kowalski",
        "Rossi", "Haddad", "Campbell", "Ibrahim", "Lefebvre", "Walker", "Sato", "Murphy", "Dubois", "Khan"]
CONTRACTORS = ["Apex Civil Constructors Ltd.", "Ridgeline Builders Inc.", "Tri-County Contracting Ltd.",
               "Blue Heron Construction", "Ironbridge Civil Ltd.", "Keystone Infrastructure Corp."]
ENGINEERS = ["Northbridge Engineering", "Halden Consulting Engineers", "Clearwater Engineering Group",
             "Meridian Design and Engineering"]
SUPPLIERS = ["Granite Peak Materials", "Northern Aggregates Ltd.", "Summit Cement Co.", "Valley Ready-Mix",
             "Ironwood Steel Supply", "Lakeland Precast", "Redstone Pipe and Fittings"]
UTILITIES = ["Waterfront Utilities", "Hydro Distribution Corp.", "Northern Gas Services", "MetroTel Communications"]
FILLER = [
    "Site housekeeping was reviewed; no concerns noted.",
    "Traffic management plan was updated for the upcoming lane closure.",
    "Contractor confirmed daily site safety tailgate meetings are being documented.",
    "Resident Engineer noted as-built survey submissions are up to date.",
    "Public notice flyers were distributed to adjacent properties on Tuesday.",
    "Shop drawing review turnaround remains within the contract timeframe.",
    "Erosion and sediment control measures were inspected and found satisfactory.",
    "The owner requested photographs of completed work be added to the weekly report.",
    "Contractor to maintain dust control during dry conditions.",
    "Testing agency reported all compaction results within specification this week.",
]
M3 = "m^3"

def lib(key, cat, aliases, cost, days, imp, owners, raised, escal, mit, closed, acts):
    return dict(key=key, cat=cat, aliases=aliases, cost=cost, days=days, imp=imp, owners=owners,
                raised=raised, escal=escal, mit=mit, closed=closed, acts=acts)

RISKS = [
    lib("supply_delay", "Supply Chain", ["Material delivery delay", "Supplier delivery slippage", "Long-lead material shortfall"],
        None, (3, 10), 4, ["Contractor Superintendent", "Contractor PM"],
        ["{supplier} advised of a possible {d1}-{d2} working day delay on the next shipment due to plant maintenance.",
         "Supplier has indicated upcoming deliveries may slip by up to {d2} working days; written confirmation pending."],
        ["Deliveries are now {d2} working days late and are constraining crews at {loc}.",
         "Second consecutive late shipment from {supplier}; work at {loc} has been partially resequenced."],
        ["Expedited deliveries arranged with {supplier}; recovery of {d1} days is expected if supply stabilizes.",
         "Alternate supplier qualified and first load delivered; stockpile is being rebuilt."],
        ["Deliveries have been on schedule for two consecutive weeks. Risk closed."],
        dict(raised=["Confirm revised delivery schedule with {supplier}.", "Provide written delivery forecast for the next four weeks."],
             escal=["Confirm alternative supply options (second source) to mitigate further delivery risk.", "Submit recovery schedule showing resequencing around late deliveries."],
             mit=["Provide weekly stockpile report until recovery is confirmed."])),
    lib("price_escalation", "Cost", ["Material price escalation", "Supplier unit price increase", "Cost escalation on remaining quantities"],
        (25000, 140000), (0, 0), 4, ["Contractor PM", "Contractor Superintendent"],
        ["Supplier notified an expected price increase of about {pct}% for deliveries after the end of the month.",
         "{supplier} advised of a {pct}% price adjustment due to raw material and energy costs."],
        ["Potential cost exposure of approximately {amt} if the {pct}% escalation is accepted on remaining quantities.",
         "Contractor formally requested a change order for the {pct}% escalation; exposure estimated at {amt}."],
        ["Owner is reviewing the supplier letter and original bid unit rates; negotiated reduction under discussion.",
         "Price hold agreed with supplier for part of the remaining quantity; residual exposure reduced."],
        ["Price escalation resolved through approved change order. Risk closed."],
        dict(raised=["Provide supplier letter and original bid unit rates for the price escalation.", "Provide detailed unit price breakdown for the escalation."],
             escal=["Submit formal change order package including cost breakdown.", "Review cumulative change order exposure against original contract value."],
             mit=["Confirm final negotiated unit price in writing."])),
    lib("utility_conflict", "Technical", ["Utility conflict", "Buried utility interference", "Third-party utility clash"],
        (10000, 45000), (2, 6), 3, ["Resident Engineer", "Contractor Superintendent"],
        ["A buried utility was exposed near {loc} that conflicts with the design alignment.",
         "Locate results show an unmarked utility crossing at {loc}; design team to assess."],
        ["Utility owner has not yet confirmed relocation; work at {loc} is on hold pending direction.",
         "Conflict at {loc} is delaying the critical path; relocation support is being scoped."],
        ["Relocation agreed with the utility owner; contractor to provide support work at {loc}.",
         "Revised alignment issued by RFI response; work at {loc} resuming."],
        ["Utility relocation complete and work at {loc} resumed. Risk closed."],
        dict(raised=["Issue RFI response for the utility conflict at {loc}.", "Obtain utility owner location confirmation for {loc}."],
             escal=["Confirm utility relocation schedule with the utility owner.", "Prepare cost estimate for relocation support at {loc}."],
             mit=["Submit as-built records of the relocated utility."])),
    lib("weather", "Schedule", ["Adverse weather delays", "Weather-related lost days", "Rain and saturated ground conditions"],
        None, (2, 6), 3, ["Contractor Superintendent"],
        ["Heavy rain this week resulted in one lost working day.", "Forecast shows extended wet weather; some earthworks may be affected."],
        ["Saturated ground conditions have now cost {d1} working days and are limiting access at {loc}.",
         "Repeated rain events are eroding the schedule float; weekend work is being considered."],
        ["Dewatering measures are in place and crews are working extended hours to recover time.",
         "Contractor has resequenced work to interior and protected areas."],
        ["Conditions have normalized and lost days have been recovered. Risk closed."],
        dict(raised=["Submit updated schedule reflecting weather delays."],
             escal=["Propose a recovery plan including weekend work options.", "Provide dewatering plan for {loc}."],
             mit=["Report recovered days weekly."])),
    lib("labour_shortage", "Resource", ["Skilled labour shortage", "Crew availability shortfall", "Trade labour constraints"],
        None, (2, 7), 3, ["Contractor Superintendent", "Contractor PM"],
        ["Contractor reported difficulty staffing the concrete finishing crew.", "Two experienced operators left the project; replacements are being sourced."],
        ["Crew productivity is about {pct}% below plan and is affecting output at {loc}.",
         "Labour shortfall persists; subcontractor has not committed additional workers."],
        ["Additional crew mobilized from another contractor site; output is improving.", "Overtime approved to compensate for reduced crew size."],
        ["Crew levels restored to plan. Risk closed."],
        dict(raised=["Provide updated manpower histogram.", "Confirm replacement crew mobilization dates."],
             escal=["Submit labour recovery plan.", "Confirm availability of additional subcontract crews."],
             mit=["Report weekly crew counts against plan."])),
    lib("permit_delay", "Regulatory", ["Permit approval delay", "Regulatory approval pending", "Authority review delay"],
        None, (3, 12), 4, ["Owner PM", "Resident Engineer"],
        ["The environmental authority has not yet issued the amended permit required for the next phase.",
         "Approval of the revised traffic plan is outstanding with the road authority."],
        ["Permit remains outstanding; the next work phase cannot start until it is issued.",
         "Authority requested additional information, extending the review by an estimated {d2} working days."],
        ["Additional information submitted; the authority has committed to a decision within one week.",
         "Interim approval granted for limited work while the final permit is processed."],
        ["Permit issued and work proceeding. Risk closed."],
        dict(raised=["Follow up with the authority on permit status.", "Submit outstanding permit documents."],
             escal=["Escalate permit status with the authority director.", "Prepare contingency sequencing if the permit is delayed further."],
             mit=["Confirm permit conditions are incorporated into the work plan."])),
    lib("design_change", "Cost", ["Design change", "Revised design requirements", "Scope change from geotechnical recommendation"],
        (30000, 90000), (2, 5), 3, ["Resident Engineer", "Owner PM"],
        ["A design revision is anticipated at {loc} following the latest field review.",
         "The consultant issued a preliminary revision affecting quantities near {loc}."],
        ["Revised drawings were issued; additional work is estimated at {amt} and {d1} working days.",
         "Design change at {loc} is confirmed; contractor to price the added scope (about {amt})."],
        ["Pricing reviewed with the owner; a reduced scope option is being considered.", "Change order under review; work proceeding at risk on the unaffected areas."],
        ["Design change incorporated through an approved change order. Risk closed."],
        dict(raised=["Provide preliminary pricing for the anticipated design change.", "Issue revised drawings for {loc}."],
             escal=["Submit change order proposal for the design change.", "Review the schedule impact of the revised design."],
             mit=["Finalize change order documentation."])),
    lib("subcontractor_default", "Contractual", ["Subcontractor performance concern", "Subcontractor financial instability", "Subcontractor default risk"],
        (20000, 80000), (4, 10), 5, ["Contractor PM", "Owner PM"],
        ["Owner raised concerns about the subcontractor's slow mobilization and unpaid supplier notices.",
         "A lien notice was received relating to a subcontractor on the project."],
        ["Subcontractor has demobilized; the contractor is assessing replacement options.", "Subcontractor has missed two commitments; default notice under consideration."],
        ["Replacement subcontractor identified and mobilizing.", "Payment arrangement reached to retain the subcontractor."],
        ["Replacement subcontractor fully mobilized. Risk closed."],
        dict(raised=["Provide confirmation of subcontractor payment status.", "Submit statutory declaration on subcontractor payments."],
             escal=["Provide a replacement subcontractor plan.", "Owner to review default provisions in the contract."],
             mit=["Confirm replacement subcontractor insurance and bonding."])),
    lib("unforeseen_conditions", "Technical", ["Differing site conditions", "Unforeseen subsurface conditions", "Unsuitable subgrade material"],
        (40000, 160000), (3, 9), 4, ["Resident Engineer", "Contractor Superintendent"],
        ["Unexpected soft material was encountered near {loc}; geotechnical consultant notified.",
         "Groundwater levels at {loc} are higher than shown in the geotechnical report."],
        ["Geotechnical review recommends deeper excavation at {loc}; estimated impact {amt} and {d1} working days.",
         "Unsuitable material extends beyond the initial extent; additional removal and replacement required."],
        ["Revised foundation detail issued; contractor proceeding with removal and replacement.", "Dewatering and stabilization measures are working; the impact area is now bounded."],
        ["Subgrade remediation complete and accepted. Risk closed."],
        dict(raised=["Obtain geotechnical recommendation for {loc}.", "Provide test pit records for {loc}."],
             escal=["Submit change order for the differing site condition.", "Issue revised foundation detail for {loc}."],
             mit=["Provide inspection records of remediated areas."])),
    lib("quality_ncr", "Quality", ["Concrete strength nonconformance", "Quality nonconformance", "Failed compaction and strength results"],
        None, (2, 5), 3, ["Resident Engineer", "Contractor Superintendent"],
        ["Two 28-day cylinder results were below the specified strength; investigation started.",
         "A nonconformance report was issued for out-of-tolerance work at {loc}."],
        ["Core testing confirms low strength in the affected section; remedial options are being evaluated.", "Rework at {loc} is required; the area is closed to further work."],
        ["Remedial plan accepted by the engineer; rework is under way.", "Retest results are improving; final acceptance pending."],
        ["Retest results meet the specification. Risk closed."],
        dict(raised=["Submit root cause analysis for the nonconformance.", "Provide revised concrete mix design for the affected section."],
             escal=["Submit remedial work plan for engineer acceptance.", "Arrange independent core testing."],
             mit=["Provide retest results to the engineer."])),
    lib("safety_incident", "Safety", ["Site safety concern", "Near-miss trend", "Safety incident follow-up"],
        None, (1, 4), 5, ["Contractor Superintendent", "Resident Engineer"],
        ["A near-miss involving excavation access was reported this week.", "A minor first-aid incident occurred; investigation under way."],
        ["A second incident in two weeks prompted a safety stand-down and an inspector visit.", "Regulator issued an order requiring improved shoring at {loc}."],
        ["Corrective actions implemented and a refresher safety orientation delivered to all crews.", "Daily safety audits are in place and trending positively."],
        ["No further incidents for three weeks and the regulator order is cleared. Risk closed."],
        dict(raised=["Submit incident investigation report.", "Provide updated hazard assessment."],
             escal=["Submit corrective action plan to the regulator.", "Hold a safety stand-down for all crews."],
             mit=["Provide weekly safety audit results."])),
    lib("environmental", "Environmental", ["Contaminated soil", "Environmental compliance issue", "Sediment discharge concern"],
        (15000, 70000), (3, 8), 4, ["Resident Engineer", "Owner PM"],
        ["Staining and odour were noted in excavated soil near {loc}; samples sent for analysis.",
         "A sediment discharge was observed at the downstream outlet."],
        ["Lab results confirm contamination above the disposal threshold; special handling is required (estimated {amt}).",
         "Regulator notified of the discharge; additional controls required at {loc}."],
        ["A disposal facility was secured and hauling has started.", "Additional controls installed; follow-up sampling is within limits."],
        ["Contaminated material removed and clearance letter received. Risk closed."],
        dict(raised=["Provide laboratory results for soil samples from {loc}.", "Inspect and repair sediment controls."],
             escal=["Submit soil management plan.", "Provide cost estimate for contaminated soil disposal."],
             mit=["Submit disposal manifests."])),
    lib("stakeholder_access", "Stakeholder", ["Property access restriction", "Adjacent owner complaint", "Stakeholder access issue"],
        None, (1, 5), 2, ["Owner PM"],
        ["An adjacent property owner has restricted access for staging near {loc}.", "Several resident complaints about noise and dust were received."],
        ["Access remains restricted; work near {loc} has been rescheduled.", "Complaints have increased; the councillor's office requested a meeting."],
        ["Access agreement reached with the adjacent owner.", "Mitigation measures including limited work hours are in effect."],
        ["Access restored and complaints have stopped. Risk closed."],
        dict(raised=["Contact the adjacent property owner regarding access.", "Issue a public notice update."],
             escal=["Hold a stakeholder meeting with affected residents.", "Prepare an alternate staging plan."],
             mit=["Report complaint log weekly."])),
    lib("equipment_failure", "Resource", ["Equipment breakdown", "Critical equipment unavailable", "Plant failure"],
        None, (1, 4), 2, ["Contractor Superintendent"],
        ["The primary excavator has been out of service for two days.", "A concrete pump failure interrupted a pour this week."],
        ["Repairs are taking longer than expected; a rental unit is being sourced.", "Second breakdown on the same unit; production at {loc} is reduced."],
        ["Rental equipment delivered and production has resumed.", "Repair completed; the contractor added a standby unit."],
        ["Equipment fully operational with standby available. Risk closed."],
        dict(raised=["Confirm repair timeline for the equipment."],
             escal=["Mobilize replacement equipment."],
             mit=["Confirm standby equipment arrangements."])),
    lib("rfi_delay", "Schedule", ["Delayed RFI responses", "Design clarification backlog", "Outstanding RFIs"],
        None, (2, 6), 3, ["Resident Engineer", "Owner PM"],
        ["Several RFIs have exceeded the five-day response target.", "A backlog of design clarifications is building up."],
        ["Delayed RFI responses are now holding up work at {loc}.", "RFI backlog has reached {d2} items, several on the critical path."],
        ["A dedicated reviewer was assigned and the backlog is reducing.", "Weekly RFI review call established with the design team."],
        ["RFI backlog cleared. Risk closed."],
        dict(raised=["Issue responses to overdue RFIs.", "Provide a priority list of outstanding RFIs."],
             escal=["Engineer to provide a response plan for the critical RFIs.", "Schedule a design coordination meeting."],
             mit=["Report open RFI count weekly."])),
    lib("payment_dispute", "Contractual", ["Payment certification dispute", "Disputed progress claim", "Holdback and certification disagreement"],
        None, (0, 2), 3, ["Contractor PM", "Owner PM"],
        ["The contractor disputes the certified amount on the last progress claim.", "Differences exist between claimed and verified quantities on the pay application."],
        ["The dispute remains unresolved and the contractor has signalled slowed mobilization.", "Contractor issued a notice regarding delayed payment."],
        ["Quantities were jointly verified on site; a revised certificate is being prepared.", "Owner and contractor agreed on a settlement approach."],
        ["Revised certificate issued and the contractor confirmed agreement. Risk closed."],
        dict(raised=["Provide quantity backup for the disputed claim.", "Joint site verification of disputed quantities."],
             escal=["Owner to respond in writing to the contractor's notice.", "Prepare a dispute resolution proposal."],
             mit=["Issue the revised payment certificate."])),
]

PROJECT_TYPES = [
    dict(code="RSW", title="Storm Sewer Upgrade", dept="Public Works Department — Capital Projects Division",
         loc=lambda r: f"Station {r.randint(0,3)}+{r.randrange(0,1000,5):03d}",
         mats=[("Portland Cement (Type GU)", "t", 1850), ("35 MPa Ready-Mix Concrete", M3, 2400), ("Reinforcing Steel (15M)", "t", 48)],
         work=["installation of {n} m of 1200 mm RCP storm sewer", "backfill and compaction of {n} m of trench", "forming and pouring of {k} manholes"],
         look=["Continue sewer installation to the next station.", "Begin lateral connections.", "Pour the next two manholes."]),
    dict(code="WMR", title="Water Main Replacement", dept="Public Works Department — Water Services Division",
         loc=lambda r: f"Station {r.randint(0,4)}+{r.randrange(0,1000,5):03d}",
         mats=[("Ductile Iron Pipe (300 mm)", "m", 3200), ("Granular A Bedding", "t", 5200), ("Gate Valves", "units", 42)],
         work=["installation of {n} m of 300 mm ductile iron water main", "pressure testing of {n} m of new main", "installation of {k} valve chambers"],
         look=["Continue main installation and tie-ins.", "Complete pressure testing and disinfection.", "Begin service reconnections."]),
    dict(code="BRG", title="Bridge Rehabilitation", dept="Transportation Services — Structures Division",
         loc=lambda r: f"Pier {r.randint(1,6)}, Span {r.randint(1,5)}",
         mats=[("Structural Concrete 40 MPa", M3, 1400), ("Reinforcing Steel (20M)", "t", 210), ("Bearing Assemblies", "units", 16)],
         work=["removal of {n} m^2 of deteriorated deck concrete", "installation of {n} t of reinforcing steel", "completion of {k} pier cap repairs"],
         look=["Continue deck removal and rebar placement.", "Pour the next deck section.", "Install the remaining bearings."]),
    dict(code="CCC", title="Community Centre Construction", dept="Facilities and Capital Projects Department",
         loc=lambda r: f"Level {r.randint(1,3)}, Grid {r.choice('ABCDEF')}{r.randint(1,8)}",
         mats=[("Structural Steel", "t", 380), ("Concrete Slab 32 MPa", M3, 1850), ("Precast Wall Panels", "units", 120)],
         work=["erection of {n} t of structural steel", "placement of {n} m^3 of slab concrete", "installation of {k} precast wall panels"],
         look=["Continue steel erection on the upper level.", "Pour the next slab section.", "Begin roofing preparation."]),
    dict(code="RDR", title="Road Reconstruction", dept="Public Works Department — Roads Division",
         loc=lambda r: f"Station {r.randint(0,6)}+{r.randrange(0,1000,5):03d}",
         mats=[("Hot Mix Asphalt", "t", 9800), ("Granular Base", "t", 14500), ("Concrete Curb and Gutter", "m", 3600)],
         work=["placement of {n} m of granular base", "paving of {n} m of base asphalt", "construction of {n} m of curb and gutter"],
         look=["Continue base preparation and curb installation.", "Place the next lift of asphalt.", "Begin driveway restoration."]),
    dict(code="WPS", title="Wastewater Pump Station Upgrade", dept="Environmental Services — Wastewater Division",
         loc=lambda r: f"Wet well, Elevation {r.randint(1,5)}",
         mats=[("Structural Concrete 35 MPa", M3, 950), ("Reinforcing Steel (15M)", "t", 120), ("Submersible Pumps", "units", 4)],
         work=["placement of {n} m^3 of wet well concrete", "installation of {n} m of force main piping", "completion of {k} pump base installations"],
         look=["Continue wet well wall pours.", "Install the remaining pump piping.", "Begin electrical rough-in."]),
]

# ----------------------------------------------------------------------------- helpers
def money(x): return f"${x:,.0f}"
def dlong(d): return f"{d.day} {d.strftime('%B %Y')}"
def dshort(d): return f"{d.day:02d} {d.strftime('%b %Y')}"
def dmed(d): return f"{d.day:02d} {d.strftime('%b')}"
def slug(s): return re.sub(r"[^a-z0-9]", "", s.lower())

def rag(score): return "Red" if score >= 15 else "Amber" if score >= 8 else "Green"

class Risk:
    def __init__(self, rng, spec, rid, start, proj):
        self.spec, self.id, self.start = spec, rid, start
        self.title = spec["aliases"][0]
        self.r_len, self.e_len, self.m_len = rng.randint(1, 3), rng.randint(0, 4), rng.randint(1, 4)
        self.persist = rng.random() < 0.2
        if self.persist and self.e_len == 0: self.e_len = 2
        lo, hi = spec["days"]
        self.days = rng.randint(lo, hi) if hi else 0
        self.d1, self.d2 = max(1, self.days - 2), max(2, self.days)
        self.amount = round(rng.randint(*spec["cost"]), -2) if spec["cost"] else 0
        self.supplier, self.loc = rng.choice(SUPPLIERS), proj["type"]["loc"](rng)
        self.pct = rng.randint(6, 15)
        self.owner_role = rng.choice(spec["owners"])
        self.has_co = False
    def stage_at(self, m):
        off = m - self.start
        if off < 0: return None
        b1, b2, b3 = self.r_len, self.r_len + self.e_len, self.r_len + self.e_len + self.m_len
        if off < b1: return "raised"
        if self.persist: return "escalated"
        if off < b2: return "escalated"
        if off < b3: return "mitigating"
        return "closed" if off == b3 else None
    def fmt(self, s):
        return s.format(supplier=self.supplier, loc=self.loc, d1=self.d1, d2=self.d2,
                        amt=money(self.amount), pct=self.pct)
    def rating(self, stage):
        L = {"raised": 3, "escalated": 4, "mitigating": 2, "closed": 1}[stage]
        I = {"raised": self.spec["imp"] - 1, "escalated": self.spec["imp"], "mitigating": self.spec["imp"] - 1, "closed": 1}[stage]
        I = max(1, min(5, I))
        return L, I, L * I

def P(text, style):
    t = esc(str(text)).replace("^3", "<super>3</super>").replace("^2", "<super>2</super>")
    t = t.replace("[b]", "<b>").replace("[/b]", "</b>")
    return Paragraph(t, style)

BASE = ParagraphStyle("b", fontName="Helvetica", fontSize=9, leading=11.5)
CELL = ParagraphStyle("c", parent=BASE, fontSize=8, leading=10)
HCELL = ParagraphStyle("h", parent=CELL, fontName="Helvetica-Bold", textColor=colors.white)
H2 = ParagraphStyle("h2", parent=BASE, fontName="Helvetica-Bold", fontSize=10.5, textColor=colors.HexColor("#16324f"), spaceBefore=9, spaceAfter=4)
TITLE = ParagraphStyle("t", parent=BASE, fontName="Helvetica-Bold", fontSize=15, leading=18, textColor=colors.HexColor("#16324f"))
NAVY = colors.HexColor("#16324f")

def table(rows, widths, header=True):
    data = []
    for i, r in enumerate(rows):
        data.append([P(c, HCELL if (header and i == 0) else CELL) for c in r])
    t = Table(data, colWidths=[w * inch for w in widths], repeatRows=1 if header else 0)
    st = [("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#b8c2cc")), ("VALIGN", (0, 0), (-1, -1), "TOP"),
          ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 3)]
    if header: st.append(("BACKGROUND", (0, 0), (-1, 0), NAVY))
    for i in range(1 if header else 0, len(rows)):
        if i % 2 == 0: st.append(("BACKGROUND", (0, i), (-1, i), colors.HexColor("#eef3f8")))
    t.setStyle(TableStyle(st))
    return t

# ----------------------------------------------------------------------------- project setup
def make_people(rng, city, contractor, engineer):
    used, people = set(), {}
    def person(role, org, domain, phone=True):
        while True:
            f, l = rng.choice(FIRST), rng.choice(LAST)
            if (f, l) not in used: used.add((f, l)); break
        email = f"{f[0].lower()}.{l.lower()}@{domain}"
        ph = f"{rng.choice(['416','647','905'])}-555-{rng.randint(100,299):04d}" if phone else ""
        short = f"{f[0]}. {l} ({org.split()[0]})"
        return dict(name=f"{f} {l}", org=org, role=role, email=email, phone=ph, short=short)
    cd, ed = slug(contractor.split()[0]) + ".ca", slug(engineer.split()[0]) + "-eng.com"
    od = slug(city) + ".ca"
    people["Owner PM"] = person("City of %s Project Manager (Owner)" % city, "City", od)
    people["Procurement Officer"] = person("Procurement Officer", "City", od, False)
    people["Contractor Superintendent"] = person("Site Superintendent", contractor, cd)
    people["Contractor PM"] = person("Project Manager (Contractor)", contractor, cd)
    people["Resident Engineer"] = person("Resident Engineer", engineer, ed)
    people["Project Coordinator"] = person("Project Coordinator / Minutes", engineer, ed, False)
    return people

def make_project(rng, idx, M, used_nos):
    t = rng.choice(PROJECT_TYPES)
    city = CITIES[idx % len(CITIES)]
    while True:
        pno = f"{t['code']}-{rng.choice([2024, 2025, 2026])}-{rng.randint(1, 199):03d}"
        if pno not in used_nos: used_nos.add(pno); break
    contractor, engineer = rng.choice(CONTRACTORS), rng.choice(ENGINEERS)
    start = date(2025, 1, 2) + timedelta(weeks=rng.randint(0, 60))
    P_ = dict(type=t, city=city, no=pno, street=rng.choice(STREETS), addr=rng.randint(100, 900),
              name=f"{rng.choice(['Riverside','Lakeview','Northgate','Central','Harbour','Elm Street','Mill Creek'])} {t['title']} — Phase {rng.randint(1,3)}",
              contractor=contractor, engineer=engineer, contract=rng.randrange(2_000_000, 18_000_000, 50_000),
              start=start, end=start + timedelta(weeks=M + 8), M=M,
              utility=rng.choice(UTILITIES), t0=rng.choice(["08:30", "09:00", "10:00"]), dur=rng.choice([60, 75, 90]))
    P_["people"] = make_people(rng, city, contractor, engineer)
    n = rng.randint(4, 7)
    specs = rng.sample(RISKS, n)
    risks = []
    for i, s in enumerate(specs):
        risks.append(Risk(rng, s, f"R-{i+1:02d}", rng.randint(0, max(1, M - 7)), P_))
    P_["risks"] = sorted(risks, key=lambda r: r.start)
    for i, r in enumerate(P_["risks"]): r.id = f"R-{i+1:02d}"
    return P_

# ----------------------------------------------------------------------------- meeting generation
def simulate_project(rng, proj, args):
    M, people, t = proj["M"], proj["people"], proj["type"]
    open_actions, ai_counter, cos, prs = [], 0, [], []
    prev_stage, prev_actual, last_claim_pct = {}, 0.0, 0.0
    meetings, register = [], {r.id: dict(risk_id=r.id, canonical_title=r.title, category=r.spec["cat"], first_meeting=None,
                                         closed_meeting=None, history=[]) for r in proj["risks"]}
    mats_used = [0.0] * 3
    for m in range(M):
        d = proj["start"] + timedelta(weeks=m)
        stages = {r.id: r.stage_at(m) for r in proj["risks"]}
        active = [r for r in proj["risks"] if stages[r.id]]
        trans = [r for r in active if stages[r.id] != prev_stage.get(r.id) and stages[r.id] != "closed"]
        # --- previous actions
        prev_rows, still_open = [], []
        for a in open_actions:
            if a["due"] <= d:
                if rng.random() < 0.7: a["status"] = "CLOSED"
                else: a["status"] = "OPEN - Overdue"
            else: a["status"] = "OPEN"
            prev_rows.append(dict(a))
            if a["status"] != "CLOSED": still_open.append(a)
        open_actions = still_open
        # --- new / carried actions
        carried = []
        for a in open_actions:
            if a["status"] == "OPEN - Overdue":
                a["due"] = d + timedelta(days=rng.choice([3, 5, 7])); a["carried"] = True
                a["status"] = "Open"; carried.append(a)
        new_actions = []
        for r in trans:
            stg = stages[r.id]
            key = {"raised": "raised", "escalated": "escal", "mitigating": "mit"}[stg]
            for txt in rng.sample(r.spec["acts"][key], k=min(len(r.spec["acts"][key]), rng.choice([1, 1, 2]))):
                ai_counter += 1
                owner = people[r.owner_role]
                a = dict(id=f"AI-{ai_counter:02d}", desc=r.fmt(txt), owner=owner["short"], due=d + timedelta(days=rng.choice([3, 5, 7, 10])),
                         status="Open", risk_id=r.id, created=m + 1, carried=False)
                new_actions.append(a); open_actions.append(a)
        if rng.random() < 0.4:
            ai_counter += 1
            who = people[rng.choice(["Contractor PM", "Resident Engineer", "Contractor Superintendent"])]
            a = dict(id=f"AI-{ai_counter:02d}", desc=rng.choice(["Submit updated three-week look-ahead schedule.", "Provide revised traffic management plan.", "Submit outstanding shop drawings for review.", "Update the project risk log."]),
                     owner=who["short"], due=d + timedelta(days=rng.choice([5, 7])), status="Open", risk_id=None, created=m + 1, carried=False)
            new_actions.append(a); open_actions.append(a)
        # --- change orders
        for r in trans:
            if r.spec["cost"] and not r.has_co and stages[r.id] == "escalated":
                r.has_co = True
                cos.append(dict(id=f"{'CO' if r.spec['key'] in ('design_change','unforeseen_conditions') else 'PCO'}-{len(cos)+1:02d}",
                                desc=f"{r.title}: {r.fmt(r.spec['escal'][0])}", amount=r.amount, days=r.days, status="Pending Submission", risk_id=r.id, shown=False))
        co_rows = []
        for c in cos:
            if c["status"] == "Pending Submission" and c["shown"]: c["status"] = "Under Review"
            elif c["status"] == "Under Review" and rng.random() < 0.45: c["status"] = rng.choice(["Approved", "Approved", "Rejected"])
            if c["status"] not in ("Approved", "Rejected") or not c.get("done"):
                co_rows.append(dict(c))
            if c["status"] in ("Approved", "Rejected"): c["done"] = True
            c["shown"] = True
        cum_pct = 100 * sum(c["amount"] for c in cos if c["status"] != "Rejected") / proj["contract"]
        # --- progress / schedule
        base = min(98.0, 100.0 * (m + 1) / (M + 1))
        mult = {"raised": 0.3, "escalated": 1.0, "mitigating": 0.6, "closed": 0.2}
        behind = int(round(sum(r.days * mult[stages[r.id]] for r in active if r.days))) + (rng.randint(0, 1) if active else 0)
        actual = max(prev_actual + 0.5, base - behind * 0.5 + rng.uniform(-0.4, 0.4))
        actual = round(min(actual, base + 0.5), 0); prev_actual = actual
        top = sorted([r for r in active if r.days and stages[r.id] != "closed"], key=lambda r: -r.days * mult[stages[r.id]])[:2]
        cause = " and ".join(r.spec["aliases"][0].lower() for r in top) if top else "normal site variability"
        # --- materials
        mat_rows, mat_remaining = [], []
        for i, (nm, unit, total) in enumerate(t["mats"]):
            used = min(total, total * actual / 100 * rng.uniform(0.97, 1.03)) if m else 0
            used = max(used, mats_used[i]); mats_used[i] = used
            rem = total - used
            rf = "On schedule. No issues."
            for r in active:
                if stages[r.id] in ("raised", "escalated") and ((i == 0 and r.spec["key"] == "supply_delay") or (i == 1 and r.spec["key"] == "price_escalation")):
                    rf = r.fmt(r.spec["raised" if stages[r.id] == "raised" else "escal"][0]) + " "
            mat_rows.append([nm, f"{total:,} {unit}", f"{used:,.0f} {unit}", f"{rem:,.0f} {unit}", rf.strip()])
            mat_remaining.append((nm, unit, round(rem)))
        # --- inconsistencies
        inj, noise = [], rng.random() < args.noise
        kinds = []
        if cos: kinds.append("cumulative_pct")
        if behind > 0: kinds.append("days_behind")
        if m > 0: kinds.append("material_qty")
        kind = rng.choice(kinds) if (noise and kinds) else None
        stated_cum, stated_behind = cum_pct, behind
        mi = 0; stated_qty = mat_remaining[mi][2]
        if kind == "cumulative_pct":
            stated_cum = cum_pct * 1.6 + 0.8; inj.append(dict(kind=kind, stated=round(stated_cum, 1), correct=round(cum_pct, 1)))
        elif kind == "days_behind":
            stated_behind = behind + rng.randint(2, 4); inj.append(dict(kind=kind, stated=stated_behind, correct=behind))
        elif kind == "material_qty":
            stated_qty = int(round(mat_remaining[mi][2] * 1.18 + 5)); inj.append(dict(kind=kind, material=mat_remaining[mi][0], stated=stated_qty, correct=mat_remaining[mi][2]))
        # --- payments
        pay_note = "No payment applications have been submitted to date."
        if m >= 1 and m % 2 == 1:
            claimed = round(proj["contract"] * max(actual - last_claim_pct, 1) / 100 * rng.uniform(0.95, 1.02), 2); last_claim_pct = actual
            prs.append(dict(no=f"PR-{len(prs)+1:02d}", period=d - timedelta(days=1), claimed=claimed, certified=None, status=f"Submitted {dmed(d)}"))
        elif m >= 2 and prs and prs[-1]["certified"] is None:
            prs[-1]["certified"] = round(prs[-1]["claimed"] * rng.uniform(0.94, 0.99), 2); prs[-1]["status"] = f"Paid {dmed(d - timedelta(days=3))}"
        pr_rows = [[p["no"], dshort(p["period"]), money(p["claimed"]), money(p["certified"]) if p["certified"] else "Under review", p["status"]] for p in prs[-2:]]
        # --- attendance
        roster = list(people.values())
        absent = [dict(name=f"K. Williams ({proj['utility']})")] if rng.random() < 0.5 else []
        att = [p for p in roster]
        if rng.random() < 0.15 and m > 0:
            gone = rng.choice([p for p in roster if p["role"].startswith(("Procurement", "Site"))]); att = [p for p in roster if p is not gone]
            absent.append(dict(name=f"{gone['name']} ({gone['org'].split()[0]})"))
        # --- risk rows
        risk_rows = []
        for r in active:
            stg = stages[r.id]; L, I, S = r.rating(stg)
            alias = r.title if rng.random() < 0.55 else rng.choice(r.spec["aliases"])
            pk = {"raised": "raised", "escalated": "escal", "mitigating": "mit", "closed": "closed"}[stg]
            phrase = r.fmt(rng.choice(r.spec[pk]))
            risk_rows.append(dict(risk=r, stage=stg, L=L, I=I, S=S, rag=rag(S) if stg != "closed" else "Closed", alias=alias, phrase=phrase))
            h = register[r.id]; h["history"].append(dict(meeting=m + 1, stage=stg, score=S))
            if h["first_meeting"] is None: h["first_meeting"] = m + 1
            if stg == "closed": h["closed_meeting"] = m + 1
        work = " ".join(w[0].upper() + w[1:] + "." for w in [x.format(n=rng.randint(40, 160), k=rng.randint(1, 4)) for x in rng.sample(t["work"], 2)]) if m else "Mobilization and site establishment."
        # --- store
        meetings.append(dict(m=m + 1, date=d, prev_rows=prev_rows, new_actions=new_actions, carried=carried, co_rows=co_rows,
                             cum_pct=cum_pct, stated_cum=stated_cum, actual=actual, base=base, behind=behind, stated_behind=stated_behind,
                             cause=cause, mat_rows=mat_rows, mat_remaining=mat_remaining, stated_qty=stated_qty, pr_rows=pr_rows, pay_note=pay_note,
                             att=att, absent=absent, risk_rows=risk_rows, work=work, inj=inj, filler=rng.sample(FILLER, 2),
                             look=rng.sample(t["look"], 2)))
        prev_stage = stages
    return meetings, list(register.values())

# ----------------------------------------------------------------------------- rendering
def render_pdf(path, proj, mt):
    t, people = proj["type"], proj["people"]
    def footer(c, doc):
        c.saveState(); c.setFont("Helvetica", 7); c.setFillColor(colors.grey)
        c.drawString(0.7 * inch, 0.45 * inch, f"City of {proj['city']} • {proj['no']} • Page {doc.page} • Synthetic sample for prototype development")
        c.restoreState()
    doc = SimpleDocTemplate(path, pagesize=letter, leftMargin=0.7 * inch, rightMargin=0.7 * inch, topMargin=0.6 * inch, bottomMargin=0.7 * inch,
                            title=f"{proj['no']}_Meeting_{mt['m']}", author=people["Project Coordinator"]["name"])
    d = mt["date"]; s = []
    end_t = (int(proj["t0"][:2]) * 60 + int(proj["t0"][3:]) + proj["dur"])
    s += [P(f"CITY OF {proj['city'].upper()}", TITLE), P(t["dept"], BASE), Spacer(1, 4), P("CONSTRUCTION PROGRESS MEETING MINUTES", ParagraphStyle("x", parent=H2, fontSize=12)),
          P(f"[b]{proj['name']}[/b]    Project No: {proj['no']}", BASE),
          P(f"Weekly Progress Meeting #{mt['m']}    Date: {dlong(d)}", BASE),
          P(f"{proj['t0']} - {end_t//60:02d}:{end_t%60:02d}    Location: Site Trailer, {proj['addr']} {proj['street']}", BASE),
          P(f"Prepared by: {people['Project Coordinator']['name']}, Project Coordinator ({proj['engineer']})    Distributed: {dlong(d)}, {rng_time(mt['m'])}", BASE)]
    s += [P("1. ATTENDEES", H2)]
    rows = [[p["name"], p["org"], p["role"], p["email"] + (" / " + p["phone"] if p["phone"] else "")] for p in mt["att"]]
    s += [table(rows, [1.2, 1.9, 1.8, 2.2], header=False)]
    s += [P("Absent: " + (", ".join(a["name"] for a in mt["absent"]) + " — notified in advance." if mt["absent"] else "None."), BASE)]
    s += [P("2. REVIEW OF PREVIOUS ACTION ITEMS", H2)]
    if mt["prev_rows"]:
        s += [table([["Item #", "Description", "Owner", "Due", "Status"]] + [[a["id"], a["desc"], a["owner"], dshort(a["due"]), a["status"]] for a in mt["prev_rows"]], [0.6, 3.4, 1.3, 0.9, 0.9])]
    else: s += [P("No open action items from the previous meeting.", BASE)]
    s += [P("3. PROGRESS and SCHEDULE UPDATE".replace("and", "&"), H2)]
    s += [P(f"[b]Work Completed (Week of {dmed(d - timedelta(days=10))} - {dshort(d - timedelta(days=5))}):[/b] {mt['work']} Overall progress against baseline: {mt['actual']:.0f}% complete (baseline target was {mt['base']:.0f}%).", BASE)]
    if mt["behind"] > 0:
        s += [P(f"[b]Schedule Variance:[/b] Currently {mt['behind']} working days behind baseline substantial completion date of {dlong(proj['end'])}. Primary cause: {mt['cause']}.", BASE)]
    else:
        s += [P(f"[b]Schedule Variance:[/b] Project is on schedule against the baseline substantial completion date of {dlong(proj['end'])}.", BASE)]
    s += [P(f"[b]Two-Week Lookahead:[/b] {' '.join(mt['look'])}", BASE), P("[b]Other Items:[/b] " + " ".join(mt["filler"]), BASE)]
    s += [P("4. MATERIALS STATUS", H2), table([["Material", "Total Required", "Used to Date", "Remaining", "Status / Remarks"]] + mt["mat_rows"], [1.7, 1.0, 1.0, 1.0, 2.4])]
    nm, unit, _ = mt["mat_remaining"][0]
    s += [P(f"Discussion: Contractor confirmed approximately {mt['stated_qty']:,} {unit} of {nm} remaining to be installed.", BASE)]
    s += [P("5. CHANGE ORDERS / POTENTIAL COST IMPACTS", H2)]
    if mt["co_rows"]:
        s += [table([["Ref", "Description", "Cost Impact", "Time Impact", "Status"]] + [[c["id"], c["desc"], "+" + money(c["amount"]) + " (est.)", f"+{c['days']} days" if c["days"] else "None", c["status"]] for c in mt["co_rows"]], [0.7, 3.3, 1.1, 0.9, 1.1])]
        s += [P(f"Cumulative approved and pending changes are approximately {mt['stated_cum']:.1f}% of the original contract value ({money(proj['contract'])}). Any unit price adjustment must follow the contract change process and requires finance review if cumulative changes exceed 5%.", BASE)]
    else: s += [P("No change orders or potential changes are currently under review.", BASE)]
    s += [P("6. PAYMENT REQUESTS", H2)]
    if mt["pr_rows"]:
        s += [table([["Pay App #", "Period Ending", "Amount Claimed", "Certified Amount", "Status"]] + mt["pr_rows"], [0.9, 1.4, 1.5, 1.5, 1.5]),
              P("Holdback remains at 10%. No payment will be certified against unapproved change order work.", BASE)]
    else: s += [P(mt["pay_note"], BASE)]
    s += [P("7. NEW / CARRIED ACTION ITEMS", H2)]
    na = mt["carried"] + mt["new_actions"]
    if na: s += [table([["Item #", "Description", "Owner", "Due", "Status"]] + [[a["id"], a["desc"] + (" (carried forward)" if a["carried"] else ""), a["owner"], dshort(a["due"]), "Open"] for a in na], [0.6, 3.4, 1.3, 0.9, 0.9])]
    else: s += [P("No new action items.", BASE)]
    s += [P("8. KEY RISKS and ISSUES".replace("and", "&"), H2)]
    if mt["risk_rows"]:
        for rr in mt["risk_rows"]:
            r = rr["risk"]
            txt = f"• {r.spec['cat']} Risk ({rr['rag']}): {rr['alias']}. {rr['phrase']}"
            s += [P(txt, BASE)]
        if mt["behind"] > 0:
            s += [P(f"• Schedule Status: {mt['stated_behind']} days behind baseline. Further delays could push substantial completion past {dlong(proj['end'])}.", BASE)]
    else: s += [P("• No significant risks or issues identified this week.", BASE)]
    nxt = d + timedelta(weeks=1)
    s += [P("9. NEXT MEETING", H2), P(f"Next Weekly Progress Meeting: {dlong(nxt)}, {proj['t0']}, Site Trailer.", BASE),
          P(f"Distribution: All attendees + {proj['utility']} + Project File. Corrections or clarifications to these minutes must be submitted in writing within three (3) business days.", BASE), Spacer(1, 6),
          P("These minutes are a record of discussion and decisions. They do not constitute a change order or direction to proceed with extra work unless formally issued under the Contract.", ParagraphStyle("i", parent=BASE, fontSize=7.5, textColor=colors.grey))]
    doc.build(s, onFirstPage=footer, onLaterPages=footer)

def rng_time(m): return f"{(13 + m % 5):d}:{(10 + 7 * m) % 60:02d} PM"

def to_scan(src, dst, rng):
    import pypdfium2 as pdfium
    from PIL import Image, ImageFilter
    pdf = pdfium.PdfDocument(src); imgs = []
    for i in range(len(pdf)):
        im = pdf[i].render(scale=2.0).to_pil().convert("L")
        im = im.rotate(rng.uniform(-0.8, 0.8), expand=False, fillcolor=255, resample=Image.BICUBIC).filter(ImageFilter.GaussianBlur(0.5))
        imgs.append(im.convert("RGB"))
    pdf.close()
    imgs[0].save(dst, "PDF", save_all=True, append_images=imgs[1:], resolution=144.0)

# ----------------------------------------------------------------------------- ground truth
def ground_truth(proj, mt, fname, scanned):
    return dict(file=fname, project_no=proj["no"], project_name=proj["name"], meeting_no=mt["m"], date=mt["date"].isoformat(), scanned=scanned,
        metrics=dict(pct_complete=mt["actual"], baseline_pct=round(mt["base"]), days_behind=mt["behind"], cumulative_change_pct=round(mt["cum_pct"], 2)),
        risks=[dict(risk_id=x["risk"].id, canonical_title=x["risk"].title, title_as_written=x["alias"], category=x["risk"].spec["cat"], stage=x["stage"],
                    likelihood=x["L"], impact=x["I"], score=x["S"], rag=x["rag"], owner_role=x["risk"].owner_role, evidence=x["phrase"]) for x in mt["risk_rows"]],
        action_items=[dict(id=a["id"], description=a["desc"], owner=a["owner"], due=a["due"].isoformat(), carried=a["carried"], risk_id=a["risk_id"]) for a in mt["carried"] + mt["new_actions"]],
        previous_actions=[dict(id=a["id"], status=a["status"]) for a in mt["prev_rows"]],
        change_orders=[dict(id=c["id"], amount=c["amount"], days=c["days"], status=c["status"], risk_id=c["risk_id"]) for c in mt["co_rows"]],
        injected_inconsistencies=mt["inj"])

# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description="Generate synthetic construction meeting-minutes PDFs (offline).")
    ap.add_argument("--projects", type=int, default=12); ap.add_argument("--meetings", type=int, default=25)
    ap.add_argument("--out", default="synthetic_minutes"); ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--scan-ratio", type=float, default=0.25, help="fraction of PDFs rendered image-only (needs pypdfium2 + pillow)")
    ap.add_argument("--noise", type=float, default=0.15, help="probability of a planted inconsistency per meeting")
    ap.add_argument("--no-zip", action="store_true")
    args = ap.parse_args()
    rng = random.Random(args.seed)
    scan_ok = True
    try: import pypdfium2, PIL  # noqa
    except ImportError:
        scan_ok = False
        if args.scan_ratio > 0: print("NOTE: pypdfium2/pillow not installed -> all PDFs will have a text layer. (pip install pypdfium2 pillow)")
    pdf_dir, gt_dir = os.path.join(args.out, "pdfs"), os.path.join(args.out, "ground_truth")
    if os.path.isdir(args.out): shutil.rmtree(args.out)
    os.makedirs(pdf_dir); os.makedirs(gt_dir)
    used, manifest, total = set(), [], 0
    for pi in range(args.projects):
        proj = make_project(rng, pi, args.meetings, used)
        meetings, register = simulate_project(rng, proj, args)
        os.makedirs(os.path.join(pdf_dir, proj["no"]), exist_ok=True)
        for mt in meetings:
            fname = f"{proj['no']}_MM-{mt['m']:02d}_{mt['date'].isoformat()}.pdf"
            path = os.path.join(pdf_dir, proj["no"], fname)
            scanned = scan_ok and rng.random() < args.scan_ratio
            if scanned:
                tmp = path + ".tmp"; render_pdf(tmp, proj, mt); to_scan(tmp, path, rng); os.remove(tmp)
            else: render_pdf(path, proj, mt)
            gt = ground_truth(proj, mt, fname, scanned)
            with open(os.path.join(gt_dir, fname.replace(".pdf", ".json")), "w", encoding="utf-8") as f: json.dump(gt, f, indent=2)
            manifest.append([fname, proj["no"], proj["name"], mt["m"], mt["date"].isoformat(), scanned, len(gt["risks"]), len(gt["injected_inconsistencies"])])
            total += 1
        with open(os.path.join(gt_dir, f"{proj['no']}_risk_register.json"), "w", encoding="utf-8") as f:
            json.dump(dict(project_no=proj["no"], project_name=proj["name"], contract_value=proj["contract"], risks=register), f, indent=2)
        print(f"[{pi+1}/{args.projects}] {proj['no']} - {len(meetings)} meetings")
    with open(os.path.join(args.out, "manifest.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["file", "project_no", "project_name", "meeting_no", "date", "scanned", "n_risks", "n_inconsistencies"]); w.writerows(manifest)
    if not args.no_zip:
        zp = shutil.make_archive(args.out, "zip", args.out); print(f"ZIP: {os.path.abspath(zp)}")
    print(f"Done: {total} PDFs in {os.path.abspath(pdf_dir)}")

if __name__ == "__main__":
    main()

from __future__ import annotations

"""
CARC (Claim Adjustment Reason Codes) and RARC (Remittance Advice Remark Codes)
reference data.

Source: X12 835 standard, published by Washington Publishing Company (WPC).
https://www.wpc-edi.com/reference/

These are the standardized codes insurers use to explain WHY a claim was
adjusted or denied. Every denial in the U.S. healthcare system references
one or more of these codes.

CARC codes are grouped into categories:
  - CO  (Contractual Obligation) — adjustments per provider-payer contract
  - PR  (Patient Responsibility) — patient owes this amount
  - OA  (Other Adjustment) — catch-all
  - PI  (Payer Initiated) — payer-side reduction, not contract or patient
  - CR  (Correction/Reversal) — fixing a prior adjudication
"""

# ── Top CARC codes by frequency in real-world claims ──────────────────
# This is a curated set of the most common and analytically useful codes.
# In production you'd scrape the full WPC list (~400 codes).

CARC_CODES: dict[str, dict] = {
    "1": {
        "description": "Deductible amount",
        "category": "PR",
        "plain_english": "Patient hasn't met their deductible yet — they owe this portion.",
        "actionable_insight": "Verify patient deductible status before submitting. Consider collecting at time of service.",
    },
    "2": {
        "description": "Coinsurance amount",
        "category": "PR",
        "plain_english": "Patient's share of the cost after deductible.",
        "actionable_insight": "Expected adjustment. Ensure correct coinsurance rate per plan.",
    },
    "3": {
        "description": "Co-payment amount",
        "category": "PR",
        "plain_english": "Patient's flat copay for this visit/service.",
        "actionable_insight": "Collect copay at time of service to reduce A/R.",
    },
    "4": {
        "description": "The procedure code is inconsistent with the modifier used, or a required modifier is missing.",
        "category": "CO",
        "plain_english": "The billing code and its modifier don't match, or a modifier is missing.",
        "actionable_insight": "Review modifier usage. Common with same-day E/M + procedure combos. Check if modifier 25 or 59 is needed.",
    },
    "5": {
        "description": "The procedure code/bill type is inconsistent with the place of service.",
        "category": "CO",
        "plain_english": "Where the service was performed doesn't match the code used.",
        "actionable_insight": "Verify place of service code matches facility type. Common when telehealth codes are used for in-person visits.",
    },
    "11": {
        "description": "The diagnosis is inconsistent with the procedure.",
        "category": "CO",
        "plain_english": "The diagnosis code doesn't justify the procedure that was performed.",
        "actionable_insight": "Ensure ICD-10 diagnosis supports the CPT procedure. Check LCD/NCD coverage rules for the payer.",
    },
    "16": {
        "description": "Claim/service lacks information or has submission/billing error(s).",
        "category": "CO",
        "plain_english": "The claim is missing required information or has errors.",
        "actionable_insight": "Review claim for missing fields: auth numbers, referring provider NPI, accident date, onset date.",
    },
    "18": {
        "description": "Exact duplicate claim/service.",
        "category": "CO",
        "plain_english": "This exact claim was already submitted and processed.",
        "actionable_insight": "Check claim status before resubmitting. May indicate a clearinghouse or PM system issue.",
    },
    "22": {
        "description": "This care may be covered by another payer per coordination of benefits.",
        "category": "CO",
        "plain_english": "Another insurance should be billed first.",
        "actionable_insight": "Verify coordination of benefits. Update primary/secondary payer order in PM system.",
    },
    "23": {
        "description": "The impact of prior payer(s) adjudication including payments and/or adjustments.",
        "category": "CO",
        "plain_english": "Adjustment based on what the primary insurer already paid.",
        "actionable_insight": "Normal for secondary claims. Ensure primary EOB is attached.",
    },
    "27": {
        "description": "Expenses incurred after coverage terminated.",
        "category": "CO",
        "plain_english": "Patient's insurance had ended by the time they were treated.",
        "actionable_insight": "Run real-time eligibility check before every encounter. Flag coverage gaps proactively.",
    },
    "29": {
        "description": "The time limit for filing has expired.",
        "category": "CO",
        "plain_english": "The claim was submitted too late — past the payer's filing deadline.",
        "actionable_insight": "Identify claims approaching timely filing limits. Medicare: 12 months. Commercial: varies (90-365 days).",
    },
    "45": {
        "description": "Charge exceeds fee schedule/maximum allowable or contracted/legislated fee arrangement.",
        "category": "CO",
        "plain_english": "You billed more than the payer's allowed amount for this service.",
        "actionable_insight": "Expected for most claims. Write off the difference per contract. If unexpected, review fee schedule.",
    },
    "50": {
        "description": "These are non-covered services because this is not deemed a 'medical necessity' by the payer.",
        "category": "CO",
        "plain_english": "The payer says this treatment wasn't medically necessary.",
        "actionable_insight": "Strengthen clinical documentation. Include failed conservative treatments, symptom duration, functional limitations. Check LCD/NCD criteria.",
    },
    "96": {
        "description": "Non-covered charge(s). At least one Remark Code must be provided.",
        "category": "CO",
        "plain_english": "This service isn't covered under the patient's plan at all.",
        "actionable_insight": "Verify coverage before rendering service. May need ABN (Advance Beneficiary Notice) for Medicare.",
    },
    "97": {
        "description": "The benefit for this service is included in the payment/allowance for another service/procedure.",
        "category": "CO",
        "plain_english": "This procedure is 'bundled' into another one and can't be billed separately.",
        "actionable_insight": "Review CCI (Correct Coding Initiative) edits. Check if modifier 59/XE/XS/XP/XU can unbundle. Document distinct services.",
    },
    "109": {
        "description": "Claim/service not covered by this payer/contractor.",
        "category": "CO",
        "plain_english": "This payer doesn't cover this type of service at all.",
        "actionable_insight": "Verify payer covers the service category. May need to bill a different entity or plan.",
    },
    "119": {
        "description": "Benefit maximum for this time period or occurrence has been reached.",
        "category": "CO",
        "plain_english": "Patient has used up their allowed visits or dollar amount for this service.",
        "actionable_insight": "Track patient benefit utilization. Notify patient before rendering services that may not be covered.",
    },
    "167": {
        "description": "This (these) diagnosis(es) is (are) not covered.",
        "category": "CO",
        "plain_english": "The diagnosis code itself isn't a covered condition under this plan.",
        "actionable_insight": "Review diagnosis coding. May need a more specific ICD-10 code or different primary diagnosis.",
    },
    "197": {
        "description": "Precertification/authorization/notification/pre-treatment absent.",
        "category": "CO",
        "plain_english": "Prior authorization was required but wasn't obtained before treatment.",
        "actionable_insight": "Build prior auth verification into scheduling workflow. Automate auth requirement lookups by CPT + payer.",
    },
    "204": {
        "description": "This service/equipment/drug is not covered under the patient's current benefit plan.",
        "category": "CO",
        "plain_english": "The patient's specific plan doesn't include this service.",
        "actionable_insight": "Run benefit verification before service. Consider alternative covered approaches.",
    },
    "236": {
        "description": "This procedure or procedure/modifier combination is not compatible with another procedure or procedure/modifier combination provided on the same day.",
        "category": "CO",
        "plain_english": "These two procedures can't both be billed on the same day per payer rules.",
        "actionable_insight": "Review NCCI edits for procedure pairs. May need modifier or to split across dates of service.",
    },
    "242": {
        "description": "Services not provided by network/primary care providers.",
        "category": "CO",
        "plain_english": "Patient went out of network without authorization.",
        "actionable_insight": "Verify network status before referrals. Obtain out-of-network authorization when needed.",
    },
    "252": {
        "description": "An attachment/other documentation is required to adjudicate this claim/service.",
        "category": "CO",
        "plain_english": "The payer needs additional documentation before they'll process this.",
        "actionable_insight": "Submit required docs proactively with initial claim. Common for high-cost procedures, DME, and unusual diagnoses.",
    },
}

# ── Denial category groupings (for dashboard aggregation) ─────────────
DENIAL_CATEGORIES = {
    "Medical Necessity": ["50", "167"],
    "Prior Auth Missing": ["197"],
    "Coding/Modifier Errors": ["4", "5", "11", "236"],
    "Missing Information": ["16", "252"],
    "Duplicate/Timely Filing": ["18", "29"],
    "Coverage/Eligibility": ["22", "27", "96", "109", "119", "204", "242"],
    "Bundling": ["97"],
    "Patient Responsibility": ["1", "2", "3"],
    "Fee Schedule": ["45"],
}


def get_carc_description(code: str) -> str:
    """Return human-readable description for a CARC code."""
    entry = CARC_CODES.get(str(code))
    if entry:
        return entry["description"]
    return f"Unknown CARC code: {code}"


def get_actionable_insight(code: str) -> str:
    """Return the actionable billing insight for a CARC code."""
    entry = CARC_CODES.get(str(code))
    if entry:
        return entry["actionable_insight"]
    return "No insight available for this code."


def get_denial_category(code: str) -> str:
    """Map a CARC code to its high-level denial category."""
    for category, codes in DENIAL_CATEGORIES.items():
        if str(code) in codes:
            return category
    return "Other"


# ── Specialty-to-CARC risk mapping ──────────────────────────────────
# Based on published denial rate research (MGMA, AAPC, CMS data briefs).
# Maps each specialty to the CARC categories most likely to cause denials.
SPECIALTY_CARC_RISK = {
    "Orthopedic Surgery": {
        "top_risks": ["Medical Necessity", "Prior Auth Missing", "Bundling"],
        "notes": "High-cost procedures (joint replacements, spinal fusions) frequently require prior auth. Bundling issues common with same-day imaging + surgery.",
    },
    "Cardiology": {
        "top_risks": ["Medical Necessity", "Coding/Modifier Errors", "Prior Auth Missing"],
        "notes": "Complex procedure combinations (cath lab + echo + stress test) create modifier and bundling challenges. Many cardiac procedures require prior auth.",
    },
    "Pain Management": {
        "top_risks": ["Medical Necessity", "Prior Auth Missing", "Coverage/Eligibility"],
        "notes": "Injection procedures under heavy LCD scrutiny. Medical necessity documentation is the #1 denial trigger for pain management.",
    },
    "Ophthalmology": {
        "top_risks": ["Bundling", "Coding/Modifier Errors", "Medical Necessity"],
        "notes": "High-volume injection codes (anti-VEGF) with complex billing. Modifier errors common with bilateral procedures.",
    },
    "Dermatology": {
        "top_risks": ["Coding/Modifier Errors", "Medical Necessity", "Missing Information"],
        "notes": "Destruction/excision codes require precise size documentation. Path lab bundling issues are common.",
    },
    "Gastroenterology": {
        "top_risks": ["Bundling", "Coding/Modifier Errors", "Prior Auth Missing"],
        "notes": "Endoscopy procedure bundles are a constant source of denials. CCI edits apply to most scope combinations.",
    },
    "Neurology": {
        "top_risks": ["Medical Necessity", "Prior Auth Missing", "Coverage/Eligibility"],
        "notes": "EEG monitoring, EMG studies, and advanced imaging require strong medical necessity documentation.",
    },
    "Psychiatry": {
        "top_risks": ["Coverage/Eligibility", "Duplicate/Timely Filing", "Missing Information"],
        "notes": "Visit frequency limits and benefit maximums are the primary denial drivers. Timely filing is a chronic issue.",
    },
    "Urology": {
        "top_risks": ["Prior Auth Missing", "Medical Necessity", "Bundling"],
        "notes": "Surgical procedures often require prior auth. In-office procedure bundling with E/M visits is a common denial trigger.",
    },
    "Pulmonary Disease": {
        "top_risks": ["Medical Necessity", "Coding/Modifier Errors", "Missing Information"],
        "notes": "PFT combinations and bronchoscopy add-ons create bundling complexity. Sleep study auth requirements vary by MAC.",
    },
}

# Default for specialties not explicitly mapped
_DEFAULT_CARC_RISK = {
    "top_risks": ["Fee Schedule", "Coverage/Eligibility", "Missing Information"],
    "notes": "Standard denial patterns. Focus on eligibility verification and clean claim submission.",
}


def get_specialty_carc_risks(specialty: str) -> dict:
    """Return the CARC risk profile for a specialty."""
    return SPECIALTY_CARC_RISK.get(specialty, _DEFAULT_CARC_RISK)


def get_all_specialty_risks_df():
    """Return all specialty CARC risk mappings as a DataFrame."""
    import pandas as pd
    rows = []
    for spec, info in SPECIALTY_CARC_RISK.items():
        rows.append({
            "specialty": spec,
            "primary_risk": info["top_risks"][0] if info["top_risks"] else "",
            "secondary_risk": info["top_risks"][1] if len(info["top_risks"]) > 1 else "",
            "tertiary_risk": info["top_risks"][2] if len(info["top_risks"]) > 2 else "",
            "notes": info["notes"],
        })
    return pd.DataFrame(rows)


def infer_denial_risks_from_data(df) -> dict[str, list[str]]:
    """
    Infer likely denial categories for each specialty based on actual
    procedure mix in the data.

    Logic:
    - High % of J-codes (drugs) → Bundling, Medical Necessity
    - High % of surgical codes (2xxxx-6xxxx) → Prior Auth, Medical Necessity
    - High % of E/M codes (992xx) → Coding/Modifier Errors
    - High % of imaging (7xxxx) → Prior Auth, Coverage/Eligibility
    - High allowed_gap → Medical Necessity, Prior Auth Missing
    """
    import pandas as pd

    results = {}

    for specialty, grp in df.groupby("specialty"):
        if "procedure_code" not in grp.columns:
            continue

        codes = grp["procedure_code"].fillna("")
        volumes = grp["total_services"]
        total = volumes.sum()
        if total == 0:
            continue

        # Categorize procedures weighted by claim volume
        j_pct = volumes[codes.str.startswith("J")].sum() / total
        surgical_pct = volumes[codes.str.match(r"^[2-6]\d{4}")].sum() / total
        em_pct = volumes[codes.str.startswith("992")].sum() / total
        imaging_pct = volumes[codes.str.startswith("7")].sum() / total

        # Allowed gap signal (claim-volume-weighted)
        if "allowed_gap_pct" in grp.columns:
            avg_allowed_gap = (grp["allowed_gap_pct"] * volumes).sum() / total
        else:
            avg_allowed_gap = 0

        risks = []

        # High drug injection volume → bundling and medical necessity
        if j_pct > 0.15:
            risks.append("Bundling")
            risks.append("Medical Necessity")

        # High surgical volume → prior auth
        if surgical_pct > 0.3:
            risks.append("Prior Auth Missing")
            if "Medical Necessity" not in risks:
                risks.append("Medical Necessity")

        # High E/M volume → coding errors
        if em_pct > 0.2:
            risks.append("Coding/Modifier Errors")

        # High imaging → prior auth, coverage
        if imaging_pct > 0.15:
            if "Prior Auth Missing" not in risks:
                risks.append("Prior Auth Missing")
            risks.append("Coverage/Eligibility")

        # High allowed gap → medical necessity
        if avg_allowed_gap > 3.0:
            if "Medical Necessity" not in risks:
                risks.insert(0, "Medical Necessity")

        # Default if nothing triggered
        if not risks:
            risks = ["Fee Schedule", "Coverage/Eligibility"]

        results[specialty] = risks[:3]  # top 3

    return results


def get_all_codes_df():
    """Return all CARC codes as a pandas DataFrame for dashboard use."""
    import pandas as pd

    rows = []
    for code, info in CARC_CODES.items():
        rows.append({
            "carc_code": code,
            "description": info["description"],
            "group": info["category"],
            "plain_english": info["plain_english"],
            "actionable_insight": info["actionable_insight"],
            "denial_category": get_denial_category(code),
        })
    return pd.DataFrame(rows)

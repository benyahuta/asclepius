from __future__ import annotations

"""
Denial pattern analysis engine.

Takes processed CMS claims data and computes aggregate denial/adjustment
patterns across multiple dimensions:
  - By specialty
  - By procedure code
  - By geography (state)
  - By place of service

Two key metrics:

1. Payment gap % = (submitted charge - Medicare payment) / submitted charge.
   This is always high (60-80%) for Medicare because providers bill at
   chargemaster rates. It's useful for RELATIVE comparison between
   specialties/procedures, not as an absolute indicator.

2. Allowed-to-payment gap = what Medicare said it would pay vs. what it
   actually paid. A nonzero gap here is a stronger signal of actual
   denials, downcoding, or adjustments.

The denial risk flag is percentile-based (top 20% of payment gaps),
which adapts to the actual data distribution.
"""

import pandas as pd
import numpy as np
from config import TOP_N_PROCEDURES

# Minimum claim volume to include a procedure in rankings.
# Filters out low-volume noise that would otherwise dominate "top risk" lists.
MIN_PROCEDURE_CLAIMS = 500

# Minimum claim volume for state-level analysis
MIN_STATE_CLAIMS = 1000

# Exclude non-US territories/military from geographic analysis
EXCLUDE_STATES = {"AE", "AP", "AA", "GU", "VI", "AS", "MP", "PR"}


def load_processed_data(path: str | None = None) -> pd.DataFrame:
    """Load processed parquet data."""
    from config import PROCESSED_DIR

    if path:
        return pd.read_parquet(path)

    default_path = PROCESSED_DIR / "provider_service_2023_processed.parquet"
    if default_path.exists():
        return pd.read_parquet(default_path)

    raise FileNotFoundError(
        f"No processed data found at {default_path}. Run ingest.py first."
    )


def _weighted_avg(group: pd.DataFrame, value_col: str, weight_col: str) -> float:
    """Compute a claim-volume-weighted average, avoiding mean-of-means bias."""
    weights = group[weight_col]
    total_w = weights.sum()
    if total_w == 0:
        return 0.0
    return (group[value_col] * weights).sum() / total_w


def denial_rate_by_specialty(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute denial risk metrics aggregated by provider specialty.
    Uses claim-volume-weighted averages (not simple mean-of-means).
    """
    results = []
    for specialty, grp in df.groupby("specialty"):
        total_svc = grp["total_services"].sum()
        results.append({
            "specialty": specialty,
            "total_claims": total_svc,
            "high_risk_claims": (grp["denial_risk_flag"] * grp["total_services"]).sum(),
            "avg_payment_gap_pct": _weighted_avg(grp, "payment_gap_pct", "total_services"),
            "avg_allowed_gap_pct": _weighted_avg(grp, "allowed_gap_pct", "total_services"),
            "avg_submitted_charge": _weighted_avg(grp, "avg_submitted_charge", "total_services"),
            "avg_medicare_payment": _weighted_avg(grp, "avg_medicare_payment", "total_services"),
            "total_revenue_at_risk": grp["total_revenue_at_risk"].sum(),
            "provider_count": grp["provider_npi"].nunique(),
        })

    grouped = pd.DataFrame(results)

    grouped["denial_risk_rate"] = (
        grouped["high_risk_claims"] / grouped["total_claims"].replace(0, np.nan) * 100
    )

    # Weighted overall average for fair comparison
    overall_avg = _weighted_avg(df, "payment_gap_pct", "total_services")
    grouped["gap_vs_avg"] = grouped["avg_payment_gap_pct"] - overall_avg

    return grouped.sort_values("avg_payment_gap_pct", ascending=False)


def denial_rate_by_procedure(
    df: pd.DataFrame,
    top_n: int = TOP_N_PROCEDURES,
    min_claims: int = MIN_PROCEDURE_CLAIMS,
) -> pd.DataFrame:
    """
    Compute denial risk metrics by procedure code (HCPCS/CPT).
    Uses claim-volume-weighted averages. Filters out low-volume procedures.
    """
    results = []
    for (code, desc), grp in df.groupby(["procedure_code", "procedure_description"]):
        total_svc = grp["total_services"].sum()
        results.append({
            "procedure_code": code,
            "procedure_description": desc,
            "total_claims": total_svc,
            "high_risk_claims": (grp["denial_risk_flag"] * grp["total_services"]).sum(),
            "avg_payment_gap_pct": _weighted_avg(grp, "payment_gap_pct", "total_services"),
            "avg_allowed_gap_pct": _weighted_avg(grp, "allowed_gap_pct", "total_services"),
            "avg_submitted_charge": _weighted_avg(grp, "avg_submitted_charge", "total_services"),
            "avg_medicare_payment": _weighted_avg(grp, "avg_medicare_payment", "total_services"),
            "total_revenue_at_risk": grp["total_revenue_at_risk"].sum(),
            "provider_count": grp["provider_npi"].nunique(),
            "state_count": grp["state"].nunique(),
        })

    grouped = pd.DataFrame(results)
    grouped = grouped[grouped["total_claims"] >= min_claims]

    grouped["denial_risk_rate"] = (
        grouped["high_risk_claims"] / grouped["total_claims"].replace(0, np.nan) * 100
    )

    overall_avg = _weighted_avg(df, "payment_gap_pct", "total_services")
    grouped["gap_vs_avg"] = grouped["avg_payment_gap_pct"] - overall_avg

    return grouped.sort_values("avg_payment_gap_pct", ascending=False).head(top_n)


def denial_rate_by_state(df: pd.DataFrame) -> pd.DataFrame:
    """
    Geographic analysis of denial patterns.

    Excludes military/overseas codes and tiny-sample territories.
    """
    # Filter out non-US territories
    filtered = df[~df["state"].isin(EXCLUDE_STATES)]

    results = []
    for state, grp in filtered.groupby("state"):
        total_svc = grp["total_services"].sum()
        results.append({
            "state": state,
            "total_claims": total_svc,
            "high_risk_claims": (grp["denial_risk_flag"] * grp["total_services"]).sum(),
            "avg_payment_gap_pct": _weighted_avg(grp, "payment_gap_pct", "total_services"),
            "avg_allowed_gap_pct": _weighted_avg(grp, "allowed_gap_pct", "total_services"),
            "total_revenue_at_risk": grp["total_revenue_at_risk"].sum(),
            "provider_count": grp["provider_npi"].nunique(),
            "specialty_count": grp["specialty"].nunique(),
        })

    grouped = pd.DataFrame(results)
    grouped = grouped[grouped["total_claims"] >= MIN_STATE_CLAIMS]

    grouped["denial_risk_rate"] = (
        grouped["high_risk_claims"] / grouped["total_claims"].replace(0, np.nan) * 100
    )

    national_avg = _weighted_avg(filtered, "payment_gap_pct", "total_services")
    grouped["gap_vs_national"] = grouped["avg_payment_gap_pct"] - national_avg

    return grouped.sort_values("avg_payment_gap_pct", ascending=False)


def denial_rate_by_specialty_and_procedure(
    df: pd.DataFrame,
    specialty: str | None = None,
    min_claims: int = 100,
) -> pd.DataFrame:
    """
    Cross-tabulation: which procedures within a specialty have highest risk?
    """
    if specialty:
        df = df[df["specialty"] == specialty]

    results = []
    for (spec, code, desc), grp in df.groupby(["specialty", "procedure_code", "procedure_description"]):
        total_svc = grp["total_services"].sum()
        results.append({
            "specialty": spec,
            "procedure_code": code,
            "procedure_description": desc,
            "total_claims": total_svc,
            "high_risk_claims": (grp["denial_risk_flag"] * grp["total_services"]).sum(),
            "avg_payment_gap_pct": _weighted_avg(grp, "payment_gap_pct", "total_services"),
            "avg_allowed_gap_pct": _weighted_avg(grp, "allowed_gap_pct", "total_services"),
            "avg_submitted_charge": _weighted_avg(grp, "avg_submitted_charge", "total_services"),
            "avg_medicare_payment": _weighted_avg(grp, "avg_medicare_payment", "total_services"),
            "total_revenue_at_risk": grp["total_revenue_at_risk"].sum(),
        })

    grouped = pd.DataFrame(results)
    grouped = grouped[grouped["total_claims"] >= min_claims]

    grouped["denial_risk_rate"] = (
        grouped["high_risk_claims"] / grouped["total_claims"].replace(0, np.nan) * 100
    )

    spec_avg = _weighted_avg(df, "payment_gap_pct", "total_services")
    grouped["gap_vs_specialty_avg"] = grouped["avg_payment_gap_pct"] - spec_avg

    return grouped.sort_values("avg_payment_gap_pct", ascending=False)


def place_of_service_analysis(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compare denial patterns between facility and office settings.
    """
    if "place_of_service_label" not in df.columns:
        return pd.DataFrame()

    results = []
    for label, grp in df.groupby("place_of_service_label"):
        total_svc = grp["total_services"].sum()
        results.append({
            "place_of_service_label": label,
            "total_claims": total_svc,
            "high_risk_claims": (grp["denial_risk_flag"] * grp["total_services"]).sum(),
            "avg_payment_gap_pct": _weighted_avg(grp, "payment_gap_pct", "total_services"),
            "avg_submitted_charge": _weighted_avg(grp, "avg_submitted_charge", "total_services"),
            "avg_medicare_payment": _weighted_avg(grp, "avg_medicare_payment", "total_services"),
        })

    grouped = pd.DataFrame(results)

    grouped["denial_risk_rate"] = (
        grouped["high_risk_claims"] / grouped["total_claims"].replace(0, np.nan) * 100
    )

    return grouped


def compute_risk_scores(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute a composite denial risk score (0-100) for each provider-procedure row.

    This is the kind of feature engineering an AI RCM system would use as input
    to a denial prediction model. The score combines four signals:

    1. Payment gap percentile within the dataset (0-100)
    2. Specialty deviation: how far this row's specialty is from the mean
    3. Geographic deviation: how far this row's state is from the mean
    4. Place-of-service risk: facility vs office gap differential

    Weights reflect relative predictive value based on denial literature:
    payment gap (40%), specialty context (25%), geography (20%), POS (15%).
    """
    scored = df.copy()

    # 1. Payment gap percentile (0-100)
    scored["gap_percentile"] = scored["payment_gap_pct"].rank(pct=True) * 100

    # 2. Specialty deviation score
    spec_avgs = scored.groupby("specialty").apply(
        lambda g: _weighted_avg(g, "payment_gap_pct", "total_services")
    ).reset_index()
    spec_avgs.columns = ["specialty", "spec_avg_gap"]
    scored = scored.merge(spec_avgs, on="specialty", how="left")
    overall_avg = _weighted_avg(scored, "payment_gap_pct", "total_services")
    spec_range = scored["spec_avg_gap"].max() - scored["spec_avg_gap"].min()
    if spec_range > 0:
        scored["spec_deviation_score"] = (
            (scored["spec_avg_gap"] - scored["spec_avg_gap"].min()) / spec_range * 100
        )
    else:
        scored["spec_deviation_score"] = 50.0

    # 3. Geographic deviation score
    state_avgs = scored.groupby("state").apply(
        lambda g: _weighted_avg(g, "payment_gap_pct", "total_services")
    ).reset_index()
    state_avgs.columns = ["state", "state_avg_gap"]
    scored = scored.merge(state_avgs, on="state", how="left")
    state_range = scored["state_avg_gap"].max() - scored["state_avg_gap"].min()
    if state_range > 0:
        scored["geo_deviation_score"] = (
            (scored["state_avg_gap"] - scored["state_avg_gap"].min()) / state_range * 100
        )
    else:
        scored["geo_deviation_score"] = 50.0

    # 4. Place-of-service risk score
    if "place_of_service" in scored.columns:
        pos_avgs = scored.groupby("place_of_service").apply(
            lambda g: _weighted_avg(g, "payment_gap_pct", "total_services")
        ).reset_index()
        pos_avgs.columns = ["place_of_service", "pos_avg_gap"]
        scored = scored.merge(pos_avgs, on="place_of_service", how="left")
        pos_range = scored["pos_avg_gap"].max() - scored["pos_avg_gap"].min()
        if pos_range > 0:
            scored["pos_risk_score"] = (
                (scored["pos_avg_gap"] - scored["pos_avg_gap"].min()) / pos_range * 100
            )
        else:
            scored["pos_risk_score"] = 50.0
    else:
        scored["pos_risk_score"] = 50.0

    # Composite: weighted combination
    scored["risk_score"] = (
        scored["gap_percentile"] * 0.40
        + scored["spec_deviation_score"] * 0.25
        + scored["geo_deviation_score"] * 0.20
        + scored["pos_risk_score"] * 0.15
    ).round(1)

    # Clean up temp columns
    for col in ["spec_avg_gap", "state_avg_gap", "pos_avg_gap"]:
        if col in scored.columns:
            scored.drop(columns=[col], inplace=True)

    return scored


def risk_score_summary(df: pd.DataFrame) -> dict:
    """Compute aggregate risk score stats for KPI display."""
    if "risk_score" not in df.columns:
        df = compute_risk_scores(df)

    high_risk = df[df["risk_score"] >= 75]
    high_risk_claims = (high_risk["total_services"]).sum()
    total_claims = df["total_services"].sum()

    # Top risk specialty by average risk score (weighted)
    spec_risks = []
    for spec, grp in df.groupby("specialty"):
        spec_risks.append({
            "specialty": spec,
            "avg_risk": _weighted_avg(grp, "risk_score", "total_services"),
        })
    spec_risk_df = pd.DataFrame(spec_risks).sort_values("avg_risk", ascending=False)

    return {
        "avg_risk_score": _weighted_avg(df, "risk_score", "total_services"),
        "high_risk_pct": high_risk_claims / total_claims * 100 if total_claims > 0 else 0,
        "highest_risk_specialty": spec_risk_df.iloc[0]["specialty"] if not spec_risk_df.empty else "N/A",
        "highest_risk_score": spec_risk_df.iloc[0]["avg_risk"] if not spec_risk_df.empty else 0,
    }


def generate_key_findings(df: pd.DataFrame) -> list[dict]:
    """
    Auto-generate the top actionable findings from the data.
    Each finding has a title, body, and severity (high/medium/info).
    """
    findings = []

    # 1. Highest-gap specialty
    spec = denial_rate_by_specialty(df)
    if not spec.empty:
        top = spec.iloc[0]
        bottom = spec.iloc[-1]
        spread = top["avg_payment_gap_pct"] - bottom["avg_payment_gap_pct"]
        findings.append({
            "title": f"{top['specialty']} has the highest payment gap",
            "body": (
                f"At {top['avg_payment_gap_pct']:.1f}%, it's {spread:.1f}pp above "
                f"{bottom['specialty']} ({bottom['avg_payment_gap_pct']:.1f}%). "
                f"This specialty accounts for {int(top['total_claims']):,} claims "
                f"across {int(top['provider_count']):,} providers."
            ),
            "severity": "high",
        })

    # 2. Highest revenue at risk
    if not spec.empty and "total_revenue_at_risk" in spec.columns:
        risk_sorted = spec.sort_values("total_revenue_at_risk", ascending=False)
        top_risk = risk_sorted.iloc[0]
        if top_risk["total_revenue_at_risk"] > 0:
            findings.append({
                "title": f"${top_risk['total_revenue_at_risk']:,.0f} underpayment gap in {top_risk['specialty']}",
                "body": (
                    f"The largest allowed-to-payment gap by dollar volume. "
                    f"This represents the difference between what Medicare approved "
                    f"and what it actually paid — a direct signal of adjustments."
                ),
                "severity": "high",
            })

    # 3. Geographic variance
    from src.analysis import EXCLUDE_STATES, MIN_STATE_CLAIMS
    state_df = denial_rate_by_state(df)
    if len(state_df) >= 5:
        state_std = state_df["avg_payment_gap_pct"].std()
        high_state = state_df.iloc[0]
        low_state = state_df.iloc[-1]
        findings.append({
            "title": f"{high_state['state']} vs. {low_state['state']}: {high_state['avg_payment_gap_pct'] - low_state['avg_payment_gap_pct']:.1f}pp gap spread",
            "body": (
                f"Geographic variation suggests different MAC contractor behavior. "
                f"{high_state['state']} averages {high_state['avg_payment_gap_pct']:.1f}% gap "
                f"vs. {low_state['state']} at {low_state['avg_payment_gap_pct']:.1f}%. "
                f"Providers in high-gap states should prioritize documentation."
            ),
            "severity": "medium",
        })

    # 4. Facility vs office spread
    pos_df = place_of_service_analysis(df)
    if len(pos_df) == 2:
        fac = pos_df[pos_df["place_of_service_label"] == "Facility"]
        ofc = pos_df[pos_df["place_of_service_label"] == "Office/Non-Facility"]
        if not fac.empty and not ofc.empty:
            fac_gap = fac.iloc[0]["avg_payment_gap_pct"]
            ofc_gap = ofc.iloc[0]["avg_payment_gap_pct"]
            diff = abs(fac_gap - ofc_gap)
            higher = "Facility" if fac_gap > ofc_gap else "Office"
            findings.append({
                "title": f"{higher} setting has {diff:.1f}pp higher gap",
                "body": (
                    f"Facility: {fac_gap:.1f}% gap | Office: {ofc_gap:.1f}% gap. "
                    f"Place-of-service mismatches are a common denial trigger (CARC-5)."
                ),
                "severity": "info",
            })

    return findings


def summary_statistics(df: pd.DataFrame) -> dict:
    """
    Compute top-level KPIs for the dashboard header.
    """
    total_claims = df["total_services"].sum()
    total_providers = df["provider_npi"].nunique()
    total_specialties = df["specialty"].nunique()
    total_procedures = df["procedure_code"].nunique()

    avg_gap = _weighted_avg(df, "payment_gap_pct", "total_services")
    median_gap = df["payment_gap_pct"].median()

    # Use allowed-to-payment gap for revenue at risk (more meaningful)
    total_at_risk = df["total_revenue_at_risk"].sum()

    high_risk_services = (df["denial_risk_flag"] * df["total_services"]).sum()
    high_risk_rate = high_risk_services / total_claims * 100 if total_claims > 0 else 0

    # Gap threshold used for flagging
    gap_threshold = df["_gap_threshold"].iloc[0] if "_gap_threshold" in df.columns else 50.0

    return {
        "total_claims": int(total_claims),
        "total_providers": int(total_providers),
        "total_specialties": int(total_specialties),
        "total_procedures": int(total_procedures),
        "avg_payment_gap_pct": round(avg_gap, 1),
        "median_payment_gap_pct": round(median_gap, 1),
        "total_revenue_at_risk": round(total_at_risk, 2),
        "high_risk_rate": round(high_risk_rate, 1),
        "gap_threshold": round(gap_threshold, 1),
    }

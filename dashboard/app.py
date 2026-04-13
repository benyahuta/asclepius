from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st
import pandas as pd

from src.analysis import (
    load_processed_data,
    denial_rate_by_specialty,
    denial_rate_by_procedure,
    denial_rate_by_state,
    denial_rate_by_specialty_and_procedure,
    place_of_service_analysis,
    summary_statistics,
    generate_key_findings,
    compute_risk_scores,
    risk_score_summary,
)
from src.anomaly import (
    detect_specialty_anomalies,
    detect_procedure_anomalies_within_specialty,
    detect_geographic_anomalies,
    generate_anomaly_alerts,
)
from src.carc_codes import get_all_codes_df, get_specialty_carc_risks, get_all_specialty_risks_df
from dashboard.components import (
    render_key_findings,
    render_kpi_cards,
    render_specialty_bar_chart,
    render_procedure_table,
    render_state_choropleth,
    render_specialty_procedure_scatter,
    render_anomaly_alerts,
    render_place_of_service_comparison,
    render_carc_reference_table,
    render_risk_score_distribution,
    render_risk_by_specialty,
)


# ── Page config ───────────────────────────────────────────────────────
st.set_page_config(
    page_title="Denial Pattern Analysis — Ember",
    page_icon="🔥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────
st.markdown("""
<style>
    .block-container { padding-top: 1rem; }
    [data-testid="stMetric"] {
        background-color: rgba(255, 154, 13, 0.08);
        border-radius: 8px;
        padding: 12px 16px;
        border-left: 3px solid #FF9A0D;
    }
    [data-testid="stMetricValue"] {
        color: inherit !important;
    }
    [data-testid="stMetricLabel"] {
        font-size: 0.85rem;
        color: inherit !important;
    }
    [data-testid="stMetricDelta"] {
        color: inherit !important;
    }
    h2 { color: #7B73FF; font-size: 1.4rem; }
    .context-note {
        background-color: rgba(88, 78, 255, 0.1);
        border-left: 3px solid #584EFF;
        padding: 10px 14px;
        border-radius: 4px;
        font-size: 0.9em;
        color: inherit;
        margin-bottom: 16px;
    }
</style>
""", unsafe_allow_html=True)


# ── Data loading ──────────────────────────────────────────────────────
@st.cache_data(ttl=3600)
def load_data():
    return load_processed_data()


# ── Sidebar ───────────────────────────────────────────────────────────
st.sidebar.title("🔥 Denial Analysis")
st.sidebar.markdown("**Medicare Claims Pattern Explorer**")
st.sidebar.markdown("---")

try:
    df = load_data()
except FileNotFoundError:
    st.error(
        "No data found. Run the data setup script first:\n\n"
        "```bash\n"
        "cd ember-denial-dashboard\n"
        "pip3 install -r requirements.txt\n"
        "python3 scripts/setup_data.py\n"
        "```"
    )
    st.stop()

# Sidebar filters
available_specialties = sorted(df["specialty"].unique())
selected_specialties = st.sidebar.multiselect(
    "Filter by Specialty",
    options=available_specialties,
    default=[],
    help="Leave empty to show all specialties",
)

available_states = sorted(df["state"].dropna().unique())
selected_states = st.sidebar.multiselect(
    "Filter by State",
    options=available_states,
    default=[],
    help="Leave empty to show all states",
)

min_claims_filter = st.sidebar.slider(
    "Minimum Claim Volume",
    min_value=10,
    max_value=1000,
    value=50,
    step=10,
    help="Filter out low-volume provider/procedure combos",
)

st.sidebar.markdown("---")
st.sidebar.markdown(
    "**Data source:** [CMS Medicare Provider Utilization](https://data.cms.gov)"
)
st.sidebar.markdown(
    "Built for [Ember AI](https://www.embercopilot.ai)"
)

# Apply filters
filtered_df = df.copy()
if selected_specialties:
    filtered_df = filtered_df[filtered_df["specialty"].isin(selected_specialties)]
if selected_states:
    filtered_df = filtered_df[filtered_df["state"].isin(selected_states)]
if "total_services" in filtered_df.columns:
    filtered_df = filtered_df[filtered_df["total_services"] >= min_claims_filter]

if filtered_df.empty:
    st.warning("No data matches the current filters. Try broadening your selection.")
    st.stop()


# ── Main content ──────────────────────────────────────────────────────
st.title("Medicare Denial Pattern Analysis")
st.markdown(
    "Real CMS claims data analyzed for payment adjustment patterns across "
    "specialties, procedures, and regions. These are the same signals an "
    "AI RCM system uses to prevent denials before they happen."
)

st.markdown(
    '<div class="context-note">'
    "<strong>Reading this data:</strong> Medicare providers typically bill at "
    "chargemaster rates 2-5x above what Medicare pays, so payment gaps of "
    "60-80% are <em>normal</em>. The value is in <strong>relative comparison</strong> — "
    "which specialties, procedures, or states have gaps significantly "
    "above or below their peers. That deviation is the denial signal."
    "</div>",
    unsafe_allow_html=True,
)

# ── Key Findings ─────────────────────────────────────────────────────
st.subheader("Key Findings")
findings = generate_key_findings(filtered_df)
render_key_findings(findings)

st.markdown("")

# ── Compute risk scores once ──────────────────────────────────────────
scored_df = compute_risk_scores(filtered_df)

# ── KPIs ──────────────────────────────────────────────────────────────
stats = summary_statistics(filtered_df)
stats.update(risk_score_summary(scored_df))
dataset_avg_gap = stats["avg_payment_gap_pct"]
render_kpi_cards(stats)

st.markdown("---")

# ── Tabs ──────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab7, tab4, tab5, tab6 = st.tabs([
    "By Specialty",
    "By Procedure",
    "By Geography",
    "Risk Scoring",
    "Anomaly Alerts",
    "Deep Dive",
    "CARC Reference",
])

# ── Tab 1: Specialty analysis ─────────────────────────────────────────
with tab1:
    st.header("Payment Gap by Specialty")
    st.markdown(
        "Specialties to the right of the dashed line have higher-than-average "
        "payment gaps. These are the areas where billing teams should focus "
        "pre-submission review and documentation improvement."
    )

    spec_df = denial_rate_by_specialty(filtered_df)
    render_specialty_bar_chart(spec_df)

    with st.expander("View raw data"):
        st.dataframe(spec_df, use_container_width=True, hide_index=True)

# ── Tab 2: Procedure analysis ─────────────────────────────────────────
with tab2:
    st.header("Highest-Gap Procedures")
    st.markdown(
        "Procedures ranked by average payment gap. Only procedures with "
        "500+ total claims are shown to filter out noise from low-volume "
        "codes. Each row represents a specific billing action where "
        "pre-submission review would prevent revenue loss."
    )

    proc_df = denial_rate_by_procedure(filtered_df, top_n=30)

    if proc_df.empty:
        st.info("No procedures meet the minimum volume threshold. Try lowering the filter.")
    else:
        render_procedure_table(proc_df)

        st.markdown("---")
        st.subheader("Payment Gap vs. Average")
        st.markdown(
            "How each procedure deviates from the dataset average gap. "
            "Red bars (right) flag procedures where the gap is unusually high — "
            "prime candidates for pre-submission review."
        )
        render_specialty_procedure_scatter(proc_df, dataset_avg=dataset_avg_gap)

# ── Tab 3: Geographic analysis ────────────────────────────────────────
with tab3:
    st.header("Geographic Payment Patterns")
    st.markdown(
        "This map shows each state's **deviation from the national average** gap. "
        "Red states have higher-than-average gaps; blue states are lower. "
        "These differences trace back to Medicare Administrative Contractors (MACs) "
        "applying different Local Coverage Determinations — a procedure "
        "paid normally in one region might be adjusted in another."
    )

    state_df = denial_rate_by_state(filtered_df)

    if state_df.empty:
        st.info("Not enough geographic data with current filters.")
    else:
        render_state_choropleth(state_df)

        with st.expander("View state-level data"):
            st.dataframe(
                state_df.sort_values("avg_payment_gap_pct", ascending=False),
                use_container_width=True,
                hide_index=True,
            )

# ── Tab 7: Risk Scoring ─────────────────────────────────────────────
with tab7:
    st.header("Composite Denial Risk Score")
    st.markdown(
        "A multi-factor risk score (0-100) combining payment gap percentile, "
        "specialty deviation, geographic deviation, and place-of-service patterns. "
        "This is the kind of **feature engineering** an AI denial prediction model "
        "would use as input — not a single metric, but a composite signal "
        "weighted by predictive value."
    )

    st.markdown(
        '<div class="context-note">'
        "<strong>Score components:</strong> "
        "Payment gap percentile (40%) + Specialty deviation (25%) + "
        "Geographic deviation (20%) + Place-of-service risk (15%). "
        "Weights reflect published denial rate research on predictive signal strength."
        "</div>",
        unsafe_allow_html=True,
    )

    rcol1, rcol2, rcol3 = st.columns(3)
    rcol1.metric("Avg Risk Score", f"{stats['avg_risk_score']:.0f}/100")
    rcol2.metric("High-Risk Claims (≥75)", f"{stats['high_risk_pct']:.1f}%")
    rcol3.metric("Highest-Risk Specialty", stats["highest_risk_specialty"])

    st.markdown("---")
    st.subheader("Claims by Risk Tier")
    st.markdown("How claims distribute across risk levels. The goal of an AI RCM system is to shift volume from right to left.")
    render_risk_score_distribution(scored_df)

    st.markdown("---")
    st.subheader("Risk Score by Specialty")
    st.markdown("Which specialties have the highest composite risk. This ranks differently from raw payment gap because it incorporates geographic and place-of-service factors.")
    render_risk_by_specialty(scored_df)

# ── Tab 4: Anomaly detection ─────────────────────────────────────────
with tab4:
    st.header("Anomaly Alerts")
    st.markdown(
        "Statistically unusual payment patterns that may indicate policy "
        "changes, new coverage restrictions, or billing issues. These are "
        "early warning signals — the kind of thing that would trigger a "
        "real-time alert in an RCM platform."
    )

    z_threshold = st.slider(
        "Sensitivity (z-score threshold)",
        min_value=0.5,
        max_value=4.0,
        value=1.5,
        step=0.25,
        help="Lower = more alerts (more sensitive). Higher = only extreme outliers.",
    )

    alerts = generate_anomaly_alerts(filtered_df, z_threshold=z_threshold)
    render_anomaly_alerts(alerts)

    st.markdown("---")
    st.subheader("Specialty Deviation Detail")
    anomaly_df = detect_specialty_anomalies(filtered_df, z_threshold=z_threshold)
    anomaly_display = anomaly_df[anomaly_df["is_anomaly"]].copy()
    if not anomaly_display.empty:
        st.dataframe(anomaly_display, use_container_width=True, hide_index=True)
    else:
        st.info("No specialty-level anomalies at this threshold.")

# ── Tab 5: Deep dive ─────────────────────────────────────────────────
with tab5:
    st.header("Specialty Deep Dive")
    st.markdown(
        "Select a specialty to drill into which specific procedures have "
        "the highest payment gaps. This is the view a billing manager "
        "would use to prioritize pre-submission review."
    )

    deep_dive_specialty = st.selectbox(
        "Choose a specialty",
        options=available_specialties,
        index=0,
    )

    if deep_dive_specialty:
        spec_proc_df = denial_rate_by_specialty_and_procedure(
            filtered_df, specialty=deep_dive_specialty
        )

        if not spec_proc_df.empty:
            # Summary metrics for this specialty
            spec_filtered = filtered_df[filtered_df["specialty"] == deep_dive_specialty]
            spec_claims = spec_filtered["total_services"].sum()
            spec_providers = spec_filtered["provider_npi"].nunique()
            spec_procs = spec_filtered["procedure_code"].nunique()

            mcol1, mcol2, mcol3 = st.columns(3)
            mcol1.metric("Total Claims", f"{spec_claims:,.0f}")
            mcol2.metric("Providers", f"{spec_providers:,.0f}")
            mcol3.metric("Unique Procedures", f"{spec_procs:,.0f}")

            # Show denial risk profile for this specialty
            carc_risk = get_specialty_carc_risks(deep_dive_specialty)
            risk_cats = ", ".join(carc_risk["top_risks"])
            st.markdown(
                f'<div class="context-note">'
                f"<strong>Denial risk profile:</strong> "
                f"Top denial categories for {deep_dive_specialty}: **{risk_cats}**. "
                f"{carc_risk['notes']}"
                f"</div>",
                unsafe_allow_html=True,
            )

            st.markdown("---")
            st.subheader(f"Highest-Gap Procedures — {deep_dive_specialty}")
            render_procedure_table(spec_proc_df.head(15))

            # Add the deviation chart for this specialty's procedures
            st.markdown("---")
            st.subheader(f"Procedure Gap Deviation — {deep_dive_specialty}")
            st.markdown(
                "How each procedure compares to this specialty's average gap. "
                "Red = above average (review priority). Purple = below average."
            )
            render_specialty_procedure_scatter(spec_proc_df.head(15), dataset_avg=dataset_avg_gap)

            st.markdown("---")
            st.subheader(f"Procedure Anomalies — {deep_dive_specialty}")
            proc_anomalies = detect_procedure_anomalies_within_specialty(
                filtered_df, deep_dive_specialty
            )
            if not proc_anomalies.empty:
                anomalous = proc_anomalies[proc_anomalies["is_anomaly"]]
                if not anomalous.empty:
                    st.dataframe(anomalous, use_container_width=True, hide_index=True)
                else:
                    st.info(f"No procedure-level anomalies in {deep_dive_specialty}.")
            else:
                st.info(f"Not enough data for anomaly detection in {deep_dive_specialty}.")

            st.markdown("---")
            st.subheader(f"Facility vs. Office — {deep_dive_specialty}")
            pos_df = place_of_service_analysis(spec_filtered)
            render_place_of_service_comparison(pos_df)
        else:
            st.warning(f"No procedures meet the minimum volume threshold for {deep_dive_specialty}.")

# ── Tab 6: CARC reference ────────────────────────────────────────────
with tab6:
    st.header("Denial Risk by Specialty")
    st.markdown(
        "Which denial categories are most likely to affect each specialty, "
        "based on procedure mix, billing patterns, and published denial rate research. "
        "This mapping is what an AI copilot would use to prioritize pre-submission checks."
    )

    # Specialty denial risk mapping
    spec_risk_df = get_all_specialty_risks_df()
    if not spec_risk_df.empty:
        for _, row in spec_risk_df.iterrows():
            with st.expander(f"**{row['specialty']}** — Primary risk: {row['primary_risk']}"):
                st.markdown(
                    f"**Top denial categories:** {row['primary_risk']}, "
                    f"{row['secondary_risk']}, {row['tertiary_risk']}"
                )
                st.markdown(f"**Why:** {row['notes']}")

                # Show relevant CARC codes for the primary risk category
                from src.carc_codes import DENIAL_CATEGORIES, CARC_CODES
                primary_codes = DENIAL_CATEGORIES.get(row['primary_risk'], [])
                if primary_codes:
                    st.markdown(f"**Key CARC codes for {row['primary_risk']}:**")
                    for code in primary_codes[:3]:
                        info = CARC_CODES.get(code, {})
                        if info:
                            st.markdown(
                                f"- **CARC-{code}:** {info.get('plain_english', '')} "
                                f"→ *{info.get('actionable_insight', '')}*"
                            )

    st.markdown("---")
    st.header("CARC Code Reference")
    st.markdown(
        "The full reference of Claim Adjustment Reason Codes — the standardized "
        "codes insurers use to explain denials and adjustments."
    )

    carc_df = get_all_codes_df()

    categories = ["All"] + sorted(carc_df["denial_category"].unique().tolist())
    selected_category = st.selectbox("Filter by category", categories)

    if selected_category != "All":
        carc_df = carc_df[carc_df["denial_category"] == selected_category]

    render_carc_reference_table(carc_df)

# ── Footer: methodology & export ────────────────────────────────────
st.markdown("---")
with st.expander("Methodology & Data Notes"):
    st.markdown(
        """
**Data source:** CMS Medicare Physician & Other Practitioners — Provider and Service
(2023 release). Accessed via CMS Data API v1.

**Payment gap calculation:** Claim-volume-weighted average of
`(submitted_charge - medicare_payment) / submitted_charge` per provider-procedure row.
Volume weighting prevents small providers from distorting specialty-level averages.

**Denial risk flag:** Percentile-based threshold (top 20% of payment gaps within
the dataset). This adapts to the actual distribution instead of an arbitrary cutoff.

**Revenue at risk:** Sum of `(allowed_amount - actual_payment) × service_count`.
This uses the allowed-to-payment gap (not the chargemaster gap) to reflect
actual underpayment, not the expected contractual discount.

**Anomaly detection:** Z-score and IQR-based methods on specialty and geographic
aggregates. Alerts fire when a group deviates beyond the configured threshold
from its peer set.

**Limitations:** This dataset does not contain actual denial codes (CARCs/RARCs)
or appeal outcomes. Payment gap is a proxy signal — it correlates with denials
and adjustments but also includes expected contractual write-offs. The CARC
reference tab provides educational context, not direct mappings to this data.
        """
    )

col_export1, col_export2 = st.columns(2)
with col_export1:
    csv_data = filtered_df.to_csv(index=False)
    st.download_button(
        label="Export filtered data (CSV)",
        data=csv_data,
        file_name="ember_denial_analysis.csv",
        mime="text/csv",
    )
with col_export2:
    spec_df_export = denial_rate_by_specialty(filtered_df)
    st.download_button(
        label="Export specialty summary (CSV)",
        data=spec_df_export.to_csv(index=False),
        file_name="ember_specialty_summary.csv",
        mime="text/csv",
    )

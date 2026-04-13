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
from src.model import train_model
from src.anomaly import (
    detect_specialty_anomalies,
    detect_procedure_anomalies_within_specialty,
    detect_geographic_anomalies,
    generate_anomaly_alerts,
)
from src.carc_codes import (
    get_all_codes_df,
    get_specialty_carc_risks,
    get_all_specialty_risks_df,
    infer_denial_risks_from_data,
    DENIAL_CATEGORIES,
    CARC_CODES,
)
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
    render_model_metrics,
    render_feature_importances,
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
    "Real CMS claims data analyzed to surface the payment adjustment signals "
    "an AI RCM system needs to predict and prevent denials. Two layers of "
    "analysis: **contractual gaps** (chargemaster vs. Medicare payment) for "
    "relative specialty/procedure ranking, and **allowed-to-payment gaps** "
    "(what Medicare approved vs. what it actually paid) as a direct "
    "underpayment signal."
)

st.markdown(
    '<div class="context-note">'
    "<strong>Two metrics, two purposes:</strong> "
    "(1) <strong>Chargemaster gap</strong> (60-80% is normal) — useful only for "
    "<em>relative</em> comparison between specialties, procedures, and states. "
    "(2) <strong>Allowed-to-payment gap</strong> — when Medicare pays less than "
    "it approved, that's a direct signal of adjustments, partial denials, or "
    "downcoding. The ML model on the Prediction tab targets this signal."
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
    "Prediction Model",
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

# ── Tab 7: Prediction Model ─────────────────────────────────────────
with tab7:
    st.header("Denial Risk Prediction Model")
    st.markdown(
        "A Random Forest classifier trained on this dataset to predict which "
        "provider-procedure combinations will have high allowed-to-payment gaps — "
        "the closest proxy to actual denials in public CMS data. "
        "This is a proof-of-concept for the kind of model an AI RCM system "
        "would run in production, with real remittance data as the target instead."
    )

    @st.cache_data(ttl=3600)
    def get_model_results(_df):
        return train_model(_df)

    with st.spinner("Training model..."):
        model_results = get_model_results(filtered_df)

    if "error" in model_results:
        st.error(model_results["error"])
    else:
        meta = model_results["metadata"]

        st.markdown(
            '<div class="context-note">'
            f"<strong>Model target:</strong> Binary classification — does this claim have "
            f"an <code>{meta['gap_column']}</code> above the {meta['gap_threshold']:.1f}% "
            f"threshold (80th percentile)? Trained on {meta['train_size']:,} rows, "
            f"tested on {meta['test_size']:,} rows. "
            f"Positive class rate: {meta['positive_rate']:.1f}%."
            "</div>",
            unsafe_allow_html=True,
        )

        st.subheader("Model Performance")
        render_model_metrics(model_results["metrics"], meta)

        st.markdown("---")
        st.subheader("What Drives Denial Risk?")
        st.markdown(
            "Feature importances show which factors the model relies on most to "
            "predict high-gap claims. In production, these same features would "
            "power real-time pre-submission checks."
        )
        render_feature_importances(model_results["feature_importances"])

        st.markdown("---")
        st.subheader("Claims by Risk Tier")
        st.markdown(
            "Distribution of claims across the composite risk score. "
            "The goal of an AI RCM system: shift volume from right (high risk) to left (low risk) "
            "through pre-submission documentation improvement."
        )
        render_risk_score_distribution(scored_df)

        st.markdown("---")
        st.subheader("Risk Score by Specialty")
        render_risk_by_specialty(scored_df)

        with st.expander("Full classification report"):
            st.code(model_results["classification_report"])

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
        "Two layers of analysis: **data-driven inference** from actual procedure "
        "mix patterns in this dataset, cross-referenced with **published denial "
        "research** from MGMA and AAPC. Where both sources agree, confidence is highest."
    )

    # Data-driven inference
    inferred_risks = infer_denial_risks_from_data(filtered_df)

    # Manual mapping for comparison
    spec_risk_df = get_all_specialty_risks_df()
    manual_map = {}
    if not spec_risk_df.empty:
        for _, row in spec_risk_df.iterrows():
            manual_map[row["specialty"]] = {
                "risks": [row["primary_risk"], row["secondary_risk"], row["tertiary_risk"]],
                "notes": row["notes"],
            }

    for specialty in sorted(inferred_risks.keys()):
        inferred = inferred_risks[specialty]
        manual = manual_map.get(specialty)
        manual_risks = manual["risks"] if manual else []

        # Check agreement between data-driven and research-based
        overlap = set(inferred) & set(manual_risks)
        agreement = "High" if len(overlap) >= 2 else "Moderate" if len(overlap) >= 1 else "Low"
        agree_color = "#38A169" if agreement == "High" else "#DD6B20" if agreement == "Moderate" else "#718096"

        primary = inferred[0] if inferred else "Unknown"
        with st.expander(f"**{specialty}** — Primary risk: {primary}"):
            col_a, col_b = st.columns(2)
            with col_a:
                st.markdown("**From data (procedure mix):**")
                for r in inferred:
                    st.markdown(f"- {r}")
            with col_b:
                st.markdown("**From published research:**")
                for r in manual_risks:
                    if r:
                        st.markdown(f"- {r}")

            st.markdown(
                f"<span style='color:{agree_color}; font-weight:bold;'>"
                f"Agreement: {agreement}</span>"
                + (f" — {manual['notes']}" if manual else ""),
                unsafe_allow_html=True,
            )

            # Show relevant CARC codes
            primary_codes = DENIAL_CATEGORIES.get(primary, [])
            if primary_codes:
                st.markdown(f"**Key CARC codes for {primary}:**")
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

# ── Ember integration framing ────────────────────────────────────────
st.markdown("---")
with st.expander("How This Connects to Ember's AI Copilot"):
    st.markdown(
        """
**What this analysis demonstrates with public data:**

This dashboard processes 30M+ Medicare claims through a full analytical pipeline:
data ingestion from the CMS API, statistical analysis with volume-weighted aggregation,
anomaly detection, and a trained Random Forest model for denial risk prediction. Every
component is built to scale — the same architecture works on 30M claims or 300M.

**What becomes possible with actual remittance data:**

The key limitation of public CMS data is the absence of actual denial codes (CARCs/RARCs)
and appeal outcomes. Ember's copilot has access to real-time remittance advice (835 files),
which means:

1. **The prediction model's target variable becomes real.** Instead of proxying denials
through allowed-to-payment gaps, the model trains on actual CARC-coded denial outcomes.
Same feature engineering, dramatically better signal.

2. **Temporal analysis becomes possible.** With claim-level timestamps, the anomaly
detection engine can flag *changes* — "Cardiology denials in Texas spiked 40% this
month" — not just static outliers.

3. **Pre-submission intervention at the claim level.** The specialty/procedure/geographic
risk patterns surface here at an aggregate level. With Ember's real-time access, the
same patterns can flag individual claims before they're submitted: "This orthopedic
surgery claim in Wisconsin has a 73% predicted denial probability — here's what to fix."

4. **The CARC mapping closes the loop.** This dashboard maps specialties to *likely*
denial categories. With real CARC data, that mapping becomes exact — and the
actionable insights become specific to the provider's actual denial history.

**Architecture decisions that transfer directly:**

- Volume-weighted aggregation (prevents small-provider bias)
- Percentile-based thresholds (adapts to any data distribution)
- Z-score + IQR anomaly detection (catches both normal and skewed outliers)
- Multi-factor risk scoring (composites are more robust than single metrics)
- Feature engineering for ML (specialty, geography, procedure category, charge magnitude)
        """
    )

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

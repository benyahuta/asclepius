from __future__ import annotations

"""
Reusable Streamlit UI components for the denial dashboard.
"""

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np


# ── Color palette (inspired by Ember's brand: orange + purple) ────────
COLORS = {
    "primary": "#FF9A0D",
    "secondary": "#584EFF",
    "danger": "#E53E3E",
    "success": "#38A169",
    "warning": "#DD6B20",
    "neutral": "#718096",
}


def render_key_findings(findings: list[dict]):
    """Render auto-generated key findings as a compact insight strip."""
    if not findings:
        return

    severity_styles = {
        "high": {"border": "#E53E3E", "bg": "rgba(229, 62, 62, 0.06)", "icon": "🔴"},
        "medium": {"border": "#DD6B20", "bg": "rgba(221, 107, 32, 0.06)", "icon": "🟡"},
        "info": {"border": "#584EFF", "bg": "rgba(88, 78, 255, 0.06)", "icon": "🔵"},
    }

    cols = st.columns(min(len(findings), 3))
    for i, finding in enumerate(findings[:3]):
        style = severity_styles.get(finding["severity"], severity_styles["info"])
        with cols[i]:
            st.markdown(
                f"""<div style="
                    border-left: 3px solid {style['border']};
                    background: {style['bg']};
                    padding: 10px 14px;
                    border-radius: 6px;
                    margin-bottom: 8px;
                    color: inherit;
                ">
                    <strong style="font-size: 0.95em;">{style['icon']} {finding['title']}</strong><br/>
                    <span style="opacity: 0.85; font-size: 0.82em; line-height: 1.4;">{finding['body']}</span>
                </div>""",
                unsafe_allow_html=True,
            )

    # Show remaining findings in an expander
    if len(findings) > 3:
        with st.expander(f"+ {len(findings) - 3} more findings"):
            for finding in findings[3:]:
                style = severity_styles.get(finding["severity"], severity_styles["info"])
                st.markdown(
                    f"**{style['icon']} {finding['title']}** — {finding['body']}"
                )


def render_kpi_cards(stats: dict):
    """Render top-level KPI metric cards with context."""
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            label="Claims Analyzed",
            value=f"{stats['total_claims']:,.0f}",
        )
    with col2:
        st.metric(
            label="Providers",
            value=f"{stats['total_providers']:,.0f}",
        )
    with col3:
        st.metric(
            label="Specialties",
            value=f"{stats['total_specialties']}",
        )
    with col4:
        st.metric(
            label="Unique Procedures",
            value=f"{stats['total_procedures']:,.0f}",
        )

    st.markdown("")  # spacer

    col5, col6, col7, col8 = st.columns(4)
    with col5:
        st.metric(
            label="Avg Payment Gap",
            value=f"{stats['avg_payment_gap_pct']:.1f}%",
            help="Average gap between billed charges and Medicare payment. "
                 "High values are normal for Medicare (chargemaster rates). "
                 "Compare between specialties and procedures for insights.",
        )
    with col6:
        if "avg_risk_score" in stats:
            st.metric(
                label="Avg Risk Score",
                value=f"{stats['avg_risk_score']:.0f}/100",
                help="Composite denial risk score combining payment gap, "
                     "specialty deviation, geographic deviation, and place of service.",
            )
        else:
            st.metric(
                label="Median Payment Gap",
                value=f"{stats['median_payment_gap_pct']:.1f}%",
            )
    with col7:
        st.metric(
            label="High-Risk Flag Rate",
            value=f"{stats['high_risk_rate']:.1f}%",
            help=f"Claims in the top 20% of payment gaps "
                 f"(above {stats['gap_threshold']:.0f}% gap).",
        )
    with col8:
        if stats["total_revenue_at_risk"] > 0:
            st.metric(
                label="Allowed-Payment Gap Total",
                value=f"${stats['total_revenue_at_risk']:,.0f}",
                help="Sum of (allowed amount - actual payment) across all claims. "
                     "This represents actual underpayment vs. what Medicare said "
                     "it would allow, not the chargemaster discount.",
            )
        else:
            st.metric(
                label="Allowed-Payment Gap Total",
                value="$0",
                help="No significant allowed-to-payment gaps detected.",
            )


def render_specialty_bar_chart(specialty_df: pd.DataFrame):
    """Horizontal bar chart of payment gap by specialty with baseline reference."""
    plot_df = specialty_df.head(20).sort_values("avg_payment_gap_pct", ascending=True)
    overall_avg = plot_df["avg_payment_gap_pct"].mean()

    fig = px.bar(
        plot_df,
        x="avg_payment_gap_pct",
        y="specialty",
        orientation="h",
        color="avg_payment_gap_pct",
        color_continuous_scale=["#584EFF", "#FF9A0D", "#E53E3E"],
        labels={
            "avg_payment_gap_pct": "Avg Payment Gap (%)",
            "specialty": "",
        },
        hover_data={
            "total_claims": ":,.0f",
            "provider_count": ":,.0f",
            "denial_risk_rate": ":.1f",
        },
    )

    # Add vertical reference line for dataset average
    fig.add_vline(
        x=overall_avg,
        line_dash="dash",
        line_color="#718096",
        annotation_text=f"Dataset avg: {overall_avg:.1f}%",
        annotation_position="top",
    )

    fig.update_layout(
        height=550,
        showlegend=False,
        coloraxis_showscale=False,
        xaxis_title="Avg Payment Gap (%)",
        margin=dict(l=10, r=10, t=30, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",

        xaxis=dict(gridcolor="rgba(128,128,128,0.15)"),
        yaxis=dict(gridcolor="rgba(128,128,128,0.15)"),
    )

    st.plotly_chart(fig, use_container_width=True)


def render_procedure_table(proc_df: pd.DataFrame):
    """Interactive table of high-risk procedures with conditional formatting."""
    display_cols = [
        "procedure_code", "procedure_description", "total_claims",
        "avg_payment_gap_pct", "avg_submitted_charge", "avg_medicare_payment",
        "provider_count",
    ]
    display_cols = [c for c in display_cols if c in proc_df.columns]
    display_df = proc_df[display_cols].copy()

    rename_map = {
        "procedure_code": "CPT/HCPCS",
        "procedure_description": "Description",
        "total_claims": "Claims",
        "avg_payment_gap_pct": "Avg Gap %",
        "avg_submitted_charge": "Avg Charge",
        "avg_medicare_payment": "Avg Payment",
        "provider_count": "Providers",
    }
    display_df = display_df.rename(columns={k: v for k, v in rename_map.items() if k in display_df.columns})

    # Use Streamlit's column_config for richer display
    col_config = {}
    if "Claims" in display_df.columns:
        col_config["Claims"] = st.column_config.NumberColumn(format="%d")
    if "Avg Gap %" in display_df.columns:
        col_config["Avg Gap %"] = st.column_config.ProgressColumn(
            "Avg Gap %",
            format="%.1f%%",
            min_value=0,
            max_value=100,
        )
    if "Avg Charge" in display_df.columns:
        col_config["Avg Charge"] = st.column_config.NumberColumn(format="$%.0f")
    if "Avg Payment" in display_df.columns:
        col_config["Avg Payment"] = st.column_config.NumberColumn(format="$%.0f")
    if "Providers" in display_df.columns:
        col_config["Providers"] = st.column_config.NumberColumn(format="%d")

    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True,
        column_config=col_config,
    )


def render_state_choropleth(state_df: pd.DataFrame):
    """US choropleth map showing deviation from national average payment gap."""
    if "gap_vs_national" not in state_df.columns:
        national_avg = state_df["avg_payment_gap_pct"].mean()
        state_df = state_df.copy()
        state_df["gap_vs_national"] = state_df["avg_payment_gap_pct"] - national_avg

    max_dev = max(abs(state_df["gap_vs_national"].max()), abs(state_df["gap_vs_national"].min()), 1)

    fig = go.Figure(
        data=go.Choropleth(
            locations=state_df["state"],
            z=state_df["gap_vs_national"],
            locationmode="USA-states",
            zmin=-max_dev,
            zmax=max_dev,
            colorscale=[
                [0, "#584EFF"],
                [0.5, "#f5f5f5"],
                [1, "#E53E3E"],
            ],
            colorbar_title="Deviation<br>from Avg (pp)",
            hovertemplate=(
                "<b>%{location}</b><br>"
                "Deviation from national avg: %{z:+.1f}pp<br>"
                "<extra></extra>"
            ),
            customdata=state_df[["avg_payment_gap_pct", "total_claims", "provider_count"]].values,
        )
    )

    # Update hover to show more detail
    fig.update_traces(
        hovertemplate=(
            "<b>%{location}</b><br>"
            "Payment Gap: %{customdata[0]:.1f}%<br>"
            "Deviation: %{z:+.1f}pp from national avg<br>"
            "Claims: %{customdata[1]:,.0f}<br>"
            "Providers: %{customdata[2]:,.0f}<br>"
            "<extra></extra>"
        ),
    )

    fig.update_layout(
        geo_scope="usa",
        height=450,
        margin=dict(l=0, r=0, t=0, b=0),
        paper_bgcolor="rgba(0,0,0,0)",
        geo=dict(
            bgcolor="rgba(0,0,0,0)",
            lakecolor="rgba(0,0,0,0)",
            landcolor="rgba(128,128,128,0.1)",
        ),
    )

    st.plotly_chart(fig, use_container_width=True)


def render_specialty_procedure_scatter(df: pd.DataFrame, dataset_avg: float | None = None):
    """
    Diverging bar chart: each procedure's payment gap vs the dataset average.
    Bar length = deviation from average. Color = above/below.
    Shows where intervention has the most impact at a glance.

    Args:
        dataset_avg: The overall dataset average gap to compare against.
                     If None, falls back to the mean of df (less useful).
    """
    plot_df = df.head(20).copy()

    if plot_df.empty:
        st.info("No procedures to display.")
        return

    avg_gap = dataset_avg if dataset_avg is not None else plot_df["avg_payment_gap_pct"].mean()
    plot_df["gap_deviation"] = plot_df["avg_payment_gap_pct"] - avg_gap
    plot_df["above_avg"] = plot_df["gap_deviation"] > 0

    # Truncate long descriptions for readable y-axis
    plot_df["short_desc"] = plot_df["procedure_description"].str[:35]
    plot_df["label"] = plot_df["procedure_code"] + " " + plot_df["short_desc"]

    plot_df = plot_df.sort_values("gap_deviation", ascending=True)

    fig = go.Figure()

    for _, row in plot_df.iterrows():
        color = COLORS["danger"] if row["above_avg"] else COLORS["secondary"]
        fig.add_trace(go.Bar(
            x=[row["gap_deviation"]],
            y=[row["label"]],
            orientation="h",
            marker_color=color,
            hovertemplate=(
                f"<b>{row['procedure_code']}</b><br>"
                f"{row['procedure_description']}<br>"
                f"Payment Gap: {row['avg_payment_gap_pct']:.1f}%<br>"
                f"Deviation from avg: {row['gap_deviation']:+.1f}pp<br>"
                f"Claims: {row['total_claims']:,.0f}<br>"
                f"Avg Charge: ${row['avg_submitted_charge']:,.0f}<br>"
                "<extra></extra>"
            ),
            showlegend=False,
        ))

    fig.add_vline(x=0, line_color="rgba(128,128,128,0.5)", line_width=1)

    fig.update_layout(
        height=max(400, len(plot_df) * 34),
        xaxis_title=f"Deviation from Avg Gap ({avg_gap:.1f}%) in percentage points",
        yaxis_title="",
        margin=dict(l=10, r=10, t=10, b=40),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(size=11),
        xaxis=dict(gridcolor="rgba(128,128,128,0.15)", zeroline=False),
        yaxis=dict(gridcolor="rgba(128,128,128,0.15)"),
    )

    st.plotly_chart(fig, use_container_width=True)


def render_anomaly_alerts(alerts: list[dict]):
    """Render anomaly detection alerts."""
    if not alerts:
        st.info("No anomalies detected at current sensitivity. Try lowering the threshold.")
        return

    for alert in alerts[:12]:
        severity = alert["severity"]
        border_color = '#E53E3E' if severity == 'high' else '#DD6B20'
        bg_color = 'rgba(229, 62, 62, 0.08)' if severity == 'high' else 'rgba(221, 107, 32, 0.08)'
        icon = "🔴" if severity == "high" else "🟡"

        st.markdown(
            f"""
            <div style="
                padding: 12px 16px;
                margin-bottom: 8px;
                border-radius: 8px;
                border-left: 4px solid {border_color};
                background-color: {bg_color};
                color: inherit;
            ">
                <strong>{icon} {alert['category']}: {alert['entity']}</strong><br/>
                <span style="opacity: 0.85; font-size: 0.9em;">{alert['message']}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_place_of_service_comparison(pos_df: pd.DataFrame):
    """Side-by-side comparison of facility vs office denial patterns."""
    if pos_df.empty:
        st.info("Place of service data not available.")
        return

    fig = px.bar(
        pos_df,
        x="place_of_service_label",
        y=["avg_payment_gap_pct", "denial_risk_rate"],
        barmode="group",
        color_discrete_sequence=[COLORS["primary"], COLORS["secondary"]],
        labels={
            "place_of_service_label": "Setting",
            "value": "Percentage (%)",
            "variable": "Metric",
        },
    )

    fig.update_layout(
        height=350,
        margin=dict(l=10, r=10, t=10, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(gridcolor="rgba(128,128,128,0.15)"),
        yaxis=dict(gridcolor="rgba(128,128,128,0.15)"),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
        ),
    )

    newnames = {
        "avg_payment_gap_pct": "Avg Payment Gap %",
        "denial_risk_rate": "High-Risk Flag Rate %",
    }
    fig.for_each_trace(lambda t: t.update(name=newnames.get(t.name, t.name)))

    st.plotly_chart(fig, use_container_width=True)


def render_risk_score_distribution(df: pd.DataFrame):
    """Histogram of composite denial risk scores across the dataset."""
    if "risk_score" not in df.columns:
        st.info("Risk scores not computed.")
        return

    fig = go.Figure()

    # Create bins
    bins = [0, 25, 50, 75, 100]
    labels = ["Low (0-25)", "Moderate (25-50)", "Elevated (50-75)", "High (75-100)"]
    colors_map = [COLORS["success"], COLORS["secondary"], COLORS["warning"], COLORS["danger"]]

    for i in range(len(bins) - 1):
        mask = (df["risk_score"] >= bins[i]) & (df["risk_score"] < bins[i + 1])
        if i == len(bins) - 2:  # last bin includes upper bound
            mask = (df["risk_score"] >= bins[i]) & (df["risk_score"] <= bins[i + 1])
        count = mask.sum()
        claims = df.loc[mask, "total_services"].sum()
        fig.add_trace(go.Bar(
            x=[labels[i]],
            y=[claims],
            marker_color=colors_map[i],
            hovertemplate=(
                f"<b>{labels[i]}</b><br>"
                f"Provider-procedure rows: {count:,}<br>"
                f"Total claims: {claims:,.0f}<br>"
                "<extra></extra>"
            ),
            showlegend=False,
        ))

    fig.update_layout(
        height=300,
        xaxis_title="Risk Tier",
        yaxis_title="Total Claims",
        margin=dict(l=10, r=10, t=10, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(gridcolor="rgba(128,128,128,0.15)"),
        yaxis=dict(gridcolor="rgba(128,128,128,0.15)"),
    )

    st.plotly_chart(fig, use_container_width=True)


def render_risk_by_specialty(df: pd.DataFrame):
    """Horizontal bar chart of average risk score by specialty."""
    if "risk_score" not in df.columns:
        return

    spec_risks = []
    for spec, grp in df.groupby("specialty"):
        weights = grp["total_services"]
        total_w = weights.sum()
        if total_w > 0:
            avg_risk = (grp["risk_score"] * weights).sum() / total_w
        else:
            avg_risk = 0
        spec_risks.append({"specialty": spec, "avg_risk_score": avg_risk, "total_claims": total_w})

    spec_df = pd.DataFrame(spec_risks).sort_values("avg_risk_score", ascending=True)

    fig = px.bar(
        spec_df,
        x="avg_risk_score",
        y="specialty",
        orientation="h",
        color="avg_risk_score",
        color_continuous_scale=[
            [0, COLORS["success"]],
            [0.5, COLORS["warning"]],
            [1, COLORS["danger"]],
        ],
        labels={"avg_risk_score": "Avg Risk Score", "specialty": ""},
        hover_data={"total_claims": ":,.0f"},
    )

    fig.update_layout(
        height=550,
        showlegend=False,
        coloraxis_showscale=False,
        xaxis_title="Composite Risk Score (0-100)",
        xaxis=dict(range=[0, 100], gridcolor="rgba(128,128,128,0.15)"),
        yaxis=dict(gridcolor="rgba(128,128,128,0.15)"),
        margin=dict(l=10, r=10, t=10, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )

    st.plotly_chart(fig, use_container_width=True)


def render_carc_reference_table(carc_df: pd.DataFrame):
    """Render the CARC code reference with actionable insights."""
    for _, row in carc_df.iterrows():
        with st.expander(f"**{row['carc_code']}** — {row['plain_english']}"):
            st.markdown(f"**Official description:** {row['description']}")
            st.markdown(f"**Category:** {row['denial_category']}")
            st.markdown(f"**Group:** {row['group']}")
            st.markdown(f"**What to do:** {row['actionable_insight']}")

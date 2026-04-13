from __future__ import annotations

"""
Anomaly detection for denial patterns.

Identifies statistically unusual spikes in denial risk that could indicate:
  - A payer quietly changed a coverage policy
  - A new LCD (Local Coverage Determination) went into effect
  - A billing practice shift at a provider or specialty level
  - Data quality issues worth investigating

Uses z-score based detection within peer groups. Also adds IQR-based
detection for robustness against skewed distributions.
"""

import pandas as pd
import numpy as np
from config import ANOMALY_Z_THRESHOLD


def detect_specialty_anomalies(
    df: pd.DataFrame,
    z_threshold: float = ANOMALY_Z_THRESHOLD,
) -> pd.DataFrame:
    """
    Flag specialties whose payment gap deviates significantly from peers.
    """
    specialty_stats = df.groupby("specialty").agg(
        mean_gap=("payment_gap_pct", "mean"),
        std_gap=("payment_gap_pct", "std"),
        median_gap=("payment_gap_pct", "median"),
        claim_volume=("total_services", "sum"),
        provider_count=("provider_npi", "nunique"),
        procedure_count=("procedure_code", "nunique"),
    ).reset_index()

    global_mean = df["payment_gap_pct"].mean()
    global_std = df["payment_gap_pct"].std()

    if global_std == 0 or np.isnan(global_std):
        specialty_stats["z_score"] = 0
    else:
        specialty_stats["z_score"] = (
            (specialty_stats["mean_gap"] - global_mean) / global_std
        )

    # Also compute IQR-based outlier detection (more robust)
    q1 = specialty_stats["mean_gap"].quantile(0.25)
    q3 = specialty_stats["mean_gap"].quantile(0.75)
    iqr = q3 - q1
    specialty_stats["is_iqr_outlier"] = (
        (specialty_stats["mean_gap"] < q1 - 1.5 * iqr)
        | (specialty_stats["mean_gap"] > q3 + 1.5 * iqr)
    )

    specialty_stats["is_anomaly"] = (
        (specialty_stats["z_score"].abs() > z_threshold)
        | specialty_stats["is_iqr_outlier"]
    )

    specialty_stats["direction"] = np.where(
        specialty_stats["mean_gap"] > global_mean,
        "Higher than average",
        "Lower than average",
    )

    specialty_stats["global_mean"] = round(global_mean, 1)

    return specialty_stats.sort_values("z_score", ascending=False)


def detect_procedure_anomalies_within_specialty(
    df: pd.DataFrame,
    specialty: str,
    z_threshold: float = ANOMALY_Z_THRESHOLD,
    min_claims: int = 100,
) -> pd.DataFrame:
    """
    Within a single specialty, find procedures with anomalous denial patterns.
    """
    spec_df = df[df["specialty"] == specialty]

    if spec_df.empty:
        return pd.DataFrame()

    proc_stats = spec_df.groupby(["procedure_code", "procedure_description"]).agg(
        mean_gap=("payment_gap_pct", "mean"),
        std_gap=("payment_gap_pct", "std"),
        claim_volume=("total_services", "sum"),
        provider_count=("provider_npi", "nunique"),
    ).reset_index()

    # Filter low-volume procedures
    proc_stats = proc_stats[proc_stats["claim_volume"] >= min_claims]

    if proc_stats.empty:
        return pd.DataFrame()

    specialty_mean = spec_df["payment_gap_pct"].mean()
    specialty_std = proc_stats["mean_gap"].std()  # use procedure-level std

    if specialty_std == 0 or np.isnan(specialty_std):
        proc_stats["z_score"] = 0
    else:
        proc_stats["z_score"] = (
            (proc_stats["mean_gap"] - specialty_mean) / specialty_std
        )

    proc_stats["is_anomaly"] = proc_stats["z_score"].abs() > z_threshold
    proc_stats["specialty"] = specialty
    proc_stats["specialty_mean"] = round(specialty_mean, 1)

    proc_stats["direction"] = np.where(
        proc_stats["z_score"] > 0, "Higher than specialty avg", "Lower than specialty avg"
    )

    return proc_stats.sort_values("z_score", ascending=False)


def detect_geographic_anomalies(
    df: pd.DataFrame,
    procedure_code: str | None = None,
    z_threshold: float = ANOMALY_Z_THRESHOLD,
) -> pd.DataFrame:
    """
    Find states where a procedure's denial rate deviates from the national norm.
    """
    from src.analysis import EXCLUDE_STATES, MIN_STATE_CLAIMS

    work_df = df[~df["state"].isin(EXCLUDE_STATES)]
    if procedure_code:
        work_df = work_df[work_df["procedure_code"] == procedure_code]

    state_stats = work_df.groupby("state").agg(
        mean_gap=("payment_gap_pct", "mean"),
        claim_volume=("total_services", "sum"),
        provider_count=("provider_npi", "nunique"),
    ).reset_index()

    state_stats = state_stats[state_stats["claim_volume"] >= MIN_STATE_CLAIMS]

    if state_stats.empty:
        return pd.DataFrame()

    national_mean = state_stats["mean_gap"].mean()
    national_std = state_stats["mean_gap"].std()

    if national_std == 0 or np.isnan(national_std):
        state_stats["z_score"] = 0
    else:
        state_stats["z_score"] = (
            (state_stats["mean_gap"] - national_mean) / national_std
        )

    state_stats["is_anomaly"] = state_stats["z_score"].abs() > z_threshold
    state_stats["national_avg_gap"] = round(national_mean, 1)

    state_stats["direction"] = np.where(
        state_stats["z_score"] > 0, "Higher than national avg", "Lower than national avg"
    )

    if procedure_code:
        state_stats["procedure_code"] = procedure_code

    return state_stats.sort_values("z_score", ascending=False)


def generate_anomaly_alerts(df: pd.DataFrame, z_threshold: float = 1.5) -> list[dict]:
    """
    Generate human-readable alert messages from detected anomalies.

    Uses a lower default threshold (1.5) than the config default (2.0)
    to ensure we actually surface useful alerts.
    """
    alerts = []

    # Specialty-level anomalies
    spec_anomalies = detect_specialty_anomalies(df, z_threshold=z_threshold)
    for _, row in spec_anomalies[spec_anomalies["is_anomaly"]].iterrows():
        direction = "above" if row["z_score"] > 0 else "below"
        alerts.append({
            "severity": "high" if abs(row["z_score"]) > 2.5 else "medium",
            "category": "Specialty",
            "entity": row["specialty"],
            "message": (
                f"{row['specialty']} has an avg payment gap of {row['mean_gap']:.1f}% "
                f"(dataset avg: {row['global_mean']:.1f}%). "
                f"This is {abs(row['z_score']):.1f} std deviations {direction} average, "
                f"affecting {int(row['claim_volume']):,} claims from "
                f"{int(row['provider_count']):,} providers."
            ),
            "z_score": round(row["z_score"], 2),
            "metric_value": round(row["mean_gap"], 1),
        })

    # Geographic anomalies
    geo_anomalies = detect_geographic_anomalies(df, z_threshold=z_threshold)
    if not geo_anomalies.empty:
        for _, row in geo_anomalies[geo_anomalies["is_anomaly"]].iterrows():
            direction = "above" if row["z_score"] > 0 else "below"
            alerts.append({
                "severity": "high" if abs(row["z_score"]) > 2.5 else "medium",
                "category": "Geographic",
                "entity": row["state"],
                "message": (
                    f"State {row['state']} has an avg payment gap of {row['mean_gap']:.1f}% "
                    f"(national avg: {row['national_avg_gap']:.1f}%). "
                    f"{abs(row['z_score']):.1f} std deviations {direction} average. "
                    f"{int(row['claim_volume']):,} claims, "
                    f"{int(row['provider_count']):,} providers."
                ),
                "z_score": round(row["z_score"], 2),
                "metric_value": round(row["mean_gap"], 1),
            })

    return sorted(alerts, key=lambda x: abs(x["z_score"]), reverse=True)

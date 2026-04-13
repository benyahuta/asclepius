from __future__ import annotations

"""
Denial risk prediction model.

Trains a real ML classifier to predict which provider-procedure combinations
will have high allowed-to-payment gaps (the closest proxy to actual denials
available in public CMS data).

Target variable: binary flag for claims where allowed_gap_pct > 80th percentile.
This captures cases where Medicare paid meaningfully less than it said it would
allow — a direct signal of adjustments, partial denials, and downcoding.

Features:
  - Specialty (categorical → one-hot encoded)
  - Place of service (F/O)
  - Log-transformed submitted charge (continuous)
  - Log-transformed service volume (continuous)
  - State-level average gap (geographic risk factor)
  - Procedure category (first character of HCPCS code)

Model: Random Forest, chosen for interpretable feature importances and
robustness to mixed feature types without extensive preprocessing.

Evaluation: Stratified train/test split (80/20), with precision, recall,
F1, and ROC-AUC reported. Feature importances show which factors drive
predicted denial risk — the same insight an AI RCM system would use to
prioritize pre-submission review.
"""

import logging
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
)
from sklearn.preprocessing import LabelEncoder

logger = logging.getLogger(__name__)


def prepare_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """
    Engineer features and target from raw claims data.

    Target: whether a claim has an allowed-to-payment gap above the 80th
    percentile (i.e., Medicare paid meaningfully less than it allowed).

    Returns:
        (X, y) — feature matrix and binary target
    """
    work = df.copy()

    # ── Target: high allowed-to-payment gap ──────────────────────────
    if "allowed_gap_pct" not in work.columns or work["allowed_gap_pct"].isna().all():
        # Fallback to payment gap if allowed gap isn't available
        gap_col = "payment_gap_pct"
    else:
        gap_col = "allowed_gap_pct"

    threshold = work[gap_col].quantile(0.80)
    work["target"] = (work[gap_col] > threshold).astype(int)

    # ── Features ─────────────────────────────────────────────────────
    # 1. Log-transformed charge (captures billing magnitude)
    work["log_charge"] = np.log1p(work["avg_submitted_charge"].clip(lower=0))

    # 2. Log-transformed volume (low volume = higher risk of issues)
    work["log_volume"] = np.log1p(work["total_services"].clip(lower=0))

    # 3. Charge-to-payment ratio (how much the payer discounted)
    work["charge_payment_ratio"] = (
        work["avg_medicare_payment"] / work["avg_submitted_charge"].replace(0, np.nan)
    ).fillna(0)

    # 4. Place of service (binary: facility=1, office=0)
    if "place_of_service" in work.columns:
        work["is_facility"] = (work["place_of_service"] == "F").astype(int)
    else:
        work["is_facility"] = 0

    # 5. Procedure category (first char of HCPCS — groups procedures by type)
    #    J = drugs, 9xxxx = surgery, 7xxxx = radiology, etc.
    if "procedure_code" in work.columns:
        work["proc_category"] = work["procedure_code"].str[0].fillna("X")
    else:
        work["proc_category"] = "X"

    # 6. Specialty (encode as integer)
    le_specialty = LabelEncoder()
    work["specialty_encoded"] = le_specialty.fit_transform(work["specialty"].fillna("Unknown"))

    # 7. State-level risk — placeholder column; actual values are computed
    #    AFTER the train/test split to prevent data leakage (see train_model).
    work["state_risk"] = 0.0

    # 8. Procedure category encoded
    le_proc = LabelEncoder()
    work["proc_cat_encoded"] = le_proc.fit_transform(work["proc_category"])

    feature_cols = [
        "log_charge",
        "log_volume",
        "charge_payment_ratio",
        "is_facility",
        "specialty_encoded",
        "state_risk",
        "proc_cat_encoded",
    ]

    X = work[feature_cols].copy()
    y = work["target"]

    # Store mappings for interpretation
    X.attrs["specialty_encoder"] = le_specialty
    X.attrs["proc_encoder"] = le_proc
    X.attrs["feature_names"] = feature_cols
    X.attrs["gap_threshold"] = threshold
    X.attrs["gap_column"] = gap_col
    X.attrs["state_series"] = work["state"]

    return X, y


def train_model(df: pd.DataFrame) -> dict:
    """
    Train a denial risk prediction model and return results.

    Returns a dict with:
        - model: trained RandomForestClassifier
        - metrics: accuracy, precision, recall, f1, roc_auc
        - feature_importances: sorted list of (feature_name, importance)
        - classification_report: full text report
        - metadata: threshold, sample sizes, etc.
    """
    X, y = prepare_features(df)

    # Drop rows with NaN features
    valid_mask = X.notna().all(axis=1)
    X = X[valid_mask]
    y = y[valid_mask]

    if len(X) < 100:
        return {"error": "Not enough data to train model (need 100+ rows)."}

    # Stratified split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Compute state_risk from TRAINING data only to prevent data leakage.
    # Uses claim-volume-weighted means so high-volume providers dominate.
    state_series = X.attrs.get("state_series")
    gap_col = X.attrs.get("gap_column", "allowed_gap_pct")
    if state_series is not None:
        train_states = state_series.loc[X_train.index]
        train_gaps = df.loc[X_train.index, gap_col]
        train_volumes = df.loc[X_train.index, "total_services"]

        # Weighted mean per state from training data only
        _num = (train_gaps * train_volumes).groupby(train_states).sum()
        _den = train_volumes.groupby(train_states).sum()
        state_means_train = (_num / _den.replace(0, np.nan)).fillna(0)
        fallback = (train_gaps * train_volumes).sum() / train_volumes.sum()

        all_states = state_series.loc[X.index]
        X_train = X_train.copy()
        X_test = X_test.copy()
        X_train["state_risk"] = all_states.loc[X_train.index].map(state_means_train).fillna(fallback)
        X_test["state_risk"] = all_states.loc[X_test.index].map(state_means_train).fillna(fallback)

    # Train
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=12,
        min_samples_leaf=20,
        class_weight="balanced",  # handle class imbalance (only ~20% positive)
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    # Predict
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    # Metrics
    metrics = {
        "accuracy": round(accuracy_score(y_test, y_pred), 3),
        "precision": round(precision_score(y_test, y_pred, zero_division=0), 3),
        "recall": round(recall_score(y_test, y_pred, zero_division=0), 3),
        "f1": round(f1_score(y_test, y_pred, zero_division=0), 3),
        "roc_auc": round(roc_auc_score(y_test, y_prob), 3),
    }

    # Feature importances
    feature_names = X.attrs.get("feature_names", X.columns.tolist())
    importances = sorted(
        zip(feature_names, model.feature_importances_),
        key=lambda x: x[1],
        reverse=True,
    )

    # Human-readable feature name mapping
    readable_names = {
        "log_charge": "Submitted Charge (log)",
        "log_volume": "Service Volume (log)",
        "charge_payment_ratio": "Payment-to-Charge Ratio",
        "is_facility": "Place of Service (Facility)",
        "specialty_encoded": "Provider Specialty",
        "state_risk": "State-Level Risk",
        "proc_cat_encoded": "Procedure Category",
    }

    importances_readable = [
        (readable_names.get(name, name), round(imp, 4))
        for name, imp in importances
    ]

    report = classification_report(y_test, y_pred, zero_division=0)

    return {
        "model": model,
        "metrics": metrics,
        "feature_importances": importances_readable,
        "classification_report": report,
        "metadata": {
            "train_size": len(X_train),
            "test_size": len(X_test),
            "positive_rate": round(y.mean() * 100, 1),
            "gap_threshold": round(X.attrs.get("gap_threshold", 0), 2),
            "gap_column": X.attrs.get("gap_column", "unknown"),
            "n_features": len(feature_names),
        },
    }

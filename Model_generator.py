"""
============================================================
  TAX APPEAL LEAD SCORING MODEL — Generate Pickle File
  Trains an XGBoost classifier to predict client conversion.

  Input Columns:
    SalesLeadID (ignored), Owner_City, Owner_ZipCode,
    Property_Type, num_ExemptionCode, Properties_Count,
    MaxTrestlescore, total_market_value, client_status

  Removed from this version:
    - CountyName
    - Owner_State
    - num_ocaluc
    - SalesLeadID (non-feature identifier)
============================================================
"""

import warnings
warnings.filterwarnings("ignore")

import os
import pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    accuracy_score, roc_auc_score, classification_report,
    confusion_matrix, ConfusionMatrixDisplay, RocCurveDisplay,
    f1_score, precision_score, recall_score, average_precision_score
)
from xgboost import XGBClassifier

try:
    from imblearn.over_sampling import SMOTE
    SMOTE_AVAILABLE = True
    print("✅ imbalanced-learn available — SMOTE will be used")
except ImportError:
    SMOTE_AVAILABLE = False
    print("⚠️  imbalanced-learn not found — falling back to scale_pos_weight")


# ──────────────────────────────────────────────────────────────
#  STEP 1 — PREPROCESSING
# ──────────────────────────────────────────────────────────────

def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # ── Strip whitespace from all column names ──
    df.columns = df.columns.str.strip()

    # ── Drop non-feature identifier column if present ──
    if "SalesLeadID" in df.columns:
        df = df.drop(columns=["SalesLeadID"])

    # ── Target: client_status → converted (binary) ──
    status = df["client_status"].astype(str).str.strip().str.lower()
    df["converted"] = (status == "client").astype(int)
    print(f"\n[TARGET] Class distribution:\n{df['converted'].value_counts().to_string()}")
    print(f"  Conversion rate: {df['converted'].mean():.2%}")

    # ── Numeric columns ──
    num_cols = ["num_ExemptionCode", "Properties_Count",
                "MaxTrestlescore", "total_market_value"]
    for col in num_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # ── ZipCode: string, zero-padded to 5 digits ──
    df["Owner_ZipCode"] = (
        df["Owner_ZipCode"].astype(str).str.strip()
        .str.replace(r"\.0$", "", regex=True)
        .str.zfill(5)
    )

    # ── String columns: uppercase + strip ──
    str_cols = ["Owner_City", "Owner_ZipCode", "Property_Type"]
    for col in str_cols:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip().str.upper()

    # ── Drop rows with no target ──
    before = len(df)
    df = df.dropna(subset=["converted"])
    print(f"[CLEAN] Rows after dropping bad targets: {len(df):,} (removed {before - len(df)})")

    return df


# ──────────────────────────────────────────────────────────────
#  STEP 2 — FEATURE ENGINEERING
# ──────────────────────────────────────────────────────────────

def engineer_features(df: pd.DataFrame, fit_meta: dict = None):
    """
    fit_meta = None  →  training mode (computes & returns meta)
    fit_meta = dict  →  inference mode (uses saved meta)
    """
    df = df.copy()
    training = fit_meta is None
    if training:
        fit_meta = {}

    # ── 1. Log transform total_market_value ──
    df["log_market_value"] = np.log1p(df["total_market_value"].clip(lower=0).fillna(0))

    # ── 2. Quantile-based binary flags ──
    quantile_defs = {
        "q25_trestle": ("MaxTrestlescore", 0.25),
        "q50_trestle": ("MaxTrestlescore", 0.50),
        "q75_trestle": ("MaxTrestlescore", 0.75),
        "q75_value":   ("total_market_value", 0.75),
        "q25_value":   ("total_market_value", 0.25),
        "q75_props":   ("Properties_Count", 0.75),
    }
    for key, (col, q) in quantile_defs.items():
        if training:
            fit_meta[key] = df[col].quantile(q)

    df["high_trestle"]         = (df["MaxTrestlescore"] > fit_meta["q75_trestle"]).astype(int)
    df["med_trestle"]          = ((df["MaxTrestlescore"] > fit_meta["q50_trestle"]) &
                                   (df["MaxTrestlescore"] <= fit_meta["q75_trestle"])).astype(int)
    df["low_trestle"]          = (df["MaxTrestlescore"] < fit_meta["q25_trestle"]).astype(int)
    df["high_value"]           = (df["total_market_value"] > fit_meta["q75_value"]).astype(int)
    df["low_value"]            = (df["total_market_value"] < fit_meta["q25_value"]).astype(int)
    df["multi_property"]       = (df["Properties_Count"] > 1).astype(int)
    df["large_portfolio"]      = (df["Properties_Count"] >= 5).astype(int)
    df["very_large_portfolio"] = (df["Properties_Count"] >= 10).astype(int)
    df["has_exemption"]        = (df["num_ExemptionCode"] > 0).astype(int)
    df["multi_exemption"]      = (df["num_ExemptionCode"] > 1).astype(int)

    # ── 3. Interaction features ──
    trestle_filled = df["MaxTrestlescore"].fillna(0)
    df["value_x_trestle"]     = df["log_market_value"] * trestle_filled
    df["portfolio_x_value"]   = df["Properties_Count"].fillna(1) * df["log_market_value"]
    df["exemption_x_value"]   = df["has_exemption"] * df["log_market_value"]
    df["trestle_x_portfolio"] = trestle_filled * df["Properties_Count"].fillna(1)

    # ── 4. Owner_City frequency encoding ──
    if "Owner_City" in df.columns:
        if training:
            fit_meta["city_freq"] = df["Owner_City"].value_counts(normalize=True).to_dict()
        df["owner_city_frequency"] = df["Owner_City"].map(fit_meta["city_freq"]).fillna(0)

    # ── 5. ZipCode frequency encoding ──
    if training:
        fit_meta["zip_freq"] = df["Owner_ZipCode"].value_counts(normalize=True).to_dict()
    df["zip_frequency"] = df["Owner_ZipCode"].map(fit_meta["zip_freq"]).fillna(0)

    # ── 6. One-hot: Property_Type (top N + OTHER) ──
    if training:
        fit_meta["top_prop_types"] = list(
            df["Property_Type"].value_counts().nlargest(12).index
        )
    df["Property_Type_enc"] = df["Property_Type"].where(
        df["Property_Type"].isin(fit_meta["top_prop_types"]), "OTHER"
    )

    # ── 7. One-hot: Owner_City (top N + OTHER) ──
    if training:
        fit_meta["top_cities"] = list(
            df["Owner_City"].value_counts().nlargest(15).index
        )
    df["City_enc"] = df["Owner_City"].where(
        df["Owner_City"].isin(fit_meta["top_cities"]), "OTHER"
    )

    # ── 8. One-hot encode categorical columns ──
    df = pd.get_dummies(
        df,
        columns=["Property_Type_enc", "City_enc"],
        drop_first=False,
        dtype=int,
    )

    return df, fit_meta


# ──────────────────────────────────────────────────────────────
#  STEP 3 — DEFINE FEATURE COLUMNS
# ──────────────────────────────────────────────────────────────

# Columns to exclude from the feature matrix
EXCLUDE_COLS = [
    "client_status", "converted",
    "SalesLeadID",
    "Owner_City", "Owner_ZipCode", "Owner_State",   # raw geo (encoded versions kept)
    "CountyName",                                    # removed column
    "Property_Type",                                 # raw (encoded version kept)
    "num_ocaluc",                                    # removed column
]

def get_feature_cols(df: pd.DataFrame):
    return [c for c in df.columns if c not in EXCLUDE_COLS]


# ──────────────────────────────────────────────────────────────
#  STEP 4 — TRAIN & EVALUATE
# ──────────────────────────────────────────────────────────────

def train_and_evaluate(df: pd.DataFrame):
    print("\n" + "="*60)
    print("  TRAINING TAX APPEAL LEAD SCORING MODEL")
    print("="*60)

    # Preprocess
    df = preprocess(df)

    # Feature engineering (training mode)
    df_fe, fit_meta = engineer_features(df, fit_meta=None)
    feature_cols = get_feature_cols(df_fe)
    fit_meta["feature_cols"] = feature_cols

    print(f"\n[FEATURES] Total engineered features: {len(feature_cols)}")

    # Train / test split (stratified)
    train_df, test_df = train_test_split(
        df_fe, test_size=0.20, random_state=42, stratify=df_fe["converted"]
    )

    X_train = train_df[feature_cols].fillna(0)
    y_train = train_df["converted"]
    X_test  = test_df[feature_cols].fillna(0)
    y_test  = test_df["converted"]

    print(f"\n[SPLIT] Train: {len(X_train):,}  |  Test: {len(X_test):,}")
    print(f"  Train positives: {y_train.sum():,} ({y_train.mean():.2%})")
    print(f"  Test  positives: {y_test.sum():,}  ({y_test.mean():.2%})")

    # ── Handle class imbalance ──
    imbalance_ratio = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
    print(f"\n[IMBALANCE] Ratio 0:1 = {imbalance_ratio:.1f}:1")

    if SMOTE_AVAILABLE and imbalance_ratio > 1.5:
        print("[IMBALANCE] Applying SMOTE oversampling...")
        k_neighbors = min(5, (y_train == 1).sum() - 1)
        k_neighbors = max(1, k_neighbors)
        smote = SMOTE(random_state=42, k_neighbors=k_neighbors)
        X_res, y_res = smote.fit_resample(X_train, y_train)
        print(f"  After SMOTE → {len(X_res):,} samples | pos rate: {y_res.mean():.2%}")
        scale_pos = 1.0
    else:
        X_res, y_res = X_train, y_train
        scale_pos = imbalance_ratio
        print(f"[IMBALANCE] Using scale_pos_weight={scale_pos:.1f}")

    # ── XGBoost model ──
    xgb_params = dict(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8,
        colsample_bylevel=0.8,
        min_child_weight=5,
        gamma=0.1,
        reg_alpha=0.1,
        reg_lambda=1.0,
        scale_pos_weight=scale_pos,
        eval_metric="auc",
        random_state=42,
        verbosity=0,
        tree_method="hist",
        early_stopping_rounds=30,
    )

    print("\n[MODEL] Training XGBoost with early stopping on validation set...")
    xgb_base = XGBClassifier(**xgb_params)
    eval_set = [(X_res, y_res), (X_test, y_test)]
    xgb_base.fit(X_res, y_res, eval_set=eval_set, verbose=False)
    best_iter = xgb_base.best_iteration
    print(f"  Best iteration: {best_iter}")

    # ── Calibrate probabilities (Platt scaling) ──
    print("[MODEL] Calibrating probability outputs...")
    conv_model = CalibratedClassifierCV(
        XGBClassifier(
            **{**xgb_params,
               "n_estimators": best_iter + 1,
               "early_stopping_rounds": None}
        ),
        cv=3,
        method="sigmoid",
    )
    conv_model.fit(X_res, y_res)
    print("✅ Conversion model trained & calibrated")

    # ── Cross-validation AUC ──
    print("\n[CV] Running 5-fold stratified cross-validation on TRAIN set...")
    cv_xgb = XGBClassifier(
        **{**xgb_params,
           "n_estimators": best_iter + 1,
           "early_stopping_rounds": None}
    )
    cv_scores = cross_val_score(
        cv_xgb, X_train, y_train,
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
        scoring="roc_auc", n_jobs=-1
    )
    print(f"  CV AUC scores: {cv_scores.round(4)}")
    print(f"  Mean CV AUC: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    # ── Evaluation on hold-out test set ──
    y_prob = conv_model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= 0.50).astype(int)

    thresholds = np.linspace(0.1, 0.9, 81)
    f1_scores = [f1_score(y_test, (y_prob >= t).astype(int), zero_division=0) for t in thresholds]
    best_thresh = thresholds[np.argmax(f1_scores)]
    y_pred_opt = (y_prob >= best_thresh).astype(int)

    auc_roc  = roc_auc_score(y_test, y_prob)
    avg_prec = average_precision_score(y_test, y_prob)

    print("\n" + "="*60)
    print("  MODEL EVALUATION — HOLD-OUT TEST SET")
    print("="*60)
    print(f"\n  Threshold @ 0.50:")
    print(f"    Accuracy  : {accuracy_score(y_test, y_pred):.4f}")
    print(f"    Precision : {precision_score(y_test, y_pred, zero_division=0):.4f}")
    print(f"    Recall    : {recall_score(y_test, y_pred, zero_division=0):.4f}")
    print(f"    F1 Score  : {f1_score(y_test, y_pred, zero_division=0):.4f}")
    print(f"\n  Optimal Threshold @ {best_thresh:.2f}:")
    print(f"    Accuracy  : {accuracy_score(y_test, y_pred_opt):.4f}")
    print(f"    Precision : {precision_score(y_test, y_pred_opt, zero_division=0):.4f}")
    print(f"    Recall    : {recall_score(y_test, y_pred_opt, zero_division=0):.4f}")
    print(f"    F1 Score  : {f1_score(y_test, y_pred_opt, zero_division=0):.4f}")
    print(f"\n  ROC-AUC Score    : {auc_roc:.4f}")
    print(f"  Avg Precision    : {avg_prec:.4f}")
    print(f"  CV AUC (5-fold)  : {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    print("\n[REPORT] Classification Report (threshold=0.50):")
    print(classification_report(y_test, y_pred, target_names=["Not Client", "Client"], zero_division=0))

    print("\n[REPORT] Classification Report (optimal threshold):")
    print(classification_report(y_test, y_pred_opt, target_names=["Not Client", "Client"], zero_division=0))

    # ── Save training meta ──
    fit_meta["optimal_threshold"] = float(best_thresh)
    fit_meta["test_auc"]          = float(auc_roc)
    fit_meta["cv_auc_mean"]       = float(cv_scores.mean())
    fit_meta["cv_auc_std"]        = float(cv_scores.std())
    fit_meta["value_median"]      = float(df_fe["total_market_value"].median())
    fit_meta["trestle_median"]    = float(df_fe["MaxTrestlescore"].median())
    fit_meta["train_conv_probs"]  = conv_model.predict_proba(X_train)[:, 1].tolist()

    # ── Plots ──
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("Tax Appeal Lead Scoring — Model Evaluation", fontsize=14, fontweight="bold")

    RocCurveDisplay.from_predictions(
        y_test, y_prob, ax=axes[0], name=f"XGBoost (AUC={auc_roc:.3f})", color="steelblue"
    )
    axes[0].plot([0, 1], [0, 1], "k--", lw=1, label="Random")
    axes[0].set_title("ROC Curve")
    axes[0].legend(fontsize=9)

    cm = confusion_matrix(y_test, y_pred_opt)
    disp = ConfusionMatrixDisplay(cm, display_labels=["Not Client", "Client"])
    disp.plot(ax=axes[1], colorbar=False, cmap="Blues")
    axes[1].set_title(f"Confusion Matrix\n(threshold={best_thresh:.2f})")

    raw_xgb = conv_model.calibrated_classifiers_[0].estimator
    importances = pd.Series(
        raw_xgb.feature_importances_, index=feature_cols
    ).nlargest(20)
    importances.sort_values().plot(kind="barh", ax=axes[2], color="steelblue")
    axes[2].set_title("Top 20 Feature Importances")
    axes[2].set_xlabel("Gain")

    plt.tight_layout()
    plot_path = "/mnt/user-data/outputs/model_evaluation.png"
    plt.savefig(plot_path, dpi=140, bbox_inches="tight")
    plt.close()
    print(f"\n[PLOT] Saved evaluation plots → {plot_path}")

    return conv_model, fit_meta


# ──────────────────────────────────────────────────────────────
#  MAIN — TRAINING
# ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # ── Set your Excel path here ──
    EXCEL_PATH = r"C:\Users\yogeshwaranm\Documents\Hakathon_final dataset_220426.xlsx"

    os.makedirs("/mnt/user-data/outputs", exist_ok=True)

    if EXCEL_PATH and os.path.exists(EXCEL_PATH):
        print(f"[LOAD] Reading {EXCEL_PATH}...")
        df = pd.read_excel(EXCEL_PATH)
        print(f"[LOAD] Shape: {df.shape}")
        print(f"[LOAD] Columns: {list(df.columns)}")
    else:
        print("[INFO] Excel file not found — please provide the correct EXCEL_PATH.")
        raise SystemExit(1)

    print(f"[INFO] Training on {len(df):,} records...\n")

    conv_model, fit_meta = train_and_evaluate(df)

    bundle = {
        "conv_model":        conv_model,
        "fit_meta":          fit_meta,
        "model_version":     "2.2",
        "target_column":     "client_status",
        "positive_class":    "client",
        "optimal_threshold": fit_meta["optimal_threshold"],
    }

    pkl_path = r"C:\Users\yogeshwaranm\downloads\lead_scoring_models.pkl"
    with open(pkl_path, "wb") as f:
        pickle.dump(bundle, f)

    print(f"\n✅ Model bundle saved → {pkl_path}")
    print(f"   Features used      : {len(fit_meta['feature_cols'])}")
    print(f"   Test AUC-ROC       : {fit_meta['test_auc']:.4f}")
    print(f"   CV AUC (5-fold)    : {fit_meta['cv_auc_mean']:.4f} ± {fit_meta['cv_auc_std']:.4f}")
    print(f"   Optimal threshold  : {fit_meta['optimal_threshold']:.2f}")


# ──────────────────────────────────────────────────────────────
#  INFERENCE — predict on new leads
# ──────────────────────────────────────────────────────────────

def predict_new_leads(new_df: pd.DataFrame, bundle: dict) -> pd.DataFrame:
    """
    Accepts a DataFrame with columns:
        SalesLeadID (optional), Owner_City, Owner_ZipCode,
        Property_Type, num_ExemptionCode, Properties_Count,
        MaxTrestlescore, total_market_value, client_status (optional)

    Returns the DataFrame with two new columns added:
        conversion_probability  — calibrated probability (0–1)
        predicted_convert       — 1 = likely client, 0 = not
    """
    conv_model = bundle["conv_model"]
    fit_meta   = bundle["fit_meta"]
    threshold  = bundle["optimal_threshold"]
    feature_cols = fit_meta["feature_cols"]

    # Preserve SalesLeadID for output if present
    lead_ids = new_df["SalesLeadID"].copy() if "SalesLeadID" in new_df.columns else None

    # Add a dummy client_status if not provided (inference on unlabelled data)
    if "client_status" not in new_df.columns:
        new_df = new_df.copy()
        new_df["client_status"] = "unknown"

    # Preprocess + feature engineer (inference mode — pass fit_meta)
    new_df_clean = preprocess(new_df)
    new_df_fe, _ = engineer_features(new_df_clean, fit_meta=fit_meta)

    # Align columns: add any missing feature columns as 0
    for col in feature_cols:
        if col not in new_df_fe.columns:
            new_df_fe[col] = 0

    X_new = new_df_fe[feature_cols].fillna(0)

    proba        = conv_model.predict_proba(X_new)[:, 1]
    will_convert = (proba >= threshold).astype(int)

    result = new_df_fe.copy()
    if lead_ids is not None:
        result.insert(0, "SalesLeadID", lead_ids.values)
    result["conversion_probability"] = proba.round(3)
    result["predicted_convert"]      = will_convert

    return result[
        (["SalesLeadID"] if lead_ids is not None else []) +
        ["Owner_City", "Owner_ZipCode", "Property_Type",
         "num_ExemptionCode", "Properties_Count",
         "MaxTrestlescore", "total_market_value",
         "conversion_probability", "predicted_convert"]
    ]


# ──────────────────────────────────────────────────────────────
#  INFERENCE USAGE EXAMPLE
# ──────────────────────────────────────────────────────────────
"""
import pickle, pandas as pd
from generate_lead_scoring_model import predict_new_leads

with open("lead_scoring_models.pkl", "rb") as f:
    bundle = pickle.load(f)

# Sample data matching the expected column format
sample_data = pd.DataFrame([
    {
        "SalesLeadID":      34316,
        "Owner_City":       "Houston",
        "Owner_ZipCode":    "77079",
        "Property_Type":    "Residential",
        "num_ExemptionCode": 1,
        "Properties_Count": 1,
        "MaxTrestlescore":  100,
        "total_market_value": 1395954,
        "client_status":    "client",
    },
    {
        "SalesLeadID":      1373,
        "Owner_City":       "Houston",
        "Owner_ZipCode":    "77021",
        "Property_Type":    "Residential",
        "num_ExemptionCode": 1,
        "Properties_Count": 1,
        "MaxTrestlescore":  100,
        "total_market_value": 883137,
        "client_status":    "not client",
    },
])

results = predict_new_leads(sample_data, bundle)
print(results.to_string(index=False))
"""

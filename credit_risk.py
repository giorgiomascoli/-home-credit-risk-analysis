"""Home Credit credit-risk analysis: train-only WoE, calibrated logistic PD, illustrative EL.

Run with: python credit_risk.py --data '/path/to/application_train.csv'
No customer-level data is written to the public repository by default.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.frozen import FrozenEstimator
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, brier_score_loss, classification_report,
                             confusion_matrix, log_loss, roc_auc_score, roc_curve,
                             precision_recall_curve)
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

FEATURES = ["AMT_INCOME_TOTAL", "AMT_CREDIT", "AMT_ANNUITY",
            "EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]
TARGET = "TARGET"
ID = "SK_ID_CURR"
RANDOM_STATE = 42
LGD_ASSUMED = 0.45


def load_data(path: str | Path) -> pd.DataFrame:
    """Read the Home Credit 25-column teaching extract or Kaggle application_train.csv."""
    frame = pd.read_csv(path, low_memory=False)
    required = [ID, TARGET, *FEATURES]
    missing = [col for col in required if col not in frame.columns]
    if missing:
        raise ValueError(f"Missing required Home Credit columns: {missing}")
    if frame[ID].isna().any() or frame[ID].duplicated().any():
        raise ValueError("Customer IDs must be unique and non-missing.")
    frame = frame[required].copy()
    for col in [TARGET, *FEATURES]:
        frame[col] = pd.to_numeric(frame[col], errors="coerce")
    if frame[TARGET].isna().any() or not frame[TARGET].isin([0, 1]).all():
        raise ValueError("TARGET must contain only 0 (non-default) and 1 (default).")
    if frame["AMT_CREDIT"].isna().any() or (frame["AMT_CREDIT"] <= 0).any():
        raise ValueError("AMT_CREDIT must be strictly positive for an EAD proxy.")
    frame[TARGET] = frame[TARGET].astype(int)
    return frame


def split_data(df: pd.DataFrame):
    """60% model development, 20% calibration, 20% final test (stratified)."""
    dev, test = train_test_split(df, test_size=0.2, stratify=df[TARGET],
                                 random_state=RANDOM_STATE)
    train, calibration = train_test_split(dev, test_size=0.25,
                                          stratify=dev[TARGET], random_state=RANDOM_STATE)
    return train, calibration, test


class TrainOnlyWoE:
    """Supervised decision-tree bins and smoothed Weight of Evidence.

    Each feature's cut points are learned from train only. Missing observations
    receive an explicit bin. WoE = log(share_of_goods / share_of_bads).
    Tree cutpoints do not rely on scorecardpy so the project runs offline.
    """

    def __init__(self, max_bins: int = 5, min_leaf_fraction: float = 0.05, smoothing: float = 0.5):
        self.max_bins = max_bins
        self.min_leaf_fraction = min_leaf_fraction
        self.smoothing = smoothing
        self.thresholds_: dict[str, np.ndarray] = {}
        self.woe_maps_: dict[str, dict[str, float]] = {}
        self.tables_: dict[str, pd.DataFrame] = {}
        self.iv_: dict[str, float] = {}

    def _assign_bins(self, s: pd.Series, feature: str) -> pd.Series:
        raw = pd.to_numeric(s, errors="coerce")
        thresholds = self.thresholds_[feature]
        result = pd.Series(index=raw.index, dtype="object")
        good = raw.notna()
        result.loc[good] = [f"bin_{i}" for i in np.searchsorted(
            thresholds, raw.loc[good].to_numpy(), side="left")]
        result.loc[~good] = "missing"
        return result

    def fit(self, x: pd.DataFrame, y: pd.Series) -> "TrainOnlyWoE":
        if x.empty or len(x) != len(y):
            raise ValueError("Feature matrix must be non-empty and align with the target.")
        if not x.index.equals(y.index):
            raise ValueError("X/y indices must match in WoE training.")
        y = y.astype(int)
        if set(y.unique()) != {0, 1}:
            raise ValueError("WoE requires both target classes (0/1).")
        self.thresholds_.clear()
        self.woe_maps_.clear()
        self.tables_.clear()
        self.iv_.clear()
        for feature in x.columns:
            series = pd.to_numeric(x[feature], errors="coerce")
            observed = series.notna()
            if observed.sum() >= 20 and series.loc[observed].nunique() > 1:
                tree = DecisionTreeClassifier(
                    max_leaf_nodes=self.max_bins, min_samples_leaf=self.min_leaf_fraction,
                    random_state=RANDOM_STATE)
                tree.fit(series.loc[observed].to_numpy().reshape(-1, 1), y.loc[observed])
                thresholds = np.sort(tree.tree_.threshold[tree.tree_.feature >= 0])
            else:
                thresholds = np.array([], dtype=float)
            self.thresholds_[feature] = np.asarray(thresholds, dtype=float)
            bins = self._assign_bins(series, feature)
            tbl = pd.DataFrame({"bin": bins, "target": y})
            counts = (tbl.groupby("bin", sort=True)["target"]
                      .agg(count="size", defaults="sum"))
            counts["non_defaults"] = counts["count"] - counts["defaults"]
            k = len(counts)
            pct_good = (counts["non_defaults"] + self.smoothing) / (
                counts["non_defaults"].sum() + self.smoothing * k)
            pct_bad = (counts["defaults"] + self.smoothing) / (
                counts["defaults"].sum() + self.smoothing * k)
            counts["woe"] = np.log(pct_good / pct_bad)
            counts["bin_iv"] = (pct_good - pct_bad) * counts["woe"]
            counts["default_rate"] = counts["defaults"] / counts["count"]
            counts = counts.reset_index()
            # Explicitly retain the ordered interval boundaries for auditability.
            def interval_label(label: str) -> str:
                if label == "missing":
                    return "Missing"
                j = int(label.split("_")[1])
                left = "-inf" if j == 0 else f"{thresholds[j - 1]:.6g}"
                right = "+inf" if j == len(thresholds) else f"{thresholds[j]:.6g}"
                return f"({left}, {right}]"
            counts["interval"] = counts["bin"].map(interval_label)
            self.tables_[feature] = counts[["bin", "interval", "count", "defaults",
                                            "default_rate", "woe", "bin_iv"]]
            self.woe_maps_[feature] = counts.set_index("bin")["woe"].to_dict()
            self.iv_[feature] = float(counts["bin_iv"].sum())
        return self

    def transform(self, x: pd.DataFrame) -> pd.DataFrame:
        if not self.woe_maps_:
            raise RuntimeError("Fit the WoE encoder on the training sample first.")
        out = pd.DataFrame(index=x.index)
        for feature in self.woe_maps_:
            bins = self._assign_bins(x[feature], feature)
            # A missing bin unseen during training is mapped to neutral WoE=0.
            # Observed numerical bins always have support because of ±infinity edges.
            out[f"{feature}_woe"] = bins.map(self.woe_maps_[feature]).fillna(0.0).astype(float)
            if not np.isfinite(out[f"{feature}_woe"]).all():
                raise ValueError(f"Missing or infinite WoE for feature {feature}")
        return out


def fit_models(train: pd.DataFrame, calibration: pd.DataFrame):
    """Keep all supervised transformations in training; Platt calibration on holdout."""
    encoder = TrainOnlyWoE().fit(train[FEATURES], train[TARGET])
    x_train = encoder.transform(train[FEATURES])
    x_cal = encoder.transform(calibration[FEATURES])
    model = LogisticRegression(class_weight="balanced", max_iter=1000,
                               random_state=RANDOM_STATE)
    model.fit(x_train, train[TARGET])
    calibrated = CalibratedClassifierCV(FrozenEstimator(model), method="sigmoid")
    calibrated.fit(x_cal, calibration[TARGET])
    return encoder, model, calibrated


def calculate_metrics(test: pd.DataFrame, pd_raw: np.ndarray,
                      pd_cal: np.ndarray, labels_balanced: np.ndarray,
                      train_default_rate: float) -> dict:
    y = test[TARGET].to_numpy()
    if (len(y) != len(pd_raw)) or (len(y) != len(pd_cal)):
        raise ValueError("PD and target lengths do not match")
    if ((np.asarray(pd_cal) < 0) | (np.asarray(pd_cal) > 1)).any():
        raise ValueError("Calibrated PD must fall in [0,1].")
    ead = test["AMT_CREDIT"].to_numpy(dtype=float)  # proxy, not actual future EAD
    el = pd_cal * LGD_ASSUMED * ead
    cm = confusion_matrix(y, labels_balanced)
    return {
        "n_test": int(len(y)),
        "default_rate": float(y.mean()),
        "mean_pd_raw": float(pd_raw.mean()),
        "mean_pd_calibrated": float(pd_cal.mean()),
        "auc_raw": float(roc_auc_score(y, pd_raw)),
        "auc_calibrated": float(roc_auc_score(y, pd_cal)),
        "average_precision_raw": float(average_precision_score(y, pd_raw)),
        "average_precision_calibrated": float(average_precision_score(y, pd_cal)),
        "brier_raw": float(brier_score_loss(y, pd_raw)),
        "brier_calibrated": float(brier_score_loss(y, pd_cal)),
        "brier_constant_baseline": float(brier_score_loss(y, np.full(len(y), train_default_rate))),
        "log_loss_raw": float(log_loss(y, pd_raw)),
        "log_loss_calibrated": float(log_loss(y, pd_cal)),
        "balanced_threshold_confusion_matrix": cm.tolist(),
        "balanced_threshold_classification_report": classification_report(
            y, labels_balanced, output_dict=True, zero_division=0),
        "ead_proxy_total": float(ead.sum()),
        "lgd_assumed": float(LGD_ASSUMED),
        "expected_loss_proxy_total": float(el.sum()),
        "expected_loss_to_ead": float(el.sum() / ead.sum()),
    }


def save_plots(y: np.ndarray, raw: np.ndarray, cal: np.ndarray, iv: pd.Series,
               out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7.6, 4.5))
    iv.sort_values().plot(kind="barh", ax=ax, color="#2362a1")
    ax.set_xlabel("Information Value (train only)")
    ax.set_title("Predictive strength of selected features")
    fig.tight_layout(); fig.savefig(out_dir / "01_information_value.png", dpi=170); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.6, 5))
    for prob, name in ((raw, "Weighted logistic"), (cal, "Calibrated PD")):
        fpr, tpr, _ = roc_curve(y, prob)
        ax.plot(fpr, tpr, label=f"{name}: AUC {roc_auc_score(y,prob):.3f}")
    ax.plot([0,1], [0,1], linestyle="--", color="gray", label="Random")
    ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate")
    ax.set_title("Out-of-sample ROC curves"); ax.legend()
    fig.tight_layout(); fig.savefig(out_dir / "02_roc_curve.png", dpi=170); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.6, 5))
    for prob, name in ((raw, "Before calibration"), (cal, "After calibration")):
        actual, predicted = calibration_curve(y, prob, n_bins=10, strategy="quantile")
        ax.plot(predicted, actual, marker="o", label=name)
    ax.plot([0,1], [0,1], linestyle="--", color="gray", label="Perfect calibration")
    ax.set_xlabel("Mean predicted PD"); ax.set_ylabel("Observed default frequency")
    ax.set_title("Reliability diagram – test sample"); ax.legend()
    fig.tight_layout(); fig.savefig(out_dir / "03_calibration.png", dpi=170); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.6, 5))
    p, recall, _ = precision_recall_curve(y, cal)
    ax.plot(recall, p, label=f"Calibrated model: AP {average_precision_score(y,cal):.3f}")
    ax.axhline(float(y.mean()), color="gray", linestyle="--", label="Default prevalence")
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
    ax.set_title("Precision–recall: minority default class"); ax.legend()
    fig.tight_layout(); fig.savefig(out_dir / "04_precision_recall.png", dpi=170); plt.close(fig)


def run_analysis(data_path: str | Path, output_dir: str | Path = "results") -> dict:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    df = load_data(data_path)
    train, cal, test = split_data(df)
    encoder, raw_model, model_cal = fit_models(train, cal)
    x_test = encoder.transform(test[FEATURES])
    pd_raw = raw_model.predict_proba(x_test)[:, 1]
    pd_cal = model_cal.predict_proba(x_test)[:, 1]
    predicted_balanced = raw_model.predict(x_test)
    metrics = calculate_metrics(test, pd_raw, pd_cal, predicted_balanced, train[TARGET].mean())
    metrics.update({
        "n_total": int(len(df)), "n_train": int(len(train)), "n_calibration": int(len(cal)),
        "overall_default_rate": float(df[TARGET].mean()),
        "sample_missing_rates": {feat: float(df[feat].isna().mean()) for feat in FEATURES},
        "method": "Decision-tree supervised WoE bins trained on train only; weighted logistic; held-out sigmoid calibration",
        "split_random_seed": RANDOM_STATE,
    })
    pd.Series(encoder.iv_, name="IV").sort_values(ascending=False).to_csv(out / "information_value.csv", header=True)
    all_bins = pd.concat([table.assign(feature=feature) for feature, table in encoder.tables_.items()],
                         ignore_index=True)
    all_bins.to_csv(out / "woe_binning_table.csv", index=False)
    pd.DataFrame({"Feature": FEATURES, "Logistic coefficient": raw_model.coef_[0]}) \
        .to_csv(out / "logistic_coefficients.csv", index=False)
    # Save anonymized aggregate decile table only; no customer data or PD rows.
    deciles = pd.qcut(pd.Series(pd_cal), q=10, labels=False, duplicates="drop")
    decile_frame = pd.DataFrame({"decile": deciles, "pd": pd_cal, "default": test[TARGET].to_numpy(),
                                 "exposure": test["AMT_CREDIT"].to_numpy(),
                                 "EL": pd_cal * LGD_ASSUMED * test["AMT_CREDIT"].to_numpy()})
    decile_summary = decile_frame.groupby("decile").agg(
        clients=("pd", "size"), mean_predicted_pd=("pd", "mean"),
        observed_default_rate=("default", "mean"),
        exposure_proxy=("exposure", "sum"), expected_loss_proxy=("EL", "sum"))
    decile_summary.to_csv(out / "pd_deciles.csv")
    save_plots(test[TARGET].to_numpy(), pd_raw, pd_cal,
               pd.Series(encoder.iv_, name="IV"), out / "figures")
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print("DATA:", len(df), "clients; train/cal/test:", len(train), len(cal), len(test))
    print("DEFAULT RATE: {:.2%}".format(df[TARGET].mean()))
    print("IV:\n", pd.Series(encoder.iv_).sort_values(ascending=False))
    print("CONFUSION MATRIX (weighted model, threshold = 0.5):\n",
          confusion_matrix(test[TARGET], predicted_balanced))
    for key in ("auc_calibrated", "average_precision_calibrated", "brier_raw",
                "brier_calibrated", "brier_constant_baseline", "default_rate",
                "mean_pd_raw", "mean_pd_calibrated", "expected_loss_proxy_total",
                "ead_proxy_total", "expected_loss_to_ead"):
        print(f"{key}: {metrics[key]:,.6f}")
    print("Saved aggregate outputs:", out.resolve())
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, help="Home Credit application CSV")
    parser.add_argument("--output", default="results")
    args = parser.parse_args()
    run_analysis(args.data, args.output)

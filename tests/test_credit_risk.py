"""Fast, offline tests independent from restricted Home Credit data."""
import numpy as np
import pandas as pd
import pytest

from credit_risk import (TrainOnlyWoE, FEATURES, ID, TARGET, load_data, split_data,
                         fit_models, calculate_metrics)


def toy_data(n=600, seed=42):
    rng = np.random.default_rng(seed)
    x = pd.DataFrame({f: rng.normal(0, 1, n) for f in FEATURES})
    log_odds = -2 + 1.4 * x["EXT_SOURCE_1"] + 0.7 * x["AMT_CREDIT"]
    p = 1 / (1 + np.exp(-log_odds))
    y = rng.binomial(1, p)
    x[TARGET] = y
    x[ID] = np.arange(n) + 1000
    x["AMT_CREDIT"] = np.abs(x["AMT_CREDIT"]) * 12000 + 100
    x.loc[::3, "EXT_SOURCE_2"] = np.nan
    return x


def test_target_and_duplicate_validation(tmp_path):
    df = toy_data()
    f = tmp_path / "data.csv"
    df.to_csv(f, index=False)
    loaded = load_data(f)
    assert loaded[TARGET].isin([0, 1]).all()
    df.loc[1, ID] = df.loc[0, ID]
    df.to_csv(f, index=False)
    with pytest.raises(ValueError, match="unique"):
        load_data(f)


def test_split_is_stratified_and_nonoverlapping():
    train, cal, test = split_data(toy_data(6000))
    assert len(train) == 3600 and len(cal) == 1200 and len(test) == 1200
    assert not set(train[ID]) & set(cal[ID])
    assert not set(train[ID]) & set(test[ID])
    assert not set(cal[ID]) & set(test[ID])
    assert abs(train[TARGET].mean() - test[TARGET].mean()) < 0.02


def test_woe_handles_missing_unknown_extreme_values():
    df = toy_data()
    train = df.iloc[:450].copy()
    test = df.iloc[450:].copy()
    enc = TrainOnlyWoE(max_bins=4).fit(train[FEATURES], train[TARGET])
    woe = enc.transform(test[FEATURES])
    assert np.isfinite(woe.to_numpy()).all()
    changed = test[FEATURES].copy()
    changed.loc[changed.index[0], "AMT_INCOME_TOTAL"] = 1e12
    changed.loc[changed.index[1], "EXT_SOURCE_3"] = np.nan
    assert np.isfinite(enc.transform(changed).to_numpy()).all()
    assert all(float(v) >= 0 for v in enc.iv_.values())


def test_woe_fit_requires_aligned_indices():
    df = toy_data()
    shifted = df[TARGET].reset_index(drop=True)
    shifted.index = range(1, len(df) + 1)
    with pytest.raises(ValueError, match="indices"):
        TrainOnlyWoE().fit(df[FEATURES], shifted)


def test_test_labels_cannot_change_fitted_woe():
    df = toy_data()
    train = df.iloc[:450].copy()
    other = df.iloc[450:].copy()
    enc = TrainOnlyWoE().fit(train[FEATURES], train[TARGET])
    old = enc.transform(other[FEATURES]).to_numpy()
    other[TARGET] = 1 - other[TARGET]
    new = enc.transform(other[FEATURES]).to_numpy()
    np.testing.assert_allclose(old, new)


def test_probability_calibration_finite():
    df = toy_data(900)
    train, cal, test = split_data(df)
    enc, model, calibrated = fit_models(train, cal)
    x = enc.transform(test[FEATURES])
    raw = model.predict_proba(x)[:, 1]
    scores = calibrated.predict_proba(x)[:, 1]
    assert np.isfinite(scores).all()
    assert np.all((scores >= 0) & (scores <= 1))
    metric = calculate_metrics(test, raw, scores, model.predict(x), train[TARGET].mean())
    assert metric["expected_loss_proxy_total"] >= 0
    assert 0 <= metric["expected_loss_to_ead"] <= 0.45
    assert len(metric["balanced_threshold_confusion_matrix"]) == 2


def test_expected_loss_formula_exact():
    test = pd.DataFrame({TARGET: [0, 1, 0, 1], "AMT_CREDIT": [100, 200, 300, 400]})
    pd_raw = np.array([0.1, 0.2, 0.3, 0.4])
    pd_cal = np.array([0.1, 0.2, 0.3, 0.4])
    metrics = calculate_metrics(test, pd_raw, pd_cal, np.array([0, 0, 1, 1]), 0.5)
    assert metrics["expected_loss_proxy_total"] == pytest.approx(0.45 * (10 + 40 + 90 + 160))
    assert metrics["expected_loss_to_ead"] == pytest.approx(0.45 * 300 / 1000)

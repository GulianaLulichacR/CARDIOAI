from __future__ import annotations

import numpy as np

from backend.app.ml_engine import (
    calibration_intercept,
    expected_calibration_error,
    full_metrics,
    hosmer_lemeshow,
)
from backend.app.public_datasets import adapt_framingham, load_framingham, resolve_features


def test_framingham_contract():
    _, raw = load_framingham()
    assert raw.shape == (4240, 16)
    adapted = adapt_framingham(raw)
    assert len(adapted) == 4240
    assert adapted["event"].sum() == 644
    assert abs(adapted["event"].mean() - 0.1519) < 0.001
    features = resolve_features(adapted)
    assert "age" in features and "systolic_bp" in features and "sex" in features
    # Variables ausentes no deben aparecer como disponibles.
    for missing in ("pm25", "green_space", "income_level", "hdl", "family_history"):
        assert missing not in features
    # Una sola región: no se inventa geografía.
    assert adapted["region"].nunique() == 1


def test_split_does_not_claim_temporal_validation():
    from backend.app.ml_engine import split_dataset

    _, raw = load_framingham()
    adapted = adapt_framingham(raw)
    split = split_dataset(adapted)
    assert split.split_type == "random"
    assert split.cutoff_year is None
    assert len(split.geographic_test) == 0
    assert len(split.train) + len(split.temporal_test) == len(adapted)
    assert "no se afirma validación temporal" in split.design_note


def test_calibration_metrics_are_computable():
    rng = np.random.default_rng(7)
    y = rng.integers(0, 2, 600)
    p = np.clip(0.15 + 0.5 * y + rng.normal(0, 0.2, 600), 0.001, 0.999)
    assert calibration_intercept(y, p) is not None
    assert expected_calibration_error(y, p) is not None
    result = hosmer_lemeshow(y, p)
    assert result is not None and 0.0 <= result["p_value"] <= 1.0
    metrics = full_metrics(y, p, 0.3)
    for key in ("roc_auc", "pr_auc", "brier", "calibration", "dca", "auc_ci"):
        assert key in metrics
    assert metrics["calibration_intercept"] is not None


def test_records_frame_coerces_additional_features():
    from backend.app.ml_engine import _records_frame

    frame = _records_frame([{"age": "55", "cigs_per_day": "10", "bp_medication": "1"}])
    assert frame.loc[0, "age"] == 55
    assert frame.loc[0, "cigs_per_day"] == 10
    assert frame.loc[0, "bp_medication"] is True or frame.loc[0, "bp_medication"] == 1

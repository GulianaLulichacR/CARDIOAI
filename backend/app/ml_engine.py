"""Motor de Machine Learning de CARDIOAI.

Implementa cinco enfoques experimentales reales sobre datos tabulares y
longitudinales:

1. Random Forest
2. XGBoost
3. Cox Proportional Hazards
4. XGBoost-Cox (híbrido)
5. LSTM-Attention-Cox (híbrido profundo)

Los modelos se entrenan con el dataset indicado (por defecto, el dataset
SINTÉTICO de demostración). Ninguna métrica debe interpretarse como evidencia
clínica: los datos de demostración no son observaciones reales.

Estrategia de validación:
- El conjunto de prueba temporal (cohortes más recientes) y el geográfico
  (región retenida) NO participan en optimización ni en selección de umbral.
- Optuna y el umbral se calculan en un subconjunto interno de entrenamiento.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import time
import traceback
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedGroupKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sqlalchemy import func
from sqlalchemy.orm import Session

from .config import MODEL_DIR, settings
from .demo_data import (
    ADDITIONAL_BOOLEAN,
    ADDITIONAL_NUMERIC,
    BOOLEAN_FEATURES,
    CATEGORICAL_FEATURES,
    FEATURE_LABELS,
    FEATURES,
    NUMERIC_FEATURES,
    REGION_INFO,
    VISITS,
    VISIT_FEATURES,
    ensure_demo_dataset,
    quality_report,
)
from .models import Alert, Dataset, ModelRun, utcnow
from .public_datasets import resolve_features

# Entradas realmente utilizables: esquema base + variables adicionales que
# algunos datasets públicos aportan (p. ej., cigarrillos/día, medicación).
NUMERIC_INPUTS = NUMERIC_FEATURES + ADDITIONAL_NUMERIC
BOOLEAN_INPUTS = BOOLEAN_FEATURES + ADDITIONAL_BOOLEAN

try:
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    OPTUNA_AVAILABLE = True
except Exception:  # pragma: no cover
    OPTUNA_AVAILABLE = False

try:
    import torch
    import torch.nn as nn

    TORCH_AVAILABLE = True
except Exception:  # pragma: no cover
    TORCH_AVAILABLE = False


ALGORITHMS: list[dict[str, str]] = [
    {
        "key": "random_forest",
        "name": "Random Forest",
        "family": "tree",
        "description": "Clasificador de bosques aleatorios para evento a 10 años.",
    },
    {
        "key": "xgboost",
        "name": "XGBoost",
        "family": "tree",
        "description": "Gradient boosting extremo para evento a 10 años.",
    },
    {
        "key": "cox_ph",
        "name": "Cox Proportional Hazards",
        "family": "survival",
        "description": "Modelo de riesgos proporcionales con línea base de Breslow.",
    },
    {
        "key": "xgboost_cox",
        "name": "XGBoost-Cox (híbrido)",
        "family": "hybrid",
        "description": "XGBoost estima el predictor lineal y Cox calibra la supervivencia.",
    },
    {
        "key": "lstm_attention_cox",
        "name": "LSTM-Attention-Cox (híbrido)",
        "family": "deep",
        "description": "Red LSTM con atención temporal y pérdida parcial de Cox.",
    },
]
ALGORITHM_MAP = {a["key"]: a for a in ALGORITHMS}
SEED = 42


@dataclass
class SplitBundle:
    train: pd.DataFrame
    temporal_test: pd.DataFrame
    geographic_test: pd.DataFrame
    geo_holdout: str | None
    cutoff_year: int | None
    split_type: str
    design_note: str


# ---------------------------------------------------------------------------
# Utilidades de datos
# ---------------------------------------------------------------------------


def horizon_labels(df: pd.DataFrame, horizon: int) -> tuple[np.ndarray, np.ndarray]:
    """Devuelve (máscara_válida, etiqueta) para evento dentro del horizonte.

    Se excluyen los individuos censurados antes del horizonte, porque su
    estado final es desconocido.
    """
    event = df["event"].astype(bool).to_numpy()
    followup = pd.to_numeric(df["followup_years"], errors="coerce").fillna(0).to_numpy()
    positive = event & (followup <= horizon)
    negative = (~event & (followup >= horizon)) | (event & (followup > horizon))
    valid = positive | negative
    return valid, positive.astype(int)


def split_dataset(df: pd.DataFrame, seed: int = SEED) -> SplitBundle:
    """Partición honesta y explícita.

    - Temporal solo si existe un año de cohorte informativo.
    - Geográfica solo si hay varias regiones y la región retenida se excluye
      por completo del entrenamiento (sin fuga).
    - En caso contrario, holdout aleatorio estratificado, declarado como tal.
    """
    df = df.dropna(subset=["followup_years", "event"]).copy()
    df["event"] = df["event"].astype(bool)
    if "cohort_year" in df.columns:
        df["cohort_year"] = pd.to_numeric(df["cohort_year"], errors="coerce")
    else:
        df["cohort_year"] = np.nan

    n = len(df)
    if n < 50:
        raise ValueError("Se requieren al menos 50 registros para particionar.")

    # 1) Retención geográfica real: se aparta antes de cualquier entrenamiento.
    geo_holdout: str | None = None
    geographic_test = df.iloc[0:0].copy()
    model_df = df
    if "region" in df.columns:
        counts = df["region"].astype(str).value_counts()
        eligible = [
            str(r)
            for r, c in counts.items()
            if len(counts) >= 2 and 40 <= int(c) <= int(0.25 * n)
        ]
        if eligible:
            candidate = sorted(eligible, key=lambda r: (-int(counts[r]), r))[0]
            candidate_test = df[df["region"].astype(str) == candidate]
            candidate_train = df[df["region"].astype(str) != candidate]
            if (
                len(candidate_test) >= 40
                and len(candidate_train) >= 200
                and candidate_test["event"].nunique() == 2
            ):
                geo_holdout = candidate
                geographic_test = candidate_test.copy()
                model_df = candidate_train

    # 2) Temporal solo si el año de cohorte existe, varía y deja grupos suficientes.
    temporal_ok = False
    cutoff: int | None = None
    years = pd.to_numeric(model_df.get("cohort_year"), errors="coerce")
    if years.notna().mean() >= 0.95 and years.nunique(dropna=True) >= 3:
        candidate_cutoff = int(years.quantile(0.8))
        candidate_test = model_df[years >= candidate_cutoff]
        candidate_train = model_df[years < candidate_cutoff]
        if (
            len(candidate_test) >= 60
            and len(candidate_train) >= 300
            and candidate_test["event"].nunique() == 2
            and candidate_train["event"].nunique() == 2
        ):
            temporal_ok = True
            cutoff = candidate_cutoff
            test_df = candidate_test
            train_df = candidate_train

    if temporal_ok:
        split_kind = "temporal"
    else:
        if model_df["event"].nunique() < 2 or len(model_df) < 100:
            raise ValueError(
                "No hay suficientes eventos ni registros para una partición de evaluación."
            )
        train_df, test_df = train_test_split(
            model_df,
            test_size=0.2,
            random_state=seed,
            stratify=model_df["event"],
        )
        split_kind = "random"

    if geo_holdout:
        split_kind = f"{split_kind}+geographic"

    if split_kind.startswith("temporal"):
        note = (
            "Holdout temporal: las cohortes más recientes se reservan para prueba; "
            "Optuna y el umbral usan solo entrenamiento/validación interna."
        )
    else:
        note = (
            "Holdout aleatorio estratificado 80/20: el dataset no contiene un año de "
            "cohorte informativo, por lo que no se afirma validación temporal."
        )
    if geo_holdout:
        note += (
            f" Retención geográfica real: la región '{geo_holdout}' se excluyó de "
            "entrenamiento y prueba; se evalúa por separado."
        )
    else:
        note += " No se aplicó retención geográfica (una sola región o tamaño insuficiente)."

    return SplitBundle(
        train=train_df,
        temporal_test=test_df,
        geographic_test=geographic_test,
        geo_holdout=geo_holdout,
        cutoff_year=cutoff,
        split_type=split_kind,
        design_note=note,
    )



def load_training_frame(db: Session, dataset_id: int | None) -> tuple[pd.DataFrame, bool, Dataset | None]:
    if dataset_id is None:
        path, df = ensure_demo_dataset()
        dataset = db.query(Dataset).filter(Dataset.name == "Demostración sintética CARDIOAI").one_or_none()
        if dataset is None:
            dataset = Dataset(
                name="Demostración sintética CARDIOAI",
                source="Generado por CARDIOAI (semilla fija)",
                license="Uso exclusivo de demostración",
                description="Cohorte sintética con fines de demostración técnica. No son datos reales.",
                version="1.0",
                is_demo=True,
                status="ready",
                rows_count=len(df),
                columns_count=df.shape[1],
                file_path=str(path),
                quality=quality_report(df),
            )
            db.add(dataset)
            db.flush()
        return df, True, dataset

    dataset = db.get(Dataset, dataset_id)
    if dataset is None or not dataset.file_path:
        raise ValueError("Dataset no encontrado o sin archivo asociado.")
    df = pd.read_csv(dataset.file_path)
    missing = [c for c in ["event", "followup_years"] if c not in df.columns]
    if missing:
        raise ValueError(f"El dataset no contiene las columnas obligatorias: {', '.join(missing)}")
    for col in FEATURES:
        if col not in df.columns:
            df[col] = np.nan
    return df, bool(dataset.is_demo), dataset


def survival_prep_frame(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in BOOLEAN_INPUTS:
        if col in out.columns:
            out[col] = out[col].astype("boolean").fillna(False).astype(int)
    return out


# ---------------------------------------------------------------------------
# Preprocesador para modelos de supervivencia (serializable)
# ---------------------------------------------------------------------------


class SurvivalPreprocessor:
    def __init__(self, features: list[str] | None = None) -> None:
        self.features = list(features or FEATURES)
        self.numeric_cols = [
            f for f in self.features if f in NUMERIC_INPUTS or f in BOOLEAN_INPUTS
        ]
        self.categorical_cols = [f for f in self.features if f in CATEGORICAL_FEATURES]
        self.medians: dict[str, float] = {}
        self.means: dict[str, float] = {}
        self.stds: dict[str, float] = {}
        self.dummy_columns: list[str] = []
        self.columns: list[str] = []
        self.feature_map: dict[str, str] = {}

    def fit(self, df: pd.DataFrame) -> "SurvivalPreprocessor":
        work = survival_prep_frame(df)
        for col in self.numeric_cols:
            values = pd.to_numeric(work[col], errors="coerce")
            median = float(values.median()) if values.notna().any() else 0.0
            filled = values.fillna(median)
            self.medians[col] = median
            self.means[col] = float(filled.mean())
            std = float(filled.std(ddof=0))
            self.stds[col] = std if std > 1e-9 else 1.0
        self.dummy_columns = []
        if self.categorical_cols:
            dummies = pd.get_dummies(
                work[self.categorical_cols].astype(str), prefix=self.categorical_cols
            )
            self.dummy_columns = list(dummies.columns)
        self.columns = self.numeric_cols + self.dummy_columns
        self.feature_map = {c: c for c in self.numeric_cols}
        for dummy in self.dummy_columns:
            base = dummy.split("_", 1)[0]
            self.feature_map[dummy] = base
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        work = survival_prep_frame(df)
        out = pd.DataFrame(index=work.index)
        for col in self.numeric_cols:
            values = pd.to_numeric(work[col], errors="coerce").fillna(self.medians[col])
            out[col] = (values - self.means[col]) / self.stds[col]
        if self.dummy_columns:
            dummies = pd.get_dummies(
                work[self.categorical_cols].astype(str), prefix=self.categorical_cols
            )
            dummies = dummies.reindex(columns=self.dummy_columns, fill_value=0)
            out = pd.concat([out, dummies.astype(float)], axis=1)
        return out.reindex(columns=self.columns, fill_value=0.0)


    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.fit(df).transform(df)


# ---------------------------------------------------------------------------
# Métricas
# ---------------------------------------------------------------------------


def _safe_auc(y: np.ndarray, p: np.ndarray) -> float | None:
    if len(np.unique(y)) < 2:
        return None
    return float(roc_auc_score(y, p))


def _safe_pr_auc(y: np.ndarray, p: np.ndarray) -> float | None:
    if len(np.unique(y)) < 2:
        return None
    return float(average_precision_score(y, p))


def _curve_points(x: np.ndarray, y: np.ndarray, max_points: int = 60) -> tuple[list[float], list[float]]:
    if len(x) > max_points:
        idx = np.linspace(0, len(x) - 1, max_points).astype(int)
        x, y = x[idx], y[idx]
    return [round(float(v), 4) for v in x], [round(float(v), 4) for v in y]


def calibration_bins(y: np.ndarray, p: np.ndarray, bins: int = 10) -> list[dict[str, float]]:
    if len(y) == 0:
        return []
    frame = pd.DataFrame({"y": y, "p": p})
    try:
        frame["bin"] = pd.qcut(frame["p"].rank(method="first"), q=min(bins, max(2, len(frame))), labels=False)
    except ValueError:
        frame["bin"] = 0
    grouped = frame.groupby("bin").agg(mean_pred=("p", "mean"), frac_pos=("y", "mean"), n=("y", "size"))
    return [
        {"mean_pred": round(float(r.mean_pred), 4), "frac_pos": round(float(r.frac_pos), 4), "n": int(r.n)}
        for r in grouped.itertuples()
    ]


def calibration_slope(y: np.ndarray, p: np.ndarray) -> float | None:
    if len(np.unique(y)) < 2:
        return None
    logit = np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6))).reshape(-1, 1)
    try:
        model = LogisticRegression(solver="lbfgs", max_iter=1000)
        model.fit(logit, y)
        return round(float(model.coef_[0][0]), 3)
    except Exception:
        return None


def calibration_intercept(y: np.ndarray, p: np.ndarray) -> float | None:
    """Calibración en grande: intercepto de logit(Y) ~ 1 + logit(p)."""
    if len(np.unique(y)) < 2:
        return None
    logit = np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6))).reshape(-1, 1)
    try:
        model = LogisticRegression(solver="lbfgs", max_iter=1000)
        model.fit(logit, y)
        return round(float(model.intercept_[0]), 3)
    except Exception:
        return None


def expected_calibration_error(y: np.ndarray, p: np.ndarray, bins: int = 10) -> float | None:
    groups = calibration_bins(y, p, bins)
    total = sum(item["n"] for item in groups)
    if not total:
        return None
    ece = sum(
        item["n"] / total * abs(item["frac_pos"] - item["mean_pred"]) for item in groups
    )
    return round(float(ece), 4)


def hosmer_lemeshow(y: np.ndarray, p: np.ndarray, groups: int = 10) -> dict[str, Any] | None:
    """Prueba de Hosmer-Lemeshow sobre deciles de probabilidad predicha."""
    if len(np.unique(y)) < 2 or len(y) < groups * 2:
        return None
    frame = pd.DataFrame({"y": y.astype(int), "p": np.clip(p, 1e-6, 1 - 1e-6)})
    frame["bin"] = pd.qcut(frame["p"].rank(method="first"), q=groups, labels=False)
    chi2 = 0.0
    used = 0
    for _, grp in frame.groupby("bin"):
        n_g = len(grp)
        o1 = float(grp["y"].sum())
        e1 = float(grp["p"].sum())
        o0 = n_g - o1
        e0 = n_g - e1
        if e1 > 0:
            chi2 += (o1 - e1) ** 2 / e1
        if e0 > 0:
            chi2 += (o0 - e0) ** 2 / e0
        used += 1
    if used < 3:
        return None
    from scipy.stats import chi2 as chi2_dist

    dof = used - 2
    return {
        "chi2": round(float(chi2), 3),
        "df": int(dof),
        "p_value": round(float(chi2_dist.sf(chi2, dof)), 4),
        "groups": int(used),
    }


def dca_points(y: np.ndarray, p: np.ndarray, thresholds: list[float] | None = None) -> list[dict[str, float]]:
    if len(y) == 0:
        return []
    n = len(y)
    prevalence = float(np.mean(y))
    thresholds = thresholds or [round(t, 3) for t in np.arange(0.02, 0.52, 0.02)]
    points = []
    for pt in thresholds:
        if pt <= 0 or pt >= 1:
            continue
        pred = p >= pt
        tp = int(np.sum(pred & (y == 1)))
        fp = int(np.sum(pred & (y == 0)))
        nb_model = tp / n - fp / n * (pt / (1 - pt))
        nb_all = prevalence - (1 - prevalence) * (pt / (1 - pt))
        points.append(
            {
                "threshold": round(float(pt), 3),
                "model": round(float(nb_model), 4),
                "treat_all": round(float(nb_all), 4),
                "treat_none": 0.0,
            }
        )
    return points


def classification_metrics(y: np.ndarray, p: np.ndarray, threshold: float) -> dict[str, Any]:
    y = np.asarray(y).astype(int)
    p = np.asarray(p, dtype=float)
    pred = (p >= threshold).astype(int)
    cm = confusion_matrix(y, pred, labels=[0, 1])
    tn, fp, fn, tp = [int(v) for v in cm.ravel()]
    sens = tp / (tp + fn) if (tp + fn) else 0.0
    spec = tn / (tn + fp) if (tn + fp) else 0.0
    return {
        "threshold": round(float(threshold), 4),
        "sensitivity": round(sens, 4),
        "specificity": round(spec, 4),
        "f1": round(float(f1_score(y, pred, zero_division=0)), 4),
        "mcc": round(float(matthews_corrcoef(y, pred)) if len(np.unique(y)) > 1 else 0.0, 4),
        "accuracy": round(float(accuracy_score(y, pred)), 4),
        "roc_auc": None if (auc := _safe_auc(y, p)) is None else round(auc, 4),
        "pr_auc": None if (prauc := _safe_pr_auc(y, p)) is None else round(prauc, 4),
        "brier": round(float(brier_score_loss(y, np.clip(p, 0, 1))), 4),
        "confusion": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
        "n": int(len(y)),
        "positives": int(y.sum()),
        "prevalence": round(float(y.mean()), 4),
    }


def bootstrap_auc_ci(y: np.ndarray, p: np.ndarray, n_boot: int = 200, seed: int = SEED) -> list[float] | None:
    if len(np.unique(y)) < 2:
        return None
    rng = np.random.default_rng(seed)
    values = []
    n = len(y)
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if len(np.unique(y[idx])) < 2:
            continue
        values.append(roc_auc_score(y[idx], p[idx]))
    if not values:
        return None
    return [round(float(np.percentile(values, 2.5)), 4), round(float(np.percentile(values, 97.5)), 4)]


def full_metrics(y: np.ndarray, p: np.ndarray, threshold: float) -> dict[str, Any]:
    metrics = classification_metrics(y, p, threshold)
    if len(y) and len(np.unique(y)) > 1:
        fpr, tpr, _ = roc_curve(y, p)
        precision, recall, _ = precision_recall_curve(y, p)
        rfp, rtp = _curve_points(fpr, tpr)
        rprec, rrec = _curve_points(recall, precision)
        metrics["roc_curve"] = {"fpr": rfp, "tpr": rtp}
        metrics["pr_curve"] = {"recall": rrec, "precision": rprec}
    else:
        metrics["roc_curve"] = {"fpr": [], "tpr": []}
        metrics["pr_curve"] = {"recall": [], "precision": []}
    metrics["calibration"] = calibration_bins(y, p)
    metrics["calibration_slope"] = calibration_slope(y, p)
    metrics["calibration_intercept"] = calibration_intercept(y, p)
    metrics["calibration_ece"] = expected_calibration_error(y, p)
    metrics["hosmer_lemeshow"] = hosmer_lemeshow(y, p)
    metrics["dca"] = dca_points(y, p)
    metrics["auc_ci"] = bootstrap_auc_ci(y, p)
    return metrics


def best_threshold(y: np.ndarray, p: np.ndarray) -> float:
    if len(y) == 0 or len(np.unique(y)) < 2:
        return 0.5
    candidates = np.unique(np.clip(p, 0.01, 0.99))
    if len(candidates) > 200:
        candidates = np.quantile(candidates, np.linspace(0, 1, 200))
    best_f1, best_t = -1.0, 0.5
    for t in candidates:
        score = f1_score(y, (p >= t).astype(int), zero_division=0)
        if score > best_f1:
            best_f1, best_t = float(score), float(t)
    return round(best_t, 4)


def risk_level(probability: float) -> str:
    if probability < settings.risk_low_max:
        return "bajo"
    if probability < settings.risk_high_min:
        return "moderado"
    return "alto"


def summary_importance(names: list[str], values: np.ndarray, feature_map: dict[str, str], top: int = 15) -> list[dict[str, Any]]:
    totals: dict[str, float] = {}
    for name, value in zip(names, values):
        base = feature_map.get(name, name)
        totals[base] = totals.get(base, 0.0) + abs(float(value))
    total = sum(totals.values()) or 1.0
    ranked = sorted(totals.items(), key=lambda kv: kv[1], reverse=True)[:top]
    return [
        {
            "feature": feature,
            "label": FEATURE_LABELS.get(feature, feature),
            "importance": round(value / total, 4),
        }
        for feature, value in ranked
    ]


# ---------------------------------------------------------------------------
# Preprocesamiento para modelos de clasificación
# ---------------------------------------------------------------------------


def build_tree_pipeline(model, features: list[str] | None = None) -> Pipeline:
    features = features or list(FEATURES)
    numeric_cols = [f for f in features if f in NUMERIC_INPUTS or f in BOOLEAN_INPUTS]
    categorical_cols = [f for f in features if f in CATEGORICAL_FEATURES]
    transformers = []
    if numeric_cols:
        transformers.append(
            (
                "num",
                Pipeline(
                    [("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]
                ),
                numeric_cols,
            )
        )
    if categorical_cols:
        transformers.append(
            (
                "cat",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                    ]
                ),
                categorical_cols,
            )
        )
    pre = ColumnTransformer(transformers, remainder="drop", verbose_feature_names_out=False)
    return Pipeline([("pre", pre), ("clf", model)])


def _tree_feature_map(names: list[str], features: list[str] | None = None) -> dict[str, str]:
    categorical = [f for f in (features or FEATURES) if f in CATEGORICAL_FEATURES]
    mapping: dict[str, str] = {}
    for name in names:
        base = name.split("__", 1)[-1]
        found = None
        for cat in categorical:
            if base.startswith(f"{cat}_"):
                found = cat
                break
        mapping[name] = found or base
    return mapping



def _tree_importance(pipe: Pipeline, feature_map: dict[str, str]) -> list[dict[str, Any]]:
    pre = pipe.named_steps["pre"]
    clf = pipe.named_steps["clf"]
    names = list(pre.get_feature_names_out())
    values = getattr(clf, "feature_importances_", None)
    if values is None:
        return []
    return summary_importance(names, np.asarray(values), feature_map)


# ---------------------------------------------------------------------------
# Modelos de supervivencia
# ---------------------------------------------------------------------------


def _fit_cox(design: pd.DataFrame, duration: np.ndarray, event: np.ndarray, penalizer: float = 0.08):
    from lifelines import CoxPHFitter

    data = design.copy()
    data["__duration"] = duration
    data["__event"] = event.astype(int)
    cph = CoxPHFitter(penalizer=penalizer)
    cph.fit(data, duration_col="__duration", event_col="__event")
    baseline = cph.baseline_cumulative_hazard_
    times = baseline.index.to_numpy(dtype=float)
    hazard = baseline.iloc[:, 0].to_numpy(dtype=float)
    return cph, times, hazard


def _h0_at(times: np.ndarray, hazard: np.ndarray, t: float) -> float:
    if len(times) == 0:
        return 0.0
    return float(np.interp(t, times, hazard))


def _cox_risk(cph, times: np.ndarray, hazard: np.ndarray, design: pd.DataFrame, horizon: int) -> np.ndarray:
    lp = cph.predict_log_partial_hazard(design).to_numpy(dtype=float)
    h0 = _h0_at(times, hazard, horizon)
    return 1.0 - np.exp(-h0 * np.exp(lp))


def _breslow_h0(lp: np.ndarray, times: np.ndarray, events: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    order = np.argsort(times)
    times_sorted = times[order]
    lp_sorted = lp[order]
    events_sorted = events[order].astype(bool)
    unique_times = np.unique(times_sorted[events_sorted])
    h0 = []
    cumulative = 0.0
    for t in unique_times:
        risk_set = times_sorted >= t
        denom = float(np.sum(np.exp(lp_sorted[risk_set])))
        d = float(np.sum((times_sorted == t) & events_sorted))
        if denom > 1e-12:
            cumulative += d / denom
        h0.append(cumulative)
    return np.asarray(unique_times, dtype=float), np.asarray(h0, dtype=float)


if TORCH_AVAILABLE:

    class LSTMAttentionCox(nn.Module):
        def __init__(self, input_size: int, hidden: int = 24) -> None:
            super().__init__()
            self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden, batch_first=True)
            self.attention = nn.Linear(hidden, 1)
            self.head = nn.Linear(hidden, 1)
            self.dropout = nn.Dropout(0.15)

        def forward(self, x, mask):
            out, _ = self.lstm(x)
            scores = self.attention(out).squeeze(-1)
            scores = scores.masked_fill(~mask, -1e9)
            weights = torch.softmax(scores, dim=1)
            context = torch.sum(out * weights.unsqueeze(-1), dim=1)
            log_hazard = self.head(self.dropout(context)).squeeze(-1)
            return log_hazard, weights


def _cox_partial_loss(log_hazard, times, events):
    order = torch.argsort(times)
    log_hazard = log_hazard[order]
    events = events[order].bool()
    log_denom = torch.logcumsumexp(torch.flip(log_hazard, [0]), 0)
    log_denom = torch.flip(log_denom, [0])
    loss = -(log_hazard - log_denom)[events]
    if loss.numel() == 0:
        return log_hazard.sum() * 0.0
    return loss.mean()


def _sequence_tensor(df: pd.DataFrame, mean: np.ndarray | None = None, std: np.ndarray | None = None):
    n = len(df)
    seq = np.zeros((n, VISITS, len(VISIT_FEATURES)), dtype=np.float32)
    mask = np.zeros((n, VISITS), dtype=bool)
    raw = np.zeros((n, VISITS, len(VISIT_FEATURES)), dtype=np.float32)
    for j, feature in enumerate(VISIT_FEATURES):
        aliases = {
            "sbp": ["systolic_bp", "visit_sbp"],
            "chol": ["cholesterol_total", "visit_chol"],
            "glucose": ["glucose", "visit_glucose"],
            "bmi": ["bmi", "visit_bmi"],
            "pm25": ["pm25", "visit_pm25"],
            "activity": ["physical_activity_min", "visit_activity"],
        }[feature]
        for v in range(VISITS):
            col = f"visit_{v}_{feature}"
            if col in df.columns:
                values = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float)
            else:
                base_col = next((a for a in aliases if a in df.columns), None)
                values = (
                    pd.to_numeric(df[base_col], errors="coerce").to_numpy(dtype=float)
                    if base_col
                    else np.full(n, np.nan)
                )
            observed = ~np.isnan(values)
            mask[:, v] |= observed
            raw[:, v, j] = values
    # Imputa hacia atrás/adelante por visita y luego con la media global.
    for j in range(len(VISIT_FEATURES)):
        column = raw[:, :, j]
        column_mean = np.nanmean(column) if not np.all(np.isnan(column)) else 0.0
        column = np.where(np.isnan(column), column_mean, column)
        raw[:, :, j] = column
    if mean is None:
        mean = raw.reshape(-1, len(VISIT_FEATURES)).mean(axis=0)
    if std is None:
        std = raw.reshape(-1, len(VISIT_FEATURES)).std(axis=0)
    std = np.where(std < 1e-8, 1.0, std)
    seq = ((raw - mean) / std).astype(np.float32)
    mask[:, 0] = True
    return seq, mask, mean, std


def _lstm_predict(model, seq: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    with torch.no_grad():
        x = torch.tensor(seq, dtype=torch.float32)
        m = torch.tensor(mask, dtype=torch.bool)
        log_hazard, weights = model(x, m)
    return log_hazard.numpy().astype(float), weights.numpy().astype(float)


def _train_lstm(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    epochs: int = 90,
    patience: int = 12,
    seed: int = SEED,
) -> dict[str, Any]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    seq_train, mask_train, mean, std = _sequence_tensor(train_df)
    seq_val, mask_val, _, _ = _sequence_tensor(val_df, mean, std)
    times_train = torch.tensor(train_df["followup_years"].to_numpy(dtype=float), dtype=torch.float32)
    events_train = torch.tensor(train_df["event"].astype(int).to_numpy(), dtype=torch.float32)
    times_val = torch.tensor(val_df["followup_years"].to_numpy(dtype=float), dtype=torch.float32)
    events_val = torch.tensor(val_df["event"].astype(int).to_numpy(), dtype=torch.float32)

    model = LSTMAttentionCox(len(VISIT_FEATURES), hidden=24)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-4)
    best_state, best_loss, best_epoch = None, math.inf, 0
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        log_hazard, _ = model(torch.tensor(seq_train), torch.tensor(mask_train))
        loss = _cox_partial_loss(log_hazard, times_train, events_train)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        model.eval()
        with torch.no_grad():
            val_hazard, _ = model(torch.tensor(seq_val), torch.tensor(mask_val))
            val_loss = float(_cox_partial_loss(val_hazard, times_val, events_val))
        if val_loss < best_loss - 1e-5:
            best_loss, best_epoch = val_loss, epoch
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        elif epoch - best_epoch > patience:
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    log_hazard, weights = _lstm_predict(model, seq_train, mask_train)
    baseline_times, baseline_hazard = _breslow_h0(
        log_hazard, train_df["followup_years"].to_numpy(dtype=float), train_df["event"].to_numpy()
    )
    return {
        "model": model,
        "state_dict": best_state or model.state_dict(),
        "seq_mean": mean,
        "seq_std": std,
        "baseline_times": baseline_times,
        "baseline_hazard": baseline_hazard,
        "epochs": best_epoch + 1,
        "val_loss": best_loss,
    }


def _lstm_risk(bundle: dict[str, Any], df: pd.DataFrame, horizon: int) -> np.ndarray:
    model = bundle["model"]
    seq, mask, _, _ = _sequence_tensor(df, bundle["seq_mean"], bundle["seq_std"])
    log_hazard, _ = _lstm_predict(model, seq, mask)
    h0 = _h0_at(bundle["baseline_times"], bundle["baseline_hazard"], horizon)
    return 1.0 - np.exp(-h0 * np.exp(log_hazard))


# ---------------------------------------------------------------------------
# Entrenamiento de cada enfoque
# ---------------------------------------------------------------------------


def _tune_tree(
    algorithm: str,
    fit_df: pd.DataFrame,
    labels: np.ndarray,
    trials: int,
    seed: int,
    features: list[str] | None = None,
) -> dict[str, Any]:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import StratifiedKFold
    from xgboost import XGBClassifier

    groups = fit_df["region"].astype(str).to_numpy() if "region" in fit_df.columns else np.array(["all"] * len(fit_df))
    n_groups = len(np.unique(groups))
    if n_groups >= 2:
        n_splits = max(2, min(settings.ml_cv_folds, n_groups))
        cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    else:
        n_splits = max(2, min(settings.ml_cv_folds, max(2, len(fit_df) // 100)))
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    if not OPTUNA_AVAILABLE:
        return (
            {"n_estimators": 300, "max_depth": 10, "min_samples_leaf": 2, "max_features": "sqrt"}
            if algorithm == "random_forest"
            else {"n_estimators": 300, "max_depth": 4, "learning_rate": 0.05, "subsample": 0.9,
                  "colsample_bytree": 0.8, "min_child_weight": 1, "reg_lambda": 1.0}
        )

    def objective(trial: "optuna.Trial") -> float:
        if algorithm == "random_forest":
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 200, 450, step=50),
                "max_depth": trial.suggest_int("max_depth", 5, 22),
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 8),
                "max_features": trial.suggest_categorical("max_features", ["sqrt", 0.5]),
            }
            model = RandomForestClassifier(**params, random_state=seed, n_jobs=2)
        else:
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 200, 450, step=50),
                "max_depth": trial.suggest_int("max_depth", 3, 8),
                "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.2, log=True),
                "subsample": trial.suggest_float("subsample", 0.7, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
                "min_child_weight": trial.suggest_int("min_child_weight", 1, 6),
                "reg_lambda": trial.suggest_float("reg_lambda", 0.5, 5.0),
            }
            model = XGBClassifier(**params, random_state=seed, n_jobs=2, eval_metric="logloss",
                                  tree_method="hist")
        pipe = build_tree_pipeline(model, features)
        scores = []
        split_iter = (
            cv.split(fit_df, labels, groups)
            if n_groups >= 2
            else cv.split(fit_df, labels)
        )
        for train_idx, val_idx in split_iter:
            pipe.fit(fit_df.iloc[train_idx], labels[train_idx])
            probs = pipe.predict_proba(fit_df.iloc[val_idx])[:, 1]
            if len(np.unique(labels[val_idx])) > 1:
                scores.append(roc_auc_score(labels[val_idx], probs))
        return float(np.mean(scores)) if scores else 0.5

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=seed),
        pruner=optuna.pruners.MedianPruner(),
    )
    study.optimize(objective, n_trials=trials, timeout=180)
    return study.best_params


def _fit_survival_model(
    algorithm: str,
    train_df: pd.DataFrame,
    horizon: int,
    trials: int,
    seed: int,
    fixed_params: dict[str, Any] | None = None,
    features: list[str] | None = None,
) -> dict[str, Any]:
    prep = SurvivalPreprocessor(features).fit(train_df)
    design = prep.transform(train_df)
    duration = train_df["followup_years"].to_numpy(dtype=float)
    event = train_df["event"].to_numpy()

    if algorithm == "cox_ph":
        cph, times, hazard = _fit_cox(design, duration, event)
        return {"prep": prep, "cph": cph, "baseline_times": times, "baseline_hazard": hazard,
                "params": {"penalizer": 0.08}}

    if algorithm == "xgboost_cox":
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.model_selection import KFold
        from xgboost import XGBRegressor

        # 1) Predictor lineal de Cox fuera de bolsa para evitar fuga de información.
        oof = np.zeros(len(train_df))
        kf = KFold(n_splits=3, shuffle=True, random_state=seed)
        for tr_idx, va_idx in kf.split(design):
            cph_fold, _, _ = _fit_cox(design.iloc[tr_idx], duration[tr_idx], event[tr_idx])
            oof[va_idx] = cph_fold.predict_log_partial_hazard(design.iloc[va_idx]).to_numpy()
        # 2) XGBoost aprende la relación no lineal con el log-hazard.
        if fixed_params is not None:
            params = dict(fixed_params)
        elif OPTUNA_AVAILABLE and trials > 0:
            def objective(trial):
                params = {
                    "n_estimators": trial.suggest_int("n_estimators", 200, 400, step=100),
                    "max_depth": trial.suggest_int("max_depth", 2, 5),
                    "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.15, log=True),
                    "subsample": trial.suggest_float("subsample", 0.7, 1.0),
                    "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
                    "reg_lambda": trial.suggest_float("reg_lambda", 1.0, 6.0),
                }
                model = XGBRegressor(**params, random_state=seed, n_jobs=2, tree_method="hist")
                model.fit(design, oof)
                pred = model.predict(design)
                return float(np.corrcoef(pred, oof)[0, 1])

            study = optuna.create_study(direction="maximize",
                                        sampler=optuna.samplers.TPESampler(seed=seed))
            study.optimize(objective, n_trials=max(4, min(trials, 10)), timeout=120)
            params = study.best_params
        else:
            params = {"n_estimators": 300, "max_depth": 3, "learning_rate": 0.05,
                      "subsample": 0.9, "colsample_bytree": 0.8, "reg_lambda": 2.0}
        xgb = XGBRegressor(**params, random_state=seed, n_jobs=2, tree_method="hist")
        xgb.fit(design, oof)
        score = xgb.predict(design)
        score_mean = float(np.mean(score))
        score_std = float(np.std(score) or 1.0)
        score_design = pd.DataFrame({"xgb_score": (score - score_mean) / score_std})
        cox_uni, times, hazard = _fit_cox(
            score_design, duration, event, penalizer=0.01
        )
        return {"prep": prep, "xgb": xgb, "cox": cox_uni, "score_mean": score_mean,
                "score_std": score_std, "baseline_times": times, "baseline_hazard": hazard,
                "params": params}

    raise ValueError(f"Algoritmo de supervivencia no soportado: {algorithm}")


def _survival_risk(bundle: dict[str, Any], df: pd.DataFrame, horizon: int) -> np.ndarray:
    design = bundle["prep"].transform(df)
    if "cph" in bundle:
        return _cox_risk(bundle["cph"], bundle["baseline_times"], bundle["baseline_hazard"],
                         design, horizon)
    score = bundle["xgb"].predict(design)
    score = (score - bundle["score_mean"]) / bundle["score_std"]
    lp = bundle["cox"].predict_log_partial_hazard(pd.DataFrame({"xgb_score": score})).to_numpy()
    h0 = _h0_at(bundle["baseline_times"], bundle["baseline_hazard"], horizon)
    return 1.0 - np.exp(-h0 * np.exp(lp))


# ---------------------------------------------------------------------------
# Entrenamiento integral
# ---------------------------------------------------------------------------


def _subgroup_report(
    frame: pd.DataFrame,
    y: np.ndarray,
    p: np.ndarray,
    threshold: float,
) -> list[dict[str, Any]]:
    """Métricas de subgrupo sobre el holdout, sin usar los datos para ajustar."""
    results: list[dict[str, Any]] = []
    candidates: list[tuple[str, str, np.ndarray]] = []
    if "sex" in frame.columns:
        sex = frame["sex"].astype(str).to_numpy()
        for value in ("F", "M"):
            candidates.append(("sex", value, sex == value))
    if "age" in frame.columns:
        age = pd.to_numeric(frame["age"], errors="coerce").to_numpy()
        candidates.append(("age", "<60", age < 60))
        candidates.append(("age", ">=60", age >= 60))
    if "education" in frame.columns:
        education = frame["education"].astype(str).to_numpy()
        for value in sorted(set(education)):
            candidates.append(("education", value, education == value))
    for variable, value, mask in candidates:
        mask = np.asarray(mask, dtype=bool)
        if mask.sum() < 20:
            continue
        yy = y[mask]
        pp = p[mask]
        if len(np.unique(yy)) < 2:
            continue
        m = classification_metrics(yy, pp, threshold)
        results.append(
            {
                "variable": variable,
                "group": value,
                "n": int(mask.sum()),
                "events": int(yy.sum()),
                "roc_auc": m["roc_auc"],
                "pr_auc": m["pr_auc"],
                "brier": m["brier"],
                "calibration_slope": calibration_slope(yy, pp),
                "calibration_intercept": calibration_intercept(yy, pp),
            }
        )
    return results


def _save_artifact_atomic(artifact: dict[str, Any], path: Path) -> str:
    """Escribe el artefacto de forma atómica y verifica que sea legible."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    joblib.dump(artifact, tmp)
    loaded = joblib.load(tmp)
    if not isinstance(loaded, dict) or loaded.get("kind") != artifact.get("kind"):
        tmp.unlink(missing_ok=True)
        raise RuntimeError("El artefacto no superó la verificación de integridad.")
    os.replace(tmp, path)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fit_and_evaluate(
    db: Session,
    run: ModelRun,
    df: pd.DataFrame,
    trials: int,
    horizon: int,
    features: list[str] | None = None,
) -> None:
    features = features or resolve_features(df)
    if len(features) < 2:
        raise ValueError(
            "El dataset no contiene al menos dos predictores utilizables con cobertura suficiente."
        )
    split = split_dataset(df)
    train_df = split.train.reset_index(drop=True)
    temporal = split.temporal_test.reset_index(drop=True)
    geographic = split.geographic_test.reset_index(drop=True)

    fit_df, val_df = train_test_split(
        train_df, test_size=0.2, random_state=SEED, stratify=train_df["event"]
    )
    valid_fit, y_fit = horizon_labels(fit_df, horizon)
    valid_val, y_val = horizon_labels(val_df, horizon)

    kind = ALGORITHM_MAP[run.algorithm]["family"]
    artifact: dict[str, Any] = {
        "kind": run.algorithm,
        "algorithm": run.algorithm,
        "features": features,
        "horizon": horizon,
        "split_type": split.split_type,
        "risk_thresholds": {
            "low_max": settings.risk_low_max,
            "high_min": settings.risk_high_min,
            "basis": "Umbrales configurables; requieren validación clínica antes de uso asistencial.",
        },
    }
    cv_metrics: dict[str, Any] = {}
    selection_auc: float | None = None
    selection_strategy = "validación interna (20 % del entrenamiento)"

    if kind == "tree":
        params = _tune_tree(
            run.algorithm,
            fit_df[valid_fit].reset_index(drop=True),
            y_fit[valid_fit],
            trials,
            SEED,
            features,
        )
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import StratifiedKFold
        from xgboost import XGBClassifier

        def _new_model():
            if run.algorithm == "random_forest":
                return RandomForestClassifier(
                    **params, random_state=SEED, n_jobs=2
                )
            return XGBClassifier(
                **params, random_state=SEED, n_jobs=2, eval_metric="logloss", tree_method="hist"
            )

        pipe = build_tree_pipeline(_new_model(), features)
        pipe.fit(fit_df[valid_fit], y_fit[valid_fit])
        val_probs = pipe.predict_proba(val_df)[:, 1]
        if valid_val.sum() and len(np.unique(y_val[valid_val])) > 1:
            threshold = best_threshold(y_val[valid_val], val_probs[valid_val])
            selection_auc = _safe_auc(y_val[valid_val], val_probs[valid_val])
        else:
            threshold = 0.5
        # Reentrena con todo el entrenamiento y el umbral ya fijado.
        train_valid, y_train = horizon_labels(train_df, horizon)
        pipe.fit(train_df[train_valid], y_train[train_valid])
        names = list(pipe.named_steps["pre"].get_feature_names_out())
        feature_map = _tree_feature_map(names, features)
        artifact.update(
            {
                "pipeline": pipe,
                "threshold": threshold,
                "feature_map": feature_map,
                "params": params,
            }
        )
        predict_fn: Callable[[pd.DataFrame], np.ndarray] = lambda d: pipe.predict_proba(d)[:, 1]

        # Validación cruzada agrupada por región o estratificada si hay una sola.
        try:
            train_valid, y_tr = horizon_labels(train_df, horizon)
            cv_df = train_df[train_valid].reset_index(drop=True)
            y_tr = y_tr[train_valid]
            groups = (
                cv_df["region"].astype(str).to_numpy()
                if "region" in cv_df.columns
                else np.array(["all"] * len(cv_df))
            )
            if len(np.unique(groups)) >= 2:
                n_splits = max(2, min(settings.ml_cv_folds, len(np.unique(groups))))
                cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
                cv_name = f"StratifiedGroupKFold por región ({n_splits} folds)"
            else:
                n_splits = max(2, min(settings.ml_cv_folds, max(2, len(cv_df) // 100)))
                cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
                cv_name = f"StratifiedKFold estratificado ({n_splits} folds)"
            scores = []
            split_iter = (
                cv.split(cv_df, y_tr, groups)
                if len(np.unique(groups)) >= 2
                else cv.split(cv_df, y_tr)
            )
            for tr_idx, va_idx in split_iter:
                fold_pipe = build_tree_pipeline(_new_model(), features)
                fold_pipe.fit(cv_df.iloc[tr_idx], y_tr[tr_idx])
                probs = fold_pipe.predict_proba(cv_df.iloc[va_idx])[:, 1]
                if len(np.unique(y_tr[va_idx])) > 1:
                    scores.append(roc_auc_score(y_tr[va_idx], probs))
            if scores:
                cv_metrics = {
                    "strategy": cv_name,
                    "folds": len(scores),
                    "roc_auc_mean": round(float(np.mean(scores)), 4),
                    "roc_auc_std": round(float(np.std(scores)), 4),
                }
        except Exception as exc:  # noqa: BLE001
            cv_metrics = {"strategy": "no disponible", "error": str(exc)[:200]}

        artifact["feature_importance"] = _tree_importance(pipe, feature_map)

    elif kind in ("survival", "hybrid"):
        bundle = _fit_survival_model(
            run.algorithm, train_df, horizon, trials, SEED, features=features
        )
        val_risk = _survival_risk(bundle, val_df, horizon)
        if valid_val.sum() and len(np.unique(y_val[valid_val])) > 1:
            threshold = best_threshold(y_val[valid_val], val_risk[valid_val])
            selection_auc = _safe_auc(y_val[valid_val], val_risk[valid_val])
        else:
            threshold = 0.5
        # Reajuste con todo el entrenamiento reutilizando hiperparámetros ya elegidos.
        bundle = _fit_survival_model(
            run.algorithm,
            train_df,
            horizon,
            0,
            SEED,
            fixed_params=bundle.get("params"),
            features=features,
        )
        artifact.update(bundle)
        artifact["threshold"] = threshold
        artifact["_background"] = bundle["prep"].transform(
            train_df.sample(min(25, len(train_df)), random_state=SEED)
        ).to_numpy()
        predict_fn = lambda d: _survival_risk(bundle, d, horizon)
        params = bundle["params"]
        prep = bundle["prep"]
        if "cph" in bundle:
            coefs = bundle["cph"].params_.to_numpy()
            names = list(prep.columns)
            artifact["feature_importance"] = summary_importance(
                names, np.abs(coefs), prep.feature_map
            )
        else:
            names = list(prep.columns)
            artifact["feature_importance"] = summary_importance(
                names, bundle["xgb"].feature_importances_, prep.feature_map
            )

    elif kind == "deep":
        if not TORCH_AVAILABLE:
            raise RuntimeError(
                "PyTorch no está disponible en este entorno. Instale torch para entrenar el modelo LSTM-Attention-Cox."
            )
        fitted = _train_lstm(fit_df, val_df, seed=SEED)
        val_risk = _lstm_risk(fitted, val_df, horizon)
        if valid_val.sum() and len(np.unique(y_val[valid_val])) > 1:
            threshold = best_threshold(y_val[valid_val], val_risk[valid_val])
            selection_auc = _safe_auc(y_val[valid_val], val_risk[valid_val])
        else:
            threshold = 0.5
        # Reentrena con todo el entrenamiento manteniendo la arquitectura.
        full = _train_lstm(train_df, val_df, epochs=max(20, fitted["epochs"]), seed=SEED)
        artifact.update(
            {
                "state_dict": full["state_dict"],
                "model_config": {"input_size": len(VISIT_FEATURES), "hidden": 24},
                "seq_mean": full["seq_mean"],
                "seq_std": full["seq_std"],
                "baseline_times": full["baseline_times"],
                "baseline_hazard": full["baseline_hazard"],
                "threshold": threshold,
                "epochs": full["epochs"],
            }
        )
        model = LSTMAttentionCox(len(VISIT_FEATURES), hidden=24)
        model.load_state_dict(full["state_dict"])
        model.eval()
        artifact["_model"] = model
        predict_fn = lambda d: _lstm_risk({**artifact, "model": model}, d, horizon)
        has_visits = any(f"visit_{v}_" in c for v in range(VISITS) for c in df.columns)
        params = {
            "hidden_size": 24,
            "visits": VISITS,
            "epochs": full["epochs"],
            "optimizer": "Adam",
            "loss": "Cox partial likelihood (Breslow)",
            "weight_decay": 1e-4,
            "longitudinal_visits_available": has_visits,
        }
        # Importancia global por oclusión (variables longitudinales).
        sample = train_df.sample(min(80, len(train_df)), random_state=SEED).reset_index(drop=True)
        base_risk = predict_fn(sample)
        importances = []
        for feature in VISIT_FEATURES:
            mutated = sample.copy()
            for v in range(VISITS):
                col = f"visit_{v}_{feature}"
                if col in mutated.columns:
                    mutated[col] = np.nan
            alt = predict_fn(mutated)
            importances.append(
                {
                    "feature": feature,
                    "label": {
                        "sbp": "Presión sistólica (longitudinal)",
                        "chol": "Colesterol (longitudinal)",
                        "glucose": "Glucosa (longitudinal)",
                        "bmi": "IMC (longitudinal)",
                        "pm25": "PM2.5 (longitudinal)",
                        "activity": "Actividad física (longitudinal)",
                    }[feature],
                    "importance": float(np.mean(np.abs(alt - base_risk))),
                }
            )
        total = sum(i["importance"] for i in importances) or 1.0
        for item in importances:
            item["importance"] = round(item["importance"] / total, 4)
        artifact["feature_importance"] = sorted(
            importances, key=lambda i: i["importance"], reverse=True
        )
    else:
        raise ValueError("Tipo de modelo no soportado.")

    artifact["trained_at"] = utcnow().isoformat()

    # Evaluación en holdout (temporal/aleatorio) y geográfico, nunca usados en ajuste.
    temporal_valid, temporal_y = horizon_labels(temporal, horizon)
    temporal_p = predict_fn(temporal)
    geo_valid, geo_y = (
        horizon_labels(geographic, horizon)
        if len(geographic)
        else (np.array([], bool), np.array([]))
    )
    geo_p = predict_fn(geographic) if len(geographic) else np.array([])

    test_metrics = (
        full_metrics(temporal_y[temporal_valid], temporal_p[temporal_valid], artifact["threshold"])
        if temporal_valid.sum()
        else {}
    )
    geo_metrics = (
        full_metrics(geo_y[geo_valid], geo_p[geo_valid], artifact["threshold"])
        if len(geo_valid) and geo_valid.sum() and len(np.unique(geo_y[geo_valid])) > 1
        else {}
    )

    if cv_metrics.get("roc_auc_mean") is not None:
        selection_auc = cv_metrics["roc_auc_mean"]
        selection_strategy = f"CV interna: {cv_metrics.get('strategy')}"
    metrics: dict[str, Any] = {
        "discrimination": test_metrics,
        "temporal": {
            "n": int(temporal_valid.sum()),
            "cutoff_year": split.cutoff_year,
            "split_type": split.split_type,
            "design_note": split.design_note,
            "prevalence": round(float(np.mean(temporal_y[temporal_valid])), 4)
            if temporal_valid.sum()
            else None,
            "metrics": test_metrics,
        },
        "geographic": {
            "region": split.geo_holdout,
            "n": int(geo_valid.sum()),
            "note": (
                "Región excluida del entrenamiento y de la prueba; se evalúa por separado."
                if split.geo_holdout
                else "No disponible: el dataset tiene una sola región o tamaño insuficiente."
            ),
            "metrics": geo_metrics,
        },
        "cv": cv_metrics,
        "selection": {
            "strategy": selection_strategy,
            "roc_auc": round(float(selection_auc), 4) if selection_auc is not None else None,
            "uses_test_set": False,
        },
        "validation_strategy": split.design_note,
        "subgroups": _subgroup_report(
            temporal[temporal_valid].reset_index(drop=True),
            temporal_y[temporal_valid],
            temporal_p[temporal_valid],
            artifact["threshold"],
        )
        if temporal_valid.sum()
        else [],
        "test_predictions": [
            {"y": int(y), "p": round(float(p), 4)}
            for y, p in zip(temporal_y[temporal_valid], temporal_p[temporal_valid])
        ],
        "features_used": features,
        "is_demo": bool(run.is_demo),
        "disclaimer": (
            "Métricas sobre una cohorte observada de acceso público (Framingham). "
            "No constituyen validación clínica externa ni evidencia de beneficio asistencial."
            if not run.is_demo
            else "Métricas sobre datos sintéticos de demostración. No constituyen evidencia clínica."
        ),
    }

    # C-index solo si el seguimiento tiene tiempos distintos (en Framingham todos son 10).
    if kind in ("survival", "hybrid", "deep") and temporal["followup_years"].nunique() > 1:
        try:
            from lifelines.utils import concordance_index

            cidx = concordance_index(
                temporal["followup_years"], -predict_fn(temporal), temporal["event"].astype(int)
            )
            metrics["c_index_temporal"] = round(float(cidx), 4)
        except Exception:
            pass

    artifact.pop("_model", None)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    artifact_path = MODEL_DIR / f"run_{run.id}_{run.algorithm}.joblib"
    artifact_sha256 = _save_artifact_atomic(artifact, artifact_path)
    params = {**params, "artifact_sha256": artifact_sha256, "features_used": features}

    run.metrics = metrics
    run.params = params
    run.feature_importance = artifact.get("feature_importance", [])
    run.artifact_path = str(artifact_path)
    run.horizon_years = horizon
    run.risk_thresholds = artifact["risk_thresholds"]
    run.train_rows = int(len(train_df))
    run.test_rows = int(len(temporal))
    run.validation_strategy = split.design_note
    run.duration_seconds = (
        round(time.time() - run._started, 1) if hasattr(run, "_started") else None
    )
    run.trained_at = utcnow()
    run.status = "ready"
    run.error = None

    if settings.mlflow_enabled:
        try:
            import mlflow

            mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
            mlflow.set_experiment("cardioai")
            with mlflow.start_run(run_name=f"{run.algorithm}-{run.version}") as mlrun:
                mlflow.log_params({k: str(v) for k, v in params.items()})
                for key in ("roc_auc", "f1", "mcc", "brier", "sensitivity", "specificity"):
                    value = metrics.get("discrimination", {}).get(key)
                    if value is not None:
                        mlflow.log_metric(key, float(value))
                mlflow.log_artifact(str(artifact_path))
                run.mlflow_run_id = mlrun.info.run_id
        except Exception:
            pass



def _training_worker(
    run_ids: list[int],
    df: pd.DataFrame,
    trials: int,
    horizon: int,
    features: list[str] | None = None,
) -> None:
    from .database import SessionLocal

    db = SessionLocal()
    try:
        for run_id in run_ids:
            run = db.get(ModelRun, run_id)
            if run is None:
                continue
            run.status = "running"
            run._started = time.time()  # type: ignore[attr-defined]
            db.commit()
            try:
                _fit_and_evaluate(db, run, df, trials, horizon, features=features)
                db.add(
                    Alert(
                        type="modelo",
                        severity="info",
                        title="Entrenamiento completado",
                        message=f"El modelo {run.name} finalizó correctamente.",
                        module="models",
                    )
                )
                db.commit()
            except Exception as exc:  # noqa: BLE001
                db.rollback()
                run = db.get(ModelRun, run_id)
                run.status = "failed"
                run.error = f"{exc}\n{traceback.format_exc()[-1500:]}"
                run.duration_seconds = round(time.time() - getattr(run, "_started", time.time()), 1)
                db.add(
                    Alert(
                        type="modelo",
                        severity="critical",
                        title="Fallo de entrenamiento",
                        message=f"El modelo {run.name} falló: {exc}",
                        module="models",
                    )
                )
                db.commit()
        _activate_best(db, run_ids)
    finally:
        db.close()


def _selection_score(run: ModelRun) -> float:
    """Puntaje de selección: validación interna o CV, nunca el holdout de prueba."""
    metrics = run.metrics or {}
    selection = metrics.get("selection") or {}
    value = selection.get("roc_auc")
    if value is None:
        value = (metrics.get("cv") or {}).get("roc_auc_mean")
    if value is None:
        # Fallback explícito: se promedia la validación interna disponible sin tocar el test.
        value = 0.0
    return float(value or 0.0)


def _activate_best(db: Session, run_ids: list[int]) -> None:
    candidates = (
        db.query(ModelRun)
        .filter(ModelRun.id.in_(run_ids), ModelRun.status == "ready")
        .all()
    )
    if not candidates:
        return
    best = max(candidates, key=_selection_score)
    selection = (best.metrics or {}).get("selection") or {}
    db.query(ModelRun).update({ModelRun.is_active: False})
    best.is_active = True
    db.add(
        Alert(
            type="modelo",
            severity="info",
            title="Modelo activo actualizado",
            message=(
                f"Se activó {best.name} por mayor AUC-ROC en validación interna "
                f"({selection.get('roc_auc')}, estrategia: {selection.get('strategy')}). "
                "El holdout de prueba no se usó para seleccionar. La activación automática "
                "es una convención de demostración."
            ),
            module="models",
        )
    )
    db.commit()


def recover_orphaned_runs(db: Session) -> int:
    """Marca como fallidos los runs que quedaron colgados tras un reinicio."""
    orphaned = (
        db.query(ModelRun)
        .filter(ModelRun.status.in_(("running", "queued")))
        .all()
    )
    if not orphaned:
        return 0
    for run in orphaned:
        run.status = "failed"
        run.error = (
            "Entrenamiento interrumpido por reinicio del servidor. "
            "Vuelva a lanzar el entrenamiento; este run no se reanuda automáticamente."
        )
    db.commit()
    return len(orphaned)


def start_training(db: Session, dataset_id: int | None, trials: int, user_id: int | None) -> list[ModelRun]:
    df, is_demo, dataset = load_training_frame(db, dataset_id)
    if len(df) < 200:
        raise ValueError("Se requieren al menos 200 registros para entrenar.")
    features = resolve_features(df)
    if len(features) < 2:
        raise ValueError(
            "El dataset no contiene al menos dos predictores utilizables con cobertura suficiente."
        )
    horizon = settings.prediction_horizon_years
    runs: list[ModelRun] = []
    for algo in ALGORITHMS:
        count = db.query(func.count(ModelRun.id)).filter(ModelRun.algorithm == algo["key"]).scalar() or 0
        run = ModelRun(
            name=f"{algo['name']} · v{count + 1}",
            algorithm=algo["key"],
            version=str(count + 1),
            status="queued",
            is_demo=is_demo,
            dataset_id=dataset.id if dataset else None,
            horizon_years=horizon,
            risk_thresholds={
                "low_max": settings.risk_low_max,
                "high_min": settings.risk_high_min,
            },
            trained_by=user_id,
            notes=algo["description"],
        )
        db.add(run)
        runs.append(run)
    db.flush()
    run_ids = [r.id for r in runs]
    db.commit()

    import threading

    thread = threading.Thread(
        target=_training_worker,
        args=(run_ids, df, max(1, trials), horizon, features),
        daemon=True,
        name="cardioai-training",
    )
    thread.start()
    return runs


# ---------------------------------------------------------------------------
# Inferencia y explicabilidad
# ---------------------------------------------------------------------------


def load_artifact(path: str) -> dict[str, Any]:
    artifact = joblib.load(path)
    if artifact.get("kind") == "lstm_attention_cox" and TORCH_AVAILABLE:
        model = LSTMAttentionCox(
            artifact["model_config"]["input_size"], artifact["model_config"]["hidden"]
        )
        model.load_state_dict(artifact["state_dict"])
        model.eval()
        artifact["_model"] = model
    return artifact


def _records_frame(records: list[dict[str, Any]]) -> pd.DataFrame:
    df = pd.DataFrame(records)
    for col in FEATURES:
        if col not in df.columns:
            df[col] = np.nan
    # Convierte booleanos textuales provenientes de formularios/CSV.
    for col in BOOLEAN_INPUTS:
        if col in df.columns:
            df[col] = df[col].map(
                lambda v: True if str(v).lower() in {"true", "1", "si", "sí", "yes"} else
                False if str(v).lower() in {"false", "0", "no"} else v
            )
    for col in NUMERIC_INPUTS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def predict_records(artifact: dict[str, Any], records: list[dict[str, Any]]) -> np.ndarray:
    df = _records_frame(records)
    horizon = int(artifact.get("horizon", settings.prediction_horizon_years))
    kind = artifact["kind"]
    if kind in ("random_forest", "xgboost"):
        return artifact["pipeline"].predict_proba(df)[:, 1]
    if kind in ("cox_ph", "xgboost_cox"):
        return _survival_risk(artifact, df, horizon)
    if kind == "lstm_attention_cox":
        return _lstm_risk({**artifact, "model": artifact.get("_model")}, df, horizon)
    raise ValueError("Modelo no soportado para inferencia.")


def _tree_shap(artifact: dict[str, Any], df: pd.DataFrame) -> tuple[np.ndarray, float, dict[str, str]]:
    import shap

    pipe: Pipeline = artifact["pipeline"]
    pre = pipe.named_steps["pre"]
    clf = pipe.named_steps["clf"]
    transformed = pre.transform(df)
    explainer = shap.TreeExplainer(clf)
    values = explainer.shap_values(transformed)
    if isinstance(values, list):
        values = values[1] if len(values) > 1 else values[0]
    values = np.asarray(values)
    if values.ndim == 3:
        values = values[:, :, 1]
    expected = explainer.expected_value
    if isinstance(expected, (list, np.ndarray)):
        expected = float(np.ravel(expected)[-1])
    return values, float(expected), artifact["feature_map"]


def _kernel_shap(artifact: dict[str, Any], df: pd.DataFrame) -> tuple[np.ndarray, float, dict[str, str]]:
    import shap

    prep: SurvivalPreprocessor = artifact["prep"]
    background = artifact.get("_background")
    if background is None:
        # Fondo sintético neutro (medianas y categorías de referencia).
        sample = pd.DataFrame([{**{c: np.nan for c in FEATURES}} for _ in range(10)])
        background = prep.transform(sample)
    background = np.asarray(background)

    if "cph" in artifact:
        cph = artifact["cph"]
        times, hazard = artifact["baseline_times"], artifact["baseline_hazard"]
        horizon = artifact["horizon"]
        h0 = _h0_at(times, hazard, horizon)

        def fn(z):
            d = pd.DataFrame(z, columns=prep.columns)
            lp = cph.predict_log_partial_hazard(d).to_numpy()
            return 1.0 - np.exp(-h0 * np.exp(lp))
    else:
        xgb, cox = artifact["xgb"], artifact["cox"]
        times, hazard = artifact["baseline_times"], artifact["baseline_hazard"]
        horizon = artifact["horizon"]
        h0 = _h0_at(times, hazard, horizon)
        mean, std = artifact["score_mean"], artifact["score_std"]

        def fn(z):
            d = pd.DataFrame(z, columns=prep.columns)
            score = (xgb.predict(d) - mean) / std
            lp = cox.predict_log_partial_hazard(pd.DataFrame({"xgb_score": score})).to_numpy()
            return 1.0 - np.exp(-h0 * np.exp(lp))

    x = prep.transform(df).to_numpy()
    explainer = shap.KernelExplainer(fn, background)
    values = explainer.shap_values(x, nsamples=120, silent=True)
    values = np.asarray(values)
    if values.ndim == 3:
        values = values[:, :, 0]
    base = float(np.ravel(explainer.expected_value)[0])
    return values, base, prep.feature_map


def _lstm_explanation(artifact: dict[str, Any], df: pd.DataFrame) -> tuple[dict[str, Any], float]:
    model = artifact.get("_model")
    seq, mask, _, _ = _sequence_tensor(df, artifact["seq_mean"], artifact["seq_std"])
    _, weights = _lstm_predict(model, seq, mask)
    h0 = _h0_at(artifact["baseline_times"], artifact["baseline_hazard"], artifact["horizon"])

    def risk_from_seq(candidate: np.ndarray) -> float:
        log_hazard, _ = _lstm_predict(model, candidate, mask)
        return float(1.0 - np.exp(-h0 * np.exp(log_hazard[0])))

    base_risk = risk_from_seq(seq)
    factors = []
    value_aliases = {
        "sbp": ["systolic_bp", "visit_0_sbp"],
        "chol": ["cholesterol_total", "visit_0_chol"],
        "glucose": ["glucose", "visit_0_glucose"],
        "bmi": ["bmi", "visit_0_bmi"],
        "pm25": ["pm25", "visit_0_pm25"],
        "activity": ["physical_activity_min", "visit_0_activity"],
    }
    for j, feature in enumerate(VISIT_FEATURES):
        # Oclusión: reemplaza la variable por su media (0 tras estandarizar)
        # en todas las visitas y mide el cambio en la probabilidad estimada.
        alt_seq = seq.copy()
        alt_seq[:, :, j] = 0.0
        alt = risk_from_seq(alt_seq)
        raw_value = next(
            (df.iloc[0][c] for c in value_aliases[feature] if c in df.columns and pd.notna(df.iloc[0][c])),
            None,
        )
        factors.append(
            {
                "feature": feature,
                "label": {
                    "sbp": "Presión sistólica",
                    "chol": "Colesterol",
                    "glucose": "Glucosa",
                    "bmi": "IMC",
                    "pm25": "PM2.5",
                    "activity": "Actividad física",
                }[feature],
                "value": None if raw_value is None else float(raw_value),
                "contribution": base_risk - alt,
                "direction": "aumenta" if base_risk - alt > 0 else "reduce",
            }
        )
    factors.sort(key=lambda f: abs(f["contribution"]), reverse=True)
    attention = [
        {"visit": v + 1, "year": v * 2, "weight": round(float(weights[0, v]), 4)}
        for v in range(VISITS)
    ]
    return {"factors": factors[:8], "attention": attention}, base_risk


def explain_prediction(
    artifact: dict[str, Any],
    record: dict[str, Any],
    probability: float,
) -> dict[str, Any]:
    df = _records_frame([record])
    kind = artifact["kind"]
    try:
        if kind in ("random_forest", "xgboost"):
            values, base, feature_map = _tree_shap(artifact, df)
            method = "SHAP (TreeExplainer)"
            values_row = values[0]
        elif kind in ("cox_ph", "xgboost_cox"):
            values, base, feature_map = _kernel_shap(artifact, df)
            method = "SHAP (KernelExplainer)"
            values_row = values[0]
        elif kind == "lstm_attention_cox":
            explanation, base = _lstm_explanation(artifact, df)
            factors = explanation["factors"]
            return {
                "method": "Atención temporal + oclusión por variable",
                "base_value": round(float(base), 4),
                "factors": factors,
                "attention": explanation["attention"],
                "notes": "La contribución se estima ocluyendo cada variable en todas las visitas. "
                         "No es SHAP clásico; se informa para no confundir métodos.",
            }
        else:
            return {"method": "No disponible", "factors": []}
    except Exception as exc:  # noqa: BLE001
        return {"method": "No disponible", "factors": [], "error": str(exc)[:300]}

    totals: dict[str, float] = {}
    for name, value in zip(list(artifact["pipeline"].named_steps["pre"].get_feature_names_out())
                           if kind in ("random_forest", "xgboost") else artifact["prep"].columns,
                           values_row):
        base_name = feature_map.get(name, name)
        totals[base_name] = totals.get(base_name, 0.0) + float(value)
    ranked = sorted(totals.items(), key=lambda kv: abs(kv[1]), reverse=True)[:8]
    factors = [
        {
            "feature": feature,
            "label": FEATURE_LABELS.get(feature, feature),
            "value": record.get(feature),
            "contribution": round(contribution, 4),
            "direction": "aumenta" if contribution > 0 else "reduce",
        }
        for feature, contribution in ranked
    ]
    return {
        "method": method,
        "base_value": round(float(base), 4),
        "factors": factors,
        "notes": "Contribuciones sobre la probabilidad estimada. No implican causalidad.",
    }


def dataset_region_summary(df: pd.DataFrame) -> list[dict[str, Any]]:
    rows = []
    for region, group in df.groupby("region"):
        rows.append(
            {
                "region": region,
                "count": int(len(group)),
                "mean_pm25": round(float(group["pm25"].mean()), 2) if group["pm25"].notna().any() else None,
                "event_rate": round(float(group["event"].mean()), 4) if "event" in group else None,
                "lat": REGION_INFO.get(region, {}).get("lat"),
                "lon": REGION_INFO.get(region, {}).get("lon"),
            }
        )
    return rows

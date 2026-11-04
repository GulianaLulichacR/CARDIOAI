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

import json
import math
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
    cutoff_year: int


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
    df = df.dropna(subset=["followup_years", "event"]).copy()
    df["event"] = df["event"].astype(bool)
    df["cohort_year"] = pd.to_numeric(df.get("cohort_year", 2015), errors="coerce").fillna(2015).astype(int)

    cutoff = int(df["cohort_year"].quantile(0.8))
    temporal_mask = df["cohort_year"] >= cutoff
    train = df[~temporal_mask]
    temporal_test = df[temporal_mask]
    if len(temporal_test) < 40 or len(train) < 200:
        train, temporal_test = train_test_split(
            df, test_size=0.2, random_state=seed, stratify=df["event"]
        )
        cutoff = int(train["cohort_year"].max()) if len(train) else 0

    region_counts = df["region"].value_counts()
    candidates = [
        r for r in ["Nariño", "Cauca", "Meta", "Tolima", "La Guajira"]
        if r in region_counts.index and region_counts[r] >= 30 and region_counts[r] <= len(df) * 0.15
    ]
    geo_holdout = candidates[0] if candidates else None
    if geo_holdout:
        geographic_test = df[df["region"] == geo_holdout]
        geo_train = df[df["region"] != geo_holdout]
        if len(geographic_test) < 25 or len(geo_train) < 150:
            geographic_test = df.iloc[0:0]
            geo_holdout = None
    else:
        geographic_test = df.iloc[0:0]

    return SplitBundle(train=train, temporal_test=temporal_test, geographic_test=geographic_test,
                       geo_holdout=geo_holdout, cutoff_year=cutoff)


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
    for col in BOOLEAN_FEATURES:
        out[col] = out[col].fillna(False).astype(int)
    return out


# ---------------------------------------------------------------------------
# Preprocesador para modelos de supervivencia (serializable)
# ---------------------------------------------------------------------------


class SurvivalPreprocessor:
    def __init__(self) -> None:
        self.numeric_cols = NUMERIC_FEATURES + BOOLEAN_FEATURES
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
        dummies = pd.get_dummies(work[CATEGORICAL_FEATURES].astype(str), prefix=CATEGORICAL_FEATURES)
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
            dummies = pd.get_dummies(work[CATEGORICAL_FEATURES].astype(str), prefix=CATEGORICAL_FEATURES)
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


def build_tree_pipeline(model) -> Pipeline:
    numeric = Pipeline(
        [("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]
    )
    categorical = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    pre = ColumnTransformer(
        [
            ("num", numeric, NUMERIC_FEATURES),
            ("cat", categorical, CATEGORICAL_FEATURES),
            ("bool", numeric, BOOLEAN_FEATURES),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    return Pipeline([("pre", pre), ("clf", model)])


def _tree_feature_map(names: list[str]) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for name in names:
        base = name.split("__", 1)[-1]
        found = None
        for cat in CATEGORICAL_FEATURES:
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
) -> dict[str, Any]:
    from sklearn.ensemble import RandomForestClassifier
    from xgboost import XGBClassifier

    groups = fit_df["region"].to_numpy()
    n_splits = min(settings.ml_cv_folds, len(np.unique(groups)))
    cv = StratifiedGroupKFold(n_splits=max(2, n_splits), shuffle=True, random_state=seed)

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
            model = RandomForestClassifier(**params, random_state=seed, n_jobs=2,
                                           class_weight="balanced_subsample")
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
        pipe = build_tree_pipeline(model)
        scores = []
        for train_idx, val_idx in cv.split(fit_df, labels, groups):
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
) -> dict[str, Any]:
    prep = SurvivalPreprocessor().fit(train_df)
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


def _fit_and_evaluate(
    db: Session,
    run: ModelRun,
    df: pd.DataFrame,
    trials: int,
    horizon: int,
) -> None:
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
        "features": FEATURES,
        "horizon": horizon,
        "risk_thresholds": {
            "low_max": settings.risk_low_max,
            "high_min": settings.risk_high_min,
            "basis": "Umbrales configurables de DEMOSTRACIÓN; requieren validación clínica.",
        },
    }
    cv_metrics: dict[str, Any] = {}

    if kind == "tree":
        params = _tune_tree(run.algorithm, fit_df[valid_fit].reset_index(drop=True),
                            y_fit[valid_fit], trials, SEED)
        from sklearn.ensemble import RandomForestClassifier
        from xgboost import XGBClassifier

        if run.algorithm == "random_forest":
            model = RandomForestClassifier(**params, random_state=SEED, n_jobs=2,
                                           class_weight="balanced_subsample")
        else:
            model = XGBClassifier(**params, random_state=SEED, n_jobs=2,
                                  eval_metric="logloss", tree_method="hist")
        pipe = build_tree_pipeline(model)
        pipe.fit(fit_df[valid_fit], y_fit[valid_fit])
        val_probs = pipe.predict_proba(val_df)[:, 1]
        if valid_val.sum() and len(np.unique(y_val[valid_val])) > 1:
            threshold = best_threshold(y_val[valid_val], val_probs[valid_val])
        else:
            threshold = 0.5
        # Reentrena con todo el conjunto de entrenamiento y el umbral ya fijado.
        train_valid, y_train = horizon_labels(train_df, horizon)
        pipe.fit(train_df[train_valid], y_train[train_valid])
        names = list(pipe.named_steps["pre"].get_feature_names_out())
        feature_map = _tree_feature_map(names)
        artifact.update({"pipeline": pipe, "threshold": threshold, "feature_map": feature_map})
        predict_fn: Callable[[pd.DataFrame], np.ndarray] = lambda d: pipe.predict_proba(d)[:, 1]

        # Validación cruzada agrupada por región (solo datos de entrenamiento).
        try:
            train_valid, y_tr = horizon_labels(train_df, horizon)
            cv_df = train_df[train_valid].reset_index(drop=True)
            y_tr = y_tr[train_valid]
            g = cv_df["region"].to_numpy()
            cv = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=SEED)
            scores = []
            for tr_idx, va_idx in cv.split(cv_df, y_tr, g):
                fold_pipe = build_tree_pipeline(
                    RandomForestClassifier(**params, random_state=SEED, n_jobs=2,
                                           class_weight="balanced_subsample")
                    if run.algorithm == "random_forest"
                    else XGBClassifier(**params, random_state=SEED, n_jobs=2, eval_metric="logloss",
                                       tree_method="hist")
                )
                fold_pipe.fit(cv_df.iloc[tr_idx], y_tr[tr_idx])
                probs = fold_pipe.predict_proba(cv_df.iloc[va_idx])[:, 1]
                if len(np.unique(y_tr[va_idx])) > 1:
                    scores.append(roc_auc_score(y_tr[va_idx], probs))
            if scores:
                cv_metrics = {"strategy": "StratifiedGroupKFold por región", "folds": len(scores),
                              "roc_auc_mean": round(float(np.mean(scores)), 4),
                              "roc_auc_std": round(float(np.std(scores)), 4)}
        except Exception:
            cv_metrics = {}

        artifact["feature_importance"] = _tree_importance(pipe, feature_map)

    elif kind in ("survival", "hybrid"):
        bundle = _fit_survival_model(run.algorithm, train_df, horizon, trials, SEED)
        val_risk = _survival_risk(bundle, val_df, horizon)
        if valid_val.sum() and len(np.unique(y_val[valid_val])) > 1:
            threshold = best_threshold(y_val[valid_val], val_risk[valid_val])
        else:
            threshold = 0.5
        # Reajuste con todo el entrenamiento reutilizando hiperparámetros ya elegidos.
        bundle = _fit_survival_model(
            run.algorithm, train_df, horizon, 0, SEED, fixed_params=bundle.get("params")
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
        else:
            threshold = 0.5
        # Reentrena con todos los datos de entrenamiento manteniendo la arquitectura.
        full = _train_lstm(train_df, val_df, epochs=max(20, fitted["epochs"]), seed=SEED)
        artifact.update({
            "state_dict": full["state_dict"],
            "model_config": {"input_size": len(VISIT_FEATURES), "hidden": 24},
            "seq_mean": full["seq_mean"],
            "seq_std": full["seq_std"],
            "baseline_times": full["baseline_times"],
            "baseline_hazard": full["baseline_hazard"],
            "threshold": threshold,
            "epochs": full["epochs"],
        })
        model = LSTMAttentionCox(len(VISIT_FEATURES), hidden=24)
        model.load_state_dict(full["state_dict"])
        model.eval()
        artifact["_model"] = model
        predict_fn = lambda d: _lstm_risk({**artifact, "model": model}, d, horizon)
        params = {"hidden_size": 24, "visits": VISITS, "epochs": full["epochs"],
                  "optimizer": "Adam", "loss": "Cox partial likelihood (Breslow)",
                  "weight_decay": 1e-4}
        # Importancia global por oclusión (6 variables longitudinales).
        sample = train_df.sample(min(80, len(train_df)), random_state=SEED).reset_index(drop=True)
        base_risk = predict_fn(sample)
        importances = []
        for j, feature in enumerate(VISIT_FEATURES):
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
        artifact["feature_importance"] = sorted(importances, key=lambda i: i["importance"], reverse=True)
    else:
        raise ValueError("Tipo de modelo no soportado.")

    artifact["trained_at"] = utcnow().isoformat()

    # Evaluación en test temporal y geográfico (nunca usados en ajuste).
    temporal_valid, temporal_y = horizon_labels(temporal, horizon)
    temporal_p = predict_fn(temporal)
    geo_valid, geo_y = horizon_labels(geographic, horizon) if len(geographic) else (np.array([], bool), np.array([]))
    geo_p = predict_fn(geographic) if len(geographic) else np.array([])

    metrics: dict[str, Any] = {
        "discrimination": full_metrics(temporal_y[temporal_valid], temporal_p[temporal_valid],
                                      artifact["threshold"]) if temporal_valid.sum() else {},
        "temporal": {
            "n": int(temporal_valid.sum()),
            "cutoff_year": split.cutoff_year,
            "metrics": full_metrics(temporal_y[temporal_valid], temporal_p[temporal_valid],
                                    artifact["threshold"]) if temporal_valid.sum() else {},
        },
        "geographic": {
            "region": split.geo_holdout,
            "n": int(geo_valid.sum()),
            "metrics": full_metrics(geo_y[geo_valid], geo_p[geo_valid], artifact["threshold"])
            if len(geo_valid) and geo_valid.sum() and len(np.unique(geo_y[geo_valid])) > 1 else {},
        },
        "cv": cv_metrics,
        "validation_strategy": "Temporal (cohortes recientes) + geográfica (región retenida); "
                               "Optuna y umbral solo en entrenamiento/validación interna.",
        "is_demo": run.is_demo,
        "disclaimer": "Métricas sobre datos sintéticos de demostración. No constituyen evidencia clínica.",
    }

    # C-index para modelos de supervivencia.
    if kind in ("survival", "hybrid", "deep"):
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
    joblib.dump(artifact, artifact_path)

    run.metrics = metrics
    run.params = params
    run.feature_importance = artifact.get("feature_importance", [])
    run.artifact_path = str(artifact_path)
    run.horizon_years = horizon
    run.risk_thresholds = artifact["risk_thresholds"]
    run.train_rows = int(len(train_df))
    run.test_rows = int(len(temporal))
    run.validation_strategy = metrics["validation_strategy"]
    run.duration_seconds = round(time.time() - run._started, 1) if hasattr(run, "_started") else None
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


def _training_worker(run_ids: list[int], df: pd.DataFrame, trials: int, horizon: int) -> None:
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
                _fit_and_evaluate(db, run, df, trials, horizon)
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


def _activate_best(db: Session, run_ids: list[int]) -> None:
    candidates = (
        db.query(ModelRun)
        .filter(ModelRun.id.in_(run_ids), ModelRun.status == "ready")
        .all()
    )
    if not candidates:
        return
    best = max(
        candidates,
        key=lambda r: (r.metrics or {}).get("discrimination", {}).get("roc_auc") or 0.0,
    )
    db.query(ModelRun).update({ModelRun.is_active: False})
    best.is_active = True
    db.add(
        Alert(
            type="modelo",
            severity="info",
            title="Modelo activo actualizado",
            message=(
                f"Se activó {best.name} por mayor AUC-ROC temporal ("
                f"{(best.metrics or {}).get('discrimination', {}).get('roc_auc')}). "
                "La selección automática es una convención de demostración."
            ),
            module="models",
        )
    )
    db.commit()


def start_training(db: Session, dataset_id: int | None, trials: int, user_id: int | None) -> list[ModelRun]:
    df, is_demo, dataset = load_training_frame(db, dataset_id)
    if len(df) < 200:
        raise ValueError("Se requieren al menos 200 registros para entrenar.")
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
        args=(run_ids, df, max(1, trials), horizon),
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
    for col in BOOLEAN_FEATURES:
        df[col] = df[col].map(
            lambda v: True if str(v).lower() in {"true", "1", "si", "sí", "yes"} else
            False if str(v).lower() in {"false", "0", "no"} else v
        )
    for col in NUMERIC_FEATURES:
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

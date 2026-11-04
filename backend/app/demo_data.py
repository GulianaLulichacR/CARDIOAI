"""Generación del dataset SINTÉTICO de demostración.

IMPORTANTE: estos datos NO son observaciones reales ni resultados clínicos.
Se generan con una semilla fija exclusivamente para demostrar el funcionamiento
técnico de la plataforma. Cualquier métrica obtenida con ellos es de DEMO.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import DEMO_DIR, settings

REGIONS: list[dict[str, Any]] = [
    {"name": "Bogotá D.C.", "lat": 4.711, "lon": -74.072, "pm25": 15.4, "weight": 0.20},
    {"name": "Antioquia", "lat": 6.245, "lon": -75.581, "pm25": 14.8, "weight": 0.16},
    {"name": "Valle del Cauca", "lat": 3.451, "lon": -76.532, "pm25": 16.2, "weight": 0.13},
    {"name": "Atlántico", "lat": 10.968, "lon": -74.781, "pm25": 13.5, "weight": 0.09},
    {"name": "Santander", "lat": 7.125, "lon": -73.122, "pm25": 12.8, "weight": 0.08},
    {"name": "Cundinamarca", "lat": 4.935, "lon": -74.295, "pm25": 11.6, "weight": 0.07},
    {"name": "Bolívar", "lat": 8.748, "lon": -75.498, "pm25": 13.9, "weight": 0.07},
    {"name": "Nariño", "lat": 1.289, "lon": -77.357, "pm25": 10.9, "weight": 0.05},
    {"name": "Tolima", "lat": 4.092, "lon": -75.154, "pm25": 12.2, "weight": 0.05},
    {"name": "Cauca", "lat": 2.444, "lon": -76.614, "pm25": 9.8, "weight": 0.04},
    {"name": "Meta", "lat": 3.988, "lon": -73.087, "pm25": 10.4, "weight": 0.03},
    {"name": "La Guajira", "lat": 11.544, "lon": -72.907, "pm25": 12.7, "weight": 0.03},
]
REGION_NAMES = [r["name"] for r in REGIONS]
REGION_INFO = {r["name"]: r for r in REGIONS}

NUMERIC_FEATURES = [
    "age",
    "bmi",
    "systolic_bp",
    "diastolic_bp",
    "cholesterol_total",
    "hdl",
    "glucose",
    "physical_activity_min",
    "alcohol_units",
    "pm25",
    "green_space",
    "stress_level",
]
CATEGORICAL_FEATURES = ["sex", "region", "education", "income_level", "smoking"]
BOOLEAN_FEATURES = ["urban", "diabetes", "hypertension", "family_history"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES + BOOLEAN_FEATURES
REQUIRED_COLUMNS = FEATURES + ["event", "followup_years"]
VISIT_FEATURES = ["sbp", "chol", "glucose", "bmi", "pm25", "activity"]
VISITS = 4

FEATURE_LABELS = {
    "age": "Edad (años)",
    "sex": "Sexo",
    "region": "Región",
    "urban": "Residencia urbana",
    "education": "Nivel educativo",
    "income_level": "Nivel de ingreso",
    "smoking": "Tabaquismo",
    "physical_activity_min": "Actividad física (min/sem)",
    "alcohol_units": "Unidades de alcohol/sem",
    "bmi": "IMC (kg/m²)",
    "systolic_bp": "Presión sistólica (mmHg)",
    "diastolic_bp": "Presión diastólica (mmHg)",
    "cholesterol_total": "Colesterol total (mg/dL)",
    "hdl": "HDL (mg/dL)",
    "glucose": "Glucosa (mg/dL)",
    "diabetes": "Diabetes",
    "hypertension": "Hipertensión",
    "family_history": "Antecedente familiar",
    "pm25": "PM2.5 (µg/m³)",
    "green_space": "Espacio verde (%)",
    "stress_level": "Estrés percibido (1-10)",
    "event": "Evento cardiovascular",
    "followup_years": "Años de seguimiento",
}

FEATURE_DESCRIPTIONS = {
    "age": "Edad al ingreso al estudio.",
    "sex": "Sexo registrado en el estudio (F/M).",
    "region": "Región territorial de residencia (agregada).",
    "urban": "Residencia urbana (verdadero) o rural (falso).",
    "education": "Máximo nivel educativo alcanzado.",
    "income_level": "Estrato de ingreso declarado (bajo/medio/alto).",
    "smoking": "Consumo de tabaco (nunca/exfumador/fumador).",
    "physical_activity_min": "Minutos de actividad física moderada por semana.",
    "alcohol_units": "Unidades estándar de alcohol por semana.",
    "bmi": "Índice de masa corporal.",
    "systolic_bp": "Presión arterial sistólica en reposo.",
    "diastolic_bp": "Presión arterial diastólica en reposo.",
    "cholesterol_total": "Colesterol total sérico.",
    "hdl": "Colesterol HDL sérico.",
    "glucose": "Glucosa en ayunas.",
    "diabetes": "Diagnóstico previo de diabetes.",
    "hypertension": "Diagnóstico previo de hipertensión.",
    "family_history": "Antecedente familiar de enfermedad cardiovascular.",
    "pm25": "Concentración media anual de material particulado PM2.5.",
    "green_space": "Porcentaje de cobertura verde del entorno residencial.",
    "stress_level": "Escala autoinformada de estrés percibido (1-10).",
    "event": "Ocurrencia de evento cardiovascular durante el seguimiento.",
    "followup_years": "Tiempo de seguimiento hasta evento o censura.",
}


def demo_dataset_path() -> Path:
    return DEMO_DIR / f"cardio_demo_seed{settings.demo_seed}.csv"


def _sigmoid(x: np.ndarray | float) -> np.ndarray | float:
    return 1.0 / (1.0 + np.exp(-x))


def generate_demo_dataframe(n: int = 1200, seed: int | None = None) -> pd.DataFrame:
    rng = np.random.default_rng(settings.demo_seed if seed is None else seed)
    regions = rng.choice(
        [r["name"] for r in REGIONS],
        size=n,
        p=np.array([r["weight"] for r in REGIONS]) / sum(r["weight"] for r in REGIONS),
    )
    pm_base = np.array([REGION_INFO[r]["pm25"] for r in regions])

    age = np.clip(rng.normal(57, 12, n), 30, 85).round().astype(int)
    sex = rng.choice(["F", "M"], n, p=[0.53, 0.47])
    urban = rng.random(n) < 0.78
    education = rng.choice(["primaria", "secundaria", "tecnica", "universitaria"], n, p=[0.22, 0.38, 0.22, 0.18])
    income = rng.choice(["bajo", "medio", "alto"], n, p=[0.32, 0.48, 0.20])
    smoking = rng.choice(["nunca", "exfumador", "fumador"], n, p=[0.50, 0.26, 0.24])
    activity = np.clip(rng.normal(150, 85, n), 0, 600).round().astype(int)
    alcohol = np.clip(rng.gamma(1.6, 2.2, n), 0, 30).round(1)
    bmi = np.clip(rng.normal(27.2, 4.6, n), 17, 46).round(1)
    sbp = np.clip(rng.normal(126, 16, n) + 0.45 * (age - 50) + 0.18 * (pm_base - 12), 85, 220).round().astype(int)
    dbp = np.clip(rng.normal(78, 9.5, n) + 0.12 * (age - 50), 50, 130).round().astype(int)
    chol = np.clip(rng.normal(198, 38, n) + 0.35 * (age - 50) + 0.7 * (bmi - 25), 100, 380).round().astype(int)
    hdl = np.clip(rng.normal(49, 12, n) - 3 * (sex == "M") + 0.12 * (activity - 150), 20, 110).round().astype(int)
    glucose = np.clip(rng.normal(99, 22, n) + 0.9 * (bmi - 25) + 0.3 * (age - 50), 60, 320).round().astype(int)
    diabetes = (rng.random(n) < _sigmoid(-4.2 + 0.055 * (age - 50) + 0.10 * (bmi - 25))).astype(bool)
    hypertension = (rng.random(n) < _sigmoid(-3.1 + 0.045 * (age - 50) + 0.018 * (sbp - 120))).astype(bool)
    family_history = rng.random(n) < 0.27
    pm25 = np.clip(pm_base + rng.normal(0, 2.4, n), 5, 35).round(1)
    green_space = np.clip(rng.normal(17, 11, n) + np.where(urban, -2, 8), 0, 65).round(1)
    stress = np.clip(rng.normal(5.1, 2.3, n), 1, 10).round().astype(int)
    cohort_year = rng.integers(2012, 2021, n)

    # Predictor lineal tipo Cox (coeficientes ilustrativos, NO clínicos).
    lp = (
        0.055 * (age - 55)
        + 0.50 * (smoking == "fumador")
        + 0.22 * (smoking == "exfumador")
        + 0.38 * diabetes
        + 0.30 * hypertension
        + 0.012 * (sbp - 130)
        + 0.0045 * (chol - 200)
        - 0.028 * (hdl - 50)
        + 0.045 * (bmi - 27)
        - 0.0018 * (activity - 150)
        + 0.28 * family_history
        + 0.014 * (pm25 - 12)
        + 0.16 * (income == "bajo")
        + 0.018 * (stress - 5)
        + 0.16 * (sex == "M")
    )
    u = rng.random(n)
    time_event = 26.0 * ((-np.log(u)) / np.exp(lp)) ** (1 / 1.25)
    censor = rng.uniform(3.0, 13.0, n)
    followup = np.minimum(time_event, censor).round(2)
    event = (time_event <= censor).astype(bool)

    df = pd.DataFrame(
        {
            "patient_code": [f"DEMO-{i + 1:05d}" for i in range(n)],
            "age": age,
            "sex": sex,
            "region": regions,
            "urban": urban,
            "education": education,
            "income_level": income,
            "smoking": smoking,
            "physical_activity_min": activity,
            "alcohol_units": alcohol,
            "bmi": bmi,
            "systolic_bp": sbp,
            "diastolic_bp": dbp,
            "cholesterol_total": chol,
            "hdl": hdl,
            "glucose": glucose,
            "diabetes": diabetes,
            "hypertension": hypertension,
            "family_history": family_history,
            "pm25": pm25,
            "green_space": green_space,
            "stress_level": stress,
            "cohort_year": cohort_year,
            "followup_years": followup,
            "event": event,
        }
    )

    # Secuencias longitudinales sintéticas con enmascaramiento por seguimiento.
    for v in range(VISITS):
        observed = df["followup_years"].to_numpy() >= v * 2
        df[f"visit_{v}_sbp"] = np.where(observed, sbp + 1.1 * v + rng.normal(0, 4, n), np.nan).round(1)
        df[f"visit_{v}_chol"] = np.where(observed, chol + 1.2 * v + rng.normal(0, 6, n), np.nan).round(1)
        df[f"visit_{v}_glucose"] = np.where(observed, glucose + 0.7 * v + rng.normal(0, 5, n), np.nan).round(1)
        df[f"visit_{v}_bmi"] = np.where(observed, bmi + 0.12 * v + rng.normal(0, 0.5, n), np.nan).round(1)
        df[f"visit_{v}_pm25"] = np.where(observed, pm25 + rng.normal(0, 1.2, n), np.nan).round(1)
        df[f"visit_{v}_activity"] = np.where(
            observed, np.clip(activity + rng.normal(0, 12, n), 0, 700), np.nan
        ).round(0)

    return df


def ensure_demo_dataset(force: bool = False) -> tuple[Path, pd.DataFrame]:
    path = demo_dataset_path()
    if path.exists() and not force:
        try:
            df = pd.read_csv(path)
            if len(df) >= 100:
                return path, df
        except Exception:
            pass
    df = generate_demo_dataframe()
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path, df


def file_checksum(path: Path | str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def quality_report(df: pd.DataFrame) -> dict[str, Any]:
    rows, cols = df.shape
    missing = df.isna().sum()
    report: dict[str, Any] = {
        "rows": int(rows),
        "columns": int(cols),
        "duplicate_rows": int(df.duplicated().sum()),
        "total_missing": int(missing.sum()),
        "missing_pct": round(float(missing.sum() / max(1, rows * cols) * 100), 2),
        "constant_columns": [c for c in df.columns if df[c].nunique(dropna=True) <= 1],
        "missing_by_column": {
            str(c): {"count": int(missing[c]), "pct": round(float(missing[c] / max(1, rows) * 100), 2)}
            for c in df.columns
            if missing[c] > 0
        },
        "dtypes": {str(c): str(df[c].dtype) for c in df.columns},
        "outliers_iqr": {},
    }
    for c in df.select_dtypes(include=[np.number]).columns:
        series = df[c].dropna()
        if len(series) < 10:
            continue
        q1, q3 = series.quantile(0.25), series.quantile(0.75)
        iqr = q3 - q1
        if iqr > 0:
            count = int(((series < q1 - 1.5 * iqr) | (series > q3 + 1.5 * iqr)).sum())
            if count:
                report["outliers_iqr"][str(c)] = count
    if "event" in df.columns:
        target = df["event"].dropna()
        report["target"] = {
            "column": "event",
            "positives": int(target.astype(bool).sum()),
            "negatives": int((~target.astype(bool)).sum()),
            "prevalence": round(float(target.astype(bool).mean()), 4) if len(target) else None,
        }
    return report

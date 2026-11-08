"""Ingesta reproducible de datasets públicos para CARDIOAI.

El repositorio incluye únicamente una cohorte sintética de demostración. Este
módulo permite incorporar una fuente observada con procedencia verificable,
sin inventar variables que el dataset no contiene.

Dataset incorporado: Framingham Heart Study, archivo docente de enfermedad
coronaria a 10 años (4.240 registros, 16 columnas). El archivo se valida por
SHA-256 y se adapta al esquema interno de CARDIOAI. Las variables ausentes
(PM2.5, espacio verde, ingreso, actividad física, alcohol, estrés, antecedente
familiar, HDL y visitas longitudinales) se dejan como faltantes explícitos y
se excluyen del entrenamiento mediante selección de predictores disponibles.
"""
from __future__ import annotations

import hashlib
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import DATA_DIR
from .demo_data import ADDITIONAL_FEATURES, FEATURES

PUBLIC_DIR = DATA_DIR / "datasets"
PUBLIC_DIR.mkdir(parents=True, exist_ok=True)

FRAMINGHAM_FILENAME = "framingham.csv"
FRAMINGHAM_SHA256 = "2c0e57dc0361b420becf1facec0a054af06c420eae0ae2faf0fdc8591fadb018"
FRAMINGHAM_URL = (
    "https://raw.githubusercontent.com/GauravPadawe/Framingham-Heart-Study/master/framingham.csv"
)
FRAMINGHAM_ROWS = 4240
FRAMINGHAM_COLUMNS = 16


@dataclass(frozen=True)
class PublicDatasetSpec:
    key: str
    name: str
    source: str
    license: str
    description: str
    citation: str
    canonical_url: str
    download_url: str
    filename: str
    sha256: str
    expected_rows: int
    expected_columns: int
    outcome: str
    followup: str
    unavailable_features: tuple[str, ...] = field(default_factory=tuple)

    @property
    def path(self) -> Path:
        return PUBLIC_DIR / self.filename

    @property
    def metadata(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "name": self.name,
            "source": self.source,
            "license": self.license,
            "citation": self.citation,
            "canonical_url": self.canonical_url,
            "sha256": self.sha256,
            "outcome": self.outcome,
            "followup": self.followup,
            "unavailable_features": list(self.unavailable_features),
        }


FRAMINGHAM = PublicDatasetSpec(
    key="framingham",
    name="Framingham Heart Study · CHD a 10 años (archivo docente)",
    source="Framingham Heart Study (NHLBI / Boston University); archivo docente desidentificado",
    license="Uso docente y de investigación; datos desidentificados. Ver condiciones de BioLINCC/NHLBI.",
    description=(
        "Cohorte observada de 4.240 participantes del Framingham Heart Study con enfermedad "
        "coronaria incidente a 10 años (TenYearCHD). Variables clínicas y educativas. No contiene "
        "PM2.5, espacio verde, ingreso, actividad física, alcohol, estrés, antecedente familiar, "
        "HDL ni visitas longitudinales; esas variables se tratan como no disponibles."
    ),
    citation=(
        "D'Agostino RB Sr, Vasan RS, Pencina MJ, Wolf PA, Cobain M, Massaro JM, Kannel WB. "
        "General cardiovascular risk profile for use in primary care: the Framingham Heart Study. "
        "Circulation. 2008;117(6):743-753. doi:10.1161/CIRCULATIONAHA.107.699579"
    ),
    canonical_url="https://biolincc.nhlbi.nih.gov/studies/framingham/",
    download_url=FRAMINGHAM_URL,
    filename=FRAMINGHAM_FILENAME,
    sha256=FRAMINGHAM_SHA256,
    expected_rows=FRAMINGHAM_ROWS,
    expected_columns=FRAMINGHAM_COLUMNS,
    outcome="TenYearCHD (enfermedad coronaria incidente a 10 años)",
    followup="Seguimiento administrativo fijo de 10 años; no hay tiempos individuales de censura.",
    unavailable_features=(
        "hdl",
        "physical_activity_min",
        "alcohol_units",
        "income_level",
        "family_history",
        "pm25",
        "green_space",
        "stress_level",
        "urban",
    ),
)

SPECS: dict[str, PublicDatasetSpec] = {FRAMINGHAM.key: FRAMINGHAM}


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def ensure_public_file(spec: PublicDatasetSpec, force: bool = False) -> Path:
    """Descarga (si hace falta) y valida el archivo público por SHA-256."""
    path = spec.path
    if path.exists() and not force:
        if _sha256(path) == spec.sha256:
            return path
        # Archivo alterado: se intenta redescargar una vez.
    spec.path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with urllib.request.urlopen(spec.download_url, timeout=60) as response:  # noqa: S310
        payload = response.read()
    tmp.write_bytes(payload)
    if hashlib.sha256(payload).hexdigest() != spec.sha256:
        tmp.unlink(missing_ok=True)
        raise ValueError(
            "El SHA-256 del archivo descargado no coincide con el esperado. "
            "Verifique la fuente o defina CARDIOAI_FRAMINGHAM_URL con un espejo confiable."
        )
    tmp.replace(path)
    return path


def load_framingham(force_download: bool = False) -> tuple[Path, pd.DataFrame]:
    path = ensure_public_file(FRAMINGHAM, force=force_download)
    df = pd.read_csv(path)
    if df.shape != (FRAMINGHAM.expected_rows, FRAMINGHAM.expected_columns):
        raise ValueError(
            f"Dimensiones inesperadas del Framingham: {df.shape}; "
            f"se esperaban {(FRAMINGHAM.expected_rows, FRAMINGHAM.expected_columns)}."
        )
    required = {
        "male", "age", "education", "currentSmoker", "prevalentHyp", "diabetes",
        "totChol", "sysBP", "diaBP", "BMI", "glucose", "TenYearCHD",
    }
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Faltan columnas del Framingham: {sorted(missing)}")
    return path, df


EDUCATION_MAP = {
    1.0: "sin_titulo_secundaria",
    2.0: "secundaria",
    3.0: "tecnica",
    4.0: "universitaria",
}


def adapt_framingham(raw: pd.DataFrame) -> pd.DataFrame:
    """Adapta el archivo original al esquema interno, sin imputar variables ausentes."""
    n = len(raw)
    out = pd.DataFrame(index=raw.index)
    out["record_id"] = np.arange(1, n + 1)
    out["patient_code"] = [f"FRM-{i:05d}" for i in range(1, n + 1)]
    out["age"] = raw["age"].astype(int)
    out["sex"] = np.where(raw["male"].astype(int) == 1, "M", "F")
    # Una sola región: la cohorte original es de Framingham, Massachusetts.
    out["region"] = "Framingham (Massachusetts, EE. UU.)"
    out["urban"] = np.nan
    out["education"] = raw["education"].map(EDUCATION_MAP)
    out["income_level"] = np.nan
    out["smoking"] = np.where(raw["currentSmoker"].astype(int) == 1, "fumador", "nunca")
    out["physical_activity_min"] = np.nan
    out["alcohol_units"] = np.nan
    out["bmi"] = raw["BMI"]
    out["systolic_bp"] = raw["sysBP"]
    out["diastolic_bp"] = raw["diaBP"]
    out["cholesterol_total"] = raw["totChol"]
    out["hdl"] = np.nan
    out["glucose"] = raw["glucose"]
    out["diabetes"] = raw["diabetes"].astype(int).astype(bool)
    out["hypertension"] = raw["prevalentHyp"].astype(int).astype(bool)
    out["family_history"] = np.nan
    out["pm25"] = np.nan
    out["green_space"] = np.nan
    out["stress_level"] = np.nan
    out["cohort_year"] = np.nan  # no disponible: no se inventa un año de cohorte
    out["followup_years"] = 10.0  # seguimiento administrativo fijo del archivo docente
    out["event"] = raw["TenYearCHD"].astype(int).astype(bool)
    # Variables originales conservadas para trazabilidad y análisis de sensibilidad.
    out["cigs_per_day"] = raw["cigsPerDay"]
    out["bp_medication"] = raw["BPMeds"]
    out["prevalent_stroke"] = raw["prevalentStroke"].astype(int).astype(bool)
    out["heart_rate"] = raw["heartRate"]
    return out


def resolve_features(df: pd.DataFrame, min_coverage: float = 0.5) -> list[str]:
    """Predictores realmente disponibles: presentes, con cobertura y variabilidad.

    Evita el error científico de imputar una variable ausente y presentarla como
    si hubiera sido medida.
    """
    features: list[str] = []
    for col in FEATURES + ADDITIONAL_FEATURES:
        if col not in df.columns:
            continue
        series = df[col]
        if series.notna().mean() < min_coverage:
            continue
        non_missing = series.dropna()
        if non_missing.nunique() < 2:
            continue
        features.append(col)
    return features


def dataset_quality(df: pd.DataFrame) -> dict[str, Any]:
    from .demo_data import quality_report

    report = quality_report(df)
    report["source_rows"] = int(len(df))
    report["available_features"] = resolve_features(df)
    report["unavailable_features"] = [f for f in FEATURES if f not in resolve_features(df)]
    report["events"] = int(df["event"].sum()) if "event" in df.columns else None
    report["event_prevalence"] = (
        round(float(df["event"].mean()), 4) if "event" in df.columns else None
    )
    return report

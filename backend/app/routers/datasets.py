from __future__ import annotations

import hashlib
from io import BytesIO
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session

from ..audit import log_action
from ..config import DATA_DIR
from ..database import get_db
from ..demo_data import (
    FEATURE_DESCRIPTIONS,
    FEATURE_LABELS,
    REQUIRED_COLUMNS,
    ensure_demo_dataset,
    file_checksum,
    quality_report,
)
from ..deps import require_permission
from ..models import Dataset, DatasetVariable, ModelRun, User
from ..utils import iso

router = APIRouter(prefix="/datasets", tags=["Datasets y calidad de datos"])
UPLOAD_DIR = DATA_DIR / "datasets"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


def dataset_payload(d: Dataset) -> dict:
    return {
        "id": d.id,
        "name": d.name,
        "source": d.source,
        "license": d.license,
        "description": d.description,
        "version": d.version,
        "is_demo": d.is_demo,
        "status": d.status,
        "rows_count": d.rows_count,
        "columns_count": d.columns_count,
        "checksum": d.checksum,
        "quality": d.quality,
        "created_at": iso(d.created_at),
        "updated_at": iso(d.updated_at),
        "download_url": f"/api/datasets/{d.id}/download" if d.file_path else None,
    }


def _variables_from_frame(db: Session, dataset: Dataset, df: pd.DataFrame) -> None:
    db.query(DatasetVariable).filter(DatasetVariable.dataset_id == dataset.id).delete()
    for col in df.columns:
        series = df[col]
        numeric = pd.to_numeric(series, errors="coerce")
        is_numeric = series.notna().any() and numeric.notna().sum() == series.notna().sum()
        db.add(
            DatasetVariable(
                dataset_id=dataset.id,
                name=col,
                label=FEATURE_LABELS.get(col, col.replace("_", " ").capitalize()),
                dtype=str(series.dtype),
                description=FEATURE_DESCRIPTIONS.get(col),
                missing_count=int(series.isna().sum()),
                missing_pct=round(float(series.isna().mean() * 100), 2),
                min_value=float(numeric.min()) if is_numeric else None,
                max_value=float(numeric.max()) if is_numeric else None,
                mean_value=float(numeric.mean()) if is_numeric else None,
                unique_count=int(series.nunique(dropna=True)),
            )
        )


@router.get("")
def list_datasets(
    user: User = Depends(require_permission("datasets", "read")),
    db: Session = Depends(get_db),
):
    rows = db.query(Dataset).order_by(Dataset.created_at.desc()).all()
    return [dataset_payload(d) for d in rows]


@router.post("/demo")
def create_demo_dataset(
    request: Request,
    user: User = Depends(require_permission("datasets", "write")),
    db: Session = Depends(get_db),
):
    existing = db.query(Dataset).filter(Dataset.name == "Demostración sintética CARDIOAI").one_or_none()
    if existing:
        return dataset_payload(existing)
    path, df = ensure_demo_dataset()
    dataset = Dataset(
        name="Demostración sintética CARDIOAI",
        source="Generado por CARDIOAI (semilla fija)",
        license="Uso exclusivo de demostración",
        description="Cohorte sintética para demostración técnica. No son datos reales.",
        version="1.0",
        is_demo=True,
        status="ready",
        rows_count=len(df),
        columns_count=df.shape[1],
        file_path=str(path),
        checksum=file_checksum(path),
        quality=quality_report(df),
        uploaded_by=user.id,
    )
    db.add(dataset)
    db.flush()
    _variables_from_frame(db, dataset, df)
    log_action(
        db, user=user, action="crear_dataset_demo", module="datasets", entity="dataset",
        entity_id=dataset.id, request=request,
    )
    db.commit()
    return dataset_payload(dataset)


@router.post("/upload", status_code=201)
async def upload_dataset(
    request: Request,
    file: UploadFile = File(...),
    name: str = Form(...),
    source: str = Form("Carga manual"),
    license: str = Form("No especificada"),
    description: str = Form(""),
    is_demo: bool = Form(False),
    user: User = Depends(require_permission("datasets", "write")),
    db: Session = Depends(get_db),
):
    if db.query(Dataset).filter(Dataset.name == name.strip()).first():
        raise HTTPException(409, "Ya existe un dataset con ese nombre.")
    content = await file.read()
    if len(content) > 60 * 1024 * 1024:
        raise HTTPException(413, "El archivo supera el límite de 60 MB.")
    suffix = Path(file.filename or "").suffix.lower()
    try:
        if suffix == ".csv":
            try:
                df = pd.read_csv(BytesIO(content), encoding="utf-8")
            except UnicodeDecodeError:
                df = pd.read_csv(BytesIO(content), encoding="latin-1")
        elif suffix == ".xlsx":
            df = pd.read_excel(BytesIO(content))
        else:
            raise HTTPException(415, "Formato no soportado. Use CSV o XLSX.")
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(422, f"No se pudo leer el archivo: {exc}") from exc

    df.columns = [str(c).strip() for c in df.columns]
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise HTTPException(
            422,
            "Faltan columnas obligatorias: " + ", ".join(missing) +
            ". Consulte el diccionario y el esquema esperado en la documentación.",
        )
    if len(df) < 50:
        raise HTTPException(422, "Se requieren al menos 50 registros para un dataset analizable.")
    if "cohort_year" not in df.columns:
        df["cohort_year"] = 2015
    # Normaliza booleanos textuales.
    for col in ["urban", "diabetes", "hypertension", "family_history", "event"]:
        if df[col].dtype == object:
            df[col] = df[col].map(
                lambda v: True if str(v).lower() in {"true", "1", "si", "sí", "yes"} else
                False if str(v).lower() in {"false", "0", "no"} else v
            )

    checksum = hashlib.sha256(content).hexdigest()
    original = UPLOAD_DIR / f"{checksum[:12]}_original_{Path(file.filename or 'dataset').name}"
    original.write_bytes(content)
    target = UPLOAD_DIR / f"{checksum[:12]}_normalized.csv"
    df.to_csv(target, index=False)

    dataset = Dataset(
        name=name.strip(),
        source=source.strip(),
        license=license.strip(),
        description=description.strip() or None,
        version="1.0",
        is_demo=is_demo,
        status="ready",
        rows_count=int(len(df)),
        columns_count=int(df.shape[1]),
        file_path=str(target),
        checksum=checksum,
        quality=quality_report(df),
        uploaded_by=user.id,
    )
    db.add(dataset)
    db.flush()
    _variables_from_frame(db, dataset, df)
    log_action(
        db, user=user, action="subir_dataset", module="datasets", entity="dataset",
        entity_id=dataset.id, detail=f"{len(df)} filas, {df.shape[1]} columnas", request=request,
    )
    db.commit()
    return dataset_payload(dataset)


@router.get("/{dataset_id}")
def get_dataset(
    dataset_id: int,
    user: User = Depends(require_permission("datasets", "read")),
    db: Session = Depends(get_db),
):
    dataset = db.get(Dataset, dataset_id)
    if dataset is None:
        raise HTTPException(404, "Dataset no encontrado.")
    payload = dataset_payload(dataset)
    payload["variables_count"] = (
        db.query(DatasetVariable).filter(DatasetVariable.dataset_id == dataset_id).count()
    )
    payload["models_trained"] = (
        db.query(ModelRun).filter(ModelRun.dataset_id == dataset_id).count()
    )
    return payload


@router.get("/{dataset_id}/variables")
def get_variables(
    dataset_id: int,
    user: User = Depends(require_permission("datasets", "read")),
    db: Session = Depends(get_db),
):
    if db.get(Dataset, dataset_id) is None:
        raise HTTPException(404, "Dataset no encontrado.")
    rows = (
        db.query(DatasetVariable)
        .filter(DatasetVariable.dataset_id == dataset_id)
        .order_by(DatasetVariable.id)
        .all()
    )
    return [
        {
            "name": v.name,
            "label": v.label,
            "dtype": v.dtype,
            "unit": v.unit,
            "description": v.description,
            "missing_count": v.missing_count,
            "missing_pct": v.missing_pct,
            "min": v.min_value,
            "max": v.max_value,
            "mean": v.mean_value,
            "unique": v.unique_count,
        }
        for v in rows
    ]


@router.get("/{dataset_id}/quality")
def get_quality(
    dataset_id: int,
    user: User = Depends(require_permission("datasets", "read")),
    db: Session = Depends(get_db),
):
    dataset = db.get(Dataset, dataset_id)
    if dataset is None:
        raise HTTPException(404, "Dataset no encontrado.")
    return {"dataset_id": dataset.id, "name": dataset.name, "quality": dataset.quality}


@router.get("/{dataset_id}/preview")
def preview_dataset(
    dataset_id: int,
    limit: int = 20,
    user: User = Depends(require_permission("datasets", "read")),
    db: Session = Depends(get_db),
):
    dataset = db.get(Dataset, dataset_id)
    if dataset is None or not dataset.file_path:
        raise HTTPException(404, "Dataset no encontrado.")
    df = pd.read_csv(dataset.file_path).head(max(1, min(limit, 100)))
    df = df.replace({np.nan: None})
    return {"columns": list(df.columns), "rows": df.to_dict(orient="records")}


@router.get("/{dataset_id}/download")
def download_dataset(
    dataset_id: int,
    user: User = Depends(require_permission("datasets", "read")),
    db: Session = Depends(get_db),
):
    dataset = db.get(Dataset, dataset_id)
    if dataset is None or not dataset.file_path or not Path(dataset.file_path).exists():
        raise HTTPException(404, "Archivo no encontrado.")
    return FileResponse(
        dataset.file_path,
        filename=f"{dataset.name.replace(' ', '_')}.csv",
        media_type="text/csv",
    )

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Request
from sqlalchemy.orm import Session

from ..audit import log_action
from ..config import DATA_DIR
from ..database import get_db
from ..demo_data import file_checksum
from ..deps import require_permission
from ..models import Dataset, User
from ..public_datasets import (
    FRAMINGHAM,
    SPECS,
    adapt_framingham,
    dataset_quality,
    ensure_public_file,
    load_framingham,
    resolve_features,
)
from .datasets import _variables_from_frame, dataset_payload

router = APIRouter(prefix="/public-datasets", tags=["Datasets públicos"])

ADAPTED_DIR = DATA_DIR / "datasets"
ADAPTED_DIR.mkdir(parents=True, exist_ok=True)


@router.get("")
def list_public_datasets(user: User = Depends(require_permission("datasets", "read"))):
    """Catálogo de fuentes públicas soportadas, con procedencia y limitaciones."""
    return [
        {
            **spec.metadata,
            "expected_rows": spec.expected_rows,
            "expected_columns": spec.expected_columns,
            "downloaded": spec.path.exists(),
            "description": spec.description,
        }
        for spec in SPECS.values()
    ]


@router.post("/{key}/register", status_code=201)
def register_public_dataset(
    key: str,
    request: Request,
    force: bool = False,
    user: User = Depends(require_permission("datasets", "write")),
    db: Session = Depends(get_db),
):
    """Descarga, valida por SHA-256, adapta y registra un dataset público."""
    if key != FRAMINGHAM.key:
        raise HTTPException(404, "Dataset público no soportado.")

    dataset_name = "Framingham Heart Study · CHD a 10 años"
    existing = db.query(Dataset).filter(Dataset.name == dataset_name).one_or_none()
    if existing and not force:
        return dataset_payload(existing)

    try:
        original_path = ensure_public_file(FRAMINGHAM, force=force)
        _, raw = load_framingham(force_download=False)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(
            503, f"No se pudo obtener el dataset público verificado: {exc}"
        ) from exc

    adapted = adapt_framingham(raw)
    checksum = file_checksum(original_path)
    if checksum != FRAMINGHAM.sha256:
        raise HTTPException(422, "La huella SHA-256 del archivo no coincide con la esperada.")

    target = ADAPTED_DIR / f"{FRAMINGHAM.key}_cardioai.csv"
    adapted.to_csv(target, index=False)

    quality = dataset_quality(adapted)
    quality["provenance"] = {
        **FRAMINGHAM.metadata,
        "original_file": original_path.name,
        "adapted_file": target.name,
        "adapted_sha256": file_checksum(target),
        "features_resolved": resolve_features(adapted),
        "design_note": (
            "Una sola región (Framingham, Massachusetts): no es posible una retención "
            "geográfica. El desenlace es administrativo a 10 años, sin tiempos "
            "individuales de censura, por lo que el C-index no es informativo."
        ),
    }

    fields = dict(
        source=FRAMINGHAM.source,
        license=FRAMINGHAM.license,
        description=FRAMINGHAM.description,
        version="1.0",
        is_demo=False,
        status="ready",
        rows_count=int(len(adapted)),
        columns_count=int(adapted.shape[1]),
        file_path=str(target),
        checksum=checksum,
        quality=quality,
        uploaded_by=user.id,
    )
    if existing:
        for field, value in fields.items():
            setattr(existing, field, value)
        dataset = existing
    else:
        dataset = Dataset(name=dataset_name, **fields)
        db.add(dataset)
    db.flush()
    _variables_from_frame(db, dataset, adapted)
    log_action(
        db,
        user=user,
        action="registrar_dataset_publico",
        module="datasets",
        entity="dataset",
        entity_id=dataset.id,
        detail=(
            f"{FRAMINGHAM.key} sha256={checksum} filas={len(adapted)} "
            f"columnas={adapted.shape[1]}"
        ),
        request=request,
    )
    db.commit()
    return dataset_payload(dataset)

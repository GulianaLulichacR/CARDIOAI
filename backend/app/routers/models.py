from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from ..audit import log_action
from ..database import get_db
from ..deps import require_permission
from ..ml_engine import ALGORITHMS, TORCH_AVAILABLE, start_training
from ..models import ModelRun, User
from ..utils import iso

router = APIRouter(prefix="/models", tags=["Modelos y MLOps"])


class TrainRequest(BaseModel):
    dataset_id: int | None = None
    trials: int = Field(default=8, ge=1, le=30)


def model_payload(run: ModelRun) -> dict:
    disc = (run.metrics or {}).get("discrimination", {})
    return {
        "id": run.id,
        "name": run.name,
        "algorithm": run.algorithm,
        "version": run.version,
        "status": run.status,
        "is_demo": run.is_demo,
        "is_active": run.is_active,
        "dataset_id": run.dataset_id,
        "horizon_years": run.horizon_years,
        "metrics": run.metrics,
        "summary": {
            "roc_auc": disc.get("roc_auc"),
            "f1": disc.get("f1"),
            "mcc": disc.get("mcc"),
            "sensitivity": disc.get("sensitivity"),
            "specificity": disc.get("specificity"),
            "brier": disc.get("brier"),
            "pr_auc": disc.get("pr_auc"),
        },
        "params": run.params,
        "feature_importance": run.feature_importance,
        "risk_thresholds": run.risk_thresholds,
        "notes": run.notes,
        "error": run.error,
        "mlflow_run_id": run.mlflow_run_id,
        "trained_at": iso(run.trained_at),
        "created_at": iso(run.created_at),
        "duration_seconds": run.duration_seconds,
        "train_rows": run.train_rows,
        "test_rows": run.test_rows,
        "validation_strategy": run.validation_strategy,
    }


@router.get("/algorithms")
def algorithms(user: User = Depends(require_permission("models", "read"))):
    return [
        {
            **algo,
            "available": True if algo["key"] != "lstm_attention_cox" else TORCH_AVAILABLE,
            "requires": "PyTorch" if algo["key"] == "lstm_attention_cox" else "Scikit-learn/XGBoost/lifelines",
        }
        for algo in ALGORITHMS
    ]


@router.get("/compare")
def compare_models(
    user: User = Depends(require_permission("models", "read")),
    db: Session = Depends(get_db),
):
    runs = db.query(ModelRun).filter(ModelRun.status == "ready").order_by(ModelRun.trained_at.desc()).all()
    latest: dict[str, ModelRun] = {}
    for run in runs:
        latest.setdefault(run.algorithm, run)
    payload = [model_payload(r) for r in latest.values()]
    payload.sort(key=lambda r: (r["summary"]["roc_auc"] or 0), reverse=True)
    return payload


@router.get("")
def list_models(
    algorithm: str | None = None,
    status_filter: str | None = Query(default=None, alias="status"),
    page: int = 1,
    page_size: int = 50,
    user: User = Depends(require_permission("models", "read")),
    db: Session = Depends(get_db),
):
    query = db.query(ModelRun)
    if algorithm:
        query = query.filter(ModelRun.algorithm == algorithm)
    if status_filter:
        query = query.filter(ModelRun.status == status_filter)
    total = query.count()
    runs = query.order_by(ModelRun.created_at.desc()).offset((page - 1) * page_size).limit(page_size).all()
    return {"items": [model_payload(r) for r in runs], "total": total, "page": page, "page_size": page_size}


@router.post("/train", status_code=202)
def train_models(
    payload: TrainRequest,
    request: Request,
    user: User = Depends(require_permission("models", "write")),
    db: Session = Depends(get_db),
):
    active = db.query(ModelRun).filter(ModelRun.status.in_(["queued", "running"])).count()
    if active:
        raise HTTPException(409, "Ya hay un entrenamiento en curso. Espere a que finalice.")
    try:
        runs = start_training(db, payload.dataset_id, payload.trials, user.id)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    log_action(
        db,
        user=user,
        action="iniciar_entrenamiento",
        module="models",
        entity="entrenamiento",
        entity_id=",".join(str(r.id) for r in runs),
        detail=f"{len(runs)} modelos, {payload.trials} trials, dataset={payload.dataset_id or 'demo'}",
        request=request,
    )
    db.commit()
    return {
        "message": "Entrenamiento iniciado en segundo plano.",
        "runs": [model_payload(r) for r in runs],
    }


@router.get("/{model_id}")
def get_model(
    model_id: int,
    user: User = Depends(require_permission("models", "read")),
    db: Session = Depends(get_db),
):
    run = db.get(ModelRun, model_id)
    if run is None:
        raise HTTPException(404, "Modelo no encontrado.")
    return model_payload(run)


@router.post("/{model_id}/activate")
def activate_model(
    model_id: int,
    request: Request,
    user: User = Depends(require_permission("models", "write")),
    db: Session = Depends(get_db),
):
    run = db.get(ModelRun, model_id)
    if run is None:
        raise HTTPException(404, "Modelo no encontrado.")
    if run.status != "ready" or not run.artifact_path:
        raise HTTPException(400, "Solo se puede activar un modelo entrenado correctamente.")
    db.query(ModelRun).update({ModelRun.is_active: False})
    run.is_active = True
    log_action(
        db, user=user, action="activar_modelo", module="models", entity="modelo",
        entity_id=run.id, detail=f"{run.name} ({run.algorithm})", request=request,
    )
    db.commit()
    return model_payload(run)


@router.get("/active/current")
def active_model(
    user: User = Depends(require_permission("models", "read")),
    db: Session = Depends(get_db),
):
    run = db.query(ModelRun).filter(ModelRun.is_active.is_(True)).order_by(ModelRun.trained_at.desc()).first()
    if run is None:
        return {"model": None, "message": "No hay modelo activo. Entrene o active un modelo."}
    return {"model": model_payload(run)}

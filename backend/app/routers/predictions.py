from __future__ import annotations

import uuid
from io import BytesIO
from typing import Any

import pandas as pd
from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile
from pydantic import BaseModel, Field
from sqlalchemy import func
from sqlalchemy.orm import Session

from ..audit import log_action
from ..config import settings
from ..database import get_db
from ..deps import require_permission
from ..ml_engine import explain_prediction, load_artifact, predict_records, risk_level
from ..models import Alert, ModelRun, Patient, Prediction, User
from ..utils import iso, paginate

router = APIRouter(prefix="/predictions", tags=["Predicción y estratificación"])

DISCLAIMER = (
    "Resultado de investigación generado por un modelo estadístico. No es un diagnóstico médico "
    "ni sustituye la valoración de un profesional de la salud. La probabilidad depende del modelo, "
    "del horizonte y de la calidad de los datos."
)


class PredictionInput(BaseModel):
    age: int = Field(ge=18, le=110)
    sex: str = Field(pattern="^(F|M|Otro)$")
    region: str
    urban: bool = True
    education: str = "secundaria"
    income_level: str = "medio"
    smoking: str = "nunca"
    physical_activity_min: float = Field(default=150, ge=0, le=1000)
    alcohol_units: float = Field(default=0, ge=0, le=100)
    bmi: float = Field(ge=12, le=70)
    systolic_bp: float = Field(ge=60, le=260)
    diastolic_bp: float = Field(default=80, ge=30, le=180)
    cholesterol_total: float = Field(ge=50, le=600)
    hdl: float = Field(default=50, ge=10, le=150)
    glucose: float = Field(ge=30, le=600)
    diabetes: bool = False
    hypertension: bool = False
    family_history: bool = False
    pm25: float = Field(default=12, ge=0, le=200)
    green_space: float = Field(default=20, ge=0, le=100)
    stress_level: int = Field(default=3, ge=1, le=10)
    patient_id: int | None = None
    model_run_id: int | None = None
    persist: bool = True
    explain: bool = True


def _get_model(db: Session, model_run_id: int | None) -> ModelRun:
    if model_run_id:
        run = db.get(ModelRun, model_run_id)
        if run is None:
            raise HTTPException(404, "Modelo no encontrado.")
    else:
        run = db.query(ModelRun).filter(ModelRun.is_active.is_(True)).order_by(ModelRun.trained_at.desc()).first()
    if run is None:
        raise HTTPException(
            409,
            "No hay modelo activo. Entrene los modelos desde el módulo de Modelos o active uno existente.",
        )
    if run.status != "ready" or not run.artifact_path:
        raise HTTPException(409, "El modelo seleccionado no está listo para inferencia.")
    return run


def _model_brief(run: ModelRun) -> dict[str, Any]:
    return {
        "id": run.id,
        "name": run.name,
        "algorithm": run.algorithm,
        "version": run.version,
        "is_demo": run.is_demo,
        "horizon_years": run.horizon_years,
    }


def _prediction_payload(p: Prediction, run: ModelRun | None = None) -> dict[str, Any]:
    return {
        "id": p.id,
        "patient_id": p.patient_id,
        "model_run_id": p.model_run_id,
        "model": _model_brief(run) if run else None,
        "probability": p.probability,
        "risk_level": p.risk_level,
        "horizon_years": p.horizon_years,
        "threshold": p.threshold,
        "explanation": p.explanation,
        "explanation_method": p.explanation_method,
        "is_demo": p.is_demo,
        "created_at": iso(p.created_at),
        "disclaimer": DISCLAIMER,
    }


def _patient_record(patient: Patient) -> dict[str, Any]:
    return {
        "age": patient.age,
        "sex": patient.sex,
        "region": patient.region,
        "urban": patient.urban,
        "education": patient.education,
        "income_level": patient.income_level,
        "smoking": patient.smoking,
        "physical_activity_min": patient.physical_activity_min,
        "alcohol_units": patient.alcohol_units,
        "bmi": patient.bmi,
        "systolic_bp": patient.systolic_bp,
        "diastolic_bp": patient.diastolic_bp,
        "cholesterol_total": patient.cholesterol_total,
        "hdl": patient.hdl,
        "glucose": patient.glucose,
        "diabetes": patient.diabetes,
        "hypertension": patient.hypertension,
        "family_history": patient.family_history,
        "pm25": patient.pm25,
        "green_space": patient.green_space,
        "stress_level": patient.stress_level,
    }


@router.post("")
def create_prediction(
    payload: PredictionInput,
    request: Request,
    user: User = Depends(require_permission("predictions", "write")),
    db: Session = Depends(get_db),
):
    run = _get_model(db, payload.model_run_id)
    patient = None
    if payload.patient_id:
        patient = db.get(Patient, payload.patient_id)
        if patient is None or not patient.is_active:
            raise HTTPException(404, "Paciente no encontrado.")
    try:
        artifact = load_artifact(run.artifact_path)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(500, f"No se pudo cargar el artefacto del modelo: {exc}") from exc

    record = payload.model_dump(
        exclude={"patient_id", "model_run_id", "persist", "explain"}
    )
    if patient:
        record = {**_patient_record(patient), **{k: v for k, v in record.items() if v is not None}}
    probability = float(predict_records(artifact, [record])[0])
    thresholds = run.risk_thresholds or {}
    low_max = float(thresholds.get("low_max", settings.risk_low_max))
    high_min = float(thresholds.get("high_min", settings.risk_high_min))
    level = "bajo" if probability < low_max else ("moderado" if probability < high_min else "alto")

    explanation: dict[str, Any] = {}
    if payload.explain:
        explanation = explain_prediction(artifact, record, probability)
    else:
        explanation = {"method": "No solicitada", "factors": []}

    prediction = Prediction(
        patient_id=payload.patient_id,
        model_run_id=run.id,
        input=record,
        probability=round(probability, 4),
        risk_level=level,
        horizon_years=run.horizon_years or settings.prediction_horizon_years,
        threshold=float(artifact.get("threshold") or 0.5),
        explanation=explanation,
        explanation_method=str(explanation.get("method", "No disponible")),
        is_demo=run.is_demo,
        created_by=user.id,
    )
    if payload.persist:
        db.add(prediction)
        db.flush()
        if level == "alto":
            db.add(
                Alert(
                    type="prediccion",
                    severity="warning",
                    title="Predicción de riesgo alto",
                    message=(
                        f"Se registró una predicción de riesgo alto ({probability:.1%}) con {run.name}. "
                        "Resultado de investigación; requiere interpretación profesional."
                    ),
                    module="predictions",
                )
            )
    log_action(
        db, user=user, action="ejecutar_prediccion", module="predictions", entity="prediccion",
        entity_id=prediction.id if payload.persist else None,
        detail=f"Modelo {run.name}; probabilidad={probability:.4f}; nivel={level}",
        request=request,
    )
    db.commit()
    result = _prediction_payload(prediction, run)
    result["input"] = record
    return result


@router.post("/batch")
async def batch_prediction(
    request: Request,
    file: UploadFile = File(...),
    model_run_id: int | None = Form(default=None),
    user: User = Depends(require_permission("predictions", "write")),
    db: Session = Depends(get_db),
):
    run = _get_model(db, model_run_id)
    if not (file.filename or "").lower().endswith(".csv"):
        raise HTTPException(415, "Envíe un archivo CSV.")
    content = await file.read()
    try:
        df = pd.read_csv(BytesIO(content))
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(422, f"No se pudo leer el CSV: {exc}") from exc
    if len(df) > 1000:
        raise HTTPException(422, "Máximo 1000 registros por lote.")
    required = ["age", "sex", "region", "bmi", "systolic_bp", "cholesterol_total", "glucose"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise HTTPException(422, "Faltan columnas: " + ", ".join(missing))
    records = df.replace({float("nan"): None}).to_dict(orient="records")
    try:
        artifact = load_artifact(run.artifact_path)
        probabilities = predict_records(artifact, records)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(500, f"Error de inferencia: {exc}") from exc

    thresholds = run.risk_thresholds or {}
    low_max = float(thresholds.get("low_max", settings.risk_low_max))
    high_min = float(thresholds.get("high_min", settings.risk_high_min))
    batch_id = uuid.uuid4().hex[:12]
    created = []
    for i, (record, probability) in enumerate(zip(records, probabilities)):
        prob = float(probability)
        level = "bajo" if prob < low_max else ("moderado" if prob < high_min else "alto")
        pred = Prediction(
            model_run_id=run.id,
            input=record,
            probability=round(prob, 4),
            risk_level=level,
            horizon_years=run.horizon_years or settings.prediction_horizon_years,
            threshold=float(artifact.get("threshold") or 0.5),
            explanation={"method": "No calculada en lote", "factors": []},
            explanation_method="No calculada en lote",
            is_demo=run.is_demo,
            batch_id=batch_id,
            created_by=user.id,
        )
        db.add(pred)
        created.append((record, pred))
    db.flush()
    log_action(
        db, user=user, action="prediccion_por_lote", module="predictions", entity="lote",
        entity_id=batch_id, detail=f"{len(created)} registros con {run.name}", request=request,
    )
    db.commit()
    distribution = {
        level: sum(1 for _, p in created if p.risk_level == level)
        for level in ["bajo", "moderado", "alto"]
    }
    return {
        "batch_id": batch_id,
        "model": _model_brief(run),
        "total": len(created),
        "distribution": distribution,
        "mean_probability": round(float(sum(p.probability for _, p in created) / max(1, len(created))), 4),
        "preview": [
            {
                "row": i + 1,
                "probability": p.probability,
                "risk_level": p.risk_level,
                "input": record,
            }
            for i, (record, p) in enumerate(created[:100])
        ],
        "disclaimer": DISCLAIMER,
    }


@router.get("")
def list_predictions(
    risk_level_filter: str | None = None,
    model_run_id: int | None = None,
    patient_id: int | None = None,
    page: int = 1,
    page_size: int = 25,
    user: User = Depends(require_permission("predictions", "read")),
    db: Session = Depends(get_db),
):
    query = db.query(Prediction)
    if risk_level_filter:
        query = query.filter(Prediction.risk_level == risk_level_filter)
    if model_run_id:
        query = query.filter(Prediction.model_run_id == model_run_id)
    if patient_id:
        query = query.filter(Prediction.patient_id == patient_id)
    query = query.order_by(Prediction.created_at.desc())
    items, total = paginate(query, page, page_size)
    runs = {r.id: r for r in db.query(ModelRun).filter(
        ModelRun.id.in_({p.model_run_id for p in items})
    ).all()}
    return {
        "items": [_prediction_payload(p, runs.get(p.model_run_id)) for p in items],
        "total": total,
        "page": page,
        "page_size": page_size,
    }


@router.get("/{prediction_id}")
def get_prediction(
    prediction_id: int,
    user: User = Depends(require_permission("predictions", "read")),
    db: Session = Depends(get_db),
):
    pred = db.get(Prediction, prediction_id)
    if pred is None:
        raise HTTPException(404, "Predicción no encontrada.")
    run = db.get(ModelRun, pred.model_run_id)
    result = _prediction_payload(pred, run)
    result["input"] = pred.input
    return result

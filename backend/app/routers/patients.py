from __future__ import annotations

import csv
import io
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from sqlalchemy import func, or_
from sqlalchemy.orm import Session

from ..audit import log_action
from ..database import get_db
from ..demo_data import REGION_NAMES
from ..deps import require_permission
from ..models import Patient, Prediction, User
from ..utils import iso, paginate

router = APIRouter(prefix="/patients", tags=["Pacientes y registros"])


class PatientInput(BaseModel):
    age: int = Field(ge=18, le=110)
    sex: str = Field(pattern="^(F|M|Otro)$")
    region: str
    urban: bool = True
    education: str = Field(default="secundaria", pattern="^(primaria|secundaria|tecnica|universitaria)$")
    income_level: str = Field(default="medio", pattern="^(bajo|medio|alto)$")
    smoking: str = Field(default="nunca", pattern="^(nunca|exfumador|fumador)$")
    physical_activity_min: int = Field(default=150, ge=0, le=1000)
    alcohol_units: float = Field(default=0, ge=0, le=100)
    bmi: float = Field(ge=12, le=70)
    systolic_bp: int = Field(ge=60, le=260)
    diastolic_bp: int = Field(ge=30, le=180)
    cholesterol_total: float = Field(ge=50, le=600)
    hdl: float = Field(ge=10, le=150)
    glucose: float = Field(ge=30, le=600)
    diabetes: bool = False
    hypertension: bool = False
    family_history: bool = False
    pm25: float = Field(default=12, ge=0, le=200)
    green_space: float = Field(default=20, ge=0, le=100)
    stress_level: int = Field(default=3, ge=1, le=10)
    event: bool | None = None
    followup_years: float | None = Field(default=None, ge=0, le=60)
    cohort_year: int | None = Field(default=None, ge=1990, le=2035)
    notes: str | None = Field(default=None, max_length=2000)


class PatientUpdate(BaseModel):
    age: int | None = Field(default=None, ge=18, le=110)
    sex: str | None = Field(default=None, pattern="^(F|M|Otro)$")
    region: str | None = None
    urban: bool | None = None
    education: str | None = None
    income_level: str | None = None
    smoking: str | None = None
    physical_activity_min: int | None = Field(default=None, ge=0, le=1000)
    alcohol_units: float | None = Field(default=None, ge=0, le=100)
    bmi: float | None = Field(default=None, ge=12, le=70)
    systolic_bp: int | None = Field(default=None, ge=60, le=260)
    diastolic_bp: int | None = Field(default=None, ge=30, le=180)
    cholesterol_total: float | None = Field(default=None, ge=50, le=600)
    hdl: float | None = Field(default=None, ge=10, le=150)
    glucose: float | None = Field(default=None, ge=30, le=600)
    diabetes: bool | None = None
    hypertension: bool | None = None
    family_history: bool | None = None
    pm25: float | None = Field(default=None, ge=0, le=200)
    green_space: float | None = Field(default=None, ge=0, le=100)
    stress_level: int | None = Field(default=None, ge=1, le=10)
    notes: str | None = Field(default=None, max_length=2000)


def patient_payload(p: Patient, last_prediction: Prediction | None = None) -> dict:
    return {
        "id": p.id,
        "code": p.code,
        "age": p.age,
        "sex": p.sex,
        "region": p.region,
        "urban": p.urban,
        "education": p.education,
        "income_level": p.income_level,
        "smoking": p.smoking,
        "physical_activity_min": p.physical_activity_min,
        "alcohol_units": p.alcohol_units,
        "bmi": p.bmi,
        "systolic_bp": p.systolic_bp,
        "diastolic_bp": p.diastolic_bp,
        "cholesterol_total": p.cholesterol_total,
        "hdl": p.hdl,
        "glucose": p.glucose,
        "diabetes": p.diabetes,
        "hypertension": p.hypertension,
        "family_history": p.family_history,
        "pm25": p.pm25,
        "green_space": p.green_space,
        "stress_level": p.stress_level,
        "event": p.event,
        "followup_years": p.followup_years,
        "cohort_year": p.cohort_year,
        "notes": p.notes,
        "is_demo": p.is_demo,
        "is_active": p.is_active,
        "created_at": iso(p.created_at),
        "updated_at": iso(p.updated_at),
        "last_prediction": None
        if last_prediction is None
        else {
            "id": last_prediction.id,
            "probability": last_prediction.probability,
            "risk_level": last_prediction.risk_level,
            "created_at": iso(last_prediction.created_at),
        },
    }


def _validate_region(region: str) -> None:
    if region not in REGION_NAMES:
        raise HTTPException(422, f"Región no reconocida. Opciones: {', '.join(REGION_NAMES)}")


def _last_prediction(db: Session, patient_id: int) -> Prediction | None:
    return (
        db.query(Prediction)
        .filter(Prediction.patient_id == patient_id)
        .order_by(Prediction.created_at.desc())
        .first()
    )


@router.get("/meta/options")
def options(user: User = Depends(require_permission("patients", "read"))):
    return {
        "regions": REGION_NAMES,
        "sex": ["F", "M", "Otro"],
        "education": ["primaria", "secundaria", "tecnica", "universitaria"],
        "income_level": ["bajo", "medio", "alto"],
        "smoking": ["nunca", "exfumador", "fumador"],
    }


@router.get("/export.csv")
def export_csv(
    user: User = Depends(require_permission("patients", "read")),
    db: Session = Depends(get_db),
):
    rows = db.query(Patient).filter(Patient.is_active.is_(True)).order_by(Patient.id).all()
    buffer = io.StringIO()
    fields = [
        "code", "age", "sex", "region", "urban", "education", "income_level", "smoking",
        "physical_activity_min", "alcohol_units", "bmi", "systolic_bp", "diastolic_bp",
        "cholesterol_total", "hdl", "glucose", "diabetes", "hypertension", "family_history",
        "pm25", "green_space", "stress_level", "event", "followup_years", "cohort_year",
    ]
    writer = csv.DictWriter(buffer, fieldnames=fields)
    writer.writeheader()
    for p in rows:
        writer.writerow({f: getattr(p, f) for f in fields})
    log_action(db, user=user, action="exportar_pacientes", module="patients", detail=f"{len(rows)} registros")
    db.commit()
    return StreamingResponse(
        iter([buffer.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": 'attachment; filename="pacientes_cardioai.csv"'},
    )


@router.get("")
def list_patients(
    q: str | None = None,
    region: str | None = None,
    sex: str | None = None,
    risk: str | None = None,
    include_inactive: bool = False,
    page: int = 1,
    page_size: int = 25,
    user: User = Depends(require_permission("patients", "read")),
    db: Session = Depends(get_db),
):
    query = db.query(Patient)
    if not include_inactive:
        query = query.filter(Patient.is_active.is_(True))
    if q:
        like = f"%{q.strip()}%"
        query = query.filter(or_(Patient.code.ilike(like), Patient.region.ilike(like)))
    if region:
        query = query.filter(Patient.region == region)
    if sex:
        query = query.filter(Patient.sex == sex)
    query = query.order_by(Patient.id.desc())
    items, total = paginate(query, page, page_size)
    return {
        "items": [patient_payload(p, _last_prediction(db, p.id)) for p in items],
        "total": total,
        "page": page,
        "page_size": page_size,
    }


@router.post("", status_code=201)
def create_patient(
    payload: PatientInput,
    request: Request,
    user: User = Depends(require_permission("patients", "write")),
    db: Session = Depends(get_db),
):
    _validate_region(payload.region)
    last_id = db.query(func.max(Patient.id)).scalar() or 0
    code = f"PAC-{datetime.utcnow().year}-{last_id + 1:04d}"
    while db.query(Patient).filter(Patient.code == code).first():
        last_id += 1
        code = f"PAC-{datetime.utcnow().year}-{last_id + 1:04d}"
    patient = Patient(
        code=code,
        **payload.model_dump(),
        is_demo=False,
        created_by=user.id,
    )
    db.add(patient)
    db.flush()
    log_action(
        db, user=user, action="crear_paciente", module="patients", entity="paciente",
        entity_id=patient.id, detail=f"Código {patient.code}", request=request,
    )
    db.commit()
    return patient_payload(patient)


@router.get("/{patient_id}")
def get_patient(
    patient_id: int,
    user: User = Depends(require_permission("patients", "read")),
    db: Session = Depends(get_db),
):
    patient = db.get(Patient, patient_id)
    if patient is None:
        raise HTTPException(404, "Registro no encontrado.")
    history = (
        db.query(Prediction)
        .filter(Prediction.patient_id == patient_id)
        .order_by(Prediction.created_at.desc())
        .limit(20)
        .all()
    )
    result = patient_payload(patient, history[0] if history else None)
    result["prediction_history"] = [
        {
            "id": p.id,
            "probability": p.probability,
            "risk_level": p.risk_level,
            "horizon_years": p.horizon_years,
            "model_run_id": p.model_run_id,
            "created_at": iso(p.created_at),
        }
        for p in history
    ]
    return result


@router.patch("/{patient_id}")
def update_patient(
    patient_id: int,
    payload: PatientUpdate,
    request: Request,
    user: User = Depends(require_permission("patients", "write")),
    db: Session = Depends(get_db),
):
    patient = db.get(Patient, patient_id)
    if patient is None or not patient.is_active:
        raise HTTPException(404, "Registro no encontrado.")
    data = payload.model_dump(exclude_unset=True)
    if "region" in data:
        _validate_region(data["region"])
    for key, value in data.items():
        setattr(patient, key, value)
    patient.updated_at = datetime.utcnow()
    log_action(
        db, user=user, action="actualizar_paciente", module="patients", entity="paciente",
        entity_id=patient.id, detail=", ".join(data.keys()), request=request,
    )
    db.commit()
    return patient_payload(patient, _last_prediction(db, patient.id))


@router.delete("/{patient_id}")
def deactivate_patient(
    patient_id: int,
    request: Request,
    user: User = Depends(require_permission("patients", "write")),
    db: Session = Depends(get_db),
):
    patient = db.get(Patient, patient_id)
    if patient is None:
        raise HTTPException(404, "Registro no encontrado.")
    patient.is_active = False
    log_action(
        db, user=user, action="desactivar_paciente", module="patients", entity="paciente",
        entity_id=patient.id, detail="Desactivación lógica; no se elimina el registro.", request=request,
    )
    db.commit()
    return {"message": "Registro desactivado. La información se conserva para trazabilidad."}


@router.get("/{patient_id}/predictions")
def patient_predictions(
    patient_id: int,
    user: User = Depends(require_permission("patients", "read")),
    db: Session = Depends(get_db),
):
    if db.get(Patient, patient_id) is None:
        raise HTTPException(404, "Registro no encontrado.")
    rows = (
        db.query(Prediction)
        .filter(Prediction.patient_id == patient_id)
        .order_by(Prediction.created_at.desc())
        .all()
    )
    return [
        {
            "id": p.id,
            "probability": p.probability,
            "risk_level": p.risk_level,
            "horizon_years": p.horizon_years,
            "model_run_id": p.model_run_id,
            "explanation": p.explanation,
            "created_at": iso(p.created_at),
        }
        for p in rows
    ]

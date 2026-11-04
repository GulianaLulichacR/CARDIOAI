from __future__ import annotations

from datetime import datetime

from fastapi import APIRouter, Depends
from sqlalchemy import func, text
from sqlalchemy.orm import Session

from ..config import settings
from ..database import get_db
from ..deps import require_permission
from ..ml_engine import TORCH_AVAILABLE, OPTUNA_AVAILABLE
from ..models import Dataset, ModelRun, Patient, Prediction, User
from ..security import permissions_for
from ..utils import iso

router = APIRouter(prefix="/system", tags=["Sistema"])


@router.get("/health")
def health(db: Session = Depends(get_db)):
    database_ok = True
    try:
        db.execute(text("SELECT 1"))
    except Exception:
        database_ok = False
    return {
        "status": "ok" if database_ok else "degraded",
        "database": "ok" if database_ok else "error",
        "version": settings.app_version,
        "environment": settings.environment,
        "demo_mode": settings.seed_demo_data,
        "timestamp": iso(datetime.utcnow()),
    }


@router.get("/info")
def info(
    user: User = Depends(require_permission("dashboard", "read")),
    db: Session = Depends(get_db),
):
    active = db.query(ModelRun).filter(ModelRun.is_active.is_(True)).first()
    return {
        "app": settings.app_name,
        "version": settings.app_version,
        "environment": settings.environment,
        "database": "PostgreSQL" if settings.database_url.startswith("postgres") else "SQLite",
        "demo_mode": settings.seed_demo_data,
        "capabilities": {
            "torch": TORCH_AVAILABLE,
            "optuna": OPTUNA_AVAILABLE,
            "mlflow": settings.mlflow_enabled,
            "llm": settings.llm_provider,
        },
        "counts": {
            "users": db.query(func.count(User.id)).scalar(),
            "patients": db.query(func.count(Patient.id)).filter(Patient.is_active.is_(True)).scalar(),
            "datasets": db.query(func.count(Dataset.id)).scalar(),
            "models": db.query(func.count(ModelRun.id)).filter(ModelRun.status == "ready").scalar(),
            "predictions": db.query(func.count(Prediction.id)).scalar(),
        },
        "risk_thresholds": {
            "low_max": settings.risk_low_max,
            "high_min": settings.risk_high_min,
            "note": "Umbrales de demostración configurables; requieren validación clínica antes de uso real.",
        },
        "prediction_horizon_years": settings.prediction_horizon_years,
        "active_model": {
            "id": active.id,
            "name": active.name,
            "algorithm": active.algorithm,
            "trained_at": iso(active.trained_at),
            "is_demo": active.is_demo,
        }
        if active else None,
        "permissions": permissions_for(user.role),
    }


@router.get("/limitations")
def limitations(user: User = Depends(require_permission("dashboard", "read"))):
    return {
        "items": [
            "El dataset incluido es SINTÉTICO y se generó con una semilla fija para demostración técnica; no representa población real.",
            "Las métricas de los modelos no constituyen evidencia clínica ni han sido validadas por pares.",
            "Los umbrales de riesgo (bajo/moderado/alto) son configurables y de demostración; no provienen de una guía clínica validada para esta población.",
            "La validación es temporal y geográfica interna sobre datos sintéticos; no sustituye validación externa.",
            "No se ha realizado recalibración con datos locales ni análisis de equidad con datos reales.",
            "La recuperación de contraseña no envía correo automático: requiere intervención administrativa.",
            "La revocación de sesiones JWT no está implementada del lado servidor; el token expira según configuración.",
            "La integración con n8n, MinIO, MLflow y un LLM requiere credenciales y servicios externos; se entregan flujos y configuración, no credenciales.",
            "Las concentraciones de PM2.5 del modo demo son ilustrativas; para uso real deben provenir de fuentes oficiales con licencia.",
            "El sistema no reemplaza la evaluación de un profesional de la salud ni debe usarse para decisiones asistenciales.",
        ],
        "responsible_use": (
            "Use la plataforma para investigación, docencia y evaluación metodológica. Documente siempre "
            "la procedencia de los datos, la versión del modelo y las limitaciones antes de comunicar resultados."
        ),
    }

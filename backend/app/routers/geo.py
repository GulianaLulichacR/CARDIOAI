from __future__ import annotations

from datetime import datetime, timedelta

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from ..config import settings
from ..database import get_db
from ..demo_data import REGION_INFO, REGION_NAMES
from ..deps import require_permission
from ..models import Patient, Prediction, User
from ..utils import iso

router = APIRouter(prefix="/geo", tags=["Análisis geoespacial"])


@router.get("/regions")
def region_analysis(
    days: int = Query(default=365, ge=7, le=3650),
    user: User = Depends(require_permission("geo", "read")),
    db: Session = Depends(get_db),
):
    start = datetime.utcnow() - timedelta(days=days)
    patients = db.query(Patient).filter(Patient.is_active.is_(True)).all()
    predictions = db.query(Prediction).filter(Prediction.created_at >= start).all()

    result = []
    for name in REGION_NAMES:
        info = REGION_INFO[name]
        reg_patients = [p for p in patients if p.region == name]
        reg_predictions = [p for p in predictions if (p.input or {}).get("region") == name]
        risks = [p.probability for p in reg_predictions]
        result.append(
            {
                "region": name,
                "lat": info["lat"],
                "lon": info["lon"],
                "patients": len(reg_patients),
                "predictions": len(reg_predictions),
                "mean_risk": round(sum(risks) / len(risks), 4) if risks else None,
                "high_risk": sum(1 for r in risks if r >= settings.risk_high_min),
                "pm25": round(sum(p.pm25 for p in reg_patients) / len(reg_patients), 2)
                if reg_patients
                else info["pm25"],
                "pm25_source": "Promedio de registros de pacientes (demo)" if reg_patients else "Valor regional ilustrativo",
                "green_space": round(sum(p.green_space for p in reg_patients) / len(reg_patients), 1)
                if reg_patients
                else None,
            }
        )
    result.sort(key=lambda r: r["patients"], reverse=True)
    return {
        "regions": result,
        "days": days,
        "environmental_note": (
            "Las concentraciones de PM2.5 y la cobertura verde son valores de demostración y no "
            "provienen de estaciones de monitoreo. Para uso real deben integrarse fuentes oficiales "
            "con trazabilidad y licencia."
        ),
        "privacy_note": "Solo se muestran agregados por región; no se exponen coordenadas de personas.",
        "generated_at": iso(datetime.utcnow()),
    }


@router.get("/regions/{region_name}")
def region_detail(
    region_name: str,
    days: int = Query(default=365, ge=7, le=3650),
    user: User = Depends(require_permission("geo", "read")),
    db: Session = Depends(get_db),
):
    if region_name not in REGION_NAMES:
        raise HTTPException(404, "Región no encontrada.")
    start = datetime.utcnow() - timedelta(days=days)
    patients = db.query(Patient).filter(Patient.region == region_name, Patient.is_active.is_(True)).all()
    predictions = db.query(Prediction).filter(Prediction.created_at >= start).all()
    predictions = [p for p in predictions if (p.input or {}).get("region") == region_name]
    ages = [p.age for p in patients]
    return {
        "region": region_name,
        "patients": len(patients),
        "predictions": len(predictions),
        "mean_risk": round(sum(p.probability for p in predictions) / len(predictions), 4)
        if predictions else None,
        "high_risk": sum(1 for p in predictions if p.risk_level == "alto"),
        "mean_age": round(sum(ages) / len(ages), 1) if ages else None,
        "mean_pm25": round(sum(p.pm25 for p in patients) / len(patients), 2) if patients else None,
        "mean_green_space": round(sum(p.green_space for p in patients) / len(patients), 1)
        if patients else None,
        "patient_sex": {
            "F": sum(1 for p in patients if p.sex == "F"),
            "M": sum(1 for p in patients if p.sex == "M"),
            "Otro": sum(1 for p in patients if p.sex == "Otro"),
        },
        "demo": all(p.is_demo for p in patients) if patients else True,
        "generated_at": iso(datetime.utcnow()),
    }

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any

from fastapi import APIRouter, Depends, Query
from sqlalchemy import func
from sqlalchemy.orm import Session

from ..config import settings
from ..database import get_db
from ..demo_data import REGION_INFO
from ..deps import require_permission
from ..models import Alert, AuditLog, Dataset, ModelRun, Patient, Prediction, User
from ..utils import iso

router = APIRouter(prefix="/dashboard", tags=["Dashboard"])

AGE_BUCKETS = [(18, 29), (30, 39), (40, 49), (50, 59), (60, 69), (70, 79), (80, 120)]


def _window_start(days: int) -> datetime:
    return datetime.utcnow() - timedelta(days=max(1, min(days, 3650)))


@router.get("/summary")
def summary(
    days: int = Query(default=180, ge=7, le=3650),
    region: str | None = None,
    user: User = Depends(require_permission("dashboard", "read")),
    db: Session = Depends(get_db),
):
    start = _window_start(days)
    patient_query = db.query(Patient).filter(Patient.is_active.is_(True))
    if region:
        patient_query = patient_query.filter(Patient.region == region)
    patients = patient_query.all()

    pred_query = db.query(Prediction).filter(Prediction.created_at >= start)
    if region:
        pred_query = pred_query.filter(Prediction.input["region"].as_string() == region)
    predictions = pred_query.all()

    total_predictions = db.query(func.count(Prediction.id)).scalar() or 0
    models_ready = db.query(ModelRun).filter(ModelRun.status == "ready").count()
    active_model = db.query(ModelRun).filter(ModelRun.is_active.is_(True)).first()
    unread_alerts = db.query(Alert).filter(Alert.is_read.is_(False)).count()

    risk_counts = {"bajo": 0, "moderado": 0, "alto": 0}
    region_risk: dict[str, list[float]] = {}
    trend: dict[str, dict[str, int]] = {}
    for pred in predictions:
        risk_counts[pred.risk_level] = risk_counts.get(pred.risk_level, 0) + 1
        reg = (pred.input or {}).get("region")
        if reg:
            region_risk.setdefault(reg, []).append(pred.probability)
        key = pred.created_at.strftime("%Y-%m")
        trend.setdefault(key, {"total": 0, "alto": 0})
        trend[key]["total"] += 1
        if pred.risk_level == "alto":
            trend[key]["alto"] += 1

    # Rellena meses vacíos del último año para una serie continua.
    months = []
    cursor = datetime.utcnow().replace(day=1)
    for _ in range(12):
        key = cursor.strftime("%Y-%m")
        months.append({"period": key, **trend.get(key, {"total": 0, "alto": 0})})
        cursor = (cursor - timedelta(days=1)).replace(day=1)
    months.reverse()

    age_counts = {f"{lo}-{hi if hi < 120 else '+'}": 0 for lo, hi in AGE_BUCKETS}
    for p in patients:
        for lo, hi in AGE_BUCKETS:
            if lo <= p.age <= hi:
                label = f"{lo}-{hi if hi < 120 else '+'}"
                age_counts[label] += 1
                break

    sex_counts = {"F": 0, "M": 0, "Otro": 0}
    for p in patients:
        sex_counts[p.sex] = sex_counts.get(p.sex, 0) + 1

    minimum_risk = [p.probability for p in predictions]
    mean_risk = round(sum(minimum_risk) / len(minimum_risk), 4) if minimum_risk else 0.0

    regions = []
    for name, info in REGION_INFO.items():
        reg_patients = [p for p in patients if p.region == name]
        risks = region_risk.get(name, [])
        regions.append(
            {
                "region": name,
                "patients": len(reg_patients),
                "predictions": len(risks),
                "mean_risk": round(sum(risks) / len(risks), 4) if risks else None,
                "high_risk": sum(1 for r in risks if r >= settings.risk_high_min),
                "pm25": round(
                    sum(p.pm25 for p in reg_patients) / len(reg_patients), 2
                ) if reg_patients else info["pm25"],
                "lat": info["lat"],
                "lon": info["lon"],
            }
        )
    regions.sort(key=lambda r: r["patients"], reverse=True)

    latest_dataset = db.query(Dataset).order_by(Dataset.created_at.desc()).first()
    quality_score = None
    if latest_dataset and latest_dataset.quality:
        missing = float(latest_dataset.quality.get("missing_pct", 0))
        dup = float(latest_dataset.quality.get("duplicate_rows", 0)) / max(1, latest_dataset.rows_count) * 100
        quality_score = round(max(0.0, 100 - missing - dup), 1)

    model_rows = (
        db.query(ModelRun).filter(ModelRun.status == "ready").order_by(ModelRun.trained_at.desc()).all()
    )
    latest_models: dict[str, ModelRun] = {}
    for run in model_rows:
        latest_models.setdefault(run.algorithm, run)
    model_comparison = []
    for run in latest_models.values():
        disc = (run.metrics or {}).get("discrimination", {})
        model_comparison.append(
            {
                "id": run.id,
                "name": run.name,
                "algorithm": run.algorithm,
                "roc_auc": disc.get("roc_auc"),
                "f1": disc.get("f1"),
                "mcc": disc.get("mcc"),
                "is_active": run.is_active,
                "train_rows": run.train_rows,
            }
        )
    model_comparison.sort(key=lambda r: (r["roc_auc"] or 0), reverse=True)

    alerts = (
        db.query(Alert)
        .filter(Alert.is_read.is_(False))
        .order_by(Alert.created_at.desc())
        .limit(6)
        .all()
    )
    recent_activity = db.query(AuditLog).order_by(AuditLog.created_at.desc()).limit(8).all()

    return {
        "filters": {"days": days, "region": region},
        "kpis": {
            "patients": len(patients),
            "datasets": db.query(func.count(Dataset.id)).scalar() or 0,
            "predictions": total_predictions,
            "predictions_window": len(predictions),
            "models_ready": models_ready,
            "high_risk_window": risk_counts.get("alto", 0),
            "high_risk_pct": round(risk_counts.get("alto", 0) / len(predictions) * 100, 1)
            if predictions else 0.0,
            "mean_risk": mean_risk,
            "unread_alerts": unread_alerts,
            "quality_score": quality_score,
        },
        "risk_distribution": [
            {"level": k, "count": v} for k, v in risk_counts.items()
        ],
        "age_distribution": [{"bucket": k, "count": v} for k, v in age_counts.items()],
        "sex_distribution": [{"sex": k, "count": v} for k, v in sex_counts.items()],
        "trend": months,
        "regions": regions,
        "model_comparison": model_comparison,
        "active_model": {
            "id": active_model.id,
            "name": active_model.name,
            "algorithm": active_model.algorithm,
            "roc_auc": (active_model.metrics or {}).get("discrimination", {}).get("roc_auc"),
            "trained_at": iso(active_model.trained_at),
            "is_demo": active_model.is_demo,
        }
        if active_model else None,
        "alerts": [
            {
                "id": a.id,
                "type": a.type,
                "severity": a.severity,
                "title": a.title,
                "message": a.message,
                "created_at": iso(a.created_at),
            }
            for a in alerts
        ],
        "recent_activity": [
            {
                "id": a.id,
                "action": a.action,
                "module": a.module,
                "user_email": a.user_email,
                "result": a.result,
                "created_at": iso(a.created_at),
            }
            for a in recent_activity
        ],
        "demo_mode": bool(active_model.is_demo if active_model else True),
        "disclaimer": (
            "Indicadores calculados sobre datos de demostración. No representan resultados "
            "científicos reales ni diagnósticos clínicos."
        ),
        "generated_at": iso(datetime.utcnow()),
    }

from __future__ import annotations

from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from ..audit import log_action
from ..database import get_db
from ..deps import require_permission
from ..models import Alert, User
from ..utils import iso, paginate

router = APIRouter(prefix="/alerts", tags=["Alertas y notificaciones"])


class AlertCreate(BaseModel):
    title: str = Field(min_length=3, max_length=200)
    message: str = Field(min_length=3, max_length=2000)
    type: str = Field(default="administrativa", max_length=60)
    severity: str = Field(default="info", pattern="^(info|warning|critical)$")
    module: str | None = Field(default=None, max_length=60)


def payload(a: Alert) -> dict:
    return {
        "id": a.id,
        "type": a.type,
        "severity": a.severity,
        "title": a.title,
        "message": a.message,
        "module": a.module,
        "is_read": a.is_read,
        "created_at": iso(a.created_at),
        "read_at": iso(a.read_at),
    }


@router.get("")
def list_alerts(
    unread_only: bool = False,
    page: int = 1,
    page_size: int = 30,
    user: User = Depends(require_permission("alerts", "read")),
    db: Session = Depends(get_db),
):
    query = db.query(Alert)
    if unread_only:
        query = query.filter(Alert.is_read.is_(False))
    query = query.order_by(Alert.created_at.desc())
    items, total = paginate(query, page, page_size)
    unread = db.query(Alert).filter(Alert.is_read.is_(False)).count()
    return {"items": [payload(a) for a in items], "total": total, "unread": unread,
            "page": page, "page_size": page_size}


@router.post("/{alert_id}/read")
def mark_read(
    alert_id: int,
    request: Request,
    user: User = Depends(require_permission("alerts", "read")),
    db: Session = Depends(get_db),
):
    alert = db.get(Alert, alert_id)
    if alert is None:
        raise HTTPException(404, "Alerta no encontrada.")
    alert.is_read = True
    alert.read_at = datetime.utcnow()
    db.commit()
    return payload(alert)


@router.post("/read-all")
def mark_all_read(
    request: Request,
    user: User = Depends(require_permission("alerts", "read")),
    db: Session = Depends(get_db),
):
    count = db.query(Alert).filter(Alert.is_read.is_(False)).update(
        {Alert.is_read: True, Alert.read_at: datetime.utcnow()}
    )
    db.commit()
    return {"updated": count}


@router.post("", status_code=201)
def create_alert(
    body: AlertCreate,
    request: Request,
    user: User = Depends(require_permission("alerts", "write")),
    db: Session = Depends(get_db),
):
    alert = Alert(**body.model_dump(), created_by=user.id)
    db.add(alert)
    db.flush()
    log_action(db, user=user, action="crear_alerta", module="alerts", entity="alerta",
               entity_id=alert.id, detail=body.title, request=request)
    db.commit()
    return payload(alert)

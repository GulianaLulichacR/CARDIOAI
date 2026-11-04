from __future__ import annotations

import csv
import io
from datetime import datetime, timedelta

from fastapi import APIRouter, Depends, Query, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from sqlalchemy import func, or_
from sqlalchemy.orm import Session

from ..audit import log_action, log_navigation
from ..database import get_db
from ..deps import require_permission
from ..models import AuditLog, User
from ..utils import iso, paginate

router = APIRouter(prefix="/audit", tags=["Bitácora y auditoría"])

MODULES = [
    "auth", "dashboard", "patients", "datasets", "models", "predictions", "geo",
    "reports", "audit", "users", "assistant", "alerts",
]


class NavigationEvent(BaseModel):
    module: str = Field(min_length=2, max_length=60)
    direction: str = Field(pattern="^(entrada|salida)$")
    route: str | None = Field(default=None, max_length=200)


def _query(
    db: Session,
    module: str | None = None,
    action: str | None = None,
    user_email: str | None = None,
    result: str | None = None,
    q: str | None = None,
    days: int | None = None,
):
    query = db.query(AuditLog)
    if module:
        query = query.filter(AuditLog.module == module)
    if action:
        query = query.filter(AuditLog.action == action)
    if user_email:
        query = query.filter(AuditLog.user_email.ilike(f"%{user_email}%"))
    if result:
        query = query.filter(AuditLog.result == result)
    if q:
        like = f"%{q}%"
        query = query.filter(
            or_(
                AuditLog.action.ilike(like),
                AuditLog.detail.ilike(like),
                AuditLog.entity_id.ilike(like),
                AuditLog.user_email.ilike(like),
            )
        )
    if days:
        query = query.filter(AuditLog.created_at >= datetime.utcnow() - timedelta(days=days))
    return query


@router.get("")
def list_audit(
    module: str | None = None,
    action: str | None = None,
    user_email: str | None = None,
    result: str | None = None,
    q: str | None = None,
    days: int | None = None,
    page: int = 1,
    page_size: int = 50,
    user: User = Depends(require_permission("audit", "read")),
    db: Session = Depends(get_db),
):
    query = _query(db, module, action, user_email, result, q, days).order_by(AuditLog.created_at.desc())
    items, total = paginate(query, page, page_size)
    return {
        "items": [
            {
                "id": a.id,
                "created_at": iso(a.created_at),
                "user_email": a.user_email,
                "action": a.action,
                "module": a.module,
                "entity": a.entity,
                "entity_id": a.entity_id,
                "result": a.result,
                "detail": a.detail,
                "ip": a.ip,
                "session_id": a.session_id,
            }
            for a in items
        ],
        "total": total,
        "page": page,
        "page_size": page_size,
        "immutable_note": "La bitácora es append-only; no existe endpoint para editar o eliminar registros.",
    }


@router.get("/stats")
def audit_stats(
    days: int = Query(default=30, ge=1, le=3650),
    user: User = Depends(require_permission("audit", "read")),
    db: Session = Depends(get_db),
):
    start = datetime.utcnow() - timedelta(days=days)
    by_module = (
        db.query(AuditLog.module, func.count(AuditLog.id))
        .filter(AuditLog.created_at >= start)
        .group_by(AuditLog.module)
        .all()
    )
    by_result = (
        db.query(AuditLog.result, func.count(AuditLog.id))
        .filter(AuditLog.created_at >= start)
        .group_by(AuditLog.result)
        .all()
    )
    by_action = (
        db.query(AuditLog.action, func.count(AuditLog.id))
        .filter(AuditLog.created_at >= start)
        .group_by(AuditLog.action)
        .order_by(func.count(AuditLog.id).desc())
        .limit(12)
        .all()
    )
    return {
        "days": days,
        "total": sum(v for _, v in by_module),
        "by_module": [{"module": m, "count": c} for m, c in by_module],
        "by_result": [{"result": r, "count": c} for r, c in by_result],
        "by_action": [{"action": a, "count": c} for a, c in by_action],
    }


@router.get("/export.csv")
def export_audit(
    module: str | None = None,
    action: str | None = None,
    user_email: str | None = None,
    result: str | None = None,
    q: str | None = None,
    days: int | None = None,
    user: User = Depends(require_permission("audit", "read")),
    db: Session = Depends(get_db),
):
    rows = _query(db, module, action, user_email, result, q, days).order_by(AuditLog.created_at.desc()).limit(10000).all()
    buffer = io.StringIO()
    writer = csv.writer(buffer)
    writer.writerow(["id", "fecha_utc", "usuario", "accion", "modulo", "entidad", "entidad_id", "resultado", "detalle", "ip", "sesion"])
    for a in rows:
        writer.writerow([a.id, iso(a.created_at), a.user_email, a.action, a.module, a.entity or "",
                         a.entity_id or "", a.result, a.detail or "", a.ip or "", a.session_id or ""])
    log_action(db, user=user, action="exportar_auditoria", module="audit",
               detail=f"{len(rows)} registros exportados")
    db.commit()
    return StreamingResponse(
        iter([buffer.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": 'attachment; filename="bitacora_cardioai.csv"'},
    )


@router.post("/navigation")
def navigation(
    event: NavigationEvent,
    request: Request,
    user: User = Depends(require_permission("dashboard", "read")),
    db: Session = Depends(get_db),
):
    log_navigation(db, user, event.module, event.direction, event.route, request)
    db.commit()
    return {"ok": True}

from __future__ import annotations

from sqlalchemy.orm import Session

from .deps import client_ip
from .models import AuditLog, User
from .security import can


def log_action(
    db: Session,
    *,
    user: User | None,
    action: str,
    module: str,
    entity: str | None = None,
    entity_id: str | int | None = None,
    result: str = "success",
    detail: str | None = None,
    request=None,
    session_id: str | None = None,
) -> AuditLog:
    """Registra un evento en la bitácora. Solo inserción (append-only)."""
    entry = AuditLog(
        user_id=user.id if user else None,
        user_email=user.email if user else "anonimo",
        action=action,
        module=module,
        entity=entity,
        entity_id=str(entity_id) if entity_id is not None else None,
        result=result,
        detail=detail,
        ip=client_ip(request),
        session_id=session_id or getattr(getattr(request, "state", None), "session_id", None),
    )
    db.add(entry)
    return entry


def log_navigation(
    db: Session, user: User, module: str, direction: str, route: str | None, request=None
) -> AuditLog:
    if not can(user.role, "dashboard", "read"):
        pass
    return log_action(
        db,
        user=user,
        action=f"navegacion_{direction}",
        module=module,
        entity="ruta",
        entity_id=route,
        detail=f"{'Ingreso' if direction == 'entrada' else 'Salida'} al módulo {module}",
        request=request,
    )

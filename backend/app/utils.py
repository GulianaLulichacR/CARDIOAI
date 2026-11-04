from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from fastapi import HTTPException


def iso(value: datetime | None) -> str | None:
    if value is None:
        return None
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def paginate(query, page: int, page_size: int) -> tuple[list[Any], int]:
    page = max(1, page)
    page_size = max(1, min(page_size, 200))
    total = query.order_by(None).count()
    items = query.offset((page - 1) * page_size).limit(page_size).all()
    return items, total


def not_found(name: str) -> HTTPException:
    return HTTPException(404, f"{name} no encontrado.")


def user_payload(user) -> dict[str, Any]:
    from .security import permissions_for

    return {
        "id": user.id,
        "email": user.email,
        "full_name": user.full_name,
        "role": user.role,
        "role_label": {
            "superadmin": "Superadministrador",
            "admin": "Administrador",
            "investigador": "Investigador",
            "profesional_salud": "Profesional de salud",
            "analista_datos": "Analista de datos",
            "auditor": "Auditor",
        }.get(user.role, user.role),
        "is_active": user.is_active,
        "must_change_password": user.must_change_password,
        "created_at": iso(user.created_at),
        "last_login_at": iso(user.last_login_at),
        "permissions": permissions_for(user.role),
    }

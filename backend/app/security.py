from __future__ import annotations

import re
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any

import bcrypt
import jwt

from .config import settings

ROLES: dict[str, dict[str, str]] = {
    "superadmin": {"label": "Superadministrador", "description": "Control total de la plataforma."},
    "admin": {"label": "Administrador", "description": "Gestión operativa, usuarios y configuración."},
    "investigador": {"label": "Investigador", "description": "Datos, modelos, predicciones y reportes."},
    "profesional_salud": {"label": "Profesional de salud", "description": "Evaluación de registros y predicciones."},
    "analista_datos": {"label": "Analista de datos", "description": "Datasets, modelos y analítica."},
    "auditor": {"label": "Auditor", "description": "Lectura de bitácora, reportes y métricas."},
}

ALL_ROLES = list(ROLES)
CLINICAL = ["superadmin", "admin", "investigador", "profesional_salud", "analista_datos"]
DATA_WRITERS = ["superadmin", "admin", "investigador", "analista_datos"]

MODULE_PERMISSIONS: dict[str, dict[str, list[str]]] = {
    "dashboard": {"read": ALL_ROLES, "write": []},
    "patients": {"read": ALL_ROLES, "write": ["superadmin", "admin", "investigador", "profesional_salud"]},
    "datasets": {"read": ["superadmin", "admin", "investigador", "analista_datos", "auditor"], "write": DATA_WRITERS},
    "models": {"read": ALL_ROLES, "write": DATA_WRITERS},
    "predictions": {"read": ALL_ROLES, "write": CLINICAL},
    "geo": {"read": ALL_ROLES, "write": []},
    "reports": {"read": ALL_ROLES, "write": CLINICAL},
    "audit": {"read": ["superadmin", "admin", "auditor", "investigador"], "write": []},
    "users": {"read": ["superadmin", "admin"], "write": ["superadmin", "admin"]},
    "assistant": {"read": ALL_ROLES, "write": []},
    "alerts": {"read": ALL_ROLES, "write": ["superadmin", "admin"]},
    "docs": {"read": ALL_ROLES, "write": []},
}


def permissions_for(role: str) -> dict[str, list[str]]:
    if role == "superadmin":
        return {m: ["read", "write"] for m in MODULE_PERMISSIONS}
    return {
        module: [action for action, roles in perms.items() if role in roles]
        for module, perms in MODULE_PERMISSIONS.items()
    }


def can(role: str, module: str, action: str = "read") -> bool:
    if role == "superadmin":
        return True
    allowed = MODULE_PERMISSIONS.get(module, {}).get(action, [])
    return role in allowed


def hash_password(password: str) -> str:
    pw = password.encode("utf-8")[:72]
    return bcrypt.hashpw(pw, bcrypt.gensalt(rounds=12)).decode("utf-8")


def verify_password(password: str, password_hash: str) -> bool:
    try:
        return bcrypt.checkpw(password.encode("utf-8")[:72], password_hash.encode("utf-8"))
    except (ValueError, TypeError):
        return False


def validate_password_strength(password: str) -> list[str]:
    problems: list[str] = []
    if len(password) < 10:
        problems.append("Debe tener al menos 10 caracteres.")
    if not re.search(r"[A-Z]", password):
        problems.append("Debe incluir una letra mayúscula.")
    if not re.search(r"[a-z]", password):
        problems.append("Debe incluir una letra minúscula.")
    if not re.search(r"\d", password):
        problems.append("Debe incluir un número.")
    if not re.search(r"[^A-Za-z0-9]", password):
        problems.append("Debe incluir un símbolo.")
    return problems


def create_token(user_id: int, role: str, token_type: str = "access", session_id: str | None = None) -> str:
    now = datetime.now(timezone.utc)
    minutes = settings.access_token_minutes if token_type == "access" else settings.refresh_token_days * 1440
    payload: dict[str, Any] = {
        "sub": str(user_id),
        "role": role,
        "type": token_type,
        "sid": session_id or uuid.uuid4().hex,
        "jti": uuid.uuid4().hex,
        "iat": now,
        "exp": now + timedelta(minutes=minutes),
        "iss": settings.app_name,
    }
    return jwt.encode(payload, settings.secret_key, algorithm=settings.jwt_algorithm)


def decode_token(token: str, expected_type: str = "access") -> dict[str, Any]:
    try:
        payload = jwt.decode(
            token,
            settings.secret_key,
            algorithms=[settings.jwt_algorithm],
            issuer=settings.app_name,
        )
    except jwt.ExpiredSignatureError as exc:
        raise ValueError("Sesión expirada.") from exc
    except jwt.InvalidTokenError as exc:
        raise ValueError("Token inválido.") from exc
    if payload.get("type") != expected_type:
        raise ValueError("Tipo de token incorrecto.")
    return payload

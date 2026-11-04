from __future__ import annotations

from typing import Any

from fastapi import Depends, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy.orm import Session

from .database import get_db
from .models import User
from .security import can, decode_token

_bearer = HTTPBearer(auto_error=False)


def client_ip(request: Request | None) -> str | None:
    if request is None:
        return None
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else None


def get_current_user(
    request: Request,
    credentials: HTTPAuthorizationCredentials | None = Depends(_bearer),
    db: Session = Depends(get_db),
) -> User:
    if credentials is None:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Se requiere autenticación.")
    try:
        payload = decode_token(credentials.credentials, "access")
    except ValueError as exc:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, str(exc)) from exc
    user = db.get(User, int(payload["sub"]))
    if user is None or not user.is_active:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Usuario inactivo o inexistente.")
    # Expone el sid del token para la bitácora dentro de la petición.
    request.state.session_id = payload.get("sid")
    request.state.token_payload = payload
    return user


def require_permission(module: str, action: str = "read"):
    def dependency(user: User = Depends(get_current_user)) -> User:
        if not can(user.role, module, action):
            raise HTTPException(
                status.HTTP_403_FORBIDDEN,
                f"El rol '{user.role}' no tiene permiso de {action} en el módulo '{module}'.",
            )
        return user

    return dependency


def require_roles(*roles: str):
    def dependency(user: User = Depends(get_current_user)) -> User:
        if user.role != "superadmin" and user.role not in roles:
            raise HTTPException(status.HTTP_403_FORBIDDEN, "No autorizado para esta operación.")
        return user

    return dependency


def audit_context(request: Request, user: User | None) -> dict[str, Any]:
    return {
        "user_id": user.id if user else None,
        "user_email": user.email if user else "anonimo",
        "ip": client_ip(request),
        "session_id": getattr(request.state, "session_id", None),
    }

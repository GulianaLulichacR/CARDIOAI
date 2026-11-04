from __future__ import annotations

from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, Request, status
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from ..audit import log_action
from ..database import get_db
from ..deps import get_current_user
from ..models import Alert, User
from ..security import (
    create_token,
    decode_token,
    hash_password,
    validate_password_strength,
    verify_password,
)
from ..utils import iso, user_payload

router = APIRouter(prefix="/auth", tags=["Autenticación"])


class LoginRequest(BaseModel):
    email: str
    password: str = Field(min_length=1, max_length=72)


class RefreshRequest(BaseModel):
    refresh_token: str


class ChangePasswordRequest(BaseModel):
    current_password: str
    new_password: str = Field(min_length=10, max_length=72)


class RecoverRequest(BaseModel):
    email: str


@router.post("/login")
def login(payload: LoginRequest, request: Request, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.email == payload.email.lower()).one_or_none()
    if user is None or not user.is_active or not verify_password(payload.password, user.password_hash):
        log_action(
            db,
            user=None,
            action="login_fallido",
            module="auth",
            entity="usuario",
            entity_id=payload.email,
            result="error",
            detail="Credenciales inválidas o usuario inactivo.",
            request=request,
        )
        db.commit()
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Credenciales inválidas.")

    session_id = create_token(user.id, user.role, "access")
    access_payload = decode_token(session_id, "access")
    refresh = create_token(user.id, user.role, "refresh", session_id=access_payload["sid"])
    user.last_login_at = datetime.utcnow()
    user.last_login_ip = request.client.host if request.client else None
    log_action(
        db,
        user=user,
        action="login",
        module="auth",
        entity="usuario",
        entity_id=user.id,
        detail="Inicio de sesión exitoso.",
        request=request,
        session_id=access_payload["sid"],
    )
    db.commit()
    return {
        "access_token": session_id,
        "refresh_token": refresh,
        "token_type": "bearer",
        "expires_in": 60 * 60 * 8,
        "user": user_payload(user),
    }


@router.post("/refresh")
def refresh(payload: RefreshRequest, request: Request, db: Session = Depends(get_db)):
    try:
        data = decode_token(payload.refresh_token, "refresh")
    except ValueError as exc:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, str(exc)) from exc
    user = db.get(User, int(data["sub"]))
    if user is None or not user.is_active:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Usuario inactivo o inexistente.")
    access = create_token(user.id, user.role, "access", session_id=data.get("sid"))
    refresh_token = create_token(user.id, user.role, "refresh", session_id=data.get("sid"))
    return {
        "access_token": access,
        "refresh_token": refresh_token,
        "token_type": "bearer",
        "user": user_payload(user),
    }


@router.get("/me")
def me(user: User = Depends(get_current_user)):
    return user_payload(user)


@router.post("/logout")
def logout(request: Request, user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    log_action(db, user=user, action="logout", module="auth", detail="Cierre de sesión.", request=request)
    db.commit()
    return {"message": "Sesión cerrada. Descarte el token en el cliente."}


@router.post("/change-password")
def change_password(
    payload: ChangePasswordRequest,
    request: Request,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    if not verify_password(payload.current_password, user.password_hash):
        raise HTTPException(400, "La contraseña actual no es correcta.")
    problems = validate_password_strength(payload.new_password)
    if problems:
        raise HTTPException(422, " ".join(problems))
    user.password_hash = hash_password(payload.new_password)
    user.must_change_password = False
    log_action(db, user=user, action="cambio_password", module="auth", request=request)
    db.commit()
    return {"message": "Contraseña actualizada."}


@router.post("/recover")
def recover(payload: RecoverRequest, request: Request, db: Session = Depends(get_db)):
    """Solicitud de recuperación.

    No hay servidor SMTP configurado en este despliegue. La solicitud se registra
    y se notifica a los administradores; el restablecimiento se realiza desde el
    módulo de usuarios. Respuesta genérica para no revelar si el correo existe.
    """
    user = db.query(User).filter(User.email == payload.email.lower()).one_or_none()
    if user:
        db.add(
            Alert(
                type="seguridad",
                severity="warning",
                title="Solicitud de recuperación de contraseña",
                message=(
                    f"El usuario {user.email} solicitó recuperación. Un administrador debe "
                    "restablecer la contraseña desde el módulo de usuarios."
                ),
                module="users",
            )
        )
    log_action(
        db,
        user=user,
        action="solicitud_recuperacion",
        module="auth",
        entity="usuario",
        entity_id=user.id if user else payload.email,
        request=request,
    )
    db.commit()
    return {
        "message": "Si el correo está registrado, se notificó al equipo administrador. "
                   "No se envía correo automático en este despliegue."
    }

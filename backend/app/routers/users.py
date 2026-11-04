from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from pydantic import BaseModel, Field
from sqlalchemy import or_
from sqlalchemy.orm import Session

from ..audit import log_action
from ..database import get_db
from ..deps import require_permission
from ..models import AuditLog, User
from ..security import ROLES, hash_password, validate_password_strength
from ..utils import iso, paginate, user_payload

router = APIRouter(prefix="/users", tags=["Usuarios y roles"])


class UserCreate(BaseModel):
    email: str
    full_name: str = Field(min_length=3, max_length=160)
    role: str
    password: str = Field(min_length=10, max_length=72)
    must_change_password: bool = True


class UserUpdate(BaseModel):
    full_name: str | None = Field(default=None, min_length=3, max_length=160)
    role: str | None = None
    is_active: bool | None = None


class ResetPassword(BaseModel):
    new_password: str = Field(min_length=10, max_length=72)
    must_change_password: bool = True


@router.get("/roles")
def list_roles(user: User = Depends(require_permission("users", "read"))):
    return [{"key": key, **meta} for key, meta in ROLES.items()]


@router.get("")
def list_users(
    q: str | None = None,
    role: str | None = None,
    active: bool | None = None,
    page: int = 1,
    page_size: int = 25,
    user: User = Depends(require_permission("users", "read")),
    db: Session = Depends(get_db),
):
    query = db.query(User)
    if q:
        like = f"%{q.strip()}%"
        query = query.filter(or_(User.email.ilike(like), User.full_name.ilike(like)))
    if role:
        query = query.filter(User.role == role)
    if active is not None:
        query = query.filter(User.is_active == active)
    query = query.order_by(User.created_at.desc())
    items, total = paginate(query, page, page_size)
    return {"items": [user_payload(u) for u in items], "total": total, "page": page, "page_size": page_size}


@router.post("", status_code=201)
def create_user(
    payload: UserCreate,
    request: Request,
    admin: User = Depends(require_permission("users", "write")),
    db: Session = Depends(get_db),
):
    if payload.role not in ROLES:
        raise HTTPException(422, "Rol no válido.")
    if admin.role == "admin" and payload.role == "superadmin":
        raise HTTPException(403, "Solo un superadministrador puede crear otro superadministrador.")
    if db.query(User).filter(User.email == payload.email.lower()).first():
        raise HTTPException(409, "Ya existe un usuario con ese correo.")
    problems = validate_password_strength(payload.password)
    if problems:
        raise HTTPException(422, " ".join(problems))
    created = User(
        email=payload.email.lower(),
        full_name=payload.full_name.strip(),
        role=payload.role,
        password_hash=hash_password(payload.password),
        must_change_password=payload.must_change_password,
    )
    db.add(created)
    db.flush()
    log_action(
        db,
        user=admin,
        action="crear_usuario",
        module="users",
        entity="usuario",
        entity_id=created.id,
        detail=f"Rol asignado: {payload.role}",
        request=request,
    )
    db.commit()
    return user_payload(created)


@router.patch("/{user_id}")
def update_user(
    user_id: int,
    payload: UserUpdate,
    request: Request,
    admin: User = Depends(require_permission("users", "write")),
    db: Session = Depends(get_db),
):
    target = db.get(User, user_id)
    if target is None:
        raise HTTPException(404, "Usuario no encontrado.")
    if target.id == admin.id and payload.is_active is False:
        raise HTTPException(400, "No puede desactivar su propia cuenta.")
    if target.role == "superadmin" and payload.role and payload.role != "superadmin":
        count = db.query(User).filter(User.role == "superadmin", User.is_active.is_(True)).count()
        if count <= 1:
            raise HTTPException(400, "Debe existir al menos un superadministrador activo.")
    if payload.role and payload.role not in ROLES:
        raise HTTPException(422, "Rol no válido.")
    if payload.role == "superadmin" and admin.role != "superadmin":
        raise HTTPException(403, "Solo un superadministrador puede asignar ese rol.")
    changes = []
    if payload.full_name is not None:
        target.full_name = payload.full_name.strip()
        changes.append("nombre")
    if payload.role is not None and payload.role != target.role:
        changes.append(f"rol {target.role}→{payload.role}")
        target.role = payload.role
    if payload.is_active is not None:
        target.is_active = payload.is_active
        changes.append("activado" if payload.is_active else "desactivado")
    log_action(
        db,
        user=admin,
        action="actualizar_usuario",
        module="users",
        entity="usuario",
        entity_id=target.id,
        detail=", ".join(changes) or "sin cambios",
        request=request,
    )
    db.commit()
    return user_payload(target)


@router.post("/{user_id}/reset-password")
def reset_password(
    user_id: int,
    payload: ResetPassword,
    request: Request,
    admin: User = Depends(require_permission("users", "write")),
    db: Session = Depends(get_db),
):
    target = db.get(User, user_id)
    if target is None:
        raise HTTPException(404, "Usuario no encontrado.")
    problems = validate_password_strength(payload.new_password)
    if problems:
        raise HTTPException(422, " ".join(problems))
    target.password_hash = hash_password(payload.new_password)
    target.must_change_password = payload.must_change_password
    log_action(
        db,
        user=admin,
        action="reset_password",
        module="users",
        entity="usuario",
        entity_id=target.id,
        detail="Restablecimiento administrativo de contraseña.",
        request=request,
    )
    db.commit()
    return {"message": f"Contraseña restablecida para {target.email}."}


@router.get("/{user_id}/activity")
def user_activity(
    user_id: int,
    page: int = 1,
    page_size: int = 20,
    admin: User = Depends(require_permission("users", "read")),
    db: Session = Depends(get_db),
):
    target = db.get(User, user_id)
    if target is None:
        raise HTTPException(404, "Usuario no encontrado.")
    query = db.query(AuditLog).filter(AuditLog.user_id == user_id).order_by(AuditLog.created_at.desc())
    items, total = paginate(query, page, page_size)
    return {
        "items": [
            {
                "id": a.id,
                "created_at": iso(a.created_at),
                "action": a.action,
                "module": a.module,
                "entity": a.entity,
                "entity_id": a.entity_id,
                "result": a.result,
                "detail": a.detail,
                "ip": a.ip,
            }
            for a in items
        ],
        "total": total,
        "page": page,
        "page_size": page_size,
    }

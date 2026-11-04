from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from ..audit import log_action
from ..database import get_db
from ..deps import require_permission
from ..models import KnowledgeDocument, User
from ..services.rag import answer_question

router = APIRouter(prefix="/assistant", tags=["Asistente documental"])


class Question(BaseModel):
    question: str = Field(min_length=3, max_length=800)


@router.post("/query")
def query(
    payload: Question,
    request: Request,
    user: User = Depends(require_permission("assistant", "read")),
    db: Session = Depends(get_db),
):
    try:
        result = answer_question(db, user, payload.question)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    log_action(
        db, user=user, action="consulta_asistente", module="assistant", entity="consulta",
        detail=payload.question[:200], request=request,
    )
    db.commit()
    return result


@router.get("/documents")
def documents(
    user: User = Depends(require_permission("assistant", "read")),
    db: Session = Depends(get_db),
):
    docs = db.query(KnowledgeDocument).order_by(KnowledgeDocument.title).all()
    return [
        {
            "id": d.id,
            "title": d.title,
            "source": d.source,
            "url": d.url,
            "year": d.year,
            "tags": d.tags.split(",") if d.tags else [],
        }
        for d in docs
    ]


@router.get("/suggestions")
def suggestions(user: User = Depends(require_permission("assistant", "read"))):
    return {
        "suggestions": [
            "¿Qué rendimiento tiene el modelo activo?",
            "¿Cómo se definen los niveles bajo, moderado y alto?",
            "¿Qué limitaciones tiene el dataset de demostración?",
            "¿Qué factores de riesgo cardiovascular menciona la OMS?",
            "¿Qué es SHAP y cómo se usa aquí?",
            "¿Qué recomienda TRIPOD+AI para reportar modelos?",
        ]
    }

"""Asistente documental con recuperación (RAG) y datos reales del sistema.

Por defecto funciona en modo extractivo: recupera fragmentos de la base
documental y los cita, sin inventar contenido. Si se configura un proveedor
LLM, se usa únicamente para redactar sobre el contexto recuperado; el modelo
predictivo nunca se sustituye por el LLM.
"""
from __future__ import annotations

import re
from typing import Any

from sqlalchemy import func
from sqlalchemy.orm import Session

from ..config import settings
from ..models import KnowledgeDocument, ModelRun, Patient, Prediction
from ..security import can

DISCLAIMER = (
    "Respuesta generada a partir de documentos citados y de las métricas registradas en la plataforma. "
    "No constituye diagnóstico ni recomendación clínica individual. Verifique siempre la fuente original "
    "y consulte a un profesional de la salud."
)

STOPWORDS = {
    "para", "como", "sobre", "entre", "desde", "hasta", "cuando", "donde", "porque", "también",
    "puede", "pueden", "debe", "deben", "cual", "cuales", "que", "los", "las", "del", "una",
    "unos", "unas", "con", "por", "sus", "este", "esta", "estos", "estas", "más", "mas", "sin",
    "the", "and", "for", "with", "from", "that", "this", "are", "was", "were", "his", "her",
}


def _tokens(text: str) -> list[str]:
    words = re.findall(r"[a-záéíóúüñ0-9]+", text.lower())
    return [w for w in words if len(w) > 2 and w not in STOPWORDS]


def _split_sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+", text.replace("\n", " "))
    return [p.strip() for p in parts if len(p.strip()) > 40]


def _system_context(db: Session) -> dict[str, Any]:
    active = db.query(ModelRun).filter(ModelRun.is_active.is_(True)).first()
    ready = db.query(ModelRun).filter(ModelRun.status == "ready").count()
    patients = db.query(func.count(Patient.id)).filter(Patient.is_active.is_(True)).scalar() or 0
    predictions = db.query(func.count(Prediction.id)).scalar() or 0
    high = db.query(func.count(Prediction.id)).filter(Prediction.risk_level == "alto").scalar() or 0
    models = []
    for run in db.query(ModelRun).filter(ModelRun.status == "ready").order_by(ModelRun.trained_at.desc()).all()[:8]:
        disc = (run.metrics or {}).get("discrimination", {})
        models.append(
            {
                "name": run.name,
                "algorithm": run.algorithm,
                "auc": disc.get("roc_auc"),
                "f1": disc.get("f1"),
                "mcc": disc.get("mcc"),
                "demo": run.is_demo,
                "active": run.is_active,
            }
        )
    return {
        "active": {
            "name": active.name,
            "algorithm": active.algorithm,
            "auc": (active.metrics or {}).get("discrimination", {}).get("roc_auc"),
            "f1": (active.metrics or {}).get("discrimination", {}).get("f1"),
            "threshold": (active.metrics or {}).get("discrimination", {}).get("threshold"),
            "horizon": active.horizon_years,
            "demo": active.is_demo,
        }
        if active else None,
        "models_ready": ready,
        "models": models,
        "patients": patients,
        "predictions": predictions,
        "high_risk": high,
    }


def _retrieve(db: Session, question: str, limit: int = 3) -> list[dict[str, Any]]:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    docs = db.query(KnowledgeDocument).all()
    if not docs:
        return []
    sentences: list[tuple[str, KnowledgeDocument]] = []
    for doc in docs:
        for sentence in _split_sentences(doc.content):
            sentences.append((sentence, doc))
    if not sentences:
        return []
    corpus = [s for s, _ in sentences] + [question]
    try:
        vectorizer = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=1)
        matrix = vectorizer.fit_transform(corpus)
    except ValueError:
        return []
    scores = cosine_similarity(matrix[-1], matrix[:-1]).ravel()
    ranked = scores.argsort()[::-1]
    results: list[dict[str, Any]] = []
    seen_docs: set[int] = set()
    for idx in ranked:
        if scores[idx] < 0.03:
            break
        sentence, doc = sentences[int(idx)]
        if doc.id in seen_docs and len(results) >= 2:
            continue
        seen_docs.add(doc.id)
        results.append(
            {
                "text": sentence,
                "title": doc.title,
                "source": doc.source,
                "url": doc.url,
                "year": doc.year,
                "score": round(float(scores[idx]), 4),
            }
        )
        if len(results) >= limit:
            break
    return results


def _platform_answer(question: str, context: dict[str, Any]) -> str | None:
    q = question.lower()
    if any(term in q for term in ["auc", "rendimiento", "métrica", "metrica", "desempeño", "modelo"]):
        if not context["active"]:
            return (
                f"No hay un modelo activo. Hay {context['models_ready']} modelos entrenados. "
                "Puede iniciar el entrenamiento desde el módulo Modelos."
            )
        active = context["active"]
        comparison = "; ".join(
            f"{m['name']}: AUC {m['auc']}" for m in context["models"] if m["auc"] is not None
        )
        return (
            f"El modelo activo es {active['name']} ({active['algorithm']}), con AUC-ROC en el "
            f"holdout de prueba {active['auc']} y F1 {active['f1']} en el conjunto retenido. Horizonte: "
            f"{active['horizon']} años. {'Es un modelo entrenado con datos sintéticos de demostración.' if active['demo'] else 'Es un modelo entrenado con una cohorte pública observada.'} "
            f"Comparación de modelos listos: {comparison or 'sin métricas disponibles'}. "
            f"Hay {context['models_ready']} modelos entrenados en total."
        )
    if any(term in q for term in ["predicción", "prediccion", "riesgo", "estratificación", "estratificacion"]):
        return (
            f"La plataforma registra {context['predictions']} predicciones, de las cuales "
            f"{context['high_risk']} fueron clasificadas como riesgo alto con los umbrales de demostración "
            f"(bajo < {settings.risk_low_max:.0%}; alto ≥ {settings.risk_high_min:.0%}). "
            f"El nivel se asigna a partir de la probabilidad del modelo activo y debe interpretarse como "
            "resultado de investigación. No es un diagnóstico."
        )
    if any(term in q for term in ["paciente", "registro", "cohorte", "datos"]):
        return (
            f"La base contiene {context['patients']} registros de pacientes activos y "
            f"{context['predictions']} predicciones. El dataset sintético de demostración está "
            "etiquetado como tal; el dataset público observado se registra con su procedencia, "
            "licencia y huella SHA-256. Ninguno de los dos debe presentarse como evidencia clínica "
            "de beneficio."
        )
    return None


def _llm_answer(question: str, chunks: list[dict[str, Any]], context: dict[str, Any]) -> str | None:
    if settings.llm_provider != "openai" or not settings.openai_api_key:
        return None
    try:
        import httpx

        context_text = "\n\n".join(
            f"[{i + 1}] {c['title']} ({c['source']}): {c['text']}" for i, c in enumerate(chunks)
        )
        system = (
            "Eres un asistente de investigación en salud pública. Responde SOLO con la información del "
            "contexto. No inventes cifras, no diagnostiques, no recomiendes tratamientos ni medicamentos. "
            "Cita las fuentes como [1], [2]. Si el contexto no responde, dilo explícitamente. "
            "Máximo 180 palabras."
        )
        payload = {
            "model": settings.openai_model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": f"Contexto:\n{context_text}\n\nPregunta: {question}"},
            ],
            "temperature": 0.1,
        }
        response = httpx.post(
            "https://api.openai.com/v1/chat/completions",
            headers={"Authorization": f"Bearer {settings.openai_api_key}"},
            json=payload,
            timeout=25,
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"].strip()
    except Exception:
        return None


def answer_question(db: Session, user, question: str) -> dict[str, Any]:
    question = question.strip()
    if len(question) < 3:
        raise ValueError("La pregunta es demasiado corta.")
    if len(question) > 800:
        raise ValueError("La pregunta supera el límite de 800 caracteres.")
    context = _system_context(db)
    clinical_terms = ["medicamento", "tratamiento", "dosis", "receta", "diagnóstico", "diagnostico", "debo tomar"]
    if any(term in question.lower() for term in clinical_terms):
        return {
            "answer": (
                "No puedo emitir recomendaciones clínicas individuales ni indicar tratamientos o dosis. "
                "La plataforma estima riesgo a partir de un modelo estadístico y sus resultados requieren "
                "interpretación por un profesional de la salud. Consulte las guías citadas y a su equipo clínico."
            ),
            "sources": [],
            "grounded": True,
            "mode": "rechazo_seguro",
            "disclaimer": DISCLAIMER,
        }

    chunks = _retrieve(db, question)
    platform = _platform_answer(question, context)
    if platform:
        answer = platform
        sources = [
            {
                "title": "Base de datos CARDIOAI (métricas y conteos registrados)",
                "source": "Plataforma CARDIOAI",
                "url": None,
                "year": None,
            }
        ]
        sources.extend(
            {
                "title": c["title"],
                "source": c["source"],
                "url": c["url"],
                "year": c["year"],
            }
            for c in chunks[:2]
        )
    else:
        if chunks:
            answer = "Según la base documental: " + " ".join(
                f"{c['text']} [{i + 1}]" for i, c in enumerate(chunks)
            )
        else:
            answer = (
                "No encontré evidencia suficiente en la base documental para responder esa pregunta. "
                "Puede reformularla o consultar las fuentes incluidas en el módulo de documentación."
            )
        sources = [
            {
                "title": c["title"],
                "source": c["source"],
                "url": c["url"],
                "year": c["year"],
            }
            for c in chunks
        ]
    llm = _llm_answer(question, chunks, context) if chunks else None
    return {
        "answer": llm or answer,
        "sources": sources,
        "grounded": True if (chunks or platform) else False,
        "mode": "llm_con_contexto" if llm else "extractivo",
        "disclaimer": DISCLAIMER,
        "suggested": [
            "¿Qué rendimiento tiene el modelo activo?",
            "¿Cómo se definen los niveles de riesgo?",
            "¿Qué limitaciones tiene el dataset de demostración?",
            "¿Qué dice la OMS sobre los factores de riesgo cardiovascular?",
        ],
    }

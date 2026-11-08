from __future__ import annotations

import json
import logging
import time
import traceback
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from sqlalchemy.exc import IntegrityError

from .config import BASE_DIR, settings
from .database import init_db
from .routers import (
    alerts,
    assistant,
    audit,
    auth,
    dashboard,
    datasets,
    geo,
    models,
    patients,
    predictions,
    public_datasets,
    reports,
    system,
    users,
)

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("cardioai")

FRONTEND_DIST = BASE_DIR / "frontend" / "dist"


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    from .database import SessionLocal

    db = SessionLocal()
    try:
        from .ml_engine import recover_orphaned_runs

        recovered = recover_orphaned_runs(db)
        if recovered:
            logger.warning(
                json.dumps({"event": "orphaned_runs_recovered", "count": recovered})
            )
    except Exception:
        db.rollback()
        logger.error(
            json.dumps({"event": "orphan_recovery_failed", "error": traceback.format_exc()[-500:]})
        )
    finally:
        db.close()
    if settings.seed_demo_data:
        from .seed import seed_all

        db = SessionLocal()
        try:
            seed_all(db)
            logger.info(json.dumps({"event": "seed_completed", "demo": True}))
        except Exception:
            db.rollback()
            logger.error(json.dumps({"event": "seed_failed", "error": traceback.format_exc()[-800:]}))
        finally:
            db.close()
    logger.info(json.dumps({"event": "startup", "app": settings.app_name, "version": settings.app_version}))
    yield
    logger.info(json.dumps({"event": "shutdown"}))


app = FastAPI(
    title="CARDIOAI · Predicción explicable de riesgo cardiovascular",
    description=(
        "Plataforma de investigación para la predicción del riesgo cardiovascular con integración de "
        "factores clínicos, conductuales y socioambientales. Los datos incluidos son sintéticos y de "
        "demostración. No es un dispositivo médico."
    ),
    version=settings.app_version,
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    openapi_url="/api/openapi.json",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origin_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["X-Request-ID", "X-Process-Time-Ms"],
)


@app.middleware("http")
async def observability_middleware(request: Request, call_next):
    request_id = request.headers.get("x-request-id") or uuid.uuid4().hex[:16]
    request.state.request_id = request_id
    start = time.perf_counter()
    try:
        response = await call_next(request)
    except Exception:
        logger.error(
            json.dumps(
                {
                    "event": "unhandled_error",
                    "request_id": request_id,
                    "path": request.url.path,
                    "error": traceback.format_exc()[-1200:],
                }
            )
        )
        raise
    elapsed = (time.perf_counter() - start) * 1000
    response.headers["X-Request-ID"] = request_id
    response.headers["X-Process-Time-Ms"] = f"{elapsed:.1f}"
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    if request.url.path.startswith("/api") and not request.url.path.startswith("/api/docs"):
        response.headers["Cache-Control"] = "no-store"
    if request.url.path.startswith("/api") and request.method != "GET":
        logger.info(
            json.dumps(
                {
                    "event": "request",
                    "request_id": request_id,
                    "method": request.method,
                    "path": request.url.path,
                    "status": response.status_code,
                    "ms": round(elapsed, 1),
                }
            )
        )
    return response


@app.exception_handler(IntegrityError)
async def integrity_handler(request: Request, exc: IntegrityError):
    return JSONResponse(
        status_code=409,
        content={
            "detail": "Conflicto de integridad de datos. Verifique duplicados o referencias existentes.",
            "request_id": getattr(request.state, "request_id", None),
        },
    )


@app.exception_handler(Exception)
async def unhandled_handler(request: Request, exc: Exception):
    detail = str(exc) if settings.environment != "production" else "Error interno del servidor."
    return JSONResponse(
        status_code=500,
        content={"detail": detail, "request_id": getattr(request.state, "request_id", None)},
    )


API = settings.api_prefix
for module in (auth, users, patients, datasets, public_datasets, models, predictions, dashboard, geo, reports, audit, alerts, assistant, system):
    app.include_router(module.router, prefix=API)


@app.get("/api", include_in_schema=False)
def api_root():
    return {"app": settings.app_name, "docs": "/api/docs", "health": "/api/system/health"}


@app.get("/{full_path:path}", include_in_schema=False)
def serve_frontend(full_path: str):
    if full_path.startswith("api"):
        return JSONResponse({"detail": "Not Found"}, status_code=404)
    index = FRONTEND_DIST / "index.html"
    candidate = (FRONTEND_DIST / full_path).resolve()
    if full_path and FRONTEND_DIST.resolve() in candidate.parents and candidate.is_file():
        return FileResponse(candidate)
    if index.exists():
        return FileResponse(index)
    return JSONResponse(
        {
            "app": settings.app_name,
            "message": "Frontend no compilado. Ejecute 'npm run build' en /frontend o use el servidor Vite en :5173.",
            "api_docs": "/api/docs",
        }
    )

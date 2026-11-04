from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from ..audit import log_action
from ..config import REPORT_DIR
from ..database import get_db
from ..deps import require_permission
from ..models import Report, User
from ..services.reports import FORMATS, REPORT_TYPES, generate_report
from ..utils import iso

router = APIRouter(prefix="/reports", tags=["Reportes"])


class ReportRequest(BaseModel):
    report_type: str
    format: str = Field(default="pdf", pattern="^(pdf|docx|xlsx)$")
    days: int = Field(default=180, ge=1, le=3650)


def payload(report: Report) -> dict:
    return {
        "id": report.id,
        "report_type": report.report_type,
        "title": REPORT_TYPES.get(report.report_type, report.report_type),
        "format": report.format,
        "filename": report.filename,
        "filters": report.filters,
        "size_bytes": report.size_bytes,
        "created_at": iso(report.created_at),
        "download_url": f"/api/reports/{report.id}/download",
    }


@router.get("/types")
def report_types(user: User = Depends(require_permission("reports", "read"))):
    return {
        "types": [{"key": k, "label": v} for k, v in REPORT_TYPES.items()],
        "formats": [{"key": k, "mime": v} for k, v in FORMATS.items()],
    }


@router.get("")
def list_reports(
    user: User = Depends(require_permission("reports", "read")),
    db: Session = Depends(get_db),
):
    rows = db.query(Report).order_by(Report.created_at.desc()).limit(100).all()
    return [payload(r) for r in rows]


@router.post("", status_code=201)
def create_report(
    body: ReportRequest,
    request: Request,
    user: User = Depends(require_permission("reports", "write")),
    db: Session = Depends(get_db),
):
    try:
        path, filters, title = generate_report(db, user, body.report_type, body.format, body.days)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(500, f"No se pudo generar el reporte: {exc}") from exc
    report = Report(
        report_type=body.report_type,
        format=body.format,
        filename=path.name,
        file_path=str(path),
        filters=filters,
        size_bytes=path.stat().st_size,
        created_by=user.id,
    )
    db.add(report)
    db.flush()
    log_action(
        db, user=user, action="generar_reporte", module="reports", entity="reporte",
        entity_id=report.id, detail=f"{title} ({body.format})", request=request,
    )
    db.commit()
    result = payload(report)
    result["title"] = title
    return result


@router.get("/{report_id}/download")
def download_report(
    report_id: int,
    user: User = Depends(require_permission("reports", "read")),
    db: Session = Depends(get_db),
):
    report = db.get(Report, report_id)
    if report is None:
        raise HTTPException(404, "Reporte no encontrado.")
    path = Path(report.file_path).resolve()
    if REPORT_DIR.resolve() not in path.parents or not path.exists():
        raise HTTPException(404, "Archivo no disponible.")
    log_action(db, user=user, action="descargar_reporte", module="reports", entity="reporte",
               entity_id=report.id)
    db.commit()
    return FileResponse(
        path,
        filename=report.filename,
        media_type=FORMATS.get(report.format, "application/octet-stream"),
    )

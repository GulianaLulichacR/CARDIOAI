from __future__ import annotations

import io
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd
from sqlalchemy import func
from sqlalchemy.orm import Session

from ..config import REPORT_DIR, settings
from ..demo_data import REGION_INFO
from ..ml_engine import ALGORITHMS
from ..models import Alert, AuditLog, Dataset, ModelRun, Patient, Prediction, User
from ..utils import iso

REPORT_TYPES = {
    "operational": "Reporte operacional",
    "management": "Reporte gerencial",
    "statistical": "Reporte estadístico",
    "predictions": "Reporte de predicciones",
    "model_performance": "Reporte de rendimiento de modelos",
    "explainability": "Reporte de explicabilidad",
    "audit": "Reporte de auditoría",
    "data_quality": "Reporte de calidad de datos",
}
FORMATS = {"pdf": "application/pdf", "docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document", "xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"}

DISCLAIMER = (
    "Documento generado por CARDIOAI con fines de investigación y demostración. Los datos y métricas "
    "provenientes del dataset sintético no constituyen evidencia clínica. No usar para diagnóstico "
    "ni para decisiones asistenciales sin validación externa y supervisión profesional."
)


def _kpis(db: Session, start: datetime) -> dict[str, Any]:
    patients = db.query(func.count(Patient.id)).filter(Patient.is_active.is_(True)).scalar() or 0
    preds_window = db.query(Prediction).filter(Prediction.created_at >= start).all()
    models = db.query(ModelRun).filter(ModelRun.status == "ready").count()
    datasets = db.query(Dataset).count()
    high = sum(1 for p in preds_window if p.risk_level == "alto")
    return {
        "Pacientes activos": patients,
        "Datasets registrados": datasets,
        "Modelos entrenados": models,
        "Predicciones en el periodo": len(preds_window),
        "Predicciones de riesgo alto": high,
        "Proporción de riesgo alto (%)": round(high / len(preds_window) * 100, 1) if preds_window else 0.0,
        "Riesgo medio estimado": round(sum(p.probability for p in preds_window) / len(preds_window), 4)
        if preds_window else 0.0,
    }


def _risk_distribution(predictions: list[Prediction]) -> tuple[list[str], list[list[str]]]:
    counts = {"bajo": 0, "moderado": 0, "alto": 0}
    for p in predictions:
        counts[p.risk_level] = counts.get(p.risk_level, 0) + 1
    return ["Nivel", "Predicciones", "Porcentaje"], [
        [
            level.capitalize(),
            str(value),
            f"{value / len(predictions) * 100:.1f}%" if predictions else "0%",
        ]
        for level, value in counts.items()
    ]


def _model_table(db: Session) -> tuple[list[str], list[list[str]]]:
    runs = db.query(ModelRun).filter(ModelRun.status == "ready").order_by(ModelRun.trained_at.desc()).all()
    latest: dict[str, ModelRun] = {}
    for run in runs:
        latest.setdefault(run.algorithm, run)
    headers = ["Modelo", "Algoritmo", "AUC-ROC", "F1", "MCC", "Sens.", "Espec.", "Brier", "C-index", "Demo"]
    rows = []
    for run in latest.values():
        disc = (run.metrics or {}).get("discrimination", {})
        rows.append(
            [
                run.name,
                run.algorithm,
                str(disc.get("roc_auc", "-")),
                str(disc.get("f1", "-")),
                str(disc.get("mcc", "-")),
                str(disc.get("sensitivity", "-")),
                str(disc.get("specificity", "-")),
                str(disc.get("brier", "-")),
                str((run.metrics or {}).get("c_index_temporal", "-")),
                "Sí" if run.is_demo else "No",
            ]
        )
    return headers, rows


def _patient_table(db: Session) -> tuple[list[str], list[list[str]]]:
    patients = db.query(Patient).filter(Patient.is_active.is_(True)).all()
    headers = ["Región", "Pacientes", "Edad media", "IMC medio", "PAS media", "PM2.5 medio", "% fumadores"]
    rows = []
    for region in REGION_INFO:
        group = [p for p in patients if p.region == region]
        if not group:
            continue
        rows.append(
            [
                region,
                str(len(group)),
                f"{sum(p.age for p in group) / len(group):.1f}",
                f"{sum(p.bmi for p in group) / len(group):.1f}",
                f"{sum(p.systolic_bp for p in group) / len(group):.1f}",
                f"{sum(p.pm25 for p in group) / len(group):.1f}",
                f"{sum(1 for p in group if p.smoking == 'fumador') / len(group) * 100:.1f}%",
            ]
        )
    return headers, rows


def _predictions_table(predictions: list[Prediction], limit: int = 40) -> tuple[list[str], list[list[str]]]:
    headers = ["Fecha", "Nivel", "Probabilidad", "Horizonte", "Modelo", "Región", "Demo"]
    rows = []
    for p in predictions[:limit]:
        rows.append(
            [
                iso(p.created_at) or "",
                p.risk_level,
                f"{p.probability:.1%}",
                f"{p.horizon_years} años",
                str(p.model_run_id),
                str((p.input or {}).get("region", "-")),
                "Sí" if p.is_demo else "No",
            ]
        )
    return headers, rows


def _importance_table(run: ModelRun) -> tuple[list[str], list[list[str]]]:
    headers = ["Variable", "Importancia relativa"]
    rows = [[item.get("label", item.get("feature", "")), f"{item.get('importance', 0):.4f}"]
            for item in (run.feature_importance or [])[:15]]
    return headers, rows


def _audit_table(db: Session, limit: int = 100) -> tuple[list[str], list[list[str]]]:
    logs = db.query(AuditLog).order_by(AuditLog.created_at.desc()).limit(limit).all()
    headers = ["Fecha", "Usuario", "Acción", "Módulo", "Entidad", "Resultado", "IP"]
    rows = [
        [
            iso(log.created_at) or "",
            log.user_email,
            log.action,
            log.module,
            f"{log.entity or '-'} {log.entity_id or ''}".strip(),
            log.result,
            log.ip or "-",
        ]
        for log in logs
    ]
    return headers, rows


def _quality_sections(db: Session) -> list[dict[str, Any]]:
    sections = []
    for dataset in db.query(Dataset).order_by(Dataset.created_at.desc()).all():
        quality = dataset.quality or {}
        headers = ["Indicador", "Valor"]
        rows = [
            ["Nombre", dataset.name],
            ["Fuente", dataset.source],
            ["Licencia", dataset.license],
            ["Filas", str(dataset.rows_count)],
            ["Columnas", str(dataset.columns_count)],
            ["Duplicados", str(quality.get("duplicate_rows", 0))],
            ["Valores faltantes (%)", f"{quality.get('missing_pct', 0)}%"],
            ["Columnas constantes", ", ".join(quality.get("constant_columns", [])) or "Ninguna"],
            ["Es demostración", "Sí" if dataset.is_demo else "No"],
            ["Checksum", dataset.checksum or "-"],
        ]
        sections.append(
            {
                "title": f"Dataset: {dataset.name}",
                "paragraphs": [dataset.description or ""],
                "tables": [{"headers": headers, "rows": rows}],
            }
        )
        missing = quality.get("missing_by_column", {})
        if missing:
            sections[-1]["tables"].append(
                {
                    "headers": ["Variable", "Faltantes", "% faltante"],
                    "rows": [[k, str(v.get("count", 0)), f"{v.get('pct', 0)}%"] for k, v in missing.items()],
                }
            )
    return sections


def _build_sections(db: Session, report_type: str, start: datetime) -> list[dict[str, Any]]:
    kpis = _kpis(db, start)
    predictions = (
        db.query(Prediction).filter(Prediction.created_at >= start).order_by(Prediction.created_at.desc()).all()
    )
    risk_headers, risk_rows = _risk_distribution(predictions)
    model_headers, model_rows = _model_table(db)
    sections: list[dict[str, Any]] = [
        {
            "title": "Indicadores generales",
            "paragraphs": [f"Periodo analizado: últimos {(datetime.utcnow() - start).days} días."],
            "tables": [{"headers": ["Indicador", "Valor"], "rows": [[k, str(v)] for k, v in kpis.items()]}],
        }
    ]
    if report_type in ("operational", "management", "statistical", "predictions"):
        sections.append(
            {
                "title": "Distribución de niveles de riesgo",
                "paragraphs": [
                    "Niveles asignados con umbrales configurables de demostración "
                    f"(bajo < {settings.risk_low_max:.0%}, alto ≥ {settings.risk_high_min:.0%})."
                ],
                "tables": [{"headers": risk_headers, "rows": risk_rows}],
                "chart": "risk",
            }
        )
    if report_type in ("operational", "predictions"):
        sections.append(
            {
                "title": "Predicciones recientes",
                "paragraphs": [],
                "tables": [{"headers": headers, "rows": rows}
                           for headers, rows in [_predictions_table(predictions, 40)]],
            }
        )
    if report_type in ("management", "statistical", "model_performance"):
        sections.append(
            {
                "title": "Rendimiento de modelos",
                "paragraphs": [
                    "Métricas calculadas sobre el conjunto temporal retenido. AUC-ROC y F1 se refieren al "
                    "evento dentro del horizonte del modelo."
                ],
                "tables": [{"headers": model_headers, "rows": model_rows}],
            }
        )
    if report_type in ("management", "statistical"):
        headers, rows = _patient_table(db)
        sections.append(
            {
                "title": "Perfil territorial de los registros",
                "paragraphs": ["Agregados por región; no se exponen identificadores individuales."],
                "tables": [{"headers": headers, "rows": rows}],
            }
        )
    if report_type == "model_performance":
        for run in db.query(ModelRun).filter(ModelRun.status == "ready").order_by(ModelRun.trained_at.desc()).all():
            metrics = run.metrics or {}
            disc = metrics.get("discrimination", {})
            temporal = metrics.get("temporal", {}).get("metrics", {})
            geo = metrics.get("geographic", {}).get("metrics", {})
            rows = [
                ["Modelo", run.name],
                ["Algoritmo", run.algorithm],
                ["Estado", run.status],
                ["Dataset demo", "Sí" if run.is_demo else "No"],
                ["Filas de entrenamiento", str(run.train_rows)],
                ["Filas de prueba temporal", str(run.test_rows)],
                ["Horizonte", f"{run.horizon_years} años"],
                ["AUC-ROC temporal", str(disc.get("roc_auc"))],
                ["Intervalo bootstrap AUC", str(disc.get("auc_ci"))],
                ["F1", str(disc.get("f1"))],
                ["MCC", str(disc.get("mcc"))],
                ["Sensibilidad", str(disc.get("sensitivity"))],
                ["Especificidad", str(disc.get("specificity"))],
                ["Brier", str(disc.get("brier"))],
                ["Pendiente de calibración", str(disc.get("calibration_slope"))],
                ["C-index temporal", str(metrics.get("c_index_temporal"))],
                ["AUC región retenida", str(geo.get("roc_auc"))],
                ["Estrategia de validación", str(metrics.get("validation_strategy"))],
            ]
            sections.append(
                {
                    "title": f"Detalle: {run.name}",
                    "paragraphs": [run.notes or ""],
                    "tables": [{"headers": ["Métrica", "Valor"], "rows": rows}],
                }
            )
    if report_type == "explainability":
        active = db.query(ModelRun).filter(ModelRun.is_active.is_(True)).first()
        if active:
            headers, rows = _importance_table(active)
            sections.append(
                {
                    "title": f"Importancia global — {active.name}",
                    "paragraphs": ["Importancia agregada por variable. No implica causalidad."],
                    "tables": [{"headers": headers, "rows": rows}],
                }
            )
        recent = [p for p in predictions if p.explanation and p.explanation.get("factors")][:10]
        for p in recent:
            rows = [
                [f.get("label", f.get("feature", "")), str(f.get("value", "-")), f"{f.get('contribution', 0):.4f}", f.get("direction", "")]
                for f in p.explanation.get("factors", [])[:8]
            ]
            sections.append(
                {
                    "title": f"Explicación local — predicción #{p.id}",
                    "paragraphs": [
                        f"Nivel {p.risk_level}; probabilidad {p.probability:.1%}; método: {p.explanation_method}."
                    ],
                    "tables": [{"headers": ["Variable", "Valor", "Contribución", "Dirección"], "rows": rows}],
                }
            )
    if report_type == "audit":
        headers, rows = _audit_table(db, 200)
        sections.append(
            {
                "title": "Bitácora de auditoría (extracto)",
                "paragraphs": [
                    "Registro append-only. La exportación completa está disponible desde el módulo de auditoría."
                ],
                "tables": [{"headers": headers, "rows": rows}],
            }
        )
    if report_type == "data_quality":
        sections.extend(_quality_sections(db))
    if report_type in ("operational", "management"):
        alerts = db.query(Alert).order_by(Alert.created_at.desc()).limit(20).all()
        sections.append(
            {
                "title": "Alertas y notificaciones",
                "paragraphs": [],
                "tables": [
                    {
                        "headers": ["Fecha", "Tipo", "Severidad", "Título", "Estado"],
                        "rows": [
                            [iso(a.created_at) or "", a.type, a.severity, a.title,
                             "Leída" if a.is_read else "No leída"]
                            for a in alerts
                        ],
                    }
                ],
            }
        )
    return sections


def _risk_chart(predictions: list[Prediction]) -> io.BytesIO | None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        counts = {"bajo": 0, "moderado": 0, "alto": 0}
        for p in predictions:
            counts[p.risk_level] = counts.get(p.risk_level, 0) + 1
        fig, ax = plt.subplots(figsize=(6, 3))
        colors = ["#2f855a", "#b7791f", "#c53030"]
        ax.bar([k.capitalize() for k in counts], list(counts.values()), color=colors)
        ax.set_ylabel("Predicciones")
        ax.set_title("Distribución de riesgo")
        ax.spines[["top", "right"]].set_visible(False)
        buffer = io.BytesIO()
        fig.tight_layout()
        fig.savefig(buffer, format="png", dpi=140)
        plt.close(fig)
        buffer.seek(0)
        return buffer
    except Exception:
        return None


def _render_pdf(path: Path, title: str, sections: list[dict[str, Any]], meta: dict[str, Any], predictions: list[Prediction]) -> None:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import mm
    from reportlab.platypus import Image, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="SmallMuted", parent=styles["Normal"], fontSize=8, textColor=colors.HexColor("#5b6b82")))
    doc = SimpleDocTemplate(str(path), pagesize=A4, leftMargin=18 * mm, rightMargin=18 * mm,
                            topMargin=18 * mm, bottomMargin=18 * mm, title=title)
    story: list[Any] = [
        Paragraph("CARDIOAI · Plataforma de investigación cardiovascular", styles["SmallMuted"]),
        Spacer(1, 4 * mm),
        Paragraph(title, styles["Title"]),
        Spacer(1, 3 * mm),
        Paragraph(
            f"Generado: {meta['generated_at']} · Usuario: {meta['user']} · Periodo: {meta['period']}",
            styles["SmallMuted"],
        ),
        Paragraph(f"Fuente: {meta['source']} · Entorno demo: {'Sí' if meta['demo'] else 'No'}", styles["SmallMuted"]),
        Spacer(1, 2 * mm),
        Paragraph(DISCLAIMER, styles["SmallMuted"]),
        Spacer(1, 6 * mm),
    ]
    chart = _risk_chart(predictions)
    for section in sections:
        story.append(Paragraph(section["title"], styles["Heading2"]))
        for text in section.get("paragraphs", []):
            if text:
                story.append(Paragraph(str(text), styles["BodyText"]))
        if section.get("chart") == "risk" and chart is not None:
            chart.seek(0)
            story.append(Image(chart, width=150 * mm, height=75 * mm))
        for table in section.get("tables", []):
            headers = [str(h) for h in table["headers"]]
            rows = table["rows"]
            if not rows:
                story.append(Paragraph("Sin registros en el periodo.", styles["SmallMuted"]))
                continue
            data = [headers] + [[str(c) for c in row] for row in rows[:35]]
            pdf_table = Table(data, repeatRows=1, hAlign="LEFT")
            pdf_table.setStyle(
                TableStyle(
                    [
                        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0b2545")),
                        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                        ("FONTSIZE", (0, 0), (-1, -1), 7.2),
                        ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#c9d4e3")),
                        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f4f7fb")]),
                        ("VALIGN", (0, 0), (-1, -1), "TOP"),
                        ("LEFTPADDING", (0, 0), (-1, -1), 3),
                        ("RIGHTPADDING", (0, 0), (-1, -1), 3),
                    ]
                )
            )
            story.append(pdf_table)
            story.append(Spacer(1, 3 * mm))
            if len(rows) > 35:
                story.append(Paragraph(f"Se muestran 35 de {len(rows)} filas. La versión XLSX contiene el detalle.", styles["SmallMuted"]))
        story.append(Spacer(1, 4 * mm))

    def footer(canvas, doc_obj):
        canvas.saveState()
        canvas.setFont("Helvetica", 7)
        canvas.setFillColor(colors.HexColor("#5b6b82"))
        canvas.drawString(18 * mm, 10 * mm, "CARDIOAI · Documento de investigación · No usar para diagnóstico")
        canvas.drawRightString(192 * mm, 10 * mm, f"Página {doc_obj.page}")
        canvas.restoreState()

    doc.build(story, onFirstPage=footer, onLaterPages=footer)


def _render_docx(path: Path, title: str, sections: list[dict[str, Any]], meta: dict[str, Any]) -> None:
    from docx import Document
    from docx.shared import Pt

    doc = Document()
    doc.add_heading(title, level=0)
    p = doc.add_paragraph(
        f"Generado: {meta['generated_at']} · Usuario: {meta['user']} · Periodo: {meta['period']}\n"
        f"Fuente: {meta['source']} · Entorno demo: {'Sí' if meta['demo'] else 'No'}"
    )
    p.runs[0].font.size = Pt(9)
    note = doc.add_paragraph(DISCLAIMER)
    note.runs[0].font.size = Pt(8)
    for section in sections:
        doc.add_heading(section["title"], level=1)
        for text in section.get("paragraphs", []):
            if text:
                doc.add_paragraph(str(text))
        for table in section.get("tables", []):
            rows = table["rows"][:100]
            if not rows:
                doc.add_paragraph("Sin registros en el periodo.")
                continue
            out = doc.add_table(rows=1, cols=len(table["headers"]))
            out.style = "Table Grid"
            for i, header in enumerate(table["headers"]):
                out.rows[0].cells[i].text = str(header)
            for row in rows:
                cells = out.add_row().cells
                for i, value in enumerate(row):
                    if i < len(cells):
                        cells[i].text = str(value)
    doc.save(str(path))


def _render_xlsx(path: Path, title: str, sections: list[dict[str, Any]], meta: dict[str, Any]) -> None:
    from openpyxl import Workbook
    from openpyxl.styles import Alignment, Font, PatternFill
    from openpyxl.utils import get_column_letter

    wb = Workbook()
    ws = wb.active
    ws.title = "Metadatos"
    ws.append(["CARDIOAI", title])
    ws.append(["Generado", meta["generated_at"]])
    ws.append(["Usuario", meta["user"]])
    ws.append(["Periodo", meta["period"]])
    ws.append(["Fuente", meta["source"]])
    ws.append(["Entorno demo", "Sí" if meta["demo"] else "No"])
    ws.append(["Aviso", DISCLAIMER])
    ws.column_dimensions["A"].width = 24
    ws.column_dimensions["B"].width = 100
    ws["B7"].alignment = Alignment(wrap_text=True, vertical="top")

    for idx, section in enumerate(sections, start=1):
        name = (section["title"][:28] or f"Hoja{idx}").replace("/", "-").replace("\\", "-")
        sheet = wb.create_sheet(title=name)
        sheet.append([section["title"]])
        sheet["A1"].font = Font(bold=True, size=13)
        for text in section.get("paragraphs", []):
            if text:
                sheet.append([str(text)])
        sheet.append([])
        for table in section.get("tables", []):
            sheet.append([str(h) for h in table["headers"]])
            header_row = sheet.max_row
            for cell in sheet[header_row]:
                cell.font = Font(bold=True, color="FFFFFF")
                cell.fill = PatternFill("solid", fgColor="0B2545")
            for row in table["rows"][:5000]:
                sheet.append([str(v) for v in row])
            sheet.append([])
        for col in range(1, 12):
            sheet.column_dimensions[get_column_letter(col)].width = 22
    wb.save(str(path))


def generate_report(db: Session, user: User, report_type: str, fmt: str, days: int) -> tuple[Path, dict[str, Any], str]:
    if report_type not in REPORT_TYPES:
        raise ValueError("Tipo de reporte no válido.")
    if fmt not in FORMATS:
        raise ValueError("Formato no válido. Use pdf, docx o xlsx.")
    start = datetime.utcnow() - timedelta(days=max(1, min(days, 3650)))
    predictions = db.query(Prediction).filter(Prediction.created_at >= start).all()
    active = db.query(ModelRun).filter(ModelRun.is_active.is_(True)).first()
    meta = {
        "generated_at": datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC"),
        "user": user.email,
        "period": f"{(datetime.utcnow() - start).days} días",
        "source": "Base de datos CARDIOAI + dataset sintético de demostración",
        "demo": bool(active.is_demo) if active else True,
    }
    sections = _build_sections(db, report_type, start)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    filename = f"cardioai_{report_type}_{stamp}.{fmt}"
    path = REPORT_DIR / filename
    title = REPORT_TYPES[report_type]
    if fmt == "pdf":
        _render_pdf(path, title, sections, meta, predictions)
    elif fmt == "docx":
        _render_docx(path, title, sections, meta)
    else:
        _render_xlsx(path, title, sections, meta)
    return path, {"days": days, "report_type": report_type, "title": title}, title

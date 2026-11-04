from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from sqlalchemy import (
    JSON,
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from .database import Base


def utcnow() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


class User(Base):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(primary_key=True)
    email: Mapped[str] = mapped_column(String(255), unique=True, index=True)
    full_name: Mapped[str] = mapped_column(String(160))
    role: Mapped[str] = mapped_column(String(40), index=True)
    password_hash: Mapped[str] = mapped_column(String(255))
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    must_change_password: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)
    last_login_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    last_login_ip: Mapped[str | None] = mapped_column(String(64), nullable=True)


class AuditLog(Base):
    """Bitácora append-only: no se expone edición ni borrado por API."""

    __tablename__ = "audit_log"
    __table_args__ = (Index("ix_audit_created_user", "created_at", "user_id"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow, index=True)
    user_id: Mapped[int | None] = mapped_column(ForeignKey("users.id"), nullable=True)
    user_email: Mapped[str] = mapped_column(String(255), default="anonimo")
    action: Mapped[str] = mapped_column(String(80), index=True)
    module: Mapped[str] = mapped_column(String(60), index=True)
    entity: Mapped[str | None] = mapped_column(String(80), nullable=True)
    entity_id: Mapped[str | None] = mapped_column(String(80), nullable=True)
    result: Mapped[str] = mapped_column(String(30), default="success")
    detail: Mapped[str | None] = mapped_column(Text, nullable=True)
    ip: Mapped[str | None] = mapped_column(String(64), nullable=True)
    session_id: Mapped[str | None] = mapped_column(String(64), index=True)
    request_id: Mapped[str | None] = mapped_column(String(64), nullable=True)


class Dataset(Base):
    __tablename__ = "datasets"

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(160), unique=True)
    source: Mapped[str] = mapped_column(String(160), default="Carga manual")
    license: Mapped[str] = mapped_column(String(160), default="No especificada")
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    version: Mapped[str] = mapped_column(String(30), default="1.0")
    is_demo: Mapped[bool] = mapped_column(Boolean, default=False)
    status: Mapped[str] = mapped_column(String(30), default="ready")
    rows_count: Mapped[int] = mapped_column(Integer, default=0)
    columns_count: Mapped[int] = mapped_column(Integer, default=0)
    file_path: Mapped[str | None] = mapped_column(String(500), nullable=True)
    checksum: Mapped[str | None] = mapped_column(String(64), nullable=True)
    quality: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    uploaded_by: Mapped[int | None] = mapped_column(ForeignKey("users.id"), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow, onupdate=utcnow)

    variables: Mapped[list[DatasetVariable]] = relationship(
        back_populates="dataset", cascade="all, delete-orphan"
    )


class DatasetVariable(Base):
    __tablename__ = "dataset_variables"
    __table_args__ = (UniqueConstraint("dataset_id", "name", name="uq_dataset_variable"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    dataset_id: Mapped[int] = mapped_column(ForeignKey("datasets.id", ondelete="CASCADE"))
    name: Mapped[str] = mapped_column(String(120))
    label: Mapped[str | None] = mapped_column(String(200), nullable=True)
    dtype: Mapped[str] = mapped_column(String(30), default="float")
    unit: Mapped[str | None] = mapped_column(String(40), nullable=True)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    missing_count: Mapped[int] = mapped_column(Integer, default=0)
    missing_pct: Mapped[float] = mapped_column(Float, default=0.0)
    min_value: Mapped[float | None] = mapped_column(Float, nullable=True)
    max_value: Mapped[float | None] = mapped_column(Float, nullable=True)
    mean_value: Mapped[float | None] = mapped_column(Float, nullable=True)
    unique_count: Mapped[int] = mapped_column(Integer, default=0)

    dataset: Mapped[Dataset] = relationship(back_populates="variables")


class Patient(Base):
    __tablename__ = "patients"

    id: Mapped[int] = mapped_column(primary_key=True)
    code: Mapped[str] = mapped_column(String(40), unique=True, index=True)
    age: Mapped[int] = mapped_column(Integer)
    sex: Mapped[str] = mapped_column(String(20))
    region: Mapped[str] = mapped_column(String(80), index=True)
    urban: Mapped[bool] = mapped_column(Boolean, default=True)
    education: Mapped[str] = mapped_column(String(40), default="secundaria")
    income_level: Mapped[str] = mapped_column(String(40), default="medio")
    smoking: Mapped[str] = mapped_column(String(30), default="nunca")
    physical_activity_min: Mapped[int] = mapped_column(Integer, default=150)
    alcohol_units: Mapped[float] = mapped_column(Float, default=0.0)
    bmi: Mapped[float] = mapped_column(Float)
    systolic_bp: Mapped[int] = mapped_column(Integer)
    diastolic_bp: Mapped[int] = mapped_column(Integer)
    cholesterol_total: Mapped[float] = mapped_column(Float)
    hdl: Mapped[float] = mapped_column(Float)
    glucose: Mapped[float] = mapped_column(Float)
    diabetes: Mapped[bool] = mapped_column(Boolean, default=False)
    hypertension: Mapped[bool] = mapped_column(Boolean, default=False)
    family_history: Mapped[bool] = mapped_column(Boolean, default=False)
    pm25: Mapped[float] = mapped_column(Float, default=12.0)
    green_space: Mapped[float] = mapped_column(Float, default=20.0)
    stress_level: Mapped[int] = mapped_column(Integer, default=3)
    event: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    followup_years: Mapped[float | None] = mapped_column(Float, nullable=True)
    cohort_year: Mapped[int | None] = mapped_column(Integer, nullable=True)
    notes: Mapped[str | None] = mapped_column(Text, nullable=True)
    is_demo: Mapped[bool] = mapped_column(Boolean, default=False)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    created_by: Mapped[int | None] = mapped_column(ForeignKey("users.id"), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow, onupdate=utcnow)


class ModelRun(Base):
    __tablename__ = "model_runs"

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(160))
    algorithm: Mapped[str] = mapped_column(String(60), index=True)
    version: Mapped[str] = mapped_column(String(20), default="1")
    status: Mapped[str] = mapped_column(String(30), default="queued", index=True)
    is_demo: Mapped[bool] = mapped_column(Boolean, default=True)
    dataset_id: Mapped[int | None] = mapped_column(ForeignKey("datasets.id"), nullable=True)
    metrics: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    params: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    feature_importance: Mapped[list[dict[str, Any]]] = mapped_column(JSON, default=list)
    artifact_path: Mapped[str | None] = mapped_column(String(500), nullable=True)
    horizon_years: Mapped[int] = mapped_column(Integer, default=10)
    risk_thresholds: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    notes: Mapped[str | None] = mapped_column(Text, nullable=True)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    mlflow_run_id: Mapped[str | None] = mapped_column(String(80), nullable=True)
    is_active: Mapped[bool] = mapped_column(Boolean, default=False, index=True)
    trained_by: Mapped[int | None] = mapped_column(ForeignKey("users.id"), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)
    trained_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    duration_seconds: Mapped[float | None] = mapped_column(Float, nullable=True)
    train_rows: Mapped[int | None] = mapped_column(Integer, nullable=True)
    test_rows: Mapped[int | None] = mapped_column(Integer, nullable=True)
    validation_strategy: Mapped[str | None] = mapped_column(String(200), nullable=True)


class Prediction(Base):
    __tablename__ = "predictions"

    id: Mapped[int] = mapped_column(primary_key=True)
    patient_id: Mapped[int | None] = mapped_column(ForeignKey("patients.id"), nullable=True)
    model_run_id: Mapped[int] = mapped_column(ForeignKey("model_runs.id"))
    input: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    probability: Mapped[float] = mapped_column(Float)
    risk_level: Mapped[str] = mapped_column(String(20), index=True)
    horizon_years: Mapped[int] = mapped_column(Integer, default=10)
    threshold: Mapped[float | None] = mapped_column(Float, nullable=True)
    explanation: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    explanation_method: Mapped[str] = mapped_column(String(80), default="SHAP")
    is_demo: Mapped[bool] = mapped_column(Boolean, default=True)
    batch_id: Mapped[str | None] = mapped_column(String(40), nullable=True, index=True)
    created_by: Mapped[int | None] = mapped_column(ForeignKey("users.id"), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow, index=True)


class Alert(Base):
    __tablename__ = "alerts"

    id: Mapped[int] = mapped_column(primary_key=True)
    type: Mapped[str] = mapped_column(String(60), default="sistema")
    severity: Mapped[str] = mapped_column(String(20), default="info")
    title: Mapped[str] = mapped_column(String(200))
    message: Mapped[str] = mapped_column(Text)
    module: Mapped[str | None] = mapped_column(String(60), nullable=True)
    is_read: Mapped[bool] = mapped_column(Boolean, default=False)
    created_by: Mapped[int | None] = mapped_column(ForeignKey("users.id"), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow, index=True)
    read_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)


class Report(Base):
    __tablename__ = "reports"

    id: Mapped[int] = mapped_column(primary_key=True)
    report_type: Mapped[str] = mapped_column(String(60))
    format: Mapped[str] = mapped_column(String(10))
    filename: Mapped[str] = mapped_column(String(255))
    file_path: Mapped[str] = mapped_column(String(500))
    filters: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    size_bytes: Mapped[int] = mapped_column(Integer, default=0)
    created_by: Mapped[int | None] = mapped_column(ForeignKey("users.id"), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow, index=True)


class KnowledgeDocument(Base):
    __tablename__ = "knowledge_documents"

    id: Mapped[int] = mapped_column(primary_key=True)
    title: Mapped[str] = mapped_column(String(255))
    source: Mapped[str] = mapped_column(String(160))
    url: Mapped[str | None] = mapped_column(String(500), nullable=True)
    year: Mapped[int | None] = mapped_column(Integer, nullable=True)
    content: Mapped[str] = mapped_column(Text)
    tags: Mapped[str] = mapped_column(String(255), default="")
    is_verified_source: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)

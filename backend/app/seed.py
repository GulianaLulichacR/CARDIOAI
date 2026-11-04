from __future__ import annotations

import pandas as pd
from sqlalchemy import func
from sqlalchemy.orm import Session

from .config import DEMO_DIR, settings
from .demo_data import (
    FEATURE_DESCRIPTIONS,
    FEATURE_LABELS,
    ensure_demo_dataset,
    file_checksum,
    quality_report,
)
from .models import Alert, Dataset, DatasetVariable, KnowledgeDocument, Patient, User
from .security import hash_password

DEMO_USERS = [
    ("admin@cardioai.local", "Administrador CARDIOAI", "superadmin", settings.admin_password),
    ("investigador@cardioai.local", "Investigadora Principal", "investigador", "Investiga2026!"),
    ("clinico@cardioai.local", "Profesional de Salud", "profesional_salud", "Clinico2026!"),
    ("analista@cardioai.local", "Analista de Datos", "analista_datos", "Analista2026!"),
    ("auditor@cardioai.local", "Auditor de Investigación", "auditor", "Auditor2026!"),
]

KNOWLEDGE_DOCUMENTS = [
    {
        "title": "Cardiovascular diseases (CVDs) — hoja informativa",
        "source": "Organización Mundial de la Salud (OMS)",
        "url": "https://www.who.int/news-room/fact-sheets/detail/cardiovascular-diseases-(cvds)",
        "year": 2021,
        "tags": "cardiovascular,epidemiologia,factores de riesgo",
        "content": (
            "Las enfermedades cardiovasculares (ECV) son un grupo de trastornos del corazón y los vasos "
            "sanguíneos. La OMS las describe como la principal causa de muerte en el mundo: aproximadamente "
            "17,9 millones de muertes al año, cerca del 32% de todas las muertes globales, y alrededor del 85% "
            "de esas muertes se atribuyen a infarto de miocardio y accidente cerebrovascular. La hoja informativa "
            "señala factores de riesgo conductuales y metabólicos: tabaquismo, dieta no saludable, obesidad, "
            "inactividad física, consumo nocivo de alcohol, hipertensión, diabetes e hiperlipidemia. También "
            "menciona la contaminación del aire como factor de riesgo cardiovascular y la importancia de la "
            "detección temprana y el control de la presión arterial. Fuente primaria verificable; este resumen "
            "no sustituye el documento oficial."
        ),
    },
    {
        "title": "HEARTS technical package for cardiovascular disease management",
        "source": "Organización Mundial de la Salud (OMS)",
        "url": "https://www.who.int/publications/i/item/9789240001367",
        "year": 2020,
        "tags": "hipertension,atencion primaria,protocolos",
        "content": (
            "HEARTS es un paquete técnico de la OMS para fortalecer el manejo de la hipertensión en atención "
            "primaria. Sus componentes incluyen hábitos saludables, protocolos estandarizados de medición y "
            "tratamiento, acceso a medicamentos esenciales, atención centrada en el paciente y sistemas de "
            "monitoreo. El paquete enfatiza la medición estandarizada de la presión arterial, el uso de "
            "protocolos simples de tratamiento escalonado y el seguimiento sistemático de cohortes. Para un "
            "sistema de predicción de riesgo, HEARTS aporta criterios de medición reproducible y recordatorios "
            "de que las estimaciones deben acompañarse de evaluación profesional, no de decisiones autónomas."
        ),
    },
    {
        "title": "Heart Disease Facts",
        "source": "Centers for Disease Control and Prevention (CDC)",
        "url": "https://www.cdc.gov/heart-disease/data-research/facts-stats/index.html",
        "year": 2024,
        "tags": "cardiovascular,estados unidos,vigilancia",
        "content": (
            "La página de datos y estadísticas del CDC describe la enfermedad cardíaca como una de las "
            "principales causas de muerte en Estados Unidos y resume factores de riesgo como hipertensión, "
            "colesterol elevado, tabaquismo, diabetes, obesidad, inactividad física, dieta poco saludable y "
            "consumo excesivo de alcohol. El CDC recomienda el control de la presión arterial y del colesterol, "
            "la actividad física regular, una alimentación saludable y el abandono del tabaco. Las cifras "
            "puntuales cambian con cada actualización; debe consultarse la fuente original antes de citarlas."
        ),
    },
    {
        "title": "2019 ACC/AHA Guideline on the Primary Prevention of Cardiovascular Disease",
        "source": "American College of Cardiology / American Heart Association",
        "url": "https://www.ahajournals.org/doi/10.1161/CIR.0000000000000678",
        "year": 2019,
        "tags": "guia,prevencion,riesgo cardiovascular",
        "content": (
            "La guía ACC/AHA de 2019 para prevención primaria enfatiza la evaluación del riesgo cardiovascular "
            "a 10 años, la promoción de estilos de vida saludables, el manejo de factores de riesgo y la toma "
            "compartida de decisiones. Recomienda estimar el riesgo con ecuaciones validadas, revisar la "
            "calibración en la población local y considerar factores de riesgo adicionales cuando la estimación "
            "es intermedia. La guía subraya que las decisiones farmacológicas deben individualizarse y que la "
            "predicción de riesgo es una herramienta de apoyo, no un diagnóstico. No se debe trasladar un umbral "
            "de una guía a otra población sin recalibración."
        ),
    },
    {
        "title": "SCORE2: Systematic COronary Risk Evaluation 2",
        "source": "European Society of Cardiology (ESC)",
        "url": "https://academic.oup.com/eurheartj/article/42/25/2439/6296097",
        "year": 2021,
        "tags": "riesgo cardiovascular,validacion,calibracion",
        "content": (
            "SCORE2 es un modelo de predicción del riesgo cardiovascular a 10 años en Europa, publicado por la "
            "Sociedad Europea de Cardiología en 2021. Utiliza edad, sexo, tabaquismo, presión arterial sistólica "
            "y colesterol para estimar el riesgo de eventos cardiovasculares fatales y no fatales. El modelo se "
            "recalibró por regiones y se acompaña de SCORE2-OP para personas mayores. El artículo ilustra la "
            "importancia de la recalibración local, la validación externa y la comunicación del riesgo absoluto. "
            "Los umbrales de decisión son específicos de cada guía y población."
        ),
    },
    {
        "title": "Framingham Risk Score — General Cardiovascular Risk Profile",
        "source": "D'Agostino et al., Circulation",
        "url": "https://www.ahajournals.org/doi/10.1161/CIRCULATIONAHA.107.699579",
        "year": 2008,
        "tags": "framingham,riesgo cardiovascular,modelo",
        "content": (
            "El perfil de riesgo cardiovascular general de Framingham estima la probabilidad de un evento "
            "cardiovascular mayor a 10 años a partir de edad, sexo, colesterol total, colesterol HDL, presión "
            "arterial sistólica, tratamiento antihipertensivo, tabaquismo y diabetes. Es un antecedente clásico "
            "de los modelos de riesgo y muestra la utilidad de combinar factores clínicos y conductuales. Su "
            "rendimiento varía al aplicarse en poblaciones distintas de la cohorte original, por lo que requiere "
            "validación y recalibración. No incorpora de forma nativa factores ambientales ni desigualdad "
            "territorial, una de las brechas que motivan este proyecto."
        ),
    },
    {
        "title": "TRIPOD+AI statement: reporting guideline for prediction models",
        "source": "BMJ (Collins et al.)",
        "url": "https://www.bmj.com/content/385/bmj-2023-078378",
        "year": 2024,
        "tags": "metodologia,reporte,validacion,sesgo",
        "content": (
            "TRIPOD+AI es una guía de reporte para estudios que desarrollan o evalúan modelos de predicción "
            "con inteligencia artificial. Extiende la declaración TRIPOD e insiste en describir la población, "
            "el desenlace, los predictores, el flujo de datos, el preprocesamiento, la validación interna y "
            "externa, la calibración, la discriminación, el análisis de subgrupos y la disponibilidad del "
            "código. Recomienda informar el uso previsto, los riesgos de sesgo y las limitaciones, y evitar "
            "presentar modelos no validados como herramientas clínicas. Es un marco de referencia para la "
            "documentación científica de esta plataforma."
        ),
    },
    {
        "title": "A Unified Approach to Interpreting Model Predictions (SHAP)",
        "source": "Lundberg y Lee, NeurIPS",
        "url": "https://arxiv.org/abs/1705.07874",
        "year": 2017,
        "tags": "explicabilidad,shap,interpretabilidad",
        "content": (
            "SHAP (SHapley Additive exPlanations) propone un marco unificado para atribuir la predicción de un "
            "modelo a sus variables de entrada, basado en valores de Shapley de la teoría de juegos. Las "
            "contribuciones son aditivas respecto a una predicción base y permiten explicaciones locales y "
            "globales. TreeSHAP hace tratable el cálculo exacto en modelos de árboles. Las explicaciones "
            "describen el comportamiento del modelo, no relaciones causales, y deben presentarse junto con las "
            "limitaciones del modelo y de los datos."
        ),
    },
    {
        "title": "WHO global air quality guidelines (2021): PM2.5",
        "source": "Organización Mundial de la Salud (OMS)",
        "url": "https://www.who.int/publications/i/item/9789240034228",
        "year": 2021,
        "tags": "contaminacion,pm2.5,ambiente,salud publica",
        "content": (
            "Las guías mundiales de calidad del aire de la OMS de 2021 recomiendan un valor guía anual de "
            "5 µg/m³ para PM2.5 y establecen objetivos intermedios para apoyar la reducción progresiva de la "
            "contaminación. La exposición a material particulado se asocia con enfermedad cardiovascular y "
            "respiratoria. Para la investigación territorial, las concentraciones deben provenir de estaciones "
            "de monitoreo, modelos de calidad del aire o fuentes oficiales documentadas; no deben imputarse "
            "valores ambientales sin trazabilidad. La fuente original contiene los valores y las condiciones "
            "de aplicación."
        ),
    },
    {
        "title": "PROBAST: risk of bias assessment for prediction models",
        "source": "BMJ (Wolff et al.)",
        "url": "https://www.bmj.com/content/364/bmj.l4898",
        "year": 2019,
        "tags": "sesgo,validacion,metodologia",
        "content": (
            "PROBAST es una herramienta para evaluar el riesgo de sesgo y la aplicabilidad de estudios de "
            "desarrollo y validación de modelos de predicción. Evalúa cuatro dominios: participantes, "
            "predictores, desenlace y análisis. Entre los problemas que señala están el uso de un único "
            "conjunto de datos sin validación, la selección de variables basada en el desenlace, el "
            "sobreajuste, la exclusión injustificada de participantes y la falta de calibración. Es útil para "
            "documentar las limitaciones de esta plataforma y para planificar una validación externa."
        ),
    },
]


def _seed_users(db: Session) -> None:
    if db.query(func.count(User.id)).scalar():
        return
    for email, name, role, password in DEMO_USERS:
        db.add(
            User(
                email=email,
                full_name=name,
                role=role,
                password_hash=hash_password(password),
                is_active=True,
                must_change_password=False,
            )
        )
    db.flush()


def _seed_demo_dataset(db: Session) -> Dataset:
    dataset = db.query(Dataset).filter(Dataset.name == "Demostración sintética CARDIOAI").one_or_none()
    path, df = ensure_demo_dataset()
    if dataset is None:
        dataset = Dataset(
            name="Demostración sintética CARDIOAI",
            source="Generado por CARDIOAI (semilla fija)",
            license="Uso exclusivo de demostración",
            description=(
                "Cohorte sintética de 1.200 registros generada con semilla fija para demostrar el "
                "funcionamiento técnico. No son datos reales ni resultados clínicos."
            ),
            version="1.0",
            is_demo=True,
            status="ready",
            rows_count=int(len(df)),
            columns_count=int(df.shape[1]),
            file_path=str(path),
            checksum=file_checksum(path),
            quality=quality_report(df),
        )
        db.add(dataset)
        db.flush()
        for col in df.columns:
            series = df[col]
            numeric = pd.to_numeric(series, errors="coerce")
            is_numeric = numeric.notna().sum() == series.notna().sum() and series.notna().any()
            db.add(
                DatasetVariable(
                    dataset_id=dataset.id,
                    name=col,
                    label=FEATURE_LABELS.get(col, col.replace("_", " ").capitalize()),
                    dtype=str(series.dtype),
                    unit=None,
                    description=FEATURE_DESCRIPTIONS.get(col),
                    missing_count=int(series.isna().sum()),
                    missing_pct=round(float(series.isna().mean() * 100), 2),
                    min_value=float(numeric.min()) if is_numeric else None,
                    max_value=float(numeric.max()) if is_numeric else None,
                    mean_value=float(numeric.mean()) if is_numeric else None,
                    unique_count=int(series.nunique(dropna=True)),
                )
            )
        db.flush()
    return dataset


def _seed_patients(db: Session) -> None:
    if db.query(func.count(Patient.id)).scalar():
        return
    _, df = ensure_demo_dataset()
    sample = df.sample(min(60, len(df)), random_state=settings.demo_seed)
    admin = db.query(User).filter(User.role == "superadmin").first()
    for _, row in sample.iterrows():
        db.add(
            Patient(
                code=str(row["patient_code"]).replace("DEMO", "PAC"),
                age=int(row["age"]),
                sex=str(row["sex"]),
                region=str(row["region"]),
                urban=bool(row["urban"]),
                education=str(row["education"]),
                income_level=str(row["income_level"]),
                smoking=str(row["smoking"]),
                physical_activity_min=int(row["physical_activity_min"]),
                alcohol_units=float(row["alcohol_units"]),
                bmi=float(row["bmi"]),
                systolic_bp=int(row["systolic_bp"]),
                diastolic_bp=int(row["diastolic_bp"]),
                cholesterol_total=float(row["cholesterol_total"]),
                hdl=float(row["hdl"]),
                glucose=float(row["glucose"]),
                diabetes=bool(row["diabetes"]),
                hypertension=bool(row["hypertension"]),
                family_history=bool(row["family_history"]),
                pm25=float(row["pm25"]),
                green_space=float(row["green_space"]),
                stress_level=int(row["stress_level"]),
                event=bool(row["event"]),
                followup_years=float(row["followup_years"]),
                cohort_year=int(row["cohort_year"]),
                notes="Registro sintético de demostración.",
                is_demo=True,
                is_active=True,
                created_by=admin.id if admin else None,
            )
        )
    db.flush()


def _seed_documents(db: Session) -> None:
    if db.query(func.count(KnowledgeDocument.id)).scalar():
        return
    for doc in KNOWLEDGE_DOCUMENTS:
        db.add(KnowledgeDocument(**doc))
    db.flush()


def _seed_alerts(db: Session) -> None:
    if db.query(func.count(Alert.id)).scalar():
        return
    db.add(
        Alert(
            type="sistema",
            severity="info",
            title="Entorno de demostración",
            message=(
                "La plataforma está operando con un dataset SINTÉTICO de demostración. "
                "Las métricas y predicciones no son evidencia clínica ni sustituyen la evaluación profesional."
            ),
            module="dashboard",
        )
    )
    db.flush()


def seed_all(db: Session) -> None:
    _seed_users(db)
    _seed_demo_dataset(db)
    _seed_patients(db)
    _seed_documents(db)
    _seed_alerts(db)
    db.commit()

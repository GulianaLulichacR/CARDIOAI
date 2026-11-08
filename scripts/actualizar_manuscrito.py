"""Actualiza el manuscrito CARDIOAI con resultados REALES de Framingham.

Lee data/results/*.csv (resultados reproducibles del motor con Framingham)
y genera un documento .docx basado en la plantilla existente, sustituyendo
los valores hipotéticos (Tablas 3-5, Figuras 3-6) por salidas observadas.

Uso:
    .venv\\Scripts\\python.exe scripts\\actualizar_manuscrito.py
"""
from __future__ import annotations

import csv
from pathlib import Path

from docx import Document
from docx.shared import Pt, Inches, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT

ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = ROOT / "Plantillas" / "Manuscrito_CARDIOAI_Q1_PlantillaMejorada.docx"
OUT_DOCX = ROOT / "Manuscrito_CARDIOAI_Actualizado_Framingham.docx"

# ---------------------------------------------------------------------------
# Cargar resultados reales de Framingham (data/results/)
# ---------------------------------------------------------------------------
RESULTS = ROOT / "data" / "results"


def _read_csv(name: str) -> list[dict]:
    p = RESULTS / name
    if not p.exists():
        return []
    with p.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


comparison = {r["model"]: r for r in _read_csv("model_comparison.csv")}
delong = _read_csv("table5_delong.csv")
calib = {r["model"]: r for r in _read_csv("table3_calibration.csv")}
subgroups = _read_csv("table4_subgroups.csv")
table1 = _read_csv("table1_baseline.csv")


def fnum(v, digits=3):
    if v is None or v == "" or v == "None":
        return "—"
    try:
        return f"{float(v):.{digits}f}"
    except ValueError:
        return str(v)


def ci(model):
    r = comparison.get(model, {})
    low, high = r.get("auc_ci_low"), r.get("auc_ci_high")
    auc = r.get("auc_test")
    if auc and low and high:
        return f"{float(auc):.3f} ({float(low):.3f}–{float(high):.3f})"
    return fnum(auc)


# ---------------------------------------------------------------------------
# Construir documento
# ---------------------------------------------------------------------------
doc = Document()

# Estilos base
normal = doc.styles["Normal"]
normal.font.name = "Calibri"
normal.font.size = Pt(11)

# Título (usando el título de la plantilla, actualizado)
title = doc.add_paragraph()
title.alignment = WD_ALIGN_PARAGRAPH.CENTER
run = title.add_run("Protocolo de evaluación de biomarcadores cardiometabólicos y factores socioambientales para predecir riesgo cardiovascular con datos de Framingham")
run.bold = True
run.font.size = Pt(16)

sub = doc.add_paragraph()
sub.alignment = WD_ALIGN_PARAGRAPH.CENTER
run = sub.add_run("Evaluation protocol for cardiometabolic biomarkers and socio-environmental factors to predict cardiovascular risk using Framingham Heart Study data")
run.italic = True
run.font.size = Pt(11)

# Aviso de integridad
aviso = doc.add_paragraph()
run = aviso.add_run(
    "AVISO DE INTEGRIDAD CIENTÍFICA. Todas las métricas de rendimiento, comparaciones de modelos, "
    "intervalos de confianza y curvas incluidas en este manuscrito proceden de una ejecución real y "
    "reproducible del motor CARDIOAI sobre el dataset público Framingham Heart Study (4.240 registros, "
    "644 eventos, prevalencia 15,19 %), entrenado con el script scripts/run_experiment.py y exportado con "
    "scripts/export_results.py. No son valores hipotéticos ni simulados."
)
run.font.size = Pt(9)
run.font.color.rgb = RGBColor(0x8B, 0x00, 0x00)

doc.add_paragraph()


def h1(text: str):
    p = doc.add_paragraph()
    r = p.add_run(text)
    r.bold = True
    r.font.size = Pt(13)
    p.space_after = Pt(6)
    return p


def h2(text: str):
    p = doc.add_paragraph()
    r = p.add_run(text)
    r.bold = True
    r.font.size = Pt(11.5)
    p.space_after = Pt(4)
    return p


def para(text: str, bold=False, italic=False, size=11, align=None):
    p = doc.add_paragraph()
    if align:
        p.alignment = align
    r = p.add_run(text)
    r.bold = bold
    r.italic = italic
    r.font.size = Pt(size)
    return p


def add_table(headers, rows, caption=None, note=None, col_widths=None):
    if caption:
        cap = doc.add_paragraph()
        r = cap.add_run(caption)
        r.bold = True
        r.font.size = Pt(10.5)
    t = doc.add_table(rows=1, cols=len(headers))
    t.style = "Table Grid"
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    hdr = t.rows[0].cells
    for i, h in enumerate(headers):
        hdr[i].text = str(h)
        for pr in hdr[i].paragraphs:
            for rn in pr.runs:
                rn.bold = True
                rn.font.size = Pt(9)
    for row in rows:
        cells = t.add_row().cells
        for i, val in enumerate(row):
            cells[i].text = str(val)
            for pr in cells[i].paragraphs:
                for rn in pr.runs:
                    rn.font.size = Pt(9)
    if col_widths:
        for i, w in enumerate(col_widths):
            for row in t.rows:
                row.cells[i].width = Inches(w)
    if note:
        n = doc.add_paragraph()
        r = n.add_run("Nota. " + note)
        r.italic = True
        r.font.size = Pt(8.5)
    doc.add_paragraph()
    return t


# ===========================================================================
# RESUMEN
# ===========================================================================
h1("Resumen")
para(
    "La predicción del riesgo cardiovascular se apoya principalmente en factores clínicos, aunque los "
    "determinantes sociales y ambientales podrían aportar información contextual cuya medición y "
    "transportabilidad son heterogéneas. El objetivo fue definir y documentar un protocolo para evaluar la "
    "integración de variables cardiometabólicas, conductuales y socioambientales en la plataforma CARDIOAI, "
    "delimitando con transparencia el alcance de su implementación. Se realizó una inspección técnica del "
    "código y un perfil descriptivo del dataset público Framingham Heart Study (4.240 registros, 29 columnas "
    "tras adaptación, 644 eventos, prevalencia 15,19 %). El protocolo contempló cinco familias de modelos, "
    "partición honesta (aleatoria estratificada al no existir año de cohorte en Framingham) y métricas de "
    "discriminación, calibración y utilidad. La cohorte generó 644 eventos (15,19 %); 848 observaciones de "
    "prueba de holdout tuvieron etiqueta binaria evaluable a diez años. XGBoost-Cox alcanzó la mayor "
    "AUC-ROC de prueba (0,703; IC 95 %: 0,658–0,754), seguido de Cox PH (0,700; 0,655–0,752) y Random Forest "
    "(0,672; 0,625–0,720). Las diferencias entre modelos fueron pequeñas respecto de la incertidumbre del "
    "intervalo de bootstrap y se contrastaron con la prueba de DeLong pareada; ninguna diferencia fue lo "
    "suficientemente grande y precisa como para afirmar superioridad general. La plataforma mostró una "
    "arquitectura capaz de integrar variables clínicas y contextuales; sin embargo, Framingham no contiene "
    "PM2.5, espacio verde, ingreso, actividad física ni visitas longitudinales, por lo que el módulo "
    "socioambiental no pudo validarse con esta fuente y los resultados no constituyen evidencia clínica, "
    "requiriendo validación externa prospectiva con datos observados."
)
para("Palabras clave: aprendizaje automático, calibración, Framingham, determinantes sociales de la salud, validación externa.", italic=True, size=10)

doc.add_paragraph()

h1("Abstract")
para(
    "Cardiovascular risk prediction relies mainly on clinical factors, although social and environmental "
    "determinants may add contextual information whose measurement and transportability are heterogeneous. "
    "The objective was to define and document a protocol for evaluating the integration of cardiometabolic, "
    "behavioral, and socio-environmental variables in the CARDIOAI platform, transparently delimiting the "
    "scope of its implementation. A technical inspection of the source code (React, TypeScript, and FastAPI) "
    "and a descriptive profile of the public Framingham Heart Study dataset (4,240 records, 29 columns after "
    "adaptation, 644 events, 15.19% prevalence) were carried out. The protocol comprised five model families, "
    "honest partitioning (stratified random holdout as Framingham lacks a cohort year) and metrics of "
    "discrimination, calibration, and utility. The cohort generated 644 events (15.19%); 848 holdout-test "
    "observations had an evaluable 10-year binary label. XGBoost-Cox achieved the highest test AUC-ROC "
    "(0.703; 95% CI: 0.658–0.754), followed by Cox PH (0.700; 0.655–0.752) and Random Forest (0.672; "
    "0.625–0.720). Differences between models were small relative to bootstrap uncertainty and were compared "
    "with paired DeLong tests; no difference was large and precise enough to claim general superiority. The "
    "platform showed an architecture able to integrate clinical and contextual variables; however, Framingham "
    "does not contain PM2.5, green space, income, physical activity, or longitudinal visits, so the "
    "socio-environmental module could not be validated with this source, and results do not constitute "
    "clinical evidence, requiring prospective external validation with observed data."
)
para("Keywords: machine learning, calibration, Framingham, social determinants of health, external validation.", italic=True, size=10)

doc.add_paragraph()

# ===========================================================================
# INTRODUCCIÓN
# ===========================================================================
h1("1. Introducción")
para(
    "Las enfermedades cardiovasculares (ECV) siguen siendo la principal causa de muerte en el mundo. La "
    "Organización Mundial de la Salud (World Health Organization [WHO], s. f.) estimó 19,8 millones de muertes "
    "por ECV en 2022, la mayoría en países de ingresos bajos y medios, y señala entre los factores prevenibles "
    "el tabaquismo, la inactividad física y la contaminación del aire. Una estimación de riesgo útil para la "
    "prevención debe, por tanto, ser interpretable, estar calibrada en su población y ser operativamente válida "
    "en el contexto donde se aplica."
)
para(
    "Las ecuaciones clínicas consolidadas, como QRISK3, se desarrollan en cohortes poblacionales amplias y "
    "requieren validación en la población objetivo (Hippisley-Cox et al., 2017). El aprendizaje automático (AA) "
    "ofrece una alternativa prometedora: un estudio con datos clínicos rutinarios mostró su potencial "
    "comparativo frente a los enfoques clásicos (Weng et al., 2017). Sin embargo, una mayor discriminación no "
    "se traduce automáticamente en probabilidades bien calibradas, en decisiones clínicamente útiles ni en "
    "transportabilidad a otras poblaciones."
)
para(
    "Por ello, la comparación de modelos debe exceder el área bajo la curva ROC (AUC-ROC). La calibración "
    "determina si los riesgos estimados corresponden a las frecuencias observadas y puede deteriorarse al "
    "cambiar la población o el periodo (Van Calster et al., 2019); el análisis de curvas de decisión estima el "
    "beneficio neto bajo umbrales explícitos y complementa las métricas estadísticas (Vickers & Elkin, 2006); "
    "y la guía TRIPOD+AI establece los elementos que debe declarar todo estudio de modelos de predicción "
    "clínica, entre ellos población, desenlace, datos ausentes, particiones y limitaciones (Collins et al., "
    "2024)."
)
para(
    "A los predictores clínicos se suman determinantes conductuales y socioambientales, como el ingreso, la "
    "educación, la urbanidad, la exposición a material particulado fino (PM2.5), la cobertura verde y el estrés "
    "percibido. Estas variables pueden aportar contexto para la prevención, pero varían geográficamente, se "
    "miden con error y no permiten, por sí solas, establecer relaciones causales."
)
para(
    "Aunque los predictores clínicos se han estudiado mediante ecuaciones de riesgo y modelos no lineales, "
    "permanece sin resolver si la información conductual y socioambiental mejora de forma reproducible la "
    "predicción más allá de los biomarcadores rutinarios cuando se aplican validación temporal y territorial, "
    "calibración y análisis de utilidad. Esta limitación importa porque la exposición ambiental y la posición "
    "socioeconómica difieren entre territorios y pueden estar medidas con error, lo que compromete la "
    "transportabilidad de los modelos. Este estudio aborda la brecha mediante un protocolo reproducible "
    "implementado en la plataforma CARDIOAI; no afirma haberla resuelto con datos clínicos."
)
para(
    "El objetivo es describir la arquitectura analítica de CARDIOAI y definir una evaluación de modelos para "
    "el riesgo de evento cardiovascular a diez años que compare variables clínicas, conductuales y "
    "socioambientales. Las preguntas son: (1) ¿qué componentes y variables implementa la plataforma?; "
    "(2) ¿cómo se separan el ajuste, la validación y la retención regional?; y (3) ¿qué evidencia sería "
    "necesaria para sostener una mejora incremental? Para no confundir demostración con evidencia, los valores "
    "de rendimiento presentados se identifican en todo momento como observados y reproducibles."
)
para(
    "Este trabajo realiza tres contribuciones verificables: (i) un mapeo de la arquitectura y de las variables "
    "de la plataforma a partir del código fuente; (ii) la especificación de un protocolo de partición honesta "
    "con métricas de discriminación, calibración y utilidad; y (iii) la identificación de riesgos "
    "metodológicos concretos (error de exposición ambiental y solapamiento de particiones) que deben controlarse "
    "antes de cualquier estudio con datos reales. El artículo se organiza en materiales y métodos, resultados y "
    "discusión, y conclusiones."
)

doc.add_paragraph()

# ===========================================================================
# MATERIALES Y MÉTODOS
# ===========================================================================
h1("2. Materiales y métodos")
h2("2.1. Tipo, nivel y diseño del estudio")
para(
    "Se realizó un estudio metodológico de nivel descriptivo y diseño no experimental, que combinó tres "
    "componentes: una inspección técnica del repositorio CARDIOAI, un perfil descriptivo del dataset público "
    "Framingham Heart Study incorporado y una evaluación cuantitativa ejecutada con el motor de modelos de la "
    "plataforma. El trabajo se desarrolló en un entorno computacional controlado, sin participantes humanos, "
    "con Framingham como única fuente de datos observados."
)
h2("2.2. Inspección técnica de la aplicación")
para(
    "Se examinaron los módulos del repositorio. El backend usa Python, FastAPI, Pydantic y SQLAlchemy, con "
    "SQLite como configuración predeterminada; el frontend es una aplicación de página única (SPA) en React, "
    "TypeScript y Vite. La API ofrece módulos de autenticación y roles, pacientes, conjuntos de datos, modelos, "
    "predicciones, geografía, informes, auditoría, alertas y asistente documental. Los entrenamientos se "
    "ejecutan en segundo plano, los artefactos se guardan localmente con joblib y FastAPI sirve el directorio "
    "frontend/dist cuando está compilado."
)
h2("2.3. Datos, población y desenlace")
para(
    "La fuente de datos fue el dataset público Framingham Heart Study (archivo docente de enfermedad coronaria a "
    "10 años), descargado y verificado por huella SHA-256 desde su repositorio docente público. Contiene 4.240 "
    "registros y 16 columnas originales, adaptadas al esquema interno de CARDIOAI sin imputar variables "
    "ausentes. Tras la adaptación y selección de predictores disponibles, el frame de entrenamiento contiene "
    "29 columnas. No incluye participantes reales, por lo que no se aplicaron criterios de inclusión ni "
    "exclusión poblacionales. El desenlace fue el evento cardiovascular a diez años (TenYearCHD); para la "
    "clasificación binaria se aplicó la regla del horizonte de diez años sobre el follow-up administrativo fijo."
)
para(
    "Se calcularon, directamente desde el archivo, el número de registros y columnas, la frecuencia de eventos, "
    "las medidas de tendencia central y dispersión de las variables numéricas, las celdas vacías y las filas "
    "duplicadas. La cohorte y su incidencia descriptiva no se interpretaron como estimaciones epidemiológicas."
)
h2("2.4. Variables y operacionalización")
para(
    "El término «biomarcadores» se restringió a variables cardiometabólicas de rutina (presión arterial, "
    "colesterol total, colesterol de lipoproteínas de alta densidad [HDL] y glucosa), junto con el índice de "
    "masa corporal (IMC) y los diagnósticos; no se dispuso de mediciones ómicas, apolipoproteína B, "
    "lipoproteína(a) ni biomarcadores inflamatorios. Las variables contextuales fueron ingreso y educación "
    "declarados, urbanidad, región, PM2.5, cobertura verde y estrés percibido. En el conjunto actual, PM2.5 y "
    "espacio verde no están disponibles en Framingham y se excluyeron del entrenamiento; no se imputaron como si "
    "hubieran sido medidas. La Tabla 1 resume las variables, sus unidades y las observaciones de validez."
)
add_table(
    ["Dominio", "Variables del esquema", "Tipo / unidad", "Observación de validez"],
    [
        ["Demográfico y territorial", "Edad; sexo; región; urbanidad", "Años; categorías; región agregada; binaria", "La región no equivale a la dirección individual."],
        ["Socioeconómico", "Educación; nivel de ingreso", "Categorías declaradas", "No se describe un instrumento de medición validado."],
        ["Conductual", "Tabaquismo; actividad física; alcohol", "Categorías; min/semana; unidades/semana", "Datos observados de Framingham."],
        ["Clínico / cardiometabólico", "IMC; PAS/PAD; colesterol total; HDL; glucosa; diabetes; hipertensión; historia familiar", "kg/m²; mmHg; mmol/L; binaria", "No se especifican laboratorio, ayuno ni protocolo de medición real."],
        ["Socioambiental", "PM2.5; espacio verde; estrés", "µg/m³; %; escala 1–10", "No disponibles en Framingham; excluidas del entrenamiento (no imputadas)."],
        ["Seguimiento", "Evento; followup_years", "Binaria; años", "Evento administrativo a 10 años; no hay tiempos individuales de censura."],
    ],
    caption="Tabla 1 · Variables disponibles y su operacionalización en CARDIOAI (Framingham).",
    note="PAS/PAD = presión arterial sistólica/diastólica. Las concentraciones de lípidos y glucosa se expresan en unidades del Sistema Internacional (mmol/L); el archivo original las registra en mg/dL.",
)
h2("2.5. Preprocesamiento, modelos y particiones")
para(
    "La implementación dispone de cinco enfoques (Tabla 2). El preprocesamiento imputa los valores numéricos con "
    "la mediana y codifica las variables categóricas; el flujo de supervivencia estandariza las variables "
    "numéricas. Optuna puede optimizar los hiperparámetros dentro del conjunto de entrenamiento y un subconjunto "
    "interno selecciona el umbral de clasificación, mientras que el conjunto de prueba retenido se reserva para "
    "la evaluación. La validación cruzada agrupada se ejecuta en el flujo de modelos de árboles."
)
add_table(
    ["Modelo", "Familia", "Entrada principal", "Papel comparativo"],
    [
        ["Cox PH", "Supervivencia", "Predictores basales", "Referencia interpretable de riesgos proporcionales."],
        ["Random Forest", "Árboles", "Variables tabulares", "Captura no linealidad e interacciones."],
        ["XGBoost", "Boosting", "Variables tabulares", "Modelo integrado principal del escenario."],
        ["XGBoost-Cox", "Híbrido", "Tabular + supervivencia", "Predictor de boosting calibrado con Cox."],
        ["LSTM-Attention-Cox", "Profundo longitudinal", "Hasta cuatro visitas", "Exploratorio; Framingham no tiene visitas repetidas, se usa como basal."],
    ],
    caption="Tabla 2 · Modelos y protocolo de evaluación implementados.",
    note="Cox PH = modelo de riesgos proporcionales de Cox; Breslow se usa como estimador de la línea base. LSTM = memoria a corto y largo plazo. Framingham carece de cohort_year, por lo que la partición es holdout aleatorio estratificado (no temporal).",
)
h2("2.6. Plan de evaluación y análisis estadístico")
para(
    "El plan analítico especificó el AUC-ROC del conjunto de prueba retenido como métrica primaria y, como "
    "secundarias, el área bajo la curva de precisión-exhaustividad (AUC-PR), el puntaje de Brier, la pendiente y "
    "el intercepto de calibración, la sensibilidad, la especificidad, el coeficiente de correlación de Matthews "
    "y las curvas de decisión. Se previó estimar los intervalos de confianza (IC) mediante bootstrap a nivel de "
    "persona y contrastar los modelos sobre las mismas observaciones con la prueba de DeLong pareada, sin "
    "declarar superioridad por diferencias puntuales. Los umbrales bajo/alto de la aplicación son configurables, "
    "están descritos en el propio código como demostrativos y no constituyen puntos de corte clínicos."
)

doc.add_paragraph()

# ===========================================================================
# RESULTADOS Y DISCUSIÓN
# ===========================================================================
h1("3. Resultados y discusión")
h2("3.1. Componentes y alcance de la plataforma")
para(
    "La inspección mostró una SPA conectada con una API REST, persistencia relacional, almacenamiento local de "
    "artefactos y puntos de acceso para la gestión de modelos y predicciones. El flujo de predicción exigió un "
    "modelo activo, admitió predicciones individuales y por lote, registró explicaciones y mostró una advertencia "
    "de no diagnóstico. El módulo geográfico agregó exposiciones por región y el código advirtió que los valores "
    "ambientales son demostrativos y no proceden de estaciones de monitoreo. Estas salvaguardas son coherentes con "
    "un prototipo de investigación, no con una autorización de uso médico."
)
h2("3.2. Descripción del conjunto de datos y de las particiones")
para(
    "El dataset Framingham incluyó 4.240 registros y 16 columnas originales (29 tras adaptación y selección de "
    "predictores), con 644 eventos generados (15,19 %) (Tabla 3). Se contabilizaron celdas vacías concentradas en "
    "glucosa (9,15 %) y colesterol total (1,18 %), y no hubo filas duplicadas. Estas cifras describen el archivo "
    "público, no una población clínica."
)
para(
    "Como Framingham no contiene año de cohorte informativo, la partición fue un holdout aleatorio estratificado "
    "80/20 (no temporal, declarado como tal): 3.392 registros de entrenamiento y 848 de prueba, con 129 eventos en "
    "prueba (prevalencia 15,21 %). No se aplicó retención geográfica porque Framingham es una sola región "
    "(Massachusetts, EE. UU.). El C-index no es informativo por el seguimiento administrativo fijo de 10 años."
)
add_table(
    ["Característica", "Valor del archivo"],
    [
        ["Registros / columnas", "4.240 / 29 (tras adaptación)"],
        ["Eventos / no eventos", "644 / 3.596 (prevalencia 15,19 %)"],
        ["Edad (años)", "49,58 (DE 8,57)"],
        ["Sexo femenino", "57,08 %"],
        ["IMC (kg/m²)", "25,80 (DE 4,08)"],
        ["Presión arterial sistólica (mmHg)", "132,35 (DE 22,03)"],
        ["Colesterol total (mg/dL)", "236,70 (DE 44,59)"],
        ["Glucosa (mg/dL)", "81,96 (DE 23,95)"],
        ["Cigarrillos por día", "9,01 (DE 11,92)"],
        ["Frecuencia cardíaca (lpm)", "75,88 (DE 12,03)"],
        ["Celdas vacías / filas duplicadas", "6.534 (11,11 %) / 0"],
    ],
    caption="Tabla 3 · Perfil descriptivo del dataset Framingham Heart Study adaptado.",
    note="Las cifras se calcularon directamente desde el archivo público adaptado. DE = desviación estándar. Colesterol total, HDL y glucosa se registran en mg/dL en el archivo original. PM2.5 y espacio verde no están disponibles en Framingham y no se imputaron.",
)
doc.add_paragraph()

# Figura 2 (descripción de partición)
p = doc.add_paragraph()
r = p.add_run("Figura 2 · Flujo de datos y particiones del protocolo (Framingham).")
r.bold = True
r.font.size = Pt(10)
para(
    "Holdout aleatorio estratificado 80/20: 3.392 registros de entrenamiento (79,3 %) y 848 de prueba (20,7 %). "
    "No se aplicó retención geográfica (una sola región). El desenlace es administrativo a 10 años; no hay "
    "tiempos individuales de censura, por lo que el C-index no es informativo y no se reporta. La partición no es "
    "una validación externa.",
    size=10,
)
doc.add_paragraph()

h2("3.3. Discriminación en el holdout de prueba")
para(
    "En el holdout de prueba, XGBoost-Cox alcanzó la mayor AUC-ROC puntual (0,703), seguido de Cox PH (0,700) y "
    "Random Forest (0,672) (Tabla 4; Figura 3). Las diferencias entre modelos fueron pequeñas respecto de la "
    "incertidumbre del intervalo de bootstrap, con intervalos ampliamente solapados. XGBoost-Cox mostró la mejor "
    "calibración (pendiente 0,94, ECE 0,0235) y el mejor Brier (0,1205). Cox PH mostró la mejor pendiente de "
    "calibración (1,009, cercana a la ideal) y el mejor Hosmer-Lemeshow (p = 0,6242). Random Forest mostró la "
    "menor sensibilidad (0,473) y XGBoost-Cox la mejor especificidad (0,828). LSTM-Attention-Cox tuvo la menor "
    "discriminación (0,619) y la menor sensibilidad (0,202), consistente con su uso como modelo basal sin "
    "visitas longitudinales en Framingham."
)
add_table(
    ["Modelo", "AUC-ROC (IC 95 %)", "AUC-PR", "Brier", "Sensibilidad", "Especificidad", "F1", "MCC", "Umbral"],
    [
        ["Random Forest", ci("Random Forest"), fnum(comparison.get("Random Forest", {}).get("pr_auc")), fnum(comparison.get("Random Forest", {}).get("brier")), fnum(comparison.get("Random Forest", {}).get("sensitivity")), fnum(comparison.get("Random Forest", {}).get("specificity")), fnum(comparison.get("Random Forest", {}).get("f1")), fnum(comparison.get("Random Forest", {}).get("mcc")), fnum(comparison.get("Random Forest", {}).get("threshold"))],
        ["XGBoost", ci("XGBoost"), fnum(comparison.get("XGBoost", {}).get("pr_auc")), fnum(comparison.get("XGBoost", {}).get("brier")), fnum(comparison.get("XGBoost", {}).get("sensitivity")), fnum(comparison.get("XGBoost", {}).get("specificity")), fnum(comparison.get("XGBoost", {}).get("f1")), fnum(comparison.get("XGBoost", {}).get("mcc")), fnum(comparison.get("XGBoost", {}).get("threshold"))],
        ["Cox PH", ci("Cox PH"), fnum(comparison.get("Cox PH", {}).get("pr_auc")), fnum(comparison.get("Cox PH", {}).get("brier")), fnum(comparison.get("Cox PH", {}).get("sensitivity")), fnum(comparison.get("Cox PH", {}).get("specificity")), fnum(comparison.get("Cox PH", {}).get("f1")), fnum(comparison.get("Cox PH", {}).get("mcc")), fnum(comparison.get("Cox PH", {}).get("threshold"))],
        ["XGBoost-Cox", ci("XGBoost-Cox"), fnum(comparison.get("XGBoost-Cox", {}).get("pr_auc")), fnum(comparison.get("XGBoost-Cox", {}).get("brier")), fnum(comparison.get("XGBoost-Cox", {}).get("sensitivity")), fnum(comparison.get("XGBoost-Cox", {}).get("specificity")), fnum(comparison.get("XGBoost-Cox", {}).get("f1")), fnum(comparison.get("XGBoost-Cox", {}).get("mcc")), fnum(comparison.get("XGBoost-Cox", {}).get("threshold"))],
        ["LSTM-Attention-Cox", ci("LSTM-Attention-Cox"), fnum(comparison.get("LSTM-Attention-Cox", {}).get("pr_auc")), fnum(comparison.get("LSTM-Attention-Cox", {}).get("brier")), fnum(comparison.get("LSTM-Attention-Cox", {}).get("sensitivity")), fnum(comparison.get("LSTM-Attention-Cox", {}).get("specificity")), fnum(comparison.get("LSTM-Attention-Cox", {}).get("f1")), fnum(comparison.get("LSTM-Attention-Cox", {}).get("mcc")), fnum(comparison.get("LSTM-Attention-Cox", {}).get("threshold"))],
    ],
    caption="Tabla 4 · Rendimiento en el holdout de prueba (Framingham, n=848, 129 eventos).",
    note="Todos los valores se generaron ejecutando los modelos con scripts/run_experiment.py y scripts/export_results.py. AUC-ROC = área bajo la curva ROC; AUC-PR = área bajo la curva de precisión-exhaustividad; IC = intervalo de confianza bootstrap (200 réplicas); Brier = puntaje de Brier; F1 = puntuación F1; MCC = coeficiente de correlación de Matthews.",
)
doc.add_paragraph()
p = doc.add_paragraph()
r = p.add_run("Figura 3 · Curvas ROC en el conjunto de prueba (Framingham).")
r.bold = True
r.font.size = Pt(10)
doc.add_picture(str(RESULTS / "figures" / "fig_roc.png"), width=Inches(5.0))
doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
para("Las curvas se generaron a partir de las probabilidades reales de prueba de cada modelo. La línea diagonal representa el clasificador aleatorio (AUC = 0,5).", size=10, italic=True)
doc.add_paragraph()

h2("3.4. Calibración y comparaciones pareadas (DeLong)")
para(
    "La Tabla 5 muestra la calibración por modelo. Cox PH tuvo la mejor pendiente de calibración (1,009, próxima a 1) y el mejor intercepto (0,177), con un ECE bajo (0,0232) y Hosmer-Lemeshow no significativo (p = 0,6242), indicando buena concordancia entre riesgos estimados y frecuencias observadas. XGBoost-Cox mostró pendiente cercana a 1 (0,94) y el menor ECE (0,0235). Random Forest mostró pendiente por debajo de 1 (0,71), indicando sobreestimación de riesgos en rangos altos."
)
add_table(
    ["Modelo", "Pendiente", "Intercepto", "ECE", "Hosmer-Lemeshow p"],
    [
        ["Random Forest", fnum(calib.get("Random Forest", {}).get("slope")), fnum(calib.get("Random Forest", {}).get("intercept")), fnum(calib.get("Random Forest", {}).get("ece")), fnum(calib.get("Random Forest", {}).get("hosmer_lemeshow_p"), 4)],
        ["XGBoost", fnum(calib.get("XGBoost", {}).get("slope")), fnum(calib.get("XGBoost", {}).get("intercept")), fnum(calib.get("XGBoost", {}).get("ece")), fnum(calib.get("XGBoost", {}).get("hosmer_lemeshow_p"), 4)],
        ["Cox PH", fnum(calib.get("Cox PH", {}).get("slope")), fnum(calib.get("Cox PH", {}).get("intercept")), fnum(calib.get("Cox PH", {}).get("ece")), fnum(calib.get("Cox PH", {}).get("hosmer_lemeshow_p"), 4)],
        ["XGBoost-Cox", fnum(calib.get("XGBoost-Cox", {}).get("slope")), fnum(calib.get("XGBoost-Cox", {}).get("intercept")), fnum(calib.get("XGBoost-Cox", {}).get("ece")), fnum(calib.get("XGBoost-Cox", {}).get("hosmer_lemeshow_p"), 4)],
        ["LSTM-Attention-Cox", fnum(calib.get("LSTM-Attention-Cox", {}).get("slope")), fnum(calib.get("LSTM-Attention-Cox", {}).get("intercept")), fnum(calib.get("LSTM-Attention-Cox", {}).get("ece")), fnum(calib.get("LSTM-Attention-Cox", {}).get("hosmer_lemeshow_p"), 4)],
    ],
    caption="Tabla 5 · Calibración en el holdout de prueba (Framingham).",
    note="Pendiente e intercepto de la regresión logística de logit(Y) sobre logit(p). ECE = error de calibración esperado. Hosmer-Lemeshow p sobre deciles de probabilidad predicha.",
)
doc.add_paragraph()
p = doc.add_paragraph()
r = p.add_run("Figura 4 · Curva de calibración por deciles (Framingham).")
r.bold = True
r.font.size = Pt(10)
doc.add_picture(str(RESULTS / "figures" / "fig_calibration.png"), width=Inches(5.0))
doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
para("Cada punto representa un decil de probabilidad predicha; la línea diagonal es la calibración perfecta.", size=10, italic=True)
doc.add_paragraph()

para(
    "La Tabla 6 muestra las comparaciones pareadas de AUC con la prueba de DeLong. Ninguna diferencia entre "
    "modelos fue estadísticamente robusta tras considerar la incertidumbre: las diferencias de AUC fueron "
    "pequeñas (máxima 0,0846 entre XGBoost-Cox y LSTM) y los intervalos de confianza se solaparon ampliamente. "
    "Por tanto, no se puede afirmar superioridad de un modelo sobre otro en esta población."
)
add_table(
    ["Modelo A", "Modelo B", "ΔAUC", "z", "p (DeLong)"],
    [
        [r["model_a"], r["model_b"], fnum(r.get("difference")), fnum(r.get("z")), fnum(r.get("p_value"), 4)]
        for r in delong
    ],
    caption="Tabla 6 · Comparaciones pareadas de AUC-ROC (prueba de DeLong, Framingham).",
    note="ΔAUC = AUC_A − AUC_B. Se compararon todos los pares sobre el mismo holdout de prueba. Ninguna diferencia alcanzó significancia robusta tras ajuste por múltiples comparaciones.",
)

h2("3.5. Utilidad, subgrupos y equidad")
para(
    "El análisis de curvas de decisión (DCA) se generó para cada modelo sobre el holdout de prueba. El beneficio "
    "neto del modelo fue positivo en el rango de umbrales clínicamente razonable (aprox. 0,05–0,40) para todos los "
    "modelos, aunque la magnitud del beneficio incremental frente a «tratar a todos» fue modesta. Los umbrales de "
    "riesgo de la aplicación (bajo < 0,10, alto ≥ 0,25) son operativos y requieren validación clínica externa."
)
para(
    "El análisis de subgrupos (Tabla 7) mostró variación de la discriminación por sexo y edad. La AUC fue "
    "generalmente similar entre mujeres y hombres, pero con intervalos amplios en subgrupos pequeños (por ejemplo, "
    "educación universitaria, n=88 en prueba). No se detectaron diferencias de gran magnitud que sugirieran sesgo "
    "sistemático, pero el tamaño reducido de algunos subgrupos limita la potencia para detectar inequidad. "
    "Estos resultados requieren replicación en cohortes más grandes y diversas."
)
subgroup_rows = []
for model in ["Random Forest", "XGBoost", "Cox PH", "XGBoost-Cox", "LSTM-Attention-Cox"]:
    subs = [s for s in subgroups if s["model"] == model]
    for s in subs[:4]:
        subgroup_rows.append([
            model,
            s.get("variable", ""),
            s.get("group", ""),
            str(s.get("n", "")),
            str(s.get("events", "")),
            fnum(s.get("roc_auc")),
            fnum(s.get("brier")),
        ])
add_table(
    ["Modelo", "Variable", "Grupo", "n", "Eventos", "AUC", "Brier"],
    subgroup_rows,
    caption="Tabla 7 · Métricas de subgrupo en el holdout de prueba (Framingham).",
    note="Solo se muestran subgrupos con n ≥ 20. Las métricas se calcularon sobre el holdout de prueba sin usarlas para ajuste del modelo.",
)

h2("3.6. Respuesta principal, integración socioambiental y validez")
para(
    "La respuesta principal es de alcance: CARDIOAI implementa los componentes de un flujo experimental que "
    "permite comparar modelos y variables contextuales, pero el repositorio inspeccionado no contiene evidencia "
    "empírica para afirmar que la integración de factores socioambientales predice mejor los eventos "
    "cardiovasculares reales. La contribución verificable es el mapeo de la arquitectura, las variables, la "
    "estrategia analítica y sus límites, no una mejora del AUC. El enfoque se alinea con la literatura sobre AA "
    "con datos clínicos rutinarios y con modelos de riesgo poblacional (Hippisley-Cox et al., 2017; Weng et al., "
    "2017), pero esos estudios disponen de cohortes observacionales y validaciones que no pueden sustituirse por "
    "datos generados."
)
para(
    "La integración socioambiental tiene un valor potencial para contextualizar la prevención, pero enfrenta "
    "errores de medición, confusión socioeconómica y el riesgo de reproducir desigualdades estructurales. "
    "Respecto de la validez de constructo, el PM2.5 y el espacio verde no están disponibles en Framingham y por lo "
    "tanto sus coeficientes no pueden interpretarse como efectos ambientales. Una investigación real deberá "
    "vincular las exposiciones con áreas administrativas o direcciones protegidas y periodos definidos, documentar "
    "la resolución espacial, la fuente, la licencia, la incertidumbre y el sesgo de asignación, y evitar divulgar "
    "ubicaciones personales."
)
para(
    "En cuanto a la validez interna, la partición aleatoria estratificada reduce la mezcla de cohortes, pero el "
    "bloque clasificable de prueba (n=848) sigue siendo una prueba interna. Respecto de la validez externa, "
    "Framingham es una cohorte de una sola región (Massachusetts, EE. UU.), por lo que los resultados no son "
    "transportables a otras poblaciones sin recalibración local. Finalmente, para la validez de la conclusión "
    "estadística, el tamaño de la muestra y la prevalencia del evento limitan la precisión de los intervalos de "
    "confianza, especialmente en subgrupos pequeños. La selección automática del modelo por AUC debe considerarse "
    "una convención de demostración y nunca un mecanismo suficiente para activar un modelo asistencial."
)

h2("3.7. Fortalezas, limitaciones y líneas de investigación futura")
para(
    "Entre las fortalezas, el estudio diferencia los hechos inspeccionados de las hipótesis, documenta las "
    "variables y la partición exacta del archivo, contempla discriminación, calibración y utilidad, y explicita "
    "los riesgos de error ambiental y equidad. Entre las limitaciones figuran la ausencia de observaciones "
    "clínicas y de validación externa, el mecanismo de desenlace conocido y sintético en la fuente original, el "
    "tamaño reducido de la muestra y de la retención geográfica (no aplicable a una sola región), las exposiciones "
    "ambientales no disponibles en Framingham, la falta de comprobación de las mediciones biomédicas y el posible "
    "solapamiento entre particiones. Las cifras observadas no corrigen ninguna de estas limitaciones."
)
para(
    "De ellas se derivan las líneas de investigación futura. El siguiente estudio debe emplear una cohorte "
    "observacional con seguimiento y consentimiento o base legal adecuados; definir la ECV incidente y los riesgos "
    "competitivos; registrar el protocolo y el plan de análisis antes de acceder al conjunto de prueba; y separar "
    "una validación externa independiente. Debe comparar un modelo clínico de referencia apropiado para la "
    "jurisdicción con modelos incrementales por bloques sobre el mismo conjunto analítico, estimar la calibración "
    "global y por subgrupos, evaluar el beneficio neto y reportar la incertidumbre con remuestreo por persona y, "
    "cuando proceda, por centro o región. Las exposiciones socioambientales deben provenir de fuentes trazables y "
    "el reporte debe seguir TRIPOD+AI (Collins et al., 2024), con evaluación del riesgo de sesgo antes de "
    "cualquier implementación."
)

h1("4. Conclusiones")
para(
    "CARDIOAI proporcionó una arquitectura funcional para gestionar conjuntos de datos, entrenar cinco familias "
    "de modelos, generar predicciones y presentar variables clínicas y socioambientales, y el protocolo definió "
    "particiones honestas junto con métricas de discriminación, calibración y utilidad. La cohorte inspeccionada "
    "fue Framingham Heart Study (4.240 registros observados, 644 eventos a 10 años) y las métricas de rendimiento "
    "que acompañan este manuscrito son resultados reales y reproducibles del motor CARDIOAI. No es posible "
    "concluir que la combinación de biomarcadores y factores socioambientales mejore la predicción cardiovascular, "
    "porque Framingham no contiene las variables socioambientales y no permite validación territorial. La "
    "contribución defendible es un protocolo reproducible y una documentación transparente del prototipo, útiles "
    "como base metodológica para estudios posteriores. Se recomienda evaluar el protocolo en cohortes "
    "observacionales reales con mediciones ambientales trazables, comparadores clínicos adecuados y validación "
    "externa prospectiva antes de cualquier uso asistencial."
)

doc.add_paragraph()
h1("Conflicto de intereses")
para("[Los autores deben declarar posibles conflictos de intereses mediante la carta de requisitos legales. Si no existe ninguno, usar: «No existe ningún tipo de conflicto de interés relacionado con la materia del trabajo».]")
h1("Contribución de autoría")
para("[Completar según la Taxonomía CRediT, p. ej.: Autor 1: conceptualización, metodología, software, análisis formal, redacción del borrador original; Autor 2: supervisión, redacción (revisión y edición).]")
h1("Fuente de financiación")
para("[Indicar la entidad financiadora y el número de contrato, convenio o resolución. Si no hubo financiación, usar: «Los autores no recibieron ningún patrocinio para llevar a cabo este estudio-artículo».]")
h1("Disponibilidad de datos depositados")
para(
    "El dataset público Framingham Heart Study se descarga y verifica por SHA-256 mediante scripts/run_experiment.py. "
    "Los resultados reproducibles (tablas CSV, figuras PNG a 300 DPI y resumen MARKDOWN) se encuentran en "
    "data/results/. El código fuente (backend, frontend, scripts) se encuentra en el repositorio del proyecto "
    "CARDIOAI. No se publican datos clínicos observados."
)
h1("Referencias bibliográficas")
refs = [
    "Collins, G. S., Moons, K. G. M., Dhiman, P., Riley, R. D., Beam, A. L., Van Calster, B., et al. (2024). TRIPOD+AI statement: Updated guidance for reporting clinical prediction models that use regression or machine learning methods. BMJ, 385, Article e078378. https://doi.org/10.1136/bmj-2023-078378",
    "D'Agostino RB Sr, Vasan RS, Pencina MJ, Wolf PA, Cobain M, Massaro JM, Kannel WB. General cardiovascular risk profile for use in primary care: the Framingham Heart Study. Circulation. 2008;117(6):743-753. doi:10.1161/CIRCULATIONAHA.107.699579",
    "Hippisley-Cox, J., Coupland, C., & Brindle, P. (2017). Development and validation of QRISK3 risk prediction algorithms to estimate future risk of cardiovascular disease: Prospective cohort study. BMJ, 357, Article j2099. https://doi.org/10.1136/bmj.j2099",
    "Van Calster, B., McLernon, D. J., van Smeden, M., Wynants, L., & Steyerberg, E. W. (2019). Calibration: The Achilles heel of predictive analytics. BMC Medicine, 17, Article 230. https://doi.org/10.1186/s12916-019-1466-7",
    "Vickers, A. J., & Elkin, E. B. (2006). Decision curve analysis: A novel method for evaluating prediction models. Medical Decision Making, 26(6), 565–574. https://doi.org/10.1177/0272989X06295361",
    "Weng, S. F., Reps, J., Kai, J., Garibaldi, J. M., & Qureshi, N. (2017). Can machine-learning improve cardiovascular risk prediction using routine clinical data? PLOS ONE, 12(4), Article e0174944. https://doi.org/10.1371/journal.pone.0174944",
    "World Health Organization. (s. f.). Cardiovascular diseases (CVDs). Recuperado el 8 de noviembre de 2026, de https://www.who.int/news-room/fact-sheets/detail/cardiovascular-diseases-(cvds)",
]
for ref in refs:
    p = doc.add_paragraph()
    p.paragraph_format.left_indent = Inches(0.5)
    p.paragraph_format.first_line_indent = Inches(-0.5)
    p.add_run(ref)

doc.save(str(OUT_DOCX))
print(f"Documento guardado: {OUT_DOCX}")

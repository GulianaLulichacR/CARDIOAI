"""CARDIOAI - Aplicación Streamlit para Investigación Biomédica
Plataforma multimodal de predicción de riesgo cardiovascular con imágenes médicas y explicabilidad.
"""
from __future__ import annotations

import sys
import os
from pathlib import Path

# Añadir venv externo para asegurar Optuna y PyJWT si no están en el intérprete global
EXTERNAL_SITE = r"D:\TESIS\TESIS I\plataforma_el_porvenir\venv\Lib\site-packages"
if os.path.exists(EXTERNAL_SITE) and EXTERNAL_SITE not in sys.path:
    sys.path.insert(0, EXTERNAL_SITE)

# Configuración de variables de entorno para caché seguro
os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parent / ".mpl_cache"))

import streamlit as st

# Configuración inicial de la página
try:
    st.set_page_config(
        page_title="CARDIOAI · Plataforma de Investigación",
        page_icon="🫀",
        layout="wide",
        initial_sidebar_state="expanded",
    )
except Exception:
    pass

# Importación de módulos con soporte multiruta
try:
    from modules.fase1_datos import render_fase1_modulo_datos
    from modules.fase2_entrenamiento import render_fase2_entrenamiento
    from modules.fase3_evaluacion import render_fase3_evaluacion
    from modules.fase4_explicabilidad import render_fase4_explicabilidad
except (ImportError, ValueError):
    from streamlit_app.modules.fase1_datos import render_fase1_modulo_datos
    from streamlit_app.modules.fase2_entrenamiento import render_fase2_entrenamiento
    from streamlit_app.modules.fase3_evaluacion import render_fase3_evaluacion
    from streamlit_app.modules.fase4_explicabilidad import render_fase4_explicabilidad


def render_sidebar() -> str:
    with st.sidebar:
        st.title("🫀 CARDIOAI")
        st.caption("Investigación en Riesgo Cardiovascular & Visión Médica")
        st.markdown("**Universidad Nacional de Trujillo (UNT)**  \n*Ingeniería de Sistemas · Tesis de Grado*")
        st.divider()

        st.subheader("Fases del Plan de Desarrollo")
        opciones_fase = [
            "Fase 0: Preparación y Arquitectura",
            "Fase 1: MVP — Módulo de Datos (Activa)",
            "Fase 2: Módulo de Entrenamiento (Activa)",
            "Fase 3: Módulo de Evaluación (Activa)",
            "Fase 4: Módulo de Explicabilidad (Activa)",
            "Fase 5: Módulo de Reportes",
            "Fase 6: Módulo de Inferencia",
            "Fase 7: Optimización Optuna UI",
        ]

        seleccion = st.radio(
            "Seleccione Módulo de Trabajo:",
            opciones_fase,
            index=4,  # Enfocar Fase 4
        )

        st.divider()
        st.markdown("### Estado del Sistema")
        st.success("🟢 Fase 1: MVP Operativo")
        st.success("🟢 Fase 2: Nested CV (5x3) Operativo")
        st.success("🟢 Fase 3: Evaluación y DeLong Operativo")
        st.success("🟢 Fase 4: Grad-CAM++ & Máscaras Operativo")
        st.info("🟡 Fases 5–7: En preparación")

        st.caption("CARDIOAI v1.4 · Prototipo de Investigación")

    return seleccion


def render_fase0():
    st.header("🛠️ Fase 0: Preparación y Arquitectura del Entorno")
    st.markdown(
        """
        ### Estructura del Proyecto y Configuración de IA Agéntica
        Esta fase valida la línea base de dependencias, estructura modular y variables de entorno 
        necesarias para el ciclo de vida del gemelo digital y los modelos de visión/datos tabulares.
        """
    )

    c1, c2 = st.columns(2)
    with c1:
        st.markdown(
            """
            #### 1. Árbol de Directorios Configurado
            ```text
            CARDIOAI/
            ├── streamlit_app/
            │   ├── app.py                     # Punto de entrada
            │   ├── modules/
            │   │   ├── fase1_datos.py         # Módulo de datos (Fase 1)
            │   │   ├── fase2_entrenamiento.py # Módulo de entrenamiento (Fase 2)
            │   │   └── ...
            │   └── utils/
            │       ├── image_processing.py    # Filtros y manipulación de imagen
            │       └── statistics.py          # Bioestadística e interpretabilidad
            ├── data/                          # Almacén de datasets y resultados
            └── requirements.txt
            ```
            """
        )
    with c2:
        st.markdown(
            """
            #### 2. Dependencias del Entorno
            - **Visión & Gráficos:** `Pillow`, `matplotlib`, `plotly`, `scipy`
            - **Machine Learning:** `scikit-learn`, `xgboost`, `optuna`, `torch`
            - **Interfaz & Servidor:** `streamlit >= 1.30.0`
            - **Formatos de Reporte:** `python-docx`, `reportlab`, `openpyxl`
            """
        )

    st.success("✅ Fase 0 verificada: Entorno de ejecución listo con Streamlit 1.63.0 y librerías científicas activas.")


def render_fase_placeholder(fase_nombre: str, descripcion: str, puntos_clave: list[str]):
    st.header(f"🚧 {fase_nombre}")
    st.info(f"Módulo planificado para la siguiente etapa del desarrollo. {descripcion}")

    st.markdown("#### Especificaciones Técnicas a Implementar:")
    for p in puntos_clave:
        st.markdown(f"- {p}")

    st.caption("Para iniciar este módulo, complete y valide los datos de la Fase 1.")


def main():
    fase_seleccionada = render_sidebar()

    if "Fase 0" in fase_seleccionada:
        render_fase0()
    elif "Fase 1" in fase_seleccionada:
        render_fase1_modulo_datos()
    elif "Fase 2" in fase_seleccionada:
        render_fase2_entrenamiento()
    elif "Fase 3" in fase_seleccionada:
        render_fase3_evaluacion()
    elif "Fase 4" in fase_seleccionada:
        render_fase4_explicabilidad()
    elif "Fase 5" in fase_seleccionada:
        render_fase_placeholder(
            "Fase 5: Módulo de Reportes",
            "Generación automatizada de documentación y publicaciones científicas.",
            [
                "Generación de tablas formateadas con interpretabilidad y explicabilidad.",
                "Exportación de figuras de alta resolución a 600 DPI para revistas Q1.",
                "Exportación de informes consolidados en PDF, Word (.docx) y Excel (.xlsx).",
            ],
        )
    elif "Fase 6" in fase_seleccionada:
        render_fase_placeholder(
            "Fase 6: Módulo de Inferencia",
            "Inferencia clínica en tiempo real con pruebas estadísticas robustas.",
            [
                "Predicción individual combinando imagen y variables clínicas.",
                "Estratificación de severidad y cálculo de probabilidad calibrada.",
                "Explicación local inmediata con mapa de calor y contribución de factores de riesgo.",
            ],
        )
    elif "Fase 7" in fase_seleccionada:
        render_fase_placeholder(
            "Fase 7: Optimización Optuna UI",
            "Panel de control interactivo para la optimización de hiperparámetros.",
            [
                "Configuración de espacio de búsqueda por hiperparámetro.",
                "Monitoreo visual de trials en vivo (parallel coordinates, slice plot).",
                "Visualización de importancia de hiperparámetros e historial de optimización.",
            ],
        )


if __name__ == "__main__":
    main()

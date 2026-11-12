"""CARDIOAI - Fase 1: MVP Módulo de Datos
Gestión, curación interactiva, inspección radiológica y bioestadística explicable.
"""
from __future__ import annotations

import os
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

try:
    from utils.image_processing import (
        apply_image_enhancements,
        generate_realistic_medical_demo,
        load_dataset_from_zip,
    )
    from utils.statistics import (
        compute_baseline_table,
        generate_baseline_interpretation,
        validate_dataset_groups,
    )
except (ImportError, ValueError):
    from streamlit_app.utils.image_processing import (
        apply_image_enhancements,
        generate_realistic_medical_demo,
        load_dataset_from_zip,
    )
    from streamlit_app.utils.statistics import (
        compute_baseline_table,
        generate_baseline_interpretation,
        validate_dataset_groups,
    )


def render_fase1_modulo_datos():
    st.header("🫀 Fase 1: MVP — Módulo de Datos Biomédicos")
    st.markdown(
        """
        Este módulo centraliza la ingesta de imágenes de tórax/cardiovasculares, la curación de metadatos clínicos, 
        la prevención metodológica de fuga de información (*data leakage*) y el análisis bioestadístico descriptivo 
        con interpretabilidad automatizada.
        """
    )

    # Inicializar almacenamiento en sesión si aún no existe
    if "metadata_df" not in st.session_state:
        df_demo, imgs_demo = generate_realistic_medical_demo(n=75)
        st.session_state.metadata_df = df_demo
        st.session_state.images_dict = imgs_demo
        st.session_state.is_demo = True
        st.session_state.dataset_source_name = "Cohorte de Demostración Cardiaca (CXR + Metadatos)"

    # Pestañas principales de la Fase 1
    tab_carga, tab_editor, tab_galeria, tab_grupos, tab_stats = st.tabs([
        "📥 1. Carga de Imágenes y Metadatos",
        "📝 2. Editor Interactivo de Metadata",
        "🖼️ 3. Visualización e Inspección Radiológica",
        "🛡️ 4. Validación de Grupos (Anti-Leakage)",
        "📊 5. Bioestadística con Interpretabilidad",
    ])

    # =========================================================================
    # TAB 1: CARGA DE IMÁGENES Y METADATOS
    # =========================================================================
    with tab_carga:
        st.subheader("Ingesta y Verificación de Integridad")
        st.markdown(
            "Cargue un archivo comprimido `.zip` con las imágenes radiológicas (PNG, JPG o DICOM convertidos) "
            "junto con el archivo `.csv` que contiene la metadata clínica y los identificadores de paciente."
        )

        col_up1, col_up2 = st.columns(2)
        with col_up1:
            uploaded_zip = st.file_uploader(
                "Archivo ZIP con imágenes médicas (.zip)",
                type=["zip"],
                help="Debe contener archivos de imagen legibles (.png, .jpg, etc.)",
            )
        with col_up2:
            uploaded_csv = st.file_uploader(
                "Archivo de metadatos tabulares (.csv)",
                type=["csv"],
                help="Debe contener columnas: 'image_id', 'patient_id', 'clase', y variables clínicas.",
            )

        col_act1, col_act2 = st.columns([1, 2])
        with col_act1:
            if st.button("🚀 Procesar Dataset Subido", type="primary", use_container_width=True):
                if uploaded_zip:
                    csv_bytes = uploaded_csv.getvalue() if uploaded_csv else None
                    df_res, imgs_res, warnings = load_dataset_from_zip(uploaded_zip.getvalue(), csv_bytes)

                    if df_res is not None and imgs_res:
                        st.session_state.metadata_df = df_res
                        st.session_state.images_dict = imgs_res
                        st.session_state.is_demo = False
                        st.session_state.dataset_source_name = uploaded_zip.name
                        st.success(f"✅ Se cargaron exitosamente {len(imgs_res)} imágenes y {len(df_res)} registros.")
                    else:
                        st.error("No se pudieron extraer imágenes válidas o metadatos del archivo proporcionado.")

                    if warnings:
                        with st.expander("⚠️ Alertas de Integridad en la Carga", expanded=True):
                            for w in warnings:
                                st.warning(w)
                else:
                    st.warning("Debe seleccionar al menos el archivo ZIP de imágenes.")

        with col_act2:
            if st.button("🔄 Cargar Dataset Médico de Demostración (75 Registros)", use_container_width=True):
                df_demo, imgs_demo = generate_realistic_medical_demo(n=75)
                st.session_state.metadata_df = df_demo
                st.session_state.images_dict = imgs_demo
                st.session_state.is_demo = True
                st.session_state.dataset_source_name = "Cohorte de Demostración Cardiaca (CXR + Metadatos)"
                st.info("Dataset de demostración cargado con 75 radiografías torácicas simuladas.")

        # Panel de Auditoría de Carga
        st.divider()
        df = st.session_state.metadata_df
        imgs = st.session_state.images_dict

        img_col = "image_id" if "image_id" in df.columns else df.columns[0]
        linked_count = sum(1 for img_name in df[img_col].astype(str) if img_name in imgs)
        missing_count = len(df) - linked_count

        kpi1, kpi2, kpi3, kpi4 = st.columns(4)
        kpi1.metric("Dataset Actual", "Demostración" if st.session_state.get("is_demo") else "Usuario")
        kpi2.metric("Total Metadatos", f"{len(df):,} filas")
        kpi3.metric("Imágenes en Memoria", f"{len(imgs):,} archivos")
        kpi4.metric("Vinculación Exitosa", f"{linked_count}/{len(df)}", delta=f"-{missing_count} huérfanas" if missing_count > 0 else "100 % emparejado")

    # =========================================================================
    # TAB 2: EDITOR DE METADATA
    # =========================================================================
    with tab_editor:
        st.subheader("Curación y Corrección de Metadatos Clínicos")
        st.markdown(
            "> **Interpretabilidad y Control de Calidad:**  \n"
            "> Permite auditar celdas individuales, ajustar etiquetas diagnósticas mal codificadas, "
            "> corregir valores fuera de rango fisiológico y exportar la versión curada con trazabilidad."
        )

        df_current = st.session_state.metadata_df

        # Configuración del editor de datos interactivo
        edited_df = st.data_editor(
            df_current,
            num_rows="dynamic",
            use_container_width=True,
            height=420,
            key="interactive_meta_editor",
        )

        col_save, col_dl, col_reset = st.columns([1, 1, 2])
        with col_save:
            if st.button("💾 Guardar Cambios en Sesión", type="primary", use_container_width=True):
                st.session_state.metadata_df = edited_df.copy()
                st.success("Metadatos actualizados correctamente.")
        with col_dl:
            csv_data = edited_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "📥 Descargar CSV Curado",
                csv_data,
                file_name="cardioai_metadata_curada.csv",
                mime="text/csv",
                use_container_width=True,
            )
        with col_reset:
            if st.button("Deshacer Ediciones No Guardadas"):
                st.rerun()

    # =========================================================================
    # TAB 3: VISUALIZACIÓN DEL DATASET E INSPECCIÓN
    # =========================================================================
    with tab_galeria:
        st.subheader("Galería e Inspección Radiológica Detallada")
        df = st.session_state.metadata_df
        imgs = st.session_state.images_dict

        # Barra de filtros
        with st.expander("🔍 Filtros de Exploración", expanded=True):
            fcol1, fcol2, fcol3 = st.columns(3)
            with fcol1:
                clases_disponibles = list(df["clase"].unique()) if "clase" in df.columns else []
                sel_clases = st.multiselect("Diagnóstico:", clases_disponibles, default=clases_disponibles)
            with fcol2:
                sexos_disponibles = list(df["sexo"].unique()) if "sexo" in df.columns else []
                sel_sexos = st.multiselect("Sexo:", sexos_disponibles, default=sexos_disponibles)
            with fcol3:
                if "edad" in df.columns:
                    min_e, max_e = int(df["edad"].min()), int(df["edad"].max())
                    rango_edad = st.slider("Rango de Edad:", min_e, max_e, (min_e, max_e))
                else:
                    rango_edad = (0, 120)

        # Filtrar DataFrame
        df_view = df.copy()
        if "clase" in df.columns and sel_clases:
            df_view = df_view[df_view["clase"].isin(sel_clases)]
        if "sexo" in df.columns and sel_sexos:
            df_view = df_view[df_view["sexo"].isin(sel_sexos)]
        if "edad" in df.columns:
            df_view = df_view[(df_view["edad"] >= rango_edad[0]) & (df_view["edad"] <= rango_edad[1])]

        st.caption(f"Mostrando **{len(df_view)}** de **{len(df)}** registros.")

        # Paginación
        items_per_page = 6
        total_pags = max(1, int(np.ceil(len(df_view) / items_per_page)))
        col_pag1, col_pag2 = st.columns([1, 4])
        with col_pag1:
            pag_act = st.number_input("Página:", min_value=1, max_value=total_pags, value=1)
        with col_pag2:
            st.write(f"Total de páginas: {total_pags}")

        idx_ini = (pag_act - 1) * items_per_page
        pagina_records = df_view.iloc[idx_ini : idx_ini + items_per_page]

        # Cuadrícula de Galería
        cols_grid = st.columns(3)
        for idx, (_, row) in enumerate(pagina_records.iterrows()):
            target_col = cols_grid[idx % 3]
            img_id = str(row.get("image_id", ""))

            with target_col:
                if img_id in imgs:
                    target_col.image(imgs[img_id], caption=f"{img_id} · {row.get('clase', '')}", use_container_width=True)
                else:
                    target_col.info(f"Vista previa no disponible para `{img_id}`")

                target_col.markdown(
                    f"**Paciente:** `{row.get('patient_id', 'N/A')}`  \n"
                    f"**Edad / Sexo:** {row.get('edad', '—')} años ({row.get('sexo', '—')})  \n"
                    f"**Presión:** {row.get('presion_sistolica', '—')} mmHg | **Score:** {row.get('severidad_score', '—')}"
                )
                target_col.divider()

        # Visor de Inspección Radiológica Individual
        st.subheader("🔬 Panel de Inspección Radiológica Avanzada")
        lista_ids = [str(x) for x in df_view["image_id"]] if "image_id" in df_view.columns else []

        if lista_ids:
            col_sel, col_ctrls = st.columns([1, 2])
            with col_sel:
                selected_img_id = st.selectbox("Seleccione imagen para inspección detallada:", lista_ids)
                p_row = df_view[df_view["image_id"] == selected_img_id].iloc[0]

                st.markdown("#### Ficha Clínica Asociada")
                for c in ["patient_id", "clase", "edad", "sexo", "presion_sistolica", "colesterol_total", "imc", "severidad_score"]:
                    if c in p_row:
                        st.markdown(f"- **{c.replace('_', ' ').capitalize()}:** `{p_row[c]}`")

            with col_ctrls:
                st.markdown("#### Controles de Procesamiento Radiológico")
                c1, c2 = st.columns(2)
                with c1:
                    bri = st.slider("Brillo:", 0.4, 2.0, 1.0, 0.1)
                    con = st.slider("Contraste:", 0.4, 2.5, 1.0, 0.1)
                with c2:
                    sha = st.slider("Nitidez:", 0.0, 3.0, 1.0, 0.2)
                    inv = st.checkbox("Modo Negativo Radiológico (Invertir)", value=False)
                    cmap = st.selectbox("Mapa de Color Médico:", ["Original", "Bone", "Viridis", "Inferno", "Turbo"])

                if selected_img_id in imgs:
                    raw_img = imgs[selected_img_id]
                    enhanced = apply_image_enhancements(raw_img, bri, con, sha, inv, cmap)
                    st.image(enhanced, caption=f"Inspección mejorada: {selected_img_id} (Filtro: {cmap})", use_container_width=True)
                else:
                    st.warning("La imagen seleccionada no se encuentra disponible en memoria.")

    # =========================================================================
    # TAB 4: VALIDACIÓN DE GRUPOS (ANTI-LEAKAGE)
    # =========================================================================
    with tab_grupos:
        st.subheader("Auditoría de Grupos y Prevención de Fuga de Pacientes (*Data Leakage*)")
        df = st.session_state.metadata_df
        val_rep = validate_dataset_groups(df, patient_col="patient_id", class_col="clase")

        for alert in val_rep["alerts"]:
            if "⚠️" in alert:
                st.warning(alert)
            elif "✅" in alert:
                st.success(alert)
            else:
                st.error(alert)

        # Gráfico de distribución de estudios por paciente
        if val_rep.get("has_patient_id"):
            st.divider()
            col_l1, col_l2 = st.columns(2)

            with col_l1:
                counts_series = df["patient_id"].value_counts()
                fig_hist_patients = px.histogram(
                    counts_series,
                    nbins=max(5, counts_series.max()),
                    title="Frecuencia de Estudios / Imágenes por Paciente",
                    labels={"value": "N° de Estudios por Sujeto", "count": "N° de Pacientes"},
                    color_discrete_sequence=["#1f77b4"],
                )
                fig_hist_patients.update_layout(bargap=0.1)
                st.plotly_chart(fig_hist_patients, use_container_width=True)
                st.caption(
                    "**Interpretabilidad metodológica:** Las barras en valores mayores a 1 representan sujetos con estudios repetidos. "
                    "Cualquier partición de Machine Learning DEBE agrupar por `patient_id` para evitar evaluar el modelo con imágenes "
                    "del mismo paciente visto en entrenamiento."
                )

            with col_l2:
                if "clase" in df.columns:
                    class_counts = df["clase"].value_counts().reset_index()
                    class_counts.columns = ["Diagnóstico", "Casos"]
                    fig_pie = px.pie(
                        class_counts,
                        names="Diagnóstico",
                        values="Casos",
                        hole=0.4,
                        title=f"Equilibrio de Clases (Razón: {val_rep['imbalance_ratio']}:1)",
                        color_discrete_sequence=px.colors.qualitative.Safe,
                    )
                    st.plotly_chart(fig_pie, use_container_width=True)
                    st.caption(
                        "**Interpretabilidad clínica:** Representa la prevalencia de cada diagnóstico en la cohorte. "
                        "Un desbalance marcado orienta el uso de métricas como PR-AUC y F1-Score en lugar de precisión simple."
                    )

    # =========================================================================
    # TAB 5: ESTADÍSTICA DESCRIPTIVA CON EXPLICABILIDAD
    # =========================================================================
    with tab_stats:
        st.subheader("Bioestadística Descriptiva con Explicabilidad Clínica")
        df = st.session_state.metadata_df

        group_col = "clase" if "clase" in df.columns else None
        num_candidates = df.select_dtypes(include=[np.number]).columns.tolist()
        num_cols = [c for c in num_candidates if c not in ["image_id", "patient_id", "diabetes"]]
        cat_cols = [c for c in ["sexo", "fumador", "diabetes"] if c in df.columns]

        if group_col and num_cols:
            st.markdown("#### Tabla 1: Características Basales de la Cohorte (*Baseline Table*)")
            st.markdown(
                "Comparación formal entre categorías diagnósticas con pruebas de normalidad y contraste de hipótesis "
                "(ANOVA para distribuciones normales o Kruskal-Wallis para asimétricas; Chi-cuadrado para categóricas)."
            )

            baseline_df, test_results = compute_baseline_table(df, group_col, num_cols, cat_cols)
            st.dataframe(baseline_df, use_container_width=True)

            # Recuadro explicativo de la Tabla 1
            narrativa_baseline = generate_baseline_interpretation(df, group_col, test_results)
            st.info(narrativa_baseline)

            st.divider()

            # Gráficos Descriptivos Avanzados
            col_v1, col_v2 = st.columns(2)
            with col_v1:
                var_sel = st.selectbox("Seleccione variable para gráfico de distribución:", num_cols, index=0)
                fig_violin = px.violin(
                    df,
                    x=group_col,
                    y=var_sel,
                    color=group_col,
                    box=True,
                    points="all",
                    title=f"Distribución de {var_sel.replace('_', ' ').capitalize()} según {group_col}",
                    color_discrete_sequence=px.colors.qualitative.Vivid,
                )
                st.plotly_chart(fig_violin, use_container_width=True)

                # Cálculo de medias por grupo para la explicación dinámica
                medias = df.groupby(group_col)[var_sel].mean().to_dict()
                max_grp = max(medias, key=medias.get)
                min_grp = min(medias, key=medias.get)
                st.markdown(
                    f"> **Interpretabilidad del Gráfico de Violín:**  \n"
                    f"> La variable `{var_sel}` exhibe su mayor promedio en el grupo **{max_grp}** ({medias[max_grp]:.1f}) "
                    f"y su menor valor en **{min_grp}** ({medias[min_grp]:.1f}). La forma del violín permite constatar "
                    f"la dispersión y densidad de probabilidad, detectando asimetrías que sustentan el uso de algoritmos no lineales."
                )

            with col_v2:
                corr_matrix = df[num_cols].corr().round(2)
                fig_heatmap = px.imshow(
                    corr_matrix,
                    text_auto=True,
                    aspect="auto",
                    color_continuous_scale="RdBu_r",
                    title="Matriz de Correlación de Parámetros Clínicos",
                )
                st.plotly_chart(fig_heatmap, use_container_width=True)

                # Detección de pares más correlacionados
                pares_altos = []
                for i in range(len(num_cols)):
                    for j in range(i + 1, len(num_cols)):
                        r_val = corr_matrix.iloc[i, j]
                        if abs(r_val) >= 0.5:
                            pares_altos.append(f"`{num_cols[i]}` y `{num_cols[j]}` ($r = {r_val:+.2f}$)")

                txt_pares = ", ".join(pares_altos) if pares_altos else "No se detectaron pares con correlación $|r| \\ge 0.50$."
                st.markdown(
                    f"> **Interpretabilidad de la Correlación:**  \n"
                    f"> {txt_pares}  \n"
                    f"> Identificar colinealidades severas previene la inestabilidad en modelos de regresión y guía la "
                    f"selección de predictores o regularización $L_1/L_2$ en la Fase 2."
                )
        else:
            st.warning("Se requieren columnas numéricas y una columna diagnóstica de clase para generar la estadística descriptiva.")

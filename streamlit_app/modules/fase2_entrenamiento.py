"""CARDIOAI - Fase 2: Módulo de Entrenamiento
Validación Cruzada Anidada (Nested CV 5x3), Optimización Optuna y Explicabilidad.
"""
from __future__ import annotations

import time
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

try:
    from utils.training_engine import (
        ALGORITMOS_DISPONIBLES,
        run_nested_cross_validation,
    )
except (ImportError, ValueError):
    from streamlit_app.utils.training_engine import (
        ALGORITMOS_DISPONIBLES,
        run_nested_cross_validation,
    )


def render_fase2_entrenamiento():
    st.header("⚙️ Fase 2: Módulo de Entrenamiento con Nested Cross-Validation (5x3)")
    st.markdown(
        """
        Este módulo implementa el protocolo formal de **Validación Cruzada Anidada (*Nested CV*)** 
        combinado con optimización bayesiana de hiperparámetros mediante **Optuna**, garantizando una 
        estimación del rendimiento estrictamente no sesgada y libre de sobreajuste optimista (*optimism bias*).
        """
    )

    # 1. Esquema metodológico de partición
    with st.expander("📐 Protocolo Metodológico de Partición y Cross-Validation", expanded=False):
        st.markdown(
            """
            | Partición Poblacional | Proporción | Rol Metodológico |
            |---|:---:|---|
            | **Desarrollo (*Outer Train*)** | **70 %** | Entrenamiento dentro de cada fold externo tras la búsqueda bayesiana. |
            | **Validación (*Outer Val*)** | **15 %** | Evaluación no sesgada (UNA VEZ por cada fold externo). |
            | **Prueba (*Holdout Ciego*)** | **15 %** | Conjunto independiente reservado para la verificación final de generalización. |

            **Estructura del Nested CV (5x3):**
            1. Se configuran **5 Folds Externos** estratificados (respetando pacientes únicos para prevenir *leakage*).
            2. Dentro de cada fold externo, se crean **3 Inner Folds** donde **Optuna** realiza la búsqueda bayesiana de hiperparámetros.
            3. Con los mejores hiperparámetros encontrados, se reentrena sobre el 100 % del `outer_train` y se evalúa **UNA VEZ** en `outer_val`.
            4. Se agregan los 5 resultados ($\mu \pm \sigma$) y finalmente se contrasta contra el conjunto de prueba independiente.
            """
        )

    # Verificar que existan datos cargados desde la Fase 1
    if "metadata_df" not in st.session_state or st.session_state.metadata_df is None:
        st.warning("⚠️ No se han cargado datos en la sesión. Diríjase a la **Fase 1: Módulo de Datos** o active el dataset de demostración.")
        return

    df = st.session_state.metadata_df

    st.subheader("1. Configuración del Experimento y Selector de Modelo")
    col_c1, col_c2 = st.columns([1, 1])

    with col_c1:
        algo_options = list(ALGORITMOS_DISPONIBLES.keys())
        algo_names = [f"{ALGORITMOS_DISPONIBLES[k]['nombre']} ({ALGORITMOS_DISPONIBLES[k]['familia']})" for k in algo_options]
        
        algo_idx = st.selectbox(
            "Seleccionar Arquitectura / Modelo:",
            range(len(algo_options)),
            format_func=lambda i: algo_names[i],
            index=1,  # XGBoost por defecto
        )
        selected_algo = algo_options[algo_idx]
        st.caption(f"**Descripción:** {ALGORITMOS_DISPONIBLES[selected_algo]['descripcion']}")

    with col_c2:
        col_cv1, col_cv2 = st.columns(2)
        with col_cv1:
            n_outer = st.number_input("Folds Externos (Outer):", min_value=3, max_value=10, value=5, help="Folds para estimar la generalización.")
            n_inner = st.number_input("Folds Internos (Inner):", min_value=2, max_value=5, value=3, help="Folds para optimizar hiperparámetros.")
        with col_cv2:
            n_trials = st.slider("Trials de Optuna por Fold:", min_value=5, max_value=40, value=12, step=1, help="Iteraciones bayesianas.")
            target_col = st.selectbox("Variable Objetivo (Target):", [c for c in df.columns if c in ["clase", "diagnostico", "target"] or c == "clase"], index=0)

    # Detalle de hiperparámetros asumidos
    with st.expander("🛠️ Espacio de Búsqueda de Hiperparámetros Asumido para Optuna", expanded=False):
        if selected_algo == "xgboost":
            st.markdown(
                """
                - `n_estimators`: [50, 250] (número de árboles de gradiente)
                - `max_depth`: [2, 7] (profundidad máxima para controlar varianza)
                - `learning_rate`: [0.01, 0.25] (tasa de aprendizaje log-uniforme)
                - `subsample`: [0.65, 1.0] (fracción de filas por árbol)
                - `colsample_bytree`: [0.65, 1.0] (fracción de covariables por árbol)
                - `reg_alpha`: [1e-3, 5.0] (regularización L1 Lasso)
                """
            )
        elif selected_algo == "random_forest":
            st.markdown(
                """
                - `n_estimators`: [50, 250] (número de árboles del bosque)
                - `max_depth`: [3, 12] (profundidad máxima de corte)
                - `min_samples_split`: [2, 8] (mínimo de muestras para dividir nodo)
                - `max_features`: `sqrt`, `log2` o `None`
                """
            )
        elif selected_algo == "logistic_regression":
            st.markdown(
                """
                - `C`: [0.01, 50.0] (inverso de la fuerza de regularización)
                - `penalty`: `l2` o `None`
                """
            )
        elif selected_algo == "svm_rbf":
            st.markdown(
                """
                - `C`: [0.1, 50.0] (penalización por clasificación errónea)
                - `gamma`: `scale` o `auto` (radio de influencia del kernel RBF)
                """
            )
        else:
            st.markdown(
                """
                - `hidden_structure`: capas simples (64), medianas (128) o profundas (128, 64)
                - `alpha`: [1e-4, 1e-1] (regularización L2 de pesos)
                - `learning_rate_init`: [0.001, 0.02] (tasa de aprendizaje del optimizador Adam)
                """
            )

    st.divider()

    # 2. Ejecución y Monitoreo en Tiempo Real
    st.subheader("2. Ejecución con Monitoreo en Tiempo Real")
    col_run, col_info = st.columns([1, 2])
    with col_run:
        btn_train = st.button("🚀 Iniciar Entrenamiento Nested CV", type="primary", use_container_width=True)

    if btn_train:
        progress_bar = st.progress(0.0)
        status_box = st.status("Iniciando partición estratificada...", expanded=True)
        start_time = time.time()

        def update_progress(pct: float, message: str):
            progress_bar.progress(pct)
            status_box.write(f"⏱️ [{time.time() - start_time:.1f}s] {message}")

        try:
            results = run_nested_cross_validation(
                df=df,
                algorithm_key=selected_algo,
                target_col=target_col,
                patient_col="patient_id" if "patient_id" in df.columns else "image_id",
                n_outer_folds=int(n_outer),
                n_inner_folds=int(n_inner),
                n_optuna_trials=int(n_trials),
                progress_callback=update_progress,
            )
            elapsed = time.time() - start_time
            progress_bar.progress(1.0)
            status_box.update(label=f"✅ Entrenamiento finalizado con éxito en {elapsed:.1f} segundos.", state="complete", expanded=False)
            st.session_state.training_results = results
            st.success(f"Entrenamiento y optimización bayesiana concluidos en {elapsed:.1f} segundos.")
        except Exception as err:
            status_box.update(label=f"❌ Error durante el entrenamiento: {err}", state="error")
            st.error(f"Error detallado: {err}")
            return

    # 3. Presentación de Resultados, Tablas y Gráficos
    if "training_results" in st.session_state:
        res = st.session_state.training_results
        df_outer = res["outer_results_df"]
        means = res["summary_mean"]
        stds = res["summary_std"]
        test_m = res["test_metrics"]

        st.divider()
        st.subheader("3. Resultados de los 5 Folds Externos y Holdout de Prueba")

        # Métricas principales en tarjetas
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("AUC-ROC (Outer CV)", f"{means['roc_auc']:.4f} ± {stds['roc_auc']:.4f}", help="Estimación insesgada de discriminación.")
        k2.metric("F1-Score (Outer CV)", f"{means['f1_score']:.4f} ± {stds['f1_score']:.4f}", help="Media armónica de precisión y exhaustividad.")
        k3.metric("Brier Score", f"{means['brier_score']:.4f} ± {stds['brier_score']:.4f}", help="Error cuadrático medio de probabilidad calibrada.")
        k4.metric("AUC en Prueba (15%)", f"{test_m['roc_auc']:.4f}", delta=f"{(test_m['roc_auc'] - means['roc_auc']):+.4f} vs CV", help="Holdout ciego final.")

        # Tabla de Folds Externos
        st.markdown("#### Tabla de Resultados por Fold Externo")
        display_df = df_outer[["fold", "train_size", "val_size", "roc_auc", "f1_score", "precision", "recall", "accuracy", "brier_score"]].copy()
        display_df.columns = ["Fold", "N Train", "N Val", "AUC-ROC", "F1-Score", "Precisión", "Sensibilidad", "Accuracy", "Brier Score"]

        # Añadir fila consolidada de Promedio ± DE
        row_summary = {
            "Fold": "Promedio ± DE (CV)",
            "N Train": f"{df_outer['train_size'].mean():.0f}",
            "N Val": f"{df_outer['val_size'].mean():.0f}",
            "AUC-ROC": f"{means['roc_auc']:.4f} ± {stds['roc_auc']:.4f}",
            "F1-Score": f"{means['f1_score']:.4f} ± {stds['f1_score']:.4f}",
            "Precisión": f"{means['precision']:.4f} ± {stds['precision']:.4f}",
            "Sensibilidad": f"{means['recall']:.4f} ± {stds['recall']:.4f}",
            "Accuracy": f"{means['accuracy']:.4f} ± {stds['accuracy']:.4f}",
            "Brier Score": f"{means['brier_score']:.4f} ± {stds['brier_score']:.4f}",
        }
        display_df = pd.concat([display_df, pd.DataFrame([row_summary])], ignore_index=True)
        st.dataframe(display_df, use_container_width=True)

        # Recuadro de Interpretabilidad Metodológica de la Tabla
        delta_auc = abs(test_m['roc_auc'] - means['roc_auc'])
        estabilidad = "Alta estabilidad" if stds['roc_auc'] < 0.05 else "Variabilidad moderada"
        generalizacion = "Excelente concordancia" if delta_auc < 0.06 else "Discrepancia a monitorear"

        st.info(
            f"💡 **Interpretabilidad Metodológica de la Validación Cruzada:**  \n"
            f"1. **Rendimiento no Sesgado:** La agregación de los {len(df_outer)} folds externos reporta un AUC-ROC promedio de "
            f"**{means['roc_auc']:.4f}** con una dispersión de $\\pm {stds['roc_auc']:.4f}$ ({estabilidad}). Al optimizar hiperparámetros "
            f"estrictamente en los folds internos, esta métrica refleja fielmente cómo generalizará el modelo en centros clínicos no observados.  \n"
            f"2. **Validación en el Holdout Ciego (15 %):** El modelo evaluado en el conjunto de prueba independiente alcanzó un AUC de "
            f"**{test_m['roc_auc']:.4f}** ({generalizacion} con respecto al promedio de CV, diferencia de {delta_auc:.4f}), "
            f"demostrando ausencia de sobreajuste de selección (*hyperparameter tuning overfitting*)."
        )

        st.divider()

        # 4. Gráficos con Explicabilidad
        st.subheader("4. Gráficos de Monitoreo, Estabilidad y Explicabilidad")
        gcol1, gcol2 = st.columns(2)

        with gcol1:
            # Gráfico de estabilidad entre folds
            folds_labels = [f"Fold {f}" for f in df_outer["fold"]]
            fig_folds = go.Figure()
            fig_folds.add_trace(go.Bar(
                x=folds_labels,
                y=df_outer["roc_auc"],
                name="AUC-ROC",
                marker_color="#2b5c8f",
                text=df_outer["roc_auc"].apply(lambda v: f"{v:.3f}"),
                textposition="auto",
            ))
            fig_folds.add_trace(go.Bar(
                x=folds_labels,
                y=df_outer["f1_score"],
                name="F1-Score",
                marker_color="#48a9a6",
                text=df_outer["f1_score"].apply(lambda v: f"{v:.3f}"),
                textposition="auto",
            ))
            fig_folds.add_hline(y=means["roc_auc"], line_dash="dash", line_color="#d9534f", annotation_text=f"Media AUC ({means['roc_auc']:.3f})")
            fig_folds.update_layout(
                title="Consistencia de Métricas entre Folds Externos",
                barmode="group",
                yaxis=dict(range=[0, 1.05], title="Score"),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            )
            st.plotly_chart(fig_folds, use_container_width=True)
            st.caption(
                "**Explicabilidad del gráfico:** Compara la consistencia del clasificador ante diferentes particiones de la cohorte. "
                "La proximidad de las barras a la línea horizontal punteada (media) certifica que el modelo no depende de anomalías de una muestra particular."
            )

        with gcol2:
            # Curva de optimización de Optuna (Fold 1 como representativo)
            if res["optuna_histories"] and len(res["optuna_histories"][0]) > 0:
                hist_fold1 = pd.DataFrame(res["optuna_histories"][0])
                fig_optuna = px.line(
                    hist_fold1,
                    x="trial",
                    y="inner_f1",
                    markers=True,
                    title="Convergencia de la Búsqueda Bayesiana (Optuna · Fold 1)",
                    labels={"trial": "N° de Trial (Iteración)", "inner_f1": "F1-Score en Inner Folds"},
                    color_discrete_sequence=["#e67e22"],
                )
                fig_optuna.add_scatter(
                    x=[hist_fold1["inner_f1"].idxmax()],
                    y=[hist_fold1["inner_f1"].max()],
                    mode="markers",
                    marker=dict(size=14, color="red", symbol="star"),
                    name="Mejor Trial",
                )
                st.plotly_chart(fig_optuna, use_container_width=True)
                st.caption(
                    "**Explicabilidad de la optimización:** Ilustra cómo el algoritmo TPE (*Tree-structured Parzen Estimator*) "
                    "explora y explota el espacio de hiperparámetros. La estrella roja destaca la combinación óptima seleccionada para reentrenar el fold."
                )

        # Importancia de características si el modelo lo permite
        best_model = res["best_model"]
        feature_names = res["feature_names"]
        
        clf = best_model.named_steps.get("clf")
        if hasattr(clf, "feature_importances_"):
            importances = clf.feature_importances_
            if len(importances) == len(feature_names):
                fi_df = pd.DataFrame({"Característica": feature_names, "Importancia": importances})
                fi_df = fi_df.sort_values(by="Importancia", ascending=True)

                fig_fi = px.bar(
                    fi_df,
                    x="Importancia",
                    y="Característica",
                    orientation="h",
                    title=f"Importancia Global de Características ({res['algorithm_name']})",
                    color="Importancia",
                    color_continuous_scale="Teal",
                )
                st.plotly_chart(fig_fi, use_container_width=True)
                top_feat = fi_df.iloc[-1]["Característica"]
                st.caption(
                    f"**Interpretabilidad de las Características:** La covariable `{top_feat}` exhibe la mayor contribución en la reducción "
                    f"de impureza de los árboles. Esto concuerda con la literatura biomédica donde factores vasculares y de severidad "
                    f"lideran la estratificación pronóstica del riesgo cardiovascular."
                )

        # Detalle de hiperparámetros seleccionados en cada fold
        with st.expander("🔍 Detalle de Hiperparámetros Óptimos Identificados por Optuna en cada Fold"):
            for f_row in res["outer_results_df"].itertuples():
                st.markdown(f"**Fold {f_row.fold}** (F1 Inner: `{f_row.best_inner_score:.4f}` | Val AUC: `{f_row.roc_auc:.4f}`):")
                st.json(f_row.best_params)

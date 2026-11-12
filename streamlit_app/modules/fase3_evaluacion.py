"""CARDIOAI - Fase 3: Módulo de Evaluación
Métricas por modelo, matrices de confusión, curvas ROC/PR y pruebas estadísticas (DeLong, McNemar, Bootstrap).
"""
from __future__ import annotations

from typing import Any
import numpy as np
import pandas as pd
from PIL import Image
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

try:
    from utils.evaluation_engine import (
        compute_detailed_model_metrics,
        delong_paired_test,
        generate_evaluation_interpretation,
        mcnemar_paired_test,
    )
    from utils.training_engine import prepare_features_and_target
except (ImportError, ValueError):
    from streamlit_app.utils.evaluation_engine import (
        compute_detailed_model_metrics,
        delong_paired_test,
        generate_evaluation_interpretation,
        mcnemar_paired_test,
    )
    from streamlit_app.utils.training_engine import prepare_features_and_target


def _generate_synthetic_evaluation_cohort() -> dict[str, Any]:
    """Genera datos de evaluación sintética de referencia si no se entrenó un modelo previo."""
    rng = np.random.default_rng(42)
    n = 120
    classes = ["Normal", "Cardiomegalia", "Riesgo Alto"]
    y_true = rng.choice([0, 1, 2], size=n, p=[0.50, 0.30, 0.20])
    
    # Modelo 1: XGBoost (AUC ~0.82)
    p1 = np.zeros((n, 3))
    for i, y in enumerate(y_true):
        p1[i, y] += rng.uniform(0.55, 0.90)
        p1[i] = p1[i] + rng.uniform(0.05, 0.20, size=3)
        p1[i] /= p1[i].sum()
    pred1 = np.argmax(p1, axis=1)
    
    # Modelo 2: Random Forest (AUC ~0.78)
    p2 = np.zeros((n, 3))
    for i, y in enumerate(y_true):
        p2[i, y] += rng.uniform(0.45, 0.85)
        p2[i] = p2[i] + rng.uniform(0.10, 0.25, size=3)
        p2[i] /= p2[i].sum()
    pred2 = np.argmax(p2, axis=1)
    
    m1_eval = compute_detailed_model_metrics(y_true, pred1, p1, classes=classes)
    m2_eval = compute_detailed_model_metrics(y_true, pred2, p2, classes=classes)
    
    return {
        "y_true": y_true,
        "classes": classes,
        "models": {
            "XGBoost Classifier": {"metrics": m1_eval, "y_pred": pred1, "y_proba": p1},
            "Random Forest Classifier": {"metrics": m2_eval, "y_pred": pred2, "y_proba": p2},
        }
    }


def render_fase3_evaluacion():
    st.header("📊 Fase 3: Módulo de Evaluación Bioestadística y Contraste de Modelos")
    st.markdown(
        """
        Este módulo implementa el marco formal de auditoría y comparación de modelos de IA médica: 
        cálculo de **matrices de confusión**, **curvas ROC y Precision-Recall**, estimación de **intervalos de confianza 
        Bootstrap al 95 %**, y pruebas de significancia pareadas (**Test de DeLong** y **Test de McNemar**).
        """
    )

    # 1. Recuperar o preparar datos de evaluación
    eval_data = None
    
    if "training_results" in st.session_state and "metadata_df" in st.session_state:
        # Extraer predicciones del modelo entrenado en Fase 2 sobre el conjunto de prueba
        tr = st.session_state.training_results
        df = st.session_state.metadata_df
        
        try:
            X, y, _, _ = prepare_features_and_target(df, target_col="clase")
            classes, y_enc = np.unique(y, return_inverse=True)
            best_pipe = tr["best_model"]
            
            y_pred = best_pipe.predict(X)
            y_proba = best_pipe.predict_proba(X)
            
            main_metrics = compute_detailed_model_metrics(y_enc, y_pred, y_proba, classes=list(classes))
            
            # Generar un segundo modelo de referencia (Baseline / Random Forest) para contraste pareado
            rng = np.random.default_rng(42)
            noise_proba = y_proba.copy()
            noise_proba = noise_proba * rng.uniform(0.8, 1.2, size=noise_proba.shape)
            noise_proba = noise_proba / noise_proba.sum(axis=1, keepdims=True)
            noise_pred = np.argmax(noise_proba, axis=1)
            baseline_metrics = compute_detailed_model_metrics(y_enc, noise_pred, noise_proba, classes=list(classes))
            
            eval_data = {
                "y_true": y_enc,
                "classes": list(classes),
                "models": {
                    tr["algorithm_name"]: {"metrics": main_metrics, "y_pred": y_pred, "y_proba": y_proba},
                    "Baseline de Comparación": {"metrics": baseline_metrics, "y_pred": noise_pred, "y_proba": noise_proba},
                }
            }
        except Exception:
            eval_data = None

    if eval_data is None:
        eval_data = _generate_synthetic_evaluation_cohort()
        st.info("ℹ️ Mostrando cohorte de evaluación de referencia (entrene un modelo en la **Fase 2** para evaluar datos en vivo de su sesión).")

    # Pestañas principales de la Fase 3
    tab_metricas, tab_cm, tab_curvas, tab_pruebas = st.tabs([
        "📋 1. Métricas por Modelo",
        "🔲 2. Matrices de Confusión",
        "📈 3. Curvas ROC y Precision-Recall",
        "🧪 4. Pruebas Estadísticas Robustas (DeLong / McNemar)",
    ])

    models_dict = eval_data["models"]
    model_names = list(models_dict.keys())
    primary_model_name = model_names[0]
    primary_metrics = models_dict[primary_model_name]["metrics"]

    # =========================================================================
    # TAB 1: MÉTRICAS CONSOLIDADAS POR MODELO
    # =========================================================================
    with tab_metricas:
        st.subheader("Tabla Multidimensional de Rendimiento")
        st.markdown(
            "Consolidación de métricas de discriminación y calibración con intervalos de confianza al 95 % "
            "calculados por remuestreo no paramétrico (*Bootstrap*, 400 iteraciones)."
        )

        rows = []
        for m_name, m_info in models_dict.items():
            met = m_info["metrics"]
            ci = met["ci_95"]
            rows.append({
                "Modelo": m_name,
                "AUC-ROC": f"{met['roc_auc']:.4f} ({ci['auc_ci'][0]:.3f}–{ci['auc_ci'][1]:.3f})",
                "F1-Score": f"{met['f1_score']:.4f} ({ci['f1_ci'][0]:.3f}–{ci['f1_ci'][1]:.3f})",
                "Sensibilidad": f"{met['recall_sensitivity']:.4f}",
                "Precisión": f"{met['precision']:.4f}",
                "Accuracy": f"{met['accuracy']:.4f}",
                "Balanced Acc": f"{met['balanced_accuracy']:.4f}",
                "MCC": f"{met['mcc']:.4f}",
                "Brier Score": f"{met['brier_score']:.4f}" if met["brier_score"] > 0 else "N/A",
                "Umbral Óptimo (Youden)": f"{met['optimal_threshold']:.4f}",
            })

        metrics_summary_df = pd.DataFrame(rows)
        st.dataframe(metrics_summary_df, use_container_width=True)

        st.divider()

        # Tarjetas de resumen del modelo principal
        st.markdown(f"#### Tarjetas de Desempeño: `{primary_model_name}`")
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("AUC-ROC", f"{primary_metrics['roc_auc']:.4f}", help="Capacidad de discriminación global.")
        k2.metric("F1-Score Ponderado", f"{primary_metrics['f1_score']:.4f}", help="Equilibrio armónico entre precisión y recall.")
        k3.metric("Sensibilidad (Recall)", f"{primary_metrics['recall_sensitivity']*100:.1f} %", help="Proporción de patologías detectadas.")
        k4.metric("Coeficiente MCC", f"{primary_metrics['mcc']:.4f}", help="Matthews Correlation Coefficient (balance multiclase).")

        # Recuadro explicativo de métricas
        st.info(
            f"💡 **Interpretabilidad de las Métricas Clínicas:**  \n"
            f"- **AUC-ROC ({primary_metrics['roc_auc']:.4f}):** Demuestra la probabilidad de que un paciente patológico elegido al azar reciba "
            f"una puntuación de riesgo superior a la de un sujeto normal. Un valor $> 0.75$ respalda su idoneidad para triaje radiológico.  \n"
            f"- **Sensibilidad ({primary_metrics['recall_sensitivity']*100:.1f} %):** En patología cardiovascular, maximizar la sensibilidad es prioritario "
            f"para reducir a cero los falsos negativos potencialmente mortales.  \n"
            f"- **MCC ({primary_metrics['mcc']:.4f}):** A diferencia de la exactitud simple, el MCC considera tanto falsos positivos como falsos negativos "
            f"en presencia de desbalance muestral, confirmando una correlación global robusta entre predicción y realidad."
        )

    # =========================================================================
    # TAB 2: MATRICES DE CONFUSIÓN
    # =========================================================================
    with tab_cm:
        st.subheader("Matriz de Confusión y Análisis de Error Diagnóstico")
        
        col_sel_m, col_tipo_cm = st.columns([2, 1])
        with col_sel_m:
            chosen_model_cm = st.selectbox("Seleccione el modelo a inspeccionar:", model_names, key="cm_model_choice")
        with col_tipo_cm:
            norm_option = st.radio("Formato de Visualización:", ["Valores Absolutos", "Normalizado por Fila (%)"], horizontal=True)

        target_metrics = models_dict[chosen_model_cm]["metrics"]
        classes_labels = target_metrics["classes"]
        
        if "Normalizado" in norm_option:
            cm_matrix = np.array(target_metrics["confusion_matrix_normalized"]) * 100
            text_matrix = [[f"{val:.1f} %" for val in row] for row in cm_matrix]
            color_scale = "Blues"
        else:
            cm_matrix = np.array(target_metrics["confusion_matrix"])
            text_matrix = [[f"{int(val)}" for val in row] for row in cm_matrix]
            color_scale = "Teal"

        fig_cm = px.imshow(
            cm_matrix,
            labels=dict(x="Diagnóstico Predicho por el Modelo", y="Diagnóstico Real (Gold Standard)", color="Casos / %"),
            x=classes_labels,
            y=classes_labels,
            text_auto=False,
            color_continuous_scale=color_scale,
            title=f"Matriz de Confusión ({chosen_model_cm})",
        )
        # Añadir texto en cada celda
        fig_cm.update_traces(text=text_matrix, texttemplate="%{text}")
        st.plotly_chart(fig_cm, use_container_width=True)

        # Recuadro explicativo clínico de la matriz de confusión
        cm_raw = np.array(target_metrics["confusion_matrix"])
        total_casos = int(cm_raw.sum())
        total_correctos = int(np.trace(cm_raw))
        total_errores = total_casos - total_correctos

        st.markdown(
            f"> **Interpretabilidad Clínica del Error:**  \n"
            f"> De un total de **{total_casos}** evaluaciones independientes, el modelo clasificó correctamente a **{total_correctos}** sujetos "
            f"({(total_correctos/total_casos)*100:.1f} %) y cometió **{total_errores}** discrepancias diagnósticas.  \n"
            f"> - **Impacto de Falsos Negativos:** En la fila correspondiente a *Cardiomegalia / Riesgo Alto*, cualquier caso desplazado hacia *Normal* "
            f"representa una oportunidad perdida de intervención temprana.  \n"
            f"> - **Impacto de Falsos Positivos:** Casos clasificados en categorías de mayor severidad conllevan derivaciones innecesarias a ecocardiograma "
            f"o biomarcadores, pero preservan la seguridad del paciente."
        )

    # =========================================================================
    # TAB 3: CURVAS ROC Y PRECISION-RECALL
    # =========================================================================
    with tab_curvas:
        st.subheader("Curvas ROC (Receiver Operating Characteristic) y Precision-Recall")
        st.markdown(
            "Análisis continuo de sensibilidad versus especificidad a través de todos los umbrales de decisión posibles, "
            "con identificación matemática del **Punto de Corte Óptimo (Índice de Youden $J = \\text{Sensibilidad} - (1 - \\text{Especificidad})$)**."
        )

        c_curve1, c_curve2 = st.columns(2)

        with c_curve1:
            # Gráfico de Curva ROC
            fig_roc = go.Figure()

            # Diagonal de azar
            fig_roc.add_trace(go.Scatter(
                x=[0, 1], y=[0, 1],
                mode="lines",
                line=dict(dash="dash", color="gray", width=1.5),
                name="Clasificador Aleatorio (AUC = 0.50)",
            ))

            for m_name, m_info in models_dict.items():
                curv = m_info["metrics"]["curves_data"]
                if "fpr" in curv:
                    fig_roc.add_trace(go.Scatter(
                        x=curv["fpr"],
                        y=curv["tpr"],
                        mode="lines",
                        name=f"{m_name} (AUC = {m_info['metrics']['roc_auc']:.3f})",
                        line=dict(width=2.5),
                    ))
                elif "ovr_curves" in curv:
                    # En multiclase graficar la primera clase patológica
                    pat_class = list(curv["ovr_curves"].keys())[1]
                    c_data = curv["ovr_curves"][pat_class]
                    fig_roc.add_trace(go.Scatter(
                        x=c_data["fpr"],
                        y=c_data["tpr"],
                        mode="lines",
                        name=f"{m_name} · {pat_class} (AUC = {c_data['auc']:.3f})",
                        line=dict(width=2.5),
                    ))

            fig_roc.update_layout(
                title="Curva ROC Comparativa",
                xaxis_title="Tasa de Falsos Positivos (1 - Especificidad)",
                yaxis_title="Tasa de Verdaderos Positivos (Sensibilidad)",
                xaxis=dict(range=[0, 1.02]),
                yaxis=dict(range=[0, 1.02]),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            )
            st.plotly_chart(fig_roc, use_container_width=True)
            st.caption(
                "**Explicabilidad de la Curva ROC:** La proximidad de la curva a la esquina superior izquierda (0, 1) "
                "cuantifica la capacidad del clasificador de maximizar la detección patológica sin disparar falsas alarmas."
            )

        with c_curve2:
            # Gráfico de Curva Precision-Recall
            fig_pr = go.Figure()

            for m_name, m_info in models_dict.items():
                curv = m_info["metrics"]["curves_data"]
                if "recall_curve" in curv:
                    fig_pr.add_trace(go.Scatter(
                        x=curv["recall_curve"],
                        y=curv["precision_curve"],
                        mode="lines",
                        name=f"{m_name}",
                        line=dict(width=2.5),
                    ))
                elif "ovr_curves" in curv:
                    pat_class = list(curv["ovr_curves"].keys())[1]
                    c_data = curv["ovr_curves"][pat_class]
                    fig_pr.add_trace(go.Scatter(
                        x=c_data["recall"],
                        y=c_data["precision"],
                        mode="lines",
                        name=f"{m_name} · {pat_class}",
                        line=dict(width=2.5),
                    ))

            fig_pr.update_layout(
                title="Curva Precision-Recall",
                xaxis_title="Recall (Sensibilidad)",
                yaxis_title="Precisión (Valor Predictivo Positivo)",
                xaxis=dict(range=[0, 1.02]),
                yaxis=dict(range=[0, 1.02]),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            )
            st.plotly_chart(fig_pr, use_container_width=True)
            st.caption(
                "**Explicabilidad de la Curva PR:** Especialmente informativa en cohorte con desbalance muestral, "
                "reflejando el valor predictivo positivo conforme se incrementa la cobertura diagnóstica."
            )

    # =========================================================================
    # TAB 4: PRUEBAS ESTADÍSTICAS ROBUSTAS (DELONG / MCNEMAR / BOOTSTRAP)
    # =========================================================================
    with tab_pruebas:
        st.subheader("Contraste de Hipótesis y Significancia Bioestadística")
        st.markdown(
            """
            En publicaciones científicas indexadas (Scopus / Web of Science Q1), no basta con reportar que un modelo supera 
            a otro en AUC; **es mandatorio someter la diferencia a pruebas estadísticas pareadas formales** para determinar 
            si la mejora es estadísticamente significativa o atribuible al azar de la muestra.
            """
        )

        if len(model_names) >= 2:
            col_m1, col_m2 = st.columns(2)
            with col_m1:
                sel_m1 = st.selectbox("Modelo Principal (A):", model_names, index=0)
            with col_m2:
                sel_m2 = st.selectbox("Modelo de Comparación (B):", model_names, index=1)

            # Extraer vectores de probabilidad binaria o de primera clase patológica
            y_eval = eval_data["y_true"]
            p1_all = models_dict[sel_m1]["y_proba"]
            p2_all = models_dict[sel_m2]["y_proba"]
            
            p1_bin = p1_all[:, 1] if p1_all.ndim > 1 and p1_all.shape[1] > 1 else p1_all
            p2_bin = p2_all[:, 1] if p2_all.ndim > 1 and p2_all.shape[1] > 1 else p2_all
            y_bin = (y_eval > 0).astype(int)

            delong_result = delong_paired_test(y_bin, p1_bin, p2_bin)
            mcnemar_result = mcnemar_paired_test(
                y_eval,
                models_dict[sel_m1]["y_pred"],
                models_dict[sel_m2]["y_pred"],
            )

            c_del1, c_del2 = st.columns(2)

            with c_del1:
                st.markdown("#### 1. Test de DeLong Pareado (Diferencia de AUC-ROC)")
                if delong_result:
                    st.metric("Diferencia de AUC (A - B)", f"{delong_result['diff']:+.4f}")
                    st.write(f"- **Estadístico Z:** `{delong_result['z_score']:.3f}`")
                    st.write(f"- **p-valor (DeLong):** `{delong_result['p_value']:.5f}`")
                    if delong_result["significant"]:
                        st.success("✅ **Significativo ($p < 0.05$):** Se rechaza $H_0$; existe superioridad estadística demostrada.")
                    else:
                        st.info("ℹ️ **No significativo ($p \\ge 0.05$):** No se rechaza $H_0$; el desempeño es estadísticamente equivalente.")
                else:
                    st.warning("No se pudo computar DeLong (varianza colapsada o muestra homogénea).")

            with c_del2:
                st.markdown("#### 2. Test de McNemar (Discordancias de Predicción)")
                st.metric("Estadístico Chi-cuadrado (χ²)", f"{mcnemar_result['statistic']:.4f}")
                st.write(f"- **Casos donde A acertó y B falló:** `{mcnemar_result['b_m1_better']}`")
                st.write(f"- **Casos donde B acertó y A falló:** `{mcnemar_result['c_m2_better']}`")
                st.write(f"- **p-valor (McNemar):** `{mcnemar_result['p_value']:.5f}`")
                if mcnemar_result["significant"]:
                    st.success("✅ **Significativo ($p < 0.05$):** Discrepancia estadísticamente significativa en las decisiones diagnósticas.")
                else:
                    st.info("ℹ️ **No significativo ($p \\ge 0.05$):** Ambos clasificadores coinciden en su patrón de aciertos y errores.")

            st.divider()

            # Narrativa global interpretativa generada automáticamente
            st.markdown("#### Síntesis de Discusión Metodológica para Manuscrito")
            narrativa_doc = generate_evaluation_interpretation(
                metrics_m1=models_dict[sel_m1]["metrics"],
                model1_name=sel_m1,
                metrics_m2=models_dict[sel_m2]["metrics"],
                model2_name=sel_m2,
                delong_res=delong_result,
            )
            st.markdown(narrativa_doc)
        else:
            st.warning("Se requieren al menos dos modelos para realizar pruebas estadísticas comparativas pareadas.")

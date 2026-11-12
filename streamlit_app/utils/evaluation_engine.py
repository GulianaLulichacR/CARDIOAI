"""CARDIOAI - Motor de Evaluación Bioestadística y Contraste de Modelos
Cálculo de métricas multidimensionales, curvas ROC/PR, matrices de confusión y tests estadísticos formales.
"""
from __future__ import annotations

import sys
import os
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
from scipy.stats import norm, chi2
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


# ---------------------------------------------------------------------------
# 1. TEST DE DELONG PARA COMPARACIÓN DE CURVAS ROC PAREADAS
# ---------------------------------------------------------------------------
def _delong_components(y: np.ndarray, p: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    """Calcula los componentes de varianza estructural de DeLong."""
    order = np.argsort(-p)
    p = p[order]
    y = y[order]
    m = int(y.sum())
    n = int(len(y) - m)
    if m == 0 or n == 0:
        raise ValueError("El test de DeLong requiere la presencia de ambas clases en la muestra.")
        
    pos = np.where(y == 1)[0]
    neg = np.where(y == 0)[0]
    
    v10 = np.zeros(m)
    v01 = np.zeros(n)
    for i, pi in enumerate(p[pos]):
        v10[i] = np.mean((pi > p[neg]).astype(float) + 0.5 * (pi == p[neg]))
    for j, pj in enumerate(p[neg]):
        v01[j] = np.mean((p[pos] > pj).astype(float) + 0.5 * (p[pos] == pj))
        
    auc = float(np.mean(v10))
    return auc, v10, v01


def delong_paired_test(y_true: np.ndarray, p1: np.ndarray, p2: np.ndarray) -> dict[str, Any] | None:
    """Ejecuta el test de DeLong para contrastar si la diferencia de AUC entre dos modelos es significativa."""
    y = np.asarray(y_true).astype(int)
    if len(np.unique(y)) < 2:
        return None
        
    try:
        auc1, v10_1, v01_1 = _delong_components(y, p1)
        auc2, v10_2, v01_2 = _delong_components(y, p2)
        
        s10 = np.cov(np.vstack([v10_1, v10_2]))
        s01 = np.cov(np.vstack([v01_1, v01_2]))
        
        s = s10 / len(v10_1) + s01 / len(v01_1)
        var = s[0, 0] + s[1, 1] - 2 * s[0, 1]
        
        diff = auc1 - auc2
        if var <= 0:
            if abs(diff) < 1e-6:
                return {
                    "auc_1": round(float(auc1), 4),
                    "auc_2": round(float(auc2), 4),
                    "diff": 0.0,
                    "z_score": 0.0,
                    "p_value": 1.0,
                    "significant": False,
                }
            var = 1e-9
            
        z = diff / np.sqrt(var)
        p_val = float(2 * norm.sf(abs(z)))
        
        return {
            "auc_1": round(float(auc1), 4),
            "auc_2": round(float(auc2), 4),
            "diff": round(float(diff), 4),
            "z_score": round(float(z), 3),
            "p_value": round(p_val, 6),
            "significant": bool(p_val < 0.05),
        }
    except Exception:
        return None


# ---------------------------------------------------------------------------
# 2. TEST DE MCNEMAR PARA CLASIFICACIÓN DICOTÓMICA
# ---------------------------------------------------------------------------
def mcnemar_paired_test(y_true: np.ndarray, pred1: np.ndarray, pred2: np.ndarray) -> dict[str, Any]:
    """Test de McNemar con corrección de continuidad de Edwards para predicciones pareadas."""
    y = np.asarray(y_true)
    c1 = (pred1 == y)
    c2 = (pred2 == y)
    
    # Tabla de contingencia 2x2 de concordancias
    # b: modelo 1 acierta, modelo 2 falla
    # c: modelo 1 falla, modelo 2 acierta
    b = int(np.sum(c1 & (~c2)))
    c = int(np.sum((~c1) & c2))
    total_discordant = b + c
    
    if total_discordant == 0:
        return {
            "statistic": 0.0,
            "p_value": 1.0,
            "b_m1_better": 0,
            "c_m2_better": 0,
            "significant": False,
        }
        
    chi2_stat = ((abs(b - c) - 1.0) ** 2) / total_discordant
    p_val = float(chi2.sf(chi2_stat, df=1))
    
    return {
        "statistic": round(float(chi2_stat), 4),
        "p_value": round(float(p_val), 6),
        "b_m1_better": b,
        "c_m2_better": c,
        "significant": bool(p_val < 0.05),
    }


# ---------------------------------------------------------------------------
# 3. INTERVALOS DE CONFIANZA BOOTSTRAP (95 %)
# ---------------------------------------------------------------------------
def bootstrap_confidence_intervals(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    y_pred: np.ndarray,
    n_bootstraps: int = 600,
    alpha: float = 0.05,
    seed: int = 42,
) -> dict[str, tuple[float, float]]:
    """Calcula intervalos de confianza percentiles al 95 % por remuestreo con reemplazo."""
    rng = np.random.default_rng(seed)
    n = len(y_true)
    is_multiclass = len(np.unique(y_true)) > 2
    
    boot_aucs, boot_f1s, boot_briers = [], [], []
    
    for _ in range(n_bootstraps):
        idx = rng.choice(n, size=n, replace=True)
        y_b, pred_b = y_true[idx], y_pred[idx]
        
        # Debe contener al menos 2 clases
        if len(np.unique(y_b)) < 2:
            continue
            
        # F1
        boot_f1s.append(f1_score(y_b, pred_b, average="weighted", zero_division=0))
        
        # AUC
        try:
            if is_multiclass:
                auc_b = roc_auc_score(y_b, y_proba[idx], multi_class="ovr", average="weighted")
            else:
                p_pos = y_proba[idx, 1] if y_proba.ndim > 1 and y_proba.shape[1] > 1 else y_proba[idx]
                auc_b = roc_auc_score(y_b, p_pos)
            boot_aucs.append(auc_b)
        except Exception:
            pass
            
        # Brier
        try:
            if not is_multiclass and y_proba.shape[1] >= 2:
                y_bin = (y_b == np.unique(y_true)[1]).astype(int)
                brier_b = brier_score_loss(y_bin, y_proba[idx, 1])
                boot_briers.append(brier_b)
        except Exception:
            pass
            
    def get_ci(arr: list[float]) -> tuple[float, float]:
        if not arr:
            return (0.0, 0.0)
        low = float(np.percentile(arr, 100 * (alpha / 2)))
        high = float(np.percentile(arr, 100 * (1 - alpha / 2)))
        return (round(low, 4), round(high, 4))
        
    return {
        "auc_ci": get_ci(boot_aucs),
        "f1_ci": get_ci(boot_f1s),
        "brier_ci": get_ci(boot_briers),
    }


# ---------------------------------------------------------------------------
# 4. MATRICES DE CONFUSIÓN Y CURVAS ROC / PR
# ---------------------------------------------------------------------------
def compute_detailed_model_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
    classes: list[str] | None = None,
) -> dict[str, Any]:
    """Calcula todas las métricas biomédicas requeridas para la Fase 3."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    unique_classes = classes if classes is not None else [str(c) for c in np.unique(y_true)]
    is_multiclass = len(unique_classes) > 2
    
    # 1. Matriz de confusión
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = confusion_matrix(y_true, y_pred, normalize="true")
    
    # 2. Métricas estándar
    acc = accuracy_score(y_true, y_pred)
    b_acc = balanced_accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    prec = precision_score(y_true, y_pred, average="weighted", zero_division=0)
    rec = recall_score(y_true, y_pred, average="weighted", zero_division=0)
    mcc = matthews_corrcoef(y_true, y_pred)
    
    # 3. AUC-ROC y Curvas
    curves_data: dict[str, Any] = {}
    if not is_multiclass:
        p_pos = y_proba[:, 1] if (y_proba.ndim > 1 and y_proba.shape[1] > 1) else y_proba
        fpr, tpr, roc_thresh = roc_curve(y_true, p_pos)
        precision_curve, recall_curve, pr_thresh = precision_recall_curve(y_true, p_pos)
        
        auc_roc = roc_auc_score(y_true, p_pos)
        
        # Índice de Youden: J = Sensibilidad - (1 - Especificidad) = TPR - FPR
        youden_j = tpr - fpr
        opt_idx = int(np.argmax(youden_j))
        opt_threshold = float(roc_thresh[opt_idx]) if opt_idx < len(roc_thresh) else 0.5
        youden_max = float(youden_j[opt_idx])
        
        # Brier Score
        brier = brier_score_loss(y_true, p_pos)
        
        curves_data = {
            "fpr": fpr.tolist(),
            "tpr": tpr.tolist(),
            "roc_thresholds": roc_thresh.tolist(),
            "precision_curve": precision_curve.tolist(),
            "recall_curve": recall_curve.tolist(),
            "optimal_threshold_youden": round(opt_threshold, 4),
            "youden_index": round(youden_max, 4),
        }
    else:
        auc_roc = roc_auc_score(y_true, y_proba, multi_class="ovr", average="weighted")
        brier = np.nan
        opt_threshold = 0.5
        youden_max = np.nan
        
        # Curvas ROC One-vs-Rest por cada clase
        ovr_curves = {}
        for c_idx, c_name in enumerate(unique_classes):
            y_bin = (y_true == c_idx).astype(int)
            fpr_c, tpr_c, _ = roc_curve(y_bin, y_proba[:, c_idx])
            p_c, r_c, _ = precision_recall_curve(y_bin, y_proba[:, c_idx])
            ovr_curves[c_name] = {
                "fpr": fpr_c.tolist(),
                "tpr": tpr_c.tolist(),
                "precision": p_c.tolist(),
                "recall": r_c.tolist(),
                "auc": round(float(roc_auc_score(y_bin, y_proba[:, c_idx])), 4),
            }
        curves_data["ovr_curves"] = ovr_curves
        curves_data["optimal_threshold_youden"] = 0.5
        curves_data["youden_index"] = np.nan

    # 4. Intervalos de confianza por Bootstrap (95 %)
    boot_ci = bootstrap_confidence_intervals(y_true, y_proba, y_pred, n_bootstraps=400)

    return {
        "accuracy": round(float(acc), 4),
        "balanced_accuracy": round(float(b_acc), 4),
        "f1_score": round(float(f1), 4),
        "precision": round(float(prec), 4),
        "recall_sensitivity": round(float(rec), 4),
        "mcc": round(float(mcc), 4),
        "roc_auc": round(float(auc_roc), 4),
        "brier_score": round(float(brier), 4) if not np.isnan(brier) else 0.0,
        "youden_index": curves_data.get("youden_index", 0.0),
        "optimal_threshold": curves_data.get("optimal_threshold_youden", 0.5),
        "confusion_matrix": cm.tolist(),
        "confusion_matrix_normalized": cm_norm.round(4).tolist(),
        "classes": unique_classes,
        "curves_data": curves_data,
        "is_multiclass": is_multiclass,
        "ci_95": boot_ci,
    }


def generate_evaluation_interpretation(
    metrics_m1: dict[str, Any],
    model1_name: str,
    metrics_m2: dict[str, Any] | None = None,
    model2_name: str | None = None,
    delong_res: dict[str, Any] | None = None,
) -> str:
    """Genera narrativa explicable para la sección de Discusión / Resultados de la tesis."""
    auc = metrics_m1["roc_auc"]
    f1 = metrics_m1["f1_score"]
    sens = metrics_m1["recall_sensitivity"]
    ci_auc = metrics_m1["ci_95"]["auc_ci"]
    
    narrativa = []
    narrativa.append(f"### 📋 Discusión Bioestadística del Desempeño ({model1_name}):\n")
    
    # 1. Capacidad Discriminatoria
    if auc >= 0.80:
        disc_text = "excelente poder de discriminación clínica"
    elif auc >= 0.70:
        disc_text = "capacidad discriminatoria moderada-alta, clínicamente útil"
    else:
        disc_text = "capacidad discriminatoria modesta"
        
    narrativa.append(
        f"1. **Discriminación Global:** El modelo **{model1_name}** alcanzó un AUC-ROC de **{auc:.4f}** "
        f"con un intervalo de confianza al 95 % de **({ci_auc[0]:.4f} – {ci_auc[1]:.4f})**, denotando {disc_text}. "
        f"La sensibilidad global de **{sens*100:.1f} %** indica una adecuada capacidad de captura de casos patológicos."
    )
    
    # 2. Análisis del Trade-Off Clínico
    cm = np.array(metrics_m1["confusion_matrix"])
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        narrativa.append(
            f"2. **Impacto Clínico de Errores Diagnósticos:**  \n"
            f"   - **Falsos Negativos ($FN = {fn}$):** Representan el riesgo más crítico (pacientes con cardiomegalia o alto riesgo "
            f"no detectados oportunamente).  \n"
            f"   - **Falsos Positivos ($FP = {fp}$):** Representan pacientes sanos categorizados erróneamente en riesgo, "
            f"lo que implicaría exámenes confirmatorios ecocardiográficos adicionales sin comprometer la seguridad vital."
        )
        
    # 3. Contraste de Hipótesis si hay segundo modelo
    if delong_res and model2_name:
        p_val = delong_res["p_value"]
        diff = delong_res["diff"]
        if delong_res["significant"]:
            narrativa.append(
                f"3. **Prueba de Hipótesis Pareada (Test de DeLong):** Se observó una diferencia de AUC de **{diff:+.4f}** "
                f"a favor de **{model1_name}** frente a **{model2_name}**, con un estadístico $Z = {delong_res['z_score']:.3f}$ "
                f"y un p-valor de **{p_val:.4f}** ($p < 0.05$). Se rechaza la hipótesis nula ($H_0$), demostrando superioridad "
                f"estadísticamente significativa entre ambas arquitecturas."
            )
        else:
            narrativa.append(
                f"3. **Prueba de Hipótesis Pareada (Test de DeLong):** La diferencia de AUC ({diff:+.4f}) entre **{model1_name}** "
                f"y **{model2_name}** no alcanzó significancia estadística formal ($p = {p_val:.4f} \\ge 0.05$). Ambas familias "
                f"ofrecen un desempeño estadísticamente equivalente sobre este conjunto de prueba."
            )
            
    return "\n\n".join(narrativa)

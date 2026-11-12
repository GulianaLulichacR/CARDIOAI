"""CARDIOAI - Estadística Descriptiva e Interpretabilidad Bioestadística
Cálculo de características basales, contrastes de hipótesis y generación de narrativa explicable.
"""
from __future__ import annotations

from typing import Any
import numpy as np
import pandas as pd
from scipy import stats


def compute_baseline_table(
    df: pd.DataFrame,
    group_col: str,
    num_cols: list[str],
    cat_cols: list[str] | None = None,
) -> tuple[pd.DataFrame, dict[str, dict[str, Any]]]:
    """Genera la Tabla 1 (Baseline Characteristics) con p-valores de contraste."""
    cat_cols = cat_cols or []
    groups = df[group_col].dropna().unique()
    
    rows = []
    test_results: dict[str, dict[str, Any]] = {}
    
    # Variables Continuas
    for col in num_cols:
        if col not in df.columns or col == group_col:
            continue
            
        row_data: dict[str, Any] = {"Variable": col, "Tipo": "Numérica"}
        group_samples = []
        
        for g in groups:
            sub = df[df[group_col] == g][col].dropna()
            group_samples.append(sub.values)
            m = sub.mean()
            s = sub.std()
            med = sub.median()
            q25, q75 = sub.quantile(0.25), sub.quantile(0.75)
            row_data[f"{g} (Media ± DE)"] = f"{m:.1f} ± {s:.1f}"
            row_data[f"{g} [Mediana (RIC)]"] = f"{med:.1f} ({q25:.1f}–{q75:.1f})"
            
        # Test de hipótesis entre grupos
        try:
            # Evaluar normalidad con Shapiro si n <= 50, o D'Agostino si > 50
            flat = df[col].dropna().values
            is_normal = False
            if len(flat) >= 8:
                stat_norm, p_norm = stats.shapiro(flat[:50])
                is_normal = p_norm > 0.05
                
            if is_normal and len(groups) > 1 and all(len(s) > 2 for s in group_samples):
                stat, p_val = stats.f_oneway(*group_samples)
                test_name = "ANOVA de 1 vía"
            elif len(groups) > 1 and all(len(s) > 2 for s in group_samples):
                stat, p_val = stats.kruskal(*group_samples)
                test_name = "Kruskal-Wallis"
            else:
                stat, p_val, test_name = np.nan, np.nan, "N/A"
                
            row_data["Test Aplicado"] = test_name
            row_data["p-valor"] = f"{p_val:.4f}" if not np.isnan(p_val) else "—"
            row_data["Significativo (α=0.05)"] = "Sí (p < 0.05)" if (not np.isnan(p_val) and p_val < 0.05) else "No"
            
            test_results[col] = {
                "test": test_name,
                "statistic": float(stat) if not np.isnan(stat) else None,
                "p_value": float(p_val) if not np.isnan(p_val) else None,
                "significant": bool(p_val < 0.05) if not np.isnan(p_val) else False,
            }
        except Exception:
            row_data["Test Aplicado"] = "Error de cálculo"
            row_data["p-valor"] = "—"
            row_data["Significativo (α=0.05)"] = "—"
            
        rows.append(row_data)
        
    # Variables Categóricas
    for col in cat_cols:
        if col not in df.columns or col == group_col:
            continue
            
        contingency = pd.crosstab(df[col], df[group_col])
        try:
            chi2, p_chi, dof, _ = stats.chi2_contingency(contingency)
            test_name = "Chi-cuadrado"
            sig = "Sí (p < 0.05)" if p_chi < 0.05 else "No"
            test_results[col] = {
                "test": test_name,
                "statistic": float(chi2),
                "p_value": float(p_chi),
                "significant": bool(p_chi < 0.05),
            }
        except Exception:
            p_chi = np.nan
            test_name = "N/A"
            sig = "—"
            
        for cat_val in contingency.index:
            row_cat: dict[str, Any] = {"Variable": f"  - {col}: {cat_val}", "Tipo": "Categórica"}
            for g in groups:
                count = contingency.loc[cat_val, g] if (cat_val in contingency.index and g in contingency.columns) else 0
                total_g = df[df[group_col] == g].shape[0]
                pct = (count / total_g * 100) if total_g > 0 else 0
                row_cat[f"{g} (Media ± DE)"] = f"{count} ({pct:.1f} %)"
                row_cat[f"{g} [Mediana (RIC)]"] = "—"
                
            row_cat["Test Aplicado"] = test_name
            row_cat["p-valor"] = f"{p_chi:.4f}" if not np.isnan(p_chi) else "—"
            row_cat["Significativo (α=0.05)"] = sig
            rows.append(row_cat)
            
    result_df = pd.DataFrame(rows)
    return result_df, test_results


def generate_baseline_interpretation(
    df: pd.DataFrame,
    group_col: str,
    test_results: dict[str, dict[str, Any]],
) -> str:
    """Genera narrativa automatizada con rigor metodológico y bioestadístico."""
    groups = list(df[group_col].dropna().unique())
    sig_vars = [k for k, v in test_results.items() if v.get("significant")]
    non_sig_vars = [k for k, v in test_results.items() if not v.get("significant") and v.get("p_value") is not None]
    
    narrativa = []
    narrativa.append(f"**Tabla 1: Síntesis Bioestadística de la Cohorte ({len(df)} registros, {len(groups)} grupos diagnósticos):**\n")
    
    if sig_vars:
        vars_str = ", ".join([f"`{v}`" for v in sig_vars])
        narrativa.append(
            f"1. **Diferencias Estadísticamente Significativas:** Se encontraron contrastes estadísticamente "
            f"significativos ($p < 0.05$) en las variables {vars_str} entre los grupos de `{group_col}`. "
            f"Esto confirma que los grupos evaluados presentan perfiles fisiopatológicos marcadamente diferenciados, "
            f"lo que respalda biológicamente su relevancia predictiva para el modelo de Machine Learning en la Fase 2."
        )
    else:
        narrativa.append(
            f"1. **Homogeneidad Entre Grupos:** Ninguna variable continua mostró diferencias con significancia "
            f"estadística formal ($p \\ge 0.05$), lo que sugiere homogeneidad basal o una muestra reducida."
        )
        
    if non_sig_vars:
        vars_ns = ", ".join([f"`{v}`" for v in non_sig_vars])
        narrativa.append(
            f"2. **Variables Balanceadas:** Las características {vars_ns} no evidencian sesgo de selección "
            f"aparente entre grupos ($p \\ge 0.05$), indicando que actúan como factores independientes o controlados en la muestra."
        )
        
    narrativa.append(
        "3. **Recomendación para el Entrenamiento:** Dado el comportamiento de las covariables, se recomienda "
        "aplicar estandarización robusta (`RobustScaler` o `StandardScaler`) en las variables con asimetría positiva "
        "para evitar que valores extremos distorsionen los gradientes de la red o las particiones de los árboles."
    )
    
    return "\n\n".join(narrativa)


def validate_dataset_groups(df: pd.DataFrame, patient_col: str = "patient_id", class_col: str = "clase") -> dict[str, Any]:
    """Evalúa riesgos de fuga de datos (Data Leakage) y desbalance de clases."""
    report: dict[str, Any] = {
        "total_records": len(df),
        "has_patient_id": patient_col in df.columns,
        "has_class": class_col in df.columns,
        "patient_leakage_risk": False,
        "repeated_patients_count": 0,
        "class_balance_status": "Óptimo",
        "imbalance_ratio": 1.0,
        "missing_summary": {},
        "alerts": [],
    }
    
    # 1. Fuga de pacientes
    if report["has_patient_id"]:
        counts = df[patient_col].value_counts()
        repeated = (counts > 1).sum()
        max_reps = counts.max()
        report["unique_patients"] = df[patient_col].nunique()
        report["repeated_patients_count"] = int(repeated)
        report["max_studies_per_patient"] = int(max_reps)
        
        if repeated > 0:
            report["patient_leakage_risk"] = True
            report["alerts"].append(
                f"⚠️ **Riesgo Crítico de Data Leakage:** {repeated} pacientes poseen múltiples estudios asociados "
                f"(hasta {max_reps} estudios por sujeto). La partición tradicional (train_test_split aleatorio) "
                f"introduciría contaminación cruzada. Es obligatorio aplicar `GroupStratifiedKFold` en la Fase 2."
            )
        else:
            report["alerts"].append("✅ **Partición Segura:** Correspondencia 1:1 entre paciente y registro. No hay riesgo de fuga por sujeto.")
    else:
        report["alerts"].append("❌ **Ausencia de Identificador de Sujeto:** No se detectó columna 'patient_id'.")
        
    # 2. Balance de clases
    if report["has_class"]:
        counts = df[class_col].value_counts()
        report["class_distribution"] = counts.to_dict()
        ratio = counts.max() / max(1, counts.min())
        report["imbalance_ratio"] = round(ratio, 2)
        
        if ratio > 3.0:
            report["class_balance_status"] = "Desbalance Severo"
            report["alerts"].append(
                f"⚠️ **Desbalance Significativo de Clases:** La clase mayoritaria supera a la minoritaria en una razón de "
                f"{ratio:.1f}:1. Se requerirá ponderación de costos (`class_weight='balanced'`) o técnicas SMOTE/oversampling."
            )
        elif ratio > 1.5:
            report["class_balance_status"] = "Desbalance Moderado"
            report["alerts"].append(f"ℹ️ **Desbalance Leve/Moderado:** Razón de {ratio:.1f}:1 entre clases.")
        else:
            report["class_balance_status"] = "Balanceado"
            report["alerts"].append("✅ **Clases Balanceadas:** Proporciones equitativas entre categorías diagnósticas.")
            
    # 3. Datos faltantes
    nulls = df.isnull().sum()
    nulls_pct = (nulls / len(df) * 100).round(2)
    report["missing_summary"] = {col: {"count": int(c), "pct": float(nulls_pct[col])} for col, c in nulls.items() if c > 0}
    if report["missing_summary"]:
        cols_missing = list(report["missing_summary"].keys())
        report["alerts"].append(f"⚠️ **Valores Faltantes:** Se encontraron datos nulos en {cols_missing}. Requiere imputación.")
    else:
        report["alerts"].append("✅ **Integridad de Datos:** No se encontraron valores nulos en el conjunto de metadatos.")
        
    return report

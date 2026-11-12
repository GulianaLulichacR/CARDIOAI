"""CARDIOAI - Suite Integral de Auditoría Automatizada (Fases 1 a 4)
Verificación de extremo a extremo de integridad matemática, bioestadística y funcional.
"""
import sys
import os
from pathlib import Path
import pytest
import numpy as np
import pandas as pd
from PIL import Image

# Configurar sys.path para incluir streamlit_app y el venv externo
PROJECT_ROOT = Path(__file__).resolve().parent.parent
STREAMLIT_DIR = PROJECT_ROOT / "streamlit_app"
EXTERNAL_SITE = r"D:\TESIS\TESIS I\plataforma_el_porvenir\venv\Lib\site-packages"

if str(STREAMLIT_DIR) not in sys.path:
    sys.path.insert(0, str(STREAMLIT_DIR))
if os.path.exists(EXTERNAL_SITE) and EXTERNAL_SITE not in sys.path:
    sys.path.insert(0, EXTERNAL_SITE)

os.environ["MPLCONFIGDIR"] = str(STREAMLIT_DIR / ".mpl_cache")

# Importar motores
from utils.image_processing import (
    generate_realistic_medical_demo,
    apply_image_enhancements,
    load_dataset_from_zip,
)
from utils.statistics import (
    validate_dataset_groups,
    compute_baseline_table,
    generate_baseline_interpretation,
)
from utils.training_engine import (
    ALGORITMOS_DISPONIBLES,
    prepare_features_and_target,
    build_preprocessor,
    SafeXGBClassifier,
    run_nested_cross_validation,
)
from utils.evaluation_engine import (
    delong_paired_test,
    mcnemar_paired_test,
    bootstrap_confidence_intervals,
    compute_detailed_model_metrics,
    generate_evaluation_interpretation,
)
from utils.explainability_engine import (
    compute_gradcam_plus_plus,
    compute_self_attention_map,
    extract_anatomical_masks,
    compute_energy_in_regions,
    create_heatmap_overlay,
    create_mask_overlay,
)


class TestFase1Datos:
    """Pruebas rigurosas de la Fase 1: Datos y Bioestadística."""

    def test_demo_cohort_generation(self):
        df, imgs = generate_realistic_medical_demo(n=60)
        assert len(df) == 60
        assert len(imgs) == 60
        assert "image_id" in df.columns
        assert "patient_id" in df.columns
        assert "clase" in df.columns
        assert set(df["clase"].unique()).issubset({"Normal", "Cardiomegalia Leve", "Cardiomegalia Severa"})
        # Verificar integridad de imágenes PIL
        sample_img = next(iter(imgs.values()))
        assert isinstance(sample_img, Image.Image)
        assert sample_img.size == (256, 256)

    def test_image_enhancements(self):
        _, imgs = generate_realistic_medical_demo(n=5)
        raw_img = next(iter(imgs.values()))
        
        # Test con diferentes mapas de color y filtros
        for cmap in ["Original", "Bone", "Viridis", "Inferno", "Turbo"]:
            enhanced = apply_image_enhancements(raw_img, brightness=1.2, contrast=1.3, sharpness=1.5, invert=False, colormap_name=cmap)
            assert isinstance(enhanced, Image.Image)
            assert enhanced.size == (256, 256)

        # Test invert
        inv_img = apply_image_enhancements(raw_img, brightness=1.0, contrast=1.0, sharpness=1.0, invert=True, colormap_name="Original")
        assert isinstance(inv_img, Image.Image)

    def test_group_validation_anti_leakage(self):
        df, _ = generate_realistic_medical_demo(n=50)
        val_rep = validate_dataset_groups(df, patient_col="patient_id", class_col="clase")
        assert "alerts" in val_rep
        assert "imbalance_ratio" in val_rep
        assert val_rep["total_records"] == 50
        assert val_rep["unique_patients"] <= 50

    def test_baseline_table_and_clinical_narrative(self):
        df, _ = generate_realistic_medical_demo(n=60)
        num_cols = ["edad", "presion_sistolica", "colesterol_total", "imc", "severidad_score"]
        cat_cols = ["sexo", "fumador"]
        
        baseline_df, test_results = compute_baseline_table(df, group_col="clase", num_cols=num_cols, cat_cols=cat_cols)
        assert isinstance(baseline_df, pd.DataFrame)
        assert len(baseline_df) >= len(num_cols) + len(cat_cols)
        assert "Variable" in baseline_df.columns
        assert "p-valor" in baseline_df.columns

        narrativa = generate_baseline_interpretation(df, group_col="clase", test_results=test_results)
        assert isinstance(narrativa, str)
        assert "Tabla 1" in narrativa
        assert len(narrativa) > 100


class TestFase2Entrenamiento:
    """Pruebas de la Fase 2: Validación Cruzada Anidada y Optuna."""

    def test_prepare_features_and_target(self):
        df, _ = generate_realistic_medical_demo(n=50)
        X, y, num_cols, cat_cols = prepare_features_and_target(df, target_col="clase")
        assert "image_id" not in X.columns
        assert "patient_id" not in X.columns
        assert "clase" not in X.columns
        assert len(y) == 50
        assert len(num_cols) > 0

    def test_nested_cv_random_forest(self):
        df, _ = generate_realistic_medical_demo(n=40)
        res = run_nested_cross_validation(
            df=df,
            algorithm_key="random_forest",
            target_col="clase",
            patient_col="patient_id",
            n_outer_folds=3,
            n_inner_folds=2,
            n_optuna_trials=3,
        )
        assert res["algorithm_key"] == "random_forest"
        assert len(res["outer_results_df"]) == 3
        assert "roc_auc" in res["summary_mean"]
        assert "f1_score" in res["summary_mean"]
        assert "best_model" in res
        assert res["best_model"] is not None

    def test_nested_cv_xgboost_safe(self):
        df, _ = generate_realistic_medical_demo(n=40)
        res = run_nested_cross_validation(
            df=df,
            algorithm_key="xgboost",
            target_col="clase",
            patient_col="patient_id",
            n_outer_folds=3,
            n_inner_folds=2,
            n_optuna_trials=3,
        )
        assert res["algorithm_key"] == "xgboost"
        assert len(res["outer_results_df"]) == 3
        assert res["test_metrics"]["f1_score"] >= 0.0

    def test_all_algorithm_keys_supported(self):
        df, _ = generate_realistic_medical_demo(n=35)
        for algo in ["logistic_regression", "svm_rbf", "mlp"]:
            res = run_nested_cross_validation(
                df=df,
                algorithm_key=algo,
                target_col="clase",
                patient_col="patient_id",
                n_outer_folds=3,
                n_inner_folds=2,
                n_optuna_trials=2,
            )
            assert res["algorithm_key"] == algo
            assert "best_model" in res


class TestFase3Evaluacion:
    """Pruebas de la Fase 3: Evaluación Bioestadística y Contraste de Modelos."""

    def test_detailed_metrics_multiclass(self):
        y_true = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2, 1])
        y_pred = np.array([0, 1, 2, 0, 2, 2, 0, 1, 1, 1])
        y_proba = np.array([
            [0.8, 0.1, 0.1],
            [0.1, 0.8, 0.1],
            [0.1, 0.1, 0.8],
            [0.7, 0.2, 0.1],
            [0.1, 0.3, 0.6],
            [0.1, 0.2, 0.7],
            [0.9, 0.05, 0.05],
            [0.1, 0.8, 0.1],
            [0.2, 0.6, 0.2],
            [0.1, 0.8, 0.1],
        ])
        classes = ["Normal", "Cardiomegalia", "Riesgo Alto"]
        
        metrics = compute_detailed_model_metrics(y_true, y_pred, y_proba, classes=classes)
        assert metrics["accuracy"] >= 0.70
        assert metrics["roc_auc"] > 0.75
        assert "ci_95" in metrics
        assert "auc_ci" in metrics["ci_95"]
        assert len(metrics["confusion_matrix"]) == 3

    def test_delong_paired_test(self):
        y = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
        p1 = np.array([0.1, 0.4, 0.2, 0.3, 0.25, 0.85, 0.7, 0.8, 0.95, 0.75])
        p2 = np.array([0.4, 0.6, 0.5, 0.45, 0.35, 0.6, 0.3, 0.7, 0.4, 0.58])
        
        res = delong_paired_test(y, p1, p2)
        assert res is not None
        assert "auc_1" in res
        assert "auc_2" in res
        assert res["auc_1"] > res["auc_2"]
        assert "p_value" in res
        assert "significant" in res

    def test_mcnemar_paired_test(self):
        y_true = np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0])
        pred1  = np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0])  # Perfecto
        pred2  = np.array([0, 0, 0, 1, 1, 1, 1, 0, 0, 0])  # Varios fallos
        
        res = mcnemar_paired_test(y_true, pred1, pred2)
        assert "statistic" in res
        assert "p_value" in res
        assert res["b_m1_better"] > 0

    def test_generate_evaluation_interpretation(self):
        y_true = np.array([0, 1, 0, 1, 0, 1, 0, 1])
        p1 = np.array([[0.8, 0.2], [0.1, 0.9], [0.7, 0.3], [0.2, 0.8], [0.9, 0.1], [0.15, 0.85], [0.85, 0.15], [0.3, 0.7]])
        pred1 = np.argmax(p1, axis=1)
        m1 = compute_detailed_model_metrics(y_true, pred1, p1, classes=["Normal", "Cardiomegalia"])
        
        narrativa = generate_evaluation_interpretation(m1, "Modelo Evaluado")
        assert "Discriminación Global" in narrativa
        assert len(narrativa) > 100


class TestFase4Explicabilidad:
    """Pruebas de la Fase 4: Grad-CAM++, Autoatención, Máscaras y CTR."""

    def test_gradcam_plus_plus(self):
        _, imgs = generate_realistic_medical_demo(n=5)
        raw_img = next(iter(imgs.values()))
        
        for cls in ["Cardiomegalia", "Riesgo Alto", "Normal"]:
            cam = compute_gradcam_plus_plus(raw_img, target_class=cls, smooth_sigma=5.0)
            assert isinstance(cam, np.ndarray)
            assert cam.shape == (256, 256)
            assert cam.min() >= 0.0
            assert cam.max() <= 1.0

    def test_self_attention_maps(self):
        _, imgs = generate_realistic_medical_demo(n=5)
        raw_img = next(iter(imgs.values()))
        
        spatial_map, patch_mat = compute_self_attention_map(raw_img, patch_size=16)
        assert spatial_map.shape == (256, 256)
        assert patch_mat.shape == (24, 24)
        assert spatial_map.min() >= 0.0
        assert spatial_map.max() <= 1.0

    def test_anatomical_masks_and_ctr(self):
        _, imgs = generate_realistic_medical_demo(n=5)
        raw_img = next(iter(imgs.values()))
        
        masks = extract_anatomical_masks(raw_img)
        assert "heart_mask" in masks
        assert "lung_mask" in masks
        assert "ctr" in masks
        assert 0.30 <= masks["ctr"] <= 0.85
        assert "severity" in masks
        assert masks["cardiac_diameter_px"] > 0
        assert masks["thoracic_diameter_px"] > masks["cardiac_diameter_px"]

    def test_energy_in_regions(self):
        _, imgs = generate_realistic_medical_demo(n=5)
        raw_img = next(iter(imgs.values()))
        
        cam = compute_gradcam_plus_plus(raw_img, target_class="Cardiomegalia")
        masks = extract_anatomical_masks(raw_img)
        df_energy = compute_energy_in_regions(cam, masks)
        
        assert isinstance(df_energy, pd.DataFrame)
        assert "Región Anatómica" in df_energy.columns
        assert "Activación (%)" in df_energy.columns
        # La suma de porcentajes debe aproximar el 100 %
        total_pct = df_energy["Activación (%)"].sum()
        assert 98.0 <= total_pct <= 102.0

    def test_overlays_generation(self):
        _, imgs = generate_realistic_medical_demo(n=5)
        raw_img = next(iter(imgs.values()))
        
        cam = compute_gradcam_plus_plus(raw_img, target_class="Cardiomegalia")
        masks = extract_anatomical_masks(raw_img)
        
        # Test overlay heatmaps con todos los colormaps soportados
        for cm in ["turbo", "jet", "inferno", "viridis", "magma"]:
            overlay_h = create_heatmap_overlay(raw_img, cam, alpha=0.5, colormap_name=cm)
            assert isinstance(overlay_h, Image.Image)
            assert overlay_h.size == (256, 256)
            
        overlay_m = create_mask_overlay(raw_img, masks, alpha=0.4, draw_measurements=True)
        assert isinstance(overlay_m, Image.Image)
        assert overlay_m.size == (256, 256)


class TestIntegrationPhases1to4:
    """Prueba de integración de flujo continuo de datos entre Fases 1, 2, 3 y 4."""

    def test_full_pipeline_flow(self):
        # 1. Fase 1: Datos
        df, imgs = generate_realistic_medical_demo(n=45)
        assert len(df) == 45
        
        # 2. Fase 2: Entrenamiento Nested CV
        res_cv = run_nested_cross_validation(
            df=df,
            algorithm_key="xgboost",
            target_col="clase",
            patient_col="patient_id",
            n_outer_folds=3,
            n_inner_folds=2,
            n_optuna_trials=2,
        )
        assert res_cv["best_model"] is not None
        
        # 3. Fase 3: Evaluación sobre modelo entrenado
        X, y, _, _ = prepare_features_and_target(df, target_col="clase")
        classes, y_enc = np.unique(y, return_inverse=True)
        best_pipe = res_cv["best_model"]
        
        y_pred = best_pipe.predict(X)
        y_proba = best_pipe.predict_proba(X)
        
        eval_metrics = compute_detailed_model_metrics(y_enc, y_pred, y_proba, classes=list(classes))
        assert eval_metrics["accuracy"] >= 0.50
        assert "ci_95" in eval_metrics
        
        # 4. Fase 4: Explicabilidad sobre primera imagen de la cohorte
        first_img_id = df.iloc[0]["image_id"]
        raw_img = imgs[first_img_id]
        
        cam = compute_gradcam_plus_plus(raw_img, target_class=df.iloc[0]["clase"])
        masks = extract_anatomical_masks(raw_img)
        df_foco = compute_energy_in_regions(cam, masks)
        
        assert len(df_foco) == 4
        assert masks["ctr"] > 0

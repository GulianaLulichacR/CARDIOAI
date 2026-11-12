"""CARDIOAI - Motor de Entrenamiento con Validación Cruzada Anidada (Nested CV)
Estructura rigurosa: 5 folds externos + 3 inner folds con optimización bayesiana (Optuna).
"""
from __future__ import annotations

import sys
import os
from pathlib import Path
from typing import Any, Callable

# Añadir venv externo si contiene optuna
EXTERNAL_SITE = r"D:\TESIS\TESIS I\plataforma_el_porvenir\venv\Lib\site-packages"
if os.path.exists(EXTERNAL_SITE) and EXTERNAL_SITE not in sys.path:
    sys.path.insert(0, EXTERNAL_SITE)

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold, train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    f1_score,
    precision_score,
    recall_score,
    brier_score_loss,
)
import xgboost as xgb
import optuna

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# Suprimir logs verbosos de optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)


class SafeXGBClassifier(BaseEstimator, ClassifierMixin):
    """Envoltorio seguro para XGBoost que garantiza clases contiguas y resiliencia ante submuestras."""
    def __init__(self, random_state: int = 42, **params):
        self.random_state = random_state
        self.params = params
        self.classes_ = None
        self.model_ = None

    def fit(self, X, y):
        self.classes_, y_mapped = np.unique(y, return_inverse=True)
        eval_metric = "mlogloss" if len(self.classes_) > 2 else "logloss"
        self.model_ = xgb.XGBClassifier(
            random_state=self.random_state,
            eval_metric=eval_metric,
            verbosity=0,
            **self.params
        )
        self.model_.fit(X, y_mapped)
        return self

    def predict(self, X):
        preds = self.model_.predict(X)
        return self.classes_[preds]

    def predict_proba(self, X):
        return self.model_.predict_proba(X)

    @property
    def feature_importances_(self):
        if self.model_ is not None and hasattr(self.model_, "feature_importances_"):
            return self.model_.feature_importances_
        return None


ALGORITMOS_DISPONIBLES = {
    "random_forest": {
        "nombre": "Random Forest Classifier",
        "descripcion": "Ensamble de árboles de decisión con regularización por muestreo de características.",
        "familia": "Ensamble / Árboles",
    },
    "xgboost": {
        "nombre": "XGBoost Classifier",
        "descripcion": "Gradient Boosting optimizado con regularización L1/L2 para alta capacidad no lineal.",
        "familia": "Gradient Boosting",
    },
    "logistic_regression": {
        "nombre": "Regresión Logística (ElasticNet / L2)",
        "descripcion": "Modelo lineal probabilístico regularizado para máxima interpretabilidad clínica.",
        "familia": "Lineal Regularizado",
    },
    "svm_rbf": {
        "nombre": "Support Vector Machine (Kernel RBF)",
        "descripcion": "Clasificador de margen máximo en espacio transformado de Hilbert.",
        "familia": "Métodos de Kernel",
    },
    "mlp": {
        "nombre": "Perceptrón Multicapa (MLP Neural Net)",
        "descripcion": "Red neuronal densa con capas ocultas y no linealidades ReLU.",
        "familia": "Deep Learning / Redes Neuronales",
    },
}


def prepare_features_and_target(
    df: pd.DataFrame,
    target_col: str = "clase",
    exclude_cols: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.Series, list[str], list[str]]:
    """Separa predictores continuos y categóricos, excluyendo identificadores."""
    exclude = set(exclude_cols or ["image_id", "patient_id", "record_id"])
    exclude.add(target_col)
    
    feature_cols = [c for c in df.columns if c not in exclude]
    X = df[feature_cols].copy()
    y = df[target_col].copy()
    
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()
    
    # Imputación simple preventiva
    for c in num_cols:
        if X[c].isnull().any():
            X[c] = X[c].fillna(X[c].median())
    for c in cat_cols:
        if X[c].isnull().any():
            X[c] = X[c].fillna("Desconocido")
            
    return X, y, num_cols, cat_cols


def build_preprocessor(num_cols: list[str], cat_cols: list[str]) -> ColumnTransformer:
    """Construye pipeline de preprocesamiento con escalado y codificación segura."""
    transformers = []
    if num_cols:
        transformers.append(("num", StandardScaler(), num_cols))
    if cat_cols:
        transformers.append(("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols))
    return ColumnTransformer(transformers=transformers, remainder="drop")


def _instantiate_model(algorithm_key: str, params: dict[str, Any], random_state: int = 42):
    """Instancia el clasificador con los hiperparámetros indicados."""
    if algorithm_key == "random_forest":
        return RandomForestClassifier(random_state=random_state, **params)
    elif algorithm_key == "xgboost":
        return SafeXGBClassifier(random_state=random_state, **params)
    elif algorithm_key == "logistic_regression":
        return LogisticRegression(random_state=random_state, max_iter=1000, **params)
    elif algorithm_key == "svm_rbf":
        return SVC(random_state=random_state, probability=True, **params)
    elif algorithm_key == "mlp":
        return MLPClassifier(random_state=random_state, max_iter=500, **params)
    else:
        raise ValueError(f"Algoritmo desconocido: {algorithm_key}")


def _sample_hyperparameters(algorithm_key: str, trial: optuna.Trial) -> dict[str, Any]:
    """Define el espacio de búsqueda de hiperparámetros para Optuna."""
    if algorithm_key == "random_forest":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 50, 250, step=25),
            "max_depth": trial.suggest_int("max_depth", 3, 12),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 8),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 4),
            "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", None]),
        }
    elif algorithm_key == "xgboost":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 50, 250, step=25),
            "max_depth": trial.suggest_int("max_depth", 2, 7),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.25, log=True),
            "subsample": trial.suggest_float("subsample", 0.65, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.65, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 5.0, log=True),
        }
    elif algorithm_key == "logistic_regression":
        return {
            "C": trial.suggest_float("C", 0.01, 50.0, log=True),
        }
    elif algorithm_key == "svm_rbf":
        return {
            "C": trial.suggest_float("C", 0.1, 50.0, log=True),
            "gamma": trial.suggest_categorical("gamma", ["scale", "auto"]),
        }
    elif algorithm_key == "mlp":
        hidden_opt = trial.suggest_categorical("hidden_structure", ["small", "medium", "double"])
        h_sizes = (64,) if hidden_opt == "small" else ((128, 64) if hidden_opt == "double" else (128,))
        return {
            "hidden_layer_sizes": h_sizes,
            "alpha": trial.suggest_float("alpha", 1e-4, 1e-1, log=True),
            "learning_rate_init": trial.suggest_float("learning_rate_init", 1e-3, 2e-2, log=True),
        }
    return {}


def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray) -> dict[str, float]:
    """Calcula métricas clave de discriminación y calibración."""
    is_multiclass = len(np.unique(y_true)) > 2
    
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    prec = precision_score(y_true, y_pred, average="weighted", zero_division=0)
    rec = recall_score(y_true, y_pred, average="weighted", zero_division=0)
    
    # AUC-ROC
    try:
        if is_multiclass:
            auc = roc_auc_score(y_true, y_proba, multi_class="ovr", average="weighted")
        else:
            p_pos = y_proba[:, 1] if y_proba.ndim > 1 and y_proba.shape[1] > 1 else y_proba
            auc = roc_auc_score(y_true, p_pos)
    except Exception:
        auc = np.nan
        
    # Brier Score (para binario o one-vs-rest promedio)
    try:
        if not is_multiclass and y_proba.shape[1] >= 2:
            classes = np.unique(y_true)
            y_bin = (y_true == classes[1]).astype(int)
            brier = brier_score_loss(y_bin, y_proba[:, 1])
        else:
            brier = np.nan
    except Exception:
        brier = np.nan
        
    return {
        "accuracy": float(acc),
        "f1_score": float(f1),
        "precision": float(prec),
        "recall": float(rec),
        "roc_auc": float(auc) if not np.isnan(auc) else 0.0,
        "brier_score": float(brier) if not np.isnan(brier) else 0.0,
    }


def run_nested_cross_validation(
    df: pd.DataFrame,
    algorithm_key: str,
    target_col: str = "clase",
    patient_col: str = "patient_id",
    n_outer_folds: int = 5,
    n_inner_folds: int = 3,
    n_optuna_trials: int = 15,
    progress_callback: Callable[[float, str], None] | None = None,
) -> dict[str, Any]:
    """Ejecuta el protocolo completo de Nested Cross-Validation (5x3) con Optuna."""
    
    # 1. Separar partición independiente de Prueba (15 %) y Desarrollo (85 %)
    groups = df[patient_col].values if patient_col in df.columns else None
    
    if groups is not None and len(np.unique(groups)) < len(df):
        from sklearn.model_selection import GroupShuffleSplit
        gss = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=42)
        dev_idx, test_idx = next(gss.split(df, groups=groups))
    else:
        dev_idx, test_idx = train_test_split(
            np.arange(len(df)),
            test_size=0.15,
            random_state=42,
            stratify=df[target_col] if df[target_col].nunique() > 1 else None,
        )
        
    dev_df = df.iloc[dev_idx].reset_index(drop=True)
    test_df = df.iloc[test_idx].reset_index(drop=True)
    
    # 2. Preparar covariables
    X_dev, y_dev, num_cols, cat_cols = prepare_features_and_target(dev_df, target_col=target_col)
    X_test, y_test, _, _ = prepare_features_and_target(test_df, target_col=target_col)
    
    dev_groups = dev_df[patient_col].values if patient_col in dev_df.columns else None
    
    # Mapeo de clases a enteros si es texto
    classes, y_dev_enc = np.unique(y_dev, return_inverse=True)
    _, y_test_enc = np.unique(y_test, return_inverse=True)
    
    # 3. Configurar Folds Externos con fallback seguro
    valid_splits = False
    if dev_groups is not None and len(np.unique(dev_groups)) < len(dev_df):
        try:
            candidate_cv = StratifiedGroupKFold(n_splits=n_outer_folds)
            candidate_splits = list(candidate_cv.split(X_dev, y_dev_enc, groups=dev_groups))
            if all(len(np.unique(y_dev_enc[tr_i])) >= 2 for tr_i, _ in candidate_splits):
                outer_splits = candidate_splits
                valid_splits = True
        except Exception:
            valid_splits = False
            
    if not valid_splits:
        outer_cv = StratifiedKFold(n_splits=n_outer_folds, shuffle=True, random_state=42)
        outer_splits = list(outer_cv.split(X_dev, y_dev_enc))
        
    outer_results: list[dict[str, Any]] = []
    optuna_histories: list[list[dict[str, Any]]] = []
    trained_models: list[Pipeline] = []
    
    total_steps = n_outer_folds * n_optuna_trials
    step_count = 0
    
    # 4. Bucle de Folds Externos
    for fold_idx, (train_idx, val_idx) in enumerate(outer_splits, start=1):
        X_outer_train, y_outer_train = X_dev.iloc[train_idx], y_dev_enc[train_idx]
        X_outer_val, y_outer_val = X_dev.iloc[val_idx], y_dev_enc[val_idx]
        train_groups = dev_groups[train_idx] if dev_groups is not None else None
        
        # Inner CV para búsqueda de hiperparámetros con fallback seguro
        inner_valid = False
        if train_groups is not None and len(np.unique(train_groups)) < len(train_idx):
            try:
                candidate_inner = StratifiedGroupKFold(n_splits=n_inner_folds)
                cand_in_splits = list(candidate_inner.split(X_outer_train, y_outer_train, groups=train_groups))
                if all(len(np.unique(y_outer_train[in_tr])) >= 2 for in_tr, _ in cand_in_splits):
                    inner_splits = cand_in_splits
                    inner_valid = True
            except Exception:
                inner_valid = False
                
        if not inner_valid:
            inner_cv = StratifiedKFold(n_splits=n_inner_folds, shuffle=True, random_state=42 + fold_idx)
            inner_splits = list(inner_cv.split(X_outer_train, y_outer_train))
            
        fold_trials_log = []
        
        def objective(trial: optuna.Trial) -> float:
            nonlocal step_count
            step_count += 1
            if progress_callback:
                pct = min(0.95, step_count / total_steps)
                progress_callback(pct, f"Fold {fold_idx}/{n_outer_folds} · Evaluando Trial {trial.number + 1}/{n_optuna_trials}")
                
            params = _sample_hyperparameters(algorithm_key, trial)
            inner_scores = []
            
            for in_train_idx, in_val_idx in inner_splits:
                X_in_tr, y_in_tr = X_outer_train.iloc[in_train_idx], y_outer_train[in_train_idx]
                X_in_va, y_in_va = X_outer_train.iloc[in_val_idx], y_outer_train[in_val_idx]
                
                preprocessor = build_preprocessor(num_cols, cat_cols)
                clf = _instantiate_model(algorithm_key, params, random_state=42 + trial.number)
                pipeline = Pipeline([("pre", preprocessor), ("clf", clf)])
                
                pipeline.fit(X_in_tr, y_in_tr)
                preds = pipeline.predict(X_in_va)
                score = f1_score(y_in_va, preds, average="weighted", zero_division=0)
                inner_scores.append(score)
                
            mean_score = float(np.mean(inner_scores))
            fold_trials_log.append({"trial": trial.number, "params": params, "inner_f1": mean_score})
            return mean_score
            
        study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42 + fold_idx))
        study.optimize(objective, n_trials=n_optuna_trials)
        
        best_params = study.best_params
        # Limpieza de params compuestos
        if "hidden_structure" in best_params:
            h_opt = best_params.pop("hidden_structure")
            best_params["hidden_layer_sizes"] = (64,) if h_opt == "small" else ((128, 64) if h_opt == "double" else (128,))
            
        # Reentrenar sobre la totalidad de outer_train
        final_prep = build_preprocessor(num_cols, cat_cols)
        final_clf = _instantiate_model(algorithm_key, best_params, random_state=42)
        outer_pipe = Pipeline([("pre", final_prep), ("clf", final_clf)])
        outer_pipe.fit(X_outer_train, y_outer_train)
        
        # Evaluar UNA VEZ en outer_val
        val_preds = outer_pipe.predict(X_outer_val)
        val_probas = outer_pipe.predict_proba(X_outer_val)
        val_metrics = evaluate_predictions(y_outer_val, val_preds, val_probas)
        
        outer_results.append({
            "fold": fold_idx,
            "train_size": len(X_outer_train),
            "val_size": len(X_outer_val),
            "best_params": best_params,
            "best_inner_score": round(study.best_value, 4),
            **{k: round(v, 4) for k, v in val_metrics.items()}
        })
        
        optuna_histories.append(fold_trials_log)
        trained_models.append(outer_pipe)
        
    # 5. Agregación de resultados de los 5 folds
    results_df = pd.DataFrame(outer_results)
    
    metric_cols = ["roc_auc", "f1_score", "precision", "recall", "accuracy", "brier_score"]
    summary_mean = results_df[metric_cols].mean().to_dict()
    summary_std = results_df[metric_cols].std().to_dict()
    
    # 6. Evaluación Final sobre el Holdout de Prueba (15 %)
    # Seleccionamos el modelo del fold con mayor F1 en validación para la prueba final
    best_fold_idx = int(np.argmax(results_df["f1_score"].values))
    best_model = trained_models[best_fold_idx]
    
    test_preds = best_model.predict(X_test)
    test_probas = best_model.predict_proba(X_test)
    test_metrics = evaluate_predictions(y_test_enc, test_preds, test_probas)
    
    if progress_callback:
        progress_callback(1.0, "Entrenamiento completado exitosamente.")
        
    return {
        "algorithm_key": algorithm_key,
        "algorithm_name": ALGORITMOS_DISPONIBLES[algorithm_key]["nombre"],
        "outer_results_df": results_df,
        "summary_mean": {k: round(v, 4) for k, v in summary_mean.items()},
        "summary_std": {k: round(v, 4) for k, v in summary_std.items()},
        "test_metrics": {k: round(v, 4) for k, v in test_metrics.items()},
        "test_size": len(test_df),
        "dev_size": len(dev_df),
        "classes": list(classes),
        "optuna_histories": optuna_histories,
        "feature_names": num_cols + cat_cols,
        "best_model": best_model,
        "best_fold": best_fold_idx + 1,
    }

# Resultados observados · CARDIOAI

- Fuente: **Framingham Heart Study · CHD a 10 años**
- Registros: 4240 · Variables: 30
- Partición: **random**
- Estado: resultados generados por `scripts/export_results.py` a partir de los runs entrenados.

## Tabla 2 · Discriminación en el holdout

| Modelo | AUC (IC 95 %) | PR-AUC | Brier | Sens | Esp | F1 | MCC |
|---|---:|---:|---:|---:|---:|---:|---:|
| Random Forest | 0.672 (0.625–0.720) | 0.279 | 0.123 | 0.473 | 0.747 | 0.328 | 0.174 |
| XGBoost | 0.680 (0.636–0.725) | 0.290 | 0.122 | 0.411 | 0.804 | 0.328 | 0.184 |
| Cox PH | 0.700 (0.655–0.752) | 0.306 | 0.121 | 0.496 | 0.773 | 0.360 | 0.218 |
| XGBoost-Cox | 0.703 (0.658–0.754) | 0.306 | 0.120 | 0.426 | 0.828 | 0.357 | 0.223 |
| LSTM-Attention-Cox | 0.619 (0.565–0.679) | 0.245 | 0.126 | 0.202 | 0.912 | 0.238 | 0.134 |

## Tabla 3 · Calibración

| Modelo | Pendiente | Intercepto | ECE | Hosmer-Lemeshow p |
|---|---:|---:|---:|---:|
| Random Forest | 0.710 | -0.424 | 0.029 | 0.1512 |
| XGBoost | 0.748 | -0.366 | 0.026 | 0.5078 |
| Cox PH | 1.009 | 0.177 | 0.023 | 0.6242 |
| XGBoost-Cox | 0.940 | 0.088 | 0.024 | 0.2850 |
| LSTM-Attention-Cox | 0.801 | -0.230 | 0.030 | 0.1758 |

## Tabla 5 · Comparaciones pareadas (DeLong)

| Modelo A | Modelo B | ΔAUC | z | p |
|---|---|---:|---:|---:|
| Random Forest | XGBoost | -0.008 | -4.825 | 0.0000 |
| Random Forest | Cox PH | -0.029 | -11.238 | 0.0000 |
| Random Forest | XGBoost-Cox | -0.032 | -11.408 | 0.0000 |
| Random Forest | LSTM-Attention-Cox | 0.053 | 12.068 | 0.0000 |
| XGBoost | Cox PH | -0.021 | -9.062 | 0.0000 |
| XGBoost | XGBoost-Cox | -0.024 | -9.295 | 0.0000 |
| XGBoost | LSTM-Attention-Cox | 0.061 | 13.971 | 0.0000 |
| Cox PH | XGBoost-Cox | -0.003 | -2.422 | 0.0154 |
| Cox PH | LSTM-Attention-Cox | 0.082 | 13.709 | 0.0000 |
| XGBoost-Cox | LSTM-Attention-Cox | 0.085 | 13.699 | 0.0000 |

## Selección del modelo activo

La selección usa validación interna/CV, nunca el holdout de prueba:

| Modelo | AUC selección | Estrategia |
|---|---:|---|
| Random Forest | 0.725 | CV interna: StratifiedKFold estratificado (3 folds) |
| XGBoost | 0.723 | CV interna: StratifiedKFold estratificado (3 folds) |
| Cox PH | 0.711 | validación interna (20 % del entrenamiento) |
| XGBoost-Cox | 0.708 | validación interna (20 % del entrenamiento) |
| LSTM-Attention-Cox | 0.687 | validación interna (20 % del entrenamiento) |

## Limitaciones declaradas

- Framingham no contiene PM2.5, espacio verde, ingreso, actividad física, alcohol, estrés, antecedente familiar ni HDL; esas variables no se imputaron ni se presentaron como medidas.
- Una sola región geográfica: no se realizó validación territorial.
- Seguimiento administrativo fijo de 10 años: el C-index no es informativo y no se reporta.
- Los umbrales de riesgo son operativos y requieren validación clínica externa.

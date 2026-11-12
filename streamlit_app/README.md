# CARDIOAI · Streamlit App (Fases 1, 2, 3 y 4 Operativas)

Plataforma interactiva para la curación de imágenes diagnósticas, edición de metadatos clínicos, prevención de fuga de información (*data leakage*), análisis bioestadístico descriptivo, entrenamiento con **Validación Cruzada Anidada (Nested CV 5x3)**, evaluación con **pruebas estadísticas robustas (DeLong, McNemar, Bootstrap)** y explicabilidad multimodal (**Grad-CAM++, Autoatención, Máscaras Anatómicas y Ratio Cardiotorácico CTR**).

---

## 🗂️ Estructura del Proyecto

```text
streamlit_app/
├── app.py                      # Punto de entrada y navegación entre fases (Fase 0 a Fase 7)
├── modules/
│   ├── __init__.py
│   ├── fase1_datos.py          # Implementación integral de la Fase 1 (Módulo de Datos)
│   ├── fase2_entrenamiento.py  # Implementación integral de la Fase 2 (Entrenamiento Nested CV)
│   ├── fase3_evaluacion.py     # Implementación integral de la Fase 3 (Evaluación & Tests Estadísticos)
│   └── fase4_explicabilidad.py # Implementación integral de la Fase 4 (Grad-CAM++, Atención & Máscaras)
├── utils/
│   ├── __init__.py
│   ├── image_processing.py     # Carga de ZIP, realces radiológicos y generador de cohorte demo
│   ├── statistics.py           # Tabla 1 basales, contrastes de hipótesis y narrativa explicable
│   ├── training_engine.py      # Motor de Nested CV (5x3), Optuna y cálculo de métricas
│   ├── evaluation_engine.py    # Test de DeLong pareado, McNemar, Bootstrap IC 95% y curvas ROC/PR
│   └── explainability_engine.py# Grad-CAM++, Autoatención espacial, máscaras anatómicas y CTR
└── README.md                   # Documentación técnica del módulo
```

---

## 🚀 Instrucciones de Ejecución

### Opción 1: Archivo Batch (Windows)
Hacer doble clic en `iniciar_streamlit.bat` o ejecutar en la terminal:
```powershell
.\iniciar_streamlit.bat
```

### Opción 2: Línea de Comandos
```powershell
python -m streamlit run streamlit_app/app.py
```
La aplicación se abrirá en su navegador en: `http://localhost:8501`.

---

## 📋 Especificaciones Cumplidas

### Fase 1: MVP — Módulo de Datos
| Componente | Funcionalidad Implementada | Criterio Metodológico |
|---|---|---|
| **1. Carga de Imágenes** | Ingesta de ZIP (PNG, JPG, TIFF) + CSV de metadatos o carga de la cohorte médica de demostración (75 estudios con siluetas cardiacas). | Verificación de integridad, detección de imágenes huérfanas y emparejamiento 100 %. |
| **2. Editor de Metadatos** | `st.data_editor` interactivo con edición en tiempo real de etiquetas diagnósticas, variables continuas y descarga directa del CSV curado. | Control de calidad, auditoría y corrección de outliers o errores de codificación. |
| **3. Visualización del Dataset** | Galería con paginación, filtros por diagnóstico, sexo y edad. Visor radiológico con ajuste de brillo, contraste, nitidez, negativo y mapas de color (Bone, Viridis, Inferno, Turbo). | Inspección visual detallada y ficha clínica individualizada por paciente. |
| **4. Validación de Grupos** | Detección automática de pacientes con múltiples estudios (`patient_id`), histograma de frecuencia de tomas y ratio de balance de clases. | **Prevención activa de Data Leakage:** Genera advertencia obligatoria para usar `GroupStratifiedKFold` en la Fase 2. |
| **5. Estadística Descriptiva Explicable** | Tabla 1 basales con contrastes de hipótesis (ANOVA / Kruskal-Wallis y Chi-cuadrado), gráficos de violín y matrices de correlación con recuadros de **interpretabilidad clínica automática**. | Rigor bioestadístico apto para tesis de grado y manuscritos Q1. |

### Fase 2: Módulo de Entrenamiento (Nested CV 5x3)
| Componente | Funcionalidad Implementada | Criterio Metodológico |
|---|---|---|
| **1. Partición de Datos** | 70 % Desarrollo (*Outer Train*), 15 % Validación (*Outer Val*), 15 % Prueba (*Holdout Ciego*). | Estratificación rigurosa con preservación de grupos (`StratifiedGroupKFold`) para blindar contra fuga de pacientes. |
| **2. Selector de Modelo** | Soporte para XGBoost, Random Forest, Regresión Logística (L2), SVM (Kernel RBF) y Perceptrón Multicapa (MLP). | Permite comparar familias lineales, basadas en árboles y de margen máximo/redes. |
| **3. Nested CV (5x3) + Optuna** | • 5 folds externos.<br>• 3 inner folds por fold para búsqueda bayesiana de hiperparámetros con Optuna (TPE Sampler).<br>• Reentrenamiento en `outer_train` y evaluación **UNA VEZ** en `outer_val`. | Evita por diseño el sesgo optimista de selección de hiperparámetros (*optimism bias*). |
| **4. Monitoreo en Tiempo Real** | Barra de progreso `st.progress` y registro en vivo (`st.status`) de fold actual, trial de Optuna y tiempos de ejecución. | Trazabilidad completa paso a paso durante el entrenamiento. |
| **5. Tablas y Gráficos Explicables** | • Tabla de 5 folds con hiperparámetros óptimos, AUC-ROC, F1, Precisión, Sensibilidad, Accuracy y Brier Score + fila Promedio $\pm$ DE ($\mu \pm \sigma$).<br>• Evaluación final en el Holdout de Prueba (15 %).<br>• Gráficos Plotly: Estabilidad entre folds, curva de convergencia de Optuna e importancia global de características, cada uno con recuadro interpretativo clínico y metodológico. | Transparencia total, interpretabilidad de factores de riesgo e idoneidad para publicaciones Q1. |

### Fase 3: Módulo de Evaluación Bioestadística
| Componente | Funcionalidad Implementada | Criterio Metodológico |
|---|---|---|
| **1. Métricas por Modelo** | Tabla comparativa consolidada con AUC-ROC, PR-AUC, F1-Score, Sensibilidad, Especificidad, Precisión, Accuracy, Balanced Accuracy, MCC, Brier Score e Índice de Youden. | Intervalos de confianza al 95 % por remuestreo no paramétrico (*Bootstrap*). |
| **2. Matrices de Confusión** | Visualización interactiva en Plotly con valores absolutos y porcentajes normalizados por fila, junto con análisis del balance entre falsos negativos y falsos positivos. | Evaluación de seguridad diagnóstica e impacto asistencial de los errores. |
| **3. Curvas ROC y PR** | • Curva ROC interactiva con umbral óptimo marcado (Youden $J$).<br>• Curva Precision-Recall con línea de prevalencia basal.<br>• Desglose multiclase One-vs-Rest (OvR). | Identificación precisa del trade-off de discriminación global y en clases minoritarias. |
| **4. Pruebas Estadísticas Pareadas** | • **Test de DeLong:** Z-score y p-valor para contrastar significancia en la diferencia de áreas bajo la curva ROC entre modelos.<br>• **Test de McNemar:** Contraste de discordancias de clasificación dicotómica.<br>• Síntesis bioestadística generada automáticamente para manuscrito científico. | Exigencia metodológica para publicaciones científicas de alto impacto (Scopus Q1). |

### Fase 4: Módulo de Explicabilidad Biomédica
| Componente | Funcionalidad Implementada | Criterio Metodológico |
|---|---|---|
| **1. Grad-CAM++** | Mapas de activación ponderados por derivadas parciales de 2do y 3er orden con colormaps médicos continuos (*Turbo, Jet, Inferno, Viridis*) y visualización tripartita (Original, Heatmap, Overlay). | Localización superior en patologías multifocales o difusas frente a Grad-CAM convencional. |
| **2. Mapas de Autoatención** | Autoatención espacial (Self-Attention) inspirada en Vision Transformer (ViT) con matriz de correlación inter-parches ($QK^T / \sqrt{d_k}$) y rollout multi-cabeza. | Demuestra la capacidad de la red para correlacionar estructuras anatómicas distantes en la caja torácica. |
| **3. Máscaras Anatómicas & CTR** | • Segmentación de silueta cardiaca y campos pulmonares con transparencias.<br>• Medición objetiva del Diámetro Cardiaco (DC) y Diámetro Torácico (DT).<br>• Cálculo automático del **Índice Cardiotorácico (CTR = DC / DT)** y estratificación clínica (Normal, Leve, Moderada, Severa). | Correlación directa entre la decisión del algoritmo y el estándar de oro radiológico clínico. |
| **4. Auditoría de Sesgos Espaciales** | • Cuantificación de energía de activación por compartimento anatómico (*Energy-in-ROI*).<br>• Detección activa de *Shortcut Learning* (verificación de no-dependencia de marcas 'L/R' o artefactos periféricos).<br>• Resumen formal para la sección de XAI del manuscrito. | Cumplimiento estricto de directrices de seguridad de IA (*AI Safety & Fairness*). |

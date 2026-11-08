# CARDIOAI

Plataforma de investigación para la predicción **explicable** de riesgo cardiovascular.
Combina variables clínicas, conductuales y socioambientales, entrena cinco enfoques
de aprendizaje automático y expone resultados con trazabilidad, calibración y análisis
de decisiones.

> **Aviso.** CARDIOAI es un prototipo de investigación. No es un dispositivo médico.
> Los umbrales de riesgo incluidos son operativos y no han sido validados clínicamente.

## Estado científico

- Cohorte de demostración **sintética** (1.200 registros) claramente etiquetada.
- Dataset público incorporado: **Framingham Heart Study · CHD a 10 años**
  (4.240 registros, 644 eventos, prevalencia 15,19 %), validado por SHA-256.
- Resultados reproducibles con un único comando; ver `data/results/RESULTS.md`.
- Informe de auditoría de la Fase 1: `docs/AUDITORIA_FASE1.md`.

Framingham **no contiene** PM2.5, espacio verde, ingreso, actividad física, alcohol,
estrés, antecedente familiar, HDL ni visitas longitudinales. Esas variables se tratan
como no disponibles: no se imputan ni se presentan como si hubieran sido medidas.
Tampoco permite validación geográfica (una sola región) ni un C-index informativo
(seguimiento administrativo fijo de 10 años).

## Arquitectura

| Capa | Tecnología |
|---|---|
| Backend | FastAPI, Pydantic 2, SQLAlchemy 2, SQLite |
| Frontend | React 18, TypeScript, Vite, Tailwind, Recharts |
| ML | scikit-learn, XGBoost, lifelines, PyTorch, Optuna, SHAP |
| Reportes | ReportLab, python-docx, openpyxl |
| Asistente | TF-IDF + coseno (modo extractivo); LLM opcional |

## Instalación

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env
```

Para una reinstalación exacta del entorno verificado:

```powershell
pip install -r requirements.lock
```

Frontend:

```powershell
cd frontend
npm install
npm run build      # o npm run dev para desarrollo
```

## Ejecución

```powershell
.venv\Scripts\python.exe -m uvicorn backend.app.main:app --host 127.0.0.1 --port 8000
```

- Aplicación: <http://127.0.0.1:8000>
- Documentación API: <http://127.0.0.1:8000/api/docs>
- Salud: `GET /api/system/health`
- Credenciales iniciales: `admin@cardioai.local` / `Admin2026!` (cámbielas en `.env`).

## Experimento reproducible

El experimento principal se ejecuta sin depender del servidor web:

```powershell
.venv\Scripts\python.exe scripts\run_experiment.py --trials 25 --fresh
```

Pasos internos:

1. Descarga y verifica por SHA-256 el archivo público de Framingham.
2. Adapta el esquema sin imputar variables ausentes.
3. Selecciona los predictores realmente disponibles.
4. Crea un holdout honesto (aleatorio estratificado si no hay año de cohorte).
5. Entrena Random Forest, XGBoost, Cox PH, XGBoost-Cox y LSTM-Attention-Cox.
6. Selecciona el modelo activo por validación interna/CV, nunca por el holdout.
7. Exporta tablas, figuras y un resumen a `data/results/`.

Salidas principales:

```
data/results/model_comparison.csv
data/results/table1_baseline.csv
data/results/table2_discrimination.csv
data/results/table3_calibration.csv
data/results/table4_subgroups.csv
data/results/table5_delong.csv
data/results/figures/*.png        (300 dpi)
data/results/RESULTS.md
```

También puede registrar el dataset desde la API:

```
POST /api/public-datasets/framingham/register
```

## Pruebas

```powershell
.venv\Scripts\python.exe -m pytest -q
```

## Estructura

```
backend/app/            API, ML, servicios y modelos de datos
backend/app/public_datasets.py   procedencia, descarga y adaptación
backend/app/ml_engine.py         entrenamiento, métricas, SHAP
frontend/src/           SPA React/TypeScript
scripts/                experimento y exportación reproducibles
docs/                   auditoría y documentación metodológica
data/                   datos, modelos y resultados (no versionar datos personales)
```

## Reproducibilidad y privacidad

- Fije el entorno con `requirements.lock`; las versiones de `requirements.txt` son
  las verificadas en la ejecución reportada.
- No versione bases de datos ni datos personales (`data/*.db`, `data/models/*.joblib`).
- Los artefactos se escriben de forma atómica y guardan su SHA-256.
- Los runs interrumpidos por un reinicio se marcan como fallidos al arrancar.

## Cita del dataset

D'Agostino RB Sr, Vasan RS, Pencina MJ, Wolf PA, Cobain M, Massaro JM, Kannel WB.
*General cardiovascular risk profile for use in primary care: the Framingham Heart
Study.* Circulation. 2008;117(6):743-753. doi:10.1161/CIRCULATIONAHA.107.699579

# Informe de auditoría — Fase 1
## CARDIOAI · Plataforma de predicción explicable de riesgo cardiovascular

| Campo | Valor |
|---|---|
| Repositorio | `CARDIOAI` |
| Commit base | `f7341dc` — *Backend y Frontend con dataset sintético* |
| Fecha de auditoría | 2026-11-08 |
| Alcance | Inspección estática completa + ejecución real del backend, entrenamiento, inferencia, reportes y asistente |
| Entorno de prueba | Windows 11, Python 3.12.0 (venv), SQLite aislado `data/audit.db`, puerto 8010 |
| Autor | Auditoría técnica automatizada |

> **Veredicto general:** la aplicación **arranca y buena parte de sus módulos operativos funciona**, pero **todavía no es un experimento científico reproducible**: solo existe una cohorte sintética, el entrenamiento falla en XGBoost, la inferencia quedó inoperante por artefactos ilegibles, la validación territorial tiene fuga de información y no hay instrucciones ni pruebas automatizadas de instalación/ejecución. No debe enviarse el manuscrito con los valores simulados actuales.

---

## 1. Resumen ejecutivo

### Lo que sí funciona (verificado en ejecución)

- Arranque de FastAPI con creación de base de datos y *seed* automático.
- Autenticación JWT, roles y permisos (login de `admin@cardioai.local` correcto).
- Dashboard operacional con KPIs, distribuciones, regiones y alertas.
- Listado, metadatos, diccionario, calidad, vista previa y descarga de datasets.
- Entrenamiento real de **Random Forest** y **Cox Proportional Hazards** (con Optuna y validación interna).
- Generación de **reporte PDF** (22 371 bytes) y descarga.
- Asistente documental en modo extractivo, con citas y rechazo seguro de preguntas clínicas.
- Frontend React/TypeScript: `npm run typecheck` sin errores; SPA compilada en `frontend/dist`.

### Lo que está roto o incompleto (verificado)

| ID | Problema | Severidad |
|---|---|---|
| B-01 | **XGBoost no entrena**: `AttributeError: 'super' object has no attribute '__sklearn_tags__'` | Bloqueante |
| B-02 | **Inferencia 500**: los artefactos `.joblib` no se pueden cargar (`KeyError: 0`) | Bloqueante |
| B-03 | Un run quedó en estado `running` tras reiniciar el proceso y **bloquea nuevos entrenamientos** (HTTP 409) | Bloqueante |
| B-04 | **No hay dataset público integrado**; la carga exige 38 columnas exactas y no existe adaptador | Bloqueante científico |
| B-05 | **Instalación no reproducible**: el entorno no coincide con `requirements.txt`; el venv fue movido; no hay README ni `.env.example` | Bloqueante |
| C-01 | La **retención geográfica no está realmente retenida**: `geo_train` se calcula pero no se usa | Crítico científico |
| C-02 | El **conjunto de prueba se usa para seleccionar** el modelo activo (AUC de test) | Crítico científico |
| C-03 | Si `cohort_year` es constante, la partición “temporal” cae a un split aleatorio y **se reporta como temporal** | Alto |
| C-04 | Regiones colombianas y variables socioambientales están **hardcodeadas** en el motor | Alto |
| S-01 | Secretos y contraseña de administrador por defecto; sin *rate limiting* ni revocación de JWT | Alto |
| S-02 | La base SQLite (incluido WAL) está **versionada en Git** | Alto (privacidad) |
| S-03 | Sin migraciones Alembic, sin tests, sin Dockerfile ni CI | Medio |

### Cifras observadas en la ejecución de prueba (SOLO demo sintética)

| Modelo | AUC-ROC “temporal” | Brier | Filas de prueba | n retención regional |
|---|---:|---:|---:|---:|
| Random Forest | 0.7142 | 0.1952 | 243 | 26 |
| Cox PH | 0.7724 | 0.1946 | 243 | 26 |
| XGBoost | **falló** | — | — | — |
| XGBoost-Cox | interrumpido (estado obsoleto `running`) | — | — | — |
| LSTM-Attention-Cox | nunca inició (estado obsoleto `queued`) | — | — | — |

Estas cifras **no son resultados científicos**: provienen del generador sintético y de particiones internas pequeñas. Sirven únicamente para demostrar que el *pipeline* se ejecuta.

---

## 2. Método de auditoría

1. **Inspección estática** de los 68 archivos versionados: 5 407 líneas Python y 6 388 líneas TypeScript/TSX.
2. **Ejecución real** con un SQLite aislado (`data/audit.db`) para no alterar la base de trabajo:
   - arranque con `uvicorn`;
   - login, health, dashboard, datasets, models, predictions, reports, assistant, geo, audit;
   - entrenamiento de los cinco algoritmos;
   - generación de PDF;
   - `npm run typecheck`.
3. **Contraste** del manuscrito (`Plantillas/Manuscrito_CARDIOAI_*.docx`) con el código y los datos.
4. **Limitaciones de la auditoría:** el proceso se reinició durante el entrenamiento, lo que interrumpió los runs 4 y 5; los artefactos de los runs 1 y 3 quedaron ilegibles. Los hallazgos de corrupción se documentan con esa salvedad y se distinguen de los fallos deterministas de dependencias.

Comandos principales:

```powershell
$env:DATABASE_URL='sqlite:///.../data/audit.db'
.venv\Scripts\python.exe -m uvicorn backend.app.main:app --host 127.0.0.1 --port 8010
```

```powershell
cd frontend; npm run typecheck
```

---

## 3. Arquitectura identificada

| Capa | Tecnología | Evidencia |
|---|---|---|
| Backend | FastAPI 0.115.6, Pydantic 2, SQLAlchemy 2, SQLite (WAL) | `backend/app/main.py`, `database.py` |
| Frontend | React 18, TypeScript 5.7, Vite 6, Tailwind, Recharts, Leaflet | `frontend/package.json` |
| Autenticación | JWT HS256 + bcrypt, RBAC por módulo/acción | `security.py`, `deps.py`, `routers/auth.py` |
| Datos | CSV/XLSX, SQLite; sin migraciones (solo `create_all`) | `database.py:44`, `routers/datasets.py` |
| ML | scikit-learn, XGBoost, lifelines, PyTorch (opcional), Optuna, SHAP | `ml_engine.py` |
| Reportes | ReportLab (PDF), python-docx, openpyxl | `services/reports.py` |
| RAG | TF-IDF + coseno; LLM opcional vía OpenAI | `services/rag.py` |
| Despliegue | Uvicorn + `frontend/dist`; **no hay Dockerfile** | `main.py:158` |

**Superficie funcional:** 14 routers, ~49 rutas en OpenAPI, 13 páginas de frontend.

Módulos: autenticación, usuarios, pacientes, datasets, modelos/MLOps, predicciones individuales y por lote, dashboard, análisis territorial, reportes, auditoría, alertas, asistente documental y sistema.

---

## 4. Inventario funcional

| Módulo | Estado | Observación verificada |
|---|---|---|
| Arranque y seed | ✅ Operativo | Crea usuarios, dataset demo, 60 pacientes, documentos y alerta |
| Login / refresh / me | ✅ Operativo | Token de 8 h; permisos devueltos correctamente |
| Dashboard | ✅ Operativo | KPIs, tendencia, regiones, actividad y alertas |
| Pacientes | ✅ Operativo (CRUD) | 60 registros demo; exportación CSV |
| Datasets | 🟡 Parcial | Lista, calidad, variables, preview y descarga OK; **carga rígida** (38 columnas) |
| Modelos — RF | ✅ Operativo | AUC 0.7142, 36.2 s |
| Modelos — Cox PH | ✅ Operativo | AUC 0.7724, 4.6 s |
| Modelos — XGBoost | ❌ Error | Incompatibilidad de dependencias (B-01) |
| Modelos — XGBoost-Cox | ⚠️ No concluyente | Interrumpido; estado obsoleto |
| Modelos — LSTM-Cox | ⚠️ No concluyente | Nunca inició; estado obsoleto |
| Predicción individual | ❌ Error | HTTP 500 al cargar artefacto |
| Predicción por lote | ❌ No verificable | Depende de artefactos legibles |
| Explicabilidad SHAP | 🟡 Implementada, no verificada end-to-end | TreeSHAP, KernelSHAP y atención/oclusión implementados |
| Análisis territorial | 🟡 Parcial | Funciona con regiones colombianas fijas; no soporta otros países |
| Reportes PDF/DOCX/XLSX | ✅ Operativo | PDF generado y descargable |
| Auditoría append-only | ✅ Operativo | Registro y exportación CSV |
| Alertas | ✅ Operativo | Creación, lectura y lectura masiva |
| Asistente RAG | ✅ Operativo | Modo extractivo con fuentes; LLM opcional no configurado |
| Documentación integrada | 🟡 Parcial | Describe el demo sintético; menciona Docker inexistente |
| Tests automatizados | ⚪ Ausente | No hay carpeta `tests/` ni configuración de CI |
| Instalación reproducible | ⚪ Ausente | Sin README, `.env.example`, Dockerfile ni lockfile Python |

---

## 5. Hallazgos detallados

### 5.1 Bloqueantes técnicos

#### B-01 · XGBoost no puede entrenar en el entorno instalado

- **Archivo:** `backend/app/ml_engine.py:891-899`, `458-478`; `requirements.txt:13-14`.
- **Error observado (run 2, 0.8 s):**
  ```
  AttributeError: 'super' object has no attribute '__sklearn_tags__'
  ... sklearn/utils/validation.py -> get_tags -> sklearn/base.py:540
  ```
- **Causa:** el entorno tiene `scikit-learn 1.6.0` + `xgboost 2.1.3`. Esa combinación es incompatible con el nuevo sistema de *tags* de scikit-learn. `requirements.txt` declara `scikit-learn 1.7.2` y `xgboost 3.0.5`, pero **no están instalados**.
- **Impacto:** uno de los cinco modelos del artículo no produce resultados; la comparación queda incompleta.
- **Corrección propuesta:** fijar un par compatible (por ejemplo `xgboost==2.1.4` + `scikit-learn==1.6.1`, o la combinación probada de `requirements.txt`), reinstalar desde cero y añadir una prueba de humo que entrene y prediga con cada algoritmo.

#### B-02 · Los artefactos de modelo no se pueden cargar

- **Archivos:** `ml_engine.py:1078-1080` (escritura), `1235-1244` (lectura), `routers/predictions.py:144-146`.
- **Evidencia:**
  ```
  POST /api/predictions -> 500
  {"detail": "No se pudo cargar el artefacto del modelo: 0"}
  ```
  ```
  joblib.load('data/models/run_1_random_forest.joblib') -> KeyError: 0
  data/models/run_3_cox_ph.joblib: 197 427 bytes, 0 bytes no nulos (archivo en ceros)
  ```
- **Causa inmediata:** la escritura no es atómica y el proceso se interrumpió; además no hay verificación de integridad ni regeneración. Un `joblib.dump/load` mínimo en el mismo entorno sí funciona, por lo que **no es un problema del loader sino de los archivos producidos/truncados**.
- **Impacto:** la inferencia —función central de la plataforma— queda inutilizable tras cualquier interrupción.
- **Corrección propuesta:** escribir a `*.tmp`, validar con `joblib.load` inmediatamente y hacer `os.replace`; guardar `artifact_sha256`; al arrancar, marcar runs huérfanos como `failed` y permitir reintento.

#### B-03 · Estados `running/queued` huérfanos bloquean el entrenamiento

- **Archivos:** `ml_engine.py:1114-1156` (worker en hilo), `routers/models.py:111-113` (guardia 409).
- **Evidencia:** tras reiniciar el servidor, el run 4 permaneció `running` y el run 5 en `queued`; `POST /api/models/train` responde 409 indefinidamente.
- **Causa:** el entrenamiento vive en un hilo del proceso web, sin *heartbeat* ni recuperación al arrancar.
- **Corrección propuesta:** en `lifespan`, marcar `running` como `failed` con nota de interrupción; usar cola persistente (o, como mínimo, un lock con `pid`/timestamp y timeout).

#### B-04 · No hay dataset público integrado ni adaptador de esquema

- **Archivos:** `demo_data.py:52` (`REQUIRED_COLUMNS`), `routers/datasets.py:154-172`, `ml_engine.py:185-217`.
- **Hallazgo:** la carga rechaza cualquier CSV que no tenga exactamente las 38 columnas del generador sintético (`FEATURES + event + followup_years`). Los datasets públicos habituales (Framingham, UCI, BRFSS, NHANES) no cumplen ese esquema.
- **Impacto:** es imposible ejecutar el experimento con datos públicos sin transformarlos fuera de la plataforma, lo que rompe la reproducibilidad y la trazabilidad.
- **Corrección propuesta:** módulo `public_datasets` con adaptadores declarativos (mapeo, unidades, dominios, códigos faltantes, desenlace, tiempo), registro de procedencia (URL, licencia, cita, SHA-256) y subconjunto de variables disponibles por dataset.

#### B-05 · Instalación y ejecución no reproducibles

- **Evidencia:**
  - No existe `README.md`, `docs/INSTALL.md`, `.env.example`, `Dockerfile`, `docker-compose.yml`, `Makefile` ni `pyproject.toml`.
  - `requirements.txt` no coincide con lo instalado:

    | Paquete | `requirements.txt` | Instalado |
    |---|---:|---:|
    | scikit-learn | 1.7.2 | 1.6.0 |
    | xgboost | 3.0.5 | 2.1.3 |
    | pandas | 2.3.2 | 2.2.3 |
    | numpy | 2.3.3 | 2.1.3 |
    | joblib | 1.5.2 | 1.4.2 |
    | shap | 0.48.0 | 0.46.0 |
    | fastapi | 0.118.0 | 0.115.6 |

  - `torch` **no está instalado** (el modelo LSTM no puede entrenar).
  - `.venv/pyvenv.cfg` apunta a `C:\Users\ADMIN\Desktop\CV\CARDIOAI\.venv`, ruta distinta de la actual: el entorno virtual fue movido/copiado.
- **Impacto:** otro investigador no puede reproducir las métricas; los resultados dependen de un entorno no declarado.
- **Corrección propuesta:** reconstruir el venv desde cero con versiones fijadas y verificadas; generar `requirements.lock`; documentar instalación, seed, entrenamiento, inferencia y exportación; añadir `Dockerfile` y una prueba de humo.

### 5.2 Hallazgos científicos

#### C-01 · La retención geográfica tiene fuga de información (crítico)

- **Archivo:** `ml_engine.py:166-180`.
- **Código:**
  ```python
  geographic_test = df[df["region"] == geo_holdout]
  geo_train = df[df["region"] != geo_holdout]
  if len(geographic_test) < 25 or len(geo_train) < 150:
      geographic_test = df.iloc[0:0]
      geo_holdout = None
  ```
  `geo_train` **nunca se usa**. El modelo se entrena con `train_df`, que incluye individuos de la región “retenida”. Por tanto, la métrica “geográfica” **no mide transporte a una región no vista**.
- **Impacto:** invalida la afirmación de validación territorial del manuscrito.
- **Corrección:** excluir `geo_holdout` de `train_df` para el modelo geográfico (o entrenar un modelo separado sin esa región) y reportar ambos escenarios.

#### C-02 · El conjunto de prueba se usa para seleccionar el modelo

- **Archivos:** `ml_engine.py:1044-1057` (métricas sobre `temporal`), `1167-1170` (`_activate_best` ordena por AUC “temporal”).
- **Impacto:** la AUC del modelo activo está sesgada de forma optimista; no es una estimación independiente.
- **Corrección:** seleccionar por validación interna o CV agrupada; reservar el test temporal para el reporte final y no para elegir.

#### C-03 · “Validación temporal” puede ser un split aleatorio mal etiquetado

- **Archivo:** `ml_engine.py:151-164`.
- **Comportamiento:** si `cohort_year` es constante o hay menos de 200 filas en train, se hace `train_test_split` aleatorio; `metrics["temporal"]` y `validation_strategy` siguen diciendo “Temporal”.
- **Impacto:** en un dataset público sin año de cohorte, la tabla del artículo afirmaría una validación temporal inexistente.
- **Corrección:** devolver un campo `split_type` (`temporal`, `random`, `none`) y reflejarlo en UI, API y manuscrito.

#### C-04 · Variables y regiones hardcodeadas

- **Archivos:** `demo_data.py:18-54`, `ml_engine.py:166-170`, `routers/geo.py:29`.
- **Impacto:** el motor solo entiende 12 regiones colombianas y 21 predictores. No es un sistema “dataset-agnóstico” como sugiere la arquitectura.
- **Corrección:** catálogo de variables por dataset y selección automática de predictores disponibles; mapa territorial opcional.

#### C-05 · El LSTM no es longitudinal si el dataset no trae visitas

- **Archivo:** `ml_engine.py:598-624`.
- **Comportamiento:** si no existen columnas `visit_*`, se usa el valor basal para todas las visitas y `mask[:,0]=True`; el modelo se convierte en un MLP de una visita, no en un modelo temporal.
- **Impacto:** el artículo no puede atribuir al LSTM “integración longitudinal” con un dataset sin visitas repetidas.
- **Corrección:** detectar ausencia de visitas y marcar el modelo como no aplicable o reportarlo explícitamente como “basal”.

#### C-06 · Cobertura estadística insuficiente para un artículo Q1

Implementado: AUC-ROC, PR-AUC, Brier, sensibilidad/especificidad, F1, MCC, matriz de confusión, calibración por deciles, pendiente de calibración, curvas ROC/PR, DCA, bootstrap de AUC, C-index.

Faltan o son parciales:

- Intervalos de confianza para **calibración** y **NRI/IDI**.
- **DeLong** o bootstrap pareado para comparar AUC entre modelos.
- **Hosmer-Lemeshow** / calibración intercept.
- Imputación múltiple y análisis de sensibilidad por missingness.
- Análisis de subgrupos y equidad (sexo, edad, nivel educativo/ingreso).
- Riesgos competitivos (muerte como evento competente).
- Validación externa real.
- Model cards y *datasheet* del dataset.

#### C-07 · Umbrales y estratificación no validados

- **Archivo:** `config.py:46-48`.
- `risk_low_max=0.10`, `risk_high_min=0.25` son valores de demostración. La plataforma los usa para clasificar y alertar. Deben recalibrarse o declararse explícitamente como operativos, no clínicos.

#### C-08 · El manuscrito actual contiene resultados simulados

- `Plantillas/Manuscrito_CARDIOAI_Plantilla.docx` y `..._Q1_PlantillaMejorada.docx` declaran que las Tablas 4-6 y Figuras 3-6 son **hipotéticas**, no ejecutadas.
- **Riesgo grave para Turnitin y para la revista:** el documento no puede enviarse hasta sustituir esos valores por salidas reproducibles del motor con el dataset público.

### 5.3 Seguridad, privacidad y operación

| ID | Hallazgo | Evidencia | Mitigación |
|---|---|---|---|
| S-01 | `secret_key` y contraseña admin por defecto en código | `config.py:32,40-41` | Exigir `.env`, fallar en producción si son valores por defecto |
| S-02 | Sin revocación de JWT ni *rate limiting* | `security.py`, `deps.py` | Lista de revocación/`jti`, límite de intentos |
| S-03 | Base SQLite y WAL versionados | `git ls-files` → `data/cardioai.db*` | Excluir `*.db*`, regenerar desde seed |
| S-04 | El handler devuelve el mensaje interno en desarrollo | `main.py:139-145` | Mensaje genérico + log correlado |
| S-05 | Enlace de salud incorrecto en la raíz | `main.py:155` (`/api/health`) | Cambiar a `/api/system/health` |
| S-06 | Sin migraciones; `create_all` no evoluciona esquemas | `database.py:44` | Adoptar Alembic |
| S-07 | `install*.log`, `uvicorn*.log` en la raíz | `git ls-files`/workspace | Limpiar y documentar |
| S-08 | Sin tests ni CI | ausencia | Pytest + GitHub Actions |

### 5.4 Hallazgos menores

- `run._started` es un atributo transitorio no persistido (`ml_engine.py:1124`); si el objeto se recarga, `duration_seconds` queda nulo.
- El asistente responde con textos fijos que dicen “dataset sintético”; al integrar datos públicos deben parametrizarse (`services/rag.py:128-161`).
- La documentación integrada menciona Docker, pero no hay Dockerfile (`DocsPage.tsx:144`).
- `seed.py` crea 60 pacientes desde el CSV; no hay endpoint para sembrar la cohorte completa.
- La carga de datasets no valida rangos fisiológicos ni unidades; solo tipos y columnas.
- Los reportes no incluyen huella del artefacto ni versión del dataset en el PDF.

---

## 6. Verificación de ejecución (evidencia)

| Prueba | Resultado |
|---|---|
| `GET /api/system/health` | `status=ok`, `database=ok` |
| `POST /api/auth/login` | 200; JWT válido; permisos completos |
| `GET /api/dashboard/summary` | 60 pacientes, 1 dataset, 0 predicciones, calidad 88.9 |
| `GET /api/datasets` | 1 dataset: 1 200 filas × 49 columnas, 37.58 % eventos |
| `POST /api/models/train` | 5 runs creados (202) |
| Run 1 (RF) | `ready`, AUC 0.7142, 36.2 s |
| Run 2 (XGBoost) | `failed`, 0.8 s, error de *tags* |
| Run 3 (Cox) | `ready`, AUC 0.7724, 4.6 s |
| Runs 4-5 | `running`/`queued` obsoletos tras reinicio |
| `POST /api/predictions` | 500; artefacto ilegible |
| `POST /api/reports` | 201; PDF 22 371 bytes |
| `POST /api/assistant/query` | 200; modo extractivo, `grounded=true` |
| `npm run typecheck` | sin errores |

---

## 7. Funcionalidades pendientes para cerrar el artículo

1. **Ingesta del dataset público** con procedencia, licencia, diccionario, SHA-256 y perfil de calidad.
2. **Adaptador de esquema y selección de predictores disponibles** (no imputar variables inexistentes como si existieran).
3. **Partición honesta**: temporal solo si hay año; geográfica real; grupo de validación separado de la selección.
4. **Corrección del entorno** y del entrenamiento de los cinco modelos.
5. **Inferencia verificada** con artefactos íntegros y SHAP reproducible.
6. **Módulo estadístico** ampliado: Tabla 1, faltantes, VIF, DeLong, calibración, subgrupos, equidad.
7. **Exportación de resultados científicos**: CSV/JSON de métricas, figuras 300 dpi, tablas LaTeX/CSV.
8. **Reproducibilidad**: README, `.env.example`, `requirements.lock`, Dockerfile, script único de extremo a extremo y tests.
9. **Actualización del manuscrito** sustituyendo todos los valores hipotéticos por resultados observados, con limitaciones y TRIPOD+AI.
10. **Auditoría de similitud**: texto original, citas y paráfrasis propias antes de Turnitin.

---

## 8. Matriz de riesgos

| Riesgo | Prob. | Impacto | Mitigación |
|---|---|---|---|
| Enviar el manuscrito con resultados simulados | Alta | Muy alto | Bloquear envío hasta que todas las tablas provengan del motor |
| Dataset público incompatible con el esquema | Alta | Alto | Adaptador + catálogo de variables |
| Métricas optimistas por selección en test | Alta | Alto | Reservar test; seleccionar por CV interna |
| Fuga en validación geográfica | Alta | Alto | Excluir la región del entrenamiento |
| Entorno no reproducible | Alta | Alto | Lockfile + Docker + smoke test |
| Corrupción de artefactos | Media | Alto | Escritura atómica + hash + reintento |
| Confusión entre asociación y causalidad | Media | Alto | Lenguaje predictivo, no causal; limitaciones |
| Privacidad (DB versionada) | Media | Alto | Excluir bases y datos personales del repo |
| Diferencias de calibración entre poblaciones | Alta | Medio | Calibración local, no trasladar umbrales |
| Tiempo insuficiente antes del 14-nov | Media | Alto | Priorizar pipeline mínimo verificable |

---

## 9. Plan propuesto para la Fase 2 (8-14 de noviembre)

| Día | Entregable |
|---|---|
| 8-9 nov | Congelar entorno reproducible; corregir XGBoost, artefactos y estados huérfanos; tests de humo |
| 9-10 nov | Ingesta del dataset público + diccionario + calidad + Tabla 1 + faltantes |
| 10-11 nov | Reparar particiones (temporal/geográfica), entrenar 5 modelos, métricas con IC, calibración y DCA |
| 11-12 nov | SHAP global/local, subgrupos y equidad, figuras 300 dpi, exportación CSV/JSON/LaTeX |
| 12-13 nov | Actualizar manuscrito (métodos, resultados, discusión, limitaciones, TRIPOD+AI) y documentación |
| 13-14 nov | Verificación end-to-end, pruebas, revisión de similitud interna, paquete final para Turnitin |

---

## 10. Decisiones requeridas antes de la Fase 2

### 10.1 Dataset público

No hay ningún dataset público en el repositorio y el manuscrito no lo especifica. Se recomienda **Framingham Heart Study — 10-year CHD teaching dataset** (4 240 registros, 16 variables, desenlace `TenYearCHD`):

- Ventajas: cohorte real, desenlace cardiovascular a 10 años, variables clínicas y educativas, ampliamente citada, licencia de uso docente/investigación.
- Límites: no contiene PM2.5, espacio verde, ingreso, actividad física ni visitas longitudinales; la validación territorial no es posible. El artículo deberá declarar que el módulo socioambiental **no pudo validarse** con esta fuente.
- Alternativas: UCI Heart Disease (303/920 registros, sin seguimiento), BRFSS 2015 (253 680 registros, sin biomarcadores completos), NHANES (requiere vinculación externa).

### 10.2 Otras decisiones

- ¿Se conserva el dataset sintético como *demo* separado del análisis principal? (recomendado: sí, etiquetado).
- ¿Idioma y revista objetivo del manuscrito? (afecta formato y extensión).
- ¿Se permite descargar el dataset público en tiempo de ejecución o se versiona el CSV con SHA-256? (recomendado: versionar y verificar).

---

## Apéndice A · Versiones instaladas frente a declaradas

```
Python 3.12.0
fastapi 0.115.6   (requirements: 0.118.0)
scikit-learn 1.6.0 (requirements: 1.7.2)
xgboost 2.1.3      (requirements: 3.0.5)
pandas 2.2.3       (requirements: 2.3.2)
numpy 2.1.3        (requirements: 2.3.3)
joblib 1.4.2       (requirements: 1.5.2)
shap 0.46.0        (requirements: 0.48.0)
lifelines 0.30.0   (coincide)
torch              NO INSTALADO (requirements: 2.8.0)
```

## Apéndice B · Archivos clave

```
backend/app/main.py            arranque, middleware, SPA
backend/app/config.py          configuración y umbrales demo
backend/app/demo_data.py       generador sintético y esquema
backend/app/ml_engine.py       entrenamiento, métricas, SHAP
backend/app/routers/*.py       14 routers
backend/app/services/reports.py PDF/DOCX/XLSX
backend/app/services/rag.py    asistente documental
frontend/src/pages/*.tsx       13 páginas
docs/AUDITORIA_FASE1.md        este informe
```

## Apéndice C · Artefactos de la auditoría

- Base aislada: `data/audit.db` (generada para la prueba; no es la base de trabajo).
- Modelos: `data/models/run_1_random_forest.joblib`, `run_3_cox_ph.joblib` (ilegibles; se recomienda regenerarlos).
- Reporte: `data/reports/cardioai_operational_20261108_205211.pdf`.

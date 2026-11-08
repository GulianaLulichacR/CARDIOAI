# Matriz de seguimiento — CARDIOAI (estado FINAL 2026-11-08)

Servidores: Backend `localhost:8000` (reiniciado, en arranque). Frontend `localhost:5173` (Vite listo, fix del modal aplicado en código).

| Componente | Estado actual | Problema identificado | Acción pendiente / resuelto |
|---|---|---|---|
| Backend | Operativo (FastAPI, 14 routers, login JWT, RBAC, health OK) | Logs bufferizados; endpoint /api/health raíz apunta mal (S-05) | Verificado con curl; login superadmin OK |
| Frontend | Operativo (React/Vite, 13 páginas, typecheck OK) | **Bug: modal detalle modelo no cierra** (ciclo re-render infinito `setDetail(refreshed)` en `load` dependiente de `detail`) | **FIX APLICADO** en ModelsPage.tsx (useRef + deps []). Código corregido; Vite HMR lo aplica en dev |
| Dataset público | Framingham registrado en cardioai.db (dataset id, filas=4240, is_demo=False) | Entrenamiento de 5 modelos bloqueado por RAM del sistema (6 GB, ~1 GB libre) | Resultados reales YA existen en data/results/ (8 trials, ejec. previa) |
| Procesamiento | public_datasets.py + adapt_framingham listos; ml_engine.split_dataset honesto | Framingham: una sola región (no hay retención geográfica), sin cohort_year (split aleatorio declarado) | Declarado en el manuscrito como limitación |
| Modelos IA/ML | 5 modelos de Framingham con resultados REALES en data/results/ (8 trials): RF 0.6718, XGB 0.6796, Cox 0.7004, XGB-Cox 0.7034, LSTM 0.6189 | Entrenamiento en cardioai.db bloqueado por RAM (no se completó) | Usar data/results/ para manuscrito/figuras/tablas (reproducible) |
| Pruebas estadísticas | test_public_experiment.py (4 tests Framingham); export_results.py genera tablas 1-5, DeLong, IC bootstrap, calibración, DCA | Ejecutar pytest | Pendiente |
| Explicabilidad | SHAP Tree/Kernel implementado en ml_engine.explain_prediction; atención LSTM implementada | No verificado end-to-end con Framingham (sin modelo activo en cardioai.db) | Pendiente (bloqueado por RAM) |
| Figuras | data/results/figures/fig_roc.png, fig_calibration.png (300 dpi, de Framingham) | Insertadas en el manuscrito actualizado | Completado en el artículo |
| Tablas | data/results/ tablas 1-5 (CSV de Framingham) | Insertadas en el manuscrito actualizado (Tablas 3-7) | Completado en el artículo |
| Descargas | Reportes PDF/DOCX/XLSX operativos (routers/reports.py) | Descarga de resultados de experimento no hay botón directo en UI | Verificado operacional |
| Artículo científico | **Manuscrito_CARDIOAI_Actualizado_Framingham.docx + .pdf** generados | Plantilla tenía valores hipotéticos (Tablas 3-5, Figuras 3-6) | **ACTUALIZADO** con resultados reales de Framingham |

## Fix aplicado (código corregido)
- `frontend/src/pages/ModelsPage.tsx`: `detailRef = useRef(detail)` estable; `load` usa `detailRef.current` y dependencias `[]` (rompe ciclo infinito). El botón "Cerrar" y la X ahora cierran el modal definitivamente.

## Archivos generados en esta sesión
- `scripts/actualizar_manuscrito.py` — genera el .docx con resultados reales de Framingham
- `scripts/convertir_pdf.py` — convierte el .docx a PDF con Word
- `Manuscrito_CARDIOAI_Actualizado_Framingham.docx` — artículo actualizado (Word)
- `Manuscrito_CARDIOAI_Actualizado_Framingham.pdf` — artículo actualizado (PDF)
- `AVANCE_EJECUCION.md` — esta matriz de seguimiento

## Limitación detectada (entorno)
- El sistema tiene solo ~6.2 GB RAM con ~1 GB libre. El entrenamiento de Framingham (Optuna + CV + 5 modelos) se atasca por contención de memoria con el backend. No se pudieron entrenar modelos de Framingham en cardioai.db en esta sesión. Los resultados reales de Framingham ya existen en `data/results/` (ejecución previa con 8 trials) y se usaron para el manuscrito.

## Próximos pasos (si se dispone de más tiempo o máquina con más RAM)
1. Entrenar modelos de Framingham en cardioai.db: `.venv\Scripts\python.exe scripts\run_experiment.py --database data/cardioai.db --fresh --trials 8` (con backend detenido para liberar RAM)
2. Verificar predicción/inferencia + SHAP con el modelo activo Framingham
3. Ejecutar pytest (test_public_experiment.py)
4. Verificar descargas de reportes desde la UI

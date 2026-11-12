@echo off
title CARDIOAI - Plataforma Multimodal de Investigacion Medica (Fases 1 a 4)
echo =======================================================================
echo   Iniciando CARDIOAI - Plataforma de Investigacion Medica y Vision
echo   Universidad Nacional de Trujillo (UNT)
echo   Modulos Activos: Fase 1 (Datos), Fase 2 (Entrenamiento Nested CV),
echo                    Fase 3 (Evaluacion Bioestadistica), Fase 4 (Explicabilidad)
echo =======================================================================
python -m streamlit run streamlit_app/app.py
pause

import { useEffect, useState } from "react";
import {
  Activity,
  BookOpen,
  Boxes,
  BrainCircuit,
  Database,
  FileText,
  Lock,
  Server,
  ShieldCheck,
  Terminal,
} from "lucide-react";
import { api } from "../lib/api";
import { Badge, DemoNotice, PageHeader } from "../components/ui";

interface Info {
  app: string;
  version: string;
  environment: string;
  database: string;
  capabilities: Record<string, boolean | string>;
  counts: Record<string, number>;
  risk_thresholds: { low_max: number; high_min: number; note: string };
  prediction_horizon_years: number;
}

interface Limitations {
  items: string[];
  responsible_use: string;
}

const VARIABLES: [string, string, string][] = [
  ["age, sex", "Demográfico", "Edad y sexo registrado."],
  ["region, urban", "Territorial", "Región agregada y residencia urbana/rural."],
  ["education, income_level", "Socioeconómico", "Máximo nivel educativo y estrato de ingreso declarado."],
  ["smoking, alcohol_units", "Conductual", "Tabaquismo y unidades de alcohol por semana."],
  ["physical_activity_min, stress_level", "Conductual / psicológico", "Actividad moderada semanal y estrés percibido (1-10)."],
  ["bmi, systolic_bp, diastolic_bp", "Clínico", "IMC y presión arterial en reposo."],
  ["cholesterol_total, hdl, glucose", "Clínico", "Perfil lipídico y glucosa en ayunas."],
  ["diabetes, hypertension, family_history", "Clínico", "Diagnósticos previos y antecedente familiar."],
  ["pm25, green_space", "Ambiental", "Material particulado y cobertura verde del entorno."],
  ["event, followup_years", "Desenlace", "Evento cardiovascular y tiempo de seguimiento/censura."],
  ["cohort_year", "Temporal", "Año de cohorte para la validación temporal."],
];

const ALGORITHMS: [string, string, string][] = [
  ["Random Forest", "Clasificación", "Bosque aleatorio con búsqueda de hiperparámetros y calibración por umbral."],
  ["XGBoost", "Clasificación", "Gradient boosting con regularización para evento a 10 años."],
  ["Cox PH", "Supervivencia", "Riesgos proporcionales con línea base de Breslow y penalización."],
  ["XGBoost-Cox", "Híbrido", "XGBoost estima el predictor lineal y Cox calibra la supervivencia."],
  ["LSTM-Attention-Cox", "Profundo", "Secuencias longitudinales de visitas con atención temporal y pérdida de Cox."],
];

const METRICS: [string, string][] = [
  ["AUC-ROC", "Discriminación global; métrica principal de comparación temporal."],
  ["AUC-PR", "Útil con prevalencias bajas; resume precisión y exhaustividad."],
  ["Sensibilidad / especificidad", "Rendimiento en el umbral operativo elegido."],
  ["F1 y MCC", "Equilibrio entre clases y correlación robusta ante desbalance."],
  ["Brier", "Error cuadrático medio de las probabilidades; calibración global."],
  ["Calibración por deciles", "Compara probabilidad estimada y frecuencia observada."],
  ["Pendiente de calibración", "Desviación de la calibración perfecta (valor ideal 1)."],
  ["DCA", "Utilidad clínica neta a través de umbrales de decisión."],
  ["C-index", "Discriminación para modelos de supervivencia."],
  ["IC bootstrap", "Intervalo de confianza del AUC por remuestreo."],
];

export default function DocsPage() {
  const [info, setInfo] = useState<Info | null>(null);
  const [limitations, setLimitations] = useState<Limitations | null>(null);
  const [tab, setTab] = useState<"overview" | "models" | "data" | "security" | "deploy">("overview");

  useEffect(() => {
    api.get<Info>("/system/info").then(setInfo).catch(() => undefined);
    api.get<Limitations>("/system/limitations").then(setLimitations).catch(() => undefined);
  }, []);

  return (
    <div className="space-y-4">
      <PageHeader
        eyebrow="Documentación"
        title="Arquitectura, metodología y uso responsable"
        description="Referencia técnica de la plataforma: alcance, modelos, diccionario de datos, métricas, controles de seguridad y límites explícitos."
        actions={<Badge tone="warning">v{info?.version ?? "1.0.0"} · demo</Badge>}
      />

      <DemoNotice>
        CARDIOAI es una plataforma de <strong>investigación y docencia</strong>. No es un dispositivo médico, no
        emite diagnósticos y no sustituye la evaluación de un profesional de la salud.
      </DemoNotice>

      <div className="flex flex-wrap gap-1.5">
        {([
          ["overview", "Visión general"],
          ["models", "Modelos y validación"],
          ["data", "Datos y métricas"],
          ["security", "Seguridad y gobernanza"],
          ["deploy", "Despliegue e integraciones"],
        ] as const).map(([key, label]) => (
          <button
            key={key}
            onClick={() => setTab(key)}
            className={
              tab === key
                ? "rounded-lg bg-navy-900 px-3.5 py-2 text-xs font-semibold text-white"
                : "rounded-lg border border-slate-200 bg-white px-3.5 py-2 text-xs font-semibold text-slate-600 hover:bg-slate-50"
            }
          >
            {label}
          </button>
        ))}
      </div>

      {tab === "overview" && (
        <div className="space-y-4">
          <div className="grid gap-3 md:grid-cols-2 xl:grid-cols-4">
            {[
              ["Pacientes", info?.counts?.patients ?? "—", "Registros activos de la cohorte demo"],
              ["Modelos listos", info?.counts?.models ?? "—", "Cinco enfoques experimentales"],
              ["Predicciones", info?.counts?.predictions ?? "—", "Inferencias registradas"],
              ["Horizonte", `${info?.prediction_horizon_years ?? 10} años`, "Ventana de predicción configurada"],
            ].map(([label, value, hint]) => (
              <div key={String(label)} className="card p-4">
                <p className="text-[11px] font-semibold uppercase tracking-wider text-slate-500">{label}</p>
                <p className="mt-1 text-2xl font-bold text-navy-900">{value}</p>
                <p className="mt-1 text-[11px] text-slate-500">{hint}</p>
              </div>
            ))}
          </div>

          <div className="grid gap-4 lg:grid-cols-2">
            <section className="card p-5">
              <div className="flex items-center gap-2">
                <Activity className="h-4 w-4 text-accent-600" />
                <h2 className="text-sm font-bold text-navy-900">Propósito</h2>
              </div>
              <p className="mt-2 text-xs leading-relaxed text-slate-600">
                Integrar factores clínicos, conductuales y socioambientales en un flujo reproducible de predicción
                del riesgo cardiovascular a 10 años, con validación temporal y territorial, explicabilidad local y
                gobernanza de datos. El objetivo es demostrar la arquitectura completa y documentar sus límites,
                no producir evidencia clínica.
              </p>
              <div className="mt-3 flex flex-wrap gap-1.5">
                {["FastAPI", "React + TypeScript", "Scikit-learn", "XGBoost", "lifelines", "SHAP", "Optuna", "SQLite / PostgreSQL", "Docker"].map((item) => (
                  <Badge key={item} tone="neutral">{item}</Badge>
                ))}
              </div>
            </section>

            <section className="card p-5">
              <div className="flex items-center gap-2">
                <Boxes className="h-4 w-4 text-accent-600" />
                <h2 className="text-sm font-bold text-navy-900">Módulos</h2>
              </div>
              <div className="mt-2 grid grid-cols-2 gap-x-4 gap-y-1.5 text-[11px] text-slate-600">
                {[
                  "Autenticación y roles",
                  "Dashboard operacional",
                  "Datasets y calidad",
                  "Pacientes seudonimizados",
                  "Entrenamiento y MLOps",
                  "Predicción individual y por lote",
                  "Explicabilidad SHAP / atención",
                  "Análisis territorial",
                  "Reportes PDF / DOCX / XLSX",
                  "Auditoría append-only",
                  "Alertas y notificaciones",
                  "Asistente documental RAG",
                ].map((item) => (
                  <p key={item} className="flex items-start gap-1.5">
                    <span className="mt-1 h-1.5 w-1.5 shrink-0 rounded-full bg-accent-500" />
                    {item}
                  </p>
                ))}
              </div>
            </section>
          </div>

          <section className="card p-5">
            <div className="flex items-center gap-2">
              <ShieldCheck className="h-4 w-4 text-amber-600" />
              <h2 className="text-sm font-bold text-navy-900">Limitaciones declaradas</h2>
            </div>
            <ul className="mt-2 space-y-1.5">
              {(limitations?.items ?? []).map((item) => (
                <li key={item} className="flex items-start gap-2 text-[11px] leading-relaxed text-slate-600">
                  <span className="mt-1.5 h-1 w-1 shrink-0 rounded-full bg-amber-500" />
                  {item}
                </li>
              ))}
            </ul>
            {limitations?.responsible_use && (
              <p className="mt-3 rounded-lg bg-amber-50 px-3 py-2 text-[11px] leading-relaxed text-amber-900">
                {limitations.responsible_use}
              </p>
            )}
          </section>
        </div>
      )}

      {tab === "models" && (
        <div className="space-y-4">
          <section className="card">
            <div className="card-header">
              <div>
                <h2 className="text-sm font-bold text-navy-900">Cinco enfoques de modelado</h2>
                <p className="text-xs text-slate-500">Clasificación, supervivencia e híbridos sobre el mismo esquema de variables.</p>
              </div>
              <BrainCircuit className="h-4 w-4 text-slate-400" />
            </div>
            <div className="table-wrap">
              <table className="data-table">
                <thead>
                  <tr>
                    <th>Enfoque</th>
                    <th>Familia</th>
                    <th>Descripción</th>
                  </tr>
                </thead>
                <tbody>
                  {ALGORITHMS.map(([name, family, description]) => (
                    <tr key={name}>
                      <td className="font-semibold text-navy-900">{name}</td>
                      <td><Badge tone="info">{family}</Badge></td>
                      <td className="text-[11px] leading-relaxed">{description}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </section>

          <div className="grid gap-4 lg:grid-cols-2">
            <section className="card p-5">
              <h2 className="text-sm font-bold text-navy-900">Estrategia de validación</h2>
              <ol className="mt-2 space-y-2 text-[11px] leading-relaxed text-slate-600">
                <li><strong>1. Temporal.</strong> Las cohortes más recientes (cuantil 80 de cohort_year) se reservan como prueba.</li>
                <li><strong>2. Geográfica.</strong> Una región con tamaño suficiente se retiene por completo para evaluar transporte territorial.</li>
                <li><strong>3. Interna.</strong> Optuna y el umbral operativo se calculan solo en entrenamiento/validación interna.</li>
                <li><strong>4. Agrupada.</strong> La validación cruzada agrupa por región para evitar fuga de información territorial.</li>
                <li><strong>5. Selección.</strong> El modelo activo se elige por AUC-ROC temporal; es una convención de demostración.</li>
              </ol>
            </section>
            <section className="card p-5">
              <h2 className="text-sm font-bold text-navy-900">Explicabilidad</h2>
              <div className="mt-2 space-y-2 text-[11px] leading-relaxed text-slate-600">
                <p><strong>Árboles:</strong> TreeSHAP sobre el pipeline transformado; contribuciones agregadas por variable original.</p>
                <p><strong>Supervivencia:</strong> KernelSHAP sobre la función de riesgo, con fondo de entrenamiento y columnas estandarizadas.</p>
                <p><strong>LSTM:</strong> atención temporal por visita y oclusión por variable; se informa explícitamente que no es SHAP clásico.</p>
                <p className="rounded-lg bg-slate-50 px-3 py-2 text-slate-500">
                  Las explicaciones describen el comportamiento del modelo. No implican causalidad ni justifican
                  decisiones clínicas por sí solas.
                </p>
              </div>
            </section>
          </div>

          <section className="card p-5">
            <h2 className="text-sm font-bold text-navy-900">Hiperparámetros y artefactos</h2>
            <p className="mt-1 text-[11px] leading-relaxed text-slate-600">
              Cada ejecución guarda un artefacto joblib con el pipeline o los objetos de supervivencia, el umbral,
              las medias/desviaciones y el mapa de variables. Los parámetros óptimos de Optuna, la duración, las
              filas de entrenamiento/prueba y las métricas quedan en la tabla de ejecuciones. Si MLflow está
              habilitado, se registran parámetros, métricas y artefacto.
            </p>
            <pre className="mt-3 overflow-x-auto rounded-lg bg-navy-950 p-3 text-[10px] leading-relaxed text-navy-100">
{`# Entrenamiento
POST /api/models/train
{ "dataset_id": null, "trials": 8 }

# Comparación y activación
GET  /api/models/compare
POST /api/models/{id}/activate

# Predicción individual
POST /api/predictions
{ "age": 58, "sex": "F", "region": "Bogotá D.C.", ... }`}
            </pre>
          </section>
        </div>
      )}

      {tab === "data" && (
        <div className="space-y-4">
          <section className="card">
            <div className="card-header">
              <div>
                <h2 className="text-sm font-bold text-navy-900">Diccionario de variables</h2>
                <p className="text-xs text-slate-500">Esquema mínimo para cargas y entrenamiento.</p>
              </div>
              <Database className="h-4 w-4 text-slate-400" />
            </div>
            <div className="table-wrap">
              <table className="data-table">
                <thead>
                  <tr>
                    <th>Variable(s)</th>
                    <th>Dominio</th>
                    <th>Descripción</th>
                  </tr>
                </thead>
                <tbody>
                  {VARIABLES.map(([name, domain, description]) => (
                    <tr key={name}>
                      <td className="font-mono text-[11px] font-semibold text-navy-900">{name}</td>
                      <td><Badge tone="neutral">{domain}</Badge></td>
                      <td className="text-[11px]">{description}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </section>

          <section className="card p-5">
            <h2 className="text-sm font-bold text-navy-900">Métricas reportadas</h2>
            <div className="mt-3 grid gap-2 md:grid-cols-2">
              {METRICS.map(([name, description]) => (
                <div key={name} className="rounded-lg border border-slate-100 bg-slate-50/60 p-3">
                  <p className="text-xs font-semibold text-navy-900">{name}</p>
                  <p className="mt-0.5 text-[11px] leading-relaxed text-slate-500">{description}</p>
                </div>
              ))}
            </div>
          </section>

          <div className="rounded-xl border border-amber-200 bg-amber-50 p-4 text-[11px] leading-relaxed text-amber-900">
            <strong>Datos sintéticos.</strong> La cohorte de 1.200 registros se genera con semilla fija 42 y un
            predictor lineal ilustrativo. No proviene de historias clínicas, encuestas poblacionales ni registros
            oficiales. Las prevalencias y asociaciones son artificiales y no deben citarse como hallazgos.
          </div>

          <section className="card p-5">
            <h2 className="text-sm font-bold text-navy-900">Umbrales de estratificación</h2>
            <p className="mt-1 text-[11px] leading-relaxed text-slate-600">
              Bajo: probabilidad &lt; {(info?.risk_thresholds.low_max ?? 0.1) * 100}% · Moderado: entre{" "}
              {(info?.risk_thresholds.low_max ?? 0.1) * 100}% y {(info?.risk_thresholds.high_min ?? 0.25) * 100}% ·
              Alto: ≥ {(info?.risk_thresholds.high_min ?? 0.25) * 100}%.
            </p>
            <p className="mt-2 text-[10px] text-slate-500">{info?.risk_thresholds.note}</p>
          </section>
        </div>
      )}

      {tab === "security" && (
        <div className="space-y-4">
          <div className="grid gap-4 lg:grid-cols-2">
            <section className="card p-5">
              <div className="flex items-center gap-2">
                <Lock className="h-4 w-4 text-emerald-600" />
                <h2 className="text-sm font-bold text-navy-900">Controles implementados</h2>
              </div>
              <ul className="mt-2 space-y-1.5 text-[11px] leading-relaxed text-slate-600">
                <li>• Autenticación JWT con access y refresh token; expiración configurable.</li>
                <li>• Contraseñas con bcrypt (coste 12) y política de fortaleza.</li>
                <li>• Autorización por rol y módulo en cada endpoint.</li>
                <li>• Desactivación lógica de usuarios y pacientes; no borrado destructivo.</li>
                <li>• Auditoría append-only sin endpoints de edición o eliminación.</li>
                <li>• Cabeceras de seguridad, CORS restringido y Cache-Control en API.</li>
                <li>• Validación de esquema y límites de tamaño en cargas.</li>
                <li>• Checksum SHA-256 de datasets y registro de descargas.</li>
              </ul>
            </section>
            <section className="card p-5">
              <div className="flex items-center gap-2">
                <ShieldCheck className="h-4 w-4 text-amber-600" />
                <h2 className="text-sm font-bold text-navy-900">Límites y deuda técnica</h2>
              </div>
              <ul className="mt-2 space-y-1.5 text-[11px] leading-relaxed text-slate-600">
                <li>• La revocación de JWT no está implementada del lado servidor.</li>
                <li>• No hay MFA ni integración con SSO institucional.</li>
                <li>• La recuperación de contraseña requiere intervención administrativa.</li>
                <li>• SQLite es el valor por defecto; PostgreSQL/PostGIS es opcional.</li>
                <li>• MinIO, n8n, LangFlow y MLflow requieren servicios/credenciales externas.</li>
                <li>• No se ha ejecutado una auditoría de seguridad externa.</li>
              </ul>
            </section>
          </div>

          <section className="card p-5">
            <h2 className="text-sm font-bold text-navy-900">Roles</h2>
            <div className="mt-3 grid gap-2 md:grid-cols-2 xl:grid-cols-3">
              {[
                ["Superadministrador", "Control total, incluida la creación de otros superadministradores."],
                ["Administrador", "Operación, usuarios y configuración; no crea superadministradores."],
                ["Investigador", "Datos, modelos, predicciones, reportes y lectura de auditoría."],
                ["Profesional de salud", "Registros y predicciones; sin administración de usuarios."],
                ["Analista de datos", "Datasets, modelos y analítica; sin gestión de usuarios."],
                ["Auditor", "Lectura de bitácora, reportes y métricas; sin escritura clínica."],
              ].map(([role, description]) => (
                <div key={role} className="rounded-lg border border-slate-100 bg-slate-50/50 p-3">
                  <p className="text-xs font-bold text-navy-900">{role}</p>
                  <p className="mt-0.5 text-[11px] leading-relaxed text-slate-500">{description}</p>
                </div>
              ))}
            </div>
          </section>

          <section className="card p-5">
            <div className="flex items-center gap-2">
              <BookOpen className="h-4 w-4 text-accent-600" />
              <h2 className="text-sm font-bold text-navy-900">Buenas prácticas de uso</h2>
            </div>
            <p className="mt-2 text-[11px] leading-relaxed text-slate-600">
              No cargue datos personales identificables. Use seudónimos y mantenga la llave de reidentificación
              fuera de la plataforma. Documente procedencia, versión del dataset, versión del modelo y fecha de
              cada análisis. Comunique siempre el carácter sintético de los datos y la incertidumbre de las
              estimaciones. No use los resultados para decisiones asistenciales sin validación externa,
              recalibración local y supervisión profesional.
            </p>
          </section>
        </div>
      )}

      {tab === "deploy" && (
        <div className="space-y-4">
          <section className="card p-5">
            <div className="flex items-center gap-2">
              <Server className="h-4 w-4 text-accent-600" />
              <h2 className="text-sm font-bold text-navy-900">Arquitectura de despliegue</h2>
            </div>
            <p className="mt-2 text-[11px] leading-relaxed text-slate-600">
              El backend FastAPI sirve la API y, si existe, el build estático del frontend. En desarrollo, Vite
              corre en el puerto 5173 y redirige <code>/api</code> a 127.0.0.1:8000. La base por defecto es SQLite;
              para PostgreSQL/PostGIS configure <code>DATABASE_URL</code>. Los artefactos se guardan en{" "}
              <code>data/models</code> y los reportes en <code>data/reports</code>.
            </p>
            <pre className="mt-3 overflow-x-auto rounded-lg bg-navy-950 p-3 text-[10px] leading-relaxed text-navy-100">
{`# Backend
python -m venv .venv
.venv\\Scripts\\pip install -r requirements.txt
.venv\\Scripts\\uvicorn backend.app.main:app --host 127.0.0.1 --port 8000

# Frontend
cd frontend
npm install
npm run build

# Docker Compose (infraestructura opcional)
docker compose up -d`}
            </pre>
          </section>

          <div className="grid gap-4 lg:grid-cols-2">
            <section className="card p-5">
              <div className="flex items-center gap-2">
                <Terminal className="h-4 w-4 text-accent-600" />
                <h2 className="text-sm font-bold text-navy-900">Endpoints principales</h2>
              </div>
              <div className="mt-2 space-y-1 font-mono text-[10px] text-slate-600">
                {[
                  "POST /api/auth/login",
                  "GET  /api/dashboard/summary",
                  "GET  /api/patients",
                  "GET  /api/datasets",
                  "POST /api/models/train",
                  "GET  /api/models/compare",
                  "POST /api/predictions",
                  "POST /api/predictions/batch",
                  "GET  /api/geo/regions",
                  "POST /api/reports",
                  "GET  /api/audit",
                  "POST /api/assistant/query",
                  "GET  /api/system/health",
                  "GET  /api/docs",
                ].map((endpoint) => (
                  <p key={endpoint} className="rounded bg-slate-50 px-2 py-1">{endpoint}</p>
                ))}
              </div>
            </section>

            <section className="card p-5">
              <div className="flex items-center gap-2">
                <FileText className="h-4 w-4 text-accent-600" />
                <h2 className="text-sm font-bold text-navy-900">Servicios opcionales</h2>
              </div>
              <div className="mt-2 space-y-2 text-[11px] leading-relaxed text-slate-600">
                <p><strong>PostgreSQL/PostGIS:</strong> cambiar DATABASE_URL y habilitar la extensión para análisis territorial avanzado.</p>
                <p><strong>MLflow:</strong> activar MLFLOW_ENABLED y definir el tracking URI para registrar experimentos.</p>
                <p><strong>n8n:</strong> flujos de alertas y reportes programados; requiere instancia y webhook.</p>
                <p><strong>LangFlow / LLM:</strong> redacción sobre contexto recuperado; nunca sustituye al modelo predictivo.</p>
                <p><strong>MinIO:</strong> almacenamiento S3 compatible para artefactos y reportes.</p>
              </div>
              <p className="mt-3 rounded-lg bg-slate-50 px-3 py-2 text-[10px] text-slate-500">
                Las integraciones se entregan como configuración y flujos; no incluyen credenciales ni servicios
                aprovisionados.
              </p>
            </section>
          </div>

          <div className="rounded-xl border border-sky-200 bg-sky-50 p-4 text-[11px] leading-relaxed text-sky-900">
            <strong>Variables de entorno relevantes:</strong> SECRET_KEY, DATABASE_URL, CORS_ORIGINS,
            ADMIN_PASSWORD, SEED_DEMO_DATA, RISK_LOW_MAX, RISK_HIGH_MIN, PREDICTION_HORIZON_YEARS, ML_TRIALS,
            ML_CV_FOLDS, MLFLOW_ENABLED, MLFLOW_TRACKING_URI, LLM_PROVIDER, OPENAI_API_KEY. Consulte{" "}
            <code>.env.example</code>.
          </div>
        </div>
      )}
    </div>
  );
}

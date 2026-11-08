import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  Activity,
  AlertTriangle,
  BarChart3,
  BrainCircuit,
  CheckCircle2,
  Clock,
  Cpu,
  Gauge,
  Play,
  RefreshCw,
  SlidersHorizontal,
  Target,
} from "lucide-react";
import {
  Bar,
  BarChart,
  CartesianGrid,
  ComposedChart,
  Line,
  LineChart,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { api, buildQuery } from "../lib/api";
import type { Dataset, ModelRun } from "../lib/types";
import { formatDateTime, formatNumber, formatPct } from "../lib/format";
import { useAuth } from "../context/AuthContext";
import { useToast } from "../context/ToastContext";
import {
  Badge,
  DemoNotice,
  EmptyState,
  ErrorBanner,
  Field,
  Modal,
  PageHeader,
  PageLoading,
  Spinner,
} from "../components/ui";

interface Algorithm {
  key: string;
  name: string;
  family: string;
  description: string;
  available: boolean;
  requires: string;
}

const STATUS_TONE: Record<string, "neutral" | "info" | "success" | "warning" | "danger"> = {
  queued: "warning",
  running: "info",
  ready: "success",
  failed: "danger",
};

export default function ModelsPage() {
  const { can } = useAuth();
  const { notify } = useToast();
  const [models, setModels] = useState<ModelRun[]>([]);
  const [compare, setCompare] = useState<ModelRun[]>([]);
  const [algorithms, setAlgorithms] = useState<Algorithm[]>([]);
  const [datasets, setDatasets] = useState<Dataset[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [trainOpen, setTrainOpen] = useState(false);
  const [trials, setTrials] = useState(8);
  const [datasetId, setDatasetId] = useState<string>("");
  const [training, setTraining] = useState(false);
  const [detail, setDetail] = useState<ModelRun | null>(null);
  const [tab, setTab] = useState<"performance" | "validation" | "calibration" | "importance" | "params">("performance");
  const detailRef = useRef(detail);
  detailRef.current = detail;

  const load = useCallback(async (quiet = false) => {
    if (!quiet) setLoading(true);
    try {
      const [list, comparison] = await Promise.all([
        api.get<{ items: ModelRun[] }>("/models" + buildQuery({ page_size: 60 })),
        api.get<ModelRun[]>("/models/compare"),
      ]);
      setModels(list.items);
      setCompare(comparison);
      setError("");
      const current = detailRef.current;
      if (current) {
        const refreshed = list.items.find((item) => item.id === current.id);
        if (refreshed) setDetail(refreshed);
      }
    } catch (err) {
      if (!quiet) setError(err instanceof Error ? err.message : "No se pudieron cargar los modelos.");
    } finally {
      if (!quiet) setLoading(false);
    }
  }, []);

  useEffect(() => {
    load();
    api.get<Algorithm[]>("/models/algorithms").then(setAlgorithms).catch(() => undefined);
    api.get<Dataset[]>("/datasets").then(setDatasets).catch(() => undefined);
  }, [load]);

  const anyRunning = useMemo(
    () => models.some((model) => model.status === "queued" || model.status === "running"),
    [models]
  );

  useEffect(() => {
    if (!anyRunning) return;
    const timer = window.setInterval(() => load(true), 4000);
    return () => window.clearInterval(timer);
  }, [anyRunning, load]);

  const train = async () => {
    setTraining(true);
    try {
      await api.post("/models/train", {
        dataset_id: datasetId ? Number(datasetId) : null,
        trials,
      });
      notify("Entrenamiento iniciado. El progreso se actualiza automáticamente.", "success");
      setTrainOpen(false);
      load(true);
    } catch (err) {
      notify(err instanceof Error ? err.message : "No se pudo iniciar el entrenamiento.", "error");
    } finally {
      setTraining(false);
    }
  };

  const activate = async (model: ModelRun) => {
    try {
      await api.post(`/models/${model.id}/activate`);
      notify(`${model.name} activado para inferencia.`, "success");
      load(true);
    } catch (err) {
      notify(err instanceof Error ? err.message : "No se pudo activar el modelo.", "error");
    }
  };

  const active = models.find((model) => model.is_active);
  const bestDetail = useMemo(() => {
    if (!detail) return null;
    const metrics = detail.metrics || {};
    const disc = metrics.discrimination || {};
    return {
      metrics,
      disc,
      temporal: metrics.temporal?.metrics || {},
      geographic: metrics.geographic?.metrics || {},
      cv: metrics.cv || {},
      roc: (disc.roc_curve?.fpr || []).map((fpr: number, index: number) => ({
        fpr,
        tpr: disc.roc_curve?.tpr?.[index] ?? 0,
      })),
      calibration: disc.calibration || [],
      dca: disc.dca || [],
    };
  }, [detail]);

  return (
    <div className="space-y-4">
      <PageHeader
        eyebrow="Machine learning"
        title="Modelos y MLOps"
        description="Entrenamiento de cinco enfoques, comparación en un holdout honesto, selección del modelo activo por validación interna y trazabilidad de hiperparámetros."
        actions={
          <>
            <button className="btn-ghost" onClick={() => load()}>
              <RefreshCw className="h-4 w-4" />
              Actualizar
            </button>
            {can("models", "write") && (
              <button className="btn-accent" onClick={() => setTrainOpen(true)} disabled={anyRunning}>
                <Play className="h-4 w-4" />
                {anyRunning ? "Entrenando…" : "Entrenar modelos"}
              </button>
            )}
          </>
        }
      />

      <DemoNotice>
        Las métricas corresponden al dataset seleccionado y se calculan en un holdout no usado para ajustar.
        La selección automática del modelo activo usa validación interna o validación cruzada, nunca el holdout
        de prueba. Los resultados no constituyen evidencia clínica.
      </DemoNotice>

      {error && <ErrorBanner message={error} onRetry={() => load()} />}

      {anyRunning && (
        <div className="flex items-center gap-2 rounded-xl border border-sky-200 bg-sky-50 px-4 py-3 text-sm text-sky-800">
          <Spinner className="h-4 w-4" />
          Entrenamiento en curso. Esta vista se actualiza cada 4 segundos; puede continuar navegando.
        </div>
      )}

      <div className="grid gap-3 sm:grid-cols-2 xl:grid-cols-4">
        <div className="card p-4">
          <p className="text-[11px] font-semibold uppercase tracking-wider text-slate-500">Modelo activo</p>
          <p className="mt-1 truncate text-sm font-bold text-navy-900">{active?.name ?? "Sin modelo activo"}</p>
          <p className="mt-1 text-xs text-slate-500">{active ? active.algorithm : "Entrene los modelos para inferir"}</p>
        </div>
        <div className="card p-4">
          <p className="text-[11px] font-semibold uppercase tracking-wider text-slate-500">AUC-ROC (holdout)</p>
          <p className="mt-1 text-2xl font-bold text-navy-900">{active?.summary.roc_auc?.toFixed(3) ?? "—"}</p>
          <p className="mt-1 text-xs text-slate-500">Conjunto de prueba no usado para ajustar</p>
        </div>
        <div className="card p-4">
          <p className="text-[11px] font-semibold uppercase tracking-wider text-slate-500">Modelos listos</p>
          <p className="mt-1 text-2xl font-bold text-navy-900">
            {models.filter((model) => model.status === "ready").length}
          </p>
          <p className="mt-1 text-xs text-slate-500">de {models.length} ejecuciones registradas</p>
        </div>
        <div className="card p-4">
          <p className="text-[11px] font-semibold uppercase tracking-wider text-slate-500">Enfoques disponibles</p>
          <p className="mt-1 text-2xl font-bold text-navy-900">
            {algorithms.filter((algorithm) => algorithm.available).length}/5
          </p>
          <p className="mt-1 text-xs text-slate-500">Incluye modelos de supervivencia</p>
        </div>
      </div>

      <section className="card">
        <div className="card-header">
          <div>
            <h2 className="text-sm font-bold text-navy-900">Ejecuciones de entrenamiento</h2>
            <p className="text-xs text-slate-500">Historial completo con estado, métricas y duración.</p>
          </div>
          <Badge tone="info">{models.length} ejecuciones</Badge>
        </div>
        {loading ? (
          <PageLoading label="Cargando modelos…" />
        ) : models.length === 0 ? (
          <div className="p-5">
            <EmptyState
              title="Aún no hay modelos"
              message="Inicie el entrenamiento de los cinco enfoques. El proceso corre en segundo plano y puede tardar varios minutos."
              action={
                can("models", "write") ? (
                  <button className="btn-accent mt-2" onClick={() => setTrainOpen(true)}>
                    <Play className="h-4 w-4" />
                    Entrenar ahora
                  </button>
                ) : undefined
              }
            />
          </div>
        ) : (
          <div className="table-wrap">
            <table className="data-table">
              <thead>
                <tr>
                  <th>Modelo</th>
                  <th>Estado</th>
                  <th>AUC</th>
                  <th>F1</th>
                  <th>MCC</th>
                  <th>Brier</th>
                  <th>C-index</th>
                  <th>Entrenado</th>
                  <th className="text-right">Acciones</th>
                </tr>
              </thead>
              <tbody>
                {models.map((model) => (
                  <tr key={model.id}>
                    <td>
                      <p className="font-semibold text-navy-900">{model.name}</p>
                      <p className="text-[10px] text-slate-400">
                        {model.algorithm} · {formatNumber(model.train_rows)} filas · {model.horizon_years} años
                      </p>
                    </td>
                    <td>
                      <Badge tone={STATUS_TONE[model.status] || "neutral"}>{model.status}</Badge>
                      {model.is_active && <span className="ml-1 text-[10px] font-bold text-emerald-600">ACTIVO</span>}
                    </td>
                    <td className="font-semibold">{model.summary.roc_auc?.toFixed(3) ?? "—"}</td>
                    <td>{model.summary.f1?.toFixed(3) ?? "—"}</td>
                    <td>{model.summary.mcc?.toFixed(3) ?? "—"}</td>
                    <td>{model.summary.brier?.toFixed(3) ?? "—"}</td>
                    <td>{model.metrics?.c_index_temporal?.toFixed?.(3) ?? "—"}</td>
                    <td>
                      <p className="text-[11px]">{model.trained_at ? formatDateTime(model.trained_at) : "—"}</p>
                      <p className="text-[10px] text-slate-400">
                        {model.duration_seconds ? `${model.duration_seconds.toFixed(0)} s` : "—"}
                      </p>
                    </td>
                    <td>
                      <div className="flex justify-end gap-1">
                        <button className="btn-ghost px-2 py-1 text-[11px]" onClick={() => { setDetail(model); setTab("performance"); }}>
                          Detalle
                        </button>
                        {can("models", "write") && model.status === "ready" && !model.is_active && (
                          <button className="btn-ghost px-2 py-1 text-[11px]" onClick={() => activate(model)}>
                            Activar
                          </button>
                        )}
                      </div>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </section>

      <div className="grid gap-4 xl:grid-cols-3">
        <section className="card xl:col-span-2">
          <div className="card-header">
            <div>
              <h2 className="text-sm font-bold text-navy-900">Comparación por enfoque</h2>
              <p className="text-xs text-slate-500">Mejor versión lista de cada algoritmo.</p>
            </div>
            <BarChart3 className="h-4 w-4 text-slate-400" />
          </div>
          {compare.length === 0 ? (
            <p className="px-5 py-8 text-sm text-slate-500">Sin modelos listos para comparar.</p>
          ) : (
            <div className="table-wrap">
              <table className="data-table">
                <thead>
                  <tr>
                    <th>Algoritmo</th>
                    <th>AUC holdout</th>
                    <th>AUC geográfica</th>
                    <th>Sensibilidad</th>
                    <th>Especificidad</th>
                    <th>MCC</th>
                  </tr>
                </thead>
                <tbody>
                  {compare.map((model) => (
                    <tr key={model.id}>
                      <td>
                        <p className="font-semibold text-navy-900">{model.name}</p>
                        <p className="text-[10px] text-slate-400">{model.validation_strategy}</p>
                      </td>
                      <td className="font-semibold">{model.summary.roc_auc?.toFixed(3) ?? "—"}</td>
                      <td>{model.metrics?.geographic?.metrics?.roc_auc?.toFixed?.(3) ?? "—"}</td>
                      <td>{model.summary.sensitivity?.toFixed(3) ?? "—"}</td>
                      <td>{model.summary.specificity?.toFixed(3) ?? "—"}</td>
                      <td>{model.summary.mcc?.toFixed(3) ?? "—"}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </section>

        <section className="card">
          <div className="card-header">
            <div>
              <h2 className="text-sm font-bold text-navy-900">Enfoques de modelado</h2>
              <p className="text-xs text-slate-500">Disponibilidad según dependencias instaladas.</p>
            </div>
          </div>
          <div className="divide-y divide-slate-50">
            {algorithms.map((algorithm) => (
              <div key={algorithm.key} className="flex items-start gap-3 px-5 py-3">
                <span className="mt-0.5 flex h-8 w-8 shrink-0 items-center justify-center rounded-lg bg-navy-50 text-navy-700">
                  {algorithm.family === "deep" ? <Cpu className="h-4 w-4" /> : <BrainCircuit className="h-4 w-4" />}
                </span>
                <div className="min-w-0 flex-1">
                  <div className="flex items-center gap-2">
                    <p className="text-xs font-semibold text-navy-900">{algorithm.name}</p>
                    {algorithm.available ? (
                      <CheckCircle2 className="h-3.5 w-3.5 text-emerald-600" />
                    ) : (
                      <AlertTriangle className="h-3.5 w-3.5 text-amber-500" />
                    )}
                  </div>
                  <p className="text-[11px] leading-snug text-slate-500">{algorithm.description}</p>
                  <p className="mt-0.5 text-[10px] text-slate-400">{algorithm.requires}</p>
                </div>
              </div>
            ))}
          </div>
        </section>
      </div>

      <section className="card p-5">
        <h2 className="text-sm font-bold text-navy-900">Protocolo de validación</h2>
        <div className="mt-3 grid gap-3 md:grid-cols-3">
          {[
            ["Partición honesta", "Temporal solo si existe un año de cohorte informativo; si no, holdout aleatorio estratificado declarado como tal. Nunca se etiqueta un split aleatorio como temporal."],
            ["Sin fuga de información", "Optuna y el umbral se calculan solo en entrenamiento/validación interna; el holdout se usa únicamente para el reporte final. La selección del modelo activo usa validación interna o CV."],
            ["Métrica principal", "AUC-ROC en el holdout, con sensibilidad, especificidad, F1, MCC, Brier, calibración, Hosmer-Lemeshow y DCA como apoyo."],
          ].map(([title, text]) => (
            <div key={title} className="rounded-lg border border-slate-100 bg-slate-50/50 p-4">
              <p className="text-xs font-bold text-navy-900">{title}</p>
              <p className="mt-1 text-[11px] leading-relaxed text-slate-500">{text}</p>
            </div>
          ))}
        </div>
        <p className="mt-3 text-[10px] leading-relaxed text-slate-400">
          La validación es interna o en un holdout del mismo dataset. No sustituye una validación externa, una
          recalibración local ni una evaluación de equidad con datos de otra población.
        </p>
      </section>

      <Modal
        open={trainOpen}
        onClose={() => setTrainOpen(false)}
        title="Entrenar los cinco modelos"
        description="El proceso se ejecuta en segundo plano. Puede cerrar esta ventana y continuar usando la plataforma."
        footer={
          <>
            <button className="btn-ghost" onClick={() => setTrainOpen(false)}>
              Cancelar
            </button>
            <button className="btn-accent" onClick={train} disabled={training}>
              {training ? <Spinner className="h-4 w-4 text-white" /> : <Play className="h-4 w-4" />}
              {training ? "Iniciando…" : "Iniciar entrenamiento"}
            </button>
          </>
        }
      >
        <div className="space-y-4">
          <Field label="Dataset de entrenamiento">
            <select className="input" value={datasetId} onChange={(event) => setDatasetId(event.target.value)}>
              <option value="">Cohorte sintética de demostración</option>
              {datasets.map((dataset) => (
                <option key={dataset.id} value={dataset.id}>
                  {dataset.name} ({dataset.rows_count} filas)
                </option>
              ))}
            </select>
          </Field>
          <Field label={`Ensayos de Optuna por modelo: ${trials}`} hint="Más ensayos mejoran la búsqueda, pero aumentan el tiempo de cómputo.">
            <input
              type="range"
              min={1}
              max={20}
              value={trials}
              onChange={(event) => setTrials(Number(event.target.value))}
              className="w-full accent-accent-600"
            />
          </Field>
          <div className="rounded-lg border border-slate-200 bg-slate-50 p-4 text-xs leading-relaxed text-slate-600">
            <p className="font-semibold text-navy-900">Se entrenarán:</p>
            <ul className="mt-2 space-y-1">
              <li>Random Forest y XGBoost (clasificación a 10 años).</li>
              <li>Cox PH, XGBoost-Cox y LSTM-Attention-Cox (supervivencia).</li>
              <li>Validación temporal cuando el dataset aporta año de cohorte; en caso contrario, holdout aleatorio estratificado.</li>
              <li>El mejor modelo por validación interna/CV se activa automáticamente al finalizar.</li>
            </ul>
          </div>
          {!algorithms.find((algorithm) => algorithm.key === "lstm_attention_cox")?.available && (
            <div className="flex gap-2 rounded-lg border border-amber-200 bg-amber-50 px-3 py-2 text-[11px] text-amber-900">
              <AlertTriangle className="mt-0.5 h-3.5 w-3.5 shrink-0" />
              PyTorch no está disponible: el modelo LSTM se marcará como fallido, pero los otros cuatro continuarán.
            </div>
          )}
        </div>
      </Modal>

      <Modal
        open={detail !== null}
        onClose={() => setDetail(null)}
        title={detail?.name ?? "Detalle del modelo"}
        description={detail ? `${detail.algorithm} · ${detail.horizon_years} años · ${formatNumber(detail.train_rows)} filas de entrenamiento` : undefined}
        size="xl"
        footer={
          detail ? (
            <>
              {detail.is_active ? (
                <Badge tone="success">Modelo activo</Badge>
              ) : (
                can("models", "write") &&
                detail.status === "ready" && (
                  <button className="btn-accent" onClick={() => activate(detail)}>
                    <Target className="h-4 w-4" />
                    Activar para inferencia
                  </button>
                )
              )}
              <button className="btn-primary" onClick={() => setDetail(null)}>
                Cerrar
              </button>
            </>
          ) : undefined
        }
      >
        {detail && bestDetail && (
          <div className="space-y-4">
            {detail.status === "failed" && detail.error && (
              <div className="rounded-lg border border-red-200 bg-red-50 p-3 text-xs text-red-800">
                <p className="font-semibold">Fallo de entrenamiento</p>
                <pre className="mt-1 max-h-32 overflow-auto whitespace-pre-wrap font-mono text-[10px]">{detail.error}</pre>
              </div>
            )}

            <div className="grid grid-cols-2 gap-2 sm:grid-cols-4">
              {[
                ["AUC-ROC", bestDetail.disc.roc_auc?.toFixed?.(3) ?? "—"],
                ["PR-AUC", bestDetail.disc.pr_auc?.toFixed?.(3) ?? "—"],
                ["Sensibilidad", bestDetail.disc.sensitivity?.toFixed?.(3) ?? "—"],
                ["Especificidad", bestDetail.disc.specificity?.toFixed?.(3) ?? "—"],
                ["F1", bestDetail.disc.f1?.toFixed?.(3) ?? "—"],
                ["MCC", bestDetail.disc.mcc?.toFixed?.(3) ?? "—"],
                ["Brier", bestDetail.disc.brier?.toFixed?.(3) ?? "—"],
                ["C-index", detail.metrics?.c_index_temporal?.toFixed?.(3) ?? "—"],
              ].map(([label, value]) => (
                <div key={label} className="rounded-lg border border-slate-100 bg-slate-50/60 p-2.5">
                  <p className="text-[9px] font-semibold uppercase tracking-wide text-slate-400">{label}</p>
                  <p className="mt-0.5 text-base font-bold text-navy-900">{value}</p>
                </div>
              ))}
            </div>

            <div className="flex flex-wrap gap-1.5">
              {([
                ["performance", "Discriminación"],
                ["validation", "Validación"],
                ["calibration", "Calibración y DCA"],
                ["importance", "Importancia"],
                ["params", "Parámetros"],
              ] as const).map(([key, label]) => (
                <button
                  key={key}
                  onClick={() => setTab(key)}
                  className={
                    tab === key
                      ? "rounded-lg bg-navy-900 px-3 py-1.5 text-xs font-semibold text-white"
                      : "rounded-lg border border-slate-200 px-3 py-1.5 text-xs font-semibold text-slate-600 hover:bg-slate-50"
                  }
                >
                  {label}
                </button>
              ))}
            </div>

            {tab === "performance" && (
              <div className="grid gap-4 lg:grid-cols-2">
                <div className="rounded-lg border border-slate-100 p-3">
                  <p className="mb-2 text-xs font-bold text-navy-900">Curva ROC · holdout</p>
                  {bestDetail.roc.length > 1 ? (
                    <div className="h-60">
                      <ResponsiveContainer width="100%" height="100%">
                        <LineChart data={bestDetail.roc} margin={{ top: 5, right: 10, left: -18, bottom: 0 }}>
                          <CartesianGrid strokeDasharray="3 3" stroke="#e6edf6" />
                          <XAxis dataKey="fpr" tick={{ fontSize: 9 }} tickLine={false} axisLine={false} domain={[0, 1]} />
                          <YAxis tick={{ fontSize: 9 }} tickLine={false} axisLine={false} domain={[0, 1]} />
                          <Tooltip contentStyle={{ fontSize: 11, borderRadius: 8 }} />
                          <ReferenceLine x={0} y={0} stroke="#cbd5e1" strokeDasharray="4 4" />
                          <Line type="monotone" dataKey="tpr" name="TPR" stroke="#2577eb" strokeWidth={2} dot={false} />
                        </LineChart>
                      </ResponsiveContainer>
                    </div>
                  ) : (
                    <p className="py-8 text-center text-xs text-slate-400">Curva no disponible.</p>
                  )}
                </div>
                <div className="rounded-lg border border-slate-100 p-3">
                  <p className="mb-2 text-xs font-bold text-navy-900">Intervalo bootstrap AUC</p>
                  <div className="flex h-32 flex-col items-center justify-center rounded-lg bg-slate-50">
                    <p className="text-3xl font-bold text-navy-900">{bestDetail.disc.roc_auc?.toFixed?.(3) ?? "—"}</p>
                    <p className="mt-1 text-[11px] text-slate-500">
                      IC 95%: {Array.isArray(bestDetail.disc.auc_ci) ? bestDetail.disc.auc_ci.join(" – ") : "no disponible"}
                    </p>
                  </div>
                  <div className="mt-3 space-y-1.5 text-[11px]">
                    <div className="flex justify-between">
                      <span className="text-slate-500">Umbral operativo</span>
                      <span className="font-semibold">{bestDetail.disc.threshold?.toFixed?.(3) ?? "—"}</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-slate-500">Pendiente de calibración</span>
                      <span className="font-semibold">{bestDetail.disc.calibration_slope ?? "—"}</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-slate-500">Prevalencia en prueba</span>
                      <span className="font-semibold">{formatPct(bestDetail.disc.prevalence)}</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-slate-500">Tamaño de prueba</span>
                      <span className="font-semibold">{formatNumber(bestDetail.disc.n)}</span>
                    </div>
                  </div>
                </div>
              </div>
            )}

            {tab === "validation" && (
              <div className="space-y-3">
                <div className="grid gap-3 md:grid-cols-2">
                  <div className="rounded-lg border border-slate-100 p-4">
                    <div className="flex items-center gap-2">
                      <Clock className="h-4 w-4 text-accent-600" />
                      <p className="text-xs font-bold text-navy-900">Holdout de prueba</p>
                    </div>
                    <p className="mt-1 text-[11px] text-slate-500">
                      {bestDetail.metrics.temporal?.split_type === "temporal"
                        ? `Cohorte desde ${bestDetail.metrics.temporal?.cutoff_year ?? "—"}`
                        : "Partición aleatoria estratificada"}{" "}
                      · n = {formatNumber(bestDetail.metrics.temporal?.n)}
                    </p>
                    <div className="mt-3 grid grid-cols-2 gap-2 text-[11px]">
                      <p>AUC: <strong>{bestDetail.temporal.roc_auc?.toFixed?.(3) ?? "—"}</strong></p>
                      <p>F1: <strong>{bestDetail.temporal.f1?.toFixed?.(3) ?? "—"}</strong></p>
                      <p>MCC: <strong>{bestDetail.temporal.mcc?.toFixed?.(3) ?? "—"}</strong></p>
                      <p>Brier: <strong>{bestDetail.temporal.brier?.toFixed?.(3) ?? "—"}</strong></p>
                    </div>
                  </div>
                  <div className="rounded-lg border border-slate-100 p-4">
                    <div className="flex items-center gap-2">
                      <Gauge className="h-4 w-4 text-accent-600" />
                      <p className="text-xs font-bold text-navy-900">Retención geográfica</p>
                    </div>
                    <p className="mt-1 text-[11px] text-slate-500">
                      Región retenida: {bestDetail.metrics.geographic?.region ?? "no aplica"} · n ={" "}
                      {formatNumber(bestDetail.metrics.geographic?.n)}
                    </p>
                    {!bestDetail.metrics.geographic?.region && (
                      <p className="mt-2 text-[10px] text-amber-700">
                        {bestDetail.metrics.geographic?.note ?? "No disponible para este dataset."}
                      </p>
                    )}
                    <div className="mt-3 grid grid-cols-2 gap-2 text-[11px]">
                      <p>AUC: <strong>{bestDetail.geographic.roc_auc?.toFixed?.(3) ?? "—"}</strong></p>
                      <p>F1: <strong>{bestDetail.geographic.f1?.toFixed?.(3) ?? "—"}</strong></p>
                      <p>MCC: <strong>{bestDetail.geographic.mcc?.toFixed?.(3) ?? "—"}</strong></p>
                      <p>Brier: <strong>{bestDetail.geographic.brier?.toFixed?.(3) ?? "—"}</strong></p>
                    </div>
                  </div>
                </div>
                <div className="rounded-lg bg-slate-50 p-4 text-[11px] leading-relaxed text-slate-600">
                  <p className="font-semibold text-navy-900">Validación cruzada agrupada</p>
                  {bestDetail.cv?.roc_auc_mean !== undefined ? (
                    <p className="mt-1">
                      {bestDetail.cv.strategy}: AUC {bestDetail.cv.roc_auc_mean} ± {bestDetail.cv.roc_auc_std} en{" "}
                      {bestDetail.cv.folds} folds. La agrupación por región evita fuga territorial.
                    </p>
                  ) : (
                    <p className="mt-1">No disponible para esta ejecución.</p>
                  )}
                  <p className="mt-2 text-slate-500">{detail.validation_strategy}</p>
                </div>
              </div>
            )}

            {tab === "calibration" && (
              <div className="grid gap-4 lg:grid-cols-2">
                <div className="rounded-lg border border-slate-100 p-3">
                  <p className="mb-2 text-xs font-bold text-navy-900">Calibración observada vs. estimada</p>
                  {bestDetail.calibration.length > 0 ? (
                    <div className="h-60">
                      <ResponsiveContainer width="100%" height="100%">
                        <LineChart data={bestDetail.calibration} margin={{ top: 5, right: 10, left: -18, bottom: 0 }}>
                          <CartesianGrid strokeDasharray="3 3" stroke="#e6edf6" />
                          <XAxis dataKey="mean_pred" tick={{ fontSize: 9 }} domain={[0, 1]} tickLine={false} axisLine={false} />
                          <YAxis tick={{ fontSize: 9 }} domain={[0, 1]} tickLine={false} axisLine={false} />
                          <Tooltip contentStyle={{ fontSize: 11, borderRadius: 8 }} />
                          <ReferenceLine segment={[{ x: 0, y: 0 }, { x: 1, y: 1 }]} stroke="#94a3b8" strokeDasharray="4 4" />
                          <Line type="monotone" dataKey="frac_pos" name="Observado" stroke="#c53030" strokeWidth={2} />
                        </LineChart>
                      </ResponsiveContainer>
                    </div>
                  ) : (
                    <p className="py-8 text-center text-xs text-slate-400">Sin datos de calibración.</p>
                  )}
                </div>
                <div className="rounded-lg border border-slate-100 p-3">
                  <p className="mb-2 text-xs font-bold text-navy-900">Curva de decisión (DCA)</p>
                  {bestDetail.dca.length > 0 ? (
                    <div className="h-60">
                      <ResponsiveContainer width="100%" height="100%">
                        <ComposedChart data={bestDetail.dca} margin={{ top: 5, right: 10, left: -18, bottom: 0 }}>
                          <CartesianGrid strokeDasharray="3 3" stroke="#e6edf6" />
                          <XAxis dataKey="threshold" tick={{ fontSize: 9 }} tickLine={false} axisLine={false} />
                          <YAxis tick={{ fontSize: 9 }} tickLine={false} axisLine={false} />
                          <Tooltip contentStyle={{ fontSize: 11, borderRadius: 8 }} />
                          <ReferenceLine y={0} stroke="#94a3b8" />
                          <Line type="monotone" dataKey="model" name="Modelo" stroke="#2577eb" strokeWidth={2} dot={false} />
                          <Line type="monotone" dataKey="treat_all" name="Tratar a todos" stroke="#c53030" strokeDasharray="4 4" dot={false} />
                        </ComposedChart>
                      </ResponsiveContainer>
                    </div>
                  ) : (
                    <p className="py-8 text-center text-xs text-slate-400">Sin datos de DCA.</p>
                  )}
                </div>
              </div>
            )}

            {tab === "importance" && (
              <div className="rounded-lg border border-slate-100 p-3">
                <p className="mb-2 text-xs font-bold text-navy-900">Importancia global por variable</p>
                {detail.feature_importance.length > 0 ? (
                  <div style={{ height: Math.max(240, detail.feature_importance.length * 30) }}>
                    <ResponsiveContainer width="100%" height="100%">
                      <BarChart
                        data={detail.feature_importance}
                        layout="vertical"
                        margin={{ top: 0, right: 20, left: 100, bottom: 0 }}
                      >
                        <CartesianGrid strokeDasharray="3 3" stroke="#e6edf6" horizontal={false} />
                        <XAxis type="number" tick={{ fontSize: 9 }} tickLine={false} axisLine={false} />
                        <YAxis type="category" dataKey="label" width={100} tick={{ fontSize: 9 }} tickLine={false} axisLine={false} />
                        <Tooltip contentStyle={{ fontSize: 11, borderRadius: 8 }} />
                        <Bar dataKey="importance" name="Importancia" fill="#4777aa" radius={[0, 4, 4, 0]} />
                      </BarChart>
                    </ResponsiveContainer>
                  </div>
                ) : (
                  <p className="py-8 text-center text-xs text-slate-400">Sin importancia registrada.</p>
                )}
                <p className="mt-1 text-[10px] text-slate-400">
                  La importancia describe el comportamiento del modelo y no implica causalidad.
                </p>
              </div>
            )}

            {tab === "params" && (
              <pre className="max-h-80 overflow-auto rounded-lg bg-navy-950 p-4 text-[11px] leading-relaxed text-navy-100">
                {JSON.stringify(detail.params, null, 2)}
              </pre>
            )}
          </div>
        )}
      </Modal>
    </div>
  );
}

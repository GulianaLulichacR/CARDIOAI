import { useCallback, useEffect, useState, type FormEvent } from "react";
import {
  CheckCircle2,
  Database,
  Download,
  FileSpreadsheet,
  Layers,
  Plus,
  UploadCloud,
} from "lucide-react";
import { api, downloadFile } from "../lib/api";
import type { Dataset } from "../lib/types";
import { formatDate, formatNumber } from "../lib/format";
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

interface Variable {
  name: string;
  label: string | null;
  dtype: string;
  unit: string | null;
  description: string | null;
  missing_count: number;
  missing_pct: number;
  min: number | null;
  max: number | null;
  mean: number | null;
  unique: number;
}

export default function DatasetsPage() {
  const { can } = useAuth();
  const { notify } = useToast();
  const [datasets, setDatasets] = useState<Dataset[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [detail, setDetail] = useState<Dataset | null>(null);
  const [variables, setVariables] = useState<Variable[]>([]);
  const [preview, setPreview] = useState<{ columns: string[]; rows: Record<string, unknown>[] } | null>(null);
  const [tab, setTab] = useState<"variables" | "quality" | "preview">("variables");
  const [detailLoading, setDetailLoading] = useState(false);
  const [uploadOpen, setUploadOpen] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [file, setFile] = useState<File | null>(null);
  const [meta, setMeta] = useState({
    name: "",
    source: "Carga manual",
    license: "No especificada",
    description: "",
    is_demo: false,
  });

  const load = useCallback(async () => {
    setLoading(true);
    setError("");
    try {
      setDatasets(await api.get<Dataset[]>("/datasets"));
    } catch (err) {
      setError(err instanceof Error ? err.message : "No se pudieron cargar los datasets.");
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    load();
  }, [load]);

  const loadDetail = async (dataset: Dataset, nextTab = tab) => {
    setDetail(dataset);
    setTab(nextTab);
    if (nextTab === "variables" && variables.length && dataset.id === detail?.id) return;
    setDetailLoading(true);
    try {
      if (nextTab === "variables") {
        setVariables(await api.get<Variable[]>(`/datasets/${dataset.id}/variables`));
      }
      if (nextTab === "preview") {
        setPreview(await api.get<{ columns: string[]; rows: Record<string, unknown>[] }>(`/datasets/${dataset.id}/preview?limit=20`));
      }
      if (nextTab === "quality") {
        const full = await api.get<Dataset & { quality: Record<string, any> }>(`/datasets/${dataset.id}`);
        setDetail(full);
      }
    } catch (err) {
      notify(err instanceof Error ? err.message : "No se pudo cargar el detalle.", "error");
    } finally {
      setDetailLoading(false);
    }
  };

  const switchTab = (next: "variables" | "quality" | "preview") => {
    if (!detail) return;
    if (next === "variables" && variables.length) {
      setTab(next);
      return;
    }
    if (next === "preview" && preview) {
      setTab(next);
      return;
    }
    loadDetail(detail, next);
  };

  const ensureDemo = async () => {
    try {
      await api.post("/datasets/demo");
      notify("Dataset de demostración disponible.", "success");
      load();
    } catch (err) {
      notify(err instanceof Error ? err.message : "No se pudo crear el dataset demo.", "error");
    }
  };

  const upload = async (event: FormEvent) => {
    event.preventDefault();
    if (!file) {
      notify("Seleccione un archivo CSV o XLSX.", "error");
      return;
    }
    setUploading(true);
    try {
      const body = new FormData();
      body.append("file", file);
      body.append("name", meta.name);
      body.append("source", meta.source);
      body.append("license", meta.license);
      body.append("description", meta.description);
      body.append("is_demo", String(meta.is_demo));
      await api.post("/datasets/upload", body);
      notify("Dataset cargado y validado.", "success");
      setUploadOpen(false);
      setFile(null);
      setMeta({ name: "", source: "Carga manual", license: "No especificada", description: "", is_demo: false });
      load();
    } catch (err) {
      notify(err instanceof Error ? err.message : "No se pudo cargar el dataset.", "error");
    } finally {
      setUploading(false);
    }
  };

  const quality = (detail?.quality ?? {}) as Record<string, any>;

  return (
    <div className="space-y-4">
      <PageHeader
        eyebrow="Datos"
        title="Datasets y calidad"
        description="Registro, validación y trazabilidad de las fuentes de datos. Cada carga calcula checksum, valores faltantes, duplicados y perfil de variables."
        actions={
          can("datasets", "write") ? (
            <button className="btn-accent" onClick={() => setUploadOpen(true)}>
              <UploadCloud className="h-4 w-4" />
              Cargar dataset
            </button>
          ) : undefined
        }
      />

      <DemoNotice>
        El dataset base es una cohorte <strong>sintética</strong> de 1.200 registros con semilla fija. No
        proviene de historias clínicas ni de encuestas reales.
      </DemoNotice>

      {error && <ErrorBanner message={error} onRetry={load} />}

      {loading ? (
        <PageLoading label="Cargando datasets…" />
      ) : datasets.length === 0 ? (
        <EmptyState
          title="Sin datasets registrados"
          message="Cargue un archivo CSV/XLSX con el esquema requerido o genere la cohorte sintética de demostración."
          action={
            <button className="btn-accent mt-2" onClick={ensureDemo}>
              <Database className="h-4 w-4" />
              Generar dataset demo
            </button>
          }
        />
      ) : (
        <div className="grid gap-4 lg:grid-cols-2 xl:grid-cols-3">
          {datasets.map((dataset) => {
            const q = (dataset.quality ?? {}) as Record<string, any>;
            return (
              <article key={dataset.id} className="card flex flex-col p-5">
                <div className="flex items-start justify-between gap-3">
                  <span className="flex h-10 w-10 items-center justify-center rounded-xl bg-navy-50 text-navy-700">
                    <Database className="h-5 w-5" />
                  </span>
                  <div className="flex flex-wrap justify-end gap-1.5">
                    {dataset.is_demo && <Badge tone="warning">Sintético</Badge>}
                    <Badge tone={dataset.status === "ready" ? "success" : "neutral"}>{dataset.status}</Badge>
                  </div>
                </div>
                <h2 className="mt-3 text-sm font-bold text-navy-900">{dataset.name}</h2>
                <p className="mt-1 line-clamp-2 min-h-8 text-xs leading-relaxed text-slate-500">
                  {dataset.description || "Sin descripción."}
                </p>
                <div className="mt-4 grid grid-cols-2 gap-2 text-xs">
                  <div className="rounded-lg bg-slate-50 p-2.5">
                    <p className="text-[10px] uppercase tracking-wide text-slate-400">Filas</p>
                    <p className="font-bold text-navy-900">{formatNumber(dataset.rows_count)}</p>
                  </div>
                  <div className="rounded-lg bg-slate-50 p-2.5">
                    <p className="text-[10px] uppercase tracking-wide text-slate-400">Columnas</p>
                    <p className="font-bold text-navy-900">{formatNumber(dataset.columns_count)}</p>
                  </div>
                  <div className="rounded-lg bg-slate-50 p-2.5">
                    <p className="text-[10px] uppercase tracking-wide text-slate-400">Faltantes</p>
                    <p className="font-bold text-navy-900">{Number(q.missing_pct ?? 0).toFixed(1)}%</p>
                  </div>
                  <div className="rounded-lg bg-slate-50 p-2.5">
                    <p className="text-[10px] uppercase tracking-wide text-slate-400">Duplicados</p>
                    <p className="font-bold text-navy-900">{formatNumber(Number(q.duplicate_rows ?? 0))}</p>
                  </div>
                </div>
                <p className="mt-3 truncate text-[10px] text-slate-400">
                  {dataset.source} · v{dataset.version} · {formatDate(dataset.created_at)}
                </p>
                <div className="mt-4 flex gap-2">
                  <button className="btn-ghost flex-1" onClick={() => loadDetail(dataset, "variables")}>
                    <Layers className="h-4 w-4" />
                    Explorar
                  </button>
                  {dataset.download_url && (
                    <button
                      className="btn-ghost"
                      title="Descargar CSV"
                      onClick={() => downloadFile(`/datasets/${dataset.id}/download`, `${dataset.name}.csv`)}
                    >
                      <Download className="h-4 w-4" />
                    </button>
                  )}
                </div>
              </article>
            );
          })}
        </div>
      )}

      <section className="card p-5">
        <div className="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
          <div>
            <h2 className="text-sm font-bold text-navy-900">Esquema esperado para cargas</h2>
            <p className="mt-1 text-xs text-slate-500">
              Se requieren variables clínicas, conductuales, ambientales y el desenlace de seguimiento. Consulte
              el diccionario completo en Documentación.
            </p>
          </div>
          {can("datasets", "write") && (
            <button className="btn-ghost" onClick={ensureDemo}>
              <Plus className="h-4 w-4" />
              Asegurar dataset demo
            </button>
          )}
        </div>
        <div className="mt-4 grid gap-2 sm:grid-cols-2 lg:grid-cols-4">
          {[
            "age, sex, region",
            "bmi, systolic_bp, diastolic_bp",
            "cholesterol_total, hdl, glucose",
            "smoking, physical_activity_min",
            "diabetes, hypertension, family_history",
            "pm25, green_space, stress_level",
            "event, followup_years",
            "cohort_year (opcional)",
          ].map((item) => (
            <div key={item} className="flex items-start gap-2 rounded-lg border border-slate-100 px-3 py-2 text-[11px] text-slate-600">
              <CheckCircle2 className="mt-0.5 h-3.5 w-3.5 shrink-0 text-emerald-600" />
              {item}
            </div>
          ))}
        </div>
      </section>

      <Modal
        open={detail !== null}
        onClose={() => setDetail(null)}
        title={detail?.name ?? "Dataset"}
        description={detail ? `${formatNumber(detail.rows_count)} filas · ${detail.columns_count} columnas · checksum ${detail.checksum?.slice(0, 12) ?? "—"}` : undefined}
        size="xl"
        footer={
          detail ? (
            <>
              {detail.download_url && (
                <button
                  className="btn-ghost"
                  onClick={() => downloadFile(`/datasets/${detail.id}/download`, `${detail.name}.csv`)}
                >
                  <Download className="h-4 w-4" />
                  Descargar CSV
                </button>
              )}
              <button className="btn-primary" onClick={() => setDetail(null)}>
                Cerrar
              </button>
            </>
          ) : undefined
        }
      >
        {detail && (
          <div className="space-y-4">
            <div className="flex flex-wrap gap-2">
              {(["variables", "quality", "preview"] as const).map((key) => (
                <button
                  key={key}
                  className={
                    tab === key
                      ? "rounded-lg bg-navy-900 px-3 py-1.5 text-xs font-semibold text-white"
                      : "rounded-lg border border-slate-200 px-3 py-1.5 text-xs font-semibold text-slate-600 hover:bg-slate-50"
                  }
                  onClick={() => switchTab(key)}
                >
                  {key === "variables" ? "Diccionario" : key === "quality" ? "Calidad" : "Vista previa"}
                </button>
              ))}
            </div>

            {detailLoading && (
              <div className="flex items-center gap-2 rounded-lg bg-slate-50 px-3 py-2 text-xs text-slate-500">
                <Spinner className="h-4 w-4" /> Cargando información…
              </div>
            )}

            {tab === "variables" && !detailLoading && (
              <div className="table-wrap rounded-lg border border-slate-100">
                <table className="data-table">
                  <thead>
                    <tr>
                      <th>Variable</th>
                      <th>Tipo</th>
                      <th>Faltantes</th>
                      <th>Rango</th>
                      <th>Media</th>
                      <th>Únicos</th>
                    </tr>
                  </thead>
                  <tbody>
                    {variables.map((variable) => (
                      <tr key={variable.name}>
                        <td>
                          <p className="font-mono text-[11px] font-semibold text-navy-900">{variable.name}</p>
                          <p className="text-[10px] text-slate-400">{variable.label}</p>
                        </td>
                        <td className="text-[11px]">{variable.dtype}</td>
                        <td className="text-[11px]">
                          {variable.missing_count} ({variable.missing_pct.toFixed(1)}%)
                        </td>
                        <td className="text-[11px]">
                          {variable.min === null ? "—" : `${variable.min.toFixed(1)} – ${variable.max?.toFixed(1)}`}
                        </td>
                        <td className="text-[11px]">{variable.mean === null ? "—" : variable.mean.toFixed(2)}</td>
                        <td className="text-[11px]">{variable.unique}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}

            {tab === "quality" && !detailLoading && (
              <div className="space-y-4">
                <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
                  {[
                    ["Filas", formatNumber(quality.rows)],
                    ["Columnas", formatNumber(quality.columns)],
                    ["Faltantes", `${Number(quality.missing_pct ?? 0).toFixed(2)}%`],
                    ["Duplicados", formatNumber(quality.duplicate_rows)],
                  ].map(([label, value]) => (
                    <div key={label} className="rounded-lg border border-slate-100 bg-slate-50/60 p-3">
                      <p className="text-[10px] font-semibold uppercase tracking-wide text-slate-400">{label}</p>
                      <p className="mt-0.5 text-lg font-bold text-navy-900">{value}</p>
                    </div>
                  ))}
                </div>
                {quality.target && (
                  <div className="rounded-lg border border-slate-100 p-4">
                    <p className="text-xs font-bold text-navy-900">Distribución del desenlace</p>
                    <p className="mt-1 text-sm text-slate-600">
                      Prevalencia observada: <strong>{(Number(quality.target.prevalence) * 100).toFixed(2)}%</strong>{" "}
                      ({formatNumber(quality.target.positives)} eventos / {formatNumber(quality.target.negatives)} sin
                      evento). Los eventos tardíos y las censuras se tratan en el motor de supervivencia.
                    </p>
                  </div>
                )}
                {quality.missing_by_column && Object.keys(quality.missing_by_column).length > 0 && (
                  <div className="table-wrap rounded-lg border border-slate-100">
                    <table className="data-table">
                      <thead>
                        <tr>
                          <th>Variable con faltantes</th>
                          <th>Cantidad</th>
                          <th>Porcentaje</th>
                        </tr>
                      </thead>
                      <tbody>
                        {Object.entries(quality.missing_by_column as Record<string, { count: number; pct: number }>).map(
                          ([name, value]) => (
                            <tr key={name}>
                              <td className="font-mono text-[11px]">{name}</td>
                              <td>{value.count}</td>
                              <td>{value.pct}%</td>
                            </tr>
                          )
                        )}
                      </tbody>
                    </table>
                  </div>
                )}
                <div className="rounded-lg bg-slate-50 p-4 text-xs leading-relaxed text-slate-600">
                  <FileSpreadsheet className="mr-1.5 inline h-3.5 w-3.5" />
                  Checksum SHA-256: <span className="break-all font-mono text-[10px]">{detail.checksum ?? "—"}</span>
                </div>
              </div>
            )}

            {tab === "preview" && !detailLoading && preview && (
              <div className="table-wrap rounded-lg border border-slate-100">
                <table className="data-table">
                  <thead>
                    <tr>
                      {preview.columns.slice(0, 14).map((column) => (
                        <th key={column}>{column}</th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {preview.rows.map((row, index) => (
                      <tr key={index}>
                        {preview.columns.slice(0, 14).map((column) => (
                          <td key={column} className="whitespace-nowrap text-[11px]">
                            {row[column] === null || row[column] === undefined ? "—" : String(row[column])}
                          </td>
                        ))}
                      </tr>
                    ))}
                  </tbody>
                </table>
                {preview.columns.length > 14 && (
                  <p className="px-3 py-2 text-[10px] text-slate-400">
                    Se muestran 14 de {preview.columns.length} columnas. Descargue el CSV para el detalle completo.
                  </p>
                )}
              </div>
            )}
          </div>
        )}
      </Modal>

      <Modal
        open={uploadOpen}
        onClose={() => setUploadOpen(false)}
        title="Cargar dataset"
        description="Formatos admitidos: CSV (UTF-8/Latin-1) y XLSX, hasta 60 MB y mínimo 50 registros."
        size="lg"
        footer={
          <>
            <button className="btn-ghost" onClick={() => setUploadOpen(false)}>
              Cancelar
            </button>
            <button className="btn-accent" form="upload-form" disabled={uploading}>
              {uploading ? "Validando…" : "Cargar y validar"}
            </button>
          </>
        }
      >
        <form id="upload-form" onSubmit={upload} className="space-y-4">
          <div className="rounded-xl border-2 border-dashed border-slate-300 bg-slate-50/60 px-4 py-8 text-center">
            <UploadCloud className="mx-auto h-8 w-8 text-slate-400" />
            <p className="mt-2 text-sm font-semibold text-navy-900">
              {file ? file.name : "Seleccione un archivo CSV o XLSX"}
            </p>
            <p className="mt-0.5 text-xs text-slate-500">
              {file ? `${(file.size / 1024 / 1024).toFixed(2)} MB` : "El esquema se valida antes de guardar."}
            </p>
            <input
              type="file"
              accept=".csv,.xlsx"
              className="mt-3 block w-full text-xs text-slate-500 file:mr-3 file:rounded-lg file:border-0 file:bg-navy-900 file:px-3 file:py-2 file:text-xs file:font-semibold file:text-white hover:file:bg-navy-800"
              onChange={(event) => {
                const selected = event.target.files?.[0] ?? null;
                setFile(selected);
                if (selected && !meta.name) {
                  setMeta((current) => ({ ...current, name: selected.name.replace(/\.(csv|xlsx)$/i, "") }));
                }
              }}
            />
          </div>
          <div className="grid gap-3 sm:grid-cols-2">
            <Field label="Nombre del dataset">
              <input className="input" value={meta.name} onChange={(e) => setMeta({ ...meta, name: e.target.value })} required />
            </Field>
            <Field label="Fuente">
              <input className="input" value={meta.source} onChange={(e) => setMeta({ ...meta, source: e.target.value })} />
            </Field>
            <Field label="Licencia">
              <input className="input" value={meta.license} onChange={(e) => setMeta({ ...meta, license: e.target.value })} />
            </Field>
            <Field label="¿Es de demostración?">
              <select className="input" value={meta.is_demo ? "true" : "false"} onChange={(e) => setMeta({ ...meta, is_demo: e.target.value === "true" })}>
                <option value="false">No</option>
                <option value="true">Sí (sintético)</option>
              </select>
            </Field>
          </div>
          <Field label="Descripción" hint="Documente procedencia, periodo y cualquier limitación conocida.">
            <textarea className="input min-h-20" value={meta.description} onChange={(e) => setMeta({ ...meta, description: e.target.value })} />
          </Field>
          <div className="rounded-lg border border-amber-200 bg-amber-50 px-3 py-2.5 text-[11px] leading-relaxed text-amber-900">
            No cargue datos personales identificables. Use seudónimos y conserve la llave de reidentificación fuera
            de la plataforma.
          </div>
        </form>
      </Modal>
    </div>
  );
}

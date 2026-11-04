import { useCallback, useEffect, useState } from "react";
import {
  Download,
  FileBarChart,
  FileSpreadsheet,
  FileText,
  Layers,
  Loader2,
  Plus,
} from "lucide-react";
import { api, downloadFile } from "../lib/api";
import { formatDateTime } from "../lib/format";
import { useAuth } from "../context/AuthContext";
import { useToast } from "../context/ToastContext";
import {
  Badge,
  DemoNotice,
  EmptyState,
  ErrorBanner,
  Field,
  PageHeader,
  PageLoading,
  Spinner,
} from "../components/ui";

interface ReportItem {
  id: number;
  report_type: string;
  title: string;
  format: string;
  filename: string;
  filters: Record<string, unknown>;
  size_bytes: number;
  created_at: string;
  download_url: string;
}

interface ReportTypes {
  types: { key: string; label: string }[];
  formats: { key: string; mime: string }[];
}

const FORMAT_ICON: Record<string, typeof FileText> = {
  pdf: FileText,
  docx: FileText,
  xlsx: FileSpreadsheet,
};

export default function ReportsPage() {
  const { can } = useAuth();
  const { notify } = useToast();
  const [items, setItems] = useState<ReportItem[]>([]);
  const [types, setTypes] = useState<ReportTypes | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [reportType, setReportType] = useState("operational");
  const [format, setFormat] = useState("pdf");
  const [days, setDays] = useState(180);
  const [generating, setGenerating] = useState(false);

  const load = useCallback(async () => {
    setLoading(true);
    setError("");
    try {
      const [catalog, reports] = await Promise.all([
        api.get<ReportTypes>("/reports/types"),
        api.get<ReportItem[]>("/reports"),
      ]);
      setTypes(catalog);
      setItems(reports);
    } catch (err) {
      setError(err instanceof Error ? err.message : "No se pudieron cargar los reportes.");
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    load();
  }, [load]);

  const generate = async () => {
    setGenerating(true);
    try {
      const result = await api.post<ReportItem>("/reports", { report_type: reportType, format, days });
      notify(`Reporte generado: ${result.filename}`, "success");
      const reports = await api.get<ReportItem[]>("/reports");
      setItems(reports);
    } catch (err) {
      notify(err instanceof Error ? err.message : "No se pudo generar el reporte.", "error");
    } finally {
      setGenerating(false);
    }
  };

  return (
    <div className="space-y-4">
      <PageHeader
        eyebrow="Documentación"
        title="Reportes"
        description="Generación de documentos PDF, DOCX y XLSX con indicadores, rendimiento de modelos, explicabilidad, auditoría y calidad de datos."
      />

      <DemoNotice>
        Los reportes incluyen una advertencia de uso responsable y la etiqueta de demostración. No deben
        presentarse como resultados clínicos validados.
      </DemoNotice>

      {error && <ErrorBanner message={error} onRetry={load} />}

      <div className="grid gap-4 xl:grid-cols-[0.85fr_1.15fr]">
        <section className="card p-5">
          <div className="flex items-center gap-2">
            <Plus className="h-4 w-4 text-accent-600" />
            <h2 className="text-sm font-bold text-navy-900">Nuevo reporte</h2>
          </div>
          <p className="mt-1 text-xs leading-relaxed text-slate-500">
            Seleccione el tipo de documento, el formato de salida y el periodo de análisis. El archivo queda
            registrado en la bitácora.
          </p>
          <div className="mt-4 space-y-3">
            <Field label="Tipo de reporte">
              <select className="input" value={reportType} onChange={(event) => setReportType(event.target.value)}>
                {(types?.types ?? []).map((item) => (
                  <option key={item.key} value={item.key}>
                    {item.label}
                  </option>
                ))}
              </select>
            </Field>
            <Field label="Formato">
              <div className="grid grid-cols-3 gap-2">
                {["pdf", "docx", "xlsx"].map((item) => {
                  const Icon = FORMAT_ICON[item] ?? FileText;
                  return (
                    <button
                      key={item}
                      type="button"
                      onClick={() => setFormat(item)}
                      className={
                        format === item
                          ? "flex flex-col items-center gap-1 rounded-lg border-2 border-accent-500 bg-accent-50 px-2 py-3 text-xs font-bold uppercase text-accent-700"
                          : "flex flex-col items-center gap-1 rounded-lg border border-slate-200 px-2 py-3 text-xs font-bold uppercase text-slate-500 hover:bg-slate-50"
                      }
                    >
                      <Icon className="h-5 w-5" />
                      {item}
                    </button>
                  );
                })}
              </div>
            </Field>
            <Field label="Periodo de análisis">
              <select className="input" value={days} onChange={(event) => setDays(Number(event.target.value))}>
                <option value={30}>Últimos 30 días</option>
                <option value={90}>Últimos 90 días</option>
                <option value={180}>Últimos 180 días</option>
                <option value={365}>Último año</option>
                <option value={730}>Últimos 2 años</option>
              </select>
            </Field>
            <button className="btn-accent w-full" onClick={generate} disabled={generating || !can("reports", "write")}>
              {generating ? <Spinner className="h-4 w-4 text-white" /> : <FileBarChart className="h-4 w-4" />}
              {generating ? "Generando documento…" : "Generar reporte"}
            </button>
            {!can("reports", "write") && (
              <p className="text-[10px] text-amber-700">
                Su rol puede consultar y descargar reportes, pero no generarlos.
              </p>
            )}
          </div>
        </section>

        <section className="card">
          <div className="card-header">
            <div>
              <h2 className="text-sm font-bold text-navy-900">Reportes generados</h2>
              <p className="text-xs text-slate-500">Historial reciente de documentos de la plataforma.</p>
            </div>
            <Badge tone="info">{items.length} documentos</Badge>
          </div>
          {loading ? (
            <PageLoading label="Cargando reportes…" />
          ) : items.length === 0 ? (
            <div className="p-5">
              <EmptyState
                title="Sin reportes"
                message="Genere el primer documento con el formulario de la izquierda. Los archivos se almacenan localmente en data/reports."
              />
            </div>
          ) : (
            <div className="table-wrap">
              <table className="data-table">
                <thead>
                  <tr>
                    <th>Documento</th>
                    <th>Tipo</th>
                    <th>Formato</th>
                    <th>Tamaño</th>
                    <th>Generado</th>
                    <th className="text-right">Acción</th>
                  </tr>
                </thead>
                <tbody>
                  {items.map((item) => {
                    const Icon = FORMAT_ICON[item.format] ?? FileText;
                    return (
                      <tr key={item.id}>
                        <td>
                          <div className="flex items-center gap-2">
                            <span className="flex h-8 w-8 items-center justify-center rounded-lg bg-navy-50 text-navy-700">
                              <Icon className="h-4 w-4" />
                            </span>
                            <div className="min-w-0">
                              <p className="max-w-[220px] truncate text-xs font-semibold text-navy-900">{item.title}</p>
                              <p className="max-w-[220px] truncate font-mono text-[10px] text-slate-400">{item.filename}</p>
                            </div>
                          </div>
                        </td>
                        <td className="text-[11px]">{item.report_type}</td>
                        <td><Badge tone="neutral">{item.format.toUpperCase()}</Badge></td>
                        <td className="text-[11px]">{(item.size_bytes / 1024).toFixed(1)} KB</td>
                        <td className="text-[11px]">{formatDateTime(item.created_at)}</td>
                        <td>
                          <button
                            className="btn-ghost ml-auto px-2 py-1 text-[11px]"
                            onClick={() => downloadFile(item.download_url.replace("/api", ""), item.filename)}
                          >
                            <Download className="h-3.5 w-3.5" />
                            Descargar
                          </button>
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          )}
        </section>
      </div>

      <div className="grid gap-3 md:grid-cols-3">
        {[
          ["Trazabilidad", "Cada descarga y generación se registra en la bitácora append-only con usuario, IP y fecha."],
          ["Contenido", "Los reportes incluyen metadatos, advertencia de uso, tablas y —en PDF— una gráfica de distribución de riesgo."],
          ["Privacidad", "Los documentos usan agregados y códigos de registro; no incluyen nombres ni identificadores personales."],
        ].map(([title, text]) => (
          <div key={title} className="card p-4">
            <div className="flex items-center gap-2">
              <Layers className="h-4 w-4 text-accent-600" />
              <p className="text-xs font-bold text-navy-900">{title}</p>
            </div>
            <p className="mt-1 text-[11px] leading-relaxed text-slate-500">{text}</p>
          </div>
        ))}
      </div>
    </div>
  );
}

import { useCallback, useEffect, useState } from "react";
import { Download, Filter, Lock, RefreshCw, Search, ShieldCheck } from "lucide-react";
import { api, buildQuery, downloadFile } from "../lib/api";
import type { AuditEntry } from "../lib/types";
import { formatDateTime } from "../lib/format";
import { Badge, DemoNotice, ErrorBanner, PageHeader, PageLoading, Pagination } from "../components/ui";

interface Stats {
  days: number;
  total: number;
  by_module: { module: string; count: number }[];
  by_result: { result: string; count: number }[];
  by_action: { action: string; count: number }[];
}

export default function AuditPage() {
  const [items, setItems] = useState<AuditEntry[]>([]);
  const [total, setTotal] = useState(0);
  const [page, setPage] = useState(1);
  const [pageSize] = useState(50);
  const [q, setQ] = useState("");
  const [module, setModule] = useState("");
  const [result, setResult] = useState("");
  const [days, setDays] = useState<number | "">("");
  const [stats, setStats] = useState<Stats | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");

  const load = useCallback(async () => {
    setLoading(true);
    setError("");
    try {
      const data = await api.get<{ items: AuditEntry[]; total: number }>(
        `/audit${buildQuery({ q, module, result, days, page, page_size: pageSize })}`
      );
      setItems(data.items);
      setTotal(data.total);
    } catch (err) {
      setError(err instanceof Error ? err.message : "No se pudo cargar la bitácora.");
    } finally {
      setLoading(false);
    }
  }, [q, module, result, days, page, pageSize]);

  const loadStats = useCallback(async () => {
    try {
      setStats(await api.get<Stats>("/audit/stats?days=30"));
    } catch {
      /* indicadores no bloqueantes */
    }
  }, []);

  useEffect(() => {
    load();
  }, [load]);

  useEffect(() => {
    loadStats();
  }, [loadStats]);

  const refresh = () => {
    load();
    loadStats();
  };

  return (
    <div className="space-y-4">
      <PageHeader
        eyebrow="Gobernanza"
        title="Bitácora de auditoría"
        description="Registro append-only de accesos, cambios, predicciones, descargas y eventos de seguridad. No existe endpoint para editar o eliminar entradas."
        actions={
          <>
            <button className="btn-ghost" onClick={refresh}>
              <RefreshCw className="h-4 w-4" />
              Actualizar
            </button>
            <button
              className="btn-ghost"
              onClick={() =>
                downloadFile(
                  `/audit/export.csv${buildQuery({ q, module, result, days })}`,
                  "bitacora_cardioai.csv"
                )
              }
            >
              <Download className="h-4 w-4" />
              Exportar CSV
            </button>
          </>
        }
      />

      <div className="flex items-start gap-2.5 rounded-xl border border-emerald-200 bg-emerald-50 px-4 py-3 text-xs leading-relaxed text-emerald-900">
        <Lock className="mt-0.5 h-4 w-4 shrink-0" />
        <p>
          La bitácora es de solo inserción. Cada entrada incluye usuario, acción, módulo, entidad, resultado, IP y
          sesión. La exportación también queda auditada.
        </p>
      </div>

      {stats && (
        <div className="grid gap-3 sm:grid-cols-2 xl:grid-cols-4">
          <div className="card p-4">
            <p className="text-[11px] font-semibold uppercase tracking-wider text-slate-500">Eventos (30 días)</p>
            <p className="mt-1 text-2xl font-bold text-navy-900">{stats.total}</p>
          </div>
          <div className="card p-4">
            <p className="text-[11px] font-semibold uppercase tracking-wider text-slate-500">Módulo más activo</p>
            <p className="mt-1 truncate text-sm font-bold text-navy-900">
              {stats.by_module.sort((a, b) => b.count - a.count)[0]?.module ?? "—"}
            </p>
            <p className="text-[10px] text-slate-400">
              {stats.by_module[0]?.count ?? 0} eventos
            </p>
          </div>
          <div className="card p-4">
            <p className="text-[11px] font-semibold uppercase tracking-wider text-slate-500">Resultado</p>
            <div className="mt-1 flex flex-wrap gap-1">
              {stats.by_result.map((item) => (
                <Badge key={item.result} tone={item.result === "success" ? "success" : "danger"}>
                  {item.result}: {item.count}
                </Badge>
              ))}
            </div>
          </div>
          <div className="card p-4">
            <p className="text-[11px] font-semibold uppercase tracking-wider text-slate-500">Acción frecuente</p>
            <p className="mt-1 truncate text-sm font-bold text-navy-900">{stats.by_action[0]?.action ?? "—"}</p>
            <p className="text-[10px] text-slate-400">{stats.by_action[0]?.count ?? 0} registros</p>
          </div>
        </div>
      )}

      <section className="card p-4">
        <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-5">
          <label className="block lg:col-span-2">
            <span className="label">Buscar</span>
            <div className="relative">
              <Search className="pointer-events-none absolute left-3 top-2.5 h-4 w-4 text-slate-400" />
              <input
                className="input pl-9"
                placeholder="Acción, detalle, usuario, entidad…"
                value={q}
                onChange={(event) => {
                  setQ(event.target.value);
                  setPage(1);
                }}
              />
            </div>
          </label>
          <label className="block">
            <span className="label">Módulo</span>
            <select
              className="input"
              value={module}
              onChange={(event) => {
                setModule(event.target.value);
                setPage(1);
              }}
            >
              <option value="">Todos</option>
              {["auth", "dashboard", "patients", "datasets", "models", "predictions", "geo", "reports", "audit", "users", "assistant", "alerts"].map((item) => (
                <option key={item} value={item}>
                  {item}
                </option>
              ))}
            </select>
          </label>
          <label className="block">
            <span className="label">Resultado</span>
            <select
              className="input"
              value={result}
              onChange={(event) => {
                setResult(event.target.value);
                setPage(1);
              }}
            >
              <option value="">Todos</option>
              <option value="success">Éxito</option>
              <option value="error">Error</option>
            </select>
          </label>
          <label className="block">
            <span className="label">Periodo</span>
            <select
              className="input"
              value={days}
              onChange={(event) => {
                setDays(event.target.value === "" ? "" : Number(event.target.value));
                setPage(1);
              }}
            >
              <option value="">Todo</option>
              <option value={7}>7 días</option>
              <option value={30}>30 días</option>
              <option value={90}>90 días</option>
              <option value={365}>1 año</option>
            </select>
          </label>
        </div>
        <div className="mt-3 flex justify-end">
          <button className="btn-ghost" onClick={refresh}>
            <Filter className="h-4 w-4" />
            Aplicar filtros
          </button>
        </div>
      </section>

      {error && <ErrorBanner message={error} onRetry={load} />}

      <section className="card">
        <div className="card-header">
          <div>
            <h2 className="text-sm font-bold text-navy-900">Eventos registrados</h2>
            <p className="text-xs text-slate-500">{total} entradas · orden descendente por fecha</p>
          </div>
          <Badge tone="success">
            <ShieldCheck className="h-3 w-3" /> Append-only
          </Badge>
        </div>
        {loading ? (
          <PageLoading label="Cargando bitácora…" />
        ) : (
          <>
            <div className="table-wrap">
              <table className="data-table">
                <thead>
                  <tr>
                    <th>Fecha (UTC)</th>
                    <th>Usuario</th>
                    <th>Acción</th>
                    <th>Módulo</th>
                    <th>Entidad</th>
                    <th>Resultado</th>
                    <th>IP</th>
                    <th>Detalle</th>
                  </tr>
                </thead>
                <tbody>
                  {items.length === 0 && (
                    <tr>
                      <td colSpan={8} className="py-8 text-center text-sm text-slate-500">
                        No hay eventos con los filtros seleccionados.
                      </td>
                    </tr>
                  )}
                  {items.map((entry) => (
                    <tr key={entry.id}>
                      <td className="whitespace-nowrap text-[11px]">{formatDateTime(entry.created_at)}</td>
                      <td className="text-[11px]">{entry.user_email}</td>
                      <td className="font-medium">{entry.action}</td>
                      <td><Badge tone="neutral">{entry.module}</Badge></td>
                      <td className="text-[11px]">
                        {entry.entity ?? "—"} {entry.entity_id ? `#${entry.entity_id}` : ""}
                      </td>
                      <td>
                        <Badge tone={entry.result === "success" ? "success" : "danger"}>{entry.result}</Badge>
                      </td>
                      <td className="font-mono text-[10px] text-slate-500">{entry.ip ?? "—"}</td>
                      <td className="max-w-[280px]">
                        <p className="line-clamp-2 text-[11px] text-slate-500">{entry.detail ?? "—"}</p>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            <Pagination page={page} pageSize={pageSize} total={total} onChange={setPage} />
          </>
        )}
      </section>

      <DemoNotice>
        La bitácora registra actividad de la plataforma, no evidencia clínica. Los datos de la cohorte son
        sintéticos y las acciones deben interpretarse en el contexto de una demostración técnica.
      </DemoNotice>
    </div>
  );
}

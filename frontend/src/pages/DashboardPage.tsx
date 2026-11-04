import { useCallback, useEffect, useState } from "react";
import { Link } from "react-router-dom";
import {
  Activity,
  AlertTriangle,
  ArrowUpRight,
  BrainCircuit,
  Database,
  Gauge,
  HeartPulse,
  RefreshCw,
  ShieldCheck,
  Users,
} from "lucide-react";
import {
  Area,
  AreaChart,
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  Pie,
  PieChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { api, buildQuery } from "../lib/api";
import type { DashboardData } from "../lib/types";
import { formatDate, formatNumber, formatPct, relativeTime, RISK_LABEL, riskColor } from "../lib/format";
import {
  Badge,
  DemoNotice,
  ErrorBanner,
  KpiCard,
  PageHeader,
  PageLoading,
  RiskBadge,
} from "../components/ui";

const RISK_FILL: Record<string, string> = { bajo: "#2f855a", moderado: "#b7791f", alto: "#c53030" };

export default function DashboardPage() {
  const [data, setData] = useState<DashboardData | null>(null);
  const [days, setDays] = useState(180);
  const [region, setRegion] = useState("");
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(true);

  const load = useCallback(async () => {
    setError("");
    try {
      const payload = await api.get<DashboardData>(`/dashboard/summary${buildQuery({ days, region })}`);
      setData(payload);
    } catch (err) {
      setError(err instanceof Error ? err.message : "No se pudo cargar el dashboard.");
    } finally {
      setLoading(false);
    }
  }, [days, region]);

  useEffect(() => {
    setLoading(true);
    load();
  }, [load]);

  if (loading && !data) return <PageLoading label="Calculando indicadores del dashboard…" />;
  if (!data) return <ErrorBanner message={error || "Sin datos disponibles."} onRetry={load} />;

  const kpis = data.kpis;
  const regions = data.regions.slice(0, 8);
  const maxRegionRisk = Math.max(0.35, ...regions.map((r) => r.mean_risk ?? 0));

  return (
    <div className="space-y-5">
      <PageHeader
        eyebrow="Panel de control"
        title="Dashboard operacional"
        description="Estado de la cohorte, actividad de predicción, calidad de datos y rendimiento de los modelos. Todos los indicadores provienen del entorno de demostración."
        actions={
          <>
            <select className="input w-auto" value={region} onChange={(event) => setRegion(event.target.value)}>
              <option value="">Todas las regiones</option>
              {data.regions.map((item) => (
                <option key={item.region} value={item.region}>
                  {item.region}
                </option>
              ))}
            </select>
            <select className="input w-auto" value={days} onChange={(event) => setDays(Number(event.target.value))}>
              <option value={30}>30 días</option>
              <option value={90}>90 días</option>
              <option value={180}>180 días</option>
              <option value={365}>365 días</option>
            </select>
            <button onClick={load} className="btn-ghost" title="Actualizar">
              <RefreshCw className="h-4 w-4" />
              Actualizar
            </button>
          </>
        }
      />

      <DemoNotice>{data.disclaimer}</DemoNotice>
      {error && <ErrorBanner message={error} onRetry={load} />}

      <div className="grid grid-cols-1 gap-3 sm:grid-cols-2 xl:grid-cols-4">
        <KpiCard
          label="Pacientes activos"
          value={formatNumber(kpis.patients)}
          hint="Registros disponibles en la cohorte"
          icon={<Users className="h-4 w-4" />}
        />
        <KpiCard
          label="Predicciones del periodo"
          value={formatNumber(kpis.predictions_window)}
          hint={`${formatNumber(kpis.predictions)} acumuladas · ${formatPct(kpis.mean_risk)} riesgo medio`}
          icon={<Activity className="h-4 w-4" />}
        />
        <KpiCard
          label="Riesgo alto"
          value={formatPct(kpis.high_risk_pct)}
          hint={`${formatNumber(kpis.high_risk_window)} casos sobre umbral de demostración`}
          icon={<HeartPulse className="h-4 w-4" />}
          tone={kpis.high_risk_pct > 20 ? "danger" : "warning"}
        />
        <KpiCard
          label="Modelos listos"
          value={formatNumber(kpis.models_ready)}
          hint={`Calidad de datos: ${kpis.quality_score === null ? "—" : `${kpis.quality_score}/100`}`}
          icon={<BrainCircuit className="h-4 w-4" />}
          tone="success"
        />
      </div>

      <div className="grid gap-4 xl:grid-cols-3">
        <section className="card xl:col-span-2">
          <div className="card-header">
            <div>
              <h2 className="text-sm font-bold text-navy-900">Tendencia de predicciones</h2>
              <p className="text-xs text-slate-500">Volumen mensual y proporción de riesgo alto.</p>
            </div>
            <Badge tone="info">Últimos 12 meses</Badge>
          </div>
          <div className="h-64 px-3 py-4">
            <ResponsiveContainer width="100%" height="100%">
              <AreaChart data={data.trend} margin={{ top: 5, right: 12, left: -14, bottom: 0 }}>
                <defs>
                  <linearGradient id="totalGradient" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor="#2577eb" stopOpacity={0.28} />
                    <stop offset="95%" stopColor="#2577eb" stopOpacity={0} />
                  </linearGradient>
                </defs>
                <CartesianGrid strokeDasharray="3 3" stroke="#e6edf6" vertical={false} />
                <XAxis dataKey="period" tick={{ fontSize: 10, fill: "#64748b" }} tickLine={false} axisLine={false} />
                <YAxis tick={{ fontSize: 10, fill: "#64748b" }} tickLine={false} axisLine={false} allowDecimals={false} />
                <Tooltip contentStyle={{ fontSize: 12, borderRadius: 8 }} />
                <Area type="monotone" dataKey="total" name="Predicciones" stroke="#2577eb" strokeWidth={2} fill="url(#totalGradient)" />
                <Area type="monotone" dataKey="alto" name="Riesgo alto" stroke="#c53030" strokeWidth={2} fill="transparent" />
              </AreaChart>
            </ResponsiveContainer>
          </div>
        </section>

        <section className="card">
          <div className="card-header">
            <div>
              <h2 className="text-sm font-bold text-navy-900">Estratificación</h2>
              <p className="text-xs text-slate-500">Distribución por nivel de riesgo.</p>
            </div>
          </div>
          <div className="h-52 px-2 pt-3">
            <ResponsiveContainer width="100%" height="100%">
              <PieChart>
                <Pie
                  data={data.risk_distribution}
                  dataKey="count"
                  nameKey="level"
                  innerRadius={46}
                  outerRadius={72}
                  paddingAngle={3}
                >
                  {data.risk_distribution.map((entry) => (
                    <Cell key={entry.level} fill={RISK_FILL[entry.level] || "#94a3b8"} />
                  ))}
                </Pie>
                <Tooltip
                  formatter={(value: number, name: string) => [value, RISK_LABEL[name] || name]}
                  contentStyle={{ fontSize: 12, borderRadius: 8 }}
                />
              </PieChart>
            </ResponsiveContainer>
          </div>
          <div className="space-y-1.5 px-5 pb-4">
            {data.risk_distribution.map((item) => (
              <div key={item.level} className="flex items-center justify-between text-xs">
                <span className="flex items-center gap-2 text-slate-600">
                  <span className="h-2 w-2 rounded-full" style={{ background: RISK_FILL[item.level] }} />
                  {RISK_LABEL[item.level] || item.level}
                </span>
                <span className="font-semibold text-navy-900">{formatNumber(item.count)}</span>
              </div>
            ))}
          </div>
        </section>
      </div>

      <div className="grid gap-4 xl:grid-cols-3">
        <section className="card">
          <div className="card-header">
            <div>
              <h2 className="text-sm font-bold text-navy-900">Perfil etario</h2>
              <p className="text-xs text-slate-500">Pacientes activos por grupo de edad.</p>
            </div>
          </div>
          <div className="h-56 px-3 py-3">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={data.age_distribution} margin={{ top: 4, right: 8, left: -20, bottom: 0 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#e6edf6" vertical={false} />
                <XAxis dataKey="bucket" tick={{ fontSize: 9, fill: "#64748b" }} tickLine={false} axisLine={false} />
                <YAxis tick={{ fontSize: 10, fill: "#64748b" }} tickLine={false} axisLine={false} allowDecimals={false} />
                <Tooltip contentStyle={{ fontSize: 12, borderRadius: 8 }} />
                <Bar dataKey="count" name="Pacientes" fill="#4777aa" radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </section>

        <section className="card">
          <div className="card-header">
            <div>
              <h2 className="text-sm font-bold text-navy-900">Riesgo medio por región</h2>
              <p className="text-xs text-slate-500">Predicciones del periodo seleccionado.</p>
            </div>
            <Link to="/geography" className="link text-[11px]">
              Ver mapa
            </Link>
          </div>
          <div className="space-y-2.5 px-5 py-4">
            {regions.map((item) => (
              <div key={item.region}>
                <div className="flex items-center justify-between text-[11px]">
                  <span className="truncate font-medium text-slate-600">{item.region}</span>
                  <span className="ml-2 shrink-0 font-semibold text-navy-900">
                    {item.mean_risk === null ? "Sin datos" : formatPct(item.mean_risk)}
                  </span>
                </div>
                <div className="mt-1 h-1.5 overflow-hidden rounded-full bg-slate-100">
                  <div
                    className="h-full rounded-full bg-accent-500"
                    style={{ width: `${Math.min(100, ((item.mean_risk ?? 0) / maxRegionRisk) * 100)}%` }}
                  />
                </div>
              </div>
            ))}
          </div>
        </section>

        <section className="card">
          <div className="card-header">
            <div>
              <h2 className="text-sm font-bold text-navy-900">Modelo activo</h2>
              <p className="text-xs text-slate-500">Selección automática por AUC temporal.</p>
            </div>
          </div>
          {data.active_model ? (
            <div className="px-5 py-5">
              <div className="flex items-center gap-3">
                <span className="flex h-11 w-11 items-center justify-center rounded-xl bg-accent-50 text-accent-700">
                  <BrainCircuit className="h-5 w-5" />
                </span>
                <div className="min-w-0">
                  <p className="truncate text-sm font-bold text-navy-900">{data.active_model.name}</p>
                  <p className="text-[11px] text-slate-500">{data.active_model.algorithm}</p>
                </div>
              </div>
              <div className="mt-5 grid grid-cols-2 gap-3">
                <div className="rounded-lg bg-slate-50 p-3">
                  <p className="text-[10px] font-semibold uppercase tracking-wide text-slate-400">AUC-ROC temporal</p>
                  <p className="mt-0.5 text-xl font-bold text-navy-900">
                    {data.active_model.roc_auc === null ? "—" : data.active_model.roc_auc.toFixed(3)}
                  </p>
                </div>
                <div className="rounded-lg bg-slate-50 p-3">
                  <p className="text-[10px] font-semibold uppercase tracking-wide text-slate-400">Entrenado</p>
                  <p className="mt-0.5 text-xs font-semibold text-navy-900">{formatDate(data.active_model.trained_at)}</p>
                </div>
              </div>
              <div className="mt-4 flex flex-wrap gap-2">
                {data.active_model.is_demo && <Badge tone="warning">Datos sintéticos</Badge>}
                <Badge tone="success">Activo</Badge>
              </div>
              <Link to="/prediction" className="btn-accent mt-5 w-full">
                <Gauge className="h-4 w-4" />
                Nueva predicción
              </Link>
            </div>
          ) : (
            <div className="px-5 py-8 text-center">
              <AlertTriangle className="mx-auto h-7 w-7 text-amber-500" />
              <p className="mt-2 text-sm font-semibold text-navy-900">Sin modelo activo</p>
              <p className="mt-1 text-xs text-slate-500">Entrene los cinco modelos para habilitar la inferencia.</p>
              <Link to="/models" className="btn-primary mt-4">
                Ir a modelos
              </Link>
            </div>
          )}
        </section>
      </div>

      <div className="grid gap-4 xl:grid-cols-3">
        <section className="card xl:col-span-2">
          <div className="card-header">
            <div>
              <h2 className="text-sm font-bold text-navy-900">Comparación de modelos</h2>
              <p className="text-xs text-slate-500">Mejor versión lista por algoritmo, ordenada por AUC temporal.</p>
            </div>
            <Link to="/models" className="link text-[11px]">
              Gestionar modelos
            </Link>
          </div>
          <div className="table-wrap">
            <table className="data-table">
              <thead>
                <tr>
                  <th>Modelo</th>
                  <th>AUC-ROC</th>
                  <th>F1</th>
                  <th>MCC</th>
                  <th>Entrenamiento</th>
                  <th>Estado</th>
                </tr>
              </thead>
              <tbody>
                {data.model_comparison.length === 0 && (
                  <tr>
                    <td colSpan={6} className="py-8 text-center text-sm text-slate-500">
                      Aún no hay modelos entrenados.
                    </td>
                  </tr>
                )}
                {data.model_comparison.map((model) => (
                  <tr key={model.id}>
                    <td>
                      <p className="font-semibold text-navy-900">{model.name}</p>
                      <p className="text-[11px] text-slate-400">{model.algorithm}</p>
                    </td>
                    <td className="font-semibold">{model.roc_auc?.toFixed(3) ?? "—"}</td>
                    <td>{model.f1?.toFixed(3) ?? "—"}</td>
                    <td>{model.mcc?.toFixed(3) ?? "—"}</td>
                    <td>{formatNumber(model.train_rows)} filas</td>
                    <td>{model.is_active ? <Badge tone="success">Activo</Badge> : <Badge>Listo</Badge>}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </section>

        <section className="card">
          <div className="card-header">
            <div>
              <h2 className="text-sm font-bold text-navy-900">Alertas recientes</h2>
              <p className="text-xs text-slate-500">Eventos del sistema y de predicción.</p>
            </div>
          </div>
          <div className="divide-y divide-slate-50">
            {data.alerts.length === 0 && <p className="px-5 py-6 text-sm text-slate-500">Sin alertas pendientes.</p>}
            {data.alerts.map((alert) => (
              <div key={alert.id} className="px-5 py-3">
                <div className="flex items-center justify-between gap-2">
                  <span className="text-[10px] font-semibold uppercase tracking-wide text-slate-400">{alert.type}</span>
                  <span className="text-[10px] text-slate-400">{relativeTime(alert.created_at)}</span>
                </div>
                <p className="mt-1 text-xs font-semibold text-navy-900">{alert.title}</p>
                <p className="mt-0.5 line-clamp-2 text-[11px] leading-snug text-slate-500">{alert.message}</p>
              </div>
            ))}
          </div>
        </section>
      </div>

      <div className="grid gap-4 xl:grid-cols-2">
        <section className="card">
          <div className="card-header">
            <div>
              <h2 className="text-sm font-bold text-navy-900">Actividad reciente</h2>
              <p className="text-xs text-slate-500">Últimos eventos registrados en la bitácora.</p>
            </div>
            <ShieldCheck className="h-4 w-4 text-emerald-600" />
          </div>
          <div className="divide-y divide-slate-50">
            {data.recent_activity.length === 0 && (
              <p className="px-5 py-6 text-sm text-slate-500">Sin actividad registrada.</p>
            )}
            {data.recent_activity.map((entry) => (
              <div key={entry.id} className="flex items-start justify-between gap-3 px-5 py-2.5">
                <div className="min-w-0">
                  <p className="truncate text-xs font-semibold text-navy-900">{entry.action}</p>
                  <p className="truncate text-[11px] text-slate-500">
                    {entry.user_email} · {entry.module}
                  </p>
                </div>
                <span className="shrink-0 text-[10px] text-slate-400">{relativeTime(entry.created_at)}</span>
              </div>
            ))}
          </div>
        </section>

        <section className="card">
          <div className="card-header">
            <div>
              <h2 className="text-sm font-bold text-navy-900">Estado de la plataforma</h2>
              <p className="text-xs text-slate-500">Resumen de capacidades y datos disponibles.</p>
            </div>
          </div>
          <div className="grid grid-cols-2 gap-3 p-5">
            <div className="rounded-lg border border-slate-100 p-3">
              <Database className="h-4 w-4 text-accent-600" />
              <p className="mt-2 text-xs font-semibold text-navy-900">{formatNumber(kpis.datasets)} datasets</p>
              <p className="text-[11px] text-slate-500">Incluye cohorte sintética base</p>
            </div>
            <div className="rounded-lg border border-slate-100 p-3">
              <ShieldCheck className="h-4 w-4 text-emerald-600" />
              <p className="mt-2 text-xs font-semibold text-navy-900">Auditoría activa</p>
              <p className="text-[11px] text-slate-500">Bitácora append-only</p>
            </div>
            <div className="rounded-lg border border-slate-100 p-3">
              <AlertTriangle className="h-4 w-4 text-amber-600" />
              <p className="mt-2 text-xs font-semibold text-navy-900">
                {formatNumber(kpis.unread_alerts)} alertas
              </p>
              <p className="text-[11px] text-slate-500">Sin leer</p>
            </div>
            <div className="rounded-lg border border-slate-100 p-3">
              <ArrowUpRight className="h-4 w-4 text-navy-600" />
              <p className="mt-2 text-xs font-semibold text-navy-900">{formatPct(kpis.mean_risk)}</p>
              <p className="text-[11px] text-slate-500">Riesgo medio del periodo</p>
            </div>
          </div>
        </section>
      </div>
    </div>
  );
}

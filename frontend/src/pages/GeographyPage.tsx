import { useCallback, useEffect, useState } from "react";
import { CircleMarker, MapContainer, Popup, TileLayer, Tooltip as LeafletTooltip } from "react-leaflet";
import "leaflet/dist/leaflet.css";
import { Globe2, Info, MapPinned, RefreshCw, ShieldCheck, Wind } from "lucide-react";
import { Bar, BarChart, CartesianGrid, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { api } from "../lib/api";
import { formatNumber, formatPct } from "../lib/format";
import { Badge, DemoNotice, ErrorBanner, PageHeader, PageLoading } from "../components/ui";

interface RegionRow {
  region: string;
  lat: number;
  lon: number;
  patients: number;
  predictions: number;
  mean_risk: number | null;
  high_risk: number;
  pm25: number;
  pm25_source: string;
  green_space: number | null;
}

interface GeoResponse {
  regions: RegionRow[];
  days: number;
  environmental_note: string;
  privacy_note: string;
  generated_at: string;
}

function riskFill(risk: number | null): string {
  if (risk === null) return "#94a3b8";
  if (risk >= 0.25) return "#c53030";
  if (risk >= 0.1) return "#b7791f";
  return "#2f855a";
}

export default function GeographyPage() {
  const [data, setData] = useState<GeoResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [days, setDays] = useState(365);

  const load = useCallback(async () => {
    setLoading(true);
    setError("");
    try {
      setData(await api.get<GeoResponse>(`/geo/regions?days=${days}`));
    } catch (err) {
      setError(err instanceof Error ? err.message : "No se pudo cargar el análisis territorial.");
    } finally {
      setLoading(false);
    }
  }, [days]);

  useEffect(() => {
    load();
  }, [load]);

  if (loading && !data) return <PageLoading label="Cargando análisis territorial…" />;
  if (!data) return <ErrorBanner message={error || "Sin datos territoriales."} onRetry={load} />;

  const maxPatients = Math.max(1, ...data.regions.map((region) => region.patients));
  const ranked = [...data.regions].sort((a, b) => (b.mean_risk ?? -1) - (a.mean_risk ?? -1));

  return (
    <div className="space-y-4">
      <PageHeader
        eyebrow="Salud territorial"
        title="Análisis geoespacial"
        description="Distribución agregada de la cohorte y del riesgo estimado por región. No se exponen coordenadas de personas ni identificadores individuales."
        actions={
          <>
            <select className="input w-auto" value={days} onChange={(event) => setDays(Number(event.target.value))}>
              <option value={90}>90 días</option>
              <option value={180}>180 días</option>
              <option value={365}>365 días</option>
            </select>
            <button className="btn-ghost" onClick={load}>
              <RefreshCw className="h-4 w-4" />
              Actualizar
            </button>
          </>
        }
      />

      <DemoNotice>
        Las coordenadas son centroides regionales aproximados y los valores de PM2.5 son ilustrativos del modo
        demo. No provienen de estaciones de monitoreo ni de datos oficiales.
      </DemoNotice>
      {error && <ErrorBanner message={error} onRetry={load} />}

      <div className="grid gap-4 xl:grid-cols-[1.4fr_1fr]">
        <section className="card overflow-hidden">
          <div className="card-header">
            <div>
              <h2 className="text-sm font-bold text-navy-900">Mapa de riesgo medio</h2>
              <p className="text-xs text-slate-500">El tamaño del círculo representa pacientes; el color, el riesgo medio.</p>
            </div>
            <Globe2 className="h-4 w-4 text-slate-400" />
          </div>
          <div className="h-[440px] w-full">
            <MapContainer center={[4.6, -74.1]} zoom={5} scrollWheelZoom className="h-full w-full">
              <TileLayer
                attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a>'
                url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
              />
              {data.regions.map((region) => (
                <CircleMarker
                  key={region.region}
                  center={[region.lat, region.lon]}
                  radius={8 + Math.sqrt(region.patients) * 2.2}
                  pathOptions={{
                    color: riskFill(region.mean_risk),
                    fillColor: riskFill(region.mean_risk),
                    fillOpacity: 0.45,
                    weight: 2,
                  }}
                >
                  <Popup>
                    <div className="min-w-[180px] text-xs">
                      <p className="font-bold text-navy-900">{region.region}</p>
                      <p className="mt-1">Pacientes: {region.patients}</p>
                      <p>Predicciones: {region.predictions}</p>
                      <p>Riesgo medio: {region.mean_risk === null ? "Sin datos" : formatPct(region.mean_risk)}</p>
                      <p>Riesgo alto: {region.high_risk}</p>
                      <p>PM2.5: {region.pm25} µg/m³</p>
                    </div>
                  </Popup>
                  <LeafletTooltip direction="top" offset={[0, -4]} opacity={0.95}>
                    {region.region}
                  </LeafletTooltip>
                </CircleMarker>
              ))}
            </MapContainer>
          </div>
          <div className="flex flex-wrap items-center gap-4 border-t border-slate-100 px-5 py-3 text-[11px] text-slate-500">
            <span className="flex items-center gap-1.5"><span className="h-2.5 w-2.5 rounded-full bg-emerald-600" /> Riesgo bajo</span>
            <span className="flex items-center gap-1.5"><span className="h-2.5 w-2.5 rounded-full bg-amber-600" /> Moderado</span>
            <span className="flex items-center gap-1.5"><span className="h-2.5 w-2.5 rounded-full bg-red-600" /> Alto</span>
            <span className="flex items-center gap-1.5"><span className="h-2.5 w-2.5 rounded-full bg-slate-400" /> Sin predicciones</span>
          </div>
        </section>

        <div className="space-y-4">
          <section className="card p-5">
            <div className="flex items-center gap-2">
              <MapPinned className="h-4 w-4 text-accent-600" />
              <h2 className="text-sm font-bold text-navy-900">Indicadores territoriales</h2>
            </div>
            <div className="mt-4 space-y-3">
              {ranked.slice(0, 6).map((region, index) => (
                <div key={region.region} className="flex items-center gap-3">
                  <span className="w-4 text-[10px] font-bold text-slate-400">{index + 1}</span>
                  <div className="min-w-0 flex-1">
                    <div className="flex items-center justify-between gap-2 text-[11px]">
                      <span className="truncate font-medium text-slate-700">{region.region}</span>
                      <span className="shrink-0 font-semibold text-navy-900">
                        {region.mean_risk === null ? "—" : formatPct(region.mean_risk)}
                      </span>
                    </div>
                    <div className="mt-1 h-1.5 overflow-hidden rounded-full bg-slate-100">
                      <div
                        className="h-full rounded-full"
                        style={{
                          width: `${Math.min(100, ((region.mean_risk ?? 0) / 0.4) * 100)}%`,
                          background: riskFill(region.mean_risk),
                        }}
                      />
                    </div>
                  </div>
                  <span className="shrink-0 text-[10px] text-slate-400">{formatNumber(region.patients)} pac.</span>
                </div>
              ))}
            </div>
          </section>

          <section className="card p-5">
            <div className="flex items-center gap-2">
              <Wind className="h-4 w-4 text-accent-600" />
              <h2 className="text-sm font-bold text-navy-900">PM2.5 y riesgo medio</h2>
            </div>
            <div className="mt-3 h-44">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart
                  data={data.regions.map((region) => ({
                    region: region.region.slice(0, 9),
                    pm25: region.pm25,
                    risk: (region.mean_risk ?? 0) * 100,
                  }))}
                  margin={{ top: 4, right: 4, left: -22, bottom: 0 }}
                >
                  <CartesianGrid strokeDasharray="3 3" stroke="#e6edf6" vertical={false} />
                  <XAxis dataKey="region" tick={{ fontSize: 8 }} tickLine={false} axisLine={false} interval={0} angle={-25} height={42} textAnchor="end" />
                  <YAxis tick={{ fontSize: 9 }} tickLine={false} axisLine={false} />
                  <Tooltip contentStyle={{ fontSize: 11, borderRadius: 8 }} />
                  <Bar dataKey="pm25" name="PM2.5 µg/m³" fill="#6a94c2" radius={[3, 3, 0, 0]} />
                  <Bar dataKey="risk" name="Riesgo medio (%)" fill="#c53030" radius={[3, 3, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </div>
          </section>

          <div className="rounded-xl border border-sky-200 bg-sky-50 p-3.5 text-[11px] leading-relaxed text-sky-900">
            <Info className="mr-1 inline h-3.5 w-3.5" />
            {data.environmental_note}
          </div>
          <div className="rounded-xl border border-emerald-200 bg-emerald-50 p-3.5 text-[11px] leading-relaxed text-emerald-900">
            <ShieldCheck className="mr-1 inline h-3.5 w-3.5" />
            {data.privacy_note}
          </div>
        </div>
      </div>

      <section className="card">
        <div className="card-header">
          <div>
            <h2 className="text-sm font-bold text-navy-900">Detalle por región</h2>
            <p className="text-xs text-slate-500">Agregados de pacientes activos y predicciones del periodo.</p>
          </div>
          <Badge tone="info">{data.regions.length} regiones</Badge>
        </div>
        <div className="table-wrap">
          <table className="data-table">
            <thead>
              <tr>
                <th>Región</th>
                <th>Pacientes</th>
                <th>Predicciones</th>
                <th>Riesgo medio</th>
                <th>Riesgo alto</th>
                <th>PM2.5 (µg/m³)</th>
                <th>Espacio verde</th>
                <th>Fuente PM2.5</th>
              </tr>
            </thead>
            <tbody>
              {data.regions.map((region) => (
                <tr key={region.region}>
                  <td className="font-medium text-navy-900">{region.region}</td>
                  <td>{formatNumber(region.patients)}</td>
                  <td>{formatNumber(region.predictions)}</td>
                  <td className="font-semibold">{region.mean_risk === null ? "—" : formatPct(region.mean_risk)}</td>
                  <td>{region.high_risk}</td>
                  <td>{region.pm25}</td>
                  <td>{region.green_space === null ? "—" : `${region.green_space}%`}</td>
                  <td className="text-[10px] text-slate-400">{region.pm25_source}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>
    </div>
  );
}

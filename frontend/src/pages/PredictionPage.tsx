import { useCallback, useEffect, useMemo, useState, type FormEvent } from "react";
import { Link, useSearchParams } from "react-router-dom";
import {
  Activity,
  AlertTriangle,
  ArrowRight,
  BarChart3,
  BrainCircuit,
  CheckCircle2,
  FileUp,
  Gauge,
  Info,
  Sparkles,
  UploadCloud,
} from "lucide-react";
import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  Line,
  LineChart,
  RadialBar,
  RadialBarChart,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { api, buildQuery } from "../lib/api";
import type { ModelRun, Patient, Prediction } from "../lib/types";
import { formatDate, formatPct, RISK_LABEL, riskColor } from "../lib/format";
import { useAuth } from "../context/AuthContext";
import { useToast } from "../context/ToastContext";
import {
  Badge,
  DemoNotice,
  ErrorBanner,
  Field,
  Modal,
  PageHeader,
  PageLoading,
  RiskBadge,
  Spinner,
} from "../components/ui";

interface FormState {
  age: number;
  sex: string;
  region: string;
  urban: boolean;
  education: string;
  income_level: string;
  smoking: string;
  physical_activity_min: number;
  alcohol_units: number;
  bmi: number;
  systolic_bp: number;
  diastolic_bp: number;
  cholesterol_total: number;
  hdl: number;
  glucose: number;
  diabetes: boolean;
  hypertension: boolean;
  family_history: boolean;
  pm25: number;
  green_space: number;
  stress_level: number;
  patient_id: number | null;
  model_run_id: number | null;
}

const DEFAULT_FORM: FormState = {
  age: 58,
  sex: "F",
  region: "Bogotá D.C.",
  urban: true,
  education: "secundaria",
  income_level: "medio",
  smoking: "nunca",
  physical_activity_min: 150,
  alcohol_units: 2,
  bmi: 27,
  systolic_bp: 128,
  diastolic_bp: 80,
  cholesterol_total: 200,
  hdl: 50,
  glucose: 100,
  diabetes: false,
  hypertension: false,
  family_history: false,
  pm25: 14,
  green_space: 18,
  stress_level: 5,
  patient_id: null,
  model_run_id: null,
};

export default function PredictionPage() {
  const { can } = useAuth();
  const { notify } = useToast();
  const [searchParams] = useSearchParams();
  const [tab, setTab] = useState<"individual" | "batch" | "history">("individual");
  const [models, setModels] = useState<ModelRun[]>([]);
  const [patients, setPatients] = useState<Patient[]>([]);
  const [form, setForm] = useState<FormState>(DEFAULT_FORM);
  const [submitting, setSubmitting] = useState(false);
  const [result, setResult] = useState<Prediction | null>(null);
  const [history, setHistory] = useState<Prediction[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");

  const [batchFile, setBatchFile] = useState<File | null>(null);
  const [batchModel, setBatchModel] = useState<string>("");
  const [batchBusy, setBatchBusy] = useState(false);
  const [batchResult, setBatchResult] = useState<any>(null);
  const [patientDetailOpen, setPatientDetailOpen] = useState(false);

  const readyModels = useMemo(() => models.filter((model) => model.status === "ready"), [models]);
  const activeModel = useMemo(() => readyModels.find((model) => model.is_active) ?? readyModels[0], [readyModels]);

  const load = useCallback(async () => {
    setLoading(true);
    setError("");
    try {
      const [modelList, patientList, historyData] = await Promise.all([
        api.get<{ items: ModelRun[] }>("/models" + buildQuery({ page_size: 60 })),
        api.get<{ items: Patient[] }>("/patients" + buildQuery({ page_size: 200 })),
        api.get<{ items: Prediction[] }>("/predictions" + buildQuery({ page_size: 12 })),
      ]);
      setModels(modelList.items);
      setPatients(patientList.items);
      setHistory(historyData.items);
      const firstReady = modelList.items.find((model) => model.status === "ready" && model.is_active)
        ?? modelList.items.find((model) => model.status === "ready");
      setForm((current) => ({
        ...current,
        model_run_id: current.model_run_id ?? firstReady?.id ?? null,
      }));
      setBatchModel((current) => current || String(firstReady?.id ?? ""));
    } catch (err) {
      setError(err instanceof Error ? err.message : "No se pudo preparar el módulo de predicción.");
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    load();
  }, [load]);

  useEffect(() => {
    const patientId = Number(searchParams.get("patient_id"));
    if (!patientId) return;
    api
      .get<Patient>(`/patients/${patientId}`)
      .then((patient) => {
        setForm((current) => ({
          ...current,
          patient_id: patient.id,
          age: patient.age,
          sex: patient.sex,
          region: patient.region,
          urban: patient.urban,
          education: patient.education,
          income_level: patient.income_level,
          smoking: patient.smoking,
          physical_activity_min: patient.physical_activity_min,
          alcohol_units: patient.alcohol_units,
          bmi: patient.bmi,
          systolic_bp: patient.systolic_bp,
          diastolic_bp: patient.diastolic_bp,
          cholesterol_total: patient.cholesterol_total,
          hdl: patient.hdl,
          glucose: patient.glucose,
          diabetes: patient.diabetes,
          hypertension: patient.hypertension,
          family_history: patient.family_history,
          pm25: patient.pm25,
          green_space: patient.green_space,
          stress_level: patient.stress_level,
        }));
        setTab("individual");
      })
      .catch(() => undefined);
  }, [searchParams]);

  const selectPatient = (value: string) => {
    const id = value ? Number(value) : null;
    if (!id) {
      setForm((current) => ({ ...current, patient_id: null }));
      return;
    }
    const patient = patients.find((item) => item.id === id);
    if (!patient) return;
    setForm((current) => ({
      ...current,
      patient_id: patient.id,
      age: patient.age,
      sex: patient.sex,
      region: patient.region,
      urban: patient.urban,
      education: patient.education,
      income_level: patient.income_level,
      smoking: patient.smoking,
      physical_activity_min: patient.physical_activity_min,
      alcohol_units: patient.alcohol_units,
      bmi: patient.bmi,
      systolic_bp: patient.systolic_bp,
      diastolic_bp: patient.diastolic_bp,
      cholesterol_total: patient.cholesterol_total,
      hdl: patient.hdl,
      glucose: patient.glucose,
      diabetes: patient.diabetes,
      hypertension: patient.hypertension,
      family_history: patient.family_history,
      pm25: patient.pm25,
      green_space: patient.green_space,
      stress_level: patient.stress_level,
    }));
  };

  const update = (key: keyof FormState, value: string | number | boolean | null) =>
    setForm((current) => ({ ...current, [key]: value }));

  const submit = async (event: FormEvent) => {
    event.preventDefault();
    if (!form.model_run_id) {
      notify("Seleccione un modelo listo para inferencia.", "error");
      return;
    }
    setSubmitting(true);
    setError("");
    try {
      const prediction = await api.post<Prediction>("/predictions", {
        ...form,
        persist: true,
        explain: true,
      });
      setResult(prediction);
      notify("Predicción calculada y registrada.", "success");
      const updated = await api.get<{ items: Prediction[] }>("/predictions" + buildQuery({ page_size: 12 }));
      setHistory(updated.items);
    } catch (err) {
      setError(err instanceof Error ? err.message : "No se pudo calcular la predicción.");
    } finally {
      setSubmitting(false);
    }
  };

  const submitBatch = async (event: FormEvent) => {
    event.preventDefault();
    if (!batchFile) {
      notify("Seleccione un archivo CSV.", "error");
      return;
    }
    setBatchBusy(true);
    try {
      const body = new FormData();
      body.append("file", batchFile);
      if (batchModel) body.append("model_run_id", batchModel);
      const data = await api.post<any>("/predictions/batch", body);
      setBatchResult(data);
      notify(`Lote procesado: ${data.total} registros.`, "success");
    } catch (err) {
      notify(err instanceof Error ? err.message : "No se pudo procesar el lote.", "error");
    } finally {
      setBatchBusy(false);
    }
  };

  if (loading) return <PageLoading label="Preparando motor de inferencia…" />;

  const noModels = readyModels.length === 0;
  const factors = result?.explanation?.factors ?? [];
  const factorData = factors.map((factor) => ({
    name: factor.label,
    contribution: factor.contribution,
    fill: factor.contribution >= 0 ? "#c53030" : "#2f855a",
  }));
  const gaugeValue = result ? Math.round(result.probability * 1000) / 10 : 0;

  return (
    <div className="space-y-4">
      <PageHeader
        eyebrow="Inferencia"
        title="Predicción y estratificación"
        description="Estimación individual o por lote con el modelo activo, umbral operativo documentado y explicación local de las contribuciones."
        actions={
          activeModel ? (
            <Badge tone="success">
              <BrainCircuit className="h-3 w-3" /> {activeModel.name}
            </Badge>
          ) : (
            <Badge tone="warning">Sin modelo activo</Badge>
          )
        }
      />

      <DemoNotice>
        Resultado de investigación sobre datos sintéticos. No es diagnóstico, no prescribe tratamiento y no
        sustituye la valoración de un profesional de la salud.
      </DemoNotice>

      {error && <ErrorBanner message={error} onRetry={load} />}

      {noModels ? (
        <div className="card flex flex-col items-center px-6 py-14 text-center">
          <span className="flex h-14 w-14 items-center justify-center rounded-2xl bg-amber-50 text-amber-600">
            <AlertTriangle className="h-7 w-7" />
          </span>
          <h2 className="mt-3 text-lg font-bold text-navy-900">No hay modelos listos</h2>
          <p className="mt-1 max-w-md text-sm text-slate-500">
            Entrene los modelos desde el módulo de Machine Learning. Al finalizar, el mejor AUC temporal se activa
            automáticamente y podrá generar predicciones.
          </p>
          <Link to="/models" className="btn-accent mt-5">
            Ir a modelos <ArrowRight className="h-4 w-4" />
          </Link>
        </div>
      ) : (
        <>
          <div className="flex flex-wrap gap-1.5">
            {([
              ["individual", "Predicción individual"],
              ["batch", "Predicción por lote"],
              ["history", "Historial reciente"],
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

          {tab === "individual" && (
            <div className="grid gap-4 xl:grid-cols-[1.05fr_0.95fr]">
              <form onSubmit={submit} className="card p-5">
                <div className="mb-4 flex items-start justify-between gap-3">
                  <div>
                    <h2 className="text-sm font-bold text-navy-900">Datos del registro</h2>
                    <p className="text-xs text-slate-500">Puede partir de un paciente existente o ingresar valores.</p>
                  </div>
                  {form.patient_id && (
                    <button type="button" className="link text-[11px]" onClick={() => setPatientDetailOpen(true)}>
                      Ver ficha
                    </button>
                  )}
                </div>

                <div className="space-y-4">
                  <div className="grid gap-3 sm:grid-cols-2">
                    <Field label="Paciente de la cohorte (opcional)">
                      <select className="input" value={form.patient_id ?? ""} onChange={(event) => selectPatient(event.target.value)}>
                        <option value="">Ingreso manual</option>
                        {patients.map((patient) => (
                          <option key={patient.id} value={patient.id}>
                            {patient.code} · {patient.age} años · {patient.region}
                          </option>
                        ))}
                      </select>
                    </Field>
                    <Field label="Modelo para inferencia">
                      <select
                        className="input"
                        value={form.model_run_id ?? ""}
                        onChange={(event) => update("model_run_id", event.target.value ? Number(event.target.value) : null)}
                        required
                      >
                        {readyModels.map((model) => (
                          <option key={model.id} value={model.id}>
                            {model.name} {model.is_active ? "· activo" : ""}
                          </option>
                        ))}
                      </select>
                    </Field>
                  </div>

                  <div className="grid gap-3 sm:grid-cols-3">
                    <Field label="Edad (años)">
                      <input type="number" className="input" value={form.age} onChange={(e) => update("age", Number(e.target.value))} required />
                    </Field>
                    <Field label="Sexo">
                      <select className="input" value={form.sex} onChange={(e) => update("sex", e.target.value)}>
                        <option value="F">F</option>
                        <option value="M">M</option>
                        <option value="Otro">Otro</option>
                      </select>
                    </Field>
                    <Field label="Región">
                      <select className="input" value={form.region} onChange={(e) => update("region", e.target.value)}>
                        {["Bogotá D.C.", "Antioquia", "Valle del Cauca", "Atlántico", "Santander", "Cundinamarca", "Bolívar", "Nariño", "Tolima", "Cauca", "Meta", "La Guajira"].map((region) => (
                          <option key={region} value={region}>
                            {region}
                          </option>
                        ))}
                      </select>
                    </Field>
                    <Field label="IMC (kg/m²)">
                      <input type="number" step="0.1" className="input" value={form.bmi} onChange={(e) => update("bmi", Number(e.target.value))} required />
                    </Field>
                    <Field label="Presión sistólica">
                      <input type="number" className="input" value={form.systolic_bp} onChange={(e) => update("systolic_bp", Number(e.target.value))} required />
                    </Field>
                    <Field label="Presión diastólica">
                      <input type="number" className="input" value={form.diastolic_bp} onChange={(e) => update("diastolic_bp", Number(e.target.value))} />
                    </Field>
                    <Field label="Colesterol total">
                      <input type="number" step="0.1" className="input" value={form.cholesterol_total} onChange={(e) => update("cholesterol_total", Number(e.target.value))} required />
                    </Field>
                    <Field label="HDL">
                      <input type="number" step="0.1" className="input" value={form.hdl} onChange={(e) => update("hdl", Number(e.target.value))} />
                    </Field>
                    <Field label="Glucosa">
                      <input type="number" step="0.1" className="input" value={form.glucose} onChange={(e) => update("glucose", Number(e.target.value))} required />
                    </Field>
                    <Field label="Tabaquismo">
                      <select className="input" value={form.smoking} onChange={(e) => update("smoking", e.target.value)}>
                        <option value="nunca">Nunca</option>
                        <option value="exfumador">Exfumador</option>
                        <option value="fumador">Fumador</option>
                      </select>
                    </Field>
                    <Field label="Actividad física (min/sem)">
                      <input type="number" className="input" value={form.physical_activity_min} onChange={(e) => update("physical_activity_min", Number(e.target.value))} />
                    </Field>
                    <Field label="Alcohol (unidades/sem)">
                      <input type="number" step="0.1" className="input" value={form.alcohol_units} onChange={(e) => update("alcohol_units", Number(e.target.value))} />
                    </Field>
                    <Field label="PM2.5 (µg/m³)">
                      <input type="number" step="0.1" className="input" value={form.pm25} onChange={(e) => update("pm25", Number(e.target.value))} />
                    </Field>
                    <Field label="Espacio verde (%)">
                      <input type="number" step="0.1" className="input" value={form.green_space} onChange={(e) => update("green_space", Number(e.target.value))} />
                    </Field>
                    <Field label="Estrés (1-10)">
                      <input type="number" className="input" value={form.stress_level} onChange={(e) => update("stress_level", Number(e.target.value))} />
                    </Field>
                  </div>

                  <div className="flex flex-wrap gap-5 rounded-lg bg-slate-50 px-4 py-3 text-sm text-slate-700">
                    <label className="flex items-center gap-2">
                      <input type="checkbox" checked={form.diabetes} onChange={(e) => update("diabetes", e.target.checked)} />
                      Diabetes
                    </label>
                    <label className="flex items-center gap-2">
                      <input type="checkbox" checked={form.hypertension} onChange={(e) => update("hypertension", e.target.checked)} />
                      Hipertensión
                    </label>
                    <label className="flex items-center gap-2">
                      <input type="checkbox" checked={form.family_history} onChange={(e) => update("family_history", e.target.checked)} />
                      Antecedente familiar
                    </label>
                    <label className="flex items-center gap-2">
                      <input type="checkbox" checked={form.urban} onChange={(e) => update("urban", e.target.checked)} />
                      Residencia urbana
                    </label>
                  </div>

                  <div className="flex flex-wrap items-center justify-between gap-2 border-t border-slate-100 pt-4">
                    <p className="text-[10px] leading-relaxed text-slate-400">
                      El resultado depende del modelo, del horizonte y de la calidad de los datos.
                    </p>
                    <button className="btn-accent" type="submit" disabled={submitting || !can("predictions", "write")}>
                      {submitting ? <Spinner className="h-4 w-4 text-white" /> : <Sparkles className="h-4 w-4" />}
                      {submitting ? "Calculando…" : "Calcular riesgo"}
                    </button>
                  </div>
                </div>
              </form>

              <div className="space-y-4">
                {result ? (
                  <>
                    <section className="card overflow-hidden">
                      <div className="border-b border-slate-100 bg-navy-900 px-5 py-4 text-white">
                        <div className="flex items-center justify-between gap-2">
                          <div>
                            <p className="text-[10px] font-semibold uppercase tracking-[0.16em] text-navy-300">
                              Resultado de la estimación
                            </p>
                            <p className="mt-0.5 text-xs text-navy-200">{result.model?.name ?? "Modelo"}</p>
                          </div>
                          <RiskBadge level={result.risk_level} />
                        </div>
                      </div>
                      <div className="grid gap-2 p-5 sm:grid-cols-[180px_1fr] sm:items-center">
                        <div className="relative h-40">
                          <ResponsiveContainer width="100%" height="100%">
                            <RadialBarChart
                              data={[{ name: "Riesgo", value: gaugeValue, fill: riskColor(result.risk_level) }]}
                              innerRadius="72%"
                              outerRadius="100%"
                              startAngle={210}
                              endAngle={-30}
                            >
                              <RadialBar dataKey="value" cornerRadius={8} background={{ fill: "#eef3f9" }} />
                            </RadialBarChart>
                          </ResponsiveContainer>
                          <div className="pointer-events-none absolute inset-0 flex flex-col items-center justify-center pt-4">
                            <p className="text-3xl font-bold text-navy-900">{formatPct(result.probability)}</p>
                            <p className="text-[10px] font-semibold uppercase tracking-wide text-slate-400">
                              a {result.horizon_years} años
                            </p>
                          </div>
                        </div>
                        <div className="space-y-2 text-xs">
                          <div className="flex justify-between rounded-lg bg-slate-50 px-3 py-2">
                            <span className="text-slate-500">Nivel</span>
                            <span className="font-bold text-navy-900">{RISK_LABEL[result.risk_level]}</span>
                          </div>
                          <div className="flex justify-between rounded-lg bg-slate-50 px-3 py-2">
                            <span className="text-slate-500">Umbral operativo</span>
                            <span className="font-semibold">{result.threshold === null ? "—" : formatPct(result.threshold)}</span>
                          </div>
                          <div className="flex justify-between rounded-lg bg-slate-50 px-3 py-2">
                            <span className="text-slate-500">Método de explicación</span>
                            <span className="font-semibold">{result.explanation?.method ?? "—"}</span>
                          </div>
                          <div className="flex justify-between rounded-lg bg-slate-50 px-3 py-2">
                            <span className="text-slate-500">Registro</span>
                            <span className="font-semibold">
                              {result.patient_id ? `#${result.patient_id}` : "Sin vincular"}
                            </span>
                          </div>
                        </div>
                      </div>
                      <div className="mx-5 mb-4 flex items-start gap-2 rounded-lg border border-amber-200 bg-amber-50 px-3 py-2 text-[10px] leading-relaxed text-amber-900">
                        <Info className="mt-0.5 h-3.5 w-3.5 shrink-0" />
                        {result.disclaimer}
                      </div>
                    </section>

                    <section className="card">
                      <div className="card-header">
                        <div>
                          <h2 className="text-sm font-bold text-navy-900">Explicación local</h2>
                          <p className="text-xs text-slate-500">
                            Contribuciones sobre la probabilidad estimada; no implican causalidad.
                          </p>
                        </div>
                        <Badge tone="info">{result.explanation?.method ?? "—"}</Badge>
                      </div>
                      {factorData.length > 0 ? (
                        <>
                          <div className="h-56 px-2 pt-2">
                            <ResponsiveContainer width="100%" height="100%">
                              <BarChart data={factorData} margin={{ top: 4, right: 12, left: 4, bottom: 0 }}>
                                <CartesianGrid strokeDasharray="3 3" stroke="#e6edf6" vertical={false} />
                                <XAxis dataKey="name" tick={{ fontSize: 8 }} interval={0} angle={-18} height={50} textAnchor="end" tickLine={false} axisLine={false} />
                                <YAxis tick={{ fontSize: 9 }} tickLine={false} axisLine={false} />
                                <Tooltip contentStyle={{ fontSize: 11, borderRadius: 8 }} />
                                <ReferenceLine y={0} stroke="#94a3b8" />
                                <Bar dataKey="contribution" name="Contribución" radius={[3, 3, 0, 0]}>
                                  {factorData.map((entry) => (
                                    <Cell key={entry.name} fill={entry.fill} />
                                  ))}
                                </Bar>
                              </BarChart>
                            </ResponsiveContainer>
                          </div>
                          <div className="table-wrap border-t border-slate-100">
                            <table className="data-table">
                              <thead>
                                <tr>
                                  <th>Variable</th>
                                  <th>Valor</th>
                                  <th>Contribución</th>
                                  <th>Dirección</th>
                                </tr>
                              </thead>
                              <tbody>
                                {factors.map((factor) => (
                                  <tr key={factor.feature}>
                                    <td className="font-medium">{factor.label}</td>
                                    <td>{factor.value === null || factor.value === undefined ? "—" : String(factor.value)}</td>
                                    <td className={factor.contribution >= 0 ? "font-semibold text-red-600" : "font-semibold text-emerald-600"}>
                                      {factor.contribution >= 0 ? "+" : ""}
                                      {factor.contribution.toFixed(4)}
                                    </td>
                                    <td className="capitalize">{factor.direction}</td>
                                  </tr>
                                ))}
                              </tbody>
                            </table>
                          </div>
                        </>
                      ) : (
                        <p className="px-5 py-6 text-sm text-slate-500">
                          Explicación no disponible para este modelo o registro.
                          {result.explanation?.error && (
                            <span className="mt-1 block text-[10px] text-slate-400">{result.explanation.error}</span>
                          )}
                        </p>
                      )}
                    </section>

                    {result.explanation?.attention && result.explanation.attention.length > 0 && (
                      <section className="card">
                        <div className="card-header">
                          <div>
                            <h2 className="text-sm font-bold text-navy-900">Atención temporal</h2>
                            <p className="text-xs text-slate-500">Peso asignado a cada visita por la red LSTM.</p>
                          </div>
                        </div>
                        <div className="h-44 px-3 py-3">
                          <ResponsiveContainer width="100%" height="100%">
                            <BarChart data={result.explanation.attention} margin={{ top: 4, right: 8, left: -20, bottom: 0 }}>
                              <CartesianGrid strokeDasharray="3 3" stroke="#e6edf6" vertical={false} />
                              <XAxis dataKey="visit" tickFormatter={(value) => `V${value}`} tick={{ fontSize: 9 }} tickLine={false} axisLine={false} />
                              <YAxis tick={{ fontSize: 9 }} tickLine={false} axisLine={false} />
                              <Tooltip contentStyle={{ fontSize: 11, borderRadius: 8 }} />
                              <Bar dataKey="weight" name="Peso de atención" fill="#3b96f6" radius={[3, 3, 0, 0]} />
                            </BarChart>
                          </ResponsiveContainer>
                        </div>
                      </section>
                    )}
                  </>
                ) : (
                  <section className="card flex min-h-[420px] flex-col items-center justify-center px-6 py-10 text-center">
                    <span className="flex h-14 w-14 items-center justify-center rounded-2xl bg-accent-50 text-accent-700">
                      <Gauge className="h-7 w-7" />
                    </span>
                    <h2 className="mt-3 text-sm font-bold text-navy-900">Esperando datos para estimar</h2>
                    <p className="mt-1 max-w-xs text-xs leading-relaxed text-slate-500">
                      Complete el formulario y calcule el riesgo. El resultado incluirá probabilidad, nivel,
                      umbral y explicación local.
                    </p>
                    <div className="mt-4 rounded-lg bg-slate-50 px-3 py-2 text-[10px] text-slate-500">
                      Umbrales demo: bajo &lt; {formatPct(activeModel?.risk_thresholds?.low_max ?? 0.1)} · alto ≥{" "}
                      {formatPct(activeModel?.risk_thresholds?.high_min ?? 0.25)}
                    </div>
                  </section>
                )}
              </div>
            </div>
          )}

          {tab === "batch" && (
            <div className="grid gap-4 xl:grid-cols-[0.95fr_1.05fr]">
              <form onSubmit={submitBatch} className="card p-5">
                <h2 className="text-sm font-bold text-navy-900">Predicción por lote</h2>
                <p className="mt-1 text-xs text-slate-500">
                  Cargue un CSV con al menos las columnas age, sex, region, bmi, systolic_bp, cholesterol_total y
                  glucose. Máximo 1.000 filas.
                </p>
                <div className="mt-4 space-y-3">
                  <Field label="Modelo">
                    <select className="input" value={batchModel} onChange={(event) => setBatchModel(event.target.value)}>
                      {readyModels.map((model) => (
                        <option key={model.id} value={model.id}>
                          {model.name} {model.is_active ? "· activo" : ""}
                        </option>
                      ))}
                    </select>
                  </Field>
                  <div className="rounded-xl border-2 border-dashed border-slate-300 bg-slate-50/60 px-4 py-8 text-center">
                    <UploadCloud className="mx-auto h-8 w-8 text-slate-400" />
                    <p className="mt-2 text-sm font-semibold text-navy-900">
                      {batchFile ? batchFile.name : "Seleccione el archivo CSV"}
                    </p>
                    <input
                      type="file"
                      accept=".csv"
                      className="mt-3 block w-full text-xs text-slate-500 file:mr-3 file:rounded-lg file:border-0 file:bg-navy-900 file:px-3 file:py-2 file:text-xs file:font-semibold file:text-white"
                      onChange={(event) => setBatchFile(event.target.files?.[0] ?? null)}
                    />
                  </div>
                  <button className="btn-accent w-full" type="submit" disabled={batchBusy}>
                    {batchBusy ? <Spinner className="h-4 w-4 text-white" /> : <FileUp className="h-4 w-4" />}
                    {batchBusy ? "Procesando lote…" : "Procesar lote"}
                  </button>
                  <p className="text-[10px] leading-relaxed text-slate-400">
                    Las predicciones por lote no calculan explicaciones individuales para evitar tiempos de espera
                    excesivos. Quedan registradas en el historial con su identificador de lote.
                  </p>
                </div>
              </form>

              <section className="card p-5">
                <h2 className="text-sm font-bold text-navy-900">Resultado del lote</h2>
                {!batchResult ? (
                  <div className="flex min-h-[320px] flex-col items-center justify-center text-center">
                    <BarChart3 className="h-9 w-9 text-slate-300" />
                    <p className="mt-2 text-sm text-slate-500">Aún no se ha procesado ningún lote.</p>
                  </div>
                ) : (
                  <div className="mt-3 space-y-4">
                    <div className="grid grid-cols-2 gap-2 sm:grid-cols-4">
                      {[
                        ["Registros", batchResult.total],
                        ["Bajo", batchResult.distribution?.bajo ?? 0],
                        ["Moderado", batchResult.distribution?.moderado ?? 0],
                        ["Alto", batchResult.distribution?.alto ?? 0],
                      ].map(([label, value]) => (
                        <div key={label} className="rounded-lg border border-slate-100 bg-slate-50/60 p-3">
                          <p className="text-[9px] font-semibold uppercase tracking-wide text-slate-400">{label}</p>
                          <p className="mt-0.5 text-lg font-bold text-navy-900">{value}</p>
                        </div>
                      ))}
                    </div>
                    <div className="rounded-lg bg-slate-50 px-3 py-2 text-xs text-slate-600">
                      Probabilidad media: <strong>{formatPct(batchResult.mean_probability)}</strong> · Lote{" "}
                      <span className="font-mono text-[10px]">{batchResult.batch_id}</span>
                    </div>
                    <div className="table-wrap max-h-72 overflow-y-auto rounded-lg border border-slate-100">
                      <table className="data-table">
                        <thead>
                          <tr>
                            <th>Fila</th>
                            <th>Probabilidad</th>
                            <th>Nivel</th>
                            <th>Región</th>
                          </tr>
                        </thead>
                        <tbody>
                          {batchResult.preview?.slice(0, 40).map((row: any) => (
                            <tr key={row.row}>
                              <td>{row.row}</td>
                              <td>{formatPct(row.probability)}</td>
                              <td><RiskBadge level={row.risk_level} /></td>
                              <td>{row.input?.region ?? "—"}</td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                    {batchResult.preview?.length > 40 && (
                      <p className="text-[10px] text-slate-400">Vista previa de 40 de {batchResult.total} registros.</p>
                    )}
                  </div>
                )}
              </section>
            </div>
          )}

          {tab === "history" && (
            <section className="card">
              <div className="card-header">
                <div>
                  <h2 className="text-sm font-bold text-navy-900">Predicciones recientes</h2>
                  <p className="text-xs text-slate-500">Últimos resultados registrados en la plataforma.</p>
                </div>
                <Badge tone="info">{history.length} registros</Badge>
              </div>
              {history.length === 0 ? (
                <p className="px-5 py-10 text-center text-sm text-slate-500">Todavía no hay predicciones registradas.</p>
              ) : (
                <div className="table-wrap">
                  <table className="data-table">
                    <thead>
                      <tr>
                        <th>Fecha</th>
                        <th>Paciente</th>
                        <th>Modelo</th>
                        <th>Probabilidad</th>
                        <th>Nivel</th>
                        <th>Horizonte</th>
                        <th>Método</th>
                      </tr>
                    </thead>
                    <tbody>
                      {history.map((item) => (
                        <tr key={item.id}>
                          <td>{formatDate(item.created_at)}</td>
                          <td>{item.patient_id ? `#${item.patient_id}` : "—"}</td>
                          <td className="text-[11px]">{item.model?.name ?? `#${item.model_run_id}`}</td>
                          <td className="font-semibold">{formatPct(item.probability)}</td>
                          <td><RiskBadge level={item.risk_level} /></td>
                          <td>{item.horizon_years} años</td>
                          <td className="text-[10px] text-slate-500">{item.explanation_method}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              )}
            </section>
          )}
        </>
      )}

      <Modal
        open={patientDetailOpen}
        onClose={() => setPatientDetailOpen(false)}
        title="Ficha del paciente"
        description="Valores precargados en el formulario de predicción."
        size="md"
        footer={
          <button className="btn-primary" onClick={() => setPatientDetailOpen(false)}>
            Cerrar
          </button>
        }
      >
        {(() => {
          const patient = patients.find((item) => item.id === form.patient_id);
          if (!patient) return <p className="text-sm text-slate-500">Paciente no disponible.</p>;
          return (
            <div className="grid grid-cols-2 gap-2 text-sm">
              {[
                ["Código", patient.code],
                ["Edad", `${patient.age} años`],
                ["Sexo", patient.sex],
                ["Región", patient.region],
                ["IMC", `${patient.bmi} kg/m²`],
                ["PA", `${patient.systolic_bp}/${patient.diastolic_bp}`],
                ["Colesterol", `${patient.cholesterol_total} mg/dL`],
                ["Glucosa", `${patient.glucose} mg/dL`],
                ["Tabaquismo", patient.smoking],
                ["Diabetes", patient.diabetes ? "Sí" : "No"],
              ].map(([label, value]) => (
                <div key={label} className="rounded-lg bg-slate-50 px-3 py-2">
                  <p className="text-[10px] uppercase tracking-wide text-slate-400">{label}</p>
                  <p className="font-semibold capitalize text-navy-900">{value}</p>
                </div>
              ))}
            </div>
          );
        })()}
      </Modal>
    </div>
  );
}

import { useCallback, useEffect, useMemo, useState, type FormEvent } from "react";
import { Link } from "react-router-dom";
import {
  Download,
  Eye,
  Filter,
  Pencil,
  Plus,
  Search,
  Sparkles,
  Stethoscope,
  UserRound,
  XCircle,
} from "lucide-react";
import { api, buildQuery, downloadFile } from "../lib/api";
import type { Patient } from "../lib/types";
import { formatDate, formatPct, RISK_LABEL } from "../lib/format";
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
  Pagination,
  RiskBadge,
} from "../components/ui";

interface Options {
  regions: string[];
  sex: string[];
  education: string[];
  income_level: string[];
  smoking: string[];
}

interface PatientForm {
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
  event: string;
  followup_years: number | "";
  cohort_year: number | "";
  notes: string;
}

const EMPTY_FORM: PatientForm = {
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
  event: "",
  followup_years: "",
  cohort_year: "",
  notes: "",
};

function fromPatient(patient: Patient): PatientForm {
  return {
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
    event: patient.event === null ? "" : patient.event ? "true" : "false",
    followup_years: patient.followup_years ?? "",
    cohort_year: patient.cohort_year ?? "",
    notes: patient.notes ?? "",
  };
}

function formPayload(form: PatientForm) {
  return {
    ...form,
    event: form.event === "" ? null : form.event === "true",
    followup_years: form.followup_years === "" ? null : Number(form.followup_years),
    cohort_year: form.cohort_year === "" ? null : Number(form.cohort_year),
    notes: form.notes.trim() || null,
  };
}

export default function PatientsPage() {
  const { can } = useAuth();
  const { notify } = useToast();
  const [items, setItems] = useState<Patient[]>([]);
  const [total, setTotal] = useState(0);
  const [page, setPage] = useState(1);
  const [pageSize] = useState(20);
  const [q, setQ] = useState("");
  const [region, setRegion] = useState("");
  const [sex, setSex] = useState("");
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [options, setOptions] = useState<Options | null>(null);

  const [formOpen, setFormOpen] = useState(false);
  const [editing, setEditing] = useState<Patient | null>(null);
  const [form, setForm] = useState<PatientForm>(EMPTY_FORM);
  const [saving, setSaving] = useState(false);

  const [detail, setDetail] = useState<Patient | null>(null);
  const [detailLoading, setDetailLoading] = useState(false);

  const load = useCallback(async () => {
    setLoading(true);
    setError("");
    try {
      const data = await api.get<{ items: Patient[]; total: number }>(
        `/patients${buildQuery({ q, region, sex, page, page_size: pageSize })}`
      );
      setItems(data.items);
      setTotal(data.total);
    } catch (err) {
      setError(err instanceof Error ? err.message : "No se pudieron cargar los pacientes.");
    } finally {
      setLoading(false);
    }
  }, [q, region, sex, page, pageSize]);

  useEffect(() => {
    load();
  }, [load]);

  useEffect(() => {
    api
      .get<Options>("/patients/meta/options")
      .then(setOptions)
      .catch(() => undefined);
  }, []);

  const openCreate = () => {
    setEditing(null);
    setForm({ ...EMPTY_FORM, region: options?.regions[0] ?? EMPTY_FORM.region });
    setFormOpen(true);
  };

  const openEdit = (patient: Patient) => {
    setEditing(patient);
    setForm(fromPatient(patient));
    setFormOpen(true);
  };

  const save = async (event: FormEvent) => {
    event.preventDefault();
    setSaving(true);
    try {
      if (editing) {
        await api.patch(`/patients/${editing.id}`, formPayload(form));
        notify("Registro actualizado.", "success");
      } else {
        await api.post("/patients", formPayload(form));
        notify("Paciente creado correctamente.", "success");
      }
      setFormOpen(false);
      load();
    } catch (err) {
      notify(err instanceof Error ? err.message : "No se pudo guardar el registro.", "error");
    } finally {
      setSaving(false);
    }
  };

  const deactivate = async (patient: Patient) => {
    if (!window.confirm(`¿Desactivar el registro ${patient.code}? Se conservará para trazabilidad.`)) return;
    try {
      await api.delete(`/patients/${patient.id}`);
      notify("Registro desactivado.", "success");
      load();
    } catch (err) {
      notify(err instanceof Error ? err.message : "No se pudo desactivar.", "error");
    }
  };

  const openDetail = async (patient: Patient) => {
    setDetail(patient);
    setDetailLoading(true);
    try {
      const full = await api.get<Patient>(`/patients/${patient.id}`);
      setDetail(full);
    } catch {
      /* mantiene el registro de la lista */
    } finally {
      setDetailLoading(false);
    }
  };

  const update = (key: keyof PatientForm, value: string | number | boolean) =>
    setForm((current) => ({ ...current, [key]: value }));

  const regionList = useMemo(() => options?.regions ?? [], [options]);

  return (
    <div className="space-y-4">
      <PageHeader
        eyebrow="Cohorte"
        title="Pacientes y registros"
        description="Gestión de registros clínicos seudonimizados. Los códigos identifican al registro, no a la persona; la plataforma no almacena datos personales identificables."
        actions={
          <>
            <button
              className="btn-ghost"
              onClick={() => downloadFile("/patients/export.csv", "pacientes_cardioai.csv")}
            >
              <Download className="h-4 w-4" />
              Exportar CSV
            </button>
            {can("patients", "write") && (
              <button className="btn-accent" onClick={openCreate}>
                <Plus className="h-4 w-4" />
                Nuevo registro
              </button>
            )}
          </>
        }
      />

      <DemoNotice>
        Los registros precargados son <strong>sintéticos</strong> (semilla fija) y se incluyen para demostrar el
        flujo completo. No corresponden a personas reales.
      </DemoNotice>

      <section className="card p-4">
        <div className="flex flex-col gap-3 lg:flex-row lg:items-end">
          <label className="block flex-1">
            <span className="label">Buscar</span>
            <div className="relative">
              <Search className="pointer-events-none absolute left-3 top-2.5 h-4 w-4 text-slate-400" />
              <input
                className="input pl-9"
                placeholder="Código o región…"
                value={q}
                onChange={(event) => {
                  setQ(event.target.value);
                  setPage(1);
                }}
              />
            </div>
          </label>
          <label className="block w-full lg:w-52">
            <span className="label">Región</span>
            <select
              className="input"
              value={region}
              onChange={(event) => {
                setRegion(event.target.value);
                setPage(1);
              }}
            >
              <option value="">Todas</option>
              {regionList.map((item) => (
                <option key={item} value={item}>
                  {item}
                </option>
              ))}
            </select>
          </label>
          <label className="block w-full lg:w-40">
            <span className="label">Sexo</span>
            <select
              className="input"
              value={sex}
              onChange={(event) => {
                setSex(event.target.value);
                setPage(1);
              }}
            >
              <option value="">Todos</option>
              <option value="F">F</option>
              <option value="M">M</option>
              <option value="Otro">Otro</option>
            </select>
          </label>
          <button className="btn-ghost" onClick={load}>
            <Filter className="h-4 w-4" />
            Aplicar
          </button>
        </div>
      </section>

      {error && <ErrorBanner message={error} onRetry={load} />}

      <section className="card">
        <div className="card-header">
          <div>
            <h2 className="text-sm font-bold text-navy-900">Registros de la cohorte</h2>
            <p className="text-xs text-slate-500">{total} registros activos · página {page}</p>
          </div>
        </div>
        {loading ? (
          <PageLoading label="Cargando registros…" />
        ) : items.length === 0 ? (
          <div className="p-5">
            <EmptyState
              title="Sin registros"
              message="No se encontraron pacientes con los filtros actuales. Ajuste la búsqueda o cree un registro."
              action={
                can("patients", "write") ? (
                  <button className="btn-accent mt-2" onClick={openCreate}>
                    <Plus className="h-4 w-4" />
                    Crear registro
                  </button>
                ) : undefined
              }
            />
          </div>
        ) : (
          <>
            <div className="table-wrap">
              <table className="data-table">
                <thead>
                  <tr>
                    <th>Código</th>
                    <th>Edad / Sexo</th>
                    <th>Región</th>
                    <th>Tabaquismo</th>
                    <th>PA</th>
                    <th>IMC</th>
                    <th>Última predicción</th>
                    <th className="text-right">Acciones</th>
                  </tr>
                </thead>
                <tbody>
                  {items.map((patient) => (
                    <tr key={patient.id}>
                      <td>
                        <button className="font-semibold text-accent-700 hover:underline" onClick={() => openDetail(patient)}>
                          {patient.code}
                        </button>
                        <p className="text-[10px] text-slate-400">{formatDate(patient.created_at)}</p>
                      </td>
                      <td>
                        {patient.age} años · {patient.sex}
                      </td>
                      <td>{patient.region}</td>
                      <td className="capitalize">{patient.smoking}</td>
                      <td>
                        {patient.systolic_bp}/{patient.diastolic_bp}
                      </td>
                      <td>{patient.bmi.toFixed(1)}</td>
                      <td>
                        {patient.last_prediction ? (
                          <div className="flex items-center gap-2">
                            <RiskBadge level={patient.last_prediction.risk_level} />
                            <span className="text-xs font-semibold">
                              {formatPct(patient.last_prediction.probability)}
                            </span>
                          </div>
                        ) : (
                          <span className="text-xs text-slate-400">Sin predicción</span>
                        )}
                      </td>
                      <td>
                        <div className="flex justify-end gap-1">
                          <button
                            className="rounded-md p-1.5 text-slate-500 hover:bg-slate-100"
                            onClick={() => openDetail(patient)}
                            title="Ver detalle"
                          >
                            <Eye className="h-4 w-4" />
                          </button>
                          {can("predictions", "write") && (
                            <Link
                              to={`/prediction?patient_id=${patient.id}`}
                              className="rounded-md p-1.5 text-accent-600 hover:bg-accent-50"
                              title="Predecir riesgo"
                            >
                              <Sparkles className="h-4 w-4" />
                            </Link>
                          )}
                          {can("patients", "write") && (
                            <>
                              <button
                                className="rounded-md p-1.5 text-slate-500 hover:bg-slate-100"
                                onClick={() => openEdit(patient)}
                                title="Editar"
                              >
                                <Pencil className="h-4 w-4" />
                              </button>
                              <button
                                className="rounded-md p-1.5 text-red-500 hover:bg-red-50"
                                onClick={() => deactivate(patient)}
                                title="Desactivar"
                              >
                                <XCircle className="h-4 w-4" />
                              </button>
                            </>
                          )}
                        </div>
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

      <Modal
        open={formOpen}
        onClose={() => setFormOpen(false)}
        title={editing ? `Editar ${editing.code}` : "Nuevo registro clínico"}
        description="Los campos clínicos alimentan el modelo de riesgo. Verifique unidades antes de guardar."
        size="xl"
        footer={
          <>
            <button className="btn-ghost" onClick={() => setFormOpen(false)}>
              Cancelar
            </button>
            <button className="btn-accent" form="patient-form" disabled={saving}>
              {saving ? "Guardando…" : editing ? "Guardar cambios" : "Crear registro"}
            </button>
          </>
        }
      >
        <form id="patient-form" onSubmit={save} className="space-y-5">
          <div>
            <p className="mb-3 text-[11px] font-bold uppercase tracking-wider text-slate-400">Datos generales</p>
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
                  {regionList.map((item) => (
                    <option key={item} value={item}>
                      {item}
                    </option>
                  ))}
                </select>
              </Field>
              <Field label="Residencia">
                <select className="input" value={form.urban ? "urbano" : "rural"} onChange={(e) => update("urban", e.target.value === "urbano")}>
                  <option value="urbano">Urbana</option>
                  <option value="rural">Rural</option>
                </select>
              </Field>
              <Field label="Educación">
                <select className="input" value={form.education} onChange={(e) => update("education", e.target.value)}>
                  {(options?.education ?? ["primaria", "secundaria", "tecnica", "universitaria"]).map((item) => (
                    <option key={item} value={item}>
                      {item}
                    </option>
                  ))}
                </select>
              </Field>
              <Field label="Ingreso">
                <select className="input" value={form.income_level} onChange={(e) => update("income_level", e.target.value)}>
                  {(options?.income_level ?? ["bajo", "medio", "alto"]).map((item) => (
                    <option key={item} value={item}>
                      {item}
                    </option>
                  ))}
                </select>
              </Field>
            </div>
          </div>

          <div>
            <p className="mb-3 text-[11px] font-bold uppercase tracking-wider text-slate-400">Factores clínicos</p>
            <div className="grid gap-3 sm:grid-cols-3">
              <Field label="Tabaquismo">
                <select className="input" value={form.smoking} onChange={(e) => update("smoking", e.target.value)}>
                  <option value="nunca">Nunca</option>
                  <option value="exfumador">Exfumador</option>
                  <option value="fumador">Fumador</option>
                </select>
              </Field>
              <Field label="IMC (kg/m²)">
                <input type="number" step="0.1" className="input" value={form.bmi} onChange={(e) => update("bmi", Number(e.target.value))} required />
              </Field>
              <Field label="Actividad física (min/sem)">
                <input type="number" className="input" value={form.physical_activity_min} onChange={(e) => update("physical_activity_min", Number(e.target.value))} />
              </Field>
              <Field label="Presión sistólica (mmHg)">
                <input type="number" className="input" value={form.systolic_bp} onChange={(e) => update("systolic_bp", Number(e.target.value))} required />
              </Field>
              <Field label="Presión diastólica (mmHg)">
                <input type="number" className="input" value={form.diastolic_bp} onChange={(e) => update("diastolic_bp", Number(e.target.value))} required />
              </Field>
              <Field label="Colesterol total (mg/dL)">
                <input type="number" step="0.1" className="input" value={form.cholesterol_total} onChange={(e) => update("cholesterol_total", Number(e.target.value))} required />
              </Field>
              <Field label="HDL (mg/dL)">
                <input type="number" step="0.1" className="input" value={form.hdl} onChange={(e) => update("hdl", Number(e.target.value))} required />
              </Field>
              <Field label="Glucosa (mg/dL)">
                <input type="number" step="0.1" className="input" value={form.glucose} onChange={(e) => update("glucose", Number(e.target.value))} required />
              </Field>
              <Field label="Alcohol (unidades/sem)">
                <input type="number" step="0.1" className="input" value={form.alcohol_units} onChange={(e) => update("alcohol_units", Number(e.target.value))} />
              </Field>
              <Field label="Estrés percibido (1-10)">
                <input type="number" className="input" value={form.stress_level} onChange={(e) => update("stress_level", Number(e.target.value))} />
              </Field>
              <Field label="PM2.5 (µg/m³)">
                <input type="number" step="0.1" className="input" value={form.pm25} onChange={(e) => update("pm25", Number(e.target.value))} />
              </Field>
              <Field label="Espacio verde (%)">
                <input type="number" step="0.1" className="input" value={form.green_space} onChange={(e) => update("green_space", Number(e.target.value))} />
              </Field>
            </div>
            <div className="mt-3 flex flex-wrap gap-5 rounded-lg bg-slate-50 px-4 py-3 text-sm text-slate-700">
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
            </div>
          </div>

          <div>
            <p className="mb-3 text-[11px] font-bold uppercase tracking-wider text-slate-400">
              Seguimiento (opcional)
            </p>
            <div className="grid gap-3 sm:grid-cols-3">
              <Field label="Año de cohorte">
                <input type="number" className="input" value={form.cohort_year} onChange={(e) => update("cohort_year", e.target.value === "" ? "" : Number(e.target.value))} placeholder="2018" />
              </Field>
              <Field label="Años de seguimiento">
                <input type="number" step="0.1" className="input" value={form.followup_years} onChange={(e) => update("followup_years", e.target.value === "" ? "" : Number(e.target.value))} placeholder="10.0" />
              </Field>
              <Field label="Evento observado">
                <select className="input" value={form.event} onChange={(e) => update("event", e.target.value)}>
                  <option value="">Desconocido / sin seguimiento</option>
                  <option value="true">Sí</option>
                  <option value="false">No</option>
                </select>
              </Field>
            </div>
            <Field label="Notas" className="mt-3">
              <textarea className="input min-h-20" value={form.notes} onChange={(e) => update("notes", e.target.value)} placeholder="Observaciones de investigación, no datos identificables." />
            </Field>
          </div>
        </form>
      </Modal>

      <Modal
        open={detail !== null}
        onClose={() => setDetail(null)}
        title={detail ? `Registro ${detail.code}` : "Detalle"}
        description={detail ? `${detail.age} años · ${detail.sex} · ${detail.region}` : undefined}
        size="lg"
        footer={
          detail ? (
            <>
              {detail.last_prediction && (
                <Link to={`/prediction?patient_id=${detail.id}`} className="btn-ghost">
                  <Sparkles className="h-4 w-4" />
                  Nueva predicción
                </Link>
              )}
              <button className="btn-primary" onClick={() => setDetail(null)}>
                Cerrar
              </button>
            </>
          ) : undefined
        }
      >
        {detailLoading && <PageLoading label="Cargando detalle…" />}
        {detail && !detailLoading && (
          <div className="space-y-5">
            <div className="flex flex-wrap items-center gap-2">
              {detail.is_demo && <Badge tone="warning">Sintético</Badge>}
              {detail.last_prediction ? <RiskBadge level={detail.last_prediction.risk_level} /> : <Badge>Sin predicción</Badge>}
              <span className="text-xs text-slate-500">Actualizado {formatDate(detail.updated_at)}</span>
            </div>
            <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
              {[
                ["IMC", `${detail.bmi.toFixed(1)} kg/m²`],
                ["PA", `${detail.systolic_bp}/${detail.diastolic_bp} mmHg`],
                ["Colesterol", `${detail.cholesterol_total.toFixed(0)} mg/dL`],
                ["HDL", `${detail.hdl.toFixed(0)} mg/dL`],
                ["Glucosa", `${detail.glucose.toFixed(0)} mg/dL`],
                ["Actividad", `${detail.physical_activity_min} min/sem`],
                ["PM2.5", `${detail.pm25.toFixed(1)} µg/m³`],
                ["Estrés", `${detail.stress_level}/10`],
              ].map(([label, value]) => (
                <div key={label} className="rounded-lg border border-slate-100 bg-slate-50/60 p-3">
                  <p className="text-[10px] font-semibold uppercase tracking-wide text-slate-400">{label}</p>
                  <p className="mt-0.5 text-sm font-semibold text-navy-900">{value}</p>
                </div>
              ))}
            </div>
            <div className="grid gap-2 rounded-lg border border-slate-100 p-4 text-sm sm:grid-cols-3">
              <p>
                <span className="text-slate-500">Tabaquismo:</span>{" "}
                <span className="font-medium capitalize">{detail.smoking}</span>
              </p>
              <p>
                <span className="text-slate-500">Diabetes:</span>{" "}
                <span className="font-medium">{detail.diabetes ? "Sí" : "No"}</span>
              </p>
              <p>
                <span className="text-slate-500">Hipertensión:</span>{" "}
                <span className="font-medium">{detail.hypertension ? "Sí" : "No"}</span>
              </p>
              <p>
                <span className="text-slate-500">Antecedente familiar:</span>{" "}
                <span className="font-medium">{detail.family_history ? "Sí" : "No"}</span>
              </p>
              <p>
                <span className="text-slate-500">Educación:</span>{" "}
                <span className="font-medium capitalize">{detail.education}</span>
              </p>
              <p>
                <span className="text-slate-500">Ingreso:</span>{" "}
                <span className="font-medium capitalize">{detail.income_level}</span>
              </p>
            </div>
            <div>
              <p className="mb-2 text-[11px] font-bold uppercase tracking-wider text-slate-400">
                Historial de predicciones
              </p>
              {detail.prediction_history && detail.prediction_history.length > 0 ? (
                <div className="table-wrap rounded-lg border border-slate-100">
                  <table className="data-table">
                    <thead>
                      <tr>
                        <th>Fecha</th>
                        <th>Nivel</th>
                        <th>Probabilidad</th>
                        <th>Horizonte</th>
                      </tr>
                    </thead>
                    <tbody>
                      {detail.prediction_history.map((item) => (
                        <tr key={item.id}>
                          <td>{formatDate(item.created_at)}</td>
                          <td>{RISK_LABEL[item.risk_level] ?? item.risk_level}</td>
                          <td>{formatPct(item.probability)}</td>
                          <td>{item.horizon_years} años</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              ) : (
                <p className="rounded-lg bg-slate-50 px-4 py-3 text-sm text-slate-500">
                  Este registro todavía no tiene predicciones.
                </p>
              )}
            </div>
            {detail.notes && (
              <div className="rounded-lg border border-slate-100 p-3 text-sm text-slate-600">{detail.notes}</div>
            )}
            <p className="text-[10px] leading-relaxed text-slate-400">
              <Stethoscope className="mr-1 inline h-3 w-3" />
              Registro de investigación. No contiene datos personales identificables y no debe usarse para
              decisiones asistenciales.
            </p>
          </div>
        )}
      </Modal>
    </div>
  );
}

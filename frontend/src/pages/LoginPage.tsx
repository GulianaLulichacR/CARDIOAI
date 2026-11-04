import { useState, type FormEvent } from "react";
import { useLocation, useNavigate } from "react-router-dom";
import { Activity, ArrowRight, Eye, EyeOff, LockKeyhole, Mail, ShieldCheck } from "lucide-react";
import { useAuth } from "../context/AuthContext";
import { ApiError } from "../lib/api";
import { Spinner } from "../components/ui";

const DEMO_USERS = [
  { role: "Superadministrador", email: "admin@cardioai.local", password: "Admin2026!" },
  { role: "Investigador", email: "investigador@cardioai.local", password: "Investiga2026!" },
  { role: "Profesional de salud", email: "clinico@cardioai.local", password: "Clinico2026!" },
  { role: "Analista de datos", email: "analista@cardioai.local", password: "Analista2026!" },
  { role: "Auditor", email: "auditor@cardioai.local", password: "Auditor2026!" },
];

export default function LoginPage() {
  const { login } = useAuth();
  const navigate = useNavigate();
  const location = useLocation();
  const [email, setEmail] = useState("admin@cardioai.local");
  const [password, setPassword] = useState("Admin2026!");
  const [show, setShow] = useState(false);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);

  const submit = async (event: FormEvent) => {
    event.preventDefault();
    setError("");
    setLoading(true);
    try {
      await login(email.trim(), password);
      const from = (location.state as { from?: string } | null)?.from;
      navigate(from && from !== "/login" ? from : "/", { replace: true });
    } catch (err) {
      setError(err instanceof ApiError ? err.detail : "No se pudo iniciar sesión. Intente nuevamente.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="grid min-h-screen bg-white lg:grid-cols-[1.05fr_0.95fr]">
      <section className="relative hidden flex-col justify-between overflow-hidden bg-navy-900 p-10 text-white lg:flex xl:p-14">
        <div className="absolute -right-32 -top-32 h-96 w-96 rounded-full bg-accent-600/20 blur-3xl" />
        <div className="absolute -bottom-40 -left-24 h-96 w-96 rounded-full bg-accent-500/10 blur-3xl" />
        <div className="relative">
          <div className="flex items-center gap-3">
            <span className="flex h-11 w-11 items-center justify-center rounded-xl bg-accent-500/15 text-accent-300 ring-1 ring-accent-400/30">
              <Activity className="h-6 w-6" />
            </span>
            <div>
              <p className="text-xl font-bold tracking-tight">CARDIOAI</p>
              <p className="text-[10px] font-semibold uppercase tracking-[0.2em] text-navy-300">
                Riesgo cardiovascular explicable
              </p>
            </div>
          </div>

          <h1 className="mt-16 max-w-xl text-4xl font-bold leading-tight tracking-tight xl:text-[2.7rem]">
            Predicción explicable del riesgo cardiovascular con validación temporal y territorial.
          </h1>
          <p className="mt-5 max-w-lg text-sm leading-relaxed text-navy-200">
            Plataforma de investigación que integra factores clínicos, conductuales y socioambientales, cinco
            enfoques de modelado y trazabilidad completa de cada decisión analítica.
          </p>

          <div className="mt-10 grid max-w-lg grid-cols-2 gap-3">
            {[
              ["5 modelos", "RF · XGBoost · Cox · híbridos"],
              ["Validación", "Temporal + geográfica"],
              ["Explicabilidad", "SHAP y atención temporal"],
              ["Gobernanza", "Roles, auditoría y reportes"],
            ].map(([title, detail]) => (
              <div key={title} className="rounded-xl border border-white/10 bg-white/5 p-3.5">
                <p className="text-sm font-semibold text-white">{title}</p>
                <p className="mt-0.5 text-[11px] leading-snug text-navy-300">{detail}</p>
              </div>
            ))}
          </div>
        </div>

        <div className="relative mt-10 flex items-start gap-2.5 rounded-xl border border-amber-400/20 bg-amber-400/10 p-3.5 text-[11px] leading-relaxed text-amber-100">
          <ShieldCheck className="mt-0.5 h-4 w-4 shrink-0" />
          <p>
            Entorno de demostración con datos <strong>sintéticos</strong>. Las métricas, predicciones y umbrales
            no constituyen evidencia clínica ni sustituyen la evaluación de un profesional de la salud.
          </p>
        </div>
      </section>

      <section className="flex items-center justify-center px-5 py-10 sm:px-10">
        <div className="w-full max-w-md">
          <div className="mb-8 flex items-center gap-3 lg:hidden">
            <span className="flex h-10 w-10 items-center justify-center rounded-xl bg-navy-900 text-accent-300">
              <Activity className="h-5 w-5" />
            </span>
            <div>
              <p className="text-lg font-bold text-navy-900">CARDIOAI</p>
              <p className="text-[10px] font-semibold uppercase tracking-[0.16em] text-slate-400">
                Riesgo explicable
              </p>
            </div>
          </div>

          <p className="text-[11px] font-bold uppercase tracking-[0.18em] text-accent-700">Acceso institucional</p>
          <h2 className="mt-1 text-2xl font-bold tracking-tight text-navy-900">Iniciar sesión</h2>
          <p className="mt-1 text-sm text-slate-500">
            Ingrese con las credenciales asignadas por el administrador de la plataforma.
          </p>

          <form onSubmit={submit} className="mt-6 space-y-4">
            <label className="block">
              <span className="label">Correo electrónico</span>
              <div className="relative">
                <Mail className="pointer-events-none absolute left-3 top-2.5 h-4 w-4 text-slate-400" />
                <input
                  type="email"
                  className="input pl-9"
                  value={email}
                  onChange={(event) => setEmail(event.target.value)}
                  autoComplete="email"
                  required
                />
              </div>
            </label>

            <label className="block">
              <span className="label">Contraseña</span>
              <div className="relative">
                <LockKeyhole className="pointer-events-none absolute left-3 top-2.5 h-4 w-4 text-slate-400" />
                <input
                  type={show ? "text" : "password"}
                  className="input px-9"
                  value={password}
                  onChange={(event) => setPassword(event.target.value)}
                  autoComplete="current-password"
                  required
                />
                <button
                  type="button"
                  onClick={() => setShow((value) => !value)}
                  className="absolute right-2.5 top-2.5 text-slate-400 hover:text-slate-600"
                  aria-label={show ? "Ocultar contraseña" : "Mostrar contraseña"}
                >
                  {show ? <EyeOff className="h-4 w-4" /> : <Eye className="h-4 w-4" />}
                </button>
              </div>
            </label>

            {error && (
              <div className="rounded-lg border border-red-200 bg-red-50 px-3 py-2 text-sm text-red-700">{error}</div>
            )}

            <button type="submit" className="btn-accent w-full" disabled={loading}>
              {loading ? <Spinner className="h-4 w-4 text-white" /> : <ArrowRight className="h-4 w-4" />}
              {loading ? "Verificando…" : "Entrar a la plataforma"}
            </button>
          </form>

          <div className="mt-8 rounded-xl border border-slate-200 bg-slate-50/70 p-4">
            <p className="text-[11px] font-bold uppercase tracking-wider text-slate-500">
              Credenciales de demostración
            </p>
            <div className="mt-2 divide-y divide-slate-200">
              {DEMO_USERS.map((item) => (
                <button
                  key={item.email}
                  onClick={() => {
                    setEmail(item.email);
                    setPassword(item.password);
                  }}
                  className="flex w-full items-center justify-between gap-2 py-2 text-left text-xs hover:text-accent-700"
                >
                  <span className="font-medium text-slate-700">{item.role}</span>
                  <span className="truncate font-mono text-[10px] text-slate-400">{item.email}</span>
                </button>
              ))}
            </div>
            <p className="mt-2 text-[10px] leading-snug text-slate-400">
              Credenciales exclusivas del entorno local. Cámbielas antes de cualquier despliegue compartido.
            </p>
          </div>

          <p className="mt-6 text-[10px] leading-relaxed text-slate-400">
            Al ingresar acepta que esta plataforma es una herramienta de investigación y docencia. No emite
            diagnósticos, no prescribe tratamientos y no reemplaza el juicio clínico.
          </p>
        </div>
      </section>
    </div>
  );
}

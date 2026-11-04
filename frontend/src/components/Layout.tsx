import { useEffect, useMemo, useState } from "react";
import { NavLink, Outlet, useLocation, useNavigate } from "react-router-dom";
import clsx from "clsx";
import {
  Activity,
  Bell,
  BookOpen,
  BrainCircuit,
  ChevronLeft,
  Database,
  FileBarChart,
  FlaskConical,
  LayoutDashboard,
  LogOut,
  MapPinned,
  Menu,
  ShieldCheck,
  Sparkles,
  Stethoscope,
  Users,
  X,
} from "lucide-react";
import { useAuth } from "../context/AuthContext";
import { api } from "../lib/api";
import type { Alert } from "../lib/types";
import { relativeTime, severityClasses } from "../lib/format";
import { Badge } from "./ui";

interface NavItem {
  to: string;
  label: string;
  module: string;
  icon: typeof LayoutDashboard;
  end?: boolean;
}

const GROUPS: { title: string; items: NavItem[] }[] = [
  {
    title: "Analítica",
    items: [
      { to: "/", label: "Dashboard", module: "dashboard", icon: LayoutDashboard, end: true },
      { to: "/patients", label: "Pacientes", module: "patients", icon: Stethoscope },
      { to: "/prediction", label: "Predicción", module: "predictions", icon: Sparkles },
      { to: "/geography", label: "Territorio", module: "geo", icon: MapPinned },
    ],
  },
  {
    title: "Ciencia de datos",
    items: [
      { to: "/datasets", label: "Datasets", module: "datasets", icon: Database },
      { to: "/models", label: "Modelos", module: "models", icon: BrainCircuit },
      { to: "/assistant", label: "Asistente", module: "assistant", icon: BookOpen },
    ],
  },
  {
    title: "Gobernanza",
    items: [
      { to: "/reports", label: "Reportes", module: "reports", icon: FileBarChart },
      { to: "/audit", label: "Auditoría", module: "audit", icon: ShieldCheck },
      { to: "/users", label: "Usuarios", module: "users", icon: Users },
      { to: "/docs", label: "Documentación", module: "docs", icon: FlaskConical },
    ],
  },
];

const MODULE_BY_PATH: Record<string, string> = {
  "/": "dashboard",
  "/patients": "patients",
  "/datasets": "datasets",
  "/models": "models",
  "/prediction": "predictions",
  "/geography": "geo",
  "/reports": "reports",
  "/audit": "audit",
  "/users": "users",
  "/assistant": "assistant",
  "/docs": "docs",
};

export default function Layout() {
  const { user, logout, can } = useAuth();
  const location = useLocation();
  const navigate = useNavigate();
  const [open, setOpen] = useState(false);
  const [alertsOpen, setAlertsOpen] = useState(false);
  const [alerts, setAlerts] = useState<Alert[]>([]);
  const [unread, setUnread] = useState(0);

  const visibleGroups = useMemo(
    () =>
      GROUPS.map((group) => ({
        ...group,
        items: group.items.filter((item) => can(item.module, "read")),
      })).filter((group) => group.items.length > 0),
    [can]
  );

  useEffect(() => {
    setOpen(false);
    setAlertsOpen(false);
  }, [location.pathname]);

  useEffect(() => {
    const module = MODULE_BY_PATH[location.pathname];
    if (!module || !user) return;
    api
      .post("/audit/navigation", { module, direction: "entrada", route: location.pathname })
      .catch(() => undefined);
  }, [location.pathname, user]);

  const loadAlerts = () => {
    api
      .get<{ items: Alert[]; unread: number }>("/alerts?page_size=8")
      .then((data) => {
        setAlerts(data.items);
        setUnread(data.unread);
      })
      .catch(() => undefined);
  };

  useEffect(() => {
    loadAlerts();
    const timer = window.setInterval(loadAlerts, 60000);
    return () => window.clearInterval(timer);
  }, []);

  const markRead = async (id: number) => {
    try {
      await api.post(`/alerts/${id}/read`);
      loadAlerts();
    } catch {
      /* silencioso: la campana no debe bloquear la navegación */
    }
  };

  const markAll = async () => {
    try {
      await api.post("/alerts/read-all");
      loadAlerts();
    } catch {
      /* silencioso */
    }
  };

  const handleLogout = async () => {
    await logout();
    navigate("/login");
  };

  return (
    <div className="min-h-screen bg-slate-50 lg:flex">
      {open && (
        <button
          className="fixed inset-0 z-40 bg-navy-950/50 backdrop-blur-[1px] lg:hidden"
          onClick={() => setOpen(false)}
          aria-label="Cerrar menú"
        />
      )}
      <aside
        className={clsx(
          "fixed inset-y-0 left-0 z-50 flex w-[272px] flex-col bg-navy-900 text-white transition-transform duration-200 lg:static lg:translate-x-0",
          open ? "translate-x-0" : "-translate-x-full"
        )}
      >
        <div className="flex items-center justify-between px-5 py-5">
          <NavLink to="/" className="flex items-center gap-3">
            <span className="flex h-10 w-10 items-center justify-center rounded-xl bg-accent-500/15 text-accent-300 ring-1 ring-accent-400/30">
              <Activity className="h-5 w-5" />
            </span>
            <span>
              <span className="block text-lg font-bold tracking-tight">CARDIOAI</span>
              <span className="block text-[10px] font-medium uppercase tracking-[0.18em] text-navy-300">
                Riesgo explicable
              </span>
            </span>
          </NavLink>
          <button className="rounded-lg p-1.5 text-navy-300 hover:bg-white/10 lg:hidden" onClick={() => setOpen(false)}>
            <X className="h-5 w-5" />
          </button>
        </div>

        <div className="mx-4 mb-4 rounded-lg border border-amber-400/20 bg-amber-400/10 px-3 py-2 text-[11px] leading-snug text-amber-200">
          Datos sintéticos de demostración. No es un dispositivo médico.
        </div>

        <nav className="flex-1 space-y-5 overflow-y-auto px-3 pb-4">
          {visibleGroups.map((group) => (
            <div key={group.title}>
              <p className="px-3 pb-1.5 text-[10px] font-bold uppercase tracking-[0.18em] text-navy-400">
                {group.title}
              </p>
              <div className="space-y-0.5">
                {group.items.map((item) => {
                  const Icon = item.icon;
                  return (
                    <NavLink
                      key={item.to}
                      to={item.to}
                      end={item.end}
                      className={({ isActive }) =>
                        clsx(
                          "flex items-center gap-3 rounded-lg px-3 py-2 text-sm font-medium transition",
                          isActive
                            ? "bg-white/10 text-white ring-1 ring-white/10"
                            : "text-navy-200 hover:bg-white/5 hover:text-white"
                        )
                      }
                    >
                      <Icon className="h-4 w-4 shrink-0" />
                      {item.label}
                    </NavLink>
                  );
                })}
              </div>
            </div>
          ))}
        </nav>

        <div className="border-t border-white/10 p-4">
          <div className="flex items-center gap-3">
            <span className="flex h-9 w-9 shrink-0 items-center justify-center rounded-full bg-accent-500/20 text-sm font-bold text-accent-200">
              {(user?.full_name || "U").slice(0, 1).toUpperCase()}
            </span>
            <div className="min-w-0 flex-1">
              <p className="truncate text-sm font-semibold">{user?.full_name}</p>
              <p className="truncate text-[11px] text-navy-300">{user?.role_label}</p>
            </div>
            <button
              onClick={handleLogout}
              className="rounded-lg p-1.5 text-navy-300 hover:bg-white/10 hover:text-white"
              title="Cerrar sesión"
            >
              <LogOut className="h-4 w-4" />
            </button>
          </div>
        </div>
      </aside>

      <div className="flex min-w-0 flex-1 flex-col">
        <header className="sticky top-0 z-30 flex h-16 items-center justify-between gap-3 border-b border-slate-200 bg-white/90 px-4 backdrop-blur sm:px-6">
          <div className="flex items-center gap-2">
            <button
              className="rounded-lg p-2 text-slate-500 hover:bg-slate-100 lg:hidden"
              onClick={() => setOpen(true)}
              aria-label="Abrir menú"
            >
              <Menu className="h-5 w-5" />
            </button>
            <div className="hidden items-center gap-2 text-xs text-slate-500 sm:flex">
              <span className="inline-flex h-2 w-2 rounded-full bg-emerald-500" />
              Sistema operativo
              <span className="text-slate-300">·</span>
              <span className="font-medium text-slate-600">Entorno demo</span>
            </div>
          </div>

          <div className="flex items-center gap-2">
            <span className="hidden rounded-full border border-slate-200 bg-slate-50 px-2.5 py-1 text-[11px] font-semibold text-slate-500 sm:inline">
              v1.0.0
            </span>
            <div className="relative">
              <button
                onClick={() => setAlertsOpen((value) => !value)}
                className="relative rounded-lg p-2 text-slate-500 hover:bg-slate-100"
                aria-label="Alertas"
              >
                <Bell className="h-5 w-5" />
                {unread > 0 && (
                  <span className="absolute -right-0.5 -top-0.5 flex h-4 min-w-4 items-center justify-center rounded-full bg-red-600 px-1 text-[9px] font-bold text-white">
                    {unread > 9 ? "9+" : unread}
                  </span>
                )}
              </button>
              {alertsOpen && (
                <div className="absolute right-0 top-12 z-50 w-[min(92vw,380px)] overflow-hidden rounded-xl border border-slate-200 bg-white shadow-pop">
                  <div className="flex items-center justify-between border-b border-slate-100 px-4 py-3">
                    <p className="text-sm font-bold text-navy-900">Alertas</p>
                    <button onClick={markAll} className="text-[11px] font-semibold text-accent-700 hover:underline">
                      Marcar leídas
                    </button>
                  </div>
                  <div className="max-h-80 overflow-y-auto">
                    {alerts.length === 0 && <p className="px-4 py-6 text-sm text-slate-500">Sin alertas.</p>}
                    {alerts.map((alert) => (
                      <button
                        key={alert.id}
                        onClick={() => markRead(alert.id)}
                        className={clsx(
                          "block w-full border-b border-slate-50 px-4 py-3 text-left hover:bg-slate-50",
                          !alert.is_read && "bg-sky-50/40"
                        )}
                      >
                        <div className="flex items-center justify-between gap-2">
                          <Badge tone={alert.severity === "critical" ? "danger" : alert.severity === "warning" ? "warning" : "info"}>
                            {alert.severity}
                          </Badge>
                          <span className="text-[10px] text-slate-400">{relativeTime(alert.created_at)}</span>
                        </div>
                        <p className="mt-1 text-xs font-semibold text-navy-900">{alert.title}</p>
                        <p className="mt-0.5 line-clamp-2 text-[11px] leading-snug text-slate-500">{alert.message}</p>
                      </button>
                    ))}
                  </div>
                </div>
              )}
            </div>
          </div>
        </header>

        <main className="mx-auto w-full max-w-[1500px] flex-1 px-4 py-6 sm:px-6 lg:px-8">
          <Outlet />
        </main>

        <footer className="border-t border-slate-200 px-6 py-4 text-center text-[11px] text-slate-400">
          CARDIOAI · Plataforma de investigación con datos sintéticos · No usar para diagnóstico ni decisiones clínicas
        </footer>
      </div>
    </div>
  );
}

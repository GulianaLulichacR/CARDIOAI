import { useEffect, type ReactNode } from "react";
import clsx from "clsx";
import { AlertTriangle, Inbox, Loader2, X } from "lucide-react";
import { RISK_LABEL, riskBadgeClasses } from "../lib/format";

export function Spinner({ className = "h-5 w-5" }: { className?: string }) {
  return <Loader2 className={clsx("animate-spin text-accent-600", className)} />;
}

export function PageLoading({ label = "Cargando información…" }: { label?: string }) {
  return (
    <div className="flex min-h-[45vh] flex-col items-center justify-center gap-3 text-sm text-slate-500">
      <Spinner className="h-7 w-7" />
      <p>{label}</p>
    </div>
  );
}

export function PageHeader({
  title,
  description,
  actions,
  eyebrow,
}: {
  title: string;
  description?: string;
  actions?: ReactNode;
  eyebrow?: string;
}) {
  return (
    <header className="mb-6 flex flex-col gap-4 sm:flex-row sm:items-end sm:justify-between">
      <div>
        {eyebrow && (
          <p className="mb-1 text-[11px] font-bold uppercase tracking-[0.18em] text-accent-700">{eyebrow}</p>
        )}
        <h1 className="text-2xl font-bold tracking-tight text-navy-900">{title}</h1>
        {description && <p className="mt-1 max-w-3xl text-sm leading-relaxed text-slate-500">{description}</p>}
      </div>
      {actions && <div className="flex flex-wrap items-center gap-2">{actions}</div>}
    </header>
  );
}

export function KpiCard({
  label,
  value,
  hint,
  icon,
  tone = "default",
}: {
  label: string;
  value: ReactNode;
  hint?: ReactNode;
  icon?: ReactNode;
  tone?: "default" | "danger" | "success" | "warning";
}) {
  return (
    <div className="card p-4">
      <div className="flex items-start justify-between gap-3">
        <p className="text-[11px] font-semibold uppercase tracking-wider text-slate-500">{label}</p>
        {icon && (
          <span
            className={clsx(
              "flex h-9 w-9 items-center justify-center rounded-lg",
              tone === "danger" && "bg-red-50 text-red-600",
              tone === "success" && "bg-emerald-50 text-emerald-600",
              tone === "warning" && "bg-amber-50 text-amber-600",
              tone === "default" && "bg-navy-50 text-navy-700"
            )}
          >
            {icon}
          </span>
        )}
      </div>
      <p className="mt-2 text-2xl font-bold tracking-tight text-navy-900">{value}</p>
      {hint && <p className="mt-1 text-xs leading-snug text-slate-500">{hint}</p>}
    </div>
  );
}

export function RiskBadge({ level }: { level: string }) {
  return <span className={clsx("badge", riskBadgeClasses(level))}>{RISK_LABEL[level] ?? level}</span>;
}

export function Badge({
  children,
  tone = "neutral",
}: {
  children: ReactNode;
  tone?: "neutral" | "info" | "success" | "warning" | "danger";
}) {
  return (
    <span
      className={clsx(
        "badge",
        tone === "neutral" && "bg-slate-100 text-slate-600 ring-slate-200",
        tone === "info" && "bg-sky-50 text-sky-700 ring-sky-200",
        tone === "success" && "bg-emerald-50 text-emerald-700 ring-emerald-200",
        tone === "warning" && "bg-amber-50 text-amber-700 ring-amber-200",
        tone === "danger" && "bg-red-50 text-red-700 ring-red-200"
      )}
    >
      {children}
    </span>
  );
}

export function Modal({
  open,
  onClose,
  title,
  description,
  children,
  footer,
  size = "md",
}: {
  open: boolean;
  onClose: () => void;
  title: string;
  description?: string;
  children: ReactNode;
  footer?: ReactNode;
  size?: "sm" | "md" | "lg" | "xl";
}) {
  useEffect(() => {
    if (!open) return;
    const onKey = (event: KeyboardEvent) => {
      if (event.key === "Escape") onClose();
    };
    document.addEventListener("keydown", onKey);
    document.body.style.overflow = "hidden";
    return () => {
      document.removeEventListener("keydown", onKey);
      document.body.style.overflow = "";
    };
  }, [open, onClose]);

  if (!open) return null;
  return (
    <div className="fixed inset-0 z-[900] flex items-end justify-center bg-navy-950/50 p-0 backdrop-blur-[2px] sm:items-center sm:p-6">
      <div
        className={clsx(
          "flex max-h-[94vh] w-full flex-col rounded-t-2xl bg-white shadow-pop sm:rounded-2xl",
          size === "sm" && "sm:max-w-md",
          size === "md" && "sm:max-w-2xl",
          size === "lg" && "sm:max-w-4xl",
          size === "xl" && "sm:max-w-6xl"
        )}
        role="dialog"
        aria-modal="true"
        aria-label={title}
      >
        <div className="flex items-start justify-between gap-4 border-b border-slate-100 px-5 py-4">
          <div>
            <h2 className="text-lg font-bold text-navy-900">{title}</h2>
            {description && <p className="mt-0.5 text-sm text-slate-500">{description}</p>}
          </div>
          <button onClick={onClose} className="rounded-lg p-1.5 text-slate-400 hover:bg-slate-100 hover:text-slate-600" aria-label="Cerrar">
            <X className="h-5 w-5" />
          </button>
        </div>
        <div className="flex-1 overflow-y-auto px-5 py-4">{children}</div>
        {footer && <div className="flex flex-wrap justify-end gap-2 border-t border-slate-100 px-5 py-3">{footer}</div>}
      </div>
    </div>
  );
}

export function EmptyState({ title, message, action }: { title: string; message: string; action?: ReactNode }) {
  return (
    <div className="flex flex-col items-center justify-center gap-2 rounded-xl border border-dashed border-slate-300 bg-white/60 px-6 py-12 text-center">
      <Inbox className="h-8 w-8 text-slate-300" />
      <h3 className="text-sm font-semibold text-slate-700">{title}</h3>
      <p className="max-w-md text-sm text-slate-500">{message}</p>
      {action}
    </div>
  );
}

export function ErrorBanner({ message, onRetry }: { message: string; onRetry?: () => void }) {
  return (
    <div className="flex flex-wrap items-center gap-3 rounded-xl border border-red-200 bg-red-50 px-4 py-3 text-sm text-red-800">
      <AlertTriangle className="h-4 w-4 shrink-0" />
      <p className="flex-1">{message}</p>
      {onRetry && (
        <button onClick={onRetry} className="font-semibold underline underline-offset-2">
          Reintentar
        </button>
      )}
    </div>
  );
}

export function Pagination({
  page,
  pageSize,
  total,
  onChange,
}: {
  page: number;
  pageSize: number;
  total: number;
  onChange: (page: number) => void;
}) {
  const pages = Math.max(1, Math.ceil(total / pageSize));
  return (
    <div className="flex flex-wrap items-center justify-between gap-3 px-1 pt-3 text-sm text-slate-500">
      <p>
        {total === 0 ? 0 : (page - 1) * pageSize + 1}–{Math.min(page * pageSize, total)} de {total}
      </p>
      <div className="flex items-center gap-1">
        <button className="btn-ghost px-2.5 py-1.5" disabled={page <= 1} onClick={() => onChange(page - 1)}>
          Anterior
        </button>
        <span className="px-2 text-xs font-semibold text-slate-600">
          {page} / {pages}
        </span>
        <button className="btn-ghost px-2.5 py-1.5" disabled={page >= pages} onClick={() => onChange(page + 1)}>
          Siguiente
        </button>
      </div>
    </div>
  );
}

export function Field({
  label,
  hint,
  error,
  children,
  className,
}: {
  label: string;
  hint?: string;
  error?: string;
  children: ReactNode;
  className?: string;
}) {
  return (
    <label className={clsx("block", className)}>
      <span className="label">{label}</span>
      {children}
      {hint && !error && <span className="mt-1 block text-[11px] text-slate-400">{hint}</span>}
      {error && <span className="mt-1 block text-[11px] font-medium text-red-600">{error}</span>}
    </label>
  );
}

export function DemoNotice({ children }: { children?: ReactNode }) {
  return (
    <div className="flex items-start gap-2.5 rounded-lg border border-amber-200 bg-amber-50/80 px-3.5 py-2.5 text-xs leading-relaxed text-amber-900">
      <AlertTriangle className="mt-0.5 h-3.5 w-3.5 shrink-0" />
      <div>
        {children ?? (
          <>
            Entorno de <strong>demostración</strong>: los datos son sintéticos y las métricas no constituyen
            evidencia clínica.
          </>
        )}
      </div>
    </div>
  );
}

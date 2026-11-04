export function formatNumber(value: number | null | undefined, digits = 0): string {
  if (value === null || value === undefined || Number.isNaN(value)) return "—";
  return new Intl.NumberFormat("es-CO", {
    minimumFractionDigits: digits,
    maximumFractionDigits: digits,
  }).format(value);
}

export function formatPct(value: number | null | undefined, digits = 1): string {
  if (value === null || value === undefined || Number.isNaN(value)) return "—";
  return `${(value * 100).toFixed(digits)}%`;
}

export function formatDate(value: string | null | undefined): string {
  if (!value) return "—";
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return "—";
  return new Intl.DateTimeFormat("es-CO", {
    year: "numeric",
    month: "short",
    day: "2-digit",
  }).format(date);
}

export function formatDateTime(value: string | null | undefined): string {
  if (!value) return "—";
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return "—";
  return new Intl.DateTimeFormat("es-CO", {
    year: "numeric",
    month: "short",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
  }).format(date);
}

export function relativeTime(value: string | null | undefined): string {
  if (!value) return "—";
  const date = new Date(value).getTime();
  const diff = Date.now() - date;
  const minutes = Math.round(diff / 60000);
  if (minutes < 1) return "ahora";
  if (minutes < 60) return `hace ${minutes} min`;
  const hours = Math.round(minutes / 60);
  if (hours < 24) return `hace ${hours} h`;
  const days = Math.round(hours / 24);
  if (days < 30) return `hace ${days} d`;
  return formatDate(value);
}

export const RISK_LABEL: Record<string, string> = {
  bajo: "Bajo",
  moderado: "Moderado",
  alto: "Alto",
};

export const ALGORITHM_LABEL: Record<string, string> = {
  random_forest: "Random Forest",
  xgboost: "XGBoost",
  cox_ph: "Cox PH",
  xgboost_cox: "XGBoost-Cox",
  lstm_attention_cox: "LSTM-Attention-Cox",
};

export function riskBadgeClasses(level: string): string {
  if (level === "alto") return "bg-red-50 text-red-700 ring-red-200";
  if (level === "moderado") return "bg-amber-50 text-amber-700 ring-amber-200";
  return "bg-emerald-50 text-emerald-700 ring-emerald-200";
}

export function riskColor(level: string): string {
  if (level === "alto") return "#c53030";
  if (level === "moderado") return "#b7791f";
  return "#2f855a";
}

export function severityClasses(severity: string): string {
  if (severity === "critical") return "bg-red-50 text-red-700 ring-red-200";
  if (severity === "warning") return "bg-amber-50 text-amber-700 ring-amber-200";
  return "bg-sky-50 text-sky-700 ring-sky-200";
}

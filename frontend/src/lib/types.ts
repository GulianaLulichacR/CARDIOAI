export type Role =
  | "superadmin"
  | "admin"
  | "investigador"
  | "profesional_salud"
  | "analista_datos"
  | "auditor";

export interface Permissions {
  [module: string]: string[];
}

export interface User {
  id: number;
  email: string;
  full_name: string;
  role: Role;
  role_label: string;
  is_active: boolean;
  must_change_password: boolean;
  created_at: string;
  last_login_at: string | null;
  permissions: Permissions;
}

export interface LoginResponse {
  access_token: string;
  refresh_token: string;
  token_type: string;
  expires_in: number;
  user: User;
}

export interface Kpis {
  patients: number;
  datasets: number;
  predictions: number;
  predictions_window: number;
  models_ready: number;
  high_risk_window: number;
  high_risk_pct: number;
  mean_risk: number;
  unread_alerts: number;
  quality_score: number | null;
}

export interface RegionSummary {
  region: string;
  patients: number;
  predictions: number;
  mean_risk: number | null;
  high_risk: number;
  pm25: number;
  lat: number;
  lon: number;
}

export interface ModelSummary {
  id: number;
  name: string;
  algorithm: string;
  roc_auc: number | null;
  f1: number | null;
  mcc: number | null;
  is_active?: boolean;
  train_rows?: number;
}

export interface DashboardData {
  filters: { days: number; region: string | null };
  kpis: Kpis;
  risk_distribution: { level: string; count: number }[];
  age_distribution: { bucket: string; count: number }[];
  sex_distribution: { sex: string; count: number }[];
  trend: { period: string; total: number; alto: number }[];
  regions: RegionSummary[];
  model_comparison: ModelSummary[];
  active_model: {
    id: number;
    name: string;
    algorithm: string;
    roc_auc: number | null;
    trained_at: string;
    is_demo: boolean;
  } | null;
  alerts: Alert[];
  recent_activity: AuditEntry[];
  demo_mode: boolean;
  disclaimer: string;
  generated_at: string;
}

export interface Alert {
  id: number;
  type: string;
  severity: "info" | "warning" | "critical";
  title: string;
  message: string;
  module?: string | null;
  is_read?: boolean;
  created_at: string;
}

export interface AuditEntry {
  id: number;
  created_at: string;
  user_email: string;
  action: string;
  module: string;
  entity?: string | null;
  entity_id?: string | null;
  result: string;
  detail?: string | null;
  ip?: string | null;
  session_id?: string | null;
}

export interface Patient {
  id: number;
  code: string;
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
  event: boolean | null;
  followup_years: number | null;
  cohort_year: number | null;
  notes: string | null;
  is_demo: boolean;
  is_active: boolean;
  created_at: string;
  updated_at: string;
  last_prediction: {
    id: number;
    probability: number;
    risk_level: string;
    created_at: string;
  } | null;
  prediction_history?: {
    id: number;
    probability: number;
    risk_level: string;
    horizon_years: number;
    model_run_id: number;
    created_at: string;
  }[];
}

export interface Dataset {
  id: number;
  name: string;
  source: string;
  license: string;
  description: string | null;
  version: string;
  is_demo: boolean;
  status: string;
  rows_count: number;
  columns_count: number;
  checksum: string | null;
  quality: Record<string, any>;
  created_at: string;
  updated_at: string;
  download_url: string | null;
  variables_count?: number;
  models_trained?: number;
}

export interface ModelRun {
  id: number;
  name: string;
  algorithm: string;
  version: string;
  status: string;
  is_demo: boolean;
  is_active: boolean;
  dataset_id: number | null;
  horizon_years: number;
  metrics: Record<string, any>;
  summary: {
    roc_auc: number | null;
    f1: number | null;
    mcc: number | null;
    sensitivity: number | null;
    specificity: number | null;
    brier: number | null;
    pr_auc: number | null;
  };
  params: Record<string, any>;
  feature_importance: { feature: string; label: string; importance: number }[];
  risk_thresholds: Record<string, any>;
  notes: string | null;
  error: string | null;
  trained_at: string | null;
  created_at: string;
  duration_seconds: number | null;
  train_rows: number | null;
  test_rows: number | null;
  validation_strategy: string | null;
}

export interface ExplanationFactor {
  feature: string;
  label: string;
  value?: number | string | null;
  contribution: number;
  direction: string;
}

export interface Prediction {
  id: number;
  patient_id: number | null;
  model_run_id: number;
  model: {
    id: number;
    name: string;
    algorithm: string;
    version: string;
    is_demo: boolean;
    horizon_years: number;
  } | null;
  probability: number;
  risk_level: string;
  horizon_years: number;
  threshold: number | null;
  explanation: {
    method: string;
    base_value?: number;
    factors: ExplanationFactor[];
    attention?: { visit: number; year: number; weight: number }[];
    notes?: string;
    error?: string;
  };
  explanation_method: string;
  is_demo: boolean;
  created_at: string;
  disclaimer: string;
  input?: Record<string, any>;
}

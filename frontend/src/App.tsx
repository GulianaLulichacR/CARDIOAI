import { Navigate, Route, Routes, useLocation } from "react-router-dom";
import { ShieldAlert } from "lucide-react";
import type { ReactNode } from "react";
import Layout from "./components/Layout";
import { useAuth } from "./context/AuthContext";
import { Spinner } from "./components/ui";
import LoginPage from "./pages/LoginPage";
import DashboardPage from "./pages/DashboardPage";
import PatientsPage from "./pages/PatientsPage";
import DatasetsPage from "./pages/DatasetsPage";
import ModelsPage from "./pages/ModelsPage";
import PredictionPage from "./pages/PredictionPage";
import GeographyPage from "./pages/GeographyPage";
import ReportsPage from "./pages/ReportsPage";
import AuditPage from "./pages/AuditPage";
import UsersPage from "./pages/UsersPage";
import AssistantPage from "./pages/AssistantPage";
import DocsPage from "./pages/DocsPage";

function FullScreenLoader() {
  return (
    <div className="flex min-h-screen items-center justify-center bg-slate-50">
      <div className="flex flex-col items-center gap-3 text-sm text-slate-500">
        <Spinner className="h-8 w-8" />
        <p>Verificando sesión…</p>
      </div>
    </div>
  );
}

function Protected({ children }: { children: ReactNode }) {
  const { user, loading } = useAuth();
  const location = useLocation();
  if (loading) return <FullScreenLoader />;
  if (!user) return <Navigate to="/login" replace state={{ from: location.pathname }} />;
  return <>{children}</>;
}

function Forbidden() {
  return (
    <div className="mx-auto flex max-w-xl flex-col items-center gap-3 py-20 text-center">
      <span className="flex h-14 w-14 items-center justify-center rounded-2xl bg-amber-50 text-amber-600">
        <ShieldAlert className="h-7 w-7" />
      </span>
      <h1 className="text-xl font-bold text-navy-900">Acceso restringido</h1>
      <p className="text-sm text-slate-500">
        Su rol no tiene permisos para este módulo. Si necesita acceso, solicítelo a un administrador de la
        plataforma.
      </p>
    </div>
  );
}

function Gate({ module, children }: { module: string; children: ReactNode }) {
  const { can } = useAuth();
  return can(module, "read") ? <>{children}</> : <Forbidden />;
}

export default function App() {
  const { user } = useAuth();
  return (
    <Routes>
      <Route path="/login" element={user ? <Navigate to="/" replace /> : <LoginPage />} />
      <Route
        element={
          <Protected>
            <Layout />
          </Protected>
        }
      >
        <Route index element={<Gate module="dashboard"><DashboardPage /></Gate>} />
        <Route path="patients" element={<Gate module="patients"><PatientsPage /></Gate>} />
        <Route path="datasets" element={<Gate module="datasets"><DatasetsPage /></Gate>} />
        <Route path="models" element={<Gate module="models"><ModelsPage /></Gate>} />
        <Route path="prediction" element={<Gate module="predictions"><PredictionPage /></Gate>} />
        <Route path="geography" element={<Gate module="geo"><GeographyPage /></Gate>} />
        <Route path="reports" element={<Gate module="reports"><ReportsPage /></Gate>} />
        <Route path="audit" element={<Gate module="audit"><AuditPage /></Gate>} />
        <Route path="users" element={<Gate module="users"><UsersPage /></Gate>} />
        <Route path="assistant" element={<Gate module="assistant"><AssistantPage /></Gate>} />
        <Route path="docs" element={<Gate module="docs"><DocsPage /></Gate>} />
      </Route>
      <Route path="*" element={<Navigate to={user ? "/" : "/login"} replace />} />
    </Routes>
  );
}

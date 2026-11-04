import { useCallback, useEffect, useState, type FormEvent } from "react";
import { KeyRound, Pencil, Plus, Search, ShieldCheck, UserCheck, UserX } from "lucide-react";
import { api, buildQuery } from "../lib/api";
import type { User } from "../lib/types";
import { formatDateTime } from "../lib/format";
import { useAuth } from "../context/AuthContext";
import { useToast } from "../context/ToastContext";
import {
  Badge,
  ErrorBanner,
  Field,
  Modal,
  PageHeader,
  PageLoading,
  Pagination,
} from "../components/ui";

interface RoleMeta {
  key: string;
  label: string;
  description: string;
}

export default function UsersPage() {
  const { user: currentUser, can } = useAuth();
  const { notify } = useToast();
  const [items, setItems] = useState<User[]>([]);
  const [roles, setRoles] = useState<RoleMeta[]>([]);
  const [total, setTotal] = useState(0);
  const [page, setPage] = useState(1);
  const [pageSize] = useState(20);
  const [q, setQ] = useState("");
  const [role, setRole] = useState("");
  const [active, setActive] = useState("");
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");

  const [createOpen, setCreateOpen] = useState(false);
  const [editOpen, setEditOpen] = useState<User | null>(null);
  const [resetOpen, setResetOpen] = useState<User | null>(null);
  const [busy, setBusy] = useState(false);

  const [newUser, setNewUser] = useState({
    email: "",
    full_name: "",
    role: "investigador",
    password: "",
    must_change_password: true,
  });
  const [editForm, setEditForm] = useState({ full_name: "", role: "", is_active: true });
  const [newPassword, setNewPassword] = useState("");

  const load = useCallback(async () => {
    setLoading(true);
    setError("");
    try {
      const data = await api.get<{ items: User[]; total: number }>(
        `/users${buildQuery({ q, role, active: active === "" ? undefined : active === "true", page, page_size: pageSize })}`
      );
      setItems(data.items);
      setTotal(data.total);
    } catch (err) {
      setError(err instanceof Error ? err.message : "No se pudieron cargar los usuarios.");
    } finally {
      setLoading(false);
    }
  }, [q, role, active, page, pageSize]);

  useEffect(() => {
    load();
  }, [load]);

  useEffect(() => {
    api.get<RoleMeta[]>("/users/roles").then(setRoles).catch(() => undefined);
  }, []);

  const createUser = async (event: FormEvent) => {
    event.preventDefault();
    setBusy(true);
    try {
      await api.post("/users", newUser);
      notify("Usuario creado correctamente.", "success");
      setCreateOpen(false);
      setNewUser({ email: "", full_name: "", role: "investigador", password: "", must_change_password: true });
      load();
    } catch (err) {
      notify(err instanceof Error ? err.message : "No se pudo crear el usuario.", "error");
    } finally {
      setBusy(false);
    }
  };

  const startEdit = (user: User) => {
    setEditForm({ full_name: user.full_name, role: user.role, is_active: user.is_active });
    setEditOpen(user);
  };

  const saveEdit = async () => {
    if (!editOpen) return;
    setBusy(true);
    try {
      await api.patch(`/users/${editOpen.id}`, editForm);
      notify("Usuario actualizado.", "success");
      setEditOpen(null);
      load();
    } catch (err) {
      notify(err instanceof Error ? err.message : "No se pudo actualizar.", "error");
    } finally {
      setBusy(false);
    }
  };

  const resetPassword = async () => {
    if (!resetOpen) return;
    setBusy(true);
    try {
      await api.post(`/users/${resetOpen.id}/reset-password`, {
        new_password: newPassword,
        must_change_password: true,
      });
      notify("Contraseña restablecida. El usuario deberá cambiarla al ingresar.", "success");
      setResetOpen(null);
      setNewPassword("");
    } catch (err) {
      notify(err instanceof Error ? err.message : "No se pudo restablecer la contraseña.", "error");
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="space-y-4">
      <PageHeader
        eyebrow="Administración"
        title="Usuarios y roles"
        description="Control de acceso basado en roles. Cada cambio queda registrado en la bitácora; no se eliminan cuentas, solo se desactivan."
        actions={
          can("users", "write") ? (
            <button className="btn-accent" onClick={() => setCreateOpen(true)}>
              <Plus className="h-4 w-4" />
              Nuevo usuario
            </button>
          ) : undefined
        }
      />

      <div className="rounded-xl border border-sky-200 bg-sky-50 px-4 py-3 text-xs leading-relaxed text-sky-900">
        <ShieldCheck className="mr-1.5 inline h-4 w-4" />
        El principio de mínimo privilegio está activo: investigadores y analistas no administran usuarios; auditores
        solo leen bitácora, reportes y métricas. Solo un superadministrador puede crear otro superadministrador.
      </div>

      {error && <ErrorBanner message={error} onRetry={load} />}

      <section className="card p-4">
        <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
          <label className="block lg:col-span-2">
            <span className="label">Buscar</span>
            <div className="relative">
              <Search className="pointer-events-none absolute left-3 top-2.5 h-4 w-4 text-slate-400" />
              <input
                className="input pl-9"
                placeholder="Nombre o correo…"
                value={q}
                onChange={(event) => {
                  setQ(event.target.value);
                  setPage(1);
                }}
              />
            </div>
          </label>
          <label className="block">
            <span className="label">Rol</span>
            <select
              className="input"
              value={role}
              onChange={(event) => {
                setRole(event.target.value);
                setPage(1);
              }}
            >
              <option value="">Todos</option>
              {roles.map((item) => (
                <option key={item.key} value={item.key}>
                  {item.label}
                </option>
              ))}
            </select>
          </label>
          <label className="block">
            <span className="label">Estado</span>
            <select
              className="input"
              value={active}
              onChange={(event) => {
                setActive(event.target.value);
                setPage(1);
              }}
            >
              <option value="">Todos</option>
              <option value="true">Activos</option>
              <option value="false">Inactivos</option>
            </select>
          </label>
        </div>
      </section>

      <section className="card">
        <div className="card-header">
          <div>
            <h2 className="text-sm font-bold text-navy-900">Cuentas de la plataforma</h2>
            <p className="text-xs text-slate-500">{total} usuarios registrados</p>
          </div>
        </div>
        {loading ? (
          <PageLoading label="Cargando usuarios…" />
        ) : (
          <>
            <div className="table-wrap">
              <table className="data-table">
                <thead>
                  <tr>
                    <th>Usuario</th>
                    <th>Rol</th>
                    <th>Estado</th>
                    <th>Último acceso</th>
                    <th>Creado</th>
                    <th className="text-right">Acciones</th>
                  </tr>
                </thead>
                <tbody>
                  {items.map((item) => (
                    <tr key={item.id}>
                      <td>
                        <div className="flex items-center gap-2.5">
                          <span className="flex h-8 w-8 items-center justify-center rounded-full bg-navy-50 text-xs font-bold text-navy-700">
                            {item.full_name.slice(0, 1).toUpperCase()}
                          </span>
                          <div>
                            <p className="text-xs font-semibold text-navy-900">{item.full_name}</p>
                            <p className="text-[10px] text-slate-400">{item.email}</p>
                          </div>
                        </div>
                      </td>
                      <td><Badge tone="info">{item.role_label}</Badge></td>
                      <td>
                        <Badge tone={item.is_active ? "success" : "danger"}>
                          {item.is_active ? "Activo" : "Inactivo"}
                        </Badge>
                      </td>
                      <td className="text-[11px]">{item.last_login_at ? formatDateTime(item.last_login_at) : "Nunca"}</td>
                      <td className="text-[11px]">{formatDateTime(item.created_at)}</td>
                      <td>
                        <div className="flex justify-end gap-1">
                          <button
                            className="rounded-md p-1.5 text-slate-500 hover:bg-slate-100"
                            title="Editar"
                            onClick={() => startEdit(item)}
                            disabled={!can("users", "write")}
                          >
                            <Pencil className="h-4 w-4" />
                          </button>
                          <button
                            className="rounded-md p-1.5 text-amber-600 hover:bg-amber-50"
                            title="Restablecer contraseña"
                            onClick={() => setResetOpen(item)}
                            disabled={!can("users", "write")}
                          >
                            <KeyRound className="h-4 w-4" />
                          </button>
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

      <section className="card p-5">
        <h2 className="text-sm font-bold text-navy-900">Matriz de roles</h2>
        <div className="mt-3 grid gap-3 md:grid-cols-2 xl:grid-cols-3">
          {roles.map((item) => (
            <div key={item.key} className="rounded-lg border border-slate-100 bg-slate-50/50 p-3.5">
              <div className="flex items-center gap-2">
                {item.key === "auditor" ? (
                  <ShieldCheck className="h-4 w-4 text-navy-600" />
                ) : item.key === "superadmin" ? (
                  <UserCheck className="h-4 w-4 text-emerald-600" />
                ) : (
                  <UserX className="h-4 w-4 text-slate-400" />
                )}
                <p className="text-xs font-bold text-navy-900">{item.label}</p>
              </div>
              <p className="mt-1 text-[11px] leading-relaxed text-slate-500">{item.description}</p>
            </div>
          ))}
        </div>
      </section>

      <Modal
        open={createOpen}
        onClose={() => setCreateOpen(false)}
        title="Nuevo usuario"
        description="La cuenta se crea activa. La contraseña debe cumplir la política de seguridad."
        size="md"
        footer={
          <>
            <button className="btn-ghost" onClick={() => setCreateOpen(false)}>Cancelar</button>
            <button className="btn-accent" form="user-form" disabled={busy}>
              {busy ? "Creando…" : "Crear usuario"}
            </button>
          </>
        }
      >
        <form id="user-form" onSubmit={createUser} className="space-y-3">
          <Field label="Nombre completo">
            <input className="input" value={newUser.full_name} onChange={(e) => setNewUser({ ...newUser, full_name: e.target.value })} required minLength={3} />
          </Field>
          <Field label="Correo electrónico">
            <input type="email" className="input" value={newUser.email} onChange={(e) => setNewUser({ ...newUser, email: e.target.value })} required />
          </Field>
          <Field label="Rol">
            <select className="input" value={newUser.role} onChange={(e) => setNewUser({ ...newUser, role: e.target.value })}>
              {roles
                .filter((item) => item.key !== "superadmin" || currentUser?.role === "superadmin")
                .map((item) => (
                  <option key={item.key} value={item.key}>
                    {item.label}
                  </option>
                ))}
            </select>
          </Field>
          <Field
            label="Contraseña temporal"
            hint="Mínimo 10 caracteres, mayúscula, minúscula, número y símbolo. El usuario deberá cambiarla."
          >
            <input type="password" className="input" value={newUser.password} onChange={(e) => setNewUser({ ...newUser, password: e.target.value })} required minLength={10} />
          </Field>
        </form>
      </Modal>

      <Modal
        open={editOpen !== null}
        onClose={() => setEditOpen(null)}
        title={editOpen ? `Editar ${editOpen.full_name}` : "Editar usuario"}
        description={editOpen?.email}
        size="sm"
        footer={
          <>
            <button className="btn-ghost" onClick={() => setEditOpen(null)}>Cancelar</button>
            <button className="btn-accent" onClick={saveEdit} disabled={busy}>
              {busy ? "Guardando…" : "Guardar cambios"}
            </button>
          </>
        }
      >
        <div className="space-y-3">
          <Field label="Nombre completo">
            <input className="input" value={editForm.full_name} onChange={(e) => setEditForm({ ...editForm, full_name: e.target.value })} />
          </Field>
          <Field label="Rol">
            <select className="input" value={editForm.role} onChange={(e) => setEditForm({ ...editForm, role: e.target.value })}>
              {roles
                .filter((item) => item.key !== "superadmin" || currentUser?.role === "superadmin")
                .map((item) => (
                  <option key={item.key} value={item.key}>
                    {item.label}
                  </option>
                ))}
            </select>
          </Field>
          <Field label="Estado">
            <select className="input" value={editForm.is_active ? "true" : "false"} onChange={(e) => setEditForm({ ...editForm, is_active: e.target.value === "true" })}>
              <option value="true">Activo</option>
              <option value="false">Inactivo</option>
            </select>
          </Field>
        </div>
      </Modal>

      <Modal
        open={resetOpen !== null}
        onClose={() => setResetOpen(null)}
        title="Restablecer contraseña"
        description={resetOpen ? `Se asignará una nueva contraseña temporal a ${resetOpen.email}.` : undefined}
        size="sm"
        footer={
          <>
            <button className="btn-ghost" onClick={() => setResetOpen(null)}>Cancelar</button>
            <button className="btn-accent" onClick={resetPassword} disabled={busy || newPassword.length < 10}>
              {busy ? "Guardando…" : "Restablecer"}
            </button>
          </>
        }
      >
        <Field label="Nueva contraseña temporal" hint="Mínimo 10 caracteres, con mayúscula, minúscula, número y símbolo.">
          <input type="password" className="input" value={newPassword} onChange={(e) => setNewPassword(e.target.value)} />
        </Field>
      </Modal>
    </div>
  );
}

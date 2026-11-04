import { createContext, useCallback, useContext, useEffect, useMemo, useState, type ReactNode } from "react";
import { api, clearSession, getToken, setSession, storedUser } from "../lib/api";
import type { LoginResponse, User } from "../lib/types";

interface AuthContextValue {
  user: User | null;
  loading: boolean;
  login: (email: string, password: string) => Promise<void>;
  logout: () => Promise<void>;
  refreshUser: () => Promise<void>;
  can: (module: string, action?: string) => boolean;
}

const AuthContext = createContext<AuthContextValue | null>(null);

export function AuthProvider({ children }: { children: ReactNode }) {
  const [user, setUser] = useState<User | null>(() => storedUser<User>());
  const [loading, setLoading] = useState(true);

  const logout = useCallback(async () => {
    try {
      if (getToken()) await api.post("/auth/logout");
    } catch {
      /* el token puede haber expirado */
    }
    clearSession();
    setUser(null);
  }, []);

  const refreshUser = useCallback(async () => {
    if (!getToken()) {
      setUser(null);
      setLoading(false);
      return;
    }
    try {
      const me = await api.get<User>("/auth/me");
      setUser(me);
      const stored = localStorage.getItem("cardioai.user");
      if (stored) {
        const current = JSON.parse(stored);
        localStorage.setItem("cardioai.user", JSON.stringify({ ...current, ...me }));
      }
    } catch {
      clearSession();
      setUser(null);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    refreshUser();
    const onUnauthorized = () => {
      setUser(null);
      if (!window.location.pathname.startsWith("/login")) {
        window.location.href = "/login";
      }
    };
    window.addEventListener("cardioai:unauthorized", onUnauthorized);
    return () => window.removeEventListener("cardioai:unauthorized", onUnauthorized);
  }, [refreshUser]);

  const login = useCallback(async (email: string, password: string) => {
    const response = await api.post<LoginResponse>("/auth/login", { email, password });
    setSession(response.access_token, response.refresh_token, response.user);
    setUser(response.user);
  }, []);

  const can = useCallback(
    (module: string, action = "read") => {
      if (!user) return false;
      if (user.role === "superadmin") return true;
      return Boolean(user.permissions?.[module]?.includes(action));
    },
    [user]
  );

  const value = useMemo(
    () => ({ user, loading, login, logout, refreshUser, can }),
    [user, loading, login, logout, refreshUser, can]
  );

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
}

export function useAuth(): AuthContextValue {
  const context = useContext(AuthContext);
  if (!context) throw new Error("useAuth debe usarse dentro de AuthProvider");
  return context;
}

import { useEffect, useRef, useState, type FormEvent } from "react";
import {
  BookOpen,
  Bot,
  ExternalLink,
  Info,
  Send,
  ShieldCheck,
  Sparkles,
  UserRound,
} from "lucide-react";
import { api } from "../lib/api";
import { Badge, DemoNotice, PageHeader, Spinner } from "../components/ui";

interface Source {
  title: string;
  source: string;
  url: string | null;
  year: number | null;
}

interface Answer {
  answer: string;
  sources: Source[];
  grounded: boolean;
  mode: string;
  disclaimer: string;
  suggested?: string[];
}

interface Message {
  role: "user" | "assistant";
  content: string;
  meta?: Answer;
}

interface DocumentItem {
  id: number;
  title: string;
  source: string;
  url: string | null;
  year: number | null;
  tags: string[];
}

const INITIAL: Message = {
  role: "assistant",
  content:
    "Soy el asistente documental de CARDIOAI. Respondo con base en los documentos citados y en las métricas registradas en la plataforma. No emito diagnósticos ni recomendaciones de tratamiento.",
};

export default function AssistantPage() {
  const [messages, setMessages] = useState<Message[]>([INITIAL]);
  const [question, setQuestion] = useState("");
  const [busy, setBusy] = useState(false);
  const [documents, setDocuments] = useState<DocumentItem[]>([]);
  const [suggestions, setSuggestions] = useState<string[]>([]);
  const scrollRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    api.get<DocumentItem[]>("/assistant/documents").then(setDocuments).catch(() => undefined);
    api
      .get<{ suggestions: string[] }>("/assistant/suggestions")
      .then((data) => setSuggestions(data.suggestions))
      .catch(() => undefined);
  }, []);

  useEffect(() => {
    scrollRef.current?.scrollTo({ top: scrollRef.current.scrollHeight, behavior: "smooth" });
  }, [messages, busy]);

  const ask = async (text: string) => {
    const trimmed = text.trim();
    if (trimmed.length < 3 || busy) return;
    setMessages((current) => [...current, { role: "user", content: trimmed }]);
    setQuestion("");
    setBusy(true);
    try {
      const answer = await api.post<Answer>("/assistant/query", { question: trimmed });
      setMessages((current) => [...current, { role: "assistant", content: answer.answer, meta: answer }]);
      if (answer.suggested) setSuggestions(answer.suggested);
    } catch (err) {
      setMessages((current) => [
        ...current,
        {
          role: "assistant",
          content: err instanceof Error ? err.message : "No se pudo obtener respuesta del asistente.",
        },
      ]);
    } finally {
      setBusy(false);
    }
  };

  const submit = (event: FormEvent) => {
    event.preventDefault();
    ask(question);
  };

  return (
    <div className="space-y-4">
      <PageHeader
        eyebrow="Conocimiento"
        title="Asistente documental"
        description="Recuperación aumentada sobre una base documental verificable y las métricas de la plataforma. Modo extractivo por defecto, con citas y rechazo explícito de consejo clínico individual."
        actions={<Badge tone="info"><BookOpen className="h-3 w-3" /> {documents.length} fuentes</Badge>}
      />

      <DemoNotice>
        Las respuestas se basan en documentos citados y en métricas de demostración. No son diagnóstico ni
        recomendación clínica. Verifique la fuente original antes de citar cifras.
      </DemoNotice>

      <div className="grid gap-4 xl:grid-cols-[1.25fr_0.75fr]">
        <section className="card flex h-[640px] flex-col overflow-hidden">
          <div className="flex items-center gap-3 border-b border-slate-100 px-5 py-3">
            <span className="flex h-9 w-9 items-center justify-center rounded-xl bg-accent-50 text-accent-700">
              <Bot className="h-5 w-5" />
            </span>
            <div>
              <p className="text-sm font-bold text-navy-900">Asistente CARDIOAI</p>
              <p className="text-[11px] text-slate-500">RAG extractivo · citas verificables · sin consejo clínico</p>
            </div>
          </div>

          <div ref={scrollRef} className="flex-1 space-y-4 overflow-y-auto bg-slate-50/50 px-4 py-4">
            {messages.map((message, index) => (
              <div key={index} className={message.role === "user" ? "flex justify-end" : "flex justify-start"}>
                <div
                  className={
                    message.role === "user"
                      ? "max-w-[85%] rounded-2xl rounded-br-sm bg-navy-900 px-4 py-2.5 text-sm leading-relaxed text-white"
                      : "max-w-[92%] rounded-2xl rounded-bl-sm border border-slate-200 bg-white px-4 py-3 text-sm leading-relaxed text-slate-700 shadow-sm"
                  }
                >
                  <div className="mb-1 flex items-center gap-2 text-[10px] font-semibold uppercase tracking-wide opacity-70">
                    {message.role === "user" ? <UserRound className="h-3 w-3" /> : <Sparkles className="h-3 w-3" />}
                    {message.role === "user" ? "Usted" : "Asistente"}
                  </div>
                  <p className="whitespace-pre-wrap">{message.content}</p>
                  {message.meta && (
                    <div className="mt-3 space-y-2 border-t border-slate-100 pt-2">
                      <div className="flex flex-wrap items-center gap-1.5">
                        <Badge tone={message.meta.grounded ? "success" : "warning"}>
                          {message.meta.grounded ? "Con fuentes" : "Sin evidencia suficiente"}
                        </Badge>
                        <Badge tone="neutral">{message.meta.mode}</Badge>
                      </div>
                      {message.meta.sources.length > 0 && (
                        <div className="space-y-1">
                          {message.meta.sources.map((source, sourceIndex) => (
                            <div key={`${source.title}-${sourceIndex}`} className="flex items-start gap-1.5 text-[11px] text-slate-500">
                              <span className="font-bold text-accent-700">[{sourceIndex + 1}]</span>
                              <span>
                                {source.url ? (
                                  <a href={source.url} target="_blank" rel="noreferrer" className="font-medium text-accent-700 hover:underline">
                                    {source.title} <ExternalLink className="inline h-2.5 w-2.5" />
                                  </a>
                                ) : (
                                  <span className="font-medium">{source.title}</span>
                                )}
                                {" — "}
                                {source.source}
                                {source.year ? ` (${source.year})` : ""}
                              </span>
                            </div>
                          ))}
                        </div>
                      )}
                      <p className="text-[10px] leading-snug text-slate-400">{message.meta.disclaimer}</p>
                    </div>
                  )}
                </div>
              </div>
            ))}
            {busy && (
              <div className="flex items-center gap-2 rounded-xl border border-slate-200 bg-white px-4 py-3 text-sm text-slate-500">
                <Spinner className="h-4 w-4" />
                Recuperando fragmentos y redactando respuesta…
              </div>
            )}
          </div>

          <div className="border-t border-slate-100 bg-white px-4 py-3">
            <div className="mb-2 flex flex-wrap gap-1.5">
              {suggestions.slice(0, 3).map((item) => (
                <button
                  key={item}
                  onClick={() => ask(item)}
                  className="rounded-full border border-slate-200 px-2.5 py-1 text-[10px] font-medium text-slate-600 hover:border-accent-300 hover:bg-accent-50 hover:text-accent-700"
                >
                  {item}
                </button>
              ))}
            </div>
            <form onSubmit={submit} className="flex items-end gap-2">
              <textarea
                value={question}
                onChange={(event) => setQuestion(event.target.value)}
                placeholder="Pregunte sobre metodología, métricas, factores de riesgo o limitaciones…"
                className="input min-h-[44px] flex-1 resize-none"
                rows={2}
                maxLength={800}
                onKeyDown={(event) => {
                  if (event.key === "Enter" && !event.shiftKey) {
                    event.preventDefault();
                    ask(question);
                  }
                }}
              />
              <button className="btn-accent h-[44px]" type="submit" disabled={busy || question.trim().length < 3}>
                <Send className="h-4 w-4" />
              </button>
            </form>
            <p className="mt-1 text-[10px] text-slate-400">
              Enter para enviar · Shift+Enter para nueva línea · máximo 800 caracteres
            </p>
          </div>
        </section>

        <div className="space-y-4">
          <section className="card">
            <div className="card-header">
              <div>
                <h2 className="text-sm font-bold text-navy-900">Base documental</h2>
                <p className="text-xs text-slate-500">Fuentes primarias y metodológicas incluidas.</p>
              </div>
            </div>
            <div className="max-h-[360px] divide-y divide-slate-50 overflow-y-auto">
              {documents.map((document) => (
                <div key={document.id} className="px-4 py-3">
                  <p className="text-xs font-semibold leading-snug text-navy-900">{document.title}</p>
                  <p className="mt-0.5 text-[10px] text-slate-500">
                    {document.source} {document.year ? `· ${document.year}` : ""}
                  </p>
                  <div className="mt-1 flex flex-wrap items-center gap-1">
                    {document.tags.slice(0, 3).map((tag) => (
                      <span key={tag} className="rounded bg-slate-100 px-1.5 py-0.5 text-[9px] text-slate-500">
                        {tag}
                      </span>
                    ))}
                    {document.url && (
                      <a href={document.url} target="_blank" rel="noreferrer" className="ml-auto text-[10px] text-accent-700 hover:underline">
                        Abrir fuente
                      </a>
                    )}
                  </div>
                </div>
              ))}
            </div>
          </section>

          <section className="card p-4">
            <div className="flex items-center gap-2">
              <ShieldCheck className="h-4 w-4 text-emerald-600" />
              <h2 className="text-sm font-bold text-navy-900">Salvaguardas</h2>
            </div>
            <ul className="mt-2 space-y-1.5 text-[11px] leading-relaxed text-slate-500">
              <li>• Rechaza solicitudes de dosis, tratamientos o diagnóstico individual.</li>
              <li>• Responde solo con fragmentos recuperados o métricas registradas.</li>
              <li>• Cita la fuente y advierte cuando no hay evidencia suficiente.</li>
              <li>• No modifica el modelo predictivo ni genera predicciones.</li>
            </ul>
          </section>

          <div className="rounded-xl border border-slate-200 bg-white p-3.5 text-[11px] leading-relaxed text-slate-500">
            <Info className="mr-1 inline h-3.5 w-3.5" />
            El modo por defecto es extractivo (TF-IDF + similitud coseno). Si se configura un proveedor LLM con
            credenciales, solo redacta sobre el contexto recuperado; el modelo predictivo nunca se sustituye.
          </div>
        </div>
      </div>
    </div>
  );
}

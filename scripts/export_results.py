"""Exporta resultados científicos reproducibles desde la base de CARDIOAI.

Uso:
    .venv\\Scripts\\python.exe scripts\\export_results.py --run-ids 1,2,3,4,5
    .venv\\Scripts\\python.exe scripts\\export_results.py --latest --dataset-id 2

Genera en data/results/:
    model_comparison.csv / .json
    table1_baseline.csv
    table2_discrimination.csv
    table3_calibration.csv
    table4_subgroups.csv
    roc_curves.csv, calibration_curves.csv, dca_curves.csv
    figures/*.png (300 dpi)
    RESULTS.md
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

DEFAULT_DB = ROOT / "data" / "experiment.db"
os.environ.setdefault("DATABASE_URL", f"sqlite:///{DEFAULT_DB.as_posix()}")

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import norm  # noqa: E402

from backend.app.database import SessionLocal  # noqa: E402
from backend.app.models import Dataset, ModelRun  # noqa: E402

RESULTS_DIR = ROOT / "data" / "results"
FIGURES_DIR = RESULTS_DIR / "figures"

ALGO_LABELS = {
    "random_forest": "Random Forest",
    "xgboost": "XGBoost",
    "cox_ph": "Cox PH",
    "xgboost_cox": "XGBoost-Cox",
    "lstm_attention_cox": "LSTM-Attention-Cox",
}
ALGO_ORDER = list(ALGO_LABELS)


# ---------------------------------------------------------------------------
# DeLong para comparar AUC pareadas
# ---------------------------------------------------------------------------
def _delong_components(y: np.ndarray, p: np.ndarray):
    order = np.argsort(-p)
    p = p[order]
    y = y[order]
    m = int(y.sum())
    n = int(len(y) - m)
    if m == 0 or n == 0:
        raise ValueError("DeLong requiere ambas clases.")
    pos = np.where(y == 1)[0]
    neg = np.where(y == 0)[0]
    # Matriz de comparaciones (m x n)
    v10 = np.zeros(m)
    v01 = np.zeros(n)
    for i, pi in enumerate(p[pos]):
        v10[i] = np.mean((pi > p[neg]).astype(float) + 0.5 * (pi == p[neg]))
    for j, pj in enumerate(p[neg]):
        v01[j] = np.mean((p[pos] > pj).astype(float) + 0.5 * (p[pos] == pj))
    auc = float(np.mean(v10))
    return auc, v10, v01


def delong_test(y: np.ndarray, p1: np.ndarray, p2: np.ndarray) -> dict[str, float] | None:
    y = np.asarray(y).astype(int)
    if len(np.unique(y)) < 2:
        return None
    auc1, v10_1, v01_1 = _delong_components(y, p1)
    auc2, v10_2, v01_2 = _delong_components(y, p2)
    s10 = np.cov(np.vstack([v10_1, v10_2]))
    s01 = np.cov(np.vstack([v01_1, v01_2]))
    s = s10 / len(v10_1) + s01 / len(v01_1)
    var = s[0, 0] + s[1, 1] - 2 * s[0, 1]
    if var <= 0:
        return None
    z = (auc1 - auc2) / np.sqrt(var)
    return {
        "auc_1": round(auc1, 4),
        "auc_2": round(auc2, 4),
        "difference": round(auc1 - auc2, 4),
        "z": round(float(z), 3),
        "p_value": round(float(2 * norm.sf(abs(z))), 6),
    }


# ---------------------------------------------------------------------------
# Selección de runs
# ---------------------------------------------------------------------------
def select_runs(db, run_ids: list[int] | None, latest: bool, dataset_id: int | None) -> list[ModelRun]:
    if run_ids:
        runs = db.query(ModelRun).filter(ModelRun.id.in_(run_ids)).all()
        found = {r.id for r in runs}
        missing = [i for i in run_ids if i not in found]
        if missing:
            raise SystemExit(f"No existen los runs: {missing}")
        return sorted(runs, key=lambda r: r.id)
    if not latest:
        raise SystemExit("Indique --run-ids o --latest.")
    query = db.query(ModelRun).filter(ModelRun.status == "ready")
    if dataset_id is not None:
        query = query.filter(ModelRun.dataset_id == dataset_id)
    rows = query.order_by(ModelRun.trained_at.desc()).all()
    best: dict[str, ModelRun] = {}
    for run in rows:
        best.setdefault(run.algorithm, run)
    return sorted(best.values(), key=lambda r: ALGO_ORDER.index(r.algorithm) if r.algorithm in ALGO_ORDER else 99)


def run_frame(run: ModelRun) -> pd.DataFrame:
    preds = (run.metrics or {}).get("test_predictions") or []
    if not preds:
        return pd.DataFrame(columns=["y", "p"])
    return pd.DataFrame(preds)


def dataset_of(db, run: ModelRun) -> Dataset | None:
    return db.get(Dataset, run.dataset_id) if run.dataset_id else None


# ---------------------------------------------------------------------------
# Tabla 1
# ---------------------------------------------------------------------------
def build_table1(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    groups = [("Global", df), ("Sin evento", df[~df["event"].astype(bool)]), ("Con evento", df[df["event"].astype(bool)])]
    numeric = [
        "age", "bmi", "systolic_bp", "diastolic_bp", "cholesterol_total",
        "glucose", "cigs_per_day", "heart_rate",
    ]
    for label, frame in groups:
        for col in numeric:
            if col not in frame.columns:
                continue
            values = pd.to_numeric(frame[col], errors="coerce").dropna()
            rows.append({
                "group": label,
                "variable": col,
                "n": int(values.size),
                "mean": round(float(values.mean()), 2) if values.size else None,
                "sd": round(float(values.std(ddof=1)), 2) if values.size > 1 else None,
                "median": round(float(values.median()), 2) if values.size else None,
                "missing_pct": round(float(frame[col].isna().mean() * 100), 2),
            })
        for col in ["sex", "smoking", "education", "diabetes", "hypertension"]:
            if col not in frame.columns:
                continue
            counts = frame[col].astype(str).value_counts(dropna=False)
            for value, count in counts.items():
                rows.append({
                    "group": label,
                    "variable": f"{col}={value}",
                    "n": int(count),
                    "mean": round(float(count / max(len(frame), 1) * 100), 2),
                    "sd": None,
                    "median": None,
                    "missing_pct": None,
                })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Exportación
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-ids", default=None, help="IDs separados por coma")
    parser.add_argument("--latest", action="store_true", help="Último run listo por algoritmo")
    parser.add_argument("--dataset-id", type=int, default=None)
    parser.add_argument("--tag", default="", help="Sufijo opcional para los archivos")
    args = parser.parse_args()

    run_ids = [int(x) for x in args.run_ids.split(",")] if args.run_ids else None
    db = SessionLocal()
    try:
        runs = select_runs(db, run_ids, args.latest, args.dataset_id)
        if not runs:
            raise SystemExit("No hay runs listos para exportar.")
        ready = [r for r in runs if r.status == "ready"]
        failed = [r for r in runs if r.status != "ready"]
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        FIGURES_DIR.mkdir(parents=True, exist_ok=True)

        suffix = f"_{args.tag}" if args.tag else ""
        split_type = None
        dataset = None
        frames: dict[str, pd.DataFrame] = {}

        comparison = []
        disc_rows = []
        calib_rows = []
        subgroup_rows = []
        roc_rows = []
        dca_rows = []
        calibration_curve_rows = []

        for run in ready:
            metrics = run.metrics or {}
            disc = metrics.get("discrimination") or {}
            temporal = metrics.get("temporal") or {}
            split_type = split_type or temporal.get("split_type")
            dataset = dataset or dataset_of(db, run)
            frame = run_frame(run)
            frames[run.algorithm] = frame
            label = ALGO_LABELS.get(run.algorithm, run.algorithm)
            summary = {
                "run_id": run.id,
                "algorithm": run.algorithm,
                "model": label,
                "status": run.status,
                "split_type": temporal.get("split_type"),
                "n_test": temporal.get("n"),
                "events_test": disc.get("positives"),
                "prevalence_test": temporal.get("prevalence"),
                "threshold": disc.get("threshold"),
                "auc_test": disc.get("roc_auc"),
                "auc_ci_low": (disc.get("auc_ci") or [None, None])[0],
                "auc_ci_high": (disc.get("auc_ci") or [None, None])[1],
                "pr_auc": disc.get("pr_auc"),
                "brier": disc.get("brier"),
                "sensitivity": disc.get("sensitivity"),
                "specificity": disc.get("specificity"),
                "f1": disc.get("f1"),
                "mcc": disc.get("mcc"),
                "accuracy": disc.get("accuracy"),
                "calibration_slope": disc.get("calibration_slope"),
                "calibration_intercept": disc.get("calibration_intercept"),
                "calibration_ece": disc.get("calibration_ece"),
                "hosmer_lemeshow_p": (disc.get("hosmer_lemeshow") or {}).get("p_value"),
                "selection_auc": (metrics.get("selection") or {}).get("roc_auc"),
                "selection_strategy": (metrics.get("selection") or {}).get("strategy"),
                "cv_auc_mean": (metrics.get("cv") or {}).get("roc_auc_mean"),
                "cv_auc_std": (metrics.get("cv") or {}).get("roc_auc_std"),
                "duration_seconds": run.duration_seconds,
                "train_rows": run.train_rows,
                "test_rows": run.test_rows,
                "features_used": ", ".join(metrics.get("features_used") or []),
                "artifact_sha256": (run.params or {}).get("artifact_sha256"),
            }
            comparison.append(summary)
            disc_rows.append({
                "model": label,
                "auc": disc.get("roc_auc"),
                "auc_ci": f"{summary['auc_ci_low']}-{summary['auc_ci_high']}",
                "pr_auc": disc.get("pr_auc"),
                "brier": disc.get("brier"),
                "sensitivity": disc.get("sensitivity"),
                "specificity": disc.get("specificity"),
                "f1": disc.get("f1"),
                "mcc": disc.get("mcc"),
                "threshold": disc.get("threshold"),
            })
            calib_rows.append({
                "model": label,
                "slope": disc.get("calibration_slope"),
                "intercept": disc.get("calibration_intercept"),
                "ece": disc.get("calibration_ece"),
                "hosmer_lemeshow_chi2": (disc.get("hosmer_lemeshow") or {}).get("chi2"),
                "hosmer_lemeshow_df": (disc.get("hosmer_lemeshow") or {}).get("df"),
                "hosmer_lemeshow_p": (disc.get("hosmer_lemeshow") or {}).get("p_value"),
            })
            for item in metrics.get("subgroups") or []:
                subgroup_rows.append({
                    "model": label,
                    **{k: item.get(k) for k in ("variable", "group", "n", "events", "roc_auc", "pr_auc", "brier", "calibration_slope", "calibration_intercept")},
                })
            roc = disc.get("roc_curve") or {}
            for fpr, tpr in zip(roc.get("fpr") or [], roc.get("tpr") or []):
                roc_rows.append({"model": label, "fpr": fpr, "tpr": tpr})
            for item in disc.get("calibration") or []:
                calibration_curve_rows.append({"model": label, **item})
            for item in disc.get("dca") or []:
                dca_rows.append({"model": label, **item})

        comparison_df = pd.DataFrame(comparison)
        comparison_df.to_csv(RESULTS_DIR / f"model_comparison{suffix}.csv", index=False)
        (RESULTS_DIR / f"model_comparison{suffix}.json").write_text(
            json.dumps({"split_type": split_type, "models": comparison}, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        pd.DataFrame(disc_rows).to_csv(RESULTS_DIR / f"table2_discrimination{suffix}.csv", index=False)
        pd.DataFrame(calib_rows).to_csv(RESULTS_DIR / f"table3_calibration{suffix}.csv", index=False)
        pd.DataFrame(subgroup_rows).to_csv(RESULTS_DIR / f"table4_subgroups{suffix}.csv", index=False)
        pd.DataFrame(roc_rows).to_csv(RESULTS_DIR / f"roc_curves{suffix}.csv", index=False)
        pd.DataFrame(calibration_curve_rows).to_csv(RESULTS_DIR / f"calibration_curves{suffix}.csv", index=False)
        pd.DataFrame(dca_rows).to_csv(RESULTS_DIR / f"dca_curves{suffix}.csv", index=False)

        if dataset and dataset.file_path and Path(dataset.file_path).exists():
            full = pd.read_csv(dataset.file_path)
            build_table1(full).to_csv(RESULTS_DIR / f"table1_baseline{suffix}.csv", index=False)

        # Comparaciones pareadas DeLong sobre el mismo holdout.
        delong_rows = []
        algos = [r.algorithm for r in ready if not frames.get(r.algorithm, pd.DataFrame()).empty]
        for i, a in enumerate(algos):
            for b in algos[i + 1:]:
                fa, fb = frames[a], frames[b]
                if len(fa) != len(fb) or not np.array_equal(fa["y"].to_numpy(), fb["y"].to_numpy()):
                    continue
                result = delong_test(fa["y"].to_numpy(), fa["p"].to_numpy(), fb["p"].to_numpy())
                if result:
                    delong_rows.append({"model_a": ALGO_LABELS.get(a, a), "model_b": ALGO_LABELS.get(b, b), **result})
        pd.DataFrame(delong_rows).to_csv(RESULTS_DIR / f"table5_delong{suffix}.csv", index=False)

        # Figuras
        _plot_roc(roc_rows, suffix)
        _plot_calibration(calibration_curve_rows, suffix)
        _plot_dca(dca_rows, suffix)
        _plot_auc_forest(comparison_df, suffix)
        _plot_importance(ready, suffix)

        _write_markdown(comparison_df, delong_rows, split_type, dataset, failed, suffix)
        print(json.dumps({
            "exported": [r.id for r in ready],
            "failed": [r.id for r in failed],
            "split_type": split_type,
            "directory": str(RESULTS_DIR),
        }, ensure_ascii=False, indent=2))
    finally:
        db.close()


def _plot_roc(rows: list[dict], suffix: str) -> None:
    df = pd.DataFrame(rows)
    if df.empty:
        return
    fig, ax = plt.subplots(figsize=(6, 6))
    for model, grp in df.groupby("model"):
        ax.plot(grp["fpr"], grp["tpr"], label=model, linewidth=1.8)
    ax.plot([0, 1], [0, 1], "--", color="grey", linewidth=1)
    ax.set_xlabel("1 − Especificidad")
    ax.set_ylabel("Sensibilidad")
    ax.set_title("Curvas ROC en el conjunto de prueba")
    ax.legend(frameon=False, fontsize=8)
    ax.set_aspect("equal")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / f"fig_roc{suffix}.png", dpi=300)
    plt.close(fig)


def _plot_calibration(rows: list[dict], suffix: str) -> None:
    df = pd.DataFrame(rows)
    if df.empty:
        return
    fig, ax = plt.subplots(figsize=(6, 5))
    for model, grp in df.groupby("model"):
        ax.plot(grp["mean_pred"], grp["frac_pos"], marker="o", markersize=4, label=model, linewidth=1.5)
    ax.plot([0, 1], [0, 1], "--", color="grey", linewidth=1)
    ax.set_xlabel("Probabilidad predicha")
    ax.set_ylabel("Frecuencia observada")
    ax.set_title("Calibración por deciles")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / f"fig_calibration{suffix}.png", dpi=300)
    plt.close(fig)


def _plot_dca(rows: list[dict], suffix: str) -> None:
    df = pd.DataFrame(rows)
    if df.empty:
        return
    fig, ax = plt.subplots(figsize=(7, 5))
    for model, grp in df.groupby("model"):
        ax.plot(grp["threshold"], grp["model"], label=model, linewidth=1.6)
    first = next(df.groupby("model"))
    ax.plot(first[1]["threshold"], first[1]["treat_all"], "--", color="grey", label="Tratar a todos", linewidth=1)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Probabilidad umbral")
    ax.set_ylabel("Beneficio neto")
    ax.set_title("Análisis de decisión clínica")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / f"fig_dca{suffix}.png", dpi=300)
    plt.close(fig)


def _plot_auc_forest(df: pd.DataFrame, suffix: str) -> None:
    if df.empty or df["auc_test"].isna().all():
        return
    plot_df = df.dropna(subset=["auc_test"]).sort_values("auc_test")
    fig, ax = plt.subplots(figsize=(6, max(3, 0.6 * len(plot_df) + 1.5)))
    y = np.arange(len(plot_df))
    low = plot_df["auc_ci_low"].fillna(plot_df["auc_test"])
    high = plot_df["auc_ci_high"].fillna(plot_df["auc_test"])
    ax.errorbar(
        plot_df["auc_test"], y,
        xerr=[plot_df["auc_test"] - low, high - plot_df["auc_test"]],
        fmt="o", capsize=4, color="#0f766e",
    )
    ax.set_yticks(y)
    ax.set_yticklabels(plot_df["model"])
    ax.set_xlabel("AUC-ROC (IC 95 %)")
    ax.set_title("Discriminación en el holdout")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / f"fig_auc_forest{suffix}.png", dpi=300)
    plt.close(fig)


def _plot_importance(runs: list[ModelRun], suffix: str) -> None:
    rows = []
    for run in runs:
        label = ALGO_LABELS.get(run.algorithm, run.algorithm)
        for item in run.feature_importance or []:
            rows.append({
                "model": label,
                "feature": item.get("label") or item.get("feature"),
                "importance": item.get("importance"),
            })
    if not rows:
        return
    df = pd.DataFrame(rows)
    top = (
        df.groupby("feature")["importance"].max().sort_values(ascending=False).head(12).index
    )
    df = df[df["feature"].isin(top)]
    pivot = df.pivot_table(index="feature", columns="model", values="importance", aggfunc="max").loc[top]
    fig, ax = plt.subplots(figsize=(8, 0.45 * len(pivot) + 1.5))
    pivot.plot(kind="barh", ax=ax, width=0.8)
    ax.set_xlabel("Importancia normalizada")
    ax.set_ylabel("")
    ax.set_title("Importancia de variables por modelo")
    ax.legend(frameon=False, fontsize=7)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / f"fig_importance{suffix}.png", dpi=300)
    plt.close(fig)


def _fmt(value, digits: int = 3) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "—"
    return f"{value:.{digits}f}" if isinstance(value, float) else str(value)


def _write_markdown(df: pd.DataFrame, delong: list[dict], split_type, dataset, failed, suffix: str) -> None:
    lines = [
        "# Resultados observados · CARDIOAI",
        "",
        f"- Fuente: **{dataset.name if dataset else 'desconocida'}**",
        f"- Registros: {dataset.rows_count if dataset else '—'} · Variables: {dataset.columns_count if dataset else '—'}",
        f"- Partición: **{split_type or '—'}**",
        f"- Estado: resultados generados por `scripts/export_results.py` a partir de los runs entrenados.",
        "",
        "## Tabla 2 · Discriminación en el holdout",
        "",
        "| Modelo | AUC (IC 95 %) | PR-AUC | Brier | Sens | Esp | F1 | MCC |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in df.iterrows():
        auc = _fmt(row["auc_test"])
        ci = (
            f"{auc} ({_fmt(row['auc_ci_low'], 3)}–{_fmt(row['auc_ci_high'], 3)})"
            if pd.notna(row["auc_ci_low"]) else auc
        )
        lines.append(
            f"| {row['model']} | {ci} | {_fmt(row['pr_auc'])} | {_fmt(row['brier'])} | "
            f"{_fmt(row['sensitivity'])} | {_fmt(row['specificity'])} | {_fmt(row['f1'])} | {_fmt(row['mcc'])} |"
        )
    lines += [
        "",
        "## Tabla 3 · Calibración",
        "",
        "| Modelo | Pendiente | Intercepto | ECE | Hosmer-Lemeshow p |",
        "|---|---:|---:|---:|---:|",
    ]
    for _, row in df.iterrows():
        lines.append(
            f"| {row['model']} | {_fmt(row['calibration_slope'])} | {_fmt(row['calibration_intercept'])} | "
            f"{_fmt(row['calibration_ece'])} | {_fmt(row['hosmer_lemeshow_p'], 4)} |"
        )
    lines += ["", "## Tabla 5 · Comparaciones pareadas (DeLong)", ""]
    if delong:
        lines += ["| Modelo A | Modelo B | ΔAUC | z | p |", "|---|---|---:|---:|---:|"]
        for row in delong:
            lines.append(
                f"| {row['model_a']} | {row['model_b']} | {_fmt(row['difference'])} | "
                f"{_fmt(row['z'])} | {_fmt(row['p_value'], 4)} |"
            )
    else:
        lines.append("No se pudieron calcular comparaciones pareadas.")
    lines += [
        "",
        "## Selección del modelo activo",
        "",
        "La selección usa validación interna/CV, nunca el holdout de prueba:",
        "",
        "| Modelo | AUC selección | Estrategia |",
        "|---|---:|---|",
    ]
    for _, row in df.iterrows():
        lines.append(
            f"| {row['model']} | {_fmt(row['selection_auc'])} | {row['selection_strategy']} |"
        )
    if failed:
        lines += ["", "## Runs no disponibles", ""]
        for run in failed:
            lines.append(f"- Run {run.id} ({run.algorithm}): **{run.status}** — {(run.error or '').splitlines()[0][:200]}")
    lines += [
        "",
        "## Limitaciones declaradas",
        "",
        "- Framingham no contiene PM2.5, espacio verde, ingreso, actividad física, alcohol, "
        "estrés, antecedente familiar ni HDL; esas variables no se imputaron ni se presentaron como medidas.",
        "- Una sola región geográfica: no se realizó validación territorial.",
        "- Seguimiento administrativo fijo de 10 años: el C-index no es informativo y no se reporta.",
        "- Los umbrales de riesgo son operativos y requieren validación clínica externa.",
        "",
    ]
    (RESULTS_DIR / f"RESULTS{suffix}.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()

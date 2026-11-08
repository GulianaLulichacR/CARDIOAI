"""Ejecuta el experimento completo sobre el dataset público, sin servidor web.

    .venv\\Scripts\\python.exe scripts\\run_experiment.py --trials 20 --fresh

Pasos: registra/valida Framingham, entrena los cinco modelos, activa el mejor
por validación interna y exporta tablas y figuras a data/results/.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

DEFAULT_DB = ROOT / "data" / "experiment.db"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--database", default=str(DEFAULT_DB))
    parser.add_argument("--trials", type=int, default=20)
    parser.add_argument("--dataset-id", type=int, default=None)
    parser.add_argument("--fresh", action="store_true", help="Borra runs previos de este dataset")
    parser.add_argument("--skip-export", action="store_true")
    args = parser.parse_args()

    db_path = Path(args.database).resolve()
    db_path.parent.mkdir(parents=True, exist_ok=True)
    os.environ["DATABASE_URL"] = f"sqlite:///{db_path.as_posix()}"
    os.environ.setdefault("SEED_DEMO_DATA", "false")

    import pandas as pd
    from sqlalchemy import func

    from backend.app.database import SessionLocal, init_db
    from backend.app.demo_data import FEATURE_DESCRIPTIONS, FEATURE_LABELS, file_checksum
    from backend.app.ml_engine import ALGORITHMS, _training_worker
    from backend.app.models import Dataset, DatasetVariable, ModelRun
    from backend.app.public_datasets import (
        FRAMINGHAM,
        adapt_framingham,
        dataset_quality,
        ensure_public_file,
        load_framingham,
        resolve_features,
    )

    init_db()
    db = SessionLocal()
    try:
        dataset = _register(db, FRAMINGHAM, adapt_framingham, dataset_quality,
                            ensure_public_file, load_framingham, resolve_features,
                            FEATURE_LABELS, FEATURE_DESCRIPTIONS, Dataset, DatasetVariable,
                            file_checksum)
        if args.fresh:
            removed = db.query(ModelRun).filter(ModelRun.dataset_id == dataset.id).delete()
            db.commit()
            print(f"runs eliminados: {removed}")

        df = pd.read_csv(dataset.file_path)
        features = resolve_features(df)
        print(f"dataset={dataset.name} filas={len(df)} features={features}")

        from backend.app.config import settings
        from backend.app.models import ModelRun as MR

        horizon = settings.prediction_horizon_years
        runs = []
        for algo in ALGORITHMS:
            count = db.query(func.count(MR.id)).filter(MR.algorithm == algo["key"]).scalar() or 0
            run = MR(
                name=f"{algo['name']} · v{count + 1}",
                algorithm=algo["key"],
                version=str(count + 1),
                status="queued",
                is_demo=False,
                dataset_id=dataset.id,
                horizon_years=horizon,
                risk_thresholds={"low_max": settings.risk_low_max, "high_min": settings.risk_high_min},
                trained_by=None,
                notes=algo["description"],
            )
            db.add(run)
            runs.append(run)
        db.commit()
        run_ids = [r.id for r in runs]
        print(f"runs creados: {run_ids}")
    finally:
        db.close()

    started = time.time()
    _training_worker(run_ids, df, max(1, args.trials), horizon, features)
    print(f"entrenamiento terminado en {round(time.time() - started, 1)} s")

    db = SessionLocal()
    try:
        for run_id in run_ids:
            run = db.get(ModelRun, run_id)
            print(f"run {run.id} {run.algorithm}: {run.status} err={(run.error or '')[:120]}")
    finally:
        db.close()

    if not args.skip_export:
        cmd = [sys.executable, str(ROOT / "scripts" / "export_results.py"), "--run-ids", ",".join(map(str, run_ids))]
        subprocess.run(cmd, check=True, env=os.environ)


def _register(db, spec, adapt_framingham, dataset_quality, ensure_public_file,
              load_framingham, resolve_features, FEATURE_LABELS, FEATURE_DESCRIPTIONS,
              Dataset, DatasetVariable, file_checksum):
    import pandas as pd

    name = "Framingham Heart Study · CHD a 10 años"
    dataset = db.query(Dataset).filter(Dataset.name == name).one_or_none()
    original = ensure_public_file(spec)
    _, raw = load_framingham()
    adapted = adapt_framingham(raw)
    target = ROOT / "data" / "datasets" / f"{spec.key}_cardioai.csv"
    adapted.to_csv(target, index=False)

    quality = dataset_quality(adapted)
    quality["provenance"] = {
        **spec.metadata,
        "original_file": original.name,
        "adapted_file": target.name,
        "adapted_sha256": file_checksum(target),
        "features_resolved": resolve_features(adapted),
        "design_note": (
            "Una sola región (Framingham, Massachusetts): no es posible una retención geográfica. "
            "El desenlace es administrativo a 10 años, sin tiempos individuales de censura, "
            "por lo que el C-index no es informativo."
        ),
    }
    fields = dict(
        source=spec.source,
        license=spec.license,
        description=spec.description,
        version="1.0",
        is_demo=False,
        status="ready",
        rows_count=int(len(adapted)),
        columns_count=int(adapted.shape[1]),
        file_path=str(target),
        checksum=file_checksum(original),
        quality=quality,
    )
    if dataset:
        for key, value in fields.items():
            setattr(dataset, key, value)
    else:
        dataset = Dataset(name=name, **fields)
        db.add(dataset)
    db.flush()
    db.query(DatasetVariable).filter(DatasetVariable.dataset_id == dataset.id).delete()
    for col in adapted.columns:
        series = adapted[col]
        numeric = pd.to_numeric(series, errors="coerce")
        is_numeric = series.notna().any() and numeric.notna().sum() == series.notna().sum()
        db.add(DatasetVariable(
            dataset_id=dataset.id,
            name=col,
            label=FEATURE_LABELS.get(col, col.replace("_", " ").capitalize()),
            dtype=str(series.dtype),
            description=FEATURE_DESCRIPTIONS.get(col),
            missing_count=int(series.isna().sum()),
            missing_pct=round(float(series.isna().mean() * 100), 2),
            min_value=float(numeric.min()) if is_numeric else None,
            max_value=float(numeric.max()) if is_numeric else None,
            mean_value=float(numeric.mean()) if is_numeric else None,
            unique_count=int(series.nunique(dropna=True)),
        ))
    db.commit()
    return dataset


if __name__ == "__main__":
    main()

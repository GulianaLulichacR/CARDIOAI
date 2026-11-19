# -*- coding: utf-8 -*-
"""Genera la Figura 2: flujo de datos y particiones del protocolo.

Salida: data/results/figures/fig_flujo_protocolo.png (1800 x 1282 px, 300 DPI).
Los conteos se leen de data/results/model_comparison.json y del dataset adaptado;
no se escriben cifras a mano.
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "data" / "results"
FIGURES = RESULTS / "figures"
DATASET = ROOT / "data" / "datasets" / "framingham_cardioai.csv"
COMPARISON = RESULTS / "model_comparison.json"
OUTPUT = FIGURES / "fig_flujo_protocolo.png"

# --- Datos reales -----------------------------------------------------------
with COMPARISON.open(encoding="utf-8") as fh:
    comparison = json.load(fh)
models = comparison["models"]
train_n = int(models[0]["train_rows"])
test_n = int(models[0]["n_test"])
test_events = int(models[0]["events_test"])

with DATASET.open(encoding="utf-8-sig", newline="") as fh:
    rows = list(csv.DictReader(fh))
total_n = len(rows)
total_events = sum(1 for row in rows if str(row["event"]).strip().lower() == "true")
train_events = total_events - test_events

# Partición interna estratificada 80/20 del entrenamiento (sklearn usa ceil).
val_n = math.ceil(0.2 * train_n)
val_events = math.ceil(0.2 * train_events)
fit_n = train_n - val_n
fit_events = train_events - val_events

expected = (4240, 644, 3392, 515, 848, 129, 2713, 412, 679, 103)
observed = (total_n, total_events, train_n, train_events, test_n, test_events,
            fit_n, fit_events, val_n, val_events)
if observed != expected:
    raise SystemExit(
        "Los conteos cambiaron respecto de la corrida documentada.\n"
        f"esperado={expected}\nobservado={observed}"
    )

# --- Lienzo ----------------------------------------------------------------
SCALE = 2
W, H = 1800, 1332
img = Image.new("RGB", (W * SCALE, H * SCALE), "white")
draw = ImageDraw.Draw(img)

TEAL = (15, 118, 110)
TEAL_SOFT = (224, 237, 236)
TEAL_PALE = (235, 243, 243)
INK = (32, 32, 32)
MUTED = (92, 92, 92)
GRAY_LINE = (115, 115, 115)
GRAY_FILL = (247, 247, 247)

FONT_REG = r"C:\Windows\Fonts\segoeui.ttf"
FONT_BOLD = r"C:\Windows\Fonts\segoeuib.ttf"
_font_cache: dict[tuple[str, int], ImageFont.FreeTypeFont] = {}


def font(path: str, size: int) -> ImageFont.FreeTypeFont:
    key = (path, size)
    if key not in _font_cache:
        _font_cache[key] = ImageFont.truetype(path, size * SCALE)
    return _font_cache[key]


def s(value: float) -> int:
    return int(round(value * SCALE))


def fmt_int(value: int) -> str:
    return f"{value:,}".replace(",", ".")


def fmt_dec(value: float, digits: int = 1) -> str:
    return f"{value:.{digits}f}".replace(".", ",")


def box(x1, y1, x2, y2, fill, outline, width=3, radius=14):
    draw.rounded_rectangle(
        (s(x1), s(y1), s(x2), s(y2)),
        radius=s(radius),
        fill=fill,
        outline=outline,
        width=s(width),
    )


def fit_font(text: str, size: int, bold: bool, max_w: float):
    path = FONT_BOLD if bold else FONT_REG
    candidate = size
    while candidate > 22:
        f = font(path, candidate)
        if draw.textlength(text, font=f) <= s(max_w):
            break
        candidate -= 1
    return font(path, candidate), candidate


def block(cx: float, cy: float, lines, max_w: float, gap: int = 3):
    """Dibuja líneas centradas vertical y horizontalmente.

    lines: [(texto, tamaño, negrita, color)]
    """
    rendered = []
    for text, size, bold, color in lines:
        f, used = fit_font(text, size, bold, max_w)
        bbox = f.getbbox("Ág")
        rendered.append((text, f, color, bbox[3] - bbox[1], used))

    heights = [h + s(gap) for _, _, _, h, _ in rendered]
    total = sum(heights) - s(gap)
    y = s(cy) - total / 2
    for (text, f, color, h, _), step in zip(rendered, heights):
        draw.text((s(cx), y), text, font=f, fill=color, anchor="mt")
        y += step


def arrow_v(x: float, y1: float, y2: float, color=TEAL, dashed=False):
    if dashed:
        yy = y1
        while yy < y2 - 14:
            draw.line([(s(x), s(yy)), (s(x), s(min(yy + 11, y2 - 14)))],
                      fill=color, width=s(3))
            yy += 20
    else:
        draw.line([(s(x), s(y1)), (s(x), s(y2 - 8))], fill=color, width=s(4))
    head = 10
    draw.polygon(
        [(s(x - head), s(y2 - head)), (s(x + head), s(y2 - head)), (s(x), s(y2))],
        fill=color,
    )


def arrow_h(x1: float, x2: float, y: float, color=TEAL):
    draw.line([(s(x1), s(y)), (s(x2 - 8), s(y))], fill=color, width=s(4))
    head = 10
    draw.polygon(
        [(s(x2 - head), s(y - head)), (s(x2 - head), s(y + head)), (s(x2), s(y))],
        fill=color,
    )


# --- Cajas -----------------------------------------------------------------
box(80, 24, 1720, 156, TEAL, TEAL)
box(80, 188, 1720, 288, TEAL_PALE, TEAL)
box(80, 334, 1100, 474, "white", TEAL)
box(1150, 334, 1720, 474, GRAY_FILL, GRAY_LINE)
box(80, 520, 1100, 674, TEAL_PALE, TEAL)
box(80, 720, 1100, 906, "white", TEAL)
box(80, 942, 1100, 1092, TEAL_PALE, TEAL)
box(80, 1128, 1100, 1308, "white", TEAL)
box(1150, 1128, 1720, 1308, TEAL, TEAL)

# --- Texto -----------------------------------------------------------------
block(900, 90, [
    ("Dataset Framingham adaptado", 38, True, "white"),
    (f"{fmt_int(total_n)} registros · 29 variables", 30, False, "white"),
    (f"{total_events} eventos (prevalencia {fmt_dec(100 * total_events / total_n, 2)} %)",
     30, False, "white"),
], max_w=1500)

block(900, 238, [
    ("Partición holdout estratificada 80/20", 36, True, INK),
    ("Semilla 42 · sin retención geográfica", 30, False, MUTED),
], max_w=1500)

block(590, 404, [
    ("Entrenamiento", 36, True, INK),
    (f"{fmt_int(train_n)} registros (80,0 %) · {train_events} eventos",
     31, False, INK),
], max_w=950)

block(1435, 404, [
    ("Prueba (holdout)", 36, True, INK),
    (f"{test_n} registros (20,0 %)", 30, False, INK),
    (f"{test_events} eventos", 30, False, INK),
], max_w=520)

block(590, 597, [
    ("Partición interna estratificada 80/20", 36, True, INK),
    (f"{fmt_int(fit_n)} para ajuste (80,0 %) · {fit_events} eventos",
     30, False, INK),
    (f"{val_n} para validación interna (20,0 %) · {val_events} eventos",
     30, False, INK),
], max_w=960)

block(590, 813, [
    ("Selección y optimización de hiperparámetros", 36, True, INK),
    ("Optuna TPE · 8 trials · semilla 42", 30, False, INK),
    ("Random Forest y XGBoost: CV estratificada de 3 folds", 30, False, INK),
    ("Cox PH, XGBoost-Cox y LSTM-Attention-Cox: validación interna 20 %",
     30, False, INK),
], max_w=960)

block(590, 1017, [
    ("Umbral que maximiza F1", 36, True, INK),
    ("Fijado en la validación interna antes del holdout", 30, False, INK),
    ("0,162 – 0,229 según el modelo", 30, False, INK),
], max_w=960)

block(590, 1218, [
    ("Reentrenamiento final", 36, True, INK),
    (f"{fmt_int(train_n)} registros de entrenamiento", 30, False, INK),
    ("Umbral previamente fijado", 30, False, INK),
], max_w=950)

block(1435, 1218, [
    ("Evaluación única en el holdout", 35, True, "white"),
    ("Discriminación: AUC-ROC · AUC-PR", 29, False, "white"),
    ("Rendimiento: F1 · MCC · Brier", 29, False, "white"),
    ("IC 95 % bootstrap (200 réplicas)", 29, False, "white"),
], max_w=520)

# Etiqueta del carril reservado.
block(1592, 790, [
    ("Holdout reservado", 27, True, MUTED),
    ("sin acceso durante", 26, False, MUTED),
    ("la selección", 26, False, MUTED),
], max_w=235, gap=1)

# --- Conectores ------------------------------------------------------------
arrow_v(900, 156, 188)
draw.line([(s(900), s(288)), (s(900), s(310))], fill=TEAL, width=s(4))
draw.line([(s(590), s(310)), (s(1435), s(310))], fill=TEAL, width=s(4))
arrow_v(590, 310, 334)
arrow_v(1435, 310, 334)
arrow_v(590, 474, 520)
arrow_v(590, 674, 720)
arrow_v(590, 906, 942)
arrow_v(590, 1092, 1128)
arrow_h(1100, 1150, 1218)

# El holdout permanece intacto hasta la evaluación final.
arrow_v(1435, 474, 1128, color=GRAY_LINE, dashed=True)

# --- Guardado --------------------------------------------------------------
FIGURES.mkdir(parents=True, exist_ok=True)
img = img.resize((W, H), Image.LANCZOS)
img.save(OUTPUT, format="PNG", dpi=(300, 300), optimize=True)
print(f"Guardada: {OUTPUT}")
print(f"Dimensiones: {W} x {H} px (300 DPI) = {W / 300:.2f} x {H / 300:.2f} in")
print(f"Conteos: total={total_n}, eventos={total_events}, "
      f"train={train_n}/{train_events}, test={test_n}/{test_events}, "
      f"ajuste={fit_n}/{fit_events}, validación={val_n}/{val_events}")

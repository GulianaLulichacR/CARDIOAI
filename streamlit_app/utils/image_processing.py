"""CARDIOAI - Procesamiento y Curación de Imágenes Médicas
Módulo de utilidades para carga, normalización, realce y generación de datos biomédicos.
"""
from __future__ import annotations

import io
import os
import zipfile
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parent.parent / ".mpl_cache"))

import numpy as np
import pandas as pd
from PIL import Image, ImageEnhance, ImageOps
import matplotlib.cm as cm


def generate_realistic_medical_demo(n: int = 75) -> tuple[pd.DataFrame, dict[str, Image.Image]]:
    """Genera un dataset de demostración médica con simulación de siluetas cardiacas.
    
    Crea imágenes sintéticas representativas de tórax con variaciones de tamaño cardiaco
    (Normal, Cardiomegalia Leve, Cardiomegalia Severa) y pacientes con estudios repetidos
    para probar activamente la detección de data leakage.
    """
    rng = np.random.default_rng(42)
    
    # 65 % de pacientes únicos y 35 % de estudios repetidos para forzar patient leakage
    num_pacientes = max(1, int(n * 0.65))
    patient_pool = [f"PAT-{i:04d}" for i in range(1, num_pacientes + 1)]
    patient_ids = [patient_pool[i] for i in range(num_pacientes)]
    
    num_repetidos = n - num_pacientes
    if num_repetidos > 0:
        pool_rep = patient_pool[:max(1, len(patient_pool) // 2)]
        patient_ids.extend(list(rng.choice(pool_rep, size=num_repetidos)))
    rng.shuffle(patient_ids)
    
    clases = ["Normal", "Cardiomegalia Leve", "Cardiomegalia Severa"]
    probs = [0.48, 0.32, 0.20]
    etiquetas = rng.choice(clases, size=n, p=probs)
    
    edades = rng.normal(loc=59.5, scale=11.5, size=n).clip(28, 88).astype(int)
    sexos = rng.choice(["M", "F"], size=n, p=[0.54, 0.46])
    presion_sist = rng.normal(loc=134.0, scale=19.0, size=n).clip(95, 210).astype(int)
    colesterol = rng.normal(loc=215.0, scale=38.0, size=n).clip(130, 360).round(1)
    imc = rng.normal(loc=27.2, scale=4.8, size=n).clip(18.5, 42.0).round(1)
    diabetes = rng.choice([0, 1], size=n, p=[0.78, 0.22])
    fumador = rng.choice(["Nunca", "Exfumador", "Activo"], size=n, p=[0.50, 0.30, 0.20])
    
    # Cálculo de severidad correlacionada con los factores clínicos
    severidad_base = (
        (edades - 40) * 0.03 +
        (presion_sist - 110) * 0.02 +
        np.where(etiquetas == "Cardiomegalia Severa", 2.2, np.where(etiquetas == "Cardiomegalia Leve", 1.1, 0.2)) +
        rng.normal(0, 0.35, n)
    ).clip(0.1, 5.0).round(2)
    
    image_ids = [f"CXR_{i:04d}.png" for i in range(1, n + 1)]
    
    df = pd.DataFrame({
        "image_id": image_ids,
        "patient_id": patient_ids,
        "clase": etiquetas,
        "edad": edades,
        "sexo": sexos,
        "presion_sistolica": presion_sist,
        "colesterol_total": colesterol,
        "imc": imc,
        "diabetes": diabetes,
        "fumador": fumador,
        "severidad_score": severidad_base,
    })
    
    # Generar imágenes sintéticas simulando tórax con silueta cardiaca
    imagenes: dict[str, Image.Image] = {}
    h, w = 256, 256
    y, x = np.ogrid[:h, :w]
    
    for img_id, cls in zip(image_ids, etiquetas):
        # Fondo base: caja torácica con gradiente
        base = np.zeros((h, w), dtype=np.float32)
        
        # Pulmón izquierdo y derecho (zonas radiolúcidas más oscuras)
        lung_left = ((x - 75) ** 2 / (38 ** 2) + (y - 120) ** 2 / (75 ** 2)) < 1.0
        lung_right = ((x - 180) ** 2 / (38 ** 2) + (y - 120) ** 2 / (75 ** 2)) < 1.0
        
        base[lung_left] = 40.0
        base[lung_right] = 40.0
        
        # Mediastino y silueta cardiaca (zona radiopaca más brillante)
        if cls == "Normal":
            r_card = 42.0
            cx, cy = 135.0, 145.0
        elif cls == "Cardiomegalia Leve":
            r_card = 58.0
            cx, cy = 142.0, 150.0
        else:  # Cardiomegalia Severa
            r_card = 74.0
            cx, cy = 148.0, 154.0
            
        heart_mask = ((x - cx) ** 2 / (r_card ** 2) + (y - cy) ** 2 / ((r_card * 0.85) ** 2)) < 1.0
        base[heart_mask] = 190.0
        
        # Mediastino superior
        mediastinum = (np.abs(x - 128) < 18) & (y < 140)
        base[mediastinum] = np.maximum(base[mediastinum], 170.0)
        
        # Añadir costillas sutiles (arcos horizontales)
        ribs = (np.sin(y / 9.0) > 0.6) & (lung_left | lung_right)
        base[ribs] += 35.0
        
        # Ruido gaussiano para realismo de sensor
        noise = rng.normal(0, 12.0, (h, w)).astype(np.float32)
        img_arr = np.clip(base + noise + 25.0, 0, 255).astype(np.uint8)
        
        imagenes[img_id] = Image.fromarray(img_arr).convert("RGB")
        
    return df, imagenes


def load_dataset_from_zip(zip_bytes: bytes, csv_bytes: bytes | None = None) -> tuple[pd.DataFrame | None, dict[str, Image.Image], list[str]]:
    """Descomprime y empareja imágenes con su archivo de metadatos CSV.
    
    Retorna:
        df: DataFrame de metadatos (si existe)
        images: Diccionario {nombre_archivo: PIL.Image}
        warnings: Lista de mensajes de auditoría/advertencia
    """
    imagenes: dict[str, Image.Image] = {}
    warnings: list[str] = []
    df: pd.DataFrame | None = None
    
    # 1. Extraer imágenes
    try:
        with zipfile.ZipFile(io.BytesIO(zip_bytes), "r") as z:
            for item in z.infolist():
                if item.is_dir():
                    continue
                filename = os.path.basename(item.filename)
                if not filename or filename.startswith("."):
                    continue
                    
                ext = filename.lower().split(".")[-1]
                if ext in ["png", "jpg", "jpeg", "bmp", "tiff", "tif"]:
                    try:
                        raw = z.read(item.filename)
                        img = Image.open(io.BytesIO(raw)).convert("RGB")
                        imagenes[filename] = img
                    except Exception as err:
                        warnings.append(f"No se pudo decodificar la imagen {filename}: {err}")
                elif ext == "csv" and csv_bytes is None:
                    # Encontrar CSV dentro del mismo zip si no fue subido por separado
                    try:
                        raw_csv = z.read(item.filename)
                        df = pd.read_csv(io.BytesIO(raw_csv))
                    except Exception as err:
                        warnings.append(f"Error al leer CSV interno {filename}: {err}")
    except Exception as err:
        warnings.append(f"Error crítico al descomprimir ZIP: {err}")
        return None, {}, warnings
        
    # 2. Cargar CSV externo si fue proporcionado
    if csv_bytes is not None and df is None:
        try:
            df = pd.read_csv(io.BytesIO(csv_bytes))
        except Exception as err:
            warnings.append(f"Error al procesar el archivo CSV de metadatos: {err}")
            
    # 3. Validar emparejamiento
    if df is not None:
        # Detectar columna de nombre de imagen
        img_col = None
        for col in ["image_id", "filename", "file_name", "image", "img", "id"]:
            if col in df.columns:
                img_col = col
                break
                
        if img_col and img_col != "image_id":
            df.rename(columns={img_col: "image_id"}, inplace=True)
            
        if "image_id" in df.columns:
            csv_imgs = set(df["image_id"].astype(str))
            zip_imgs = set(imagenes.keys())
            
            missing_in_zip = csv_imgs - zip_imgs
            orphan_in_zip = zip_imgs - csv_imgs
            
            if missing_in_zip:
                warnings.append(f"{len(missing_in_zip)} imágenes declaradas en el CSV no se encontraron en el ZIP.")
            if orphan_in_zip:
                warnings.append(f"{len(orphan_in_zip)} imágenes del ZIP no tienen registro en el CSV.")
        else:
            warnings.append("El CSV no contiene una columna 'image_id' identificable para vincular con las imágenes.")
            
    return df, imagenes, warnings


def apply_image_enhancements(
    img: Image.Image,
    brightness: float = 1.0,
    contrast: float = 1.0,
    sharpness: float = 1.0,
    invert: bool = False,
    colormap: str = "Original",
    colormap_name: str | None = None,
) -> Image.Image:
    """Aplica filtros radiológicos y de inspección visual médica a una imagen PIL."""
    if colormap_name is not None:
        colormap = colormap_name
    processed = img.copy()
    
    # 1. Inversión (negativo radiológico)
    if invert:
        processed = ImageOps.invert(processed.convert("L")).convert("RGB")
        
    # 2. Brillo
    if brightness != 1.0:
        enhancer = ImageEnhance.Brightness(processed)
        processed = enhancer.enhance(brightness)
        
    # 3. Contraste
    if contrast != 1.0:
        enhancer = ImageEnhance.Contrast(processed)
        processed = enhancer.enhance(contrast)
        
    # 4. Nitidez
    if sharpness != 1.0:
        enhancer = ImageEnhance.Sharpness(processed)
        processed = enhancer.enhance(sharpness)
        
    # 5. Colormap médico
    if colormap != "Original":
        gray = np.array(processed.convert("L")) / 255.0
        cmap_name = colormap.lower()
        import matplotlib as mpl
        try:
            cmap = mpl.colormaps[cmap_name]
            colored = (cmap(gray)[:, :, :3] * 255).astype(np.uint8)
            processed = Image.fromarray(colored)
        except Exception:
            pass  # Mantener original si el colormap no existe
            
    return processed

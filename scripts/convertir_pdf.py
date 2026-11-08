"""Convierte el manuscrito .docx a PDF usando automatización de Microsoft Word."""
from __future__ import annotations

import os
from pathlib import Path

import win32com.client

ROOT = Path(__file__).resolve().parents[1]
DOCX = ROOT / "Manuscrito_CARDIOAI_Actualizado_Framingham.docx"
PDF = ROOT / "Manuscrito_CARDIOAI_Actualizado_Framingham.pdf"


def main() -> None:
    if not DOCX.exists():
        raise FileNotFoundError(f"No se encontró el archivo: {DOCX}")

    word = win32com.client.Dispatch("Word.Application")
    word.Visible = False
    try:
        doc = word.Documents.Open(str(DOCX.resolve()))
        # wdFormatPDF = 17
        doc.SaveAs(str(PDF.resolve()), FileFormat=17)
        doc.Close(False)
        print(f"PDF generado: {PDF}")
    finally:
        word.Quit()


if __name__ == "__main__":
    main()

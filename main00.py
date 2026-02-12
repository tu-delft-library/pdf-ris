from pdf2image import convert_from_path
from pathlib import Path
import pytesseract
import cv2
import pandas as pd
import numpy as np
import re
from openai import OpenAI
from dotenv import load_dotenv
import os

# ----------------------------
# LOAD OPENAI API KEY
# ----------------------------
load_dotenv()  # loads .env file

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OpenAI API key not found. Check your .env file.")

client = OpenAI(api_key=api_key)

# ----------------------------
# INPUT PDF
# ----------------------------
pdf_path = Path(r"E:\XRZONE_Files\PDFExtractor\pdf-ris\samples\257.pdf")
base_name = pdf_path.stem
output_dir = pdf_path.parent
ocr_total_path = output_dir / f"{base_name}_total.txt"
ris_path = output_dir / f"{base_name}.ris"

# ----------------------------
# OCR SETTINGS
# ----------------------------
OCR_DPI = 450
TESSERACT_CONFIG = "--oem 3 --psm 12"
POPPLER_PATH = r"E:\XRZONE_Files\PDFExtractor\pdf-ris\poppler-25.11.0\Library\bin"

# ----------------------------
# PAGE RANGE
# ----------------------------
first_page = 1
last_page = 2

# ----------------------------
# OCR OR LOAD EXISTING TEXT
# ----------------------------
if ocr_total_path.exists():
    print(f"📄 OCR text exists. Loading: {ocr_total_path}")
    ocr_text = ocr_total_path.read_text(encoding="utf-8")
else:
    print(f"🔍 Converting PDF pages {first_page} to {last_page} to images...")
    if POPPLER_PATH:
        pages = convert_from_path(
            str(pdf_path),
            OCR_DPI,
            poppler_path=POPPLER_PATH,
            first_page=first_page,
            last_page=last_page
        )
    else:
        pages = convert_from_path(
            str(pdf_path),
            OCR_DPI,
            first_page=first_page,
            last_page=last_page
        )

    master_lines = []
    print("🔠 Running Tesseract OCR...")
    for i, page in enumerate(pages, start=first_page):
        gray = cv2.cvtColor(np.array(page), cv2.COLOR_RGB2GRAY)
        _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

        data = pytesseract.image_to_data(
            thresh,
            config=TESSERACT_CONFIG,
            output_type="dict"
        )

        df = pd.DataFrame(data)
        df = df[df["conf"].astype(float) > 0]

        page_dict = {}
        for _, row in df.iterrows():
            key = f"{i}_{row['par_num']}_{row['line_num']}"
            text = str(row["text"]).strip()
            if not text:
                continue
            page_dict.setdefault(key, "")
            page_dict[key] += (" " if page_dict[key] else "") + text

        for key in sorted(page_dict.keys()):
            master_lines.append(page_dict[key])

    ocr_text = " ".join(master_lines)
    ocr_total_path.write_text(ocr_text, encoding="utf-8")
    print(f"✅ OCR done. Saved to: {ocr_total_path}")

# ----------------------------
# CHECK IF RIS ALREADY EXISTS
# ----------------------------
if ris_path.exists():
    print(f"📄 RIS file exists: {ris_path}")
    print("⏩ No OpenAI processing.")
else:
    # Count words in OCR text
    ocr_word_count = len(ocr_text.split())
    print(f"📝 OCR word count: {ocr_word_count}")

    # Truncate OCR text for OpenAI
    snippet = ocr_text[:8000]  # only first 8k chars

    # Prepare prompt
    prompt = f"""
    Extract metadata from this research paper text and return it in RIS format.
    Include the following tags if possible: T1 (title), AU (authors), PY (year), AB (abstract), 
    KW (keywords), DO (DOI), BT (book/journal title), PB (publisher), 
    T2 (conference title), Y2 (conference dates), ER (end of record).

    OCR text:
    {snippet}
    """

    print("🤖 Sending OCR text to OpenAI to generate RIS...")

    # Send to OpenAI
    response = client.responses.create(
        model="gpt-4o-mini",
        input=prompt
    )

    ris_text = response.output_text

    # Save RIS file
    ris_path.write_text(ris_text, encoding="utf-8")
    print(f"✅ RIS file saved: {ris_path}")


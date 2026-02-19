from pdf2image import convert_from_path
from pathlib import Path
import pytesseract
import cv2
import pandas as pd
import numpy as np
from openai import OpenAI
from dotenv import load_dotenv
import os
import time

# ============================================================
# CONFIGURATION
# ============================================================

INPUT_DIR = Path(r"E:\XRZONE_Files\PDFExtractor\pdf-ris\samples\batches\01")
POPPLER_PATH = r"E:\XRZONE_Files\PDFExtractor\pdf-ris\poppler-25.11.0\Library\bin"

OCR_DPI = 450
TESSERACT_CONFIG = "--oem 3 --psm 12"

FIRST_PAGE = 1
LAST_PAGE = 2

MAX_PDFS = 100
MODEL_NAME = "gpt-4o-mini"
MODEL_PRICE_PER_1K = 0.0008  # USD per 1k tokens

# ============================================================
# LOAD OPENAI KEY
# ============================================================

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OpenAI API key not found. Check your .env file.")

client = OpenAI(api_key=api_key)

# ============================================================
# FIND PDF FILES
# ============================================================

all_pdfs = sorted(INPUT_DIR.glob("*.pdf"))

pending_pdfs = [
    pdf for pdf in all_pdfs
    if not (pdf.parent / f"{pdf.stem}.ris").exists()
]

pdf_files = pending_pdfs[:MAX_PDFS]

print("=" * 60)
print(f"📂 Total PDFs found: {len(all_pdfs)}")
print(f"⏳ Pending (no RIS yet): {len(pending_pdfs)}")
print(f"🚀 Processing up to: {len(pdf_files)} (cap {MAX_PDFS})")
print("=" * 60)

if not pdf_files:
    print("✅ Nothing to process.")
    exit()

# ============================================================
# BATCH PROCESSING
# ============================================================

total_tokens_used = 0
total_cost_estimate = 0.0

for idx, pdf_path in enumerate(pdf_files, start=1):

    print("\n" + "=" * 60)
    print(f"[{idx}/{len(pdf_files)}] 📄 Processing: {pdf_path.name}")

    base_name = pdf_path.stem
    output_dir = pdf_path.parent
    ocr_total_path = output_dir / f"{base_name}_total.txt"
    ris_path = output_dir / f"{base_name}.ris"

    try:
        # ----------------------------------------------------
        # OCR OR LOAD EXISTING TEXT
        # ----------------------------------------------------
        if ocr_total_path.exists():
            print("📄 OCR text exists. Loading.")
            ocr_text = ocr_total_path.read_text(encoding="utf-8")

        else:
            print("🔍 Converting PDF to images...")
            pages = convert_from_path(
                str(pdf_path),
                OCR_DPI,
                poppler_path=POPPLER_PATH,
                first_page=FIRST_PAGE,
                last_page=LAST_PAGE
            )

            master_lines = []
            print("🔠 Running Tesseract OCR...")

            for i, page in enumerate(pages, start=FIRST_PAGE):
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
            print("✅ OCR saved.")

        # ----------------------------------------------------
        # WORD COUNT
        # ----------------------------------------------------
        ocr_word_count = len(ocr_text.split())
        print(f"📝 OCR word count: {ocr_word_count}")

        # ----------------------------------------------------
        # OPENAI CALL
        # ----------------------------------------------------
        snippet = ocr_text[:8000]

        prompt = f"""
        Extract metadata from this research paper text and return it in RIS format.
        Include the following tags if possible: T1 (title), AU (authors), PY (year),
        AB (abstract), KW (keywords), DO (DOI), BT (book/journal title),
        PB (publisher), T2 (conference title), Y2 (conference dates), ER.
        Return ONLY valid RIS.
                
        OCR text:
        {snippet}
        """

        print("🤖 Sending to OpenAI...")

        response = client.responses.create(
            model=MODEL_NAME,
            input=prompt
        )

        ris_text = response.output_text
        ris_path.write_text(ris_text, encoding="utf-8")
        print(f"✅ RIS saved: {ris_path.name}")

    except Exception as e:
        print(f"❌ Error processing {pdf_path.name}: {e}")
        continue

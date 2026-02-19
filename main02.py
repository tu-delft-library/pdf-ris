from pdf2image import convert_from_path
from pathlib import Path
import pytesseract
import cv2
import pandas as pd
import numpy as np
from openai import OpenAI
from dotenv import load_dotenv
import os
import json

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
pdf_path = Path(r"E:\XRZONE_Files\PDFExtractor\pdf-ris\samples\266.pdf")
base_name = pdf_path.stem
output_dir = pdf_path.parent
ocr_total_path = output_dir / f"{base_name}_total.txt"
json_path = output_dir / f"{base_name}.json"

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
# CHECK IF JSON ALREADY EXISTS
# ----------------------------
if json_path.exists():
    print(f"📄 JSON file exists: {json_path}")
    print("⏩ No OpenAI processing.")
else:
    # Count words in OCR text
    ocr_word_count = len(ocr_text.split())
    print(f"📝 OCR word count: {ocr_word_count}")

    # Truncate OCR text for OpenAI
    snippet = ocr_text[:8000]

    # ----------------------------
    # PREPARE PROMPT FOR JSON
    # ----------------------------
    prompt_json = f"""
Extract structured metadata from this research paper text.
Return ONLY valid JSON with the following fields:
title, authors (array), year, abstract, keywords (array), doi,
journal, publisher, conference_title, conference_dates.

OCR text:
{snippet}
    """

    print("🤖 Sending OCR snippet to OpenAI for JSON metadata...")

    # ----------------------------
    # SEND TO OPENAI
    # ----------------------------
    response = client.responses.create(
        model="gpt-4o-mini",
        input=prompt_json
    )

    raw_output = response.output_text.strip()

    # Clean code fences if present
    if raw_output.startswith("```"):
        raw_output = raw_output.strip("`").replace("json", "", 1).strip()

    # ----------------------------
    # PARSE JSON
    # ----------------------------
    try:
        metadata_json = json.loads(raw_output)
    except json.JSONDecodeError as e:
        print(f"❌ Failed to parse JSON: {e}")
        print("Raw output:", raw_output)
        raise

    # Save JSON
    json_path.write_text(json.dumps(metadata_json, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"✅ JSON metadata saved: {json_path}")

    # ----------------------------
# CONVERT JSON TO RIS
# ----------------------------
def json_to_ris(metadata):
    ris_lines = []

    def add(tag, value):
        if value:
            if isinstance(value, list):
                for v in value:
                    ris_lines.append(f"{tag}  - {v}")
            else:
                ris_lines.append(f"{tag}  - {value}")

    add("TY", "JOUR")  # default to journal article
    add("T1", metadata.get("title"))
    add("AU", metadata.get("authors"))
    add("PY", metadata.get("year"))
    add("AB", metadata.get("abstract"))
    add("KW", metadata.get("keywords"))
    add("DO", metadata.get("doi"))
    add("BT", metadata.get("journal"))
    add("PB", metadata.get("publisher"))
    add("T2", metadata.get("conference_title"))
    add("Y2", metadata.get("conference_dates"))
    ris_lines.append("ER  - ")

    return "\n".join(ris_lines)

ris_path = output_dir / f"{base_name}.ris"
ris_text = json_to_ris(metadata_json)
ris_path.write_text(ris_text, encoding="utf-8")
print(f"✅ RIS saved: {ris_path}")


# ----------------------------
# CONVERT JSON TO CERIF/XML
# ----------------------------
def json_to_cerif_xml(metadata):
    import xml.etree.ElementTree as ET

    # Root element
    root = ET.Element("publication")

    def add_element(tag, value):
        if value:
            if isinstance(value, list):
                for v in value:
                    el = ET.SubElement(root, tag)
                    el.text = str(v)
            else:
                el = ET.SubElement(root, tag)
                el.text = str(value)

    add_element("title", metadata.get("title"))
    add_element("authors", metadata.get("authors"))
    add_element("year", metadata.get("year"))
    add_element("abstract", metadata.get("abstract"))
    add_element("keywords", metadata.get("keywords"))
    add_element("doi", metadata.get("doi"))
    add_element("journal", metadata.get("journal"))
    add_element("publisher", metadata.get("publisher"))
    add_element("conference_title", metadata.get("conference_title"))
    add_element("conference_dates", metadata.get("conference_dates"))

    return ET.tostring(root, encoding="unicode")

xml_path = output_dir / f"{base_name}.xml"
xml_text = json_to_cerif_xml(metadata_json)
xml_path.write_text(xml_text, encoding="utf-8")
print(f"✅ CERIF/XML saved: {xml_path}")


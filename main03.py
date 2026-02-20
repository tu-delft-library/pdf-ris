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
import xml.etree.ElementTree as ET
import uuid

# ----------------------------
# LOAD OPENAI API KEY
# ----------------------------
load_dotenv()
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
    ocr_text = ocr_total_path.read_text(encoding="utf-8")
else:
    pages = convert_from_path(str(pdf_path), OCR_DPI, poppler_path=POPPLER_PATH,
                              first_page=first_page, last_page=last_page)
    master_lines = []
    for i, page in enumerate(pages, start=first_page):
        gray = cv2.cvtColor(np.array(page), cv2.COLOR_RGB2GRAY)
        _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        data = pytesseract.image_to_data(thresh, config=TESSERACT_CONFIG, output_type="dict")
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

# ----------------------------
# JSON METADATA
# ----------------------------
if json_path.exists():
    metadata_json = json.loads(json_path.read_text(encoding="utf-8"))
else:
    snippet = ocr_text[:8000]
    prompt_json = f"""
Extract structured metadata from this research paper text.
Return ONLY valid JSON with the following fields:
- title
- subtitle (optional)
- authors (array of full names)
- year
- abstract
- keywords (array)
- doi
- publication_type
- publisher
- conference_name (optional)
- conference_acronym (optional)
- conference_place (optional)
- conference_country (optional)
- conference_dates (optional; start_date, end_date)

OCR text:
{snippet}
    """
    response = client.responses.create(model="gpt-4o-mini", input=prompt_json)
    raw_output = response.output_text.strip()
    if raw_output.startswith("```"):
        raw_output = raw_output.strip("`").replace("json", "", 1).strip()
    metadata_json = json.loads(raw_output)
    json_path.write_text(json.dumps(metadata_json, indent=2, ensure_ascii=False), encoding="utf-8")

# ----------------------------
# JSON → OAI-PMH CERIF XML
# ----------------------------
def json_to_oai_pmh(metadata, pdf_file=None):
    """
    Convert metadata JSON to full OAI-PMH + CERIF/OpenAIRE XML suitable for Pure upload.
    """
    import xml.etree.ElementTree as ET
    import uuid

    # Namespaces
    ns_oai = "http://www.openarchives.org/OAI/2.0/"
    ns_xsi = "http://www.w3.org/2001/XMLSchema-instance"
    ns_cerif = "https://www.openaire.eu/cerif-profile/1.2/"
    ns_pubt = "https://www.openaire.eu/cerif-profile/vocab/COAR_Publication_Types"
    ns_ar = "http://purl.org/coar/access_right"
    ns_pure = "https://pure.elsevier.com/schema/extensions/oai_cerif_openaire"

    # Register namespaces
    ET.register_namespace("", ns_oai)
    ET.register_namespace("cerif", ns_cerif)
    ET.register_namespace("pubt", ns_pubt)
    ET.register_namespace("ar", ns_ar)
    ET.register_namespace("pure", ns_pure)

    # Root OAI-PMH element
    root = ET.Element(
        f"{{{ns_oai}}}OAI-PMH",
        attrib={
            f"{{{ns_xsi}}}schemaLocation": (
                f"{ns_oai} {ns_oai}/OAI-PMH.xsd "
                f"{ns_cerif} https://www.openaire.eu/schema/cris/current/openaire-cerif-profile.xsd"
            )
        }
    )

    # ListRecords
    list_records = ET.SubElement(root, f"{{{ns_oai}}}ListRecords")
    record = ET.SubElement(list_records, f"{{{ns_oai}}}record")
    metadata_el = ET.SubElement(record, "metadata")

    # CERIF Publication
    pub_id = str(uuid.uuid4())
    pub_el = ET.SubElement(metadata_el, f"{{{ns_cerif}}}Publication", id=pub_id)

    # Type (publication_type from JSON)
    pub_type = ET.SubElement(pub_el, f"{{{ns_pubt}}}Type", attrib={
        f"{{{ns_pure}}}peerReviewed": "true",
        f"{{{ns_pure}}}publicationCategory": "/dk/atira/pure/researchoutput/category/scientific"
    })
    pub_type.text = metadata.get("publication_type", "http://purl.org/coar/resource_type/c_5794")

    # Language
    lang = ET.SubElement(pub_el, f"{{{ns_cerif}}}Language")
    lang.text = "en"

    # Title / Subtitle
    title = ET.SubElement(pub_el, f"{{{ns_cerif}}}Title", attrib={"xml:lang": "en"})
    title.text = metadata.get("title", "")
    if metadata.get("subtitle"):
        subtitle = ET.SubElement(pub_el, f"{{{ns_cerif}}}Subtitle", attrib={"xml:lang": "en"})
        subtitle.text = metadata.get("subtitle")

    # Abstract
    abstract = ET.SubElement(pub_el, f"{{{ns_cerif}}}Abstract", attrib={"xml:lang": "en"})
    abstract.text = metadata.get("abstract", "")

    # Keywords
    for kw in metadata.get("keywords", []):
        k = ET.SubElement(pub_el, f"{{{ns_cerif}}}Keyword", attrib={"xml:lang": "en"})
        k.text = kw

    # DOI
    doi_raw = metadata.get("doi", "")
    if doi_raw.startswith("https://doi.org/"):
        doi_raw = doi_raw.replace("https://doi.org/", "")
    doi = ET.SubElement(pub_el, f"{{{ns_cerif}}}DOI")
    doi.text = doi_raw

    # Authors
    authors_el = ET.SubElement(pub_el, f"{{{ns_cerif}}}Authors")
    for author_name in metadata.get("authors", []):
        author_el = ET.SubElement(authors_el, f"{{{ns_cerif}}}Author")
        person_el = ET.SubElement(author_el, f"{{{ns_cerif}}}Person", id=str(uuid.uuid4()))
        name_el = ET.SubElement(person_el, f"{{{ns_cerif}}}PersonName")
        if " " in author_name:
            first, last = author_name.split(" ", 1)
        else:
            first, last = author_name, ""
        ET.SubElement(name_el, f"{{{ns_cerif}}}FirstNames").text = first
        ET.SubElement(name_el, f"{{{ns_cerif}}}FamilyNames").text = last
        ET.SubElement(author_el, f"{{{ns_cerif}}}Affiliation")  # empty

    # Journal / Conference as PartOf
    if metadata.get("journal") or metadata.get("conference_name"):
        part_of = ET.SubElement(pub_el, f"{{{ns_cerif}}}PartOf")
        pub_part = ET.SubElement(part_of, f"{{{ns_cerif}}}Publication")
        ET.SubElement(pub_part, f"{{{ns_pubt}}}Type").text = "http://purl.org/coar/resource_type/c_0640"
        if metadata.get("journal"):
            ET.SubElement(pub_part, f"{{{ns_cerif}}}Title", attrib={"xml:lang": "en"}).text = metadata.get("journal")
        elif metadata.get("conference_name"):
            ET.SubElement(pub_part, f"{{{ns_cerif}}}Title", attrib={"xml:lang": "en"}).text = metadata.get("conference_name")

    # Conference details
    if metadata.get("conference_name"):
        presented_at = ET.SubElement(pub_el, f"{{{ns_cerif}}}PresentedAt")
        event = ET.SubElement(presented_at, f"Event")
        for key in ["conference_acronym", "conference_name", "conference_place", "conference_country"]:
            if metadata.get(key):
                tag = key.replace("conference_", "").capitalize()
                ET.SubElement(event, tag).text = metadata[key]
        # Conference dates
        if metadata.get("conference_dates"):
            ET.SubElement(event, "StartDate").text = metadata["conference_dates"].get("start_date", "")
            ET.SubElement(event, "EndDate").text = metadata["conference_dates"].get("end_date", "")

    # Publication year
    pub_date = ET.SubElement(pub_el, f"{{{ns_cerif}}}PublicationDate")
    pub_date.text = str(metadata.get("year", ""))

    # Status
    status = ET.SubElement(pub_el, f"{{{ns_cerif}}}Status", scheme="/dk/atira/pure/researchoutput/status")
    status.text = "published"

    # File location (optional)
    if pdf_file:
        files_el = ET.SubElement(pub_el, f"{{{ns_cerif}}}FileLocations")
        medium_el = ET.SubElement(files_el, f"{{{ns_cerif}}}Medium")
        ET.SubElement(medium_el, f"{{{ns_cerif}}}Type", attrib={"scheme": "/dk/atira/pure/researchoutput/electronicversion/versiontype"}).text = "publishersversion"
        ET.SubElement(medium_el, f"{{{ns_cerif}}}Title", attrib={"xml:lang": "en"}).text = pdf_file.get("title", "")
        ET.SubElement(medium_el, f"{{{ns_cerif}}}URI").text = pdf_file.get("uri", "")
        ET.SubElement(medium_el, f"{{{ns_cerif}}}MimeType").text = "application/pdf"
        ET.SubElement(medium_el, f"{{{ns_cerif}}}Size").text = str(pdf_file.get("size", 0))
        ET.SubElement(medium_el, f"{{{ns_ar}}}Access").text = "http://purl.org/coar/access_right/c_abf2"

    # Access
    access_el = ET.SubElement(pub_el, f"{{{ns_ar}}}Access")
    access_el.text = "http://purl.org/coar/access_right/c_abf2"

    return ET.tostring(root, encoding="unicode")

# ----------------------------
# SAVE XML
# ----------------------------
xml_text = json_to_oai_pmh(metadata_json)
xml_path = output_dir / f"{base_name}.xml"
xml_path.write_text(xml_text, encoding="utf-8")
print(f"✅ OAI-PMH / CERIF XML saved: {xml_path}")
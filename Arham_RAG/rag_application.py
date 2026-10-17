import os
import re
import json
import base64
import logging
from pathlib import Path

import anthropic

try:
    from sarvamai import SarvamAI
except ImportError:
    SarvamAI = None
import psycopg
from psycopg.rows import dict_row
from pgvector import Vector
from pgvector.psycopg import register_vector
from sentence_transformers import SentenceTransformer
from pypdf import PdfReader
from docx import Document
import fitz
import pytesseract
from PIL import Image
from io import BytesIO

import ollama
import numpy as np

# ============================================================
# DAY 12 - RETRIEVAL QUALITY CONFIGURATION
# ============================================================

DATA_FOLDER = Path("data")

EMBEDDING_MODEL = "intfloat/multilingual-e5-base"
EMBEDDING_DIM = 768

# Retrieval experiments
TOP_K = 5
RETRIEVAL_CANDIDATES = 100
ENABLE_RERANKING = True

# Chunk experiments
CHUNK_SIZE = 1200
CHUNK_OVERLAP = 200

# Metadata/filtering experiments
SOURCE_FILTER = None       # Example: "output_hindi.txt"
FILE_TYPE_FILTER = None    # Example: ".txt"

# Local LLM
LOCAL_LLM_MODEL = "qwen3:4b"
PREFER_CLAUDE = True  # Use Claude for normal RAG answers; requires ANTHROPIC_API_KEY

# PostgreSQL
DB_HOST = "localhost"
DB_PORT = 5432
DB_NAME = "ragdb"
DB_USER = "postgres"
DB_PASSWORD = "root"

# Current Claude model
CLAUDE_MODEL = "claude-opus-5-5"
SARVAM_MODEL = "sarvam-105b"
# Ragas evaluation settings. Ragas is optional and runs only when menu option 10 is selected.
RAGAS_MAX_QUESTIONS = 5  # Start small because local LLM evaluation can be slow.
RAGAS_OUTPUT_FILE = Path("ragas_evaluation_results.csv")

# ============================================================
# DAY 12 - 20 RETRIEVAL TEST QUESTIONS
# ============================================================

RETRIEVAL_TESTS = [
    {"id": 1, "question": "एक धनुष कितने हाथ का होता है?", "keywords": ["एक धनुष चार हाथ का होता है"]},
    {"id": 2, "question": "भगवान ऋषभदेव की काया कितने धनुष की थी?", "keywords": ["500 धनुष", "ऋषभदेव"]},
    {"id": 3, "question": "बाहुबली की काया कितने धनुष की बताई गई है?", "keywords": ["525 धनुष", "बाहुबली"]},
    {"id": 4, "question": "2000 हाथ की काया किसके संदर्भ में बताई गई है?", "keywords": ["2000 हाथ"]},
    {"id": 5, "question": "ऋषभदेव और बाहुबली की काया में कितने धनुष का अंतर था?", "keywords": ["500 धनुष", "525 धनुष"]},
    {"id": 6, "question": "एक धनुष को हाथ में कैसे मापा गया है?", "keywords": ["एक धनुष चार हाथ"]},
    {"id": 7, "question": "बाहुबली की काया के बारे में ग्रंथ में क्या बताया गया है?", "keywords": ["बाहुबली", "525 धनुष"]},
    {"id": 8, "question": "ऋषभदेव की काया और बाहुबली की काया में क्या संबंध बताया गया है?", "keywords": ["ऋषभदेव", "बाहुबली"]},
    {"id": 9, "question": "किसकी काया 525 धनुष की थी?", "keywords": ["बाहुबली", "525 धनुष"]},
    {"id": 10, "question": "500 धनुष की काया किसकी थी?", "keywords": ["ऋषभदेव", "500 धनुष"]},
    {"id": 11, "question": "धनुष और हाथ के बीच क्या संबंध है?", "keywords": ["एक धनुष चार हाथ"]},
    {"id": 12, "question": "ऋषभदेव की काया को हाथ में व्यक्त करें।", "keywords": ["500 धनुष", "चार हाथ"]},
    {"id": 13, "question": "बाहुबली की काया को हाथ में व्यक्त करें।", "keywords": ["525 धनुष", "चार हाथ"]},
    {"id": 14, "question": "525 धनुष कितने हाथ होते हैं?", "keywords": ["525", "चार हाथ"]},
    {"id": 15, "question": "500 धनुष कितने हाथ होते हैं?", "keywords": ["500", "चार हाथ"]},
    {"id": 16, "question": "ऋषभदेव की काया कितने हाथ की थी?", "keywords": ["500 धनुष", "चार हाथ"]},
    {"id": 17, "question": "बाहुबली की काया कितने हाथ की थी?", "keywords": ["525 धनुष", "चार हाथ"]},
    {"id": 18, "question": "ग्रंथ में भगवान की काया 2000 हाथ क्यों बताई गई है?", "keywords": ["2000 हाथ"]},
    {"id": 19, "question": "धनुष की लंबाई के बारे में दिए गए कथन को खोजिए।", "keywords": ["एक धनुष चार हाथ"]},
    {"id": 20, "question": "ऋषभदेव, बाहुबली और धनुष के माप से संबंधित सभी संख्याएँ बताइए।", "keywords": ["500 धनुष", "525 धनुष", "चार हाथ", "2000 हाथ"], "multi_chunk": True},
]

# Never hard-code API keys in source code. Set ANTHROPIC_API_KEY in your
# Windows environment or in a local .env file (if your project loads dotenv).
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "").strip()
client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY) if ANTHROPIC_API_KEY else None
SARVAM_API_KEY = os.getenv("SARVAM_API_KEY", "").strip()
sarvam_client = (
    SarvamAI(api_subscription_key=SARVAM_API_KEY)
    if SarvamAI is not None and SARVAM_API_KEY else None
)

# Detailed diagnostics for PDF heading detection. Set HEADING_LOG_LEVEL=DEBUG
# in the environment to see OCR candidate lines and prompts/results.
HEADING_LOG_LEVEL = os.getenv("HEADING_LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, HEADING_LOG_LEVEL, logging.INFO),
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
heading_logger = logging.getLogger("rag.heading_detection")
# ============================================================
 
# ============================================================
# DATABASE CONNECTION
# ============================================================

def get_connection():
    conn = psycopg.connect(
        host=DB_HOST,
        port=DB_PORT,
        dbname=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD,
        row_factory=dict_row,
    )
    register_vector(conn)
    return conn


# ============================================================
# DATABASE SETUP
# ============================================================

def setup_database():
    print("\nSetting up PostgreSQL database...")

    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")

            cur.execute(
                f"""
                CREATE TABLE IF NOT EXISTS documents (
                    id BIGSERIAL PRIMARY KEY,
                    content TEXT NOT NULL,
                    source TEXT NOT NULL,
                    chunk_number INTEGER NOT NULL,
                    embedding VECTOR({EMBEDDING_DIM}) NOT NULL,
                    file_type TEXT DEFAULT '.txt',
                    page_number INTEGER,
                    section_name TEXT
                );
                """
            )

            # Add metadata column to older databases without destroying data.
            cur.execute(
                "ALTER TABLE documents ADD COLUMN IF NOT EXISTS file_type TEXT DEFAULT '.txt';"
            )
            # Citation metadata: page number for PDFs, section heading for all supported formats.
            cur.execute(
                "ALTER TABLE documents ADD COLUMN IF NOT EXISTS page_number INTEGER;"
            )
            cur.execute(
                "ALTER TABLE documents ADD COLUMN IF NOT EXISTS section_name TEXT;"
            )

            cur.execute(
                """
                CREATE INDEX IF NOT EXISTS documents_embedding_hnsw_idx
                ON documents
                USING hnsw (embedding vector_cosine_ops);
                """
            )

            cur.execute(
                """
                CREATE INDEX IF NOT EXISTS documents_source_idx
                ON documents(source);
                """
            )

        conn.commit()

    print("Database setup completed.")


def sanitize_extracted_text(text):
    """Remove malformed PDF/control-character noise while preserving readable text."""
    if not text:
        return ""

    text = str(text)
    text = text.replace("\x00", " ")
    text = text.replace("\ufeff", "")
    text = text.replace("\u200b", "").replace("\u200c", "").replace("\u200d", "")
    text = text.replace("\u00ad", "")
    text = text.replace("\ufffd", "")

    # Remove control chars that commonly appear in bad PDF font extraction.
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]", " ", text)

    # Remove common bad-character runs created by broken PDF font/cmap decoding.
    text = re.sub(r"(?<!\w)[^\w\s\-.,;:()[]/\\'\"\u0900-\u097F\u2000-\u206F\u00A0-\u00FF]+", " ", text)

    # Collapse artificial spacing and trim lines.
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def extract_pdf_text_with_fitz(file_path):
    """Use PyMuPDF for cleaner PDF text extraction when font encoding is broken."""
    doc = fitz.open(str(file_path))
    pages = []
    for page_num in range(doc.page_count):
        page = doc[page_num]
        text = page.get_text("text", sort=True)
        if text:
            pages.append((page_num + 1, sanitize_extracted_text(text)))
    doc.close()
    return pages


# ============================================================
# DOCUMENT READING
# ============================================================

def read_txt_file(file_path):
    try:
        return file_path.read_text(encoding="utf-8", errors="ignore")
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return ""


def looks_like_heading(line):
    """Conservative heading heuristic for PDF/TXT files without heading styles."""
    line = line.strip()
    if not line or len(line) > 100:
        return False
    if line.endswith((".", "।", ",", ";", ":")):
        return False
    if len(line.split()) > 12:
        return False
    # Common numbered headings: 1, 1.2, Chapter 1, अध्याय १, etc.
    if re.match(r"^(chapter|section|अध्याय|खंड|भाग)\s*[-.:]?\s*[0-9०-९IVXivx]+", line, re.IGNORECASE):
        return True
    if re.match(r"^[0-9०-९]+(?:\.[0-9०-९]+)*[.)]?\s+\S+", line):
        return True
    # Short title-like lines are plausible headings, but may be imperfect.
    return len(line.split()) <= 8 and line == line.strip()


def split_text_into_section_records(text, source):
    """Split plain text into heading-based records; page number is unavailable for TXT."""
    records = []
    current_section = "Unspecified section"
    current_lines = []

    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if looks_like_heading(stripped):
            if current_lines:
                records.append({
                    "source": source,
                    "page_number": None,
                    "section_name": current_section,
                    "text": "\n".join(current_lines),
                })
                current_lines = []
            current_section = stripped[:200]
        else:
            current_lines.append(stripped)

    if current_lines:
        records.append({
            "source": source,
            "page_number": None,
            "section_name": current_section,
            "text": "\n".join(current_lines),
        })
    if not records and text.strip():
        records.append({"source": source, "page_number": None,
                        "section_name": "Unspecified section", "text": text.strip()})
    return records


def _ocr_lines_with_layout(image):
    """Return OCR lines with approximate position and confidence using Tesseract."""
    data = pytesseract.image_to_data(
        image, lang="hin+eng", config="--psm 6",
        output_type=pytesseract.Output.DICT,
    )
    grouped = {}
    count = len(data.get("text", []))
    for i in range(count):
        word = (data["text"][i] or "").strip()
        if not word:
            continue
        try:
            confidence = float(data["conf"][i])
        except (TypeError, ValueError):
            confidence = -1
        key = (data["block_num"][i], data["par_num"][i], data["line_num"][i])
        item = grouped.setdefault(key, {
            "words": [], "left": data["left"][i], "top": data["top"][i],
            "right": data["left"][i] + data["width"][i],
            "bottom": data["top"][i] + data["height"][i], "confidences": []
        })
        item["words"].append(word)
        item["left"] = min(item["left"], data["left"][i])
        item["top"] = min(item["top"], data["top"][i])
        item["right"] = max(item["right"], data["left"][i] + data["width"][i])
        item["bottom"] = max(item["bottom"], data["top"][i] + data["height"][i])
        if confidence >= 0:
            item["confidences"].append(confidence)

    lines = []
    for item in grouped.values():
        text = sanitize_extracted_text(" ".join(item["words"]))
        if text:
            item["text"] = text
            item["confidence"] = (
                sum(item["confidences"]) / len(item["confidences"])
                if item["confidences"] else 0
            )
            lines.append(item)
    return sorted(lines, key=lambda x: (x["top"], x["left"]))


def _is_number_only(line):
    return bool(re.fullmatch(r"[0-9०-९]{1,3}[.)।:]?", line.strip()))


def _looks_like_title_text(line):
    """Reject sentence-like OCR lines; accept short title-shaped Hindi/English lines."""
    line = line.strip()
    if not line or len(line) > 90 or len(line.split()) > 8:
        return False
    if line.endswith(("।", "|", ".", ",", ";", ":")):
        return False
    # A prose sentence often has a verb / sentence punctuation; this is only a heuristic.
    prose_markers = (" करेगा", " करेगी", " करेंगे", " होता है", " होती है", " होता", " रहेगी", " करेगा |")
    if any(marker in line for marker in prose_markers):
        return False
    return bool(re.search(r"[A-Za-z\u0900-\u097F]", line))


def _detect_chapter_heading(ocr_lines, image_width, image_height):
    """Detect a chapter number/title pair near the top, using both text and layout.

    Returns a normalized heading or None. Does not classify arbitrary short body lines
    as headings, which was causing verse numbers and prose to become section names.
    """
    top_lines = [
        line for line in ocr_lines
        if line["top"] < image_height * 0.38
        and line["confidence"] >= 20
    ]
    heading_logger.debug(
        "Rule detector: image=%dx%d; top_lines=%d; lines=%s",
        image_width, image_height, len(top_lines),
        [line.get("text", "") for line in top_lines[:15]],
    )
    for i, line in enumerate(top_lines):
        text = line["text"].strip()
        centered = abs(((line["left"] + line["right"]) / 2) - image_width / 2) < image_width * 0.22
        if not centered:
            continue

        # Pattern A: title line followed by a standalone chapter number (as in the sample).
        if _looks_like_title_text(text) and i + 1 < len(top_lines):
            nxt = top_lines[i + 1]
            gap = nxt["top"] - line["bottom"]
            next_centered = abs(((nxt["left"] + nxt["right"]) / 2) - image_width / 2) < image_width * 0.22
            if _is_number_only(nxt["text"]) and next_centered and 0 <= gap < image_height * 0.12:
                number = re.sub(r"[^0-9०-९]", "", nxt["text"])
                heading = f"{number} — {text}"
                heading_logger.info("Rule detector found title-then-number heading: %r", heading)
                return heading

        # Pattern B: standalone number followed by a centered title.
        if _is_number_only(text) and i + 1 < len(top_lines):
            nxt = top_lines[i + 1]
            gap = nxt["top"] - line["bottom"]
            next_centered = abs(((nxt["left"] + nxt["right"]) / 2) - image_width / 2) < image_width * 0.22
            if _looks_like_title_text(nxt["text"]) and next_centered and 0 <= gap < image_height * 0.12:
                number = re.sub(r"[^0-9०-९]", "", text)
                heading = f"{number} — {nxt['text']}"
                heading_logger.info("Rule detector found number-then-title heading: %r", heading)
                return heading
    heading_logger.debug("Rule detector found no chapter heading")
    return None



def _normalize_chapter_number(value):
    """Normalize Devanagari digits to Arabic digits for consistent metadata."""
    devanagari = "०१२३४५६७८९"
    return "".join(str(devanagari.index(ch)) if ch in devanagari else ch for ch in str(value))


def detect_heading_with_ai(image, ocr_lines, page_number=None):
    """Use Sarvam chat completion to classify OCR/layout clues as a chapter heading.

    Sarvam chat completion is text-based here (not image/vision input), so it uses
    OCR lines plus their approximate position, alignment and confidence. The
    function returns a normalized heading or None and emits diagnostic logs.
    """
    page_label = f"page={page_number}" if page_number is not None else "page=unknown"
    heading_logger.info("[%s] AI heading detection started", page_label)

    if sarvam_client is None:
        reason = "sarvamai SDK missing" if SarvamAI is None else "SARVAM_API_KEY not configured"
        heading_logger.warning("[%s] Sarvam unavailable (%s); skipping AI heading detection", page_label, reason)
        return None

    upper = [line for line in ocr_lines if line.get("top", image.height) < image.height * 0.72]
    has_number = any(_is_number_only(line.get("text", "")) for line in upper)
    has_short_title = any(
        _looks_like_title_text(line.get("text", ""))
        and abs(((line.get("left", 0) + line.get("right", 0)) / 2) - image.width / 2) < image.width * 0.35
        for line in upper
    )
    heading_logger.debug(
        "[%s] OCR precheck: upper_lines=%d has_number=%s has_short_centered_title=%s",
        page_label, len(upper), has_number, has_short_title
    )
   # if not (has_number and has_short_title):
   #     heading_logger.info("[%s] AI skipped: OCR precheck did not find both a number and a plausible title", page_label)
   #     return None

    # Include OCR text and coordinates so Sarvam can reason about visual hierarchy
    # even though this endpoint receives text rather than the page image.
    formatted_lines = []
    for line in upper[:35]:
        text = str(line.get("text", "")).strip()
        if not text:
            continue
        left = int(line.get("left", 0))
        right = int(line.get("right", 0))
        top = int(line.get("top", 0))
        bottom = int(line.get("bottom", top))
        center_ratio = ((left + right) / 2) / max(image.width, 1)
        formatted_lines.append(
            f"y={top}:{bottom}, center_x={center_ratio:.2f}, "
            f"confidence={float(line.get('confidence', 0)):.0f}: {text}"
        )
    ocr_preview = "\n".join(formatted_lines)[:5000]
    prompt = f"""You classify OCR output from a scanned Hindi/Sanskrit book for RAG metadata.
Determine whether this page starts a new chapter/section. Use the OCR text, vertical positions,
center alignment and confidence. A common layout is a short centered Hindi title near the top,
then a large blank gap, then a standalone chapter number, then Sanskrit verses.
Example: title 'शरीर कैसा है?' plus number '18' means section '18 — शरीर कैसा है?'.
OCR can misread a chapter number (for example, 18 as 8); only correct it when the supplied OCR/layout evidence supports the correction. Do not guess a number solely from the example.
Do not classify verse lines, page numbers, or ordinary prose as chapter headings. If uncertain, return false.

OCR lines (y positions are pixels from the top; center_x is a 0-1 page-width ratio):
{ocr_preview}

Return ONLY valid JSON, no markdown:
{{"is_chapter_heading": true, "chapter_number": "18", "chapter_title": "शरीर कैसा है?", "confidence": 0.90}}
If not a heading, return {{"is_chapter_heading": false, "chapter_number": null, "chapter_title": null, "confidence": 0.0}}.
"""

    try:
        heading_logger.debug("[%s] Calling Sarvam model=%s; OCR preview chars=%d", page_label, SARVAM_MODEL, len(ocr_preview))
        response = sarvam_client.chat.completions(
            model=SARVAM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=300,
        )
        raw = response.choices[0].message.content or ""
        heading_logger.debug("[%s] Sarvam raw response: %s", page_label, raw[:1500])
        match = re.search(r"\{.*\}", raw, flags=re.DOTALL)
        if not match:
            heading_logger.warning("[%s] Sarvam response did not contain JSON; response=%r", page_label, raw[:500])
            return None

        result = json.loads(match.group(0))
        if not result.get("is_chapter_heading"):
            heading_logger.info("[%s] Sarvam classified page as NOT a chapter heading", page_label)
            return None

        confidence = float(result.get("confidence", 0))
        number = _normalize_chapter_number(str(result.get("chapter_number") or "").strip())
        title = str(result.get("chapter_title") or "").strip()
        if not number or not title:
            heading_logger.warning("[%s] Invalid heading response: missing number/title: %s", page_label, result)
            return None
        if not re.fullmatch(r"\d{1,3}", number) or len(title) > 100:
            heading_logger.warning("[%s] Rejected implausible heading number/title: number=%r title=%r", page_label, number, title)
            return None
        if confidence < 0.60:
            heading_logger.warning("[%s] Rejected low-confidence heading confidence=%.2f title=%r number=%s", page_label, confidence, title, number)
            return None

        section = f"{number} — {title}"
        heading_logger.info("[%s] Sarvam detected heading: %r (confidence=%.2f)", page_label, section, confidence)
        return section
    except Exception:
        heading_logger.exception("[%s] Sarvam heading detection failed; caller will keep rule result/current section", page_label)
        return None


def read_pdf_records(file_path):
    """OCR PDF pages with layout-aware chapter detection and stable section metadata."""
    records = []
    current_section = "Unspecified section"
    try:
        with fitz.open(str(file_path)) as doc:
            for page_number, page in enumerate(doc, start=1):
                pix = page.get_pixmap(dpi=300, alpha=False)
                image = Image.open(BytesIO(pix.tobytes("png")))
                ocr_lines = _ocr_lines_with_layout(image)
                page_text = "\n".join(line["text"] for line in ocr_lines)
                # Try inexpensive deterministic detection first, then use Claude
                # vision only when the rules cannot confidently pair title/number.
                chapter_heading = _detect_chapter_heading(
                    ocr_lines, image.width, image.height
                )
                heading_method = "rules" if chapter_heading else None
                if not chapter_heading:
                    chapter_heading = detect_heading_with_ai(image, ocr_lines, page_number=page_number)
                    if chapter_heading:
                        heading_method = "AI vision"

                if chapter_heading:
                    current_section = chapter_heading[:200]

                if page_text.strip():
                    records.append({
                        "source": file_path.name,
                        "page_number": page_number,
                        "section_name": current_section,
                        "text": page_text,
                    })

                heading_logger.info(
                    "PDF page %d result: section=%r; detected=%s; method=%s; OCR_lines=%d; text_chars=%d",
                    page_number, current_section, bool(chapter_heading),
                    heading_method or "none", len(ocr_lines), len(page_text)
                )
    except Exception as e:
        print(f"Error OCR-reading PDF {file_path}: {e}")
    return records


def read_docx_records(file_path):
    """Extract DOCX paragraphs and group them under Word heading styles.

    python-docx does not calculate rendered page numbers, so page_number is None.
    """
    records = []
    current_section = "Unspecified section"
    section_lines = []
    try:
        doc = Document(str(file_path))
        for paragraph in doc.paragraphs:
            text = paragraph.text.strip()
            if not text:
                continue
            style_name = getattr(getattr(paragraph, "style", None), "name", "") or ""
            if style_name.lower().startswith("heading") or style_name.lower() == "title":
                if section_lines:
                    records.append({"source": file_path.name, "page_number": None,
                                    "section_name": current_section, "text": "\n".join(section_lines)})
                    section_lines = []
                current_section = text[:200]
            else:
                section_lines.append(text)

        # Include table content under the current section.
        for table in doc.tables:
            for row in table.rows:
                cells = [cell.text.strip().replace("\n", " ") for cell in row.cells]
                row_text = " | ".join(cell for cell in cells if cell)
                if row_text:
                    section_lines.append(row_text)

        if section_lines:
            records.append({"source": file_path.name, "page_number": None,
                            "section_name": current_section, "text": "\n".join(section_lines)})
    except Exception as e:
        print(f"Error reading Word document {file_path}: {e}")
    return records


def read_document_records(file_path):
    """Return source-aware text records for PDF, TXT and DOCX documents."""
    suffix = file_path.suffix.lower()
    if suffix == ".pdf":
        return read_pdf_records(file_path)
    if suffix == ".txt":
        return split_text_into_section_records(read_txt_file(file_path), file_path.name)
    if suffix == ".docx":
        return read_docx_records(file_path)
    return []


def clean_text(text):
    text = text.replace("\x00", " ")
    lines = []
    for line in text.splitlines():
        line = line.strip()
        if line:
            lines.append(line)
    return "\n".join(lines)


def chunk_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    if not text:
        return []
    if overlap >= chunk_size:
        raise ValueError("CHUNK_OVERLAP must be smaller than CHUNK_SIZE")

    chunks = []
    start = 0
    text_length = len(text)

    while start < text_length:
        end = start + chunk_size
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start = end - overlap

    return chunks


# ============================================================
# EMBEDDINGS
# ============================================================

def load_embedding_model():
    print("\nLoading embedding model...")
    model = SentenceTransformer(EMBEDDING_MODEL)
    print("Embedding model loaded.")
    return model


def create_embeddings(model, chunks):
    print("\nCreating embeddings...")
    texts = [f"passage: {chunk}" for chunk in chunks]

    embeddings = model.encode(
        texts,
        normalize_embeddings=True,
        show_progress_bar=True,
    )

    return embeddings


# ============================================================
# INGESTION
# ============================================================

def clear_documents():
    print("\nClearing existing documents...")
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM documents;")
        conn.commit()
    print("Existing documents removed.")


def ingest_documents(model):
    print("\nStarting document ingestion...")

    if not DATA_FOLDER.exists():
        print(f"Data folder does not exist: {DATA_FOLDER.resolve()}")
        return

    files = [
        p for p in DATA_FOLDER.rglob("*")
        if p.is_file() and p.suffix.lower() in [".pdf", ".txt", ".docx"]
    ]

    if not files:
        print(f"No PDF, TXT or DOCX files found in {DATA_FOLDER.resolve()}")
        return

    print(f"Found {len(files)} document(s).")
    print(f"Chunk size: {CHUNK_SIZE}; overlap: {CHUNK_OVERLAP}")
    total_chunks = 0

    with get_connection() as conn:
        with conn.cursor() as cur:
            for file_path in files:
                print(f"\nProcessing: {file_path.name}")
                records = read_document_records(file_path)
                file_chunks = []
                chunk_number = 0

                for record in records:
                    text = clean_text(record["text"])
                    if not text:
                        continue
                    for chunk in chunk_text(text):
                        chunk_number += 1
                        file_chunks.append({
                            "content": chunk,
                            "source": record["source"],
                            "page_number": record["page_number"],
                            "section_name": record["section_name"],
                            "chunk_number": chunk_number,
                        })

                if not file_chunks:
                    print(f"No text extracted from {file_path.name}")
                    continue
                print(f"chunks: {file_chunks[:3]}")  # Show first 3 chunks for verification
                print(f"Created {len(file_chunks)} chunks with source/page/section metadata.")
                embeddings = create_embeddings(model, [item["content"] for item in file_chunks])
                
                for item, embedding in zip(file_chunks, embeddings):
                    cur.execute(
                        """
                        INSERT INTO documents
                            (content, source, chunk_number, embedding, file_type, page_number, section_name)
                        VALUES (%s, %s, %s, %s, %s, %s, %s)
                        """,
                        (
                            item["content"], item["source"], item["chunk_number"],
                            Vector(embedding.tolist()), file_path.suffix.lower(),
                            item["page_number"], item["section_name"],
                        ),
                    )
                total_chunks += len(file_chunks)

        conn.commit()

    print(f"\nIngestion completed. Total chunks inserted: {total_chunks}")
    print("Citation metadata is stored. Re-ingest existing files to populate it for old chunks.")


# ============================================================
# RETRIEVAL + METADATA FILTERING
# ============================================================

def search_similar_chunks(
    query_vector,
    top_k=TOP_K,
    source_filter=SOURCE_FILTER,
    file_type_filter=FILE_TYPE_FILTER,
):
    """Vector search with optional metadata filters."""

    conditions = []
    params = []

    if source_filter:
        conditions.append("source = %s")
        params.append(source_filter)

    if file_type_filter:
        conditions.append("file_type = %s")
        params.append(file_type_filter)

    where_clause = ""
    if conditions:
        where_clause = "WHERE " + " AND ".join(conditions)

    sql = f"""
        SELECT
            id,
            content,
            source,
            chunk_number,
            file_type,
            page_number,
            section_name,
            embedding <=> %s AS distance
        FROM documents
        {where_clause}
        ORDER BY embedding <=> %s
        LIMIT %s
    """

    params_for_query = [query_vector] + params + [query_vector, top_k]

    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, params_for_query)
            return cur.fetchall()


# ============================================================
# SIMPLE RERANKING
# ============================================================

def tokenize(text):
    return set(re.findall(r"[\w\u0900-\u097F]+", text.lower()))


def lexical_score(query, content):
    query_words = tokenize(query)
    content_words = tokenize(content)
    if not query_words:
        return 0.0
    return len(query_words & content_words) / len(query_words)


def rerank_chunks(question, rows, top_k=TOP_K):
    """
    Lightweight reranker for Day 12.

    It combines vector similarity with lexical overlap. This is intentionally
    simple so the experiment runs locally without downloading another model.
    """

    reranked = []

    for row in rows:
        similarity = 1.0 - float(row["distance"])
        lexical = lexical_score(question, row["content"])

        # Vector similarity remains dominant; lexical matching is a boost.
        rerank_score = (0.80 * similarity) + (0.20 * lexical)

        item = dict(row)
        item["vector_similarity"] = similarity
        item["lexical_score"] = lexical
        item["rerank_score"] = rerank_score
        reranked.append(item)

    reranked.sort(key=lambda x: x["rerank_score"], reverse=True)
    return reranked[:top_k]


# ============================================================
# DAY 12 RETRIEVAL QUALITY TESTS
# ============================================================

def keyword_coverage(rows, keywords):
    """
    Check evidence cumulatively across retrieved chunks.

    This fixes the old evaluation problem where ALL keywords had to occur
    in one chunk. A multi-fact RAG question may legitimately require several
    chunks, so each expected fact is tracked independently.

    Returns:
        covered_keywords: keywords found in the supplied rows
        evidence: mapping keyword -> first rank containing that keyword
    """
    covered_keywords = set()
    evidence = {}

    for rank, row in enumerate(rows, start=1):
        content = row["content"].lower()
        for keyword in keywords:
            if keyword.lower() in content and keyword not in covered_keywords:
                covered_keywords.add(keyword)
                evidence[keyword] = rank

    return covered_keywords, evidence


def coverage_rank(rows, keywords):
    """
    Return the first K at which ALL expected facts are covered by the
    cumulative top-K chunks.

    Example:
        Chunk 1 -> fact A
        Chunk 2 -> fact B
        Chunk 3 -> fact C

    The question gets coverage rank 3, rather than being marked MISS.
    """
    covered = set()

    for rank, row in enumerate(rows, start=1):
        content = row["content"].lower()

        for keyword in keywords:
            if keyword.lower() in content:
                covered.add(keyword)

        if len(covered) == len(keywords):
            return rank

    return None


def evaluate_retrieval(
    question,
    keywords,
    model,
    top_k=TOP_K,
    rerank=ENABLE_RERANKING,
):
    query_text = f"query: {question}"
    query_embedding = model.encode(query_text, normalize_embeddings=True)
    query_vector = Vector(query_embedding.tolist())

    print(f"question: {question}")
    # Retrieve a larger candidate pool before reranking.
    candidate_rows = search_similar_chunks(
        query_vector,
        top_k=max(RETRIEVAL_CANDIDATES, top_k),
        source_filter=SOURCE_FILTER,
        file_type_filter=FILE_TYPE_FILTER,
    )
    print(f"candidate pool size: {len(candidate_rows)}")
    # IMPORTANT:
    # coverage_rank evaluates cumulative evidence, not whether one chunk
    # contains every keyword.
    vector_rank = coverage_rank(candidate_rows, keywords)
    print("Vector rank (cumulative evidence):", vector_rank)
    final_rows = candidate_rows
    rerank_scores = []
    if rerank:
        final_rows = rerank_chunks(question, candidate_rows, top_k=top_k)

        # For evaluation we need to see the full reranked candidate list,
        # not only TOP_K. Otherwise a fact at rerank position 6 can never be
        # evaluated for Hit@10.
        reranked_all = []
        for row in candidate_rows:
            similarity = 1.0 - float(row["distance"])
            lexical = lexical_score(question, row["content"])
            item = dict(row)
            item["vector_similarity"] = similarity
            item["lexical_score"] = lexical
            item["rerank_score"] = (0.80 * similarity) + (0.20 * lexical)
            #2print("Rerank score for row id", row["id"], ":", item["rerank_score"])
            reranked_all.append(item)

        reranked_all.sort(key=lambda x: x["rerank_score"], reverse=True)
        rerank_scores = reranked_all
        rerank_rank = coverage_rank(reranked_all, keywords)
    else:
        rerank_rank = vector_rank

    # Evidence detail is useful for debugging multi-chunk questions.
    _, vector_evidence = keyword_coverage(candidate_rows, keywords)
    _, rerank_evidence = keyword_coverage(
        rerank_scores if rerank_scores else candidate_rows,
        keywords,
    )

    return {
        "candidate_rows": candidate_rows,
        "final_rows": final_rows,
        "vector_rank": vector_rank,
        "rerank_rank": rerank_rank,
        "vector_evidence": vector_evidence,
        "rerank_evidence": rerank_evidence,
    }


def format_evidence(evidence):
    """Format keyword -> first supporting chunk rank for console output."""
    if not evidence:
        return "none"
    return "; ".join(f"{keyword}:R{rank}" for keyword, rank in evidence.items())


def run_retrieval_quality_tests(model):
    print("\\n" + "=" * 90)
    print("DAY 12 - RETRIEVAL QUALITY TEST")
    print("=" * 90)
    print(f"Chunk size       : {CHUNK_SIZE}")
    print(f"Chunk overlap    : {CHUNK_OVERLAP}")
    print(f"Top-K            : {TOP_K}")
    print(f"Candidates       : {RETRIEVAL_CANDIDATES}")
    print(f"Reranking        : {ENABLE_RERANKING}")
    print(f"Source filter    : {SOURCE_FILTER}")
    print(f"File type filter : {FILE_TYPE_FILTER}")
    print("Evaluation       : cumulative multi-chunk evidence coverage")

    results = []

    for test in RETRIEVAL_TESTS:
        try:
            result = evaluate_retrieval(
                test["question"],
                test["keywords"],
                model,
            )

            vector_rank = result["vector_rank"]
            rerank_rank = result["rerank_rank"]
            results.append((vector_rank, rerank_rank))

            vector_text = str(vector_rank) if vector_rank else "MISS"
            rerank_text = str(rerank_rank) if rerank_rank else "MISS"
            mode = "MULTI" if test.get("multi_chunk", False) else "SINGLE/MULTI"

            print(
                f"Q{test['id']:02d} | "
                f"Vector coverage: {vector_text:>4} | "
                f"Rerank coverage: {rerank_text:>4} | "
                f"{mode:11} | "
                f"{test['question']}"
            )

            # For multi-fact questions, show exactly where each fact was found.
            if test.get("multi_chunk", False):
                print(
                    "     Vector evidence: "
                    + format_evidence(result["vector_evidence"])
                )
                print(
                    "     Rerank evidence: "
                    + format_evidence(result["rerank_evidence"])
                )

        except Exception as e:
            print(f"Q{test['id']:02d} | ERROR | {type(e).__name__}: {e}")
            results.append((None, None))

    print("\\n" + "-" * 90)
    print("CUMULATIVE COVERAGE SUMMARY")
    print("-" * 90)

    vector_hits_1 = sum(1 for v, _ in results if v and v <= 1)
    vector_hits_3 = sum(1 for v, _ in results if v and v <= 3)
    vector_hits_5 = sum(1 for v, _ in results if v and v <= 5)

    rerank_hits_1 = sum(1 for _, r in results if r and r <= 1)
    rerank_hits_3 = sum(1 for _, r in results if r and r <= 3)
    rerank_hits_5 = sum(1 for _, r in results if r and r <= 5)

    total = len(results)

    print(f"Vector Coverage@1 : {vector_hits_1}/{total}")
    print(f"Vector Coverage@3 : {vector_hits_3}/{total}")
    print(f"Vector Coverage@5 : {vector_hits_5}/{total}")
    print(f"Rerank Coverage@1 : {rerank_hits_1}/{total}")
    print(f"Rerank Coverage@3 : {rerank_hits_3}/{total}")
    print(f"Rerank Coverage@5 : {rerank_hits_5}/{total}")
    print("=" * 90)
    print(
        "Interpretation: Coverage@K means ALL expected facts are found "
        "across the first K retrieved chunks, not necessarily in one chunk."
    )
    print(
        "This is especially important for questions whose answer is spread "
        "across multiple chunks."
    )
    print("=" * 90)


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def vector_to_numpy(value):
    """Convert a pgvector.Vector or list-like value into a 1-D NumPy float array."""
    if value is None:
        return np.array([], dtype=float)

    if isinstance(value, Vector):
        if hasattr(value, "to_numpy"):
            value = value.to_numpy()
        elif hasattr(value, "to_list"):
            value = value.to_list()
        else:
            text = str(value)
            inner = text.removeprefix("Vector(").removesuffix(")")
            inner = inner.strip()
            if inner.startswith("[") and inner.endswith("]"):
                inner = inner[1:-1]
            value = [float(part.strip()) for part in inner.split(",") if part.strip()]
    elif hasattr(value, "tolist"):
        try:
            value = value.tolist()
        except Exception:
            pass

    if isinstance(value, str):
        cleaned = value.strip().strip("[]")
        if not cleaned:
            return np.array([], dtype=float)
        value = [float(part.strip()) for part in cleaned.split(",") if part.strip()]

    if not hasattr(value, "__iter__") or isinstance(value, (str, bytes)):
        raise TypeError(f"Expected a vector-like value, got {type(value).__name__}")

    value = list(value)
    return np.asarray(value, dtype=float).reshape(-1)


# ============================================================
# DEBUG TEST - DHANUSH
# ============================================================

def test_vector_search(model, search_text="धनुष"):
    print("\n" + "=" * 70)
    print("VECTOR SEARCH TEST")
    print("=" * 70)
    print(f"Question: {search_text}")

    query_embedding = model.encode(
        f"query: {search_text}",
        normalize_embeddings=True,
    )
    query_vector = Vector(query_embedding.tolist())

    rows = search_similar_chunks(query_vector, top_k=10)

    found_target = False

    for rank, row in enumerate(rows, start=1):
        similarity = 1 - float(row["distance"])

        print("\n" + "-" * 70)
        print(f"Rank       : {rank}")
        print(f"ID         : {row['id']}")
        print(f"Source     : {row['source']}")
        print(f"Chunk      : {row['chunk_number']}")
        print(f"Similarity : {similarity:.4f}")

        if "एक धनुष चार हाथ का होता है" in row["content"]:
            print("\n🎯 TARGET SENTENCE FOUND!")
            found_target = True

        print("\nContent preview:")
        print(row["content"][:700])

    print("\n" + "=" * 70)
    print("✅ SUCCESS!" if found_target else "❌ TARGET SENTENCE NOT FOUND IN TOP 10.")
    print("=" * 70)


# ============================================================
# SEND RETRIEVED CONTEXT TO SARVAM
# ============================================================

def ask_sarvam(question, rows):
    """Generate a grounded RAG answer using Sarvam Chat Completions."""
    if not rows:
        return "I could not find any relevant information in the provided documents."

#    from sarvamai import SarvamAI

#    api_key = "sk_bj9bk4vb_nhS6PFFIHACOZGHDlrLHJmgi"
#    if not api_key:
#        raise RuntimeError(
#            "SARVAM_API_KEY is not set. Configure it as a Windows environment variable "
#            "before running this program."
#        )

    context_parts = []
    for rank, row in enumerate(rows, start=1):
        content = row.get("content")
        source = row.get("source")
        chunk_number = row.get("chunk_number")
        similarity = row.get("distance")
        similarity_text = (
            f"{1 - float(similarity):.4f}" if similarity is not None else "not available"
        )

        if not content:
            continue

        context_parts.append(
            f"SOURCE {rank}\n"
            f"File: {source}\n"
            f"Chunk: {chunk_number}\n"
            f"Similarity: {similarity_text}\n\n"
            f"{content}"
        )

    context = "\n\n---\n\n".join(context_parts)
    system_prompt = (
        "You are a research assistant."
        "Answer the user's question using the supplied retrieved documents."
        "IMPORTANT RULES:"
        "1. Use the retrieved context as your primary source."
        "2. Do not invent facts that are not supported by the context."
        "3. If the context does not contain enough information,"
        "   clearly say that the information is not available."
        "4. Cite the source file and section for factual claims."
        "5. Include the PDF page number when provided. Do not invent page numbers."
        "6. Use only the file/page/section metadata supplied in the context."
        "7. Give a clear and concise answer."
        "8. Conclude in one or 2 lines if possible."
        )
    user_prompt = (
        "Use the retrieved context below as the evidence for your answer.\n\n"
        f"CONTEXT:\n{context}\n\n"
        f"QUESTION:\n{question}\n\n"
    )
    print("user_prompt:", user_prompt)
#    client = SarvamAI(api_subscription_key=api_key)
    response = sarvam_client.chat.completions(
        model=SARVAM_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.7,
        max_tokens=1200,
        reasoning_effort=None,  # Disable reasoning-token consumption
    )

    answer = response.choices[0].message.content
    if not answer:
        raise RuntimeError("Sarvam returned an empty answer.")
    return answer




# ============================================================
# 11. SEND RETRIEVED CONTEXT TO CLAUDE
# ============================================================

def ask_claude(question, rows):

    print("\nSending retrieved context to Claude...")

    context_parts = []

    for rank, row in enumerate(rows, start=1):
        if isinstance(row, dict):
            content = row.get("content")
            source = row.get("source")
            chunk_number = row.get("chunk_number")
            page_number = row.get("page_number")
            section_name = row.get("section_name")
        else:
            content = row[1] if len(row) > 1 else None
            source = row[2] if len(row) > 2 else None
            chunk_number = row[3] if len(row) > 3 else None
            page_number = None
            section_name = None

        if content is None:
            continue

        context_parts.append(
            f"""
SOURCE {rank}
File: {source}
Page: {page_number if page_number is not None else 'not available for this format'}
Section: {section_name or 'Unspecified section'}
Chunk: {chunk_number}

{content}
"""
        )

    context = "\n".join(context_parts)

    prompt = f"""
You are a research assistant.

Answer the user's question using the supplied
retrieved documents.

IMPORTANT RULES:

1. Use the retrieved context as your primary source.
2. Do not invent facts that are not supported by the context.
3. If the context does not contain enough information,
   clearly say that the information is not available.
4. Cite the source file and section for factual claims.
5. Include the PDF page number when provided. Do not invent page numbers.
6. Use only the file/page/section metadata supplied in the context.
7. Give a clear and concise answer.
8. Conclude in one or 2 lines if possible.

USER QUESTION:
{question}

RETRIEVED CONTEXT:
{context}
"""


    message = client.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=1200,
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    answer_parts = []

    for block in message.content:
        if block.type == "text":
            answer_parts.append(block.text)

    return "\n".join(answer_parts)
# ============================================================
# LOCAL LLM
# ============================================================

def ask_local_llm(question, rows):
    if not rows:
        return "I could not find any relevant information in the provided documents."

    print("\nSending retrieved context to local LLM...")
    context_parts = []
    for row in rows:
        context_parts.append(
            f"Source: {row['source']}\n"
            f"Chunk: {row['chunk_number']}\n"
            f"Page: {row.get('page_number') if row.get('page_number') is not None else 'not available for this format'}\n"
            f"Section: {row.get('section_name') or 'Unspecified section'}\n"
            f"Similarity: {1 - float(row['distance']):.4f}\n"
            f"{row['content']}"
        )

    context = "\n\n---\n\n".join(context_parts)

    prompt = f"""
You are a strict evidence-based RAG assistant.

Your job is to answer ONLY using the retrieved context below.
Do NOT use outside knowledge.
Do NOT infer unsupported facts.
Do NOT invent data, numbers, names, page numbers, or section names.
Cite factual claims with the supplied source filename, page number (when available), and section name.
If the answer is not explicitly present in the retrieved context, say:
"I could not find the answer in the provided documents."

RETRIEVED CONTEXT:
{context}

USER QUESTION:
{question}

Return a short, factual answer in Hindi/English as appropriate.
"""

    try:
        response = ollama.chat(
            model=LOCAL_LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.message.content
    except Exception as e:
        return (
            "Error communicating with Ollama.\n\n"
            f"{type(e).__name__}: {e}"
        )


# ============================================================
# ASK QUESTION
# ============================================================

def ask_question(question, embedding_model):
    print("\nCreating question embedding...%s", question)

    query_text = f"query: {question}"
    query_embedding = embedding_model.encode(
        query_text,
        normalize_embeddings=True,
    )
    query_vector = Vector(query_embedding.tolist())

    print("Searching PostgreSQL + pgvector...")

    candidates = search_similar_chunks(
        query_vector,
        top_k=max(RETRIEVAL_CANDIDATES, TOP_K),
        source_filter=SOURCE_FILTER,
        file_type_filter=FILE_TYPE_FILTER,
    )
    print(f"Found {len(candidates)} candidate chunks.")
    rows = rerank_chunks(question, candidates, TOP_K) if ENABLE_RERANKING else candidates[:TOP_K]
    print(f"Returning top {len(rows)} chunks after reranking." if ENABLE_RERANKING else f"Returning top {len(rows)} chunks without reranking.")
    if not rows:
        print("\nNo matching documents found.")
        return

    print(f"Retrieved {len(rows)} relevant chunks.")
    print("\nGenerating answer using grounded retriever context...")

    if PREFER_CLAUDE:
        answer = ask_sarvam(question, rows)
        #answer = ask_claude(question, rows)
    else:
        answer = ask_local_llm(question, rows)

    print("\n" + "=" * 70)
    print("ANSWER")
    print("=" * 70)
    print(answer)

    print("\n" + "=" * 70)
    print("RETRIEVED SOURCES")
    print("=" * 70)

    for rank, row in enumerate(rows, start=1):
        page = row.get("page_number")
        section = row.get("section_name") or "Unspecified section"
        page_text = f"page {page}" if page is not None else "page unavailable"
        print(
            f"{rank}. {row['source']} | {page_text} | section: {section} "
            f"| chunk {row['chunk_number']} | similarity: {1 - float(row['distance']):.4f}"
        )


# ============================================================
# DAY 13 - RAGAS EVALUATION
# ============================================================

def run_ragas_evaluation(embedding_model):
    """Evaluate answer grounding and retrieval using Ragas + Claude.

    Install optional dependencies in the active Python environment:
        pip install "ragas>=0.2,<1.0" langchain-anthropic langchain-huggingface

    This function is deliberately separate from normal question answering.
    Claude generates RAG answers and also acts as the Ragas LLM judge.
    ANTHROPIC_API_KEY must be configured as an environment variable.
    """
    try:
        from ragas import EvaluationDataset, evaluate
        from ragas.llms import LangchainLLMWrapper
        from ragas.metrics import (
            Faithfulness,
            AnswerRelevancy,
            LLMContextPrecisionWithReference,
            LLMContextRecall,
        )
        from langchain_anthropic import ChatAnthropic
        from langchain_huggingface import HuggingFaceEmbeddings
        from ragas.embeddings import LangchainEmbeddingsWrapper
    except ImportError as exc:
        print("\nRagas dependencies are missing or incompatible.")
        print("Install them in the same Python environment running this program:")
        print('  pip install "ragas>=0.2,<1.0" langchain-anthropic langchain-huggingface')
        print(f"Import error: {exc}")
        return

    # References are manually curated from the known Ishtopdesh facts.
    # Keep these verified; Ragas scores are only as trustworthy as references.
    reference_answers = {
        1: "एक धनुष चार हाथ का होता है।",
        2: "भगवान ऋषभदेव की काया 500 धनुष की थी।",
        3: "बाहुबली की काया 525 धनुष की थी।",
        5: "ऋषभदेव की काया 500 धनुष और बाहुबली की काया 525 धनुष की थी; अंतर 25 धनुष था।",
        7: "ग्रंथ के अनुसार बाहुबली की काया 525 धनुष की थी।",
        9: "बाहुबली की काया 525 धनुष की थी।",
        10: "भगवान ऋषभदेव की काया 500 धनुष की थी।",
        11: "एक धनुष चार हाथ का होता है।",
        20: "ऋषभदेव की काया 500 धनुष, बाहुबली की काया 525 धनुष की थी, एक धनुष चार हाथ का होता है, और पाठ में भगवान की काया 2000 हाथ बताई गई है।",
    }

    # Five questions by default to keep evaluation time manageable on CPU.
    selected_tests = [
        test for test in RETRIEVAL_TESTS
        if test["id"] in reference_answers
    ][:max(1, RAGAS_MAX_QUESTIONS)]

    print("\n" + "=" * 90)
    print("DAY 13 - RAGAS EVALUATION")
    print("=" * 90)
    print(f"Questions to evaluate : {len(selected_tests)}")
    print(f"Ragas evaluator       : Claude / {CLAUDE_MODEL}")
    print("Metrics               : faithfulness, answer relevancy, context precision, context recall")
    print("Note                  : This may take several minutes on a CPU-only laptop.")

    samples = []
    for index, test in enumerate(selected_tests, start=1):
        question = test["question"]
        print(f"\n[{index}/{len(selected_tests)}] Retrieving evidence for: {question}")
        try:
            query_embedding = embedding_model.encode(
                f"query: {question}", normalize_embeddings=True
            )
            query_vector = Vector(query_embedding.tolist())
            candidates = search_similar_chunks(
                query_vector,
                top_k=max(RETRIEVAL_CANDIDATES, TOP_K),
                source_filter=SOURCE_FILTER,
                file_type_filter=FILE_TYPE_FILTER,
            )
            rows = (
                rerank_chunks(question, candidates, TOP_K)
                if ENABLE_RERANKING else candidates[:TOP_K]
            )
            if not rows:
                print("  Skipped: no retrieved chunks.")
                continue

            # Generate a fresh answer using the same Claude model used by the app.
            answer = ask_claude(question, rows)
            contexts = [str(row["content"]) for row in rows]
            samples.append({
                "user_input": question,
                "retrieved_contexts": contexts,
                "response": answer,
                "reference": reference_answers[test["id"]],
            })
            print(f"  Retrieved contexts: {len(contexts)}; answer generated.")
        except Exception as exc:
            print(f"  Failed: {type(exc).__name__}: {exc}")

    if not samples:
        print("No Ragas samples were created; check database, Anthropic API key, and Claude model configuration.")
        return

    try:
        # Use Claude for LLM-judge metrics and the local multilingual
        # embedding model for answer relevancy. ANTHROPIC_API_KEY is required.
       
        evaluator_chat_model = ChatAnthropic(
            model=CLAUDE_MODEL,
            max_tokens=512,
        )
        evaluator_llm = LangchainLLMWrapper(evaluator_chat_model, bypass_temperature=True )
        evaluator_embedding_model = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            encode_kwargs={"normalize_embeddings": True},
        )
        evaluator_embeddings = LangchainEmbeddingsWrapper(evaluator_embedding_model)

        metrics = [
            Faithfulness(llm=evaluator_llm),
            AnswerRelevancy(llm=evaluator_llm, embeddings=evaluator_embeddings),
            LLMContextPrecisionWithReference(llm=evaluator_llm),
            LLMContextRecall(llm=evaluator_llm),
        ]

        dataset = EvaluationDataset.from_list(samples)
        print("\nRunning Ragas metrics. This makes additional Claude API calls and may incur cost...")
        result = evaluate(dataset=dataset, metrics=metrics)
        results_df = result.to_pandas()
        results_df.to_csv(RAGAS_OUTPUT_FILE, index=False, encoding="utf-8-sig")

        print("\nRAGAS RESULTS")
        print("=" * 90)
        print(results_df.to_string(index=False))
        print(f"\nSaved per-question results to: {RAGAS_OUTPUT_FILE.resolve()}")
        print("\nScores generally range from 0 to 1; higher is better.")
        print("Review sample answers manually, especially for Hindi text and numeric facts.")
    except Exception as exc:
        print("\nRagas evaluation failed.")
        print(f"{type(exc).__name__}: {exc}")
        print("Check the installed Ragas/LangChain versions and that Ollama is running.")


# ============================================================
# CHECKS / STATISTICS
# ============================================================

def check_database():
    print("\nChecking PostgreSQL connection...")
    try:
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT version();")
                result = cur.fetchone()
                print("PostgreSQL connection successful.")
                print(f"Database: {DB_NAME}")
                print(f"Server: {result['version']}")
                cur.execute("SELECT COUNT(*) AS count FROM documents;")
                print(f"Stored chunks: {cur.fetchone()['count']}")
        return True
    except Exception as e:
        print(f"\nPostgreSQL connection failed: {type(e).__name__}: {e}")
        return False


def check_ollama():
    print("\nChecking Ollama...")
    try:
        models = ollama.list()
        model_names = [model.model for model in models.models]
        if LOCAL_LLM_MODEL in model_names or any(n.startswith(LOCAL_LLM_MODEL) for n in model_names):
            print(f"Ollama model '{LOCAL_LLM_MODEL}' is installed.")
            return True
        print(f"Model '{LOCAL_LLM_MODEL}' was not found.")
        print(f"Run: ollama pull {LOCAL_LLM_MODEL}")
        return False
    except Exception as e:
        print(f"\nCould not connect to Ollama: {type(e).__name__}: {e}")
        return False


def show_database_stats():
    print("\nDatabase statistics")
    print("-" * 50)
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT COUNT(*) AS total_chunks,
                       COUNT(DISTINCT source) AS total_sources
                FROM documents;
                """
            )
            row = cur.fetchone()
            print(f"Documents: {row['total_sources']}")
            print(f"Chunks:    {row['total_chunks']}")


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 90)
    print("ISHTOPDESH RAG APPLICATION - DAY 12 RETRIEVAL QUALITY")
    print("=" * 90)
    print(f"Embedding model : {EMBEDDING_MODEL}")
    print(f"Chunk size      : {CHUNK_SIZE}")
    print(f"Chunk overlap   : {CHUNK_OVERLAP}")
    print(f"Top-K           : {TOP_K}")
    print(f"Reranking       : {ENABLE_RERANKING}")

    if not check_ollama():
        return

    setup_database()
    embedding_model = load_embedding_model()

    print("\n" + "=" * 90)
    print("MENU")
    print("=" * 90)
    print("1. Re-ingest documents")
    print("2. Run 20 retrieval-quality tests")
    print("3. Test 'धनुष' retrieval")
    print("4. Ask a RAG question")
    print("5. Show database statistics")
    print("6. Run Ragas evaluation (local LLM; may take several minutes)")
    print("7. Exit")

    while True:
        choice = input("\nSelect option (1-10): ").strip()

        if choice == "1":
            clear_documents()
            ingest_documents(embedding_model)

        elif choice == "2":
            run_retrieval_quality_tests(embedding_model)

        elif choice == "3":
            test_vector_search(embedding_model, "धनुष")

        elif choice == "4":
            question = input("Question: ").strip()
            if question:
                ask_question(question, embedding_model)

        elif choice == "5":
            show_database_stats()

        elif choice == "6":
            run_ragas_evaluation(embedding_model)

        elif choice == "7":
            print("\nExiting application.")
            break
        else:
            print("Invalid choice. Please select 1-10.")


if __name__ == "__main__":
    main()

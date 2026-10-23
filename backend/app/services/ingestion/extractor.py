"""
Document text extractor for PDF, DOCX, TXT, and Markdown.

PDF uses PyMuPDF (reading-order block extraction).
DOCX uses python-docx (paragraphs + tables).
TXT uses chardet for encoding detection.
MD strips markup and splits on headings.
"""

from __future__ import annotations

import io
import re
from dataclasses import dataclass, field
from pathlib import Path

import chardet
import structlog

from app.core.exceptions import ExtractionError, UnsupportedFileType

log = structlog.get_logger(__name__)


@dataclass
class ExtractedPage:
    """Text content of a single page or section."""
    page_number: int
    text: str
    heading: str | None = None   # Section heading if detected


@dataclass
class ExtractionResult:
    """Complete extraction result for one document."""
    filename: str
    file_type: str
    pages: list[ExtractedPage] = field(default_factory=list)
    language_hint: str | None = None  # ISO 639-1 code detected by langdetect

    @property
    def full_text(self) -> str:
        return "\n\n".join(p.text for p in self.pages if p.text.strip())

    @property
    def total_chars(self) -> int:
        return sum(len(p.text) for p in self.pages)


def extract(file_path: Path, filename: str) -> ExtractionResult:
    """Extract text from a file.  Dispatches based on extension.

    Args:
        file_path:  Path to the file on disk.
        filename:   Original filename (used for extension detection and logging).

    Returns:
        ExtractionResult with pages list and metadata.

    Raises:
        UnsupportedFileType: if the file extension is not supported.
        ExtractionError: if parsing fails.
    """
    suffix = Path(filename).suffix.lower().lstrip(".")
    log.info("extractor.start", filename=filename, file_type=suffix)

    extractors = {
        "pdf": _extract_pdf,
        "docx": _extract_docx,
        "txt": _extract_txt,
        "md": _extract_markdown,
        "markdown": _extract_markdown,
    }

    if suffix not in extractors:
        raise UnsupportedFileType(
            f"Unsupported file type: .{suffix}. Allowed: pdf, docx, txt, md"
        )

    try:
        result = extractors[suffix](file_path, filename)
        log.info(
            "extractor.done",
            filename=filename,
            pages=len(result.pages),
            chars=result.total_chars,
        )
        return result
    except (UnsupportedFileType, ExtractionError):
        raise
    except Exception as exc:
        raise ExtractionError(
            f"Failed to extract text from {filename}: {exc}"
        ) from exc


# ─── PDF ─────────────────────────────────────────────────────────────────────

def _extract_pdf(file_path: Path, filename: str) -> ExtractionResult:
    """Extract text from PDF using PyMuPDF."""
    try:
        import fitz  # PyMuPDF
    except ImportError as exc:
        raise ExtractionError("pymupdf not installed. Run: pip install pymupdf") from exc

    result = ExtractionResult(filename=filename, file_type="pdf")
    doc = fitz.open(str(file_path))

    for page_num, page in enumerate(doc, start=1):
        # get_text("blocks") returns text in reading order (top-left → bottom-right)
        blocks = page.get_text("blocks")
        page_text_parts = []
        for block in blocks:
            # block: (x0, y0, x1, y1, text, block_num, block_type)
            if block[6] == 0:   # block_type 0 = text (not image)
                text = block[4].strip()
                if text:
                    page_text_parts.append(text)

        page_text = "\n".join(page_text_parts)
        if page_text.strip():
            result.pages.append(ExtractedPage(page_number=page_num, text=page_text))

    doc.close()

    if not result.pages:
        raise ExtractionError(
            f"{filename} appears to be a scanned PDF with no extractable text. "
            "OCR support is not yet implemented."
        )
    return result


# ─── DOCX ────────────────────────────────────────────────────────────────────

def _extract_docx(file_path: Path, filename: str) -> ExtractionResult:
    """Extract text from DOCX, treating each section as a 'page'."""
    try:
        import docx
    except ImportError as exc:
        raise ExtractionError("python-docx not installed. Run: pip install python-docx") from exc

    result = ExtractionResult(filename=filename, file_type="docx")
    document = docx.Document(str(file_path))

    current_section: list[str] = []
    section_num = 1
    current_heading: str | None = None

    for para in document.paragraphs:
        text = para.text.strip()
        if not text:
            continue

        # Detect headings by style name
        if para.style.name.startswith("Heading"):
            if current_section:
                result.pages.append(ExtractedPage(
                    page_number=section_num,
                    text="\n".join(current_section),
                    heading=current_heading,
                ))
                section_num += 1
                current_section = []
            current_heading = text
        else:
            current_section.append(text)

    # Don't forget the last section
    if current_section:
        result.pages.append(ExtractedPage(
            page_number=section_num,
            text="\n".join(current_section),
            heading=current_heading,
        ))

    # Also extract text from tables
    for table in document.tables:
        table_text = []
        for row in table.rows:
            row_text = " | ".join(cell.text.strip() for cell in row.cells)
            if row_text.strip():
                table_text.append(row_text)
        if table_text:
            result.pages.append(ExtractedPage(
                page_number=section_num,
                text="\n".join(table_text),
                heading="[Table]",
            ))
            section_num += 1

    if not result.pages:
        raise ExtractionError(f"{filename} contains no extractable text.")
    return result


# ─── TXT ─────────────────────────────────────────────────────────────────────

def _extract_txt(file_path: Path, filename: str) -> ExtractionResult:
    """Extract text from plain text file with auto-detected encoding."""
    raw = file_path.read_bytes()
    detected = chardet.detect(raw)
    encoding = detected.get("encoding") or "utf-8"

    try:
        text = raw.decode(encoding, errors="replace")
    except Exception:
        text = raw.decode("utf-8", errors="replace")

    result = ExtractionResult(filename=filename, file_type="txt")
    # Split into pseudo-pages of ~3000 chars each
    chunk_size = 3000
    for i, start in enumerate(range(0, len(text), chunk_size), start=1):
        chunk = text[start : start + chunk_size].strip()
        if chunk:
            result.pages.append(ExtractedPage(page_number=i, text=chunk))

    if not result.pages:
        raise ExtractionError(f"{filename} contains no text content.")
    return result


# ─── Markdown ────────────────────────────────────────────────────────────────

def _extract_markdown(file_path: Path, filename: str) -> ExtractionResult:
    """Extract text from Markdown by stripping markup."""
    try:
        import markdown as md_lib
    except ImportError as exc:
        raise ExtractionError("markdown not installed. Run: pip install markdown") from exc

    raw = file_path.read_bytes()
    detected = chardet.detect(raw)
    encoding = detected.get("encoding") or "utf-8"
    text = raw.decode(encoding, errors="replace")

    # Convert MD → HTML → strip tags → plain text
    html = md_lib.markdown(text)
    plain = re.sub(r"<[^>]+>", " ", html)
    plain = re.sub(r"\s+", " ", plain).strip()

    result = ExtractionResult(filename=filename, file_type="md")
    # Split on headers (## or # lines) to create sections
    sections = re.split(r"\n(?=#{1,3} )", text)
    for i, section in enumerate(sections, start=1):
        section_plain = re.sub(r"[#*`_>\[\]()]+", " ", section)
        section_plain = re.sub(r"\s+", " ", section_plain).strip()
        if section_plain:
            result.pages.append(ExtractedPage(page_number=i, text=section_plain))

    if not result.pages and plain:
        result.pages.append(ExtractedPage(page_number=1, text=plain))

    if not result.pages:
        raise ExtractionError(f"{filename} contains no text content.")
    return result

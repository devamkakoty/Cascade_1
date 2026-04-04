"""
Parse 2D engineering drawings (PNG/JPEG/PDF) to extract dimensions,
tolerances, materials, and part information.

Two extraction strategies:
  1. OCR-based: Uses pytesseract to extract text + bounding boxes from images,
     then applies regex to find dimension callouts.
  2. Fallback regex: If pytesseract is unavailable, converts to text via
     basic image processing and pattern matching.

Both strategies look for:
  - Linear dimensions:  "123.4", "1500 mm", "25.0 ±0.5"
  - Diameter dims:      "Ø12.5", "⌀50 ±0.01"
  - Radius dims:        "R10", "R25.0"
  - Tolerances:         "±0.1", "+0.05 / -0.02", "12.5 +0.1 -0.05"
  - Material callouts:  "Material: AL 6061-T6", "AH36 Marine Steel"
  - Hole callouts:      "4x Ø8.5 THRU"
  - Surface finish:     "Ra 3.2", "Ra 1.6"
  - General notes:      "ALL DIMS IN mm", "TOLERANCES: ±0.5mm"
"""

from __future__ import annotations

import io
import re
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

# Try to import image/OCR libraries — graceful fallback if unavailable
try:
    from PIL import Image
    _HAS_PIL = True
except ImportError:
    _HAS_PIL = False

try:
    import pytesseract
    _HAS_TESSERACT = True
except ImportError:
    _HAS_TESSERACT = False


# ── Data classes ────────────────────────────────────────────────────

@dataclass
class ExtractedDimension:
    """A single dimension extracted from a drawing."""
    dim_type: str          # "linear", "diameter", "radius", "angle"
    value: float
    unit: str = "mm"
    tolerance_plus: float = 0.0
    tolerance_minus: float = 0.0
    raw_text: str = ""
    label: str = ""        # inferred semantic label
    confidence: float = 0.0  # 0-1 confidence score
    bbox: Optional[Tuple[float, float, float, float]] = None  # (x, y, w, h) normalized


@dataclass
class ExtractedNote:
    """A note or callout extracted from a drawing."""
    category: str          # "material", "surface_finish", "tolerance", "hole_pattern", "general"
    text: str
    value: str = ""        # parsed value if applicable
    confidence: float = 0.0


@dataclass
class DrawingParseResult:
    """Complete extraction result from a 2D drawing."""
    dimensions: List[ExtractedDimension] = field(default_factory=list)
    notes: List[ExtractedNote] = field(default_factory=list)
    part_name: str = ""
    material: str = ""
    sector: str = ""       # inferred sector
    raw_text: str = ""     # all OCR text
    method: str = ""       # "ocr" or "regex_fallback"
    ocr_available: bool = False


# ── Regex patterns ──────────────────────────────────────────────────

# Diameter: Ø50.0, ⌀12.5, DIA 25
_RE_DIAMETER = re.compile(
    r'(?:[Øø⌀]|DIA\.?\s*)'
    r'(\d+\.?\d*)'
    r'(?:\s*[±]\s*(\d+\.?\d*))?',
    re.IGNORECASE
)

# Radius: R10, R25.0
_RE_RADIUS = re.compile(
    r'\bR(\d+\.?\d*)\b'
)

# Linear dimension with optional tolerance: 125.0 ±0.5, 1500 mm
_RE_LINEAR = re.compile(
    r'(?<![ØøR⌀])'  # not preceded by diameter/radius symbol
    r'(?<![A-Za-z])'  # not preceded by a letter (avoids material grades)
    r'(\d+\.?\d*)'  # value
    r'(?:\s*[±+]\s*(\d+\.?\d*))?'  # optional ± tolerance
    r'(?:\s*(mm|in|cm|m)\b)?'  # optional unit
    r'(?![\-][A-Za-z])',  # not followed by "-letter" (avoids "2024-T3")
    re.IGNORECASE
)

# Dimension with tolerance on its own line: "8.2 ±0.35"
_RE_DIM_WITH_TOL = re.compile(
    r'(\d+\.?\d+)\s*[±]\s*(\d+\.?\d+)',
)

# Bilateral tolerance: +0.05 / -0.02 or +0.1 -0.05
_RE_BILATERAL_TOL = re.compile(
    r'(\d+\.?\d*)\s*'
    r'[+]\s*(\d+\.?\d*)\s*'
    r'(?:/\s*)?'
    r'[-]\s*(\d+\.?\d*)'
)

# Hole pattern: 4x Ø8.5 THRU, 6x Ø12 ON PCD
_RE_HOLE_PATTERN = re.compile(
    r'(\d+)\s*[xX×]\s*[Øø⌀]?\s*(\d+\.?\d*)'
    r'(?:\s*(THRU|BLIND|CSK|CBORE))?'
    r'(?:\s*ON\s*.*?PCD)?',
    re.IGNORECASE
)

# Material callout
_RE_MATERIAL = re.compile(
    r'(?:MATERIAL|MAT|MATL)[:\s]+(.+?)(?:\n|$)',
    re.IGNORECASE
)

# Surface finish: Ra 3.2, Ra 1.6
_RE_SURFACE_FINISH = re.compile(
    r'Ra\s*(\d+\.?\d*)',
    re.IGNORECASE
)

# General tolerance note
_RE_GENERAL_TOL = re.compile(
    r'(?:GENERAL\s+)?TOLERANCES?[:\s]+[±]?\s*(\d+\.?\d*)\s*(mm|in)?',
    re.IGNORECASE
)

# Known materials for matching
_KNOWN_MATERIALS = [
    "AH36", "DH36", "EH36", "Marine Steel",
    "AL 6061", "AL 7075", "AL 2024", "AL 5083",
    "CFRP", "GFRP",
    "Titanium", "Ti-6Al-4V", "Inconel",
    "Steel 4340", "Steel A36", "Stainless 304", "Stainless 316",
    "Bronze", "Copper", "Brass",
]

# Sector keywords
_SECTOR_KEYWORDS = {
    "naval": ["hull", "deck", "bulkhead", "marine", "keel", "rudder", "propeller", "vessel", "ship", "dnv"],
    "aerospace": ["fuselage", "wing", "spar", "aircraft", "windshield", "airframe", "nacelle", "avionics"],
    "automotive_ev": ["battery", "cell", "module", "pack", "cooling", "bms", "ev", "motor", "chassis"],
}


# Material grade numbers to ignore as dimensions
_MATERIAL_GRADE_NUMBERS = {
    2024, 5083, 6061, 7075, 4340, 304, 316, 718,
    93200, 36, 4130, 4140, 17, 15, 321,
}

# ── Core extraction functions ───────────────────────────────────────

def _is_material_context(text: str, match_start: int) -> bool:
    """Check if a number appears in a material-grade context."""
    # Look at the 20 chars before the match
    prefix = text[max(0, match_start - 20):match_start].upper()
    material_prefixes = ["AL ", "AL-", "STEEL", "STAINLESS", "TI-", "INCONEL",
                         "BRONZE", "CFRP", "MATL", "MATERIAL", "MAT:", "GR."]
    return any(mp in prefix for mp in material_prefixes)


def _extract_dimensions_from_text(text: str) -> List[ExtractedDimension]:
    """Extract dimension values from OCR/text using regex."""
    dims = []
    seen_values = set()

    # Diameters
    for m in _RE_DIAMETER.finditer(text):
        val = float(m.group(1))
        tol = float(m.group(2)) if m.group(2) else 0.0
        if val not in seen_values:
            seen_values.add(val)
            dims.append(ExtractedDimension(
                dim_type="diameter",
                value=val,
                tolerance_plus=tol,
                tolerance_minus=-tol,
                raw_text=m.group(0).strip(),
                confidence=0.85,
            ))

    # Radii
    for m in _RE_RADIUS.finditer(text):
        val = float(m.group(1))
        if val not in seen_values:
            seen_values.add(val)
            dims.append(ExtractedDimension(
                dim_type="radius",
                value=val,
                raw_text=m.group(0).strip(),
                confidence=0.80,
            ))

    # Dimensions with explicit tolerance (high confidence)
    for m in _RE_DIM_WITH_TOL.finditer(text):
        val = float(m.group(1))
        tol = float(m.group(2))
        if val in seen_values:
            continue
        if _is_material_context(text, m.start()):
            continue
        seen_values.add(val)
        dims.append(ExtractedDimension(
            dim_type="linear",
            value=val,
            tolerance_plus=tol,
            tolerance_minus=-tol,
            raw_text=m.group(0).strip(),
            confidence=0.90,
        ))

    # Linear dimensions
    for m in _RE_LINEAR.finditer(text):
        val = float(m.group(1))
        if val in seen_values:
            continue
        # Skip values that are too small (likely noise) or too large
        if val < 0.5 or val > 100000:
            continue
        # Skip material grade numbers
        if int(val) in _MATERIAL_GRADE_NUMBERS and val == int(val):
            continue
        if _is_material_context(text, m.start()):
            continue
        tol = float(m.group(2)) if m.group(2) else 0.0
        unit = m.group(3) if m.group(3) else "mm"
        seen_values.add(val)
        dims.append(ExtractedDimension(
            dim_type="linear",
            value=val,
            unit=unit,
            tolerance_plus=tol,
            tolerance_minus=-tol,
            raw_text=m.group(0).strip(),
            confidence=0.70,
        ))

    # Bilateral tolerances (attach to nearest dimension)
    for m in _RE_BILATERAL_TOL.finditer(text):
        base_val = float(m.group(1))
        tol_plus = float(m.group(2))
        tol_minus = float(m.group(3))
        # Find matching dimension and update
        for d in dims:
            if abs(d.value - base_val) < 0.01:
                d.tolerance_plus = tol_plus
                d.tolerance_minus = -tol_minus
                d.confidence = min(d.confidence + 0.1, 1.0)
                break

    return dims


def _extract_notes_from_text(text: str) -> List[ExtractedNote]:
    """Extract notes, materials, and callouts from text."""
    notes = []

    # Material
    for m in _RE_MATERIAL.finditer(text):
        notes.append(ExtractedNote(
            category="material",
            text=m.group(0).strip(),
            value=m.group(1).strip(),
            confidence=0.90,
        ))

    # Also scan for known material names
    for mat in _KNOWN_MATERIALS:
        if mat.lower() in text.lower():
            # Check we didn't already find it
            already_found = any(mat.lower() in n.value.lower() for n in notes if n.category == "material")
            if not already_found:
                notes.append(ExtractedNote(
                    category="material",
                    text=mat,
                    value=mat,
                    confidence=0.75,
                ))

    # Surface finish
    for m in _RE_SURFACE_FINISH.finditer(text):
        notes.append(ExtractedNote(
            category="surface_finish",
            text=m.group(0).strip(),
            value=f"Ra {m.group(1)}",
            confidence=0.85,
        ))

    # General tolerances
    for m in _RE_GENERAL_TOL.finditer(text):
        unit = m.group(2) if m.group(2) else "mm"
        notes.append(ExtractedNote(
            category="tolerance",
            text=m.group(0).strip(),
            value=f"±{m.group(1)} {unit}",
            confidence=0.80,
        ))

    # Hole patterns
    for m in _RE_HOLE_PATTERN.finditer(text):
        count = int(m.group(1))
        dia = float(m.group(2))
        through = m.group(3) or ""
        notes.append(ExtractedNote(
            category="hole_pattern",
            text=m.group(0).strip(),
            value=f"{count}x Ø{dia} {through}".strip(),
            confidence=0.85,
        ))

    return notes


def _infer_sector(text: str) -> str:
    """Infer sector from text content."""
    text_lower = text.lower()
    scores = {}
    for sector, keywords in _SECTOR_KEYWORDS.items():
        scores[sector] = sum(1 for kw in keywords if kw in text_lower)
    if max(scores.values(), default=0) > 0:
        return max(scores, key=scores.get)
    return ""


def _infer_dimension_labels(dims: List[ExtractedDimension]) -> None:
    """Apply heuristic labels to dimensions based on value ranges."""
    if not dims:
        return

    # Sort by value
    sorted_dims = sorted(dims, key=lambda d: d.value)

    for d in dims:
        if d.dim_type == "diameter":
            if d.value < 25:
                d.label = "hole_diameter"
            elif d.value < 100:
                d.label = "bore_diameter"
            else:
                d.label = "outer_diameter"
        elif d.dim_type == "radius":
            if d.value < 20:
                d.label = "fillet_radius"
            else:
                d.label = "feature_radius"
        elif d.dim_type == "linear":
            if d.value < 30:
                d.label = "thickness"
            elif d.value < 200:
                d.label = "width"
            else:
                d.label = "length"


# ── OCR-based extraction ───────────────────────────────────────────

def _ocr_extract(image: "Image.Image") -> Tuple[str, List[dict]]:
    """Run OCR on an image and return (full_text, word_boxes)."""
    # Get full text
    full_text = pytesseract.image_to_string(image)

    # Get word-level bounding boxes
    word_data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
    word_boxes = []
    w_img, h_img = image.size
    for i in range(len(word_data["text"])):
        txt = word_data["text"][i].strip()
        if not txt:
            continue
        conf = int(word_data["conf"][i]) if word_data["conf"][i] != "-1" else 0
        x = word_data["left"][i] / w_img
        y = word_data["top"][i] / h_img
        w = word_data["width"][i] / w_img
        h = word_data["height"][i] / h_img
        word_boxes.append({
            "text": txt,
            "confidence": conf / 100.0,
            "bbox": (x, y, w, h),
        })

    return full_text, word_boxes


def _attach_bboxes(dims: List[ExtractedDimension], word_boxes: List[dict]) -> None:
    """Try to attach bounding boxes from OCR word data to extracted dimensions."""
    for dim in dims:
        # Search for the dimension value in OCR words
        val_str = f"{dim.value:.1f}" if dim.value != int(dim.value) else f"{dim.value:.0f}"
        for wb in word_boxes:
            if val_str in wb["text"]:
                dim.bbox = wb["bbox"]
                dim.confidence = max(dim.confidence, wb["confidence"])
                break


# ── Public API ──────────────────────────────────────────────────────

def parse_drawing_image(image_data: bytes, filename: str = "") -> DrawingParseResult:
    """
    Parse a 2D engineering drawing image and extract dimensions + notes.

    Parameters
    ----------
    image_data : bytes
        Raw image bytes (PNG, JPEG, TIFF, or PDF first page).
    filename : str
        Original filename (used for format detection).

    Returns
    -------
    DrawingParseResult
        Extracted dimensions, notes, material, sector info.
    """
    result = DrawingParseResult()
    result.ocr_available = _HAS_TESSERACT and _HAS_PIL

    if not _HAS_PIL:
        result.method = "unavailable"
        result.notes.append(ExtractedNote(
            category="general",
            text="Pillow (PIL) not installed — cannot process images. Install with: pip install Pillow",
            confidence=1.0,
        ))
        return result

    # Load image
    try:
        image = Image.open(io.BytesIO(image_data))
        if image.mode != "RGB":
            image = image.convert("RGB")
    except Exception as e:
        result.method = "error"
        result.notes.append(ExtractedNote(
            category="general",
            text=f"Failed to open image: {e}",
            confidence=1.0,
        ))
        return result

    # Strategy 1: OCR if available
    if _HAS_TESSERACT:
        try:
            full_text, word_boxes = _ocr_extract(image)
            result.raw_text = full_text
            result.method = "ocr"

            # Extract dimensions and notes from OCR text
            result.dimensions = _extract_dimensions_from_text(full_text)
            result.notes = _extract_notes_from_text(full_text)

            # Attach bounding boxes
            _attach_bboxes(result.dimensions, word_boxes)

        except Exception as e:
            # Fall through to regex fallback
            result.method = "ocr_failed"
            result.notes.append(ExtractedNote(
                category="general",
                text=f"OCR failed ({e}), using regex fallback",
                confidence=1.0,
            ))

    # Strategy 2: Regex fallback — try to extract any text-like content
    if not result.dimensions and result.method != "ocr":
        result.method = "regex_fallback"
        # If we already have text from OCR failure, try regex on that
        if result.raw_text:
            result.dimensions = _extract_dimensions_from_text(result.raw_text)
            result.notes = _extract_notes_from_text(result.raw_text)

    # Infer semantic labels
    _infer_dimension_labels(result.dimensions)

    # Infer sector
    result.sector = _infer_sector(result.raw_text)

    # Extract material (first material note)
    mat_notes = [n for n in result.notes if n.category == "material"]
    if mat_notes:
        result.material = mat_notes[0].value

    return result


def parse_drawing_pdf(pdf_data: bytes) -> DrawingParseResult:
    """
    Parse a PDF engineering drawing.

    Extracts text directly from the PDF, then optionally renders pages
    to images for OCR-based dimension extraction.
    """
    result = DrawingParseResult()

    # Try text extraction first
    try:
        import pypdf
        reader = pypdf.PdfReader(io.BytesIO(pdf_data))
        text_parts = []
        for page in reader.pages:
            text_parts.append(page.extract_text() or "")
        full_text = "\n".join(text_parts)
        result.raw_text = full_text
    except ImportError:
        result.raw_text = ""
        result.notes.append(ExtractedNote(
            category="general",
            text="pypdf not installed — cannot extract PDF text. Install with: pip install pypdf",
            confidence=1.0,
        ))
    except Exception as e:
        result.raw_text = ""
        result.notes.append(ExtractedNote(
            category="general",
            text=f"PDF text extraction failed: {e}",
            confidence=1.0,
        ))

    if result.raw_text:
        result.dimensions = _extract_dimensions_from_text(result.raw_text)
        result.notes.extend(_extract_notes_from_text(result.raw_text))
        result.method = "pdf_text"

    # Try rendering PDF to image for OCR
    if _HAS_PIL and _HAS_TESSERACT:
        try:
            import pdf2image
            images = pdf2image.convert_from_bytes(pdf_data, dpi=200, first_page=1, last_page=1)
            if images:
                img_result = parse_drawing_image(_pil_to_bytes(images[0]))
                # Merge results (OCR may find dimensions that text extraction missed)
                existing_values = {d.value for d in result.dimensions}
                for d in img_result.dimensions:
                    if d.value not in existing_values:
                        result.dimensions.append(d)
                        existing_values.add(d.value)
                # Merge notes
                existing_notes = {n.text for n in result.notes}
                for n in img_result.notes:
                    if n.text not in existing_notes:
                        result.notes.append(n)
                result.method = "pdf_text+ocr"
        except ImportError:
            pass  # pdf2image not available, text-only extraction is fine
        except Exception:
            pass  # OCR enhancement failed, text extraction is sufficient

    # Infer labels and sector
    _infer_dimension_labels(result.dimensions)
    result.sector = _infer_sector(result.raw_text)

    mat_notes = [n for n in result.notes if n.category == "material"]
    if mat_notes:
        result.material = mat_notes[0].value

    return result


def _pil_to_bytes(image: "Image.Image") -> bytes:
    """Convert PIL Image to PNG bytes."""
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return buf.getvalue()


def parse_drawing(file_data: bytes, filename: str = "") -> DrawingParseResult:
    """
    Auto-detect format and parse a 2D engineering drawing.

    Supports: PNG, JPEG, TIFF, BMP, PDF.
    """
    fname_lower = filename.lower()

    if fname_lower.endswith(".pdf"):
        return parse_drawing_pdf(file_data)
    elif fname_lower.endswith((".png", ".jpg", ".jpeg", ".tiff", ".tif", ".bmp")):
        return parse_drawing_image(file_data, filename)
    else:
        # Try as image first
        try:
            return parse_drawing_image(file_data, filename)
        except Exception:
            return DrawingParseResult(
                method="unsupported",
                notes=[ExtractedNote(
                    category="general",
                    text=f"Unsupported file format: {filename}",
                    confidence=1.0,
                )],
            )

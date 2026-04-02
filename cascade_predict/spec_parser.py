"""
Spec document parser — extract component parameters from uploaded files.

Supports:
  - CSV: structured table with component/property/value columns
  - PDF: text extraction → regex-based parameter detection
  - Plain text: regex-based parameter detection
"""

from __future__ import annotations

import csv
import io
import re
from dataclasses import dataclass


@dataclass
class ParameterOverride:
    """A single parameter extracted from an uploaded document."""
    component_id: str
    property_name: str
    value: float
    unit: str = ""
    source: str = "uploaded document"


# ── Known parameter patterns for free-text / PDF extraction ──────────
# Maps regex pattern → (component_id, property_name)
_KNOWN_PARAMS: list[tuple[str, str, str]] = [
    # Aircraft
    (r"battery\s+capacity[\s:=]+([0-9.]+)\s*(kWh|kwh)?", "battery", "capacity_kwh"),
    (r"battery\s+mass[\s:=]+([0-9.]+)\s*(kg)?", "battery", "mass_kg"),
    (r"battery\s+c[\s_-]?rate[\s:=]+([0-9.]+)", "battery", "c_rate"),
    (r"windshield\s+curvature[\s:=]+([0-9.]+)\s*(1/m|m\^-1)?", "windshield", "curvature"),
    (r"cabin\s+temp(?:erature)?[\s:=]+([0-9.]+)\s*(K|°?C)?", "cabin", "temperature"),
    (r"cruise\s+(?:l/d|lift.to.drag)[\s:=]+([0-9.]+)", "wing", "cruise_ld"),
    (r"mtow[\s:=]+([0-9.]+)\s*(kg)?", "airframe", "mtow_kg"),
    (r"range[\s:=]+([0-9.]+)\s*(km|nm)?", "mission", "range_km"),
    # EV / Battery
    (r"cell\s+capacity[\s:=]+([0-9.]+)\s*(Ah|ah)?", "cell", "capacity_ah"),
    (r"cell\s+resistance[\s:=]+([0-9.]+)\s*(mOhm|mohm|ohm)?", "cell", "internal_resistance"),
    (r"pack\s+voltage[\s:=]+([0-9.]+)\s*(V|v)?", "pack", "voltage"),
    (r"coolant\s+flow[\s:=]+([0-9.]+)\s*(L/min|l/min)?", "cooling", "flow_rate"),
]


def parse_csv(file_content: bytes | str) -> list[ParameterOverride]:
    """Parse a CSV with flexible column matching."""
    if isinstance(file_content, bytes):
        file_content = file_content.decode("utf-8", errors="replace")

    reader = csv.DictReader(io.StringIO(file_content))
    if reader.fieldnames is None:
        return []

    # Normalize column names for flexible matching
    col_map = {}
    for col in reader.fieldnames:
        lower = col.strip().lower()
        if lower in ("component", "component_id", "part", "subsystem"):
            col_map["component"] = col
        elif lower in ("property", "property_name", "param", "parameter", "name"):
            col_map["property"] = col
        elif lower in ("value", "val", "amount"):
            col_map["value"] = col
        elif lower in ("unit", "units", "uom"):
            col_map["unit"] = col

    if not all(k in col_map for k in ("component", "property", "value")):
        return []

    overrides = []
    for row in reader:
        try:
            comp = row[col_map["component"]].strip()
            prop = row[col_map["property"]].strip()
            val = float(row[col_map["value"]].strip())
            unit = row.get(col_map.get("unit", ""), "").strip() if "unit" in col_map else ""
            overrides.append(ParameterOverride(
                component_id=comp, property_name=prop, value=val, unit=unit,
                source="CSV upload",
            ))
        except (ValueError, KeyError):
            continue

    return overrides


def parse_text(text: str) -> list[ParameterOverride]:
    """Extract parameters from free-form text using regex patterns."""
    overrides = []
    for pattern, comp_id, prop_name in _KNOWN_PARAMS:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            try:
                val = float(match.group(1))
                unit = match.group(2) if match.lastindex >= 2 and match.group(2) else ""
                overrides.append(ParameterOverride(
                    component_id=comp_id, property_name=prop_name,
                    value=val, unit=unit or "", source="text extraction",
                ))
            except (ValueError, IndexError):
                continue
    return overrides


def parse_pdf(file_content: bytes) -> list[ParameterOverride]:
    """Extract text from PDF then parse for parameters."""
    text = extract_pdf_text(file_content)
    return parse_text(text)


def extract_pdf_text(file_content: bytes) -> str:
    """Raw text extraction from PDF bytes."""
    try:
        from pypdf import PdfReader
        reader = PdfReader(io.BytesIO(file_content))
        pages = []
        for page in reader.pages:
            pages.append(page.extract_text() or "")
        return "\n".join(pages)
    except ImportError:
        return "[pypdf not installed — install with: pip install pypdf]"
    except Exception as e:
        return f"[PDF read error: {e}]"

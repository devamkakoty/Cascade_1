"""
Spec document parser — extract component parameters from uploaded files.

Supports:
  - CSV: structured component specs (component, property, value, unit)
  - PDF: text extraction with regex-based parameter detection
  - Plain text: same regex extraction as PDF
  - Onshape URL: stores link for CAD viewer embedding

The parser returns a list of ParameterOverride objects that can be
applied to any template's component properties before cascade analysis.
"""

from __future__ import annotations
import csv
import io
import re
from dataclasses import dataclass


@dataclass
class ParameterOverride:
    """A single parameter value extracted from an uploaded document."""

    component_id: str
    property_name: str
    value: float
    unit: str = ""
    source: str = "uploaded document"
    confidence: str = "high"  # high (CSV exact), medium (PDF regex), low (guessed)


def parse_csv(content: str | bytes) -> list[ParameterOverride]:
    """Parse a CSV file with columns: component, property, value, unit.

    Accepts flexible column names (case-insensitive, partial match).
    """
    if isinstance(content, bytes):
        content = content.decode("utf-8", errors="replace")

    overrides = []
    reader = csv.DictReader(io.StringIO(content))

    if reader.fieldnames is None:
        return overrides

    # Flexible column mapping
    col_map = {}
    for col in reader.fieldnames:
        cl = col.strip().lower()
        if "component" in cl or "part" in cl:
            col_map["component"] = col
        elif "property" in cl or "param" in cl or "name" in cl:
            col_map["property"] = col
        elif "value" in cl or "val" in cl:
            col_map["value"] = col
        elif "unit" in cl:
            col_map["unit"] = col

    if "component" not in col_map or "value" not in col_map:
        return overrides

    for row in reader:
        try:
            comp = row[col_map["component"]].strip()
            prop = row.get(col_map.get("property", ""), comp).strip()
            val = float(row[col_map["value"]].strip())
            unit = row.get(col_map.get("unit", ""), "").strip()
            if comp and prop:
                overrides.append(ParameterOverride(
                    component_id=comp,
                    property_name=prop,
                    value=val,
                    unit=unit,
                    source="CSV upload",
                    confidence="high",
                ))
        except (ValueError, KeyError):
            continue

    return overrides


def parse_text(content: str | bytes) -> list[ParameterOverride]:
    """Extract parameter values from plain text or PDF text.

    Uses regex patterns to find lines like:
      - "thermal_conductivity: 1.4 W/(m·K)"
      - "mass = 12.0 kg"
      - "curvature 0.25 1/m"
      - "MTOW: 6350 kg"
    """
    if isinstance(content, bytes):
        content = content.decode("utf-8", errors="replace")

    overrides = []

    # Known parameter patterns (property_name → component mapping)
    _KNOWN_PARAMS = {
        "thermal_conductivity": "windshield",
        "solar_transmittance": "windshield",
        "curvature": "windshield",
        "windshield_mass": "windshield",
        "cooling_capacity": "hvac_unit",
        "cop": "hvac_unit",
        "total_capacity": "battery_pack",
        "specific_energy": "battery_pack",
        "battery_mass": "battery_pack",
        "structural_mass": "wing",
        "wing_area": "wing",
        "insulation_rvalue": "fuselage",
        "rated_power": "propulsion_motors",
        # EV battery
        "nominal_capacity": "battery_cell",
        "internal_resistance": "battery_cell",
        "thermal_runaway_onset": "battery_cell",
        "interface_resistance": "cooling_system",
        "chiller_capacity": "cooling_system",
    }

    # Regex: "property_name" followed by separator then number and optional unit
    pattern = re.compile(
        r'(?P<name>[a-z][a-z0-9_]*(?:\s+[a-z]+)*)'
        r'\s*[:=]\s*'
        r'(?P<value>[-+]?\d+\.?\d*(?:[eE][-+]?\d+)?)'
        r'\s*(?P<unit>[^\n,;]*)',
        re.IGNORECASE
    )

    for match in pattern.finditer(content):
        name_raw = match.group("name").strip().lower().replace(" ", "_")
        try:
            val = float(match.group("value"))
        except ValueError:
            continue
        unit = match.group("unit").strip().rstrip(".,;")

        # Try to match to a known parameter
        comp_id = _KNOWN_PARAMS.get(name_raw, "")
        if not comp_id:
            # Fuzzy match: check if any known param is a substring
            for known, comp in _KNOWN_PARAMS.items():
                if known in name_raw or name_raw in known:
                    comp_id = comp
                    name_raw = known
                    break

        if comp_id:
            overrides.append(ParameterOverride(
                component_id=comp_id,
                property_name=name_raw,
                value=val,
                unit=unit,
                source="text extraction",
                confidence="medium",
            ))

    return overrides


def parse_pdf(file_bytes: bytes) -> list[ParameterOverride]:
    """Extract text from a PDF and parse parameters."""
    try:
        from pypdf import PdfReader
        reader = PdfReader(io.BytesIO(file_bytes))
        text = "\n".join(page.extract_text() or "" for page in reader.pages)
        return parse_text(text)
    except ImportError:
        return []
    except Exception:
        return []


def extract_pdf_text(file_bytes: bytes) -> str:
    """Extract raw text from PDF for display."""
    try:
        from pypdf import PdfReader
        reader = PdfReader(io.BytesIO(file_bytes))
        return "\n\n".join(page.extract_text() or "" for page in reader.pages)
    except ImportError:
        return "(pypdf not installed — run: pip install pypdf)"
    except Exception as e:
        return f"(PDF extraction failed: {e})"

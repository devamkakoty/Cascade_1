"""
Sample CAD/drawing library for quick testing.

Provides pre-built sample files that users can select instead of
uploading their own, enabling immediate testing of the cascade pipeline.
"""

from __future__ import annotations

import struct
import math
import random
from dataclasses import dataclass
from typing import List, Tuple


@dataclass
class SampleFile:
    """A sample file for the library."""
    name: str
    description: str
    sector: str
    file_type: str        # "stl", "drawing_png"
    data: bytes = b""     # raw file bytes (generated on demand)


def _generate_box_stl(lx: float, ly: float, lz: float) -> bytes:
    """Generate a binary STL for a rectangular box."""
    # 12 triangles for a box
    triangles = []
    # Bottom face (z=0)
    triangles.append(((0, 0, -1), (0, 0, 0), (lx, 0, 0), (lx, ly, 0)))
    triangles.append(((0, 0, -1), (0, 0, 0), (lx, ly, 0), (0, ly, 0)))
    # Top face (z=lz)
    triangles.append(((0, 0, 1), (0, 0, lz), (lx, ly, lz), (lx, 0, lz)))
    triangles.append(((0, 0, 1), (0, 0, lz), (0, ly, lz), (lx, ly, lz)))
    # Front face (y=0)
    triangles.append(((0, -1, 0), (0, 0, 0), (lx, 0, lz), (lx, 0, 0)))
    triangles.append(((0, -1, 0), (0, 0, 0), (0, 0, lz), (lx, 0, lz)))
    # Back face (y=ly)
    triangles.append(((0, 1, 0), (0, ly, 0), (lx, ly, 0), (lx, ly, lz)))
    triangles.append(((0, 1, 0), (0, ly, 0), (lx, ly, lz), (0, ly, lz)))
    # Left face (x=0)
    triangles.append(((-1, 0, 0), (0, 0, 0), (0, ly, 0), (0, ly, lz)))
    triangles.append(((-1, 0, 0), (0, 0, 0), (0, ly, lz), (0, 0, lz)))
    # Right face (x=lx)
    triangles.append(((1, 0, 0), (lx, 0, 0), (lx, 0, lz), (lx, ly, lz)))
    triangles.append(((1, 0, 0), (lx, 0, 0), (lx, ly, lz), (lx, ly, 0)))

    # Binary STL format
    header = b'\x00' * 80
    n_tris = len(triangles)
    data = header + struct.pack('<I', n_tris)
    for normal, v1, v2, v3 in triangles:
        data += struct.pack('<fff', *normal)
        data += struct.pack('<fff', *v1)
        data += struct.pack('<fff', *v2)
        data += struct.pack('<fff', *v3)
        data += struct.pack('<H', 0)  # attribute byte count

    return data


def _generate_cylinder_stl(radius: float, height: float, segments: int = 24) -> bytes:
    """Generate a binary STL for a cylinder."""
    triangles = []
    for i in range(segments):
        a1 = 2 * math.pi * i / segments
        a2 = 2 * math.pi * (i + 1) / segments
        x1, y1 = radius * math.cos(a1), radius * math.sin(a1)
        x2, y2 = radius * math.cos(a2), radius * math.sin(a2)

        # Bottom cap
        triangles.append(((0, 0, -1), (0, 0, 0), (x2, y2, 0), (x1, y1, 0)))
        # Top cap
        triangles.append(((0, 0, 1), (0, 0, height), (x1, y1, height), (x2, y2, height)))
        # Side (two triangles per segment)
        nx = math.cos((a1 + a2) / 2)
        ny = math.sin((a1 + a2) / 2)
        triangles.append(((nx, ny, 0), (x1, y1, 0), (x2, y2, 0), (x2, y2, height)))
        triangles.append(((nx, ny, 0), (x1, y1, 0), (x2, y2, height), (x1, y1, height)))

    header = b'\x00' * 80
    n_tris = len(triangles)
    data = header + struct.pack('<I', n_tris)
    for normal, v1, v2, v3 in triangles:
        data += struct.pack('<fff', *normal)
        data += struct.pack('<fff', *v1)
        data += struct.pack('<fff', *v2)
        data += struct.pack('<fff', *v3)
        data += struct.pack('<H', 0)

    return data


def _generate_l_bracket_stl(leg_a: float, leg_b: float, thickness: float, width: float) -> bytes:
    """Generate binary STL for an L-bracket (extruded L profile)."""
    # L-profile cross section, extruded along width
    # Points of the L (in XZ plane, extruded along Y)
    pts_2d = [
        (0, 0),
        (leg_a, 0),
        (leg_a, thickness),
        (thickness, thickness),
        (thickness, leg_b),
        (0, leg_b),
    ]

    triangles = []
    n = len(pts_2d)

    # Front face (y=0) — triangulate as fan from first point
    for i in range(1, n - 1):
        nx, nz = 0, 0
        triangles.append(
            ((0, -1, 0),
             (pts_2d[0][0], 0, pts_2d[0][1]),
             (pts_2d[i][0], 0, pts_2d[i][1]),
             (pts_2d[i+1][0], 0, pts_2d[i+1][1]))
        )

    # Back face (y=width)
    for i in range(1, n - 1):
        triangles.append(
            ((0, 1, 0),
             (pts_2d[0][0], width, pts_2d[0][1]),
             (pts_2d[i+1][0], width, pts_2d[i+1][1]),
             (pts_2d[i][0], width, pts_2d[i][1]))
        )

    # Side faces (extrude each edge)
    for i in range(n):
        j = (i + 1) % n
        x1, z1 = pts_2d[i]
        x2, z2 = pts_2d[j]
        dx, dz = x2 - x1, z2 - z1
        length = math.sqrt(dx*dx + dz*dz)
        if length < 1e-6:
            continue
        # Outward normal
        nx_n, nz_n = dz / length, -dx / length
        triangles.append(
            ((nx_n, 0, nz_n),
             (x1, 0, z1), (x2, 0, z2), (x2, width, z2))
        )
        triangles.append(
            ((nx_n, 0, nz_n),
             (x1, 0, z1), (x2, width, z2), (x1, width, z1))
        )

    header = b'\x00' * 80
    n_tris = len(triangles)
    data = header + struct.pack('<I', n_tris)
    for normal, v1, v2, v3 in triangles:
        data += struct.pack('<fff', *normal)
        data += struct.pack('<fff', *v1)
        data += struct.pack('<fff', *v2)
        data += struct.pack('<fff', *v3)
        data += struct.pack('<H', 0)

    return data


# ── Sample Library ──────────────────────────────────────────────────

SAMPLE_LIBRARY: List[SampleFile] = []


def _init_library():
    """Build the sample library (called once)."""
    if SAMPLE_LIBRARY:
        return

    SAMPLE_LIBRARY.extend([
        # Naval
        SampleFile(
            name="Hull Plate (12mm steel, 3m×1.5m)",
            description="Flat hull plate — typical midship panel. AH36 marine steel, 12mm thick.",
            sector="naval",
            file_type="stl",
            data=_generate_box_stl(3000, 1500, 12),  # 3m × 1.5m × 12mm
        ),
        SampleFile(
            name="Deck Stiffener Bracket",
            description="L-shaped stiffener bracket for deck attachment. Steel, 150×100×8mm.",
            sector="naval",
            file_type="stl",
            data=_generate_l_bracket_stl(150, 100, 8, 80),
        ),

        # Aerospace
        SampleFile(
            name="Fuselage Skin Panel (2mm AL, 1.2m×0.6m)",
            description="Curved fuselage skin panel. AL 2024-T3, 2mm thick.",
            sector="aerospace",
            file_type="stl",
            data=_generate_box_stl(1200, 600, 2),
        ),
        SampleFile(
            name="Wing Spar Section (AL 7075)",
            description="I-beam wing spar section, 800mm long.",
            sector="aerospace",
            file_type="stl",
            data=_generate_box_stl(800, 60, 120),
        ),

        # Automotive / EV
        SampleFile(
            name="Battery Cell (21700 cylindrical)",
            description="Cylindrical Li-ion cell, 21mm diameter × 70mm height.",
            sector="automotive_ev",
            file_type="stl",
            data=_generate_cylinder_stl(10.5, 70, 24),
        ),
        SampleFile(
            name="Cooling Plate (3mm AL, 300×200mm)",
            description="Battery pack cold plate. AL 6061-T6, 3mm thick.",
            sector="automotive_ev",
            file_type="stl",
            data=_generate_box_stl(300, 200, 3),
        ),

        # Robotics
        SampleFile(
            name="Robot Arm Link (AL tube, 400mm)",
            description="Hollow aluminum tube link, 40mm OD × 3mm wall × 400mm long.",
            sector="robotics",
            file_type="stl",
            data=_generate_cylinder_stl(20, 400, 24),
        ),
        SampleFile(
            name="Gripper Bracket (AL 6061)",
            description="L-bracket for gripper mounting, 60×40×5mm.",
            sector="robotics",
            file_type="stl",
            data=_generate_l_bracket_stl(60, 40, 5, 30),
        ),
        SampleFile(
            name="Motor Housing (cylinder, Ø80×50mm)",
            description="Servo motor housing, steel cylinder.",
            sector="robotics",
            file_type="stl",
            data=_generate_cylinder_stl(40, 50, 24),
        ),
    ])


def get_sample_library() -> List[SampleFile]:
    """Get the sample library (initializes on first call)."""
    _init_library()
    return SAMPLE_LIBRARY


def get_sample_by_name(name: str) -> SampleFile:
    """Get a sample file by name."""
    _init_library()
    for s in SAMPLE_LIBRARY:
        if s.name == name:
            return s
    raise KeyError(f"Sample not found: {name}")

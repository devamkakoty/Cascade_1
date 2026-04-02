"""
CAD file parser — extract geometric parameters from STL and STEP files.

Supports:
  - STL (ASCII & binary): mesh analysis → bounding box, volume, surface area,
    curvature estimate, wall thickness estimate
  - STEP / STP: text-based extraction of named shapes, radii, lengths,
    plus fallback triangulation for geometry metrics
"""

from __future__ import annotations

import io
import re
import struct
import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class GeometryParameter:
    """A single geometric measurement extracted from a CAD file."""
    name: str
    value: float
    unit: str = "mm"
    description: str = ""
    category: str = "geometry"  # geometry, curvature, mass, structural


@dataclass
class CADAnalysisResult:
    """Full analysis result from a CAD file."""
    filename: str
    file_type: str  # "stl" or "step"
    parameters: list[GeometryParameter] = field(default_factory=list)
    vertices: Optional[np.ndarray] = None  # Nx3 for 3D viewer
    faces: Optional[np.ndarray] = None     # Mx3 triangle indices
    normals: Optional[np.ndarray] = None   # Mx3 face normals
    raw_stl_bytes: Optional[bytes] = None  # for JS viewer

    def param_dict(self) -> dict[str, float]:
        """Return {name: value} for easy lookup."""
        return {p.name: p.value for p in self.parameters}


# ── STL Parsing ─────────────────────────────────────────────────────

def _parse_stl_ascii(text: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Parse ASCII STL, return (vertices Nx3, faces Mx3, normals Mx3)."""
    vertices = []
    normals = []
    faces = []
    current_normal = [0.0, 0.0, 0.0]
    tri_verts = []

    for line in text.splitlines():
        line = line.strip().lower()
        if line.startswith("facet normal"):
            parts = line.split()
            current_normal = [float(parts[2]), float(parts[3]), float(parts[4])]
        elif line.startswith("vertex"):
            parts = line.split()
            v = [float(parts[1]), float(parts[2]), float(parts[3])]
            tri_verts.append(len(vertices))
            vertices.append(v)
        elif line.startswith("endfacet"):
            if len(tri_verts) == 3:
                faces.append(tri_verts)
                normals.append(current_normal)
            tri_verts = []

    return (
        np.array(vertices, dtype=np.float32) if vertices else np.zeros((0, 3), dtype=np.float32),
        np.array(faces, dtype=np.int32) if faces else np.zeros((0, 3), dtype=np.int32),
        np.array(normals, dtype=np.float32) if normals else np.zeros((0, 3), dtype=np.float32),
    )


def _parse_stl_binary(data: bytes) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Parse binary STL, return (vertices, faces, normals)."""
    # Skip 80-byte header
    n_triangles = struct.unpack_from("<I", data, 80)[0]
    vertices = []
    normals = []
    faces = []
    offset = 84

    for i in range(n_triangles):
        nx, ny, nz = struct.unpack_from("<3f", data, offset)
        offset += 12
        tri_verts = []
        for _ in range(3):
            x, y, z = struct.unpack_from("<3f", data, offset)
            offset += 12
            tri_verts.append(len(vertices))
            vertices.append([x, y, z])
        normals.append([nx, ny, nz])
        faces.append(tri_verts)
        offset += 2  # attribute byte count

    return (
        np.array(vertices, dtype=np.float32) if vertices else np.zeros((0, 3), dtype=np.float32),
        np.array(faces, dtype=np.int32) if faces else np.zeros((0, 3), dtype=np.int32),
        np.array(normals, dtype=np.float32) if normals else np.zeros((0, 3), dtype=np.float32),
    )


def parse_stl(data: bytes) -> CADAnalysisResult:
    """Parse an STL file (auto-detects ASCII vs binary) and extract geometry."""
    # Detect ASCII vs binary
    try:
        text = data.decode("ascii", errors="strict")
        if text.strip().lower().startswith("solid") and "facet" in text.lower():
            vertices, faces, normals = _parse_stl_ascii(text)
        else:
            vertices, faces, normals = _parse_stl_binary(data)
    except (UnicodeDecodeError, ValueError):
        vertices, faces, normals = _parse_stl_binary(data)

    params = _analyze_mesh(vertices, faces, normals)

    return CADAnalysisResult(
        filename="",
        file_type="stl",
        parameters=params,
        vertices=vertices,
        faces=faces,
        normals=normals,
        raw_stl_bytes=data,
    )


# ── STEP Parsing ────────────────────────────────────────────────────

def parse_step(data: bytes) -> CADAnalysisResult:
    """Parse a STEP/STP file and extract geometric parameters."""
    text = data.decode("utf-8", errors="replace")
    params = []

    # Extract named shapes
    shape_names = re.findall(r"PRODUCT\s*\(\s*'([^']+)'", text, re.IGNORECASE)

    # Extract circle/cylinder radii
    radii = []
    for m in re.finditer(r"CIRCLE\s*\([^,]*,\s*[^,]*,\s*([0-9.eE+-]+)\s*\)", text):
        try:
            radii.append(float(m.group(1)))
        except ValueError:
            pass
    for m in re.finditer(r"CYLINDRICAL_SURFACE\s*\([^,]*,\s*[^,]*,\s*([0-9.eE+-]+)\s*\)", text):
        try:
            radii.append(float(m.group(1)))
        except ValueError:
            pass

    # Extract cartesian points for bounding box
    points = []
    for m in re.finditer(
        r"CARTESIAN_POINT\s*\(\s*'[^']*'\s*,\s*\(\s*([0-9.eE+-]+)\s*,\s*([0-9.eE+-]+)\s*,\s*([0-9.eE+-]+)\s*\)\s*\)",
        text,
    ):
        try:
            points.append([float(m.group(1)), float(m.group(2)), float(m.group(3))])
        except ValueError:
            pass

    # Extract axis placements for orientation info
    directions = []
    for m in re.finditer(
        r"DIRECTION\s*\(\s*'[^']*'\s*,\s*\(\s*([0-9.eE+-]+)\s*,\s*([0-9.eE+-]+)\s*,\s*([0-9.eE+-]+)\s*\)\s*\)",
        text,
    ):
        try:
            directions.append([float(m.group(1)), float(m.group(2)), float(m.group(3))])
        except ValueError:
            pass

    # Compute bounding box from points
    if points:
        pts = np.array(points, dtype=np.float64)
        bb_min = pts.min(axis=0)
        bb_max = pts.max(axis=0)
        dims = bb_max - bb_min
        sorted_dims = sorted(dims, reverse=True)

        params.append(GeometryParameter(
            "length", sorted_dims[0], "mm", "Largest bounding box dimension"))
        params.append(GeometryParameter(
            "width", sorted_dims[1], "mm", "Middle bounding box dimension"))
        params.append(GeometryParameter(
            "height", sorted_dims[2], "mm", "Smallest bounding box dimension"))
        params.append(GeometryParameter(
            "bounding_volume", float(np.prod(dims)), "mm^3", "Bounding box volume"))
        params.append(GeometryParameter(
            "aspect_ratio", sorted_dims[0] / max(sorted_dims[2], 1e-6), "",
            "Length-to-height ratio"))

        # Estimate surface area as bounding box surface
        sa = 2 * (dims[0]*dims[1] + dims[1]*dims[2] + dims[0]*dims[2])
        params.append(GeometryParameter(
            "est_surface_area", float(sa), "mm^2", "Estimated surface area (bounding box)"))

    # Add radii
    if radii:
        params.append(GeometryParameter(
            "min_radius", min(radii), "mm", "Smallest circle/cylinder radius"))
        params.append(GeometryParameter(
            "max_radius", max(radii), "mm", "Largest circle/cylinder radius"))
        params.append(GeometryParameter(
            "mean_radius", sum(radii) / len(radii), "mm", "Average radius"))
        if max(radii) > 0:
            params.append(GeometryParameter(
                "max_curvature", 1.0 / min(r for r in radii if r > 0), "1/mm",
                "Maximum curvature (1/min_radius)", category="curvature"))

    # Shape count
    params.append(GeometryParameter(
        "n_shapes", len(shape_names), "", "Number of named product shapes"))
    params.append(GeometryParameter(
        "n_points", len(points), "", "Number of cartesian points in file"))

    # Generate a simple point cloud for viewer (no triangulation)
    vertices = np.array(points, dtype=np.float32) if points else np.zeros((0, 3), dtype=np.float32)

    return CADAnalysisResult(
        filename="",
        file_type="step",
        parameters=params,
        vertices=vertices,
        faces=None,
        normals=None,
    )


# ── Mesh Analysis (shared) ──────────────────────────────────────────

def _analyze_mesh(
    vertices: np.ndarray, faces: np.ndarray, normals: np.ndarray,
) -> list[GeometryParameter]:
    """Extract geometric parameters from a triangle mesh."""
    params = []
    if len(vertices) == 0 or len(faces) == 0:
        return params

    # Bounding box
    bb_min = vertices.min(axis=0)
    bb_max = vertices.max(axis=0)
    dims = bb_max - bb_min
    sorted_dims = sorted(dims, reverse=True)

    params.append(GeometryParameter(
        "length", float(sorted_dims[0]), "mm", "Largest bounding box dimension"))
    params.append(GeometryParameter(
        "width", float(sorted_dims[1]), "mm", "Middle bounding box dimension"))
    params.append(GeometryParameter(
        "height", float(sorted_dims[2]), "mm", "Smallest bounding box dimension"))

    # Aspect ratio
    if sorted_dims[2] > 1e-6:
        params.append(GeometryParameter(
            "aspect_ratio", float(sorted_dims[0] / sorted_dims[2]), "",
            "Length-to-height ratio"))

    # Surface area (sum of triangle areas)
    total_area = 0.0
    areas = []
    for face in faces:
        v0, v1, v2 = vertices[face[0]], vertices[face[1]], vertices[face[2]]
        area = 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0))
        areas.append(area)
        total_area += area
    params.append(GeometryParameter(
        "surface_area", float(total_area), "mm^2", "Total mesh surface area"))

    # Volume (signed volume method for closed meshes)
    volume = 0.0
    for face in faces:
        v0, v1, v2 = vertices[face[0]], vertices[face[1]], vertices[face[2]]
        volume += np.dot(v0, np.cross(v1, v2)) / 6.0
    volume = abs(volume)
    params.append(GeometryParameter(
        "volume", float(volume), "mm^3", "Estimated mesh volume (assumes watertight)"))

    # Centroid
    centroid = vertices.mean(axis=0)
    params.append(GeometryParameter(
        "centroid_x", float(centroid[0]), "mm", "Centroid X"))
    params.append(GeometryParameter(
        "centroid_y", float(centroid[1]), "mm", "Centroid Y"))
    params.append(GeometryParameter(
        "centroid_z", float(centroid[2]), "mm", "Centroid Z"))

    # Curvature estimation from normal variation
    if len(normals) > 1:
        # Average angle between adjacent face normals
        n_samples = min(500, len(normals))
        rng = np.random.RandomState(0)
        indices = rng.choice(len(normals), n_samples, replace=False) if len(normals) > n_samples else np.arange(len(normals))
        angle_sum = 0.0
        count = 0
        for i in range(len(indices) - 1):
            n1 = normals[indices[i]]
            n2 = normals[indices[i + 1]]
            mag1 = np.linalg.norm(n1)
            mag2 = np.linalg.norm(n2)
            if mag1 > 1e-8 and mag2 > 1e-8:
                cos_angle = np.clip(np.dot(n1, n2) / (mag1 * mag2), -1.0, 1.0)
                angle_sum += math.acos(cos_angle)
                count += 1
        if count > 0:
            mean_angle = angle_sum / count
            params.append(GeometryParameter(
                "mean_normal_deviation", float(math.degrees(mean_angle)), "deg",
                "Mean angle between adjacent face normals", category="curvature"))
            # Approximate curvature: angle / average edge length
            avg_area = total_area / max(len(faces), 1)
            avg_edge = math.sqrt(avg_area) if avg_area > 0 else 1.0
            curvature_est = mean_angle / max(avg_edge, 1e-6)
            params.append(GeometryParameter(
                "est_curvature", float(curvature_est), "1/mm",
                "Estimated mean curvature from normal variation", category="curvature"))

    # Wall thickness estimate (ray-casting approximation)
    if volume > 0 and total_area > 0:
        thickness_est = volume / (total_area / 2.0)  # V ≈ SA/2 * t for thin shells
        params.append(GeometryParameter(
            "est_wall_thickness", float(thickness_est), "mm",
            "Estimated wall thickness (V / half-SA)", category="structural"))

    # Triangle count
    params.append(GeometryParameter(
        "n_triangles", float(len(faces)), "", "Number of mesh triangles"))
    params.append(GeometryParameter(
        "n_vertices", float(len(vertices)), "", "Number of mesh vertices"))

    return params


# ── Public API ──────────────────────────────────────────────────────

def parse_cad_file(filename: str, data: bytes) -> CADAnalysisResult:
    """Auto-detect file type and parse."""
    lower = filename.lower()
    if lower.endswith(".stl"):
        result = parse_stl(data)
    elif lower.endswith((".step", ".stp")):
        result = parse_step(data)
    else:
        # Try STL first, fall back to STEP
        try:
            result = parse_stl(data)
            if len(result.parameters) == 0:
                result = parse_step(data)
        except Exception:
            result = parse_step(data)

    result.filename = filename
    return result

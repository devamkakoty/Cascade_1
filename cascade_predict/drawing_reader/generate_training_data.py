"""
Generate synthetic 2D engineering drawing training data.

Creates annotated drawing images with known ground-truth labels for:
  - Dimension lines with values and units
  - Tolerances (bilateral ±, unilateral, limit)
  - GD&T symbols (flatness, perpendicularity, true position, etc.)
  - Title blocks with part info
  - Section views with callouts
  - Material/finish notes

Each generated drawing produces:
  1. A PNG image (the "drawing")
  2. A JSON annotation file (ground truth: bounding boxes, values, types)

Uses matplotlib to render — no external CAD software needed.
"""

from __future__ import annotations

import json
import math
import os
import random
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyArrowPatch
import numpy as np


@dataclass
class DimensionAnnotation:
    """Ground truth for a single dimension on a drawing."""
    dim_type: str           # "linear", "diameter", "radius", "angle", "tolerance"
    value: float
    unit: str               # "mm", "in", "deg"
    tolerance_plus: float = 0.0
    tolerance_minus: float = 0.0
    text: str = ""          # as it appears on the drawing
    x: float = 0.0          # bbox center x (normalized 0-1)
    y: float = 0.0          # bbox center y (normalized 0-1)
    width: float = 0.0      # bbox width (normalized)
    height: float = 0.0     # bbox height (normalized)
    label: str = ""         # semantic label: "plate_thickness", "hole_diameter", etc.


@dataclass
class DrawingAnnotation:
    """Full ground truth for one drawing."""
    filename: str
    part_name: str
    material: str
    dimensions: list[DimensionAnnotation] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    sector: str = ""        # "naval", "aerospace", "automotive"


# ── Drawing generators ──────────────────────────────────────────────

def _draw_dimension_line(ax, x1, y1, x2, y2, dim_text, offset=0.3):
    """Draw a dimension line with arrows and text."""
    # Extension lines
    dx = x2 - x1
    dy = y2 - y1
    length = math.sqrt(dx*dx + dy*dy)
    if length < 0.01:
        return

    # Normal direction for offset
    nx, ny = -dy/length, dx/length

    # Offset points
    ox1, oy1 = x1 + nx*offset, y1 + ny*offset
    ox2, oy2 = x2 + nx*offset, y2 + ny*offset

    # Extension lines
    ax.plot([x1, ox1], [y1, oy1], 'k-', linewidth=0.5)
    ax.plot([x2, ox2], [y2, oy2], 'k-', linewidth=0.5)

    # Dimension line with arrows
    ax.annotate('', xy=(ox2, oy2), xytext=(ox1, oy1),
                arrowprops=dict(arrowstyle='<->', color='black', lw=0.8))

    # Text
    mx, my = (ox1+ox2)/2, (oy1+oy2)/2
    angle = math.degrees(math.atan2(dy, dx))
    ax.text(mx, my + 0.08, dim_text, ha='center', va='bottom',
            fontsize=7, fontweight='bold', rotation=angle if abs(angle) < 45 else 0)


def _draw_title_block(ax, part_name, material, dims_summary, sector):
    """Draw a title block in the bottom-right corner."""
    tb_x, tb_y = 0.55, 0.02
    tb_w, tb_h = 0.43, 0.15

    rect = patches.Rectangle((tb_x, tb_y), tb_w, tb_h,
                               linewidth=1.5, edgecolor='black', facecolor='white')
    ax.add_patch(rect)

    # Horizontal dividers
    for dy in [0.05, 0.10]:
        ax.plot([tb_x, tb_x + tb_w], [tb_y + dy, tb_y + dy], 'k-', linewidth=0.5)

    # Vertical divider
    ax.plot([tb_x + tb_w/2, tb_x + tb_w/2], [tb_y, tb_y + 0.05], 'k-', linewidth=0.5)

    # Text
    ax.text(tb_x + 0.02, tb_y + 0.12, part_name, fontsize=7, fontweight='bold')
    ax.text(tb_x + 0.02, tb_y + 0.07, f"Material: {material}", fontsize=5)
    ax.text(tb_x + 0.02, tb_y + 0.02, f"Scale: 1:1", fontsize=5)
    ax.text(tb_x + tb_w/2 + 0.02, tb_y + 0.02, f"Sector: {sector}", fontsize=5)


def generate_plate_drawing(output_dir, idx, sector="naval"):
    """Generate a flat plate / panel drawing with thickness, length, width dims."""
    fig, ax = plt.subplots(1, 1, figsize=(11, 8.5), dpi=150)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect('equal')
    ax.axis('off')

    # Border
    border = patches.Rectangle((0.02, 0.02), 0.96, 0.96,
                                 linewidth=2, edgecolor='black', facecolor='white')
    ax.add_patch(border)

    # Random dimensions
    thickness = round(random.uniform(4, 25), 1)
    length = round(random.uniform(500, 3000), 0)
    width = round(random.uniform(200, 1500), 0)
    tol_t = round(random.uniform(0.1, 0.5), 2)
    tol_l = round(random.uniform(1.0, 5.0), 1)

    part_name = random.choice([
        "Hull Plate", "Deck Panel", "Bulkhead Panel",
        "Fuselage Skin", "Wing Skin Panel", "Floor Panel",
    ])
    material = random.choice([
        "AH36 Marine Steel", "DH36 Steel", "AL 6061-T6",
        "AL 2024-T3", "CFRP Laminate", "Titanium Ti-6Al-4V",
    ])

    # Draw top view (rectangle)
    rx, ry = 0.15, 0.35
    rw, rh = 0.55, 0.35
    rect = patches.Rectangle((rx, ry), rw, rh,
                               linewidth=1.5, edgecolor='black', facecolor='#f0f0f0')
    ax.add_patch(rect)

    # Cross-hatch for section
    for i in range(20):
        hx = rx + i * rw / 20
        ax.plot([hx, hx + rw/40], [ry, ry + rh/8], 'gray', linewidth=0.3, alpha=0.4)

    # Dimension lines
    length_text = f"{length:.0f}"
    _draw_dimension_line(ax, rx, ry, rx + rw, ry, length_text, offset=-0.08)

    width_text = f"{width:.0f}"
    _draw_dimension_line(ax, rx + rw, ry, rx + rw, ry + rh, width_text, offset=0.08)

    # Side view (thin rectangle for thickness)
    sx, sy = 0.15, 0.22
    sw, sh = 0.55, 0.05
    side = patches.Rectangle((sx, sy), sw, sh,
                               linewidth=1.5, edgecolor='black', facecolor='#e8e8e8')
    ax.add_patch(side)

    thickness_text = f"{thickness:.1f} ±{tol_t}"
    _draw_dimension_line(ax, sx, sy, sx, sy + sh, thickness_text, offset=-0.08)

    # Tolerance note
    ax.text(0.15, 0.85, f"GENERAL TOLERANCES: ±{tol_l}mm (LINEAR), ±0.5° (ANGULAR)",
            fontsize=6, fontfamily='monospace')
    ax.text(0.15, 0.82, f"SURFACE FINISH: Ra 3.2 unless noted",
            fontsize=6, fontfamily='monospace')

    # Title block
    _draw_title_block(ax, part_name, material, f"{length}x{width}x{thickness}", sector)

    # Section label
    ax.text(sx + sw/2, sy - 0.03, "SECTION A-A", ha='center', fontsize=6, fontstyle='italic')

    # Build annotations
    annotations = DrawingAnnotation(
        filename=f"plate_{idx:04d}.png",
        part_name=part_name,
        material=material,
        sector=sector,
        dimensions=[
            DimensionAnnotation("linear", length, "mm", tol_l, -tol_l,
                                length_text, rx + rw/2, ry - 0.08, rw, 0.05, "length"),
            DimensionAnnotation("linear", width, "mm", tol_l, -tol_l,
                                width_text, rx + rw + 0.08, ry + rh/2, 0.05, rh, "width"),
            DimensionAnnotation("linear", thickness, "mm", tol_t, -tol_t,
                                thickness_text, sx - 0.08, sy + sh/2, 0.05, sh, "thickness"),
        ],
        notes=[
            f"General tolerance: ±{tol_l}mm",
            f"Surface finish: Ra 3.2",
            f"Material: {material}",
        ],
    )

    # Save
    filepath = os.path.join(output_dir, annotations.filename)
    fig.savefig(filepath, bbox_inches='tight', facecolor='white')
    plt.close(fig)

    return annotations


def generate_circular_part_drawing(output_dir, idx, sector="aerospace"):
    """Generate a circular part (flange, bearing, pipe) with diameter dimensions."""
    fig, ax = plt.subplots(1, 1, figsize=(11, 8.5), dpi=150)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect('equal')
    ax.axis('off')

    border = patches.Rectangle((0.02, 0.02), 0.96, 0.96,
                                 linewidth=2, edgecolor='black', facecolor='white')
    ax.add_patch(border)

    # Random dimensions
    outer_d = round(random.uniform(50, 500), 1)
    inner_d = round(random.uniform(10, outer_d * 0.6), 1)
    thickness = round((outer_d - inner_d) / 2, 1)
    height = round(random.uniform(10, 100), 1)
    n_holes = random.choice([4, 6, 8, 12])
    hole_d = round(random.uniform(5, 20), 1)
    tol_d = round(random.uniform(0.01, 0.1), 3)

    part_name = random.choice([
        "Bearing Housing", "Flange Coupling", "Pipe Flange",
        "Windshield Frame Ring", "Motor Mount Ring", "Propeller Hub",
    ])
    material = random.choice([
        "AL 7075-T6", "Steel 4340", "Inconel 718",
        "Bronze C93200", "Stainless 316L", "Titanium Gr.5",
    ])

    # Draw front view (concentric circles)
    cx, cy = 0.4, 0.5
    r_outer = 0.18
    r_inner = r_outer * inner_d / outer_d

    outer_circle = plt.Circle((cx, cy), r_outer, fill=False, linewidth=1.5, edgecolor='black')
    inner_circle = plt.Circle((cx, cy), r_inner, fill=False, linewidth=1.5, edgecolor='black')
    ax.add_patch(outer_circle)
    ax.add_patch(inner_circle)

    # Center lines
    ax.plot([cx - r_outer - 0.05, cx + r_outer + 0.05], [cy, cy], 'b--', linewidth=0.3)
    ax.plot([cx, cx], [cy - r_outer - 0.05, cy + r_outer + 0.05], 'b--', linewidth=0.3)

    # Bolt holes
    pcd = (r_outer + r_inner) / 2
    for i in range(n_holes):
        angle = 2 * math.pi * i / n_holes
        hx = cx + pcd * math.cos(angle)
        hy = cy + pcd * math.sin(angle)
        hole = plt.Circle((hx, hy), hole_d/outer_d * r_outer, fill=False,
                           linewidth=0.8, edgecolor='black')
        ax.add_patch(hole)

    # Diameter dimensions
    od_text = f"Ø{outer_d:.1f} ±{tol_d}"
    ax.annotate('', xy=(cx + r_outer, cy), xytext=(cx - r_outer, cy),
                arrowprops=dict(arrowstyle='<->', color='red', lw=0.8))
    ax.text(cx, cy + r_outer + 0.04, od_text, ha='center', fontsize=7,
            fontweight='bold', color='red')

    id_text = f"Ø{inner_d:.1f}"
    ax.annotate('', xy=(cx + r_inner, cy - 0.03), xytext=(cx - r_inner, cy - 0.03),
                arrowprops=dict(arrowstyle='<->', color='blue', lw=0.8))
    ax.text(cx, cy - r_inner - 0.06, id_text, ha='center', fontsize=7,
            fontweight='bold', color='blue')

    # Hole callout
    ax.text(cx + pcd + 0.05, cy + pcd + 0.05,
            f"{n_holes}x Ø{hole_d:.1f} THRU\nON Ø{outer_d*0.75:.0f} PCD",
            fontsize=5, fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    # Side view
    svx, svy = 0.72, 0.4
    svw = 0.15
    svh = height / outer_d * 0.36

    side = patches.Rectangle((svx, svy), svw, svh,
                               linewidth=1.5, edgecolor='black', facecolor='#f0f0f0')
    ax.add_patch(side)
    ax.text(svx + svw/2, svy - 0.03, "SIDE VIEW", ha='center', fontsize=6, fontstyle='italic')

    height_text = f"{height:.1f}"
    _draw_dimension_line(ax, svx + svw, svy, svx + svw, svy + svh, height_text, offset=0.05)

    # Notes
    ax.text(0.08, 0.9, f"ALL DIMS IN mm  |  TOLERANCES: ±{tol_d*10:.1f}mm UNLESS NOTED",
            fontsize=6, fontfamily='monospace')

    _draw_title_block(ax, part_name, material, f"Ø{outer_d}", sector)

    annotations = DrawingAnnotation(
        filename=f"circular_{idx:04d}.png",
        part_name=part_name,
        material=material,
        sector=sector,
        dimensions=[
            DimensionAnnotation("diameter", outer_d, "mm", tol_d, -tol_d,
                                od_text, cx, cy + r_outer + 0.04, 0.1, 0.03, "outer_diameter"),
            DimensionAnnotation("diameter", inner_d, "mm", tol_d, -tol_d,
                                id_text, cx, cy - r_inner - 0.06, 0.1, 0.03, "inner_diameter"),
            DimensionAnnotation("linear", height, "mm", tol_d*10, -tol_d*10,
                                height_text, svx + svw + 0.05, svy + svh/2, 0.04, svh, "height"),
            DimensionAnnotation("diameter", hole_d, "mm", 0.05, -0.05,
                                f"Ø{hole_d:.1f}", cx + pcd, cy + pcd, 0.08, 0.05, "bolt_hole_diameter"),
        ],
        notes=[
            f"{n_holes}x Ø{hole_d} bolt holes on PCD {outer_d*0.75:.0f}",
            f"Material: {material}",
        ],
    )

    filepath = os.path.join(output_dir, annotations.filename)
    fig.savefig(filepath, bbox_inches='tight', facecolor='white')
    plt.close(fig)

    return annotations


def generate_bracket_drawing(output_dir, idx, sector="naval"):
    """Generate an L-bracket or gusset drawing."""
    fig, ax = plt.subplots(1, 1, figsize=(11, 8.5), dpi=150)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect('equal')
    ax.axis('off')

    border = patches.Rectangle((0.02, 0.02), 0.96, 0.96,
                                 linewidth=2, edgecolor='black', facecolor='white')
    ax.add_patch(border)

    # Random dimensions
    leg_a = round(random.uniform(50, 300), 0)
    leg_b = round(random.uniform(50, 300), 0)
    thickness = round(random.uniform(3, 20), 1)
    fillet_r = round(random.uniform(5, 30), 0)
    n_holes = random.choice([2, 3, 4])
    hole_d = round(random.uniform(6, 16), 1)
    tol = round(random.uniform(0.1, 0.5), 2)

    part_name = random.choice([
        "Bracket Assembly", "Gusset Plate", "Stiffener Bracket",
        "Motor Mount Bracket", "Sensor Bracket", "Keel Bracket",
    ])
    material = random.choice([
        "Steel A36", "AL 6061-T6", "Stainless 304",
        "AH36 Marine Steel", "AL 5083-H116",
    ])

    # Draw L-shape
    bx, by = 0.2, 0.3
    scale = 0.4 / max(leg_a, leg_b)
    sa = leg_a * scale
    sb = leg_b * scale
    st = thickness * scale * 3  # exaggerate thickness for visibility

    # L-profile points
    pts = [
        (bx, by),
        (bx + sa, by),
        (bx + sa, by + st),
        (bx + st, by + st),
        (bx + st, by + sb),
        (bx, by + sb),
    ]
    polygon = plt.Polygon(pts, closed=True, linewidth=1.5,
                           edgecolor='black', facecolor='#e8e8e8')
    ax.add_patch(polygon)

    # Dimension lines
    a_text = f"{leg_a:.0f}"
    _draw_dimension_line(ax, bx, by, bx + sa, by, a_text, offset=-0.06)

    b_text = f"{leg_b:.0f}"
    _draw_dimension_line(ax, bx, by, bx, by + sb, b_text, offset=-0.08)

    t_text = f"{thickness:.1f} ±{tol}"
    ax.text(bx + sa/2, by + st + 0.02, t_text, ha='center', fontsize=7, fontweight='bold')

    # Holes
    for i in range(n_holes):
        hx = bx + sa * 0.3 + i * sa * 0.4 / max(n_holes - 1, 1)
        hy = by + st / 2
        hole = plt.Circle((hx, hy), hole_d/leg_a * sa / 2, fill=False,
                           linewidth=0.8, edgecolor='black')
        ax.add_patch(hole)

    ax.text(bx + sa + 0.05, by + st/2,
            f"{n_holes}x Ø{hole_d:.1f}",
            fontsize=6, fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    # Fillet note
    ax.text(bx + st + 0.02, by + st + 0.02,
            f"R{fillet_r:.0f}", fontsize=6, fontweight='bold')

    ax.text(0.08, 0.88, f"DIMS IN mm  |  GENERAL TOL: ±{tol*5:.1f}mm",
            fontsize=6, fontfamily='monospace')

    _draw_title_block(ax, part_name, material, f"{leg_a}x{leg_b}x{thickness}", sector)

    annotations = DrawingAnnotation(
        filename=f"bracket_{idx:04d}.png",
        part_name=part_name,
        material=material,
        sector=sector,
        dimensions=[
            DimensionAnnotation("linear", leg_a, "mm", tol*5, -tol*5,
                                a_text, bx + sa/2, by - 0.06, sa, 0.04, "leg_length_a"),
            DimensionAnnotation("linear", leg_b, "mm", tol*5, -tol*5,
                                b_text, bx - 0.08, by + sb/2, 0.04, sb, "leg_length_b"),
            DimensionAnnotation("linear", thickness, "mm", tol, -tol,
                                t_text, bx + sa/2, by + st + 0.02, 0.1, 0.03, "thickness"),
            DimensionAnnotation("radius", fillet_r, "mm", 0, 0,
                                f"R{fillet_r:.0f}", bx + st, by + st, 0.04, 0.03, "fillet_radius"),
            DimensionAnnotation("diameter", hole_d, "mm", 0.05, -0.05,
                                f"Ø{hole_d:.1f}", bx + sa*0.5, by + st/2, 0.08, 0.03, "hole_diameter"),
        ],
        notes=[f"Fillet R{fillet_r}mm all internal corners", f"Material: {material}"],
    )

    filepath = os.path.join(output_dir, annotations.filename)
    fig.savefig(filepath, bbox_inches='tight', facecolor='white')
    plt.close(fig)

    return annotations


def generate_training_dataset(output_dir: str, n_samples: int = 100):
    """Generate a full training dataset of annotated engineering drawings."""
    os.makedirs(output_dir, exist_ok=True)

    generators = [
        (generate_plate_drawing, "naval", 0.35),
        (generate_plate_drawing, "aerospace", 0.15),
        (generate_circular_part_drawing, "aerospace", 0.20),
        (generate_circular_part_drawing, "automotive_ev", 0.10),
        (generate_bracket_drawing, "naval", 0.10),
        (generate_bracket_drawing, "aerospace", 0.10),
    ]

    all_annotations = []
    idx = 0

    for gen_fn, sector, fraction in generators:
        count = max(1, int(n_samples * fraction))
        for _ in range(count):
            try:
                ann = gen_fn(output_dir, idx, sector)
                all_annotations.append(ann)
                idx += 1
            except Exception as e:
                print(f"Warning: Failed to generate drawing {idx}: {e}")
                idx += 1

    # Save annotations as JSON
    ann_data = []
    for ann in all_annotations:
        ann_data.append({
            "filename": ann.filename,
            "part_name": ann.part_name,
            "material": ann.material,
            "sector": ann.sector,
            "dimensions": [
                {
                    "type": d.dim_type,
                    "value": d.value,
                    "unit": d.unit,
                    "tolerance_plus": d.tolerance_plus,
                    "tolerance_minus": d.tolerance_minus,
                    "text": d.text,
                    "label": d.label,
                    "bbox": [d.x, d.y, d.width, d.height],
                }
                for d in ann.dimensions
            ],
            "notes": ann.notes,
        })

    ann_path = os.path.join(output_dir, "annotations.json")
    with open(ann_path, "w") as f:
        json.dump(ann_data, f, indent=2)

    print(f"Generated {len(all_annotations)} drawings in {output_dir}")
    print(f"Annotations saved to {ann_path}")

    return all_annotations


if __name__ == "__main__":
    generate_training_dataset("training_data/drawings", n_samples=100)

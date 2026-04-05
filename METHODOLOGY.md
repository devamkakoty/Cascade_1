# Cascade Failure Prediction — Methodology & Architecture

## 1. Executive Summary

This platform predicts the downstream engineering impact of a design change across interconnected subsystems. Given a single parameter change (e.g., increasing hull plate thickness by 3mm), it computes every affected parameter — mass, stress, thermal load, drag, cost, schedule — and flags certification violations.

**Core principle:** The entire cascade propagation is deterministic and physics-based. No LLMs, no black-box ML models in the prediction path. Every result is fully reproducible and auditable.

---

## 2. Architecture Overview

```
┌──────────────────────────────────────────────────────────────┐
│                      Streamlit UI                             │
│  Step 1: Upload (CAD/Drawing/Spec/Sample Library)            │
│  Step 2: Identify Part (sector + part type)                  │
│  Step 3: Configure Cascade (template or auto-assemble)       │
│  Step 4: Propagate & View Results                            │
├──────────────────────────────────────────────────────────────┤
│                    Cascade Engine                              │
│  • BFS graph traversal (single-pass for DAGs)                │
│  • Constraint checking against regulatory standards          │
│  • Cost & schedule roll-up                                   │
│  • Optional: Bayesian Monte Carlo uncertainty                │
├────────────┬─────────────┬───────────────────────────────────┤
│ Templates  │ Auto-       │ Universal Physics Library          │
│ (hand-     │ Assembler   │ 20+ parameterized models           │
│  built)    │ (geometry + │ Material DB (12 materials)         │
│            │  material)  │                                    │
├────────────┴─────────────┴───────────────────────────────────┤
│                    Input Parsers                               │
│  CAD (STL/STEP) · 2D Drawing (Tesseract OCR) · Spec (CSV)   │
│  Part Identifier · Sample Library (9 models)                 │
├──────────────────────────────────────────────────────────────┤
│                 Data Pipeline (future)                         │
│  ECR Fitting · Simulation Surrogates · Operational Sensors   │
└──────────────────────────────────────────────────────────────┘
```

---

## 3. Core Methodology: Deterministic Graph Propagation

### 3.1 Graph Representation

Every system is a **directed acyclic graph (DAG)**:

- **Nodes** = engineering parameters (scalar values with units)
  - Examples: `hull_plate_thickness` (25.0 mm), `total_mass` (1200.0 kg), `max_stress` (180.0 MPa)
  - Each node belongs to a subsystem (structural, thermal, electrical, performance, cost, schedule)
  - Nodes can have regulatory limits (e.g., max stress ≤ 250 MPa per DNV)

- **Edges** = physics coupling equations
  - Each edge has: source node, target node, sensitivity coefficient, physics equation name
  - Example: `hull_plate_thickness → hull_mass` with coefficient `ρ × A × 0.001` (density × area × mm-to-m)

### 3.2 Propagation Algorithm

```
Input: trigger_node, delta_value
Output: list of (affected_node, delta, violation?)

1. Queue ← [(trigger_node, delta_value)]
2. While Queue not empty:
   a. (node, delta) ← dequeue
   b. For each outgoing edge from node:
      c. downstream_delta = delta × edge.sensitivity
      d. Update target node value
      e. Check regulatory limit → flag violation if exceeded
      f. Enqueue (target_node, downstream_delta)
3. Return all affected nodes with deltas and violations
```

**Mode: single_pass** — each node processed once in topological order. Required for auto-assembled DAGs.

**Mode: iterative** (templates only) — allows cycles with damping factor (0.7) for convergence. Used when template graphs have feedback loops.

### 3.3 Sensitivity Coefficients

Edge sensitivities are **not learned** — they are computed from physics:

| Coupling | Equation | Units |
|----------|----------|-------|
| Thickness → Mass | `ρ × A × 0.001` | kg/mm |
| Mass → Weight | `g = 9.81` | N/kg |
| Thickness → Section Modulus | `b × t / 3` | mm³/mm |
| Load → Stress | `1 / (A × 1e6)` | MPa/N |
| Stress → Fatigue Life | `-N₀ × m / σ₀` (S-N curve) | cycles/MPa |
| ΔT → Heat Flow | `k × A / L` (Fourier) | W/K |
| Mass → Drag | `½ × ρ_air × v² × Cd × ΔA` | N/kg |
| Drag → Power | `v` | W/N |
| Power → Range | `-R₀ / P₀` | km/kW |

All coefficients are parameterized by **material properties** from the material database, not hard-coded per application.

---

## 4. Universal Physics Library (Option A)

### 4.1 Design Decision

We evaluated four approaches for physics coupling:

| Option | Approach | Decision |
|--------|----------|----------|
| **A** | Universal parameterized physics models | **Selected** — deterministic, auditable |
| B | LLM-generated coupling graphs | Rejected — hallucination risk, human-in-loop negates automation |
| C | Engineering Change Record fitting | In pipeline — calibration data source |
| D | Simulation surrogates (FEA/CFD) | In pipeline — for complex geometries |

### 4.2 Physics Model Catalog

20+ models organized by domain:

**Mass Models:**
- `ThicknessToMass` — plate: `Δm = ρ × L × W × Δt × 0.001`
- `VolumeToMass` — general: `Δm = ρ × ΔV`
- `DiameterToMass` — solid cylinder: `Δm = ρ × π/4 × L × 2D × ΔD × 0.001`
- `TubeDiameterToMass` — hollow tube: `Δm = ρ × π × L × t × ΔD × 0.001`

**Structural Models:**
- `MassToWeight` — `ΔW = g × Δm`
- `ThicknessToSectionModulus` — rectangular: `ΔZ = b × 2t × Δt / 6`
- `StressFromLoad` — `Δσ = ΔF / A` (output in MPa)
- `BendingMomentFromLoad` — `ΔM = L/2 × ΔF`
- `FatigueLifeFromStress` — S-N curve: `ΔN = -N₀ × m / σ₀ × Δσ`

**Thermal Models:**
- `FourierConductionUniversal` — `Δq = k × A / L × ΔT`
- `HeatToTemperature` — `ΔT = ΔQ / (m × Cp)`
- `PowerToHeat` — `ΔQ = η_loss × ΔP`

**Fluid/Performance Models:**
- `MassToDrag` — `ΔD = ½ρv²Cd × ΔA_frontal`
- `DragToPower` — `ΔP = v × ΔD`
- `PowerToFuelConsumption` — `Δfuel = BSFC × ΔP`
- `PowerToRange` — `ΔR = -R₀/P₀ × ΔP`
- `MassToRange` — `ΔR = -R₀/M₀ × ΔM`

**Electrical Models:**
- `PowerBalance` — `ΔP_total = Σ ΔP_components`
- `ResistiveHeating` — `ΔQ = I²R × Δt`

### 4.3 Material Database

12 materials with full property sets:

| Material | Density (kg/m³) | Yield (MPa) | k (W/mK) | Cost ($/kg) |
|----------|-----------------|-------------|-----------|-------------|
| Mild Steel | 7,850 | 250 | 50 | 0.80 |
| High-Strength Steel | 7,850 | 690 | 40 | 2.50 |
| AL 6061-T6 | 2,700 | 276 | 167 | 3.50 |
| AL 7075-T6 | 2,810 | 503 | 130 | 6.00 |
| AL 2024-T3 | 2,780 | 345 | 121 | 5.00 |
| Ti-6Al-4V | 4,430 | 880 | 6.7 | 25.00 |
| CFRP | 1,600 | 600 | 5.0 | 40.00 |
| GFRP | 2,000 | 300 | 0.3 | 15.00 |
| Copper | 8,960 | 210 | 385 | 8.00 |
| Inconel 718 | 8,190 | 1,034 | 11.4 | 35.00 |
| Stainless 316L | 7,990 | 205 | 16.3 | 4.00 |
| Marine Grade AL 5083 | 2,650 | 215 | 117 | 4.50 |

Each entry also includes: `ultimate_strength`, `youngs_modulus`, `specific_heat`, `fatigue_endurance`.

Fuzzy name matching handles OCR variants (e.g., "AL-6061" → "al_6061_t6").

---

## 5. Graph Auto-Assembly

For parts without a pre-built template, the assembler builds a cascade graph automatically:

### 5.1 Input
- **Geometry** — from CAD parser (STL bounding box, STEP dimensions) or drawing OCR (dimensions, tolerances)
- **Material** — selected from material database
- **Sector** — determines safety factors, temperature limits, constraint standards

### 5.2 Process

1. **Classify shape** — plate-like, cylindrical, hollow tube, or general solid (based on aspect ratios)
2. **Select physics chain** — e.g., plate → ThicknessToMass → MassToWeight → StressFromLoad → FatigueLifeFromStress
3. **Compute coefficients** — from geometry dimensions and material properties (with mm→m unit conversion)
4. **Add thermal chain** — FourierConduction + HeatToTemperature if applicable
5. **Add performance chain** — drag, power, range if sector requires it
6. **Add cost/schedule nodes** — material cost, manufacturing cost, tooling cost, lead time, certification time
7. **Add constraints** — safety factors, temperature limits, stress limits from sector-specific standards

### 5.3 Unit Handling

- CAD/drawing dimensions are in **mm** (stored in nodes as mm)
- Physics models expect **meters** internally
- Conversion factor `0.001` applied in edge coefficients (not node values)
- Stress output in **MPa** (not Pa) — division by `1e6` in coefficient

### 5.4 Sector-Specific Constants

| Sector | Safety Factor | Temp Limit (°C) | Max Stress Fraction | Standards |
|--------|--------------|-----------------|---------------------|-----------|
| Aerospace | 1.5 | 150 | 0.6 × yield | FAR 25, CS-25 |
| Naval | 2.0 | 200 | 0.5 × yield | DNV GL |
| Automotive EV | 1.8 | 85 | 0.55 × yield | UN R100, ISO 6469 |
| Robotics | 1.5 | 85 | 0.6 × yield | ISO 10218, ISO 9283, IEC 60034 |

---

## 6. Pre-Built Templates

Templates are hand-crafted cascade graphs for common systems:

### 6.1 Electric Aircraft (Eviation-style)
- **Subsystems:** structural, thermal, electrical, performance
- **Trigger examples:** windshield material change, wing spar thickness
- **Key couplings:** mass → drag → power → range, thermal conductivity → heat flow → battery temp

### 6.2 EV Battery Pack
- **Subsystems:** cell, cooling, electrical, mechanical
- **Trigger examples:** cell thickness change, cooling plate modification
- **Key couplings:** cell dimensions → pack mass → range, cooling efficiency → thermal management

### 6.3 Robotic Arm (6-DOF)
- **33 nodes, 37 edges** across structural, actuator, electrical, thermal, performance, end_effector
- **Trigger examples:** payload upgrade (5→8 kg), link extension (400→500 mm), grip force increase
- **Key couplings:**
  - `link_length → mass` (ρ × A_tube × ΔL)
  - `link_length → tip_deflection` (cantilever L³ relationship)
  - `link_length → natural_frequency` (drops as L²)
  - `payload → joint_torque → actuator_mass → power → heat`
  - `grip_force → grippable_mass` (friction coefficient μ=0.6)
- **Constraints:** ISO 10218 vibration (15Hz min), ISO 9283 accuracy (0.2mm max), IEC 60034 thermal (85°C max)

---

## 7. Input Parsers

### 7.1 CAD Parser (STL / STEP)
- **STL:** Binary/ASCII parsing → bounding box → thickness, length, width, volume
- **STEP:** Text parsing for geometry parameters (when available)
- **Output:** `CADAnalysisResult` with `GeometryParameter` list
- **3D Viewer:** STL rendered via Three.js component in the browser

### 7.2 2D Drawing Parser (OCR)
- **Engine:** Tesseract OCR (with regex fallback if Tesseract unavailable)
- **Extracts:** Dimensions (diameter, radius, linear), tolerances (bilateral), material callouts, surface finish, hole patterns
- **Smart filtering:** Material grade numbers (2024, 6061, 7075) excluded from dimension detection using context-aware prefix checking
- **Sector inference:** Keyword matching for automatic sector suggestion

### 7.3 Spec Parser (CSV / PDF / Text)
- **CSV:** Column-based parameter extraction
- **PDF:** Text extraction via `pypdf`, then regex matching
- **Text:** Direct regex matching for component.property = value patterns

### 7.4 Part Identifier
- **4 sectors** × **4-5 part types each** = ~20 pre-defined part profiles
- **Auto-suggest:** Heuristic classification from geometry (volume, aspect ratio)
- **Custom parts:** User can describe any part; physics inferred from geometry features

### 7.5 Sample Library
- **9 pre-built STL models** generated programmatically (no external files needed)
- Shapes: box, cylinder, L-bracket
- Samples: Hull Plate, Deck Stiffener, Fuselage Panel, Wing Spar, Battery Cell, Cooling Plate, Robot Arm Link, Gripper Bracket, Motor Housing

---

## 8. Data Pipeline (Future Integration)

### 8.1 Option C — Engineering Change Records (ECR)

**Purpose:** Calibrate coupling coefficients from historical change-impact data.

**Method:**
1. Ingest ECR/ECO records with trigger parameter and observed downstream effects
2. Group by (source_node, target_node) pairs
3. Fit linear sensitivity coefficient via least-squares regression
4. Replace or blend with analytical coefficient in the graph

**Current state:** Synthetic data generator produces realistic ECR records for all 4 sectors. Fitting pipeline (`fit_coupling_coefficients`, `discover_all_couplings`) is implemented and tested.

### 8.2 Option D — Simulation Surrogates

**Purpose:** Replace analytical physics models with fitted response surfaces from FEA/CFD/modal analysis where the analytical equation is insufficient.

**Method:**
1. Ingest simulation results (parameter sweeps from ANSYS, ABAQUS, etc.)
2. Fit linear (`y = ax + b`) or quadratic (`y = ax² + bx + c`) response surface
3. Wrap as `SurrogatePhysicsModel` — drop-in replacement for any graph edge

**Current state:** Synthetic generators for plate stress, thermal, and vibration studies. Fitting pipeline with R² validation. `SurrogatePhysicsModel` wraps any fitted surrogate as a standard `PhysicsModel`.

### 8.3 Option E — Operational Sensor Data

**Purpose:** Detect when in-service parameters drift from design values, triggering cascade re-evaluation.

**Method:**
1. Map sensor IDs to graph node IDs via `SensorConfig`
2. Ingest real-time or batch sensor readings
3. Compute moving-average drift from design baseline
4. If drift exceeds threshold, trigger cascade propagation from that node

**Current state:** Sensor configurations defined for all 4 sectors. Synthetic stream generator with realistic drift and anomaly injection. Drift detection algorithm implemented.

---

## 9. Bayesian Uncertainty Analysis

Optional Monte Carlo mode for risk assessment:

1. Define uncertainty distributions on edge sensitivities (Normal, Uniform, or LogNormal)
2. Draw N samples (default 500) of each edge coefficient
3. Run N cascade propagations
4. Compute per-node distributions: mean, std, 5th/95th percentile
5. Compute per-constraint violation probability

**Output:** Probability of violation rather than binary pass/fail. Useful for trade studies where manufacturing tolerances or material batch variability matter.

Currently implemented for Electric Aircraft and EV Battery Pack templates.

---

## 10. Cost & Schedule Model

Every template/auto-assembled graph includes cost and schedule nodes:

**Cost nodes:**
- `material_cost` — driven by mass delta × material $/kg
- `manufacturing_cost` — driven by geometry complexity changes
- `tooling_cost` — driven by design changes requiring new tooling
- `total_cost_delta` — sum of above

**Schedule nodes:**
- `manufacturing_lead_time` — weeks impact from design changes
- `certification_time` — weeks for re-certification if violations exist
- `total_schedule_delta` — sum of above

Rates are sector-specific (e.g., aerospace certification is longer than automotive).

---

## 11. Deployment

### Docker
```bash
docker build -t cascade-predict .
docker run -p 8501:8501 cascade-predict
```

### Direct
```bash
pip install -r requirements.txt
apt-get install tesseract-ocr  # optional, for 2D drawing OCR
streamlit run app.py
```

### Cloud Deployment
Compatible with:
- **Streamlit Community Cloud** — push to GitHub, connect repo
- **AWS ECS / GCP Cloud Run / Azure Container Apps** — use Dockerfile
- **Heroku** — add `setup.sh` with Streamlit config and `Procfile`

The app is stateless (no database), so horizontal scaling is straightforward.

### Requirements
- Python 3.10+
- ~200 MB installed (without torch)
- Tesseract OCR optional (only for 2D drawing parsing)

---

## 12. What Is NOT Used

To be explicit about scope:

- **No LLMs** — no GPT, Claude, or any language model in the prediction pipeline
- **No neural networks** — cascade propagation is BFS on a DAG with analytical coefficients
- **No training data required** — physics models are equations, not learned functions
- **No external API calls** — everything runs locally, no cloud dependencies at runtime
- **No database** — all state is in-memory per session

The ML training infrastructure (`train_model.py`, `training_data/`) exists for a separate prediction model but is **not used** in the cascade propagation engine.

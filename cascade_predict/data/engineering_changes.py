"""
Option C — Data-driven coupling discovery from engineering change records.

Engineering Change Records (ECR/ECO) capture what parameter changed, what
downstream effects were observed, and the actual deltas. Over time, these
records can be used to:
  1. Validate physics model coefficients (is our ρ×A×Δt correct?)
  2. Discover couplings not in the physics model (supply chain, human factors)
  3. Fit empirical correction factors

This module provides:
  - Synthetic ECR data generator (for development/demo)
  - Data schema for real ECR ingestion
  - Coupling coefficient fitting from historical data
  - Pipeline stub for connecting to real PLM/ERP systems

Future: connect to Siemens Teamcenter, PTC Windchill, Jira, SAP, etc.
"""

from __future__ import annotations

import json
import math
import os
import random
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Tuple


# ── Data Schema ─────────────────────────────────────────────────────

@dataclass
class ParameterDelta:
    """A single observed parameter change."""
    node_id: str
    old_value: float
    new_value: float
    unit: str = ""
    subsystem: str = ""

    @property
    def delta(self) -> float:
        return self.new_value - self.old_value

    @property
    def pct_change(self) -> float:
        if self.old_value == 0:
            return float("inf")
        return (self.new_value - self.old_value) / abs(self.old_value) * 100


@dataclass
class EngineeringChangeRecord:
    """A single engineering change record (ECR/ECO)."""
    ecr_id: str
    timestamp: str
    sector: str
    part_type: str
    material: str
    description: str

    # What was changed (the trigger)
    trigger: ParameterDelta

    # What downstream effects were observed
    observed_effects: List[ParameterDelta] = field(default_factory=list)

    # Cost/schedule impact
    cost_delta_usd: float = 0.0
    schedule_delta_weeks: float = 0.0

    # Metadata
    source: str = ""           # "PLM", "ERP", "manual", "synthetic"
    confidence: float = 1.0     # 0-1, data quality
    tags: List[str] = field(default_factory=list)


@dataclass
class CouplingObservation:
    """An observed coupling between two parameters from ECR data."""
    source_node: str
    target_node: str
    observed_sensitivity: float  # Δtarget / Δsource
    n_observations: int
    std_dev: float = 0.0
    confidence: float = 0.0


# ── Synthetic ECR Generator ────────────────────────────────────────

# Templates for realistic engineering changes
_ECR_TEMPLATES = {
    "naval": [
        {
            "desc": "Hull plate thickness increase for fatigue life extension",
            "part": "hull_plate", "material": "AH36 Marine Steel",
            "trigger": ("thickness", 12.0, 15.0, "mm"),
            "effects": [
                ("part_mass", 423.9, 530.0, "kg"),
                ("hull_drag", 45.0, 48.2, "kN"),
                ("fuel_consumption", 320, 342, "L/h"),
                ("fatigue_life", 5e6, 12e6, "cycles"),
                ("material_cost", 0, 371, "$"),
            ],
            "cost": 15200, "schedule": 2.5,
        },
        {
            "desc": "Hull plate thickness reduction for weight saving",
            "part": "hull_plate", "material": "AH36 Marine Steel",
            "trigger": ("thickness", 12.0, 9.0, "mm"),
            "effects": [
                ("part_mass", 423.9, 318.0, "kg"),
                ("hull_drag", 45.0, 42.1, "kN"),
                ("safety_margin", 0.45, 0.22, "ratio"),
                ("material_cost", 0, -265, "$"),
            ],
            "cost": -8500, "schedule": 1.5,
        },
        {
            "desc": "Propulsion power upgrade for speed increase",
            "part": "propulsion", "material": "Steel 4340",
            "trigger": ("propulsion_power", 2400, 3200, "kW"),
            "effects": [
                ("fuel_consumption", 320, 426, "L/h"),
                ("max_speed", 25, 28.5, "kts"),
                ("engine_heat_rejection", 480, 640, "kW"),
                ("engine_room_temp", 45, 52, "°C"),
            ],
            "cost": 450000, "schedule": 12,
        },
    ],
    "aerospace": [
        {
            "desc": "Windshield thickness increase for bird strike",
            "part": "windshield", "material": "Glass/Acrylic laminate",
            "trigger": ("thickness", 8.0, 12.0, "mm"),
            "effects": [
                ("part_mass", 12.5, 18.7, "kg"),
                ("thermal_conductivity", 0.8, 0.8, "W/mK"),
                ("cabin_heat_load", 2.1, 2.4, "kW"),
                ("oew", 4850, 4856, "kg"),
                ("range_nm", 450, 447, "nm"),
            ],
            "cost": 8200, "schedule": 4,
        },
        {
            "desc": "Battery capacity upgrade for range extension",
            "part": "battery_pack", "material": "NMC 811",
            "trigger": ("battery_capacity", 200, 250, "kWh"),
            "effects": [
                ("battery_mass", 1200, 1500, "kg"),
                ("mtow", 6500, 6800, "kg"),
                ("range_nm", 450, 520, "nm"),
                ("wing_loading", 226, 237, "kg/m²"),
                ("stall_speed", 55.2, 56.5, "kts"),
            ],
            "cost": 125000, "schedule": 8,
        },
        {
            "desc": "Wing spar material change Al→CFRP",
            "part": "wing_spar", "material": "CFRP Laminate",
            "trigger": ("wing_mass", 380, 245, "kg"),
            "effects": [
                ("oew", 4850, 4715, "kg"),
                ("mtow", 6500, 6365, "kg"),
                ("wing_root_bending", 42000, 38500, "Nm"),
                ("range_nm", 450, 468, "nm"),
            ],
            "cost": 95000, "schedule": 16,
        },
    ],
    "automotive_ev": [
        {
            "desc": "Cell capacity increase for range extension",
            "part": "battery_cell", "material": "NMC 622",
            "trigger": ("cell_capacity", 50, 65, "Ah"),
            "effects": [
                ("pack_energy", 75, 97.5, "kWh"),
                ("pack_mass", 450, 520, "kg"),
                ("vehicle_range", 350, 420, "km"),
                ("charge_time_10_80", 32, 38, "min"),
                ("vehicle_curb_weight", 1800, 1870, "kg"),
            ],
            "cost": 3200, "schedule": 6,
        },
        {
            "desc": "Cooling plate thickness increase for thermal management",
            "part": "cooling_plate", "material": "AL 6061-T6",
            "trigger": ("thickness", 3.0, 5.0, "mm"),
            "effects": [
                ("part_mass", 7.8, 13.0, "kg"),
                ("max_cell_temp", 42, 38, "°C"),
                ("thermal_resistance", 0.8, 0.48, "K/W"),
                ("pack_mass", 450, 455, "kg"),
            ],
            "cost": 85, "schedule": 1,
        },
    ],
    "robotics": [
        {
            "desc": "Actuator torque upgrade for higher payload",
            "part": "joint_actuator", "material": "Steel 4340",
            "trigger": ("rated_torque", 50, 80, "Nm"),
            "effects": [
                ("actuator_mass", 2.5, 3.8, "kg"),
                ("power_consumption", 150, 240, "W"),
                ("payload_capacity", 5.0, 8.0, "kg"),
                ("joint_temperature", 55, 68, "°C"),
                ("cycle_time", 1.2, 1.4, "s"),
            ],
            "cost": 1200, "schedule": 3,
        },
        {
            "desc": "Link length extension for workspace increase",
            "part": "arm_link", "material": "AL 7075-T6",
            "trigger": ("link_length", 400, 500, "mm"),
            "effects": [
                ("link_mass", 1.8, 2.25, "kg"),
                ("workspace_radius", 850, 1050, "mm"),
                ("tip_deflection", 0.15, 0.29, "mm"),
                ("natural_frequency", 45, 32, "Hz"),
                ("max_payload_at_reach", 5.0, 3.2, "kg"),
            ],
            "cost": 350, "schedule": 2,
        },
        {
            "desc": "Gripper force increase for heavier objects",
            "part": "end_effector", "material": "AL 6061-T6",
            "trigger": ("grip_force", 50, 100, "N"),
            "effects": [
                ("gripper_mass", 0.8, 1.4, "kg"),
                ("power_consumption", 15, 30, "W"),
                ("grippable_mass", 3.0, 6.0, "kg"),
                ("cycle_time", 0.5, 0.7, "s"),
            ],
            "cost": 450, "schedule": 1.5,
        },
    ],
}


def _add_noise(value: float, noise_pct: float = 5.0) -> float:
    """Add Gaussian noise to a value."""
    return value * (1 + random.gauss(0, noise_pct / 100))


def generate_synthetic_ecr_dataset(
    output_path: str,
    n_records: int = 200,
    seed: int = 42,
) -> List[EngineeringChangeRecord]:
    """
    Generate synthetic engineering change records.

    Creates realistic ECR data with noise, covering all sectors.
    Each record represents a historical design change with observed
    downstream effects — the exact data needed for coupling discovery.
    """
    random.seed(seed)
    records = []

    sectors = list(_ECR_TEMPLATES.keys())
    sector_weights = [0.25, 0.25, 0.25, 0.25]  # equal distribution

    for i in range(n_records):
        sector = random.choices(sectors, weights=sector_weights)[0]
        template = random.choice(_ECR_TEMPLATES[sector])

        # Random timestamp in last 3 years
        days_ago = random.randint(0, 1095)
        ts = datetime(2024, 1, 1) - timedelta(days=days_ago)

        # Add noise to all values
        trig = template["trigger"]
        trigger = ParameterDelta(
            node_id=trig[0],
            old_value=_add_noise(trig[1], 3),
            new_value=_add_noise(trig[2], 5),
            unit=trig[3],
        )

        effects = []
        for eff in template["effects"]:
            effects.append(ParameterDelta(
                node_id=eff[0],
                old_value=_add_noise(eff[1], 3),
                new_value=_add_noise(eff[2], 8),  # more noise in observed effects
                unit=eff[3],
            ))

        record = EngineeringChangeRecord(
            ecr_id=f"ECR-{sector[:3].upper()}-{i:04d}",
            timestamp=ts.strftime("%Y-%m-%d"),
            sector=sector,
            part_type=template["part"],
            material=template["material"],
            description=template["desc"],
            trigger=trigger,
            observed_effects=effects,
            cost_delta_usd=_add_noise(template["cost"], 15),
            schedule_delta_weeks=_add_noise(template["schedule"], 20),
            source="synthetic",
            confidence=round(random.uniform(0.7, 1.0), 2),
            tags=[sector, template["part"]],
        )
        records.append(record)

    # Save to JSON
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    data = []
    for r in records:
        data.append({
            "ecr_id": r.ecr_id,
            "timestamp": r.timestamp,
            "sector": r.sector,
            "part_type": r.part_type,
            "material": r.material,
            "description": r.description,
            "trigger": {
                "node_id": r.trigger.node_id,
                "old_value": round(r.trigger.old_value, 4),
                "new_value": round(r.trigger.new_value, 4),
                "unit": r.trigger.unit,
            },
            "observed_effects": [
                {
                    "node_id": e.node_id,
                    "old_value": round(e.old_value, 4),
                    "new_value": round(e.new_value, 4),
                    "unit": e.unit,
                }
                for e in r.observed_effects
            ],
            "cost_delta_usd": round(r.cost_delta_usd, 2),
            "schedule_delta_weeks": round(r.schedule_delta_weeks, 2),
            "source": r.source,
            "confidence": r.confidence,
            "tags": r.tags,
        })

    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)

    return records


# ── Coupling Coefficient Fitting ────────────────────────────────────

def fit_coupling_coefficients(
    records: List[EngineeringChangeRecord],
    source_node: str,
    target_node: str,
) -> Optional[CouplingObservation]:
    """
    Fit a linear coupling coefficient from ECR data.

    Collects all records where source_node was the trigger and
    target_node was an observed effect, then fits:
      sensitivity = mean(Δtarget / Δsource)
    """
    sensitivities = []

    for r in records:
        if r.trigger.node_id != source_node:
            continue
        delta_source = r.trigger.delta
        if abs(delta_source) < 1e-10:
            continue

        for eff in r.observed_effects:
            if eff.node_id == target_node:
                sens = eff.delta / delta_source
                sensitivities.append(sens)

    if not sensitivities:
        return None

    n = len(sensitivities)
    mean_sens = sum(sensitivities) / n
    if n > 1:
        variance = sum((s - mean_sens) ** 2 for s in sensitivities) / (n - 1)
        std_dev = math.sqrt(variance)
    else:
        std_dev = 0.0

    # Confidence based on sample size and consistency
    cv = std_dev / abs(mean_sens) if abs(mean_sens) > 1e-10 else 1.0
    confidence = min(1.0, n / 10.0) * max(0.0, 1.0 - cv)

    return CouplingObservation(
        source_node=source_node,
        target_node=target_node,
        observed_sensitivity=mean_sens,
        n_observations=n,
        std_dev=std_dev,
        confidence=round(confidence, 3),
    )


def discover_all_couplings(
    records: List[EngineeringChangeRecord],
    min_observations: int = 3,
) -> List[CouplingObservation]:
    """
    Discover all coupling relationships from ECR data.

    Returns observed couplings with enough data to be statistically meaningful.
    """
    # Collect all (source, target) pairs
    pairs: Dict[Tuple[str, str], int] = {}
    for r in records:
        src = r.trigger.node_id
        for eff in r.observed_effects:
            key = (src, eff.node_id)
            pairs[key] = pairs.get(key, 0) + 1

    # Fit coefficients for pairs with enough observations
    couplings = []
    for (src, tgt), count in pairs.items():
        if count < min_observations:
            continue
        obs = fit_coupling_coefficients(records, src, tgt)
        if obs and obs.confidence > 0.1:
            couplings.append(obs)

    return sorted(couplings, key=lambda c: -c.confidence)


def load_ecr_dataset(path: str) -> List[EngineeringChangeRecord]:
    """Load ECR records from a JSON file."""
    with open(path) as f:
        data = json.load(f)

    records = []
    for d in data:
        trigger = ParameterDelta(**d["trigger"])
        effects = [ParameterDelta(**e) for e in d["observed_effects"]]
        records.append(EngineeringChangeRecord(
            ecr_id=d["ecr_id"],
            timestamp=d["timestamp"],
            sector=d["sector"],
            part_type=d["part_type"],
            material=d["material"],
            description=d["description"],
            trigger=trigger,
            observed_effects=effects,
            cost_delta_usd=d["cost_delta_usd"],
            schedule_delta_weeks=d["schedule_delta_weeks"],
            source=d.get("source", ""),
            confidence=d.get("confidence", 1.0),
            tags=d.get("tags", []),
        ))
    return records


if __name__ == "__main__":
    records = generate_synthetic_ecr_dataset("training_data/ecr_dataset.json", n_records=200)
    print(f"Generated {len(records)} ECR records")

    # Demo: discover couplings
    couplings = discover_all_couplings(records, min_observations=3)
    print(f"\nDiscovered {len(couplings)} coupling relationships:")
    for c in couplings[:10]:
        print(f"  {c.source_node} → {c.target_node}: "
              f"sensitivity={c.observed_sensitivity:.4f} "
              f"(n={c.n_observations}, σ={c.std_dev:.4f}, conf={c.confidence:.2f})")

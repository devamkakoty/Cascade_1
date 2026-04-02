"""
Part identification module — maps uploaded CAD geometry to a known part type,
selects the right sector/template, and connects extracted dimensions to the
correct physics models and cascade parameters.

Flow:
  1. User uploads CAD file → geometry extracted (bounding box, curvature, etc.)
  2. User selects sector + part type (or auto-detect in future)
  3. Module returns: template_id, component_id, parameter mappings, relevant physics
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class PhysicsMapping:
    """Maps a CAD geometry parameter to a cascade graph input."""
    cad_param: str          # e.g. "est_curvature"
    component_id: str       # e.g. "windshield"
    property_name: str      # e.g. "curvature"
    unit_conversion: float = 1.0  # multiply CAD value by this
    description: str = ""


@dataclass
class PartProfile:
    """Full identification of what the uploaded part is and how to cascade it."""
    part_type: str              # e.g. "windshield"
    part_label: str             # e.g. "Aircraft Windshield"
    sector: str                 # e.g. "Aerospace"
    template_id: str            # e.g. "electric_aircraft"
    component_id: str           # e.g. "windshield"
    mappings: list[PhysicsMapping] = field(default_factory=list)
    physics_models: list[str] = field(default_factory=list)
    description: str = ""


# ── Sector definitions ──────────────────────────────────────────────

SECTORS = {
    "aerospace": {
        "label": "Aerospace",
        "icon": "airplane",
        "template_id": "electric_aircraft",
        "parts": {
            "windshield": PartProfile(
                part_type="windshield",
                part_label="Aircraft Windshield / Canopy",
                sector="Aerospace",
                template_id="electric_aircraft",
                component_id="windshield",
                description="Transparent panel (glass/acrylic/polycarbonate) forming the cockpit glazing.",
                mappings=[
                    PhysicsMapping("est_curvature", "windshield", "curvature", 1.0,
                                   "Surface curvature drives aero drag, mass, solar load, and stress margin"),
                    PhysicsMapping("est_wall_thickness", "windshield", "thermal_conductivity", 0.5,
                                   "Thicker glass changes thermal conductivity path"),
                    PhysicsMapping("surface_area", "windshield", "thermal_conductivity", 0.001,
                                   "Larger area increases solar heat gain"),
                ],
                physics_models=[
                    "CurvatureDragEffect — curvature changes form drag, affects L/D",
                    "CurvatureMassScaling — more curvature = thicker edges = more mass",
                    "CurvatureSolarCapture — curvature changes solar projection area",
                    "CurvaturePressureStress — curvature converts bending to membrane stress",
                    "FourierConduction — heat transfer through glazing",
                ],
            ),
            "fuselage_panel": PartProfile(
                part_type="fuselage_panel",
                part_label="Fuselage Skin Panel",
                sector="Aerospace",
                template_id="electric_aircraft",
                component_id="fuselage",
                description="Structural skin panel of the aircraft fuselage.",
                mappings=[
                    PhysicsMapping("est_wall_thickness", "fuselage", "insulation_rvalue", 0.4,
                                   "Wall thickness affects insulation R-value"),
                    PhysicsMapping("surface_area", "fuselage", "insulation_rvalue", 0.0001,
                                   "Panel area affects heat transfer"),
                ],
                physics_models=[
                    "InsulationResistance — heat through fuselage wall",
                    "BeamBending — structural loads on fuselage",
                    "SafetyMargin — structural margin check",
                ],
            ),
            "battery_enclosure": PartProfile(
                part_type="battery_enclosure",
                part_label="Aircraft Battery Enclosure",
                sector="Aerospace",
                template_id="electric_aircraft",
                component_id="battery_pack",
                description="Housing for the aircraft battery pack.",
                mappings=[
                    PhysicsMapping("volume", "battery_pack", "specific_energy", 0.0001,
                                   "Enclosure volume constrains battery capacity"),
                    PhysicsMapping("est_wall_thickness", "battery_pack", "specific_energy", 0.1,
                                   "Wall thickness adds structural mass"),
                ],
                physics_models=[
                    "BatterySizing — mass from capacity and specific energy",
                    "ThermalMassScaling — thermal equipment scales with battery",
                    "BreguetRange — battery weight affects range",
                ],
            ),
            "wing_spar": PartProfile(
                part_type="wing_spar",
                part_label="Wing Spar / Structural Member",
                sector="Aerospace",
                template_id="electric_aircraft",
                component_id="wing",
                description="Primary load-carrying structural element of the wing.",
                mappings=[
                    PhysicsMapping("length", "wing", "span", 0.001,
                                   "Spar length relates to wing span"),
                    PhysicsMapping("volume", "wing", "structural_mass", 0.0028,
                                   "Volume x density = structural mass (aluminum)"),
                ],
                physics_models=[
                    "BeamBending — root bending moment from wing loading",
                    "SafetyMargin — structural safety factor",
                    "DirectMassSum — structural weight adds to OEW",
                ],
            ),
            "generic": PartProfile(
                part_type="generic",
                part_label="Generic Aerospace Component",
                sector="Aerospace",
                template_id="electric_aircraft",
                component_id="windshield",
                description="Unspecified aerospace component — map parameters manually.",
                mappings=[
                    PhysicsMapping("volume", "battery_pack", "specific_energy", 0.0001, "Volume-based sizing"),
                    PhysicsMapping("surface_area", "windshield", "thermal_conductivity", 0.001, "Surface area for thermal"),
                ],
                physics_models=["Select parameters manually below"],
            ),
        },
    },
    "automotive_ev": {
        "label": "Automotive / EV",
        "icon": "car",
        "template_id": "ev_battery_pack",
        "parts": {
            "battery_cell": PartProfile(
                part_type="battery_cell",
                part_label="Battery Cell (Cylindrical / Pouch / Prismatic)",
                sector="Automotive / EV",
                template_id="ev_battery_pack",
                component_id="battery_cell",
                description="Individual Li-ion cell for EV battery pack.",
                mappings=[
                    PhysicsMapping("volume", "battery_cell", "nominal_capacity", 0.00018,
                                   "Cell volume correlates with capacity (energy density)"),
                    PhysicsMapping("length", "battery_cell", "mass", 0.045,
                                   "Cell length/diameter drives mass"),
                    PhysicsMapping("est_wall_thickness", "battery_cell", "internal_resistance", 50.0,
                                   "Casing thickness affects thermal path and resistance"),
                ],
                physics_models=[
                    "JouleHeating — I²R heat generation in cell",
                    "ThermalResistanceToTemp — temperature rise through cell casing",
                    "CellEnergy — voltage × capacity",
                    "ThermalMargin — margin to thermal runaway onset",
                ],
            ),
            "battery_module": PartProfile(
                part_type="battery_module",
                part_label="Battery Module Assembly",
                sector="Automotive / EV",
                template_id="ev_battery_pack",
                component_id="battery_module",
                description="Module housing multiple cells with interconnects and cooling.",
                mappings=[
                    PhysicsMapping("volume", "battery_module", "mass_overhead", 0.001,
                                   "Module volume drives overhead mass (housing, busbars)"),
                    PhysicsMapping("surface_area", "battery_module", "mass_overhead", 0.0005,
                                   "Surface area for cooling interface sizing"),
                ],
                physics_models=[
                    "CountBasedMass — module mass from cell count",
                    "ModuleCountScaling — pack parameters from module count",
                    "HeatGenScaling — total heat from cell count",
                ],
            ),
            "cooling_plate": PartProfile(
                part_type="cooling_plate",
                part_label="Cold Plate / Cooling System Component",
                sector="Automotive / EV",
                template_id="ev_battery_pack",
                component_id="cooling_system",
                description="Liquid cooling cold plate for battery thermal management.",
                mappings=[
                    PhysicsMapping("est_wall_thickness", "cooling_system", "interface_resistance", 2.0,
                                   "Plate thickness affects thermal interface resistance"),
                    PhysicsMapping("surface_area", "cooling_system", "coolant_flow_rate", 0.0001,
                                   "Plate area drives required coolant flow"),
                ],
                physics_models=[
                    "ConvectiveCooling — coolant flow effect on temperature",
                    "ThermalResistanceToTemp — temperature through interface",
                    "COPPowerDraw — chiller power requirement",
                ],
            ),
            "pack_enclosure": PartProfile(
                part_type="pack_enclosure",
                part_label="Battery Pack Enclosure / Housing",
                sector="Automotive / EV",
                template_id="ev_battery_pack",
                component_id="battery_module",
                description="Structural housing for the complete battery pack.",
                mappings=[
                    PhysicsMapping("volume", "battery_module", "mass_overhead", 0.002,
                                   "Enclosure volume adds to pack mass"),
                    PhysicsMapping("est_wall_thickness", "cooling_system", "interface_resistance", 1.5,
                                   "Wall thickness affects heat dissipation"),
                ],
                physics_models=[
                    "DirectMassSum — enclosure mass adds to vehicle weight",
                    "WeightToConsumption — heavier pack = more energy use",
                    "ConsumptionToRange — energy consumption affects range",
                ],
            ),
            "generic": PartProfile(
                part_type="generic",
                part_label="Generic EV Component",
                sector="Automotive / EV",
                template_id="ev_battery_pack",
                component_id="battery_cell",
                description="Unspecified EV component — map parameters manually.",
                mappings=[
                    PhysicsMapping("volume", "battery_cell", "nominal_capacity", 0.0001, "Volume-based sizing"),
                    PhysicsMapping("est_wall_thickness", "cooling_system", "interface_resistance", 1.0, "Thickness for thermal"),
                ],
                physics_models=["Select parameters manually below"],
            ),
        },
    },
}


def get_sectors():
    """Return list of (sector_key, sector_label) tuples."""
    return [(k, v["label"]) for k, v in SECTORS.items()]


def get_parts_for_sector(sector_key):
    """Return list of (part_key, part_label) tuples for a sector."""
    sector = SECTORS.get(sector_key, {})
    parts = sector.get("parts", {})
    return [(k, v.part_label) for k, v in parts.items()]


def get_part_profile(sector_key, part_key):
    """Get the full PartProfile for a sector + part combination."""
    return SECTORS.get(sector_key, {}).get("parts", {}).get(part_key)


def auto_suggest_part(geometry_params, sector_key=None):
    """
    Heuristic: suggest a part type based on extracted geometry.
    Returns (sector_key, part_key, confidence) or None.
    """
    params = {p.name: p.value for p in geometry_params}

    suggestions = []

    # Curvature detected → likely windshield/canopy
    if params.get("est_curvature", 0) > 0.01 or params.get("max_curvature", 0) > 0.01:
        if params.get("est_wall_thickness", 0) < 10:
            suggestions.append(("aerospace", "windshield", 0.7))

    # Small cylindrical shape → battery cell
    vol = params.get("volume", 0)
    aspect = params.get("aspect_ratio", 1)
    if 500 < vol < 50000 and aspect > 2.5:
        suggestions.append(("automotive_ev", "battery_cell", 0.6))

    # Large flat shape with channels → cooling plate
    if params.get("est_wall_thickness", 0) < 5 and params.get("surface_area", 0) > 10000:
        if aspect < 3:
            suggestions.append(("automotive_ev", "cooling_plate", 0.5))

    # Long thin shape → wing spar
    if aspect > 8 and vol > 100000:
        suggestions.append(("aerospace", "wing_spar", 0.5))

    # Filter by sector if specified
    if sector_key:
        suggestions = [s for s in suggestions if s[0] == sector_key]

    if suggestions:
        suggestions.sort(key=lambda x: x[2], reverse=True)
        return suggestions[0]
    return None

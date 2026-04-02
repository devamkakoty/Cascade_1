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
    "naval": {
        "label": "Naval / Maritime",
        "icon": "ship",
        "template_id": "naval_vessel",
        "parts": {
            "hull_plate": PartProfile(
                part_type="hull_plate",
                part_label="Hull Plating / Shell Panel",
                sector="Naval / Maritime",
                template_id="naval_vessel",
                component_id="hull_plating",
                description="Steel plate forming the hull shell — AH36 or equivalent marine grade.",
                mappings=[
                    PhysicsMapping("est_wall_thickness", "hull_plating", "plate_thickness", 1.0,
                                   "Plate thickness directly drives hull mass and structural capacity"),
                    PhysicsMapping("surface_area", "hull_plating", "plate_thickness", 0.000001,
                                   "Plate area contributes to wetted surface and drag"),
                ],
                physics_models=[
                    "LinearModel — hull mass scales with plate thickness (dm = rho * A * dt)",
                    "DirectMassSum — hull mass change affects displacement",
                    "Froude scaling — displacement drives hull resistance",
                    "SafetyMargin — section modulus must exceed DNV minimum",
                    "S-N Fatigue — thicker plates extend fatigue life",
                ],
            ),
            "propeller": PartProfile(
                part_type="propeller",
                part_label="Propeller / Thruster",
                sector="Naval / Maritime",
                template_id="naval_vessel",
                component_id="propulsion",
                description="Fixed or controllable pitch propeller for main propulsion.",
                mappings=[
                    PhysicsMapping("length", "propulsion", "rated_power", 0.8,
                                   "Propeller diameter relates to power absorption"),
                    PhysicsMapping("volume", "propulsion", "rated_power", 0.001,
                                   "Blade volume relates to thrust capability"),
                ],
                physics_models=[
                    "Power-speed — P = R * V / eta_prop",
                    "Fuel consumption — SFOC scaling from power",
                    "Range — fuel capacity / consumption rate",
                    "Shaft torque — T = P / (2*pi*n)",
                ],
            ),
            "rudder": PartProfile(
                part_type="rudder",
                part_label="Rudder / Steering Gear",
                sector="Naval / Maritime",
                template_id="naval_vessel",
                component_id="hull_plating",
                description="Rudder blade and steering mechanism.",
                mappings=[
                    PhysicsMapping("surface_area", "hull_plating", "plate_thickness", 0.00001,
                                   "Rudder area adds to appendage drag"),
                    PhysicsMapping("est_wall_thickness", "hull_plating", "plate_thickness", 0.8,
                                   "Rudder plate thickness for structural sizing"),
                ],
                physics_models=[
                    "Appendage drag — rudder area adds to total resistance",
                    "Structural — rudder stock bending from hydrodynamic force",
                    "Stability — rudder effectiveness affects maneuverability",
                ],
            ),
            "deck_structure": PartProfile(
                part_type="deck_structure",
                part_label="Deck Plating / Bulkhead",
                sector="Naval / Maritime",
                template_id="naval_vessel",
                component_id="hull_plating",
                description="Internal deck or bulkhead structural panel.",
                mappings=[
                    PhysicsMapping("est_wall_thickness", "hull_plating", "plate_thickness", 0.9,
                                   "Deck plate thickness contributes to structural weight"),
                    PhysicsMapping("surface_area", "hull_plating", "plate_thickness", 0.0000005,
                                   "Deck area contributes to total steel weight"),
                ],
                physics_models=[
                    "DirectMassSum — deck mass adds to displacement",
                    "Section modulus — deck plating contributes to hull girder strength",
                    "Stability — weight distribution affects GM",
                ],
            ),
            "heat_exchanger": PartProfile(
                part_type="heat_exchanger",
                part_label="Heat Exchanger / Cooler",
                sector="Naval / Maritime",
                template_id="naval_vessel",
                component_id="cooling",
                description="Seawater or freshwater heat exchanger for engine/HVAC cooling.",
                mappings=[
                    PhysicsMapping("surface_area", "cooling", "cooling_capacity", 0.05,
                                   "Heat transfer area determines cooling capacity"),
                    PhysicsMapping("est_wall_thickness", "cooling", "cooling_capacity", 10.0,
                                   "Tube/plate thickness affects thermal resistance"),
                ],
                physics_models=[
                    "HVACSizing — cooling system sized above waste heat",
                    "FourierConduction — heat transfer through exchanger walls",
                    "COPPowerDraw — electrical power for cooling pumps",
                ],
            ),
            "generic": PartProfile(
                part_type="generic",
                part_label="Generic Naval Component",
                sector="Naval / Maritime",
                template_id="naval_vessel",
                component_id="hull_plating",
                description="Unspecified naval component — map parameters manually.",
                mappings=[
                    PhysicsMapping("est_wall_thickness", "hull_plating", "plate_thickness", 1.0, "Thickness-based"),
                    PhysicsMapping("volume", "hull_plating", "plate_thickness", 0.0001, "Volume-based"),
                ],
                physics_models=["Select parameters manually below"],
            ),
        },
    },
}


# ── Custom / Other part (not in any dropdown) ──────────────────────

def build_custom_part_profile(
    sector_key,
    part_name,
    part_description,
    geometry_params,
):
    """
    Build a PartProfile on-the-fly for a part type not in the dropdown.
    Uses geometry features to infer which physics are relevant.

    Args:
        sector_key: which sector the user picked
        part_name: user-typed name for the part
        part_description: user-typed description
        geometry_params: list of GeometryParameter from CAD parser

    Returns:
        PartProfile with auto-generated mappings
    """
    params = {p.name: p for p in geometry_params}
    sector_info = SECTORS.get(sector_key, {})
    template_id = sector_info.get("template_id", "electric_aircraft")

    # Pick the first available component in the template as default
    _DEFAULT_COMPONENTS = {
        "aerospace": "windshield",
        "automotive_ev": "battery_cell",
        "naval": "hull_plating",
    }
    default_comp = _DEFAULT_COMPONENTS.get(sector_key, "windshield")

    mappings = []
    physics = []

    # ── Auto-detect relevant physics from geometry features ──────

    # Thickness detected → structural / mass physics
    if "est_wall_thickness" in params:
        t = params["est_wall_thickness"].value
        if sector_key == "naval":
            mappings.append(PhysicsMapping("est_wall_thickness", "hull_plating", "plate_thickness", 1.0,
                                           f"Wall thickness {t:.2f}mm → plate thickness"))
            physics.append("LinearModel — mass scales with thickness")
            physics.append("SafetyMargin — structural capacity check")
        elif sector_key == "aerospace":
            mappings.append(PhysicsMapping("est_wall_thickness", "windshield", "thermal_conductivity", 0.5,
                                           f"Wall thickness {t:.2f}mm → thermal conductivity scaling"))
            physics.append("FourierConduction — heat transfer through wall")
        elif sector_key == "automotive_ev":
            mappings.append(PhysicsMapping("est_wall_thickness", "cooling_system", "interface_resistance", 2.0,
                                           f"Wall thickness {t:.2f}mm → thermal interface resistance"))
            physics.append("ThermalResistanceToTemp — temperature through wall")

    # Curvature detected → aero/hydro physics
    if "est_curvature" in params or "max_curvature" in params:
        curv_key = "est_curvature" if "est_curvature" in params else "max_curvature"
        if sector_key == "aerospace":
            mappings.append(PhysicsMapping(curv_key, "windshield", "curvature", 1.0,
                                           "Curvature drives aero drag and structural stress"))
            physics.append("CurvatureDragEffect — form drag from curvature")
            physics.append("CurvaturePressureStress — membrane vs bending stress")
        else:
            physics.append("Curvature detected — may affect hydrodynamic or structural behavior")

    # Volume detected → sizing / capacity physics
    if "volume" in params:
        v = params["volume"].value
        if sector_key == "automotive_ev":
            mappings.append(PhysicsMapping("volume", "battery_cell", "nominal_capacity", 0.00018,
                                           f"Volume {v:.0f}mm³ → capacity estimate"))
            physics.append("CellEnergy — energy from voltage × capacity")
        elif sector_key == "aerospace":
            mappings.append(PhysicsMapping("volume", "battery_pack", "specific_energy", 0.0001,
                                           f"Volume {v:.0f}mm³ → battery sizing"))
            physics.append("BatterySizing — mass from capacity and specific energy")
        elif sector_key == "naval":
            mappings.append(PhysicsMapping("volume", "hull_plating", "plate_thickness", 0.0001,
                                           f"Volume {v:.0f}mm³ → structural contribution"))

    # Surface area → thermal physics
    if "surface_area" in params:
        sa = params["surface_area"].value
        if sector_key == "naval":
            mappings.append(PhysicsMapping("surface_area", "cooling", "cooling_capacity", 0.05,
                                           f"Surface area {sa:.0f}mm² → heat transfer area"))
            physics.append("HVACSizing — cooling capacity from heat transfer area")
        else:
            physics.append(f"Surface area {sa:.0f}mm² — relevant for thermal analysis")

    # Aspect ratio → structural form
    if "aspect_ratio" in params:
        ar = params["aspect_ratio"].value
        if ar > 5:
            physics.append(f"High aspect ratio ({ar:.1f}) — beam-like behavior, check bending")
        elif ar < 1.5:
            physics.append(f"Low aspect ratio ({ar:.1f}) — plate/shell behavior, check buckling")

    if not physics:
        physics.append("No specific physics auto-detected — select parameters manually below")

    # Always add cost & schedule physics (every part affects these)
    physics.append("--- Cost & Schedule ---")
    physics.append("Material cost — scales with mass change ($/kg)")
    physics.append("Manufacturing cost — scales with mass and complexity")
    physics.append("Lead time — heavier/complex parts take longer")
    physics.append("Certification time — constraint changes add review cycles")

    return PartProfile(
        part_type="custom",
        part_label=part_name or "Custom Part",
        sector=sector_info.get("label", "Unknown"),
        template_id=template_id,
        component_id=default_comp,
        mappings=mappings,
        physics_models=physics,
        description=part_description or "Custom part — physics inferred from extracted geometry features.",
    )


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

    # Thick flat plate → hull plating
    thickness = params.get("est_wall_thickness", 0)
    if 5 < thickness < 30 and params.get("surface_area", 0) > 50000:
        suggestions.append(("naval", "hull_plate", 0.55))

    # Large volume, low aspect → hull section / enclosure
    if vol > 500000 and aspect < 3:
        suggestions.append(("naval", "deck_structure", 0.4))

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

"""
Template: Electric Aircraft (Eviation-style)

Industry: Aerospace
Reference scenario: Windshield material change cascading through the full aircraft system.

Subsystems: thermal, hvac, electrical, mass, aerodynamic, structural, propulsion
"""

from cascade_predict.subsystems.aircraft import build_electric_aircraft
from cascade_predict.templates import TemplateInfo, register_template


TEMPLATE_INFO = TemplateInfo(
    template_id="electric_aircraft",
    name="Electric Aircraft",
    description=(
        "9-passenger electric aircraft (Eviation Alice-class). "
        "Models thermal, HVAC, electrical, mass, aero, structural, "
        "and propulsion coupling. Reference scenario: windshield swap "
        "cascading to MTOW violation."
    ),
    industry="Aerospace",
    build_fn=build_electric_aircraft,
    icon="aircraft",
    presets=[
        {
            "name": "Eviation: Windshield k 1.0 -> 1.4",
            "component_id": "windshield",
            "property_name": "thermal_conductivity",
            "new_value": 1.4,
        },
        {
            "name": "Eviation: Windshield k 1.0 -> 2.0",
            "component_id": "windshield",
            "property_name": "thermal_conductivity",
            "new_value": 2.0,
        },
        {
            "name": "Battery upgrade: 220 -> 280 Wh/kg",
            "component_id": "battery_pack",
            "property_name": "specific_energy",
            "new_value": 280.0,
        },
        {
            "name": "Insulation downgrade: R 2.5 -> 1.5",
            "component_id": "fuselage",
            "property_name": "insulation_rvalue",
            "new_value": 1.5,
        },
        {
            "name": "Windshield curvature increase: 0.15 -> 0.35",
            "component_id": "windshield",
            "property_name": "curvature",
            "new_value": 0.35,
        },
        {
            "name": "Windshield flatten: 0.15 -> 0.05",
            "component_id": "windshield",
            "property_name": "curvature",
            "new_value": 0.05,
        },
    ],
)

register_template(TEMPLATE_INFO)

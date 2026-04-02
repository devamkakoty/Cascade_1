"""
Template: Naval Patrol Vessel (diesel-electric)

Industry: Naval / Maritime
Reference scenario: Hull plate thickness change cascading through propulsion,
electrical, thermal, structural, and stability systems.

Subsystems: hull, propulsion, electrical, thermal, structural, navigation
"""

from cascade_predict.subsystems.naval_vessel import build_naval_vessel
from cascade_predict.templates import TemplateInfo, register_template


TEMPLATE_INFO = TemplateInfo(
    template_id="naval_vessel",
    name="Naval Patrol Vessel",
    description=(
        "450-tonne diesel-electric patrol vessel. "
        "Models hull, propulsion, electrical, thermal, structural, "
        "and navigation/stability coupling. "
        "Reference scenario: hull plate thickness change cascading "
        "to displacement, power, range, and classification violations."
    ),
    industry="Naval",
    build_fn=build_naval_vessel,
    icon="ship",
    presets=[
        {
            "name": "Hull thickening: 12mm -> 15mm (corrosion allowance)",
            "component_id": "hull_plating",
            "property_name": "plate_thickness",
            "new_value": 15.0,
        },
        {
            "name": "Hull thinning: 12mm -> 9mm (weight reduction)",
            "component_id": "hull_plating",
            "property_name": "plate_thickness",
            "new_value": 9.0,
        },
        {
            "name": "Power upgrade: 2400 -> 3200 kW",
            "component_id": "propulsion",
            "property_name": "rated_power",
            "new_value": 3200.0,
        },
        {
            "name": "Power downgrade: 2400 -> 1800 kW",
            "component_id": "propulsion",
            "property_name": "rated_power",
            "new_value": 1800.0,
        },
    ],
)

register_template(TEMPLATE_INFO)

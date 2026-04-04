"""
Robotics template registration.
"""
from cascade_predict.templates import TemplateInfo, register_template
from cascade_predict.subsystems.robotic_arm import build_robotic_arm

TEMPLATE_INFO = TemplateInfo(
    template_id="robotic_arm",
    name="Industrial Robot Arm (6-DOF)",
    description=(
        "6-DOF industrial manipulator with cascading dependencies across "
        "structural (links, deflections), actuator (torques, motors), "
        "control (accuracy, cycle time), electrical, and thermal subsystems."
    ),
    industry="Robotics",
    build_fn=build_robotic_arm,
    icon="robot",
    presets=[
        {
            "name": "Payload upgrade 5→8 kg",
            "component_id": "controller",
            "property_name": "payload_capacity",
            "new_value": 8.0,
        },
        {
            "name": "Payload downgrade 5→3 kg",
            "component_id": "controller",
            "property_name": "payload_capacity",
            "new_value": 3.0,
        },
        {
            "name": "Link 1 extension 400→500 mm",
            "component_id": "arm_link",
            "property_name": "link1_length",
            "new_value": 500,
        },
        {
            "name": "Grip force increase 50→100 N",
            "component_id": "end_effector",
            "property_name": "grip_force",
            "new_value": 100,
        },
    ],
)

register_template(TEMPLATE_INFO)

"""
Robotic arm / manipulator subsystem model.

Models a 6-DOF industrial robot arm with cascading dependencies across:
  - Structural (link masses, deflections, natural frequencies)
  - Actuator (torques, power, thermal)
  - Control (accuracy, cycle time, payload)
  - Electrical (power supply, battery/tether)
  - Thermal (motor heating, joint limits)

Cascade chains:
  link_length → link_mass → required_torque → motor_power → heat_generation
  link_length → tip_deflection → positioning_accuracy → cycle_time
  payload → required_torque → motor_sizing → total_mass → power_consumption
  grip_force → gripper_mass → payload_at_reach → workspace_utilization
"""

from __future__ import annotations

import math

from cascade_predict.graph.dependency_graph import (
    DependencyGraph,
    SubsystemNode,
    CouplingEdge,
)
from cascade_predict.subsystems.component import Component, CertConstraint
from cascade_predict.subsystems.cost_schedule import inject_cost_schedule_nodes, COST_RATES
from cascade_predict.physics_models import LinearModel


# ── Add robotics cost rates ─────────────────────────────────────────

COST_RATES["robotics"] = {
    "material_per_kg": 15.0,        # $/kg (aluminum/steel mix)
    "manufacturing_per_kg": 80.0,   # $/kg (CNC machining + assembly)
    "tooling_base": 25000.0,        # $ base tooling
    "tooling_per_change": 5000.0,   # $ per geometry change
    "cert_base": 10000.0,           # $ safety certification (ISO 10218)
    "cert_per_violation": 15000.0,  # $ per safety revalidation
    "mfg_lead_weeks": 6.0,
    "cert_lead_weeks": 4.0,
    "supplier_lead_weeks": 8.0,
}


def build_robotic_arm():
    """Build a 6-DOF industrial robot arm dependency graph."""

    graph = DependencyGraph()

    # ═══════════════════════════════════════════════════════════════
    # NODES
    # ═══════════════════════════════════════════════════════════════

    # ── Structural (links) ──────────────────────────────────────
    graph.add_node(SubsystemNode(
        "link1_length", "structural", 400, "mm",
        "Link 1 (shoulder→elbow) length",
    ))
    graph.add_node(SubsystemNode(
        "link2_length", "structural", 350, "mm",
        "Link 2 (elbow→wrist) length",
    ))
    graph.add_node(SubsystemNode(
        "link1_mass", "structural", 3.5, "kg",
        "Link 1 mass (AL 7075 tube)",
    ))
    graph.add_node(SubsystemNode(
        "link2_mass", "structural", 2.2, "kg",
        "Link 2 mass",
    ))
    graph.add_node(SubsystemNode(
        "total_arm_mass", "structural", 18.0, "kg",
        "Total arm mass (all links + actuators + wiring)",
    ))
    graph.add_node(SubsystemNode(
        "tip_deflection", "structural", 0.15, "mm",
        "Static tip deflection under rated payload",
        bounds=(0, 1.0),
    ))
    graph.add_node(SubsystemNode(
        "natural_frequency", "structural", 45.0, "Hz",
        "First natural frequency of arm structure",
        bounds=(15.0, float("inf")),
        regulatory_limit=15.0,
        regulatory_ref="ISO 10218 vibration limit",
    ))

    # ── Actuators ───────────────────────────────────────────────
    graph.add_node(SubsystemNode(
        "joint1_rated_torque", "actuator", 80.0, "Nm",
        "Joint 1 (shoulder) rated torque",
    ))
    graph.add_node(SubsystemNode(
        "joint2_rated_torque", "actuator", 50.0, "Nm",
        "Joint 2 (elbow) rated torque",
    ))
    graph.add_node(SubsystemNode(
        "wrist_torque", "actuator", 15.0, "Nm",
        "Wrist (joints 4-6) combined torque",
    ))
    graph.add_node(SubsystemNode(
        "actuator_mass", "actuator", 8.0, "kg",
        "Total actuator mass (all joints)",
    ))

    # ── Power / Electrical ──────────────────────────────────────
    graph.add_node(SubsystemNode(
        "peak_power", "electrical", 800.0, "W",
        "Peak electrical power consumption",
    ))
    graph.add_node(SubsystemNode(
        "continuous_power", "electrical", 350.0, "W",
        "Continuous power consumption at rated load",
    ))
    graph.add_node(SubsystemNode(
        "supply_voltage", "electrical", 48.0, "V",
        "DC bus voltage",
    ))
    graph.add_node(SubsystemNode(
        "peak_current", "electrical", 16.7, "A",
        "Peak current draw (800W / 48V)",
    ))

    # ── Thermal ─────────────────────────────────────────────────
    graph.add_node(SubsystemNode(
        "motor_heat_gen", "thermal", 120.0, "W",
        "Heat generated in motors (I²R losses)",
    ))
    graph.add_node(SubsystemNode(
        "joint_temperature", "thermal", 55.0, "°C",
        "Hottest joint temperature",
        regulatory_limit=85.0,
        regulatory_ref="Motor thermal class B limit",
    ))
    graph.add_node(SubsystemNode(
        "thermal_margin", "thermal", 0.35, "ratio",
        "Thermal margin (1 - T/T_max)",
        bounds=(0, float("inf")),
    ))

    # ── Performance / Control ───────────────────────────────────
    graph.add_node(SubsystemNode(
        "payload_capacity", "performance", 5.0, "kg",
        "Rated payload at full reach",
    ))
    graph.add_node(SubsystemNode(
        "workspace_radius", "performance", 850.0, "mm",
        "Maximum reach radius",
    ))
    graph.add_node(SubsystemNode(
        "positioning_accuracy", "performance", 0.05, "mm",
        "Repeatability (ISO 9283)",
        bounds=(0, 0.2),
        regulatory_limit=0.2,
        regulatory_ref="ISO 9283 positioning accuracy",
    ))
    graph.add_node(SubsystemNode(
        "cycle_time", "performance", 1.2, "s",
        "Pick-and-place cycle time (standard test)",
    ))
    graph.add_node(SubsystemNode(
        "max_speed", "performance", 2.0, "m/s",
        "Maximum TCP speed",
    ))

    # ── End effector ────────────────────────────────────────────
    graph.add_node(SubsystemNode(
        "grip_force", "end_effector", 50.0, "N",
        "Maximum gripper force",
    ))
    graph.add_node(SubsystemNode(
        "gripper_mass", "end_effector", 0.8, "kg",
        "Gripper mass",
    ))
    graph.add_node(SubsystemNode(
        "grippable_mass", "end_effector", 3.0, "kg",
        "Maximum object mass that can be gripped (friction limited)",
    ))

    # ═══════════════════════════════════════════════════════════════
    # EDGES (Physics Couplings)
    # ═══════════════════════════════════════════════════════════════

    # Link length → link mass (longer tube = more material)
    # m = ρ × π × (D² - d²)/4 × L;  linearized: Δm = ρ×A_cross × ΔL
    # AL 7075 tube ~40mm OD, 3mm wall: A_cross ≈ 3.49e-4 m²
    graph.add_edge(CouplingEdge(
        source_id="link1_length", target_id="link1_mass",
        model=LinearModel(
            coefficient=2810 * 3.49e-4 * 0.001,  # kg per mm (AL 7075, tube section)
            desc="Link mass from length (AL 7075 tube, 40mm OD × 3mm wall)",
            eq="Δm = ρ × A_tube × ΔL = 0.00098 kg/mm",
        ),
        description="Link 1 length → mass",
    ))

    graph.add_edge(CouplingEdge(
        source_id="link2_length", target_id="link2_mass",
        model=LinearModel(
            coefficient=2810 * 2.54e-4 * 0.001,  # smaller tube for link 2
            desc="Link 2 mass from length (smaller tube section)",
            eq="Δm = ρ × A_tube × ΔL",
        ),
        description="Link 2 length → mass",
    ))

    # Link masses + actuator mass → total arm mass
    graph.add_edge(CouplingEdge(
        source_id="link1_mass", target_id="total_arm_mass",
        model=LinearModel(coefficient=1.0),
        physics_equation="Δm_total = Δm_link1",
        description="Link 1 mass → total mass",
    ))
    graph.add_edge(CouplingEdge(
        source_id="link2_mass", target_id="total_arm_mass",
        model=LinearModel(coefficient=1.0),
        physics_equation="Δm_total = Δm_link2",
        description="Link 2 mass → total mass",
    ))
    graph.add_edge(CouplingEdge(
        source_id="actuator_mass", target_id="total_arm_mass",
        model=LinearModel(coefficient=1.0),
        physics_equation="Δm_total = Δm_actuators",
        description="Actuator mass → total mass",
    ))
    graph.add_edge(CouplingEdge(
        source_id="gripper_mass", target_id="total_arm_mass",
        model=LinearModel(coefficient=1.0),
        physics_equation="Δm_total = Δm_gripper",
        description="Gripper mass → total mass",
    ))

    # Link length → workspace radius
    graph.add_edge(CouplingEdge(
        source_id="link1_length", target_id="workspace_radius",
        model=LinearModel(coefficient=1.0),
        physics_equation="ΔR = ΔL1 (direct reach extension)",
        description="Link 1 length → workspace radius",
    ))
    graph.add_edge(CouplingEdge(
        source_id="link2_length", target_id="workspace_radius",
        model=LinearModel(coefficient=1.0),
        physics_equation="ΔR = ΔL2",
        description="Link 2 length → workspace radius",
    ))

    # Link length → tip deflection (δ = FL³/3EI for cantilever)
    # Linearized: Δδ = 3×δ₀/L₀ × ΔL (δ ∝ L³)
    graph.add_edge(CouplingEdge(
        source_id="link1_length", target_id="tip_deflection",
        model=LinearModel(
            coefficient=3 * 0.15 / 400,  # 3 × δ₀ / L₀ mm/mm
            desc="Tip deflection scales as L³ (linearized)",
            eq="Δδ ≈ 3×δ₀/L₀ × ΔL",
        ),
        description="Link 1 length → tip deflection (cantilever L³)",
    ))

    # Link length → natural frequency (f ∝ 1/L² for cantilever beam)
    # Linearized: Δf = -2×f₀/L₀ × ΔL
    graph.add_edge(CouplingEdge(
        source_id="link1_length", target_id="natural_frequency",
        model=LinearModel(
            coefficient=-2 * 45.0 / 400,  # -2×f₀/L₀ Hz/mm
            desc="Natural frequency drops with length² (linearized)",
            eq="Δf ≈ -2×f₀/L₀ × ΔL",
        ),
        description="Link 1 length → natural frequency (drops as L²)",
    ))

    # Payload + arm mass → required shoulder torque
    # τ = (m_payload + m_arm/2) × g × L_reach
    graph.add_edge(CouplingEdge(
        source_id="payload_capacity", target_id="joint1_rated_torque",
        model=LinearModel(
            coefficient=9.81 * 0.75,  # g × effective arm length (0.75m)
            desc="Shoulder torque from payload (τ = m × g × L)",
            eq="Δτ = Δm_payload × g × L_eff",
        ),
        description="Payload → shoulder torque",
    ))
    graph.add_edge(CouplingEdge(
        source_id="total_arm_mass", target_id="joint1_rated_torque",
        model=LinearModel(
            coefficient=9.81 * 0.375,  # g × L/2 (center of mass)
            desc="Shoulder torque from arm self-weight",
            eq="Δτ = Δm_arm × g × L/2",
        ),
        description="Arm mass → shoulder torque",
    ))

    # Payload → elbow torque
    graph.add_edge(CouplingEdge(
        source_id="payload_capacity", target_id="joint2_rated_torque",
        model=LinearModel(
            coefficient=9.81 * 0.35,  # g × link2 length
            desc="Elbow torque from payload",
            eq="Δτ = Δm × g × L2",
        ),
        description="Payload → elbow torque",
    ))

    # Torque → motor sizing → actuator mass
    # Larger torque = bigger motor. ~0.04 kg/Nm typical
    graph.add_edge(CouplingEdge(
        source_id="joint1_rated_torque", target_id="actuator_mass",
        model=LinearModel(
            coefficient=0.04,
            desc="Motor mass scales with torque (~0.04 kg/Nm)",
            eq="Δm_motor = 0.04 × Δτ",
        ),
        description="Shoulder torque → actuator mass",
    ))
    graph.add_edge(CouplingEdge(
        source_id="joint2_rated_torque", target_id="actuator_mass",
        model=LinearModel(coefficient=0.04),
        description="Elbow torque → actuator mass",
    ))

    # Torque → power: P = τ × ω (at max speed ~3 rad/s for J1)
    graph.add_edge(CouplingEdge(
        source_id="joint1_rated_torque", target_id="peak_power",
        model=LinearModel(
            coefficient=3.0,  # W per Nm at 3 rad/s
            desc="Peak power from torque (P = τ × ω)",
            eq="ΔP = Δτ × ω_max(3 rad/s)",
        ),
        description="Shoulder torque → peak power",
    ))
    graph.add_edge(CouplingEdge(
        source_id="joint2_rated_torque", target_id="peak_power",
        model=LinearModel(coefficient=4.0),  # faster joint
        description="Elbow torque → peak power",
    ))

    # Continuous power ≈ 40% of peak (duty cycle)
    graph.add_edge(CouplingEdge(
        source_id="peak_power", target_id="continuous_power",
        model=LinearModel(coefficient=0.4),
        physics_equation="P_cont ≈ 0.4 × P_peak (duty cycle)",
        description="Peak power → continuous power",
    ))

    # Power → current
    graph.add_edge(CouplingEdge(
        source_id="peak_power", target_id="peak_current",
        model=LinearModel(
            coefficient=1.0 / 48.0,  # I = P / V
            desc="Current from power (I = P/V at 48V)",
            eq="ΔI = ΔP / V(48V)",
        ),
        description="Peak power → peak current",
    ))

    # Power → heat generation (motor losses ~15%)
    graph.add_edge(CouplingEdge(
        source_id="continuous_power", target_id="motor_heat_gen",
        model=LinearModel(
            coefficient=0.15,  # 15% efficiency loss → heat
            desc="Motor losses → heat (η_loss = 15%)",
            eq="ΔQ = 0.15 × ΔP",
        ),
        description="Power → motor heat generation",
    ))

    # Heat → joint temperature (thermal resistance ~0.25 °C/W)
    graph.add_edge(CouplingEdge(
        source_id="motor_heat_gen", target_id="joint_temperature",
        model=LinearModel(
            coefficient=0.25,
            desc="Temperature rise from heat (R_th = 0.25 °C/W)",
            eq="ΔT = R_th × ΔQ",
        ),
        description="Heat generation → joint temperature",
    ))

    # Joint temperature → thermal margin
    graph.add_edge(CouplingEdge(
        source_id="joint_temperature", target_id="thermal_margin",
        model=LinearModel(
            coefficient=-1.0 / 85.0,  # margin = 1 - T/85
            desc="Thermal margin erodes with temperature",
            eq="Δmargin = -ΔT / T_max(85°C)",
        ),
        description="Temperature → thermal margin",
    ))

    # Tip deflection → positioning accuracy (directly related)
    graph.add_edge(CouplingEdge(
        source_id="tip_deflection", target_id="positioning_accuracy",
        model=LinearModel(
            coefficient=0.3,  # accuracy degrades ~30% of deflection
            desc="Deflection degrades positioning accuracy",
            eq="Δacc = 0.3 × Δdeflection",
        ),
        description="Tip deflection → positioning accuracy",
    ))

    # Total arm mass → max speed (heavier = slower: v ∝ τ/m)
    graph.add_edge(CouplingEdge(
        source_id="total_arm_mass", target_id="max_speed",
        model=LinearModel(
            coefficient=-2.0 / 18.0,  # -v₀/m₀ m/s per kg
            desc="Heavier arm reduces max speed",
            eq="Δv = -v₀/m₀ × Δm",
        ),
        description="Arm mass → max speed (inverse)",
    ))

    # Max speed + accuracy → cycle time
    graph.add_edge(CouplingEdge(
        source_id="max_speed", target_id="cycle_time",
        model=LinearModel(
            coefficient=-1.2 / 2.0,  # -t₀/v₀
            desc="Faster = shorter cycle time",
            eq="Δt = -t₀/v₀ × Δv",
        ),
        description="Max speed → cycle time",
    ))
    graph.add_edge(CouplingEdge(
        source_id="positioning_accuracy", target_id="cycle_time",
        model=LinearModel(
            coefficient=2.0,  # worse accuracy needs slower approach
            desc="Worse accuracy → slower approach phase",
            eq="Δt_cycle ≈ 2 × Δacc (settling time)",
        ),
        description="Accuracy → cycle time (settling)",
    ))

    # Grip force → gripper mass (~0.012 kg/N for pneumatic gripper)
    graph.add_edge(CouplingEdge(
        source_id="grip_force", target_id="gripper_mass",
        model=LinearModel(
            coefficient=0.012,
            desc="Gripper mass scales with force capacity",
            eq="Δm_gripper = 0.012 × ΔF_grip",
        ),
        description="Grip force → gripper mass",
    ))

    # Grip force → grippable mass (friction coefficient ~0.6)
    graph.add_edge(CouplingEdge(
        source_id="grip_force", target_id="grippable_mass",
        model=LinearModel(
            coefficient=0.6 / 9.81,  # m = μ×F / g
            desc="Grippable mass from friction (μ=0.6)",
            eq="Δm_grip = μ × ΔF / g",
        ),
        description="Grip force → grippable mass (friction limit)",
    ))

    # ── Cost/schedule injection ─────────────────────────────────
    inject_cost_schedule_nodes(
        graph, "robotics",
        mass_node_id="total_arm_mass",
        power_node_id="peak_power",
    )

    # ═══════════════════════════════════════════════════════════════
    # COMPONENTS
    # ═══════════════════════════════════════════════════════════════

    components = {}

    comp_link = Component("arm_link", "Arm Links", "structural",
                          "Upper and lower arm links (AL 7075 tubes)")
    comp_link.add_property("link1_length", 400, "mm", "link1_length", "Link 1 length")
    comp_link.add_property("link2_length", 350, "mm", "link2_length", "Link 2 length")
    components["arm_link"] = comp_link

    comp_act = Component("joint_actuator", "Joint Actuators", "actuator",
                         "Servo motors with harmonic drive reducers")
    comp_act.add_property("joint1_rated_torque", 80, "Nm", "joint1_rated_torque", "Shoulder torque")
    comp_act.add_property("joint2_rated_torque", 50, "Nm", "joint2_rated_torque", "Elbow torque")
    components["joint_actuator"] = comp_act

    comp_grip = Component("end_effector", "End Effector / Gripper", "end_effector",
                          "Pneumatic parallel gripper")
    comp_grip.add_property("grip_force", 50, "N", "grip_force", "Maximum grip force")
    components["end_effector"] = comp_grip

    comp_ctrl = Component("controller", "Motion Controller", "performance",
                          "Servo controller + trajectory planner")
    comp_ctrl.add_property("payload_capacity", 5.0, "kg", "payload_capacity", "Rated payload")
    comp_ctrl.add_property("max_speed", 2.0, "m/s", "max_speed", "Maximum TCP speed")
    components["controller"] = comp_ctrl

    comp_power = Component("power_supply", "Power Supply", "electrical",
                           "48V DC power supply unit")
    comp_power.add_property("supply_voltage", 48, "V", "supply_voltage", "Bus voltage")
    components["power_supply"] = comp_power

    # ═══════════════════════════════════════════════════════════════
    # CONSTRAINTS
    # ═══════════════════════════════════════════════════════════════

    constraints = [
        CertConstraint(
            "iso_10218_vibration", "ISO 10218-1:2011", "5.4",
            "Vibration Limit",
            "Natural frequency must exceed 15 Hz to avoid resonance with typical industrial vibrations",
            "natural_frequency", 15.0, "min", "Hz",
        ),
        CertConstraint(
            "iso_9283_accuracy", "ISO 9283:1998", "6.2",
            "Positioning Repeatability",
            "Repeatability must be within ±0.2mm for industrial classification",
            "positioning_accuracy", 0.2, "max", "mm",
        ),
        CertConstraint(
            "motor_thermal_class", "IEC 60034-1", "8.1",
            "Motor Temperature Limit",
            "Motor temperature must not exceed thermal class B (130°C) with margin",
            "joint_temperature", 85.0, "max", "°C",
        ),
        CertConstraint(
            "iso_10218_speed", "ISO 10218-1:2011", "5.3.5",
            "Maximum TCP Speed",
            "TCP speed limited for collaborative operation (if applicable)",
            "max_speed", 2.5, "max", "m/s",
        ),
        CertConstraint(
            "tip_deflection", "Application-specific", "structural",
            "Maximum Tip Deflection",
            "Tip deflection under rated load must not exceed 1mm",
            "tip_deflection", 1.0, "max", "mm",
        ),
    ]

    return graph, components, constraints

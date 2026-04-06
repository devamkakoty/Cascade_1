"""
Visual system maps — interactive zone selector per sector/sub-type.

Each system type has a schematic layout with labeled zones. When a user
selects a zone, the system auto-populates:
  - Part location (structural role)
  - Primary load types
  - Connected subsystems / interfaces
  - Operating environment (temp, exposure)
  - Applicable constraints / standards

This replaces dropdown guessing with "point at where it sits."
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional


# ── Zone Definition ──────────────────────────────────────────────────

@dataclass
class SystemZone:
    """A zone/section within a system schematic."""
    zone_id: str
    label: str
    description: str

    # Visual placement (normalized 0-1 coordinates for plotly)
    x: float               # center x
    y: float               # center y
    width: float = 0.15    # box width
    height: float = 0.10   # box height
    color: str = "#ffe8d6"

    # Auto-populated context when this zone is selected
    location_type: str = "internal_structural_primary"
    load_types: List[str] = field(default_factory=lambda: ["combined"])
    operating_temp_c: float = 25.0
    interfaces: List[str] = field(default_factory=list)
    environment: str = "normal"  # normal, wet, submerged, vacuum, high_temp, cryogenic
    standards: List[str] = field(default_factory=list)


@dataclass
class SystemMap:
    """Complete system schematic with selectable zones."""
    system_type: str       # e.g. "industrial_arm", "warship", "quadcopter"
    label: str             # display name
    description: str
    zones: List[SystemZone] = field(default_factory=list)


# ── Sector Sub-Types ─────────────────────────────────────────────────

SECTOR_SUBTYPES = {
    "robotics": [
        ("industrial_arm", "Industrial Robot Arm (6-DOF)"),
        ("mobile_wheeled", "Mobile Robot (Wheeled / Tracked)"),
        ("quadcopter", "Aerial Drone (Quadcopter)"),
        ("fixed_wing_uav", "Fixed-Wing UAV"),
        ("underwater_rov", "Underwater ROV"),
        ("humanoid", "Humanoid / Bipedal Robot"),
    ],
    "naval": [
        ("warship", "Warship / Combatant"),
        ("submarine", "Submarine"),
        ("patrol_vessel", "Patrol Vessel / OPV"),
        ("cargo_ship", "Cargo / Container Ship"),
        ("offshore_platform", "Offshore Platform"),
    ],
    "aerospace": [
        ("commercial_aircraft", "Commercial Aircraft"),
        ("electric_aircraft", "Electric / Hybrid Aircraft"),
        ("fighter_jet", "Fighter / Military Aircraft"),
        ("helicopter", "Helicopter / Rotorcraft"),
        ("spacecraft", "Spacecraft / Satellite"),
    ],
    "automotive_ev": [
        ("passenger_ev", "Passenger EV (Car / SUV)"),
        ("commercial_ev", "Commercial EV (Bus / Truck)"),
        ("two_wheeler_ev", "Two-Wheeler EV"),
        ("autonomous_vehicle", "Autonomous Vehicle"),
    ],
}


def get_subtypes(sector: str) -> List[Tuple[str, str]]:
    """Get available sub-types for a sector."""
    return SECTOR_SUBTYPES.get(sector, [("generic", "Generic System")])


# ── System Map Definitions ───────────────────────────────────────────

def _build_industrial_arm() -> SystemMap:
    return SystemMap(
        system_type="industrial_arm",
        label="Industrial Robot Arm (6-DOF)",
        description="Articulated robot arm with base, shoulder, elbow, wrist joints and end effector.",
        zones=[
            SystemZone("base_mount", "Base / Mount", "Foundation mounting plate and base rotation mechanism",
                        x=0.5, y=0.05, width=0.25, height=0.08, color="#d4725c",
                        location_type="internal_structural_primary",
                        load_types=["static", "torsion", "vibration"],
                        operating_temp_c=30.0,
                        interfaces=["structural_frame", "electrical_harness", "mounting_hardware"],
                        standards=["ISO 10218"]),
            SystemZone("shoulder", "Shoulder Joint (J1-J2)", "Major load-bearing joints, highest torque",
                        x=0.5, y=0.2, width=0.2, height=0.08, color="#e8a87c",
                        location_type="moving_joint",
                        load_types=["cyclic_fatigue", "torsion"],
                        operating_temp_c=45.0,
                        interfaces=["structural_frame", "electrical_harness", "control_system", "sensors"],
                        standards=["ISO 10218", "IEC 60034"]),
            SystemZone("upper_arm", "Upper Arm (Link 1)", "Primary structural link between shoulder and elbow",
                        x=0.5, y=0.35, width=0.12, height=0.12, color="#ffe8d6",
                        location_type="internal_structural_primary",
                        load_types=["cyclic_fatigue", "vibration"],
                        operating_temp_c=35.0,
                        interfaces=["structural_frame", "electrical_harness"],
                        standards=["ISO 10218"]),
            SystemZone("elbow", "Elbow Joint (J3)", "Mid-arm articulation, moderate torque",
                        x=0.5, y=0.5, width=0.18, height=0.08, color="#e8a87c",
                        location_type="moving_joint",
                        load_types=["cyclic_fatigue", "torsion"],
                        operating_temp_c=40.0,
                        interfaces=["structural_frame", "electrical_harness", "control_system"],
                        standards=["ISO 10218", "IEC 60034"]),
            SystemZone("forearm", "Forearm (Link 2)", "Secondary structural link between elbow and wrist",
                        x=0.5, y=0.65, width=0.10, height=0.10, color="#ffe8d6",
                        location_type="internal_structural_secondary",
                        load_types=["cyclic_fatigue", "vibration"],
                        operating_temp_c=35.0,
                        interfaces=["structural_frame", "electrical_harness", "sensors"],
                        standards=["ISO 10218"]),
            SystemZone("wrist", "Wrist (J4-J5-J6)", "Fine positioning joints, high speed, low torque",
                        x=0.5, y=0.78, width=0.18, height=0.08, color="#e8a87c",
                        location_type="moving_joint",
                        load_types=["cyclic_fatigue", "vibration"],
                        operating_temp_c=40.0,
                        interfaces=["electrical_harness", "control_system", "sensors"],
                        standards=["ISO 9283", "IEC 60034"]),
            SystemZone("end_effector", "End Effector / Gripper", "Tool or gripper at the tip — interacts with payload",
                        x=0.5, y=0.92, width=0.22, height=0.08, color="#c5b4e3",
                        location_type="interface_boundary",
                        load_types=["impact", "cyclic_fatigue", "static"],
                        operating_temp_c=30.0,
                        interfaces=["control_system", "sensors", "human_operator", "adjacent_parts"],
                        standards=["ISO 10218", "ISO 9283"]),
            SystemZone("controller", "Controller / Electronics", "Motor drivers, PLCs, safety system",
                        x=0.15, y=0.15, width=0.18, height=0.12, color="#a8d8ea",
                        location_type="internal_non_structural",
                        load_types=["vibration", "thermal_cycling"],
                        operating_temp_c=50.0,
                        interfaces=["electrical_harness", "control_system", "software_controls", "cooling_system"],
                        standards=["IEC 60034", "ISO 13849"]),
            SystemZone("power_supply", "Power Supply / Cabling", "Power distribution, cables, connectors",
                        x=0.85, y=0.15, width=0.18, height=0.12, color="#ffd700",
                        location_type="internal_non_structural",
                        load_types=["thermal_cycling"],
                        operating_temp_c=45.0,
                        interfaces=["electrical_harness", "control_system"],
                        standards=["IEC 60034"]),
        ],
    )


def _build_mobile_wheeled() -> SystemMap:
    return SystemMap(
        system_type="mobile_wheeled",
        label="Mobile Robot (Wheeled)",
        description="Wheeled mobile platform with drive system, sensors, and payload bay.",
        zones=[
            SystemZone("chassis", "Chassis / Frame", "Main structural platform, carries all subsystems",
                        x=0.5, y=0.5, width=0.6, height=0.25, color="#ffe8d6",
                        location_type="internal_structural_primary",
                        load_types=["vibration", "impact", "static"],
                        operating_temp_c=35.0,
                        interfaces=["structural_frame", "electrical_harness", "mounting_hardware"],
                        standards=["ISO 13482"]),
            SystemZone("drive_left", "Left Drive Unit", "Motor, gearbox, wheel assembly",
                        x=0.15, y=0.75, width=0.18, height=0.15, color="#e8a87c",
                        location_type="moving_joint",
                        load_types=["cyclic_fatigue", "vibration", "impact"],
                        operating_temp_c=50.0,
                        interfaces=["structural_frame", "electrical_harness", "control_system"],
                        standards=["IEC 60034"]),
            SystemZone("drive_right", "Right Drive Unit", "Motor, gearbox, wheel assembly",
                        x=0.85, y=0.75, width=0.18, height=0.15, color="#e8a87c",
                        location_type="moving_joint",
                        load_types=["cyclic_fatigue", "vibration", "impact"],
                        operating_temp_c=50.0,
                        interfaces=["structural_frame", "electrical_harness", "control_system"],
                        standards=["IEC 60034"]),
            SystemZone("battery", "Battery Pack", "Rechargeable battery, BMS, thermal management",
                        x=0.5, y=0.75, width=0.25, height=0.12, color="#ffd700",
                        location_type="internal_structural_secondary",
                        load_types=["vibration", "thermal_cycling"],
                        operating_temp_c=40.0,
                        interfaces=["electrical_harness", "cooling_system", "software_controls"],
                        standards=["UN R100", "IEC 62619"]),
            SystemZone("sensor_suite", "Sensor Suite", "LiDAR, cameras, IMU, ultrasonic",
                        x=0.5, y=0.15, width=0.3, height=0.12, color="#a8d8ea",
                        location_type="external_exposed",
                        load_types=["vibration"],
                        operating_temp_c=35.0,
                        interfaces=["electrical_harness", "control_system", "sensors", "software_controls"],
                        standards=["ISO 13482"]),
            SystemZone("compute", "Compute / Controller", "Main compute, navigation, SLAM",
                        x=0.5, y=0.35, width=0.2, height=0.1, color="#a8d8ea",
                        location_type="internal_non_structural",
                        load_types=["vibration", "thermal_cycling"],
                        operating_temp_c=55.0,
                        interfaces=["electrical_harness", "control_system", "software_controls", "cooling_system"]),
            SystemZone("payload", "Payload Bay", "Cargo area, manipulator mount, or sensor payload",
                        x=0.5, y=0.05, width=0.25, height=0.08, color="#c5b4e3",
                        location_type="interface_boundary",
                        load_types=["static", "vibration"],
                        operating_temp_c=30.0,
                        interfaces=["structural_frame", "electrical_harness", "mounting_hardware"]),
        ],
    )


def _build_quadcopter() -> SystemMap:
    return SystemMap(
        system_type="quadcopter",
        label="Aerial Drone (Quadcopter)",
        description="Multi-rotor UAV with 4 motor arms, flight controller, and payload.",
        zones=[
            SystemZone("center_body", "Center Body / Fuselage", "Main structural hub, electronics bay",
                        x=0.5, y=0.5, width=0.2, height=0.2, color="#ffe8d6",
                        location_type="internal_structural_primary",
                        load_types=["vibration", "impact"],
                        operating_temp_c=35.0,
                        interfaces=["structural_frame", "electrical_harness", "control_system"],
                        standards=["EASA Specific Category"]),
            SystemZone("arm_fl", "Front-Left Arm + Motor", "Motor arm, ESC, propeller",
                        x=0.2, y=0.2, width=0.18, height=0.12, color="#e8a87c",
                        location_type="external_exposed",
                        load_types=["vibration", "cyclic_fatigue"],
                        operating_temp_c=45.0,
                        interfaces=["structural_frame", "electrical_harness"],
                        environment="normal"),
            SystemZone("arm_fr", "Front-Right Arm + Motor", "Motor arm, ESC, propeller",
                        x=0.8, y=0.2, width=0.18, height=0.12, color="#e8a87c",
                        location_type="external_exposed",
                        load_types=["vibration", "cyclic_fatigue"],
                        operating_temp_c=45.0,
                        interfaces=["structural_frame", "electrical_harness"]),
            SystemZone("arm_rl", "Rear-Left Arm + Motor", "Motor arm, ESC, propeller",
                        x=0.2, y=0.8, width=0.18, height=0.12, color="#e8a87c",
                        location_type="external_exposed",
                        load_types=["vibration", "cyclic_fatigue"],
                        operating_temp_c=45.0,
                        interfaces=["structural_frame", "electrical_harness"]),
            SystemZone("arm_rr", "Rear-Right Arm + Motor", "Motor arm, ESC, propeller",
                        x=0.8, y=0.8, width=0.18, height=0.12, color="#e8a87c",
                        location_type="external_exposed",
                        load_types=["vibration", "cyclic_fatigue"],
                        operating_temp_c=45.0,
                        interfaces=["structural_frame", "electrical_harness"]),
            SystemZone("flight_controller", "Flight Controller", "Autopilot, IMU, GPS, barometer",
                        x=0.5, y=0.35, width=0.15, height=0.08, color="#a8d8ea",
                        location_type="internal_non_structural",
                        load_types=["vibration"],
                        operating_temp_c=50.0,
                        interfaces=["electrical_harness", "control_system", "sensors", "software_controls"]),
            SystemZone("battery_drone", "Battery", "LiPo pack, voltage regulator",
                        x=0.5, y=0.65, width=0.18, height=0.08, color="#ffd700",
                        location_type="internal_structural_secondary",
                        load_types=["vibration", "impact", "thermal_cycling"],
                        operating_temp_c=40.0,
                        interfaces=["electrical_harness", "structural_frame"]),
            SystemZone("payload_drone", "Payload / Camera Gimbal", "Camera, sensor, delivery mechanism",
                        x=0.5, y=0.88, width=0.2, height=0.1, color="#c5b4e3",
                        location_type="external_exposed",
                        load_types=["vibration"],
                        operating_temp_c=30.0,
                        interfaces=["electrical_harness", "mounting_hardware", "control_system"]),
            SystemZone("landing_gear", "Landing Gear", "Legs, skids, shock absorption",
                        x=0.5, y=0.98, width=0.4, height=0.04, color="#d4725c",
                        location_type="external_exposed",
                        load_types=["impact", "static"],
                        operating_temp_c=25.0,
                        interfaces=["structural_frame", "mounting_hardware"]),
        ],
    )


def _build_underwater_rov() -> SystemMap:
    return SystemMap(
        system_type="underwater_rov",
        label="Underwater ROV",
        description="Remotely operated vehicle for subsea inspection, maintenance, and intervention.",
        zones=[
            SystemZone("hull_rov", "Pressure Hull", "Sealed electronics housing, rated to depth",
                        x=0.5, y=0.4, width=0.3, height=0.2, color="#ffe8d6",
                        location_type="external_pressurized",
                        load_types=["pressure", "static"],
                        operating_temp_c=10.0,
                        interfaces=["structural_frame", "electrical_harness", "seals_gaskets"],
                        environment="submerged",
                        standards=["DNV-OS-E301"]),
            SystemZone("thruster_fwd", "Forward Thrusters", "Horizontal propulsion, vectored thrust",
                        x=0.2, y=0.3, width=0.15, height=0.1, color="#e8a87c",
                        location_type="external_submerged",
                        load_types=["cyclic_fatigue", "vibration"],
                        operating_temp_c=8.0,
                        interfaces=["structural_frame", "electrical_harness", "propulsion"],
                        environment="submerged"),
            SystemZone("thruster_aft", "Aft Thrusters", "Lateral and vertical positioning",
                        x=0.8, y=0.3, width=0.15, height=0.1, color="#e8a87c",
                        location_type="external_submerged",
                        load_types=["cyclic_fatigue", "vibration"],
                        operating_temp_c=8.0,
                        interfaces=["structural_frame", "electrical_harness", "propulsion"],
                        environment="submerged"),
            SystemZone("thruster_vert", "Vertical Thrusters", "Depth control",
                        x=0.5, y=0.15, width=0.2, height=0.08, color="#e8a87c",
                        location_type="external_submerged",
                        load_types=["cyclic_fatigue", "vibration"],
                        operating_temp_c=8.0,
                        interfaces=["structural_frame", "electrical_harness", "propulsion"],
                        environment="submerged"),
            SystemZone("manipulator", "Manipulator Arm", "Subsea intervention, cutting, gripping",
                        x=0.5, y=0.7, width=0.25, height=0.1, color="#c5b4e3",
                        location_type="external_submerged",
                        load_types=["cyclic_fatigue", "impact", "static"],
                        operating_temp_c=8.0,
                        interfaces=["structural_frame", "hydraulic_lines", "control_system", "sensors"],
                        environment="submerged"),
            SystemZone("camera_lights", "Cameras & Lights", "Inspection cameras, LED arrays, sonar",
                        x=0.5, y=0.55, width=0.3, height=0.06, color="#a8d8ea",
                        location_type="external_submerged",
                        load_types=["pressure"],
                        operating_temp_c=8.0,
                        interfaces=["electrical_harness", "sensors", "seals_gaskets"],
                        environment="submerged"),
            SystemZone("tether", "Tether / Umbilical", "Power, data, fiber optic connection to surface",
                        x=0.5, y=0.05, width=0.1, height=0.08, color="#ffd700",
                        location_type="interface_boundary",
                        load_types=["cyclic_fatigue", "static"],
                        operating_temp_c=10.0,
                        interfaces=["electrical_harness", "external_environment"],
                        environment="submerged"),
            SystemZone("buoyancy", "Buoyancy / Flotation", "Syntactic foam, trim system",
                        x=0.5, y=0.85, width=0.4, height=0.06, color="#90ee90",
                        location_type="external_submerged",
                        load_types=["pressure"],
                        operating_temp_c=8.0,
                        interfaces=["structural_frame"],
                        environment="submerged"),
        ],
    )


def _build_warship() -> SystemMap:
    return SystemMap(
        system_type="warship",
        label="Warship / Combatant",
        description="Surface combatant with hull, superstructure, propulsion, weapons, and mission systems.",
        zones=[
            SystemZone("bow", "Bow Section", "Forward hull, sonar dome, anchor handling",
                        x=0.12, y=0.5, width=0.12, height=0.2, color="#ffe8d6",
                        location_type="external_exposed",
                        load_types=["impact", "cyclic_fatigue", "pressure"],
                        operating_temp_c=35.0,
                        interfaces=["structural_frame", "external_environment", "sensors"],
                        environment="wet",
                        standards=["DNV GL"]),
            SystemZone("fwd_hull", "Forward Hull / Accommodation", "Crew quarters, stores, forward systems",
                        x=0.28, y=0.5, width=0.12, height=0.25, color="#ffe8d6",
                        location_type="internal_structural_primary",
                        load_types=["cyclic_fatigue", "vibration"],
                        operating_temp_c=30.0,
                        interfaces=["structural_frame", "cooling_system", "electrical_harness"],
                        standards=["DNV GL"]),
            SystemZone("midships", "Midships / Mission Bay", "Combat systems, magazines, mission equipment",
                        x=0.45, y=0.5, width=0.12, height=0.3, color="#d4725c",
                        location_type="internal_structural_primary",
                        load_types=["impact", "vibration", "static"],
                        operating_temp_c=35.0,
                        interfaces=["structural_frame", "electrical_harness", "cooling_system", "hydraulic_lines"],
                        standards=["DNV GL"]),
            SystemZone("engine_room", "Engine Room", "Main engines, generators, auxiliary machinery",
                        x=0.62, y=0.5, width=0.12, height=0.3, color="#ffd700",
                        location_type="high_vibration_zone",
                        load_types=["vibration", "thermal_cycling", "cyclic_fatigue"],
                        operating_temp_c=55.0,
                        interfaces=["structural_frame", "propulsion", "fuel_system", "cooling_system", "electrical_harness"],
                        standards=["DNV GL", "SOLAS"]),
            SystemZone("aft_hull", "Aft Hull / Steering", "Rudder, stern tube, aft peak",
                        x=0.78, y=0.5, width=0.12, height=0.2, color="#ffe8d6",
                        location_type="external_exposed",
                        load_types=["cyclic_fatigue", "vibration"],
                        operating_temp_c=35.0,
                        interfaces=["structural_frame", "propulsion", "hydraulic_lines", "external_environment"],
                        environment="wet",
                        standards=["DNV GL"]),
            SystemZone("superstructure", "Superstructure / Bridge", "Bridge, communications, radar, mast",
                        x=0.45, y=0.15, width=0.2, height=0.15, color="#a8d8ea",
                        location_type="external_exposed",
                        load_types=["vibration", "static"],
                        operating_temp_c=30.0,
                        interfaces=["structural_frame", "electrical_harness", "sensors", "control_system"],
                        standards=["DNV GL"]),
            SystemZone("weapons", "Weapons Systems", "Gun mount, missile launchers, CIWS",
                        x=0.3, y=0.2, width=0.12, height=0.1, color="#c5b4e3",
                        location_type="external_exposed",
                        load_types=["impact", "vibration", "static"],
                        operating_temp_c=40.0,
                        interfaces=["structural_frame", "electrical_harness", "control_system", "hydraulic_lines"],
                        standards=["DNV GL", "NATO STANAG"]),
            SystemZone("below_waterline", "Below Waterline", "Keel, sea chests, hull fittings, shaft",
                        x=0.5, y=0.85, width=0.5, height=0.1, color="#90c8e8",
                        location_type="external_submerged",
                        load_types=["pressure", "cyclic_fatigue"],
                        operating_temp_c=15.0,
                        interfaces=["structural_frame", "propulsion", "external_environment", "seals_gaskets"],
                        environment="submerged",
                        standards=["DNV GL"]),
        ],
    )


def _build_electric_aircraft() -> SystemMap:
    return SystemMap(
        system_type="electric_aircraft",
        label="Electric / Hybrid Aircraft",
        description="Fixed-wing electric aircraft with battery, motors, and conventional airframe.",
        zones=[
            SystemZone("nose", "Nose / Cockpit", "Windshield, avionics, forward pressure bulkhead",
                        x=0.08, y=0.5, width=0.1, height=0.15, color="#a8d8ea",
                        location_type="external_pressurized",
                        load_types=["pressure", "impact", "thermal_cycling"],
                        operating_temp_c=25.0,
                        interfaces=["structural_frame", "electrical_harness", "control_system", "sensors"],
                        standards=["FAR 25", "CS-25"]),
            SystemZone("fwd_fuselage", "Forward Fuselage", "Cabin, passengers, floor structure",
                        x=0.25, y=0.5, width=0.12, height=0.18, color="#ffe8d6",
                        location_type="internal_structural_primary",
                        load_types=["pressure", "cyclic_fatigue"],
                        operating_temp_c=22.0,
                        interfaces=["structural_frame", "thermal_management", "electrical_harness"],
                        standards=["FAR 25"]),
            SystemZone("wing_root", "Wing Root / Center Box", "Wing-fuselage joint, main spar carry-through",
                        x=0.4, y=0.5, width=0.1, height=0.12, color="#d4725c",
                        location_type="internal_structural_primary",
                        load_types=["cyclic_fatigue", "static"],
                        operating_temp_c=30.0,
                        interfaces=["structural_frame"],
                        standards=["FAR 25.305"]),
            SystemZone("wing", "Wing Structure", "Spars, ribs, skin panels, control surfaces",
                        x=0.4, y=0.2, width=0.25, height=0.1, color="#ffe8d6",
                        location_type="external_exposed",
                        load_types=["cyclic_fatigue", "vibration", "static"],
                        operating_temp_c=0.0,
                        interfaces=["structural_frame", "electrical_harness", "fuel_system"],
                        standards=["FAR 25.305", "FAR 25.571"]),
            SystemZone("battery_bay", "Battery Bay", "Battery packs, BMS, thermal management",
                        x=0.5, y=0.7, width=0.2, height=0.12, color="#ffd700",
                        location_type="internal_structural_secondary",
                        load_types=["vibration", "thermal_cycling", "impact"],
                        operating_temp_c=35.0,
                        interfaces=["electrical_harness", "cooling_system", "structural_frame", "software_controls"],
                        standards=["FAR 25.863", "DO-311A"]),
            SystemZone("aft_fuselage", "Aft Fuselage", "Tail cone, empennage attachment, APU bay",
                        x=0.7, y=0.5, width=0.12, height=0.15, color="#ffe8d6",
                        location_type="internal_structural_primary",
                        load_types=["cyclic_fatigue", "vibration"],
                        operating_temp_c=30.0,
                        interfaces=["structural_frame", "control_system"],
                        standards=["FAR 25"]),
            SystemZone("empennage", "Tail / Empennage", "Horizontal/vertical stabilizers, rudder, elevator",
                        x=0.88, y=0.4, width=0.12, height=0.2, color="#e8a87c",
                        location_type="external_exposed",
                        load_types=["cyclic_fatigue", "vibration"],
                        operating_temp_c=0.0,
                        interfaces=["structural_frame", "control_system"],
                        standards=["FAR 25.305"]),
            SystemZone("propulsion", "Electric Motors / Propulsion", "Motors, propellers, nacelles, ESCs",
                        x=0.35, y=0.12, width=0.15, height=0.06, color="#e8a87c",
                        location_type="high_vibration_zone",
                        load_types=["vibration", "cyclic_fatigue", "thermal_cycling"],
                        operating_temp_c=80.0,
                        interfaces=["structural_frame", "electrical_harness", "control_system", "cooling_system"],
                        standards=["FAR 33", "CS-E"]),
            SystemZone("landing_gear_ac", "Landing Gear", "Main and nose gear, shock absorbers, brakes",
                        x=0.35, y=0.85, width=0.2, height=0.1, color="#d4725c",
                        location_type="external_exposed",
                        load_types=["impact", "cyclic_fatigue", "static"],
                        operating_temp_c=25.0,
                        interfaces=["structural_frame", "hydraulic_lines", "electrical_harness"],
                        standards=["FAR 25.473"]),
        ],
    )


def _build_passenger_ev() -> SystemMap:
    return SystemMap(
        system_type="passenger_ev",
        label="Passenger EV (Car / SUV)",
        description="Battery electric vehicle with skateboard platform.",
        zones=[
            SystemZone("frunk", "Front Trunk / Crumple Zone", "Front structure, energy absorption, frunk",
                        x=0.1, y=0.5, width=0.12, height=0.2, color="#ffe8d6",
                        location_type="external_exposed",
                        load_types=["impact"],
                        operating_temp_c=25.0,
                        interfaces=["structural_frame", "cooling_system"],
                        standards=["FMVSS", "Euro NCAP"]),
            SystemZone("front_axle", "Front Axle / Motor", "Front drive unit, suspension, steering",
                        x=0.2, y=0.5, width=0.1, height=0.25, color="#e8a87c",
                        location_type="high_vibration_zone",
                        load_types=["vibration", "cyclic_fatigue", "impact"],
                        operating_temp_c=50.0,
                        interfaces=["structural_frame", "electrical_harness", "control_system", "cooling_system"],
                        standards=["IEC 60034"]),
            SystemZone("cabin", "Passenger Cabin", "Occupant space, dashboard, seats, HVAC",
                        x=0.45, y=0.35, width=0.25, height=0.15, color="#a8d8ea",
                        location_type="internal_non_structural",
                        load_types=["vibration"],
                        operating_temp_c=22.0,
                        interfaces=["thermal_management", "electrical_harness", "human_operator"],
                        standards=["FMVSS"]),
            SystemZone("battery_floor", "Battery Pack (Floor)", "Cells, modules, BMS, cooling, enclosure",
                        x=0.45, y=0.7, width=0.35, height=0.12, color="#ffd700",
                        location_type="internal_structural_primary",
                        load_types=["vibration", "impact", "thermal_cycling"],
                        operating_temp_c=35.0,
                        interfaces=["structural_frame", "cooling_system", "electrical_harness", "software_controls"],
                        standards=["UN R100", "ISO 6469", "IEC 62619"]),
            SystemZone("rear_axle", "Rear Axle / Motor", "Rear drive unit, differential, suspension",
                        x=0.75, y=0.5, width=0.1, height=0.25, color="#e8a87c",
                        location_type="high_vibration_zone",
                        load_types=["vibration", "cyclic_fatigue"],
                        operating_temp_c=55.0,
                        interfaces=["structural_frame", "electrical_harness", "control_system", "cooling_system"],
                        standards=["IEC 60034"]),
            SystemZone("trunk", "Trunk / Rear Structure", "Rear crumple zone, cargo, rear systems",
                        x=0.9, y=0.5, width=0.12, height=0.2, color="#ffe8d6",
                        location_type="internal_non_structural",
                        load_types=["impact"],
                        operating_temp_c=25.0,
                        interfaces=["structural_frame"],
                        standards=["FMVSS"]),
            SystemZone("underbody", "Underbody / Chassis", "Subframe, cross-members, protection plates",
                        x=0.45, y=0.9, width=0.5, height=0.08, color="#d4725c",
                        location_type="external_exposed",
                        load_types=["vibration", "impact", "cyclic_fatigue"],
                        operating_temp_c=30.0,
                        interfaces=["structural_frame", "external_environment"],
                        standards=["FMVSS"]),
        ],
    )


# ── Map Registry ─────────────────────────────────────────────────────

_MAP_BUILDERS = {
    "industrial_arm": _build_industrial_arm,
    "mobile_wheeled": _build_mobile_wheeled,
    "quadcopter": _build_quadcopter,
    "underwater_rov": _build_underwater_rov,
    "warship": _build_warship,
    "electric_aircraft": _build_electric_aircraft,
    "passenger_ev": _build_passenger_ev,
    # Aliases — map to closest available
    "fixed_wing_uav": _build_quadcopter,
    "humanoid": _build_industrial_arm,
    "submarine": _build_underwater_rov,
    "patrol_vessel": _build_warship,
    "cargo_ship": _build_warship,
    "offshore_platform": _build_warship,
    "commercial_aircraft": _build_electric_aircraft,
    "fighter_jet": _build_electric_aircraft,
    "helicopter": _build_quadcopter,
    "spacecraft": _build_electric_aircraft,
    "commercial_ev": _build_passenger_ev,
    "two_wheeler_ev": _build_mobile_wheeled,
    "autonomous_vehicle": _build_passenger_ev,
}


def get_system_map(subtype: str) -> Optional[SystemMap]:
    """Get a system map for a given sub-type."""
    builder = _MAP_BUILDERS.get(subtype)
    if builder:
        return builder()
    return None


def get_zone_by_id(system_map: SystemMap, zone_id: str) -> Optional[SystemZone]:
    """Get a specific zone from a system map."""
    for z in system_map.zones:
        if z.zone_id == zone_id:
            return z
    return None

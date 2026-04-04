"""
Operational data ingestion pipeline.

Connects real-time sensor/telemetry data to the cascade graph to:
  1. Update graph node values from live measurements
  2. Detect when parameters drift from design values
  3. Trigger automatic re-propagation when thresholds are crossed
  4. Build historical trend data for coupling validation

Data sources (future integrations):
  - IoT sensors (MQTT, OPC-UA)
  - SCADA systems
  - Fleet telemetry (CAN bus, ARINC 429)
  - PLM/MES systems
  - Manual inspection logs

For now: synthetic sensor data generator + ingestion pipeline stub.
"""

from __future__ import annotations

import json
import math
import os
import random
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Callable, Tuple


# ── Data Schema ─────────────────────────────────────────────────────

@dataclass
class SensorReading:
    """A single sensor measurement."""
    sensor_id: str
    timestamp: str
    value: float
    unit: str
    quality: float = 1.0      # 0-1, signal quality
    source: str = "sensor"     # "sensor", "manual", "derived"


@dataclass
class SensorConfig:
    """Configuration for a sensor mapping to graph node."""
    sensor_id: str
    node_id: str               # graph node this maps to
    description: str
    unit: str
    nominal_value: float       # design value
    warning_threshold: float   # % deviation before warning
    alarm_threshold: float     # % deviation before alarm
    sample_rate_hz: float = 1.0


@dataclass
class DriftAlert:
    """Alert when a parameter drifts from design value."""
    sensor_id: str
    node_id: str
    severity: str              # "warning", "alarm", "critical"
    current_value: float
    nominal_value: float
    deviation_pct: float
    timestamp: str
    message: str


@dataclass
class OperationalSnapshot:
    """Complete system state from operational data at a point in time."""
    timestamp: str
    readings: Dict[str, float]   # node_id → current value
    alerts: List[DriftAlert] = field(default_factory=list)
    source: str = "operational"


# ── Synthetic Sensor Data Generator ─────────────────────────────────

# Sensor configurations per sector
_SENSOR_CONFIGS = {
    "naval": [
        SensorConfig("hull_strain_01", "applied_stress", "Hull midship strain gauge", "MPa", 120, 10, 25),
        SensorConfig("hull_temp_01", "temperature_rise", "Hull plate temperature", "°C", 35, 15, 30),
        SensorConfig("engine_power_01", "propulsion_power", "Main engine power output", "kW", 2400, 10, 20),
        SensorConfig("fuel_flow_01", "fuel_consumption", "Fuel flow meter", "L/h", 320, 15, 25),
        SensorConfig("vibration_01", "fatigue_life", "Hull vibration sensor", "mm/s", 2.5, 20, 40),
        SensorConfig("speed_log", "max_speed", "GPS speed over ground", "kts", 25, 5, 15),
        SensorConfig("draft_sensor", "displacement", "Draft sensor amidships", "m", 3.2, 5, 10),
    ],
    "aerospace": [
        SensorConfig("cabin_temp_01", "cabin_temperature", "Cabin temperature sensor", "°C", 22, 15, 25),
        SensorConfig("battery_temp_01", "battery_temperature", "Battery pack temperature", "°C", 35, 10, 20),
        SensorConfig("motor_power_01", "motor_power", "Propulsion motor power", "kW", 450, 10, 20),
        SensorConfig("airspeed_01", "approach_speed", "Indicated airspeed", "kts", 130, 5, 10),
        SensorConfig("wing_strain_01", "wing_root_bending", "Wing root strain gauge", "Nm", 42000, 15, 30),
        SensorConfig("battery_soc", "battery_capacity", "Battery state of charge", "%", 100, 20, 35),
    ],
    "automotive_ev": [
        SensorConfig("cell_temp_01", "max_cell_temp", "Cell temperature sensor", "°C", 35, 10, 20),
        SensorConfig("pack_voltage", "pack_voltage", "Pack voltage sensor", "V", 400, 5, 10),
        SensorConfig("coolant_flow", "coolant_flow", "Coolant flow meter", "L/min", 8, 15, 25),
        SensorConfig("cell_ir", "cell_resistance", "Cell internal resistance", "mΩ", 1.2, 20, 40),
        SensorConfig("motor_temp", "motor_temperature", "Motor winding temperature", "°C", 80, 10, 20),
        SensorConfig("range_est", "vehicle_range", "Range estimator", "km", 350, 10, 25),
    ],
    "robotics": [
        SensorConfig("joint1_torque", "joint_torque", "Joint 1 torque sensor", "Nm", 50, 10, 25),
        SensorConfig("joint1_temp", "joint_temperature", "Joint 1 temperature", "°C", 45, 15, 30),
        SensorConfig("tip_force", "end_effector_force", "End effector force sensor", "N", 20, 10, 20),
        SensorConfig("arm_vib", "natural_frequency", "Arm vibration sensor", "Hz", 45, 10, 25),
        SensorConfig("motor_current", "power_consumption", "Motor current sensor", "A", 5.0, 15, 25),
        SensorConfig("position_error", "positioning_accuracy", "Position encoder error", "mm", 0.05, 20, 40),
    ],
}


def _generate_sensor_stream(
    config: SensorConfig,
    duration_hours: float = 24,
    degradation_rate: float = 0.001,
    anomaly_probability: float = 0.02,
    seed: int = 42,
) -> List[SensorReading]:
    """Generate a synthetic sensor time series with drift and anomalies."""
    random.seed(seed + hash(config.sensor_id))
    readings = []
    n_samples = int(duration_hours * 3600 * config.sample_rate_hz)
    # Limit to reasonable number
    n_samples = min(n_samples, 10000)

    start_time = datetime(2024, 6, 1)
    dt = timedelta(seconds=1.0 / config.sample_rate_hz)

    for i in range(n_samples):
        t = start_time + dt * i

        # Base value with gradual drift (simulating wear/aging)
        drift = config.nominal_value * degradation_rate * (i / n_samples)
        value = config.nominal_value + drift

        # Normal noise
        noise = random.gauss(0, config.nominal_value * 0.01)
        value += noise

        # Occasional anomalies
        if random.random() < anomaly_probability:
            anomaly = random.choice([
                config.nominal_value * 0.15,   # spike up
                -config.nominal_value * 0.10,  # dip down
                config.nominal_value * 0.25,   # large spike
            ])
            value += anomaly

        # Sensor quality degrades slightly over time
        quality = max(0.5, 1.0 - 0.1 * (i / n_samples) - random.uniform(0, 0.05))

        readings.append(SensorReading(
            sensor_id=config.sensor_id,
            timestamp=t.strftime("%Y-%m-%dT%H:%M:%S"),
            value=round(value, 4),
            unit=config.unit,
            quality=round(quality, 3),
            source="synthetic_sensor",
        ))

    return readings


def generate_operational_dataset(
    output_dir: str,
    sector: str = "naval",
    duration_hours: float = 24,
    seed: int = 42,
) -> Dict[str, List[SensorReading]]:
    """Generate synthetic operational sensor data for a sector."""
    os.makedirs(output_dir, exist_ok=True)

    configs = _SENSOR_CONFIGS.get(sector, _SENSOR_CONFIGS["naval"])
    all_streams = {}

    for config in configs:
        stream = _generate_sensor_stream(config, duration_hours, seed=seed)
        all_streams[config.sensor_id] = stream

    # Save
    data = {
        "sector": sector,
        "duration_hours": duration_hours,
        "sensor_configs": [
            {
                "sensor_id": c.sensor_id,
                "node_id": c.node_id,
                "description": c.description,
                "unit": c.unit,
                "nominal_value": c.nominal_value,
                "warning_threshold_pct": c.warning_threshold,
                "alarm_threshold_pct": c.alarm_threshold,
            }
            for c in configs
        ],
        "streams": {
            sid: [{"ts": r.timestamp, "v": r.value, "q": r.quality} for r in readings[-100:]]
            # Save last 100 readings per sensor (to keep file size manageable)
            for sid, readings in all_streams.items()
        },
    }

    path = os.path.join(output_dir, f"operational_{sector}.json")
    with open(path, "w") as f:
        json.dump(data, f, indent=2)

    print(f"Generated operational data for {sector}: "
          f"{len(configs)} sensors, {duration_hours}h duration")

    return all_streams


# ── Drift Detection ─────────────────────────────────────────────────

def detect_drift(
    readings: List[SensorReading],
    config: SensorConfig,
    window_size: int = 10,
) -> List[DriftAlert]:
    """Detect parameter drift from operational data."""
    alerts = []
    if len(readings) < window_size:
        return alerts

    # Moving average over window
    for i in range(window_size, len(readings)):
        window = readings[i - window_size:i]
        avg_value = sum(r.value for r in window) / window_size
        deviation_pct = abs(avg_value - config.nominal_value) / abs(config.nominal_value) * 100

        if deviation_pct > config.alarm_threshold:
            severity = "critical" if deviation_pct > config.alarm_threshold * 1.5 else "alarm"
            alerts.append(DriftAlert(
                sensor_id=config.sensor_id,
                node_id=config.node_id,
                severity=severity,
                current_value=round(avg_value, 4),
                nominal_value=config.nominal_value,
                deviation_pct=round(deviation_pct, 2),
                timestamp=readings[i].timestamp,
                message=f"{config.description}: {deviation_pct:.1f}% deviation from design "
                        f"({avg_value:.2f} vs {config.nominal_value} {config.unit})",
            ))
        elif deviation_pct > config.warning_threshold:
            alerts.append(DriftAlert(
                sensor_id=config.sensor_id,
                node_id=config.node_id,
                severity="warning",
                current_value=round(avg_value, 4),
                nominal_value=config.nominal_value,
                deviation_pct=round(deviation_pct, 2),
                timestamp=readings[i].timestamp,
                message=f"{config.description}: {deviation_pct:.1f}% drift detected",
            ))

    # Deduplicate consecutive same-severity alerts
    deduped = []
    for alert in alerts:
        if not deduped or deduped[-1].severity != alert.severity:
            deduped.append(alert)
    return deduped


def create_operational_snapshot(
    streams: Dict[str, List[SensorReading]],
    configs: List[SensorConfig],
) -> OperationalSnapshot:
    """Create a system state snapshot from latest sensor readings."""
    readings = {}
    alerts = []

    for config in configs:
        if config.sensor_id in streams:
            stream = streams[config.sensor_id]
            if stream:
                latest = stream[-1]
                readings[config.node_id] = latest.value

                # Check for drift
                drift_alerts = detect_drift(stream[-50:], config, window_size=10)
                alerts.extend(drift_alerts)

    return OperationalSnapshot(
        timestamp=datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
        readings=readings,
        alerts=alerts,
        source="synthetic",
    )


def get_sensor_configs(sector: str) -> List[SensorConfig]:
    """Get sensor configurations for a sector."""
    return _SENSOR_CONFIGS.get(sector, [])


def load_operational_dataset(path: str) -> Tuple[List[SensorConfig], Dict[str, List[SensorReading]]]:
    """Load operational data from JSON."""
    with open(path) as f:
        data = json.load(f)

    configs = [
        SensorConfig(
            sensor_id=c["sensor_id"],
            node_id=c["node_id"],
            description=c["description"],
            unit=c["unit"],
            nominal_value=c["nominal_value"],
            warning_threshold=c["warning_threshold_pct"],
            alarm_threshold=c["alarm_threshold_pct"],
        )
        for c in data["sensor_configs"]
    ]

    streams = {}
    for sid, readings_data in data.get("streams", {}).items():
        streams[sid] = [
            SensorReading(
                sensor_id=sid,
                timestamp=r["ts"],
                value=r["v"],
                quality=r.get("q", 1.0),
                unit=next((c.unit for c in configs if c.sensor_id == sid), ""),
                source="loaded",
            )
            for r in readings_data
        ]

    return configs, streams


if __name__ == "__main__":
    for sector in ["naval", "aerospace", "automotive_ev", "robotics"]:
        streams = generate_operational_dataset(
            "training_data/operational", sector=sector, duration_hours=24,
        )

        configs = get_sensor_configs(sector)
        snapshot = create_operational_snapshot(streams, configs)
        print(f"  Snapshot: {len(snapshot.readings)} readings, {len(snapshot.alerts)} alerts")

"""
Component model — the unit of design change.

A Component represents a physical part (windshield, battery cell, wing spar)
that has a set of measurable properties. When an engineer changes a component
(swap material, resize, etc.), the changed properties propagate through the
dependency graph.

This is the entry point for users: "I'm changing this component's property
from X to Y — what breaks?"

MVP data sources (per your spec):
  1. Component specs / requirements (properties, values, units)
  2. Certification constraints (FAR/CS limits tied to system-level parameters)

Future data sources (plugged in later to calibrate sensitivities):
  - CAD model properties
  - Simulation results
  - Experimental test data
  - Historic failure data
  - Operational data
  - Manufacturing constraints
"""

from __future__ import annotations
from dataclasses import dataclass, field
from cascade_predict.graph.dependency_graph import Subsystem


@dataclass
class ComponentProperty:
    """A single measurable property of a component."""
    name: str               # e.g. "thermal_conductivity"
    value: float
    unit: str
    description: str = ""
    source: str = ""        # where this value came from: "datasheet", "test", "sim", etc.


@dataclass
class Component:
    """
    A physical part in the system.

    Examples:
      - Windshield (material, thermal conductivity, solar transmittance, mass)
      - Battery cell (capacity, mass, specific energy, thermal runaway temp)
      - Wing spar (material, yield strength, cross-section area, mass)

    Each component belongs to a primary subsystem and has properties that
    map to nodes in the dependency graph.
    """
    component_id: str
    name: str
    subsystem: Subsystem
    description: str = ""
    properties: dict[str, ComponentProperty] = field(default_factory=dict)

    # Maps property names to graph node IDs
    # e.g. {"thermal_conductivity": "windshield_thermal_conductivity"}
    property_to_node: dict[str, str] = field(default_factory=dict)

    def add_property(
        self, name: str, value: float, unit: str,
        graph_node_id: str = "", description: str = "", source: str = "spec",
    ) -> None:
        self.properties[name] = ComponentProperty(
            name=name, value=value, unit=unit,
            description=description, source=source,
        )
        if graph_node_id:
            self.property_to_node[name] = graph_node_id

    def get_graph_deltas(self, changes: dict[str, float]) -> dict[str, float]:
        """
        Convert component property changes to graph node deltas.

        Args:
            changes: {property_name: new_value} for changed properties

        Returns:
            {graph_node_id: delta} for affected graph nodes
        """
        deltas = {}
        for prop_name, new_value in changes.items():
            if prop_name in self.properties and prop_name in self.property_to_node:
                old_value = self.properties[prop_name].value
                node_id = self.property_to_node[prop_name]
                deltas[node_id] = new_value - old_value
        return deltas


@dataclass
class CertConstraint:
    """
    A certification requirement that constrains a system parameter.

    Maps directly to regulatory_limit on graph nodes.
    """
    constraint_id: str
    standard: str           # e.g. "FAR 25.303", "CS 25.473"
    section: str            # e.g. "25.303"
    title: str              # e.g. "Factor of safety"
    description: str
    parameter_node_id: str  # graph node this constrains
    limit_value: float
    limit_type: str = "max"  # "max", "min", "range"
    unit: str = ""

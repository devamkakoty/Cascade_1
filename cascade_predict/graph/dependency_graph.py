"""
Cross-subsystem dependency graph.

Models an aircraft (or any complex system) as a directed graph where:
  - Nodes represent subsystem parameters (e.g. battery_mass, hvac_power_draw,
    wing_loading, cabin_temperature)
  - Edges represent physics-based coupling: when one parameter changes,
    connected parameters must change according to a transfer function

This is the core data structure for cascade prediction. A perturbation
to any node propagates through the graph following physics constraints,
exactly like the Eviation windshield → thermal → HVAC → battery → weight
→ structures cascade.

Each edge carries:
  - A transfer function: delta_output = f(delta_input, system_state)
  - Sensitivity (partial derivative ∂output/∂input at operating point)
  - Domain crossing flag (same subsystem or cross-domain)
"""

from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
import numpy as np


class Subsystem(Enum):
    """Major aircraft subsystems."""
    THERMAL = "thermal"
    ELECTRICAL = "electrical"
    STRUCTURAL = "structural"
    AERODYNAMIC = "aerodynamic"
    PROPULSION = "propulsion"
    HVAC = "hvac"
    REGULATORY = "regulatory"
    MASS = "mass"


@dataclass
class SubsystemNode:
    """A parameter within a subsystem.

    Example: node_id='battery_mass', subsystem=ELECTRICAL,
             value=800.0, unit='kg', bounds=(400, 1200)
    """
    node_id: str
    subsystem: Subsystem
    value: float
    unit: str
    description: str = ""
    bounds: tuple[float, float] = (float("-inf"), float("inf"))

    # Regulatory limits (if applicable)
    regulatory_limit: float | None = None
    regulatory_ref: str = ""  # e.g. "FAR 25.303"

    def is_violated(self) -> bool:
        """Check if current value violates bounds or regulatory limits."""
        if self.value < self.bounds[0] or self.value > self.bounds[1]:
            return True
        if self.regulatory_limit is not None and self.value > self.regulatory_limit:
            return True
        return False

    def margin(self) -> float:
        """Fractional margin to nearest limit. <0 means violated."""
        limits = [self.bounds[1]]
        if self.regulatory_limit is not None:
            limits.append(self.regulatory_limit)
        nearest = min(limits)
        if nearest == float("inf"):
            return float("inf")
        return (nearest - self.value) / abs(nearest) if nearest != 0 else float("inf")


@dataclass
class CouplingEdge:
    """Physics-based coupling between two parameters.

    The transfer function maps a change in the source parameter to
    a change in the target parameter:
        delta_target = sensitivity * delta_source  (linear approx)
    Or for nonlinear:
        new_target = transfer_fn(new_source, system_state)
    """
    source_id: str
    target_id: str
    sensitivity: float  # ∂target/∂source (linearized)
    description: str = ""
    physics_equation: str = ""  # human-readable physics law

    # For nonlinear transfer functions (optional)
    transfer_fn: callable | None = None

    # Metadata
    is_cross_domain: bool = False  # crosses subsystem boundary
    confidence: float = 1.0  # 0-1, how well-characterized is this coupling
    latency: float = 0.0  # time delay for this coupling to manifest [s]

    def propagate(self, delta_source: float, system_state: dict | None = None) -> float:
        """Compute the change in target given a change in source."""
        if self.transfer_fn is not None and system_state is not None:
            return self.transfer_fn(delta_source, system_state)
        return self.sensitivity * delta_source


class DependencyGraph:
    """
    Directed graph of subsystem parameter dependencies.

    Supports:
    - Adding nodes (parameters) and edges (couplings)
    - Querying paths between any two parameters
    - Sensitivity analysis (total derivative through chain rule)
    - Cycle detection (feedback loops are real and important!)
    - Cascade simulation from a perturbation
    """

    def __init__(self):
        self.nodes: dict[str, SubsystemNode] = {}
        self.edges: list[CouplingEdge] = []
        self._adj: dict[str, list[CouplingEdge]] = {}  # forward adjacency
        self._rev: dict[str, list[CouplingEdge]] = {}  # reverse adjacency

    def add_node(self, node: SubsystemNode) -> None:
        self.nodes[node.node_id] = node
        if node.node_id not in self._adj:
            self._adj[node.node_id] = []
            self._rev[node.node_id] = []

    def add_edge(self, edge: CouplingEdge) -> None:
        # Auto-detect cross-domain
        if edge.source_id in self.nodes and edge.target_id in self.nodes:
            src_sub = self.nodes[edge.source_id].subsystem
            tgt_sub = self.nodes[edge.target_id].subsystem
            edge.is_cross_domain = src_sub != tgt_sub

        self.edges.append(edge)
        self._adj.setdefault(edge.source_id, []).append(edge)
        self._rev.setdefault(edge.target_id, []).append(edge)

    def get_downstream(self, node_id: str) -> list[CouplingEdge]:
        """Get all edges where this node is the source."""
        return self._adj.get(node_id, [])

    def get_upstream(self, node_id: str) -> list[CouplingEdge]:
        """Get all edges where this node is the target."""
        return self._rev.get(node_id, [])

    def find_paths(
        self, source_id: str, target_id: str, max_depth: int = 10
    ) -> list[list[CouplingEdge]]:
        """Find all paths from source to target (DFS, cycle-aware)."""
        paths = []
        self._dfs_paths(source_id, target_id, [], set(), paths, max_depth)
        return paths

    def _dfs_paths(self, current, target, path, visited, all_paths, max_depth):
        if len(path) > max_depth:
            return
        if current == target and len(path) > 0:
            all_paths.append(list(path))
            return
        visited.add(current)
        for edge in self._adj.get(current, []):
            if edge.target_id not in visited:
                path.append(edge)
                self._dfs_paths(edge.target_id, target, path, visited, all_paths, max_depth)
                path.pop()
        visited.remove(current)

    def path_sensitivity(self, path: list[CouplingEdge]) -> float:
        """Total sensitivity along a path (chain rule: product of edge sensitivities)."""
        sensitivity = 1.0
        for edge in path:
            sensitivity *= edge.sensitivity
        return sensitivity

    def total_sensitivity(self, source_id: str, target_id: str) -> float:
        """Total sensitivity summing over all paths (superposition)."""
        paths = self.find_paths(source_id, target_id)
        return sum(self.path_sensitivity(p) for p in paths)

    def detect_cycles(self) -> list[list[str]]:
        """Find all feedback loops in the graph."""
        cycles = []
        for node_id in self.nodes:
            paths = self.find_paths(node_id, node_id)
            for path in paths:
                cycle = [e.source_id for e in path] + [path[-1].target_id]
                # Normalize cycle (start from smallest node_id)
                min_idx = cycle.index(min(cycle[:-1]))
                normalized = cycle[min_idx:-1] + cycle[:min_idx] + [cycle[min_idx]]
                if normalized not in cycles:
                    cycles.append(normalized)
        return cycles

    def get_subsystem_nodes(self, subsystem: Subsystem) -> list[SubsystemNode]:
        """Get all nodes belonging to a subsystem."""
        return [n for n in self.nodes.values() if n.subsystem == subsystem]

    def get_cross_domain_edges(self) -> list[CouplingEdge]:
        """Get all edges that cross subsystem boundaries."""
        return [e for e in self.edges if e.is_cross_domain]

    def get_violated_nodes(self) -> list[SubsystemNode]:
        """Get all nodes that currently violate their constraints."""
        return [n for n in self.nodes.values() if n.is_violated()]

    def summary(self) -> dict:
        """Quick stats about the graph."""
        return {
            "n_nodes": len(self.nodes),
            "n_edges": len(self.edges),
            "n_subsystems": len(set(n.subsystem for n in self.nodes.values())),
            "n_cross_domain_edges": len(self.get_cross_domain_edges()),
            "n_violated": len(self.get_violated_nodes()),
        }

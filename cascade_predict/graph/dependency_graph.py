"""
Cross-subsystem dependency graph — domain-agnostic core.

Models any complex system as a directed graph where:
  - Nodes represent system parameters with values, units, and constraints
  - Edges represent physics-based couplings between parameters
  - Subsystem labels are plain strings — defined by each domain template,
    not hardcoded in the engine

The engine doesn't know or care whether it's running on an aircraft,
an EV battery pack, a power plant, or a submarine. The domain knowledge
lives entirely in the template that builds the graph.

Each edge carries:
  - A transfer function: delta_output = f(delta_input, system_state)
  - Sensitivity (partial derivative at operating point)
  - Domain crossing flag (auto-detected from subsystem labels)
"""

from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np


@dataclass
class SubsystemNode:
    """A parameter within a subsystem.

    subsystem is a plain string — defined by the domain template.
    Examples: "thermal", "drivetrain", "hull_structure", "grid_connection"
    """
    node_id: str
    subsystem: str
    value: float
    unit: str
    description: str = ""
    bounds: tuple[float, float] = (float("-inf"), float("inf"))

    # Regulatory / constraint limits
    regulatory_limit: float | None = None
    regulatory_ref: str = ""  # e.g. "FAR 25.303", "UN ECE R100", "IEC 62619"

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

    Three ways to define the coupling (checked in order of priority):
      1. model: a PhysicsModel instance — preferred, provides compute + metadata
      2. transfer_fn: a callable for custom nonlinear behavior
      3. sensitivity: a float for simple linear ∂target/∂source

    When a model is attached, it auto-populates sensitivity, description,
    and physics_equation from the model at construction time.
    """
    source_id: str
    target_id: str
    sensitivity: float = 0.0  # ∂target/∂source (linearized)
    description: str = ""
    physics_equation: str = ""  # human-readable physics law

    # Physics model — the preferred way to define couplings
    model: object | None = None  # PhysicsModel instance

    # For custom nonlinear transfer functions (optional)
    transfer_fn: callable | None = None

    # Metadata
    is_cross_domain: bool = False  # crosses subsystem boundary
    confidence: float = 1.0  # 0-1, how well-characterized is this coupling
    latency: float = 0.0  # time delay for this coupling to manifest [s]

    def __post_init__(self):
        if self.model is not None:
            # Model provides defaults for metadata
            if self.sensitivity == 0.0:
                self.sensitivity = self.model.nominal_sensitivity()
            if not self.description:
                self.description = self.model.description
            if not self.physics_equation:
                self.physics_equation = self.model.equation

    def propagate(self, delta_source: float, system_state: dict | None = None) -> float:
        """Compute the change in target given a change in source."""
        if self.model is not None and system_state is not None:
            return self.model.compute_delta(delta_source, system_state)
        if self.transfer_fn is not None and system_state is not None:
            return self.transfer_fn(delta_source, system_state)
        return self.sensitivity * delta_source


class DependencyGraph:
    """
    Directed graph of system parameter dependencies.

    Domain-agnostic: works with any set of subsystem labels.
    """

    def __init__(self):
        self.nodes: dict[str, SubsystemNode] = {}
        self.edges: list[CouplingEdge] = []
        self._adj: dict[str, list[CouplingEdge]] = {}
        self._rev: dict[str, list[CouplingEdge]] = {}

    def add_node(self, node: SubsystemNode) -> None:
        self.nodes[node.node_id] = node
        if node.node_id not in self._adj:
            self._adj[node.node_id] = []
            self._rev[node.node_id] = []

    def add_edge(self, edge: CouplingEdge) -> None:
        # Auto-detect cross-domain from subsystem labels
        if edge.source_id in self.nodes and edge.target_id in self.nodes:
            src_sub = self.nodes[edge.source_id].subsystem
            tgt_sub = self.nodes[edge.target_id].subsystem
            edge.is_cross_domain = src_sub != tgt_sub

        self.edges.append(edge)
        self._adj.setdefault(edge.source_id, []).append(edge)
        self._rev.setdefault(edge.target_id, []).append(edge)

    def get_downstream(self, node_id: str) -> list[CouplingEdge]:
        return self._adj.get(node_id, [])

    def get_upstream(self, node_id: str) -> list[CouplingEdge]:
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
        """Total sensitivity along a path (chain rule: product of sensitivities)."""
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
                min_idx = cycle.index(min(cycle[:-1]))
                normalized = cycle[min_idx:-1] + cycle[:min_idx] + [cycle[min_idx]]
                if normalized not in cycles:
                    cycles.append(normalized)
        return cycles

    def get_subsystem_nodes(self, subsystem: str) -> list[SubsystemNode]:
        return [n for n in self.nodes.values() if n.subsystem == subsystem]

    def get_cross_domain_edges(self) -> list[CouplingEdge]:
        return [e for e in self.edges if e.is_cross_domain]

    def get_violated_nodes(self) -> list[SubsystemNode]:
        return [n for n in self.nodes.values() if n.is_violated()]

    def subsystems(self) -> list[str]:
        """All unique subsystem labels in the graph."""
        return sorted(set(n.subsystem for n in self.nodes.values()))

    def summary(self) -> dict:
        return {
            "n_nodes": len(self.nodes),
            "n_edges": len(self.edges),
            "n_subsystems": len(self.subsystems()),
            "n_cross_domain_edges": len(self.get_cross_domain_edges()),
            "n_violated": len(self.get_violated_nodes()),
        }

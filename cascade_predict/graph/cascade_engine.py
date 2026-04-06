"""
Cascade propagation engine.

Given a perturbation to one or more nodes, propagates the effects through
the dependency graph using topological ordering (or iterative relaxation
for graphs with cycles/feedback loops).

This models exactly the kind of cascade Eviation experienced:
  windshield_material → cabin_thermal_load → hvac_sizing → power_draw
  → battery_capacity → battery_mass → MTOW → wing_loading → structural_margin

Each step of the cascade:
  1. Apply perturbation to source node
  2. Compute downstream effects via coupling edges
  3. Update affected nodes
  4. Check for constraint/regulatory violations
  5. Recurse until no more propagation or convergence
"""

from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np

from .dependency_graph import DependencyGraph, SubsystemNode, CouplingEdge


@dataclass
class CascadeStep:
    """One step in a cascade propagation."""
    step_num: int
    edge: CouplingEdge
    source_node: str
    target_node: str
    source_subsystem: str
    target_subsystem: str
    delta_input: float
    delta_output: float
    new_value: float
    old_value: float
    is_cross_domain: bool
    causes_violation: bool = False
    violation_detail: str = ""


@dataclass
class CascadeResult:
    """Full result of a cascade propagation."""
    trigger_node: str
    trigger_delta: float
    steps: list[CascadeStep] = field(default_factory=list)
    initial_state: dict[str, float] = field(default_factory=dict)
    final_state: dict[str, float] = field(default_factory=dict)
    violations: list[dict] = field(default_factory=list)
    converged: bool = False
    n_iterations: int = 0

    @property
    def n_subsystems_affected(self) -> int:
        subs = set()
        for step in self.steps:
            subs.add(step.source_subsystem)
            subs.add(step.target_subsystem)
        return len(subs)

    @property
    def n_cross_domain_hops(self) -> int:
        return sum(1 for s in self.steps if s.is_cross_domain)

    @property
    def cascade_depth(self) -> int:
        """Longest chain of unique nodes affected."""
        if not self.steps:
            return 0
        # Count unique target nodes in order of first appearance
        seen = set()
        depth = 0
        for step in self.steps:
            if step.target_node not in seen:
                seen.add(step.target_node)
                depth += 1
        return depth

    @property
    def affected_nodes(self) -> list[str]:
        nodes = []
        seen = set()
        for step in self.steps:
            if step.target_node not in seen:
                seen.add(step.target_node)
                nodes.append(step.target_node)
        return nodes

    def trace_path(self) -> list[str]:
        """Return the cascade path as a list of node IDs in order."""
        path = [self.trigger_node]
        for step in self.steps:
            if step.target_node not in path:
                path.append(step.target_node)
        return path

    @property
    def max_pct_change(self) -> float:
        """Largest absolute % change in any downstream node."""
        max_pct = 0.0
        for step in self.steps:
            baseline = self.initial_state.get(step.target_node, step.old_value)
            if abs(baseline) > 1e-10:
                pct = abs(step.delta_output / baseline * 100)
                max_pct = max(max_pct, pct)
        return max_pct

    def summary(self, total_graph_nodes: int = 0) -> dict:
        from cascade_predict.cost_predictor import compute_propagation_score
        n_affected = len(self.affected_nodes)
        total = total_graph_nodes if total_graph_nodes > 0 else max(n_affected + 5, 20)
        prop_score = compute_propagation_score(
            n_affected, total,
            len(self.violations),
            self.n_cross_domain_hops,
            self.max_pct_change,
        )
        return {
            "trigger": self.trigger_node,
            "trigger_delta": self.trigger_delta,
            "nodes_affected": n_affected,
            "cascade_depth": self.cascade_depth,
            "subsystems_affected": self.n_subsystems_affected,
            "cross_domain_hops": self.n_cross_domain_hops,
            "violations": len(self.violations),
            "converged": self.converged,
            "iterations": self.n_iterations,
            "propagation_score": prop_score,
            "max_pct_change": round(self.max_pct_change, 2),
        }


class CascadeEngine:
    """
    Propagates perturbations through the dependency graph.

    Supports two modes:
    1. Single-pass (DAG): topological propagation, no iteration
    2. Iterative relaxation: for graphs with feedback loops,
       iterate until convergence or max iterations
    """

    def __init__(
        self,
        graph: DependencyGraph,
        convergence_tol: float = 1e-4,
        max_iterations: int = 50,
        damping: float = 0.7,  # for iterative: blend old and new values
    ):
        self.graph = graph
        self.convergence_tol = convergence_tol
        self.max_iterations = max_iterations
        self.damping = damping

    def propagate(
        self,
        trigger_node: str,
        trigger_delta: float,
        mode: str = "iterative",
    ) -> CascadeResult:
        """
        Propagate a perturbation from trigger_node through the graph.

        Args:
            trigger_node: ID of the node to perturb
            trigger_delta: magnitude of the perturbation (in node's units)
            mode: 'single_pass' or 'iterative'

        Returns:
            CascadeResult with full propagation trace
        """
        # Snapshot initial state
        initial_state = {nid: n.value for nid, n in self.graph.nodes.items()}

        result = CascadeResult(
            trigger_node=trigger_node,
            trigger_delta=trigger_delta,
            initial_state=dict(initial_state),
        )

        # Apply trigger
        self.graph.nodes[trigger_node].value += trigger_delta

        if mode == "single_pass":
            self._propagate_single_pass(trigger_node, trigger_delta, result)
        else:
            self._propagate_iterative(trigger_node, trigger_delta, result)

        # Record final state
        result.final_state = {nid: n.value for nid, n in self.graph.nodes.items()}

        # Check violations
        for node in self.graph.get_violated_nodes():
            result.violations.append({
                "node_id": node.node_id,
                "subsystem": node.subsystem,
                "value": node.value,
                "unit": node.unit,
                "bound_upper": node.bounds[1],
                "regulatory_limit": node.regulatory_limit,
                "regulatory_ref": node.regulatory_ref,
                "margin": node.margin(),
            })

        return result

    def _propagate_single_pass(
        self, trigger_node: str, trigger_delta: float, result: CascadeResult
    ) -> None:
        """BFS-style single-pass propagation (no cycles)."""
        queue = [(trigger_node, trigger_delta)]
        visited_edges = set()
        step_num = 0

        while queue:
            source_id, delta = queue.pop(0)

            for edge in self.graph.get_downstream(source_id):
                edge_key = (edge.source_id, edge.target_id)
                if edge_key in visited_edges:
                    continue
                visited_edges.add(edge_key)

                system_state = {nid: n.value for nid, n in self.graph.nodes.items()}
                delta_out = edge.propagate(delta, system_state)

                if abs(delta_out) < self.convergence_tol:
                    continue

                target = self.graph.nodes[edge.target_id]
                old_val = target.value
                target.value += delta_out

                step = CascadeStep(
                    step_num=step_num,
                    edge=edge,
                    source_node=edge.source_id,
                    target_node=edge.target_id,
                    source_subsystem=self.graph.nodes[edge.source_id].subsystem,
                    target_subsystem=target.subsystem,
                    delta_input=delta,
                    delta_output=delta_out,
                    old_value=old_val,
                    new_value=target.value,
                    is_cross_domain=edge.is_cross_domain,
                    causes_violation=target.is_violated(),
                    violation_detail=(
                        f"Exceeds {target.regulatory_ref} limit of {target.regulatory_limit} {target.unit}"
                        if target.is_violated() and target.regulatory_limit
                        else ""
                    ),
                )
                result.steps.append(step)
                step_num += 1

                queue.append((edge.target_id, delta_out))

        result.converged = True
        result.n_iterations = 1

    def _propagate_iterative(
        self, trigger_node: str, trigger_delta: float, result: CascadeResult
    ) -> None:
        """
        Iterative relaxation for graphs with feedback loops.

        Repeatedly sweeps the graph, updating each node based on its
        inputs, until changes fall below convergence tolerance.
        """
        # Track deltas per node
        deltas = {nid: 0.0 for nid in self.graph.nodes}
        deltas[trigger_node] = trigger_delta
        step_num = 0

        for iteration in range(self.max_iterations):
            max_change = 0.0
            new_deltas = {nid: 0.0 for nid in self.graph.nodes}

            # Propagate all current deltas forward
            for source_id, delta in deltas.items():
                if abs(delta) < self.convergence_tol:
                    continue

                for edge in self.graph.get_downstream(source_id):
                    system_state = {nid: n.value for nid, n in self.graph.nodes.items()}
                    delta_out = edge.propagate(delta, system_state)

                    if abs(delta_out) < self.convergence_tol:
                        continue

                    # Damping for stability
                    damped = self.damping * delta_out
                    new_deltas[edge.target_id] += damped

                    target = self.graph.nodes[edge.target_id]
                    old_val = target.value

                    step = CascadeStep(
                        step_num=step_num,
                        edge=edge,
                        source_node=edge.source_id,
                        target_node=edge.target_id,
                        source_subsystem=self.graph.nodes[edge.source_id].subsystem,
                        target_subsystem=target.subsystem,
                        delta_input=delta,
                        delta_output=damped,
                        old_value=old_val,
                        new_value=old_val + damped,
                        is_cross_domain=edge.is_cross_domain,
                    )
                    result.steps.append(step)
                    step_num += 1

            # Apply accumulated deltas
            for nid, d in new_deltas.items():
                if abs(d) > self.convergence_tol:
                    self.graph.nodes[nid].value += d
                    max_change = max(max_change, abs(d))

            # Check violations this iteration
            for node in self.graph.get_violated_nodes():
                for step in result.steps:
                    if step.target_node == node.node_id:
                        step.causes_violation = True
                        if node.regulatory_limit is not None:
                            step.violation_detail = (
                                f"Exceeds {node.regulatory_ref} limit "
                                f"of {node.regulatory_limit} {node.unit}"
                            )

            deltas = new_deltas
            result.n_iterations = iteration + 1

            if max_change < self.convergence_tol:
                result.converged = True
                break

    def reset_to_initial(self, result: CascadeResult) -> None:
        """Reset graph nodes to their pre-cascade values."""
        for nid, val in result.initial_state.items():
            if nid in self.graph.nodes:
                self.graph.nodes[nid].value = val

    def sensitivity_report(self, trigger_node: str) -> dict[str, float]:
        """
        Compute total sensitivity of every other node to the trigger.
        Uses the graph's path-based chain rule.
        """
        report = {}
        for nid in self.graph.nodes:
            if nid != trigger_node:
                sens = self.graph.total_sensitivity(trigger_node, nid)
                if abs(sens) > 1e-10:
                    report[nid] = sens
        return dict(sorted(report.items(), key=lambda x: abs(x[1]), reverse=True))

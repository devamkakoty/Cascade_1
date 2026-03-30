"""
Monte Carlo cascade propagation with Bayesian uncertainty.

Runs the deterministic cascade engine N times, each time sampling
edge sensitivities from their uncertainty distributions. Aggregates
results into:
  - Per-node: mean, std, percentiles (5th, 50th, 95th)
  - Violation probabilities (fraction of runs where each constraint is hit)
  - Sensitivity distributions (which edges matter most?)
"""

from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np

from cascade_predict.graph.dependency_graph import DependencyGraph, SubsystemNode, CouplingEdge
from cascade_predict.graph.cascade_engine import CascadeEngine, CascadeResult
from .uncertainty import UncertaintySpec


@dataclass
class NodeDistribution:
    """Statistical summary for a single node across MC runs."""

    node_id: str
    subsystem: str
    unit: str
    baseline: float
    samples: np.ndarray  # raw MC samples of final value

    @property
    def mean(self) -> float:
        return float(np.mean(self.samples))

    @property
    def std(self) -> float:
        return float(np.std(self.samples))

    @property
    def p5(self) -> float:
        return float(np.percentile(self.samples, 5))

    @property
    def p50(self) -> float:
        return float(np.percentile(self.samples, 50))

    @property
    def p95(self) -> float:
        return float(np.percentile(self.samples, 95))

    @property
    def mean_delta(self) -> float:
        return self.mean - self.baseline

    @property
    def ci_90(self) -> tuple[float, float]:
        """90% confidence interval."""
        return (self.p5, self.p95)


@dataclass
class ViolationProbability:
    """Probability of a specific constraint being violated."""

    node_id: str
    regulatory_ref: str
    regulatory_limit: float | None
    unit: str
    probability: float  # 0-1
    mean_value: float
    p95_value: float
    mean_margin: float  # positive = safe, negative = violated on average


@dataclass
class BayesianCascadeResult:
    """Full result of a Bayesian (Monte Carlo) cascade analysis."""

    trigger_node: str
    trigger_delta: float
    n_samples: int
    node_distributions: dict[str, NodeDistribution] = field(default_factory=dict)
    violation_probabilities: list[ViolationProbability] = field(default_factory=list)
    deterministic_result: CascadeResult | None = None

    def summary(self) -> dict:
        n_likely_violations = sum(
            1 for v in self.violation_probabilities if v.probability > 0.5
        )
        max_violation_prob = max(
            (v.probability for v in self.violation_probabilities), default=0.0
        )
        return {
            "n_samples": self.n_samples,
            "nodes_with_uncertainty": len(self.node_distributions),
            "likely_violations": n_likely_violations,
            "max_violation_probability": max_violation_prob,
        }


class BayesianCascadeEngine:
    """
    Monte Carlo wrapper around the deterministic CascadeEngine.

    For each MC sample:
      1. Rebuild a fresh graph from the template
      2. Perturb edge sensitivities by sampling from uncertainty distributions
      3. Run the deterministic cascade
      4. Record final node values
    After N samples, compute statistics and violation probabilities.
    """

    def __init__(
        self,
        build_fn,
        uncertainty_spec: UncertaintySpec,
        n_samples: int = 500,
        seed: int = 42,
    ):
        """
        Args:
            build_fn: callable returning (graph, components, constraints)
            uncertainty_spec: edge/node uncertainty definitions
            n_samples: number of Monte Carlo samples
            seed: random seed for reproducibility
        """
        self.build_fn = build_fn
        self.uncertainty = uncertainty_spec
        self.n_samples = n_samples
        self.rng = np.random.RandomState(seed)

    def propagate(
        self,
        component_id: str,
        property_name: str,
        new_value: float,
        mode: str = "single_pass",
    ) -> BayesianCascadeResult:
        """
        Run Monte Carlo cascade propagation.

        Args:
            component_id: which component to change
            property_name: which property to change
            new_value: the new property value
            mode: cascade propagation mode

        Returns:
            BayesianCascadeResult with distributions and violation probabilities
        """
        # First run: deterministic (nominal sensitivities)
        graph, components, constraints = self.build_fn()
        comp = components[component_id]
        deltas = comp.get_graph_deltas({property_name: new_value})
        if not deltas:
            return BayesianCascadeResult(
                trigger_node="", trigger_delta=0.0, n_samples=0
            )

        trigger_node = list(deltas.keys())[0]
        trigger_delta = list(deltas.values())[0]

        engine = CascadeEngine(graph)
        det_result = engine.propagate(trigger_node, trigger_delta, mode=mode)

        # Pre-sample all edge multipliers
        edge_samples = {}
        for edge_key, unc in self.uncertainty.edge_uncertainties.items():
            edge_samples[edge_key] = unc.sample(self.rng, self.n_samples)

        # Collect node IDs to track
        all_node_ids = list(det_result.initial_state.keys())

        # Storage for MC samples (node_id -> array of final values)
        mc_values = {nid: np.zeros(self.n_samples) for nid in all_node_ids}

        # Monte Carlo loop
        for i in range(self.n_samples):
            g_i, comps_i, _ = self.build_fn()

            # Perturb edge sensitivities
            for edge in g_i.edges:
                key = (edge.source_id, edge.target_id)
                if key in edge_samples:
                    multiplier = edge_samples[key][i]
                    edge.sensitivity *= multiplier
                    # If model exists, wrap compute_delta to scale output
                    if edge.model is not None:
                        original_model = edge.model
                        edge.model = _ScaledModel(original_model, multiplier)

            # Run cascade
            comp_i = comps_i[component_id]
            deltas_i = comp_i.get_graph_deltas({property_name: new_value})
            engine_i = CascadeEngine(g_i)
            result_i = engine_i.propagate(trigger_node, deltas_i[trigger_node], mode=mode)

            for nid in all_node_ids:
                mc_values[nid][i] = result_i.final_state.get(nid, det_result.initial_state[nid])

        # Build node distributions
        node_dists = {}
        for nid in all_node_ids:
            baseline = det_result.initial_state[nid]
            samples = mc_values[nid]
            # Only include nodes that actually changed
            if np.std(samples) > 1e-10 or abs(np.mean(samples) - baseline) > 1e-10:
                node = graph.nodes[nid]
                node_dists[nid] = NodeDistribution(
                    node_id=nid,
                    subsystem=node.subsystem,
                    unit=node.unit,
                    baseline=baseline,
                    samples=samples,
                )

        # Compute violation probabilities
        violation_probs = []
        for nid, node in graph.nodes.items():
            if node.regulatory_limit is None:
                continue
            if nid not in mc_values:
                continue

            samples = mc_values[nid]
            n_violated = np.sum(samples > node.regulatory_limit)
            prob = float(n_violated) / self.n_samples
            mean_val = float(np.mean(samples))
            p95_val = float(np.percentile(samples, 95))
            mean_margin = (node.regulatory_limit - mean_val) / abs(node.regulatory_limit) \
                if node.regulatory_limit != 0 else float("inf")

            violation_probs.append(ViolationProbability(
                node_id=nid,
                regulatory_ref=node.regulatory_ref,
                regulatory_limit=node.regulatory_limit,
                unit=node.unit,
                probability=prob,
                mean_value=mean_val,
                p95_value=p95_val,
                mean_margin=mean_margin,
            ))

        violation_probs.sort(key=lambda v: -v.probability)

        return BayesianCascadeResult(
            trigger_node=trigger_node,
            trigger_delta=trigger_delta,
            n_samples=self.n_samples,
            node_distributions=node_dists,
            violation_probabilities=violation_probs,
            deterministic_result=det_result,
        )


class _ScaledModel:
    """Wrapper that scales a physics model's output by a multiplier."""

    def __init__(self, original_model, multiplier: float):
        self._model = original_model
        self._multiplier = multiplier

    def compute_delta(self, delta_source: float, system_state: dict | None = None) -> float:
        return self._model.compute_delta(delta_source, system_state) * self._multiplier

    def nominal_sensitivity(self) -> float:
        return self._model.nominal_sensitivity() * self._multiplier

    @property
    def description(self):
        return self._model.description

    @property
    def equation(self):
        return self._model.equation

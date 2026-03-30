"""
Bayesian uncertainty layer for cascade prediction.

Adds probabilistic reasoning on top of the deterministic physics engine:
  - Parameter uncertainties (priors from engineering judgment, posteriors from test data)
  - Monte Carlo propagation through the dependency graph
  - Violation probabilities with confidence intervals
  - Sensitivity distributions instead of point estimates

The deterministic physics models remain the backbone. This layer samples
from uncertainty distributions on edge sensitivities and node values,
runs the cascade N times, and aggregates results into probability
distributions.
"""

from .uncertainty import ParameterUncertainty, UncertaintySpec
from .monte_carlo import BayesianCascadeEngine, BayesianCascadeResult

__all__ = [
    "ParameterUncertainty",
    "UncertaintySpec",
    "BayesianCascadeEngine",
    "BayesianCascadeResult",
]

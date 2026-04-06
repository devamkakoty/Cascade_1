"""
Cost Variance Predictor — ML-based cost overrun prediction.

Predicts how much actual cost will deviate from estimated cost for a
design change, using features extracted from the change request and
cascade results. Inspired by MDSL warship change data patterns.

Uses RandomForestRegressor trained on historical ECR data.
Outputs predicted cost variance % and risk tier (1-4).
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple

import numpy as np

try:
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    _HAS_SKLEARN = True
except ImportError:
    _HAS_SKLEARN = False


# ── Data Schema ──────────────────────────────────────────────────────

@dataclass
class ChangeRequest:
    """A design change request with features for cost prediction."""
    change_type: str           # structural, electrical, thermal, etc.
    affected_subsystems: List[str]
    change_cause: str          # customer_requirement, integration_issue, supplier_innovation, regulatory, design_optimization
    severity: str              # high, medium, low
    regulatory_involved: bool
    estimated_cost: float      # estimated cost in base currency
    propagation_score: float   # 0-1, from cascade analysis
    n_nodes_affected: int
    n_subsystems_affected: int
    n_violations: int
    n_cross_domain_hops: int
    cascade_depth: int
    max_pct_change: float      # largest % change in any downstream node
    sector: str = ""
    duration_days: int = 0     # approval pipeline duration


@dataclass
class CostPrediction:
    """Output of cost variance prediction."""
    predicted_variance_pct: float
    risk_tier: int             # 1-4
    risk_label: str            # Fast-track, Standard, Senior Review, Deep Analysis
    confidence: float          # model confidence
    feature_importance: Dict[str, float] = field(default_factory=dict)
    predicted_actual_cost: float = 0.0


# ── Risk Tiers ───────────────────────────────────────────────────────

_RISK_TIERS = [
    (15.0, 1, "Fast-track",     "Low risk — proceed with standard approval"),
    (30.0, 2, "Standard Review", "Moderate risk — standard review process"),
    (45.0, 3, "Senior Review",   "High risk — senior engineering review required"),
    (float("inf"), 4, "Deep Analysis", "Very high risk — deep cost/schedule analysis needed"),
]


def _classify_risk(variance_pct: float) -> Tuple[int, str, str]:
    for threshold, tier, label, action in _RISK_TIERS:
        if variance_pct < threshold:
            return tier, label, action
    return 4, "Deep Analysis", "Very high risk — deep cost/schedule analysis needed"


# ── Propagation Score ────────────────────────────────────────────────

def compute_propagation_score(
    n_affected: int,
    total_nodes: int,
    n_violations: int,
    n_cross_domain: int,
    max_pct_change: float,
) -> float:
    """
    Compute a 0-1 propagation severity score from cascade results.

    Factors:
    - Coverage: fraction of total graph nodes affected
    - Violation severity: number of regulatory violations
    - Cross-domain spread: how many subsystem boundaries crossed
    - Impact magnitude: largest % change in downstream parameters
    """
    if total_nodes == 0:
        return 0.0

    coverage = min(1.0, n_affected / max(total_nodes, 1))
    violation_factor = min(1.0, n_violations / 3.0)  # saturates at 3 violations
    cross_domain_factor = min(1.0, n_cross_domain / 5.0)  # saturates at 5 hops
    magnitude_factor = min(1.0, abs(max_pct_change) / 50.0)  # saturates at 50%

    # Weighted combination
    score = (
        0.25 * coverage +
        0.30 * violation_factor +
        0.20 * cross_domain_factor +
        0.25 * magnitude_factor
    )
    return round(min(1.0, max(0.0, score)), 3)


# ── Feature Engineering ─────────────────────────────────────────────

_ALL_SUBSYSTEMS = [
    "structural", "thermal", "electrical", "performance",
    "propulsion", "hull", "outfit", "weapon", "stability",
    "actuator", "end_effector", "cost", "schedule",
]

_CHANGE_TYPES = [
    "structural", "electrical", "thermal", "weapon",
    "stability", "outfit", "propulsion", "performance",
    "actuator", "integration",
]

_CHANGE_CAUSES = [
    "customer_requirement", "integration_issue",
    "supplier_innovation", "regulatory", "design_optimization",
]

_SEVERITIES = ["high", "medium", "low"]


def _encode_features(req: ChangeRequest) -> np.ndarray:
    """Convert a ChangeRequest into a feature vector."""
    features = []

    # Numeric features
    features.append(req.propagation_score)
    features.append(req.n_nodes_affected)
    features.append(req.n_subsystems_affected)
    features.append(req.n_violations)
    features.append(req.n_cross_domain_hops)
    features.append(req.cascade_depth)
    features.append(req.max_pct_change)
    features.append(req.estimated_cost)
    features.append(req.duration_days)
    features.append(1.0 if req.regulatory_involved else 0.0)

    # Subsystem binary flags
    affected_lower = [s.lower() for s in req.affected_subsystems]
    for sub in _ALL_SUBSYSTEMS:
        features.append(1.0 if sub in affected_lower else 0.0)

    # Change type one-hot (drop first)
    ct = req.change_type.lower()
    for t in _CHANGE_TYPES[1:]:  # drop first category
        features.append(1.0 if ct == t else 0.0)

    # Change cause one-hot (drop first)
    cc = req.change_cause.lower()
    for c in _CHANGE_CAUSES[1:]:  # drop first category
        features.append(1.0 if cc == c else 0.0)

    # Severity one-hot (drop first)
    sev = req.severity.lower()
    for s in _SEVERITIES[1:]:  # drop first category
        features.append(1.0 if sev == s else 0.0)

    return np.array(features, dtype=np.float64)


def _feature_names() -> List[str]:
    names = [
        "propagation_score", "n_nodes_affected", "n_subsystems_affected",
        "n_violations", "n_cross_domain_hops", "cascade_depth",
        "max_pct_change", "estimated_cost", "duration_days",
        "regulatory_involved",
    ]
    for sub in _ALL_SUBSYSTEMS:
        names.append(f"sub_{sub}")
    for t in _CHANGE_TYPES[1:]:
        names.append(f"type_{t}")
    for c in _CHANGE_CAUSES[1:]:
        names.append(f"cause_{c}")
    for s in _SEVERITIES[1:]:
        names.append(f"severity_{s}")
    return names


# ── Synthetic Training Data ──────────────────────────────────────────

def _generate_training_data(n_samples: int = 2000, seed: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate synthetic training data for cost variance prediction.

    Cost variance is driven by:
    - High propagation score → higher overrun
    - More violations → higher overrun
    - Cross-domain hops → harder to estimate accurately
    - Regulatory involvement → approval delays compound cost
    - Severity → high severity → more unknowns
    - Change cause → integration issues have highest overrun
    """
    rng = random.Random(seed)
    np_rng = np.random.RandomState(seed)

    X_list = []
    y_list = []

    sectors = ["naval", "aerospace", "automotive_ev", "robotics"]

    for _ in range(n_samples):
        sector = rng.choice(sectors)
        change_type = rng.choice(_CHANGE_TYPES)
        n_affected_subs = rng.randint(1, min(5, len(_ALL_SUBSYSTEMS)))
        affected = rng.sample(_ALL_SUBSYSTEMS, n_affected_subs)

        cause = rng.choice(_CHANGE_CAUSES)
        severity = rng.choice(_SEVERITIES)
        regulatory = rng.random() < 0.35

        n_nodes = rng.randint(3, 30)
        n_subs = len(set(affected))
        n_violations = rng.randint(0, 4)
        n_cross = rng.randint(0, min(n_subs, 6))
        depth = rng.randint(2, min(n_nodes, 12))
        max_pct = rng.uniform(1, 60)
        est_cost = rng.uniform(500, 500000)
        duration = rng.randint(14, 180)

        prop_score = compute_propagation_score(
            n_nodes, max(n_nodes + 5, 30), n_violations, n_cross, max_pct
        )

        req = ChangeRequest(
            change_type=change_type,
            affected_subsystems=affected,
            change_cause=cause,
            severity=severity,
            regulatory_involved=regulatory,
            estimated_cost=est_cost,
            propagation_score=prop_score,
            n_nodes_affected=n_nodes,
            n_subsystems_affected=n_subs,
            n_violations=n_violations,
            n_cross_domain_hops=n_cross,
            cascade_depth=depth,
            max_pct_change=max_pct,
            sector=sector,
            duration_days=duration,
        )

        features = _encode_features(req)
        X_list.append(features)

        # Generate target: cost variance % driven by features
        base_variance = 8.0  # minimum baseline overrun

        # Propagation score effect (biggest driver)
        variance = base_variance + prop_score * 25.0

        # Violations amplify overrun
        variance += n_violations * 4.5

        # Cross-domain hops add estimation difficulty
        variance += n_cross * 2.0

        # Regulatory adds approval delay costs
        if regulatory:
            variance += 5.0

        # Severity effect
        if severity == "high":
            variance += 6.0
        elif severity == "medium":
            variance += 2.0

        # Cause effect (integration issues worst)
        if cause == "integration_issue":
            variance += 7.0
        elif cause == "regulatory":
            variance += 4.0
        elif cause == "supplier_innovation":
            variance += 3.0

        # Large changes are harder to estimate
        variance += max_pct * 0.1

        # Duration effect (longer = more unknowns)
        variance += duration * 0.02

        # Add noise
        variance += np_rng.normal(0, 2.5)

        # Clamp to realistic range (IQR-style)
        variance = max(5.0, min(55.0, variance))

        y_list.append(variance)

    return np.array(X_list), np.array(y_list)


# ── Model ────────────────────────────────────────────────────────────

class CostVariancePredictor:
    """
    Predicts cost variance % for design changes using Random Forest.

    Train on historical ECR data or synthetic data, then predict
    cost overrun risk for new changes.
    """

    def __init__(self):
        self.model: Optional[RandomForestRegressor] = None
        self.is_trained = False
        self.feature_names = _feature_names()
        self.train_metrics: Dict[str, float] = {}

    def train(self, n_samples: int = 2000, seed: int = 42) -> Dict[str, float]:
        """Train on synthetic data. Returns evaluation metrics."""
        if not _HAS_SKLEARN:
            raise RuntimeError("scikit-learn is required: pip install scikit-learn")

        X, y = _generate_training_data(n_samples, seed)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=seed,
        )

        self.model = RandomForestRegressor(
            n_estimators=50,
            random_state=seed,
            n_jobs=-1,
            max_depth=12,
            min_samples_leaf=5,
        )
        self.model.fit(X_train, y_train)

        y_pred = self.model.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        mape = np.mean(np.abs((y_test - y_pred) / np.maximum(y_test, 1e-10))) * 100

        self.train_metrics = {
            "r2": round(r2, 4),
            "mse": round(mse, 4),
            "mae": round(mae, 4),
            "mape": round(mape, 2),
            "n_train": len(X_train),
            "n_test": len(X_test),
        }
        self.is_trained = True
        return self.train_metrics

    def predict(self, request: ChangeRequest) -> CostPrediction:
        """Predict cost variance for a single change request."""
        if not self.is_trained or self.model is None:
            # Fallback: heuristic prediction if model not trained
            return self._heuristic_predict(request)

        features = _encode_features(request).reshape(1, -1)
        variance_pct = float(self.model.predict(features)[0])
        variance_pct = max(0.0, variance_pct)

        tier, label, _action = _classify_risk(variance_pct)

        # Feature importance
        importances = dict(zip(
            self.feature_names,
            self.model.feature_importances_,
        ))
        # Keep top 10
        top_features = dict(sorted(
            importances.items(), key=lambda x: x[1], reverse=True
        )[:10])

        predicted_actual = request.estimated_cost * (1 + variance_pct / 100)

        return CostPrediction(
            predicted_variance_pct=round(variance_pct, 2),
            risk_tier=tier,
            risk_label=label,
            confidence=self.train_metrics.get("r2", 0.0),
            feature_importance=top_features,
            predicted_actual_cost=round(predicted_actual, 2),
        )

    def _heuristic_predict(self, request: ChangeRequest) -> CostPrediction:
        """Simple rule-based fallback when sklearn is not available."""
        variance = 10.0
        variance += request.propagation_score * 20.0
        variance += request.n_violations * 4.0
        variance += request.n_cross_domain_hops * 2.0
        if request.regulatory_involved:
            variance += 5.0
        if request.severity == "high":
            variance += 5.0

        tier, label, _action = _classify_risk(variance)
        predicted_actual = request.estimated_cost * (1 + variance / 100)

        return CostPrediction(
            predicted_variance_pct=round(variance, 2),
            risk_tier=tier,
            risk_label=label,
            confidence=0.5,
            predicted_actual_cost=round(predicted_actual, 2),
        )

    def get_feature_importance(self) -> Dict[str, float]:
        """Return feature importance from trained model."""
        if not self.is_trained or self.model is None:
            return {}
        return dict(zip(self.feature_names, self.model.feature_importances_))


# ── Singleton ────────────────────────────────────────────────────────

_predictor: Optional[CostVariancePredictor] = None


def get_predictor() -> CostVariancePredictor:
    """Get or create the singleton predictor, training if needed."""
    global _predictor
    if _predictor is None:
        _predictor = CostVariancePredictor()
        try:
            _predictor.train()
        except RuntimeError:
            pass  # sklearn not available, will use heuristic
    return _predictor

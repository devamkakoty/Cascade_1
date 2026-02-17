"""
Historic failure knowledge base.

Stores past failure records — what failed, why, where, and under what
conditions. When an engineer selects a component or changes a property,
the system checks against known failures and surfaces warnings.

"This inline fuse failed on the two-wheeler project because of thermal
overload under sustained high-current draw — consider before picking that."

Data sources (per user spec):
  - Historic failure data: why it failed, what failed, how, where
  - Operational data: field returns, warranty claims
  - Test data: qualification test failures

Each failure record is domain-agnostic — works across industries.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum


class Severity(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class FailureRecord:
    """A single historic failure event."""

    failure_id: str
    title: str                          # short summary
    component_type: str                 # generic: "fuse", "battery_cell", "windshield", "connector"
    failure_mode: str                   # "thermal_overload", "vibration_fatigue", "corrosion", etc.
    root_cause: str                     # why it actually failed
    description: str                    # full narrative

    # Context — when/where this happened
    product_type: str = ""              # "two_wheeler", "electric_aircraft", "ev_sedan"
    subsystem: str = ""                 # "electrical", "thermal", "structural"
    operating_conditions: str = ""      # "sustained 40A draw", "hot climate", "high vibration"
    environment: str = ""               # "tropical", "arctic", "marine", "urban"

    # What was affected
    affected_parameters: list[str] = field(default_factory=list)  # graph node IDs this relates to
    affected_properties: list[str] = field(default_factory=list)  # component property names

    # Severity and outcome
    severity: Severity = Severity.MEDIUM
    consequence: str = ""               # "field recall", "test failure", "certification delay"
    corrective_action: str = ""         # what was done to fix it

    # Traceability
    source: str = ""                    # "project_alpha_2023", "warranty_claim_4521", "DVP test 12"
    date: str = ""                      # "2023-06"
    reference: str = ""                 # document/report number

    # Matching hints — properties and thresholds that trigger this warning
    # e.g. {"current_rating": {"max": 30, "unit": "A"}} means warn if current > 30A
    trigger_conditions: dict = field(default_factory=dict)

    # Tags for flexible matching
    tags: list[str] = field(default_factory=list)


@dataclass
class FailureWarning:
    """A warning surfaced to the engineer during cascade analysis."""

    failure: FailureRecord
    match_reason: str                   # why this warning was triggered
    relevance: float = 1.0             # 0-1, how relevant to current context
    recommendation: str = ""            # what the engineer should consider


class FailureDatabase:
    """
    In-memory failure knowledge base with matching.

    Checks component selections, property changes, and cascade results
    against known failures. Returns warnings when matches are found.
    """

    def __init__(self):
        self.records: list[FailureRecord] = []

    def add(self, record: FailureRecord) -> None:
        self.records.append(record)

    def add_many(self, records: list[FailureRecord]) -> None:
        self.records.extend(records)

    def check_component(
        self,
        component_type: str,
        subsystem: str = "",
        properties: dict[str, float] | None = None,
        product_type: str = "",
        tags: list[str] | None = None,
    ) -> list[FailureWarning]:
        """
        Check if selecting this component type has known failure history.

        Args:
            component_type: generic type ("fuse", "battery_cell", "connector")
            subsystem: subsystem label
            properties: current property values to check against trigger_conditions
            product_type: current product context
            tags: additional matching tags

        Returns:
            List of FailureWarning objects, sorted by relevance (highest first)
        """
        warnings = []
        tags = tags or []

        for rec in self.records:
            match_reasons = []
            relevance = 0.0

            # Match on component type (exact or substring)
            if rec.component_type and (
                rec.component_type == component_type
                or rec.component_type in component_type
                or component_type in rec.component_type
            ):
                match_reasons.append(f"Same component type: {rec.component_type}")
                relevance += 0.5

            if not match_reasons:
                # No component match — require at least 2 tag overlaps
                # to avoid overly broad matches (e.g. just "thermal")
                overlap = set(rec.tags) & set(tags)
                if len(overlap) >= 2:
                    match_reasons.append(f"Matching tags: {', '.join(overlap)}")
                    relevance += 0.3
                else:
                    continue  # no match at all

            # Boost relevance for same subsystem
            if subsystem and rec.subsystem and rec.subsystem == subsystem:
                match_reasons.append(f"Same subsystem: {subsystem}")
                relevance += 0.2

            # Boost for same product type
            if product_type and rec.product_type and rec.product_type == product_type:
                match_reasons.append(f"Same product type: {product_type}")
                relevance += 0.2

            # Check trigger conditions against current properties
            if properties and rec.trigger_conditions:
                for prop_name, condition in rec.trigger_conditions.items():
                    if prop_name in properties:
                        val = properties[prop_name]
                        triggered = False
                        if "max" in condition and val > condition["max"]:
                            match_reasons.append(
                                f"Property {prop_name}={val} exceeds failure threshold "
                                f"{condition['max']} {condition.get('unit', '')}"
                            )
                            triggered = True
                        if "min" in condition and val < condition["min"]:
                            match_reasons.append(
                                f"Property {prop_name}={val} below failure threshold "
                                f"{condition['min']} {condition.get('unit', '')}"
                            )
                            triggered = True
                        if triggered:
                            relevance += 0.3

            relevance = min(relevance, 1.0)

            if match_reasons:
                recommendation = rec.corrective_action or "Review this failure record before proceeding."
                warnings.append(FailureWarning(
                    failure=rec,
                    match_reason="; ".join(match_reasons),
                    relevance=relevance,
                    recommendation=recommendation,
                ))

        # Sort by relevance (highest first), then severity
        severity_order = {Severity.CRITICAL: 0, Severity.HIGH: 1, Severity.MEDIUM: 2, Severity.LOW: 3}
        warnings.sort(key=lambda w: (-w.relevance, severity_order.get(w.failure.severity, 9)))
        return warnings

    def check_cascade_result(
        self,
        affected_node_ids: list[str],
        violated_node_ids: list[str] | None = None,
    ) -> list[FailureWarning]:
        """
        Check cascade results against known failures that affected the same parameters.

        Args:
            affected_node_ids: graph node IDs that changed in the cascade
            violated_node_ids: nodes that hit violations

        Returns:
            Warnings for failures that involved the same system parameters
        """
        warnings = []
        violated = set(violated_node_ids or [])

        for rec in self.records:
            if not rec.affected_parameters:
                continue

            overlap = set(rec.affected_parameters) & set(affected_node_ids)
            if not overlap:
                continue

            violation_overlap = set(rec.affected_parameters) & violated
            relevance = 0.3 + 0.1 * len(overlap)
            match_reasons = [f"Cascade affects same parameters: {', '.join(overlap)}"]

            if violation_overlap:
                match_reasons.append(f"Violations on previously-failed parameters: {', '.join(violation_overlap)}")
                relevance += 0.3

            relevance = min(relevance, 1.0)

            warnings.append(FailureWarning(
                failure=rec,
                match_reason="; ".join(match_reasons),
                relevance=relevance,
                recommendation=rec.corrective_action or "This parameter was involved in a past failure.",
            ))

        severity_order = {Severity.CRITICAL: 0, Severity.HIGH: 1, Severity.MEDIUM: 2, Severity.LOW: 3}
        warnings.sort(key=lambda w: (-w.relevance, severity_order.get(w.failure.severity, 9)))
        return warnings

    def search(self, query: str) -> list[FailureRecord]:
        """Simple text search across all fields."""
        q = query.lower()
        results = []
        for rec in self.records:
            searchable = " ".join([
                rec.title, rec.component_type, rec.failure_mode,
                rec.root_cause, rec.description, rec.product_type,
                rec.subsystem, rec.operating_conditions, rec.consequence,
                " ".join(rec.tags),
            ]).lower()
            if q in searchable:
                results.append(rec)
        return results

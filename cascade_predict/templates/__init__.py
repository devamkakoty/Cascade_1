"""
Template registry — domain-agnostic template discovery.

Each template is a module with:
  - TEMPLATE_INFO dict: name, description, industry, icon
  - build() function returning (DependencyGraph, dict[str, Component], list[CertConstraint])

The engine doesn't care what domain it's running on. Templates encode the domain.
"""

from __future__ import annotations
from typing import Callable
from cascade_predict.graph.dependency_graph import DependencyGraph
from cascade_predict.subsystems.component import Component, CertConstraint


# Type alias for a template builder function
BuildFn = Callable[[], tuple[DependencyGraph, dict[str, Component], list[CertConstraint]]]


class TemplateInfo:
    """Metadata about a registered template."""
    def __init__(
        self,
        template_id: str,
        name: str,
        description: str,
        industry: str,
        build_fn: BuildFn,
        icon: str = "",
        presets: list[dict] | None = None,
    ):
        self.template_id = template_id
        self.name = name
        self.description = description
        self.industry = industry
        self.build_fn = build_fn
        self.icon = icon
        self.presets = presets or []

    def build(self):
        return self.build_fn()


# Global registry
_REGISTRY: dict[str, TemplateInfo] = {}


def register_template(info: TemplateInfo) -> None:
    _REGISTRY[info.template_id] = info


def get_template(template_id: str) -> TemplateInfo:
    return _REGISTRY[template_id]


def list_templates() -> list[TemplateInfo]:
    return list(_REGISTRY.values())


def template_ids() -> list[str]:
    return list(_REGISTRY.keys())


# Auto-discover and register built-in templates
from cascade_predict.templates import electric_aircraft  # noqa: E402, F401
from cascade_predict.templates import ev_battery_pack  # noqa: E402, F401
from cascade_predict.templates import naval_vessel  # noqa: E402, F401
from cascade_predict.templates import robotic_arm  # noqa: E402, F401

# Global failure knowledge base (shared across all templates)
from cascade_predict.knowledge.seed_data import build_default_failure_db  # noqa: E402

_FAILURE_DB = None


def get_failure_db():
    """Get the global failure database (lazy-loaded)."""
    global _FAILURE_DB
    if _FAILURE_DB is None:
        _FAILURE_DB = build_default_failure_db()
    return _FAILURE_DB


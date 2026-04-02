"""
Cost and schedule cascade nodes — injectable into any sector template.

Adds cost ($) and time (weeks) nodes that cascade from mass, power, and
complexity changes. Every template gets these same nodes and edges so that
any design change automatically shows budget and timeline impact.

Cost model (parametric, ROM-level):
  - Material cost ∝ mass × $/kg rate
  - Manufacturing cost ∝ mass × complexity factor
  - Tooling cost: step function when geometry changes significantly
  - Certification cost: fixed + variable per violation
  - Total cost = material + manufacturing + tooling + certification

Schedule model:
  - Manufacturing lead time ∝ mass^0.6 (learning curve)
  - Certification time: fixed + variable per constraint change
  - Supplier lead time: fixed + variable per new part
  - Total schedule = max(manufacturing, supplier) + certification
"""

from __future__ import annotations

from cascade_predict.graph.dependency_graph import (
    DependencyGraph,
    SubsystemNode,
    CouplingEdge,
)
from cascade_predict.physics_models import LinearModel


# ── Sector-specific cost rates ──────────────────────────────────────

COST_RATES = {
    "aerospace": {
        "material_per_kg": 120.0,       # $/kg (aerospace-grade aluminum/composite)
        "manufacturing_per_kg": 350.0,  # $/kg (machining, layup, assembly)
        "tooling_base": 50000.0,        # $ base tooling cost
        "tooling_per_change": 15000.0,  # $ per significant geometry change
        "cert_base": 25000.0,           # $ base certification cost
        "cert_per_violation": 40000.0,  # $ per constraint that needs revalidation
        "mfg_lead_weeks": 12.0,         # baseline manufacturing lead time
        "cert_lead_weeks": 8.0,         # baseline certification lead time
        "supplier_lead_weeks": 6.0,     # baseline supplier lead time
    },
    "automotive": {
        "material_per_kg": 8.0,
        "manufacturing_per_kg": 25.0,
        "tooling_base": 200000.0,       # stamping dies are expensive
        "tooling_per_change": 50000.0,
        "cert_base": 15000.0,
        "cert_per_violation": 20000.0,
        "mfg_lead_weeks": 8.0,
        "cert_lead_weeks": 6.0,
        "supplier_lead_weeks": 10.0,
    },
    "naval": {
        "material_per_kg": 3.5,         # marine steel
        "manufacturing_per_kg": 12.0,   # welding, assembly
        "tooling_base": 30000.0,
        "tooling_per_change": 8000.0,
        "cert_base": 20000.0,           # classification society survey
        "cert_per_violation": 30000.0,
        "mfg_lead_weeks": 16.0,
        "cert_lead_weeks": 10.0,
        "supplier_lead_weeks": 12.0,
    },
}


def inject_cost_schedule_nodes(
    graph: DependencyGraph,
    sector: str,
    mass_node_id: str,
    power_node_id: str = "",
):
    """
    Add cost and schedule nodes + edges to an existing dependency graph.

    Args:
        graph: the dependency graph to augment
        sector: one of "aerospace", "automotive", "naval"
        mass_node_id: the node ID that represents total system mass (e.g. "mtow", "displacement")
        power_node_id: optional node for power/energy (e.g. "propulsion_power")
    """
    rates = COST_RATES.get(sector, COST_RATES["aerospace"])

    # ── Cost nodes ──────────────────────────────────────────────────
    graph.add_node(SubsystemNode(
        node_id="material_cost", subsystem="cost",
        value=0.0, unit="$",
        description="Material cost delta from mass change",
    ))
    graph.add_node(SubsystemNode(
        node_id="manufacturing_cost", subsystem="cost",
        value=0.0, unit="$",
        description="Manufacturing cost delta from mass/complexity change",
    ))
    graph.add_node(SubsystemNode(
        node_id="tooling_cost", subsystem="cost",
        value=rates["tooling_base"], unit="$",
        description="Tooling cost (base + geometry change surcharge)",
    ))
    graph.add_node(SubsystemNode(
        node_id="total_cost_delta", subsystem="cost",
        value=0.0, unit="$",
        description="Total cost impact of design change",
    ))

    # ── Schedule nodes ──────────────────────────────────────────────
    graph.add_node(SubsystemNode(
        node_id="manufacturing_lead_time", subsystem="schedule",
        value=rates["mfg_lead_weeks"], unit="weeks",
        description="Manufacturing lead time",
    ))
    graph.add_node(SubsystemNode(
        node_id="certification_time", subsystem="schedule",
        value=rates["cert_lead_weeks"], unit="weeks",
        description="Certification / recertification time",
    ))
    graph.add_node(SubsystemNode(
        node_id="total_schedule_delta", subsystem="schedule",
        value=0.0, unit="weeks",
        description="Total schedule impact of design change",
    ))

    # ── Cost edges ──────────────────────────────────────────────────

    # Mass → material cost
    graph.add_edge(CouplingEdge(
        source_id=mass_node_id, target_id="material_cost",
        model=LinearModel(coefficient=rates["material_per_kg"]),
        physics_equation=f"cost = delta_mass * ${rates['material_per_kg']:.0f}/kg",
        description="Material cost scales with mass change",
    ))

    # Mass → manufacturing cost
    graph.add_edge(CouplingEdge(
        source_id=mass_node_id, target_id="manufacturing_cost",
        model=LinearModel(coefficient=rates["manufacturing_per_kg"]),
        physics_equation=f"cost = delta_mass * ${rates['manufacturing_per_kg']:.0f}/kg",
        description="Manufacturing cost scales with mass change",
    ))

    # Material cost → total cost
    graph.add_edge(CouplingEdge(
        source_id="material_cost", target_id="total_cost_delta",
        model=LinearModel(coefficient=1.0),
        physics_equation="total += material_cost",
        description="Material cost feeds into total",
    ))

    # Manufacturing cost → total cost
    graph.add_edge(CouplingEdge(
        source_id="manufacturing_cost", target_id="total_cost_delta",
        model=LinearModel(coefficient=1.0),
        physics_equation="total += manufacturing_cost",
        description="Manufacturing cost feeds into total",
    ))

    # Tooling → total cost
    graph.add_edge(CouplingEdge(
        source_id="tooling_cost", target_id="total_cost_delta",
        model=LinearModel(coefficient=1.0),
        physics_equation="total += tooling_cost",
        description="Tooling cost feeds into total",
    ))

    # ── Schedule edges ──────────────────────────────────────────────

    # Mass → manufacturing lead time (heavier = longer to build)
    mass_to_time = rates["mfg_lead_weeks"] * 0.0001  # weeks per kg
    graph.add_edge(CouplingEdge(
        source_id=mass_node_id, target_id="manufacturing_lead_time",
        model=LinearModel(coefficient=mass_to_time),
        physics_equation=f"lead_time += delta_mass * {mass_to_time:.5f} weeks/kg",
        description="Heavier parts take longer to manufacture",
    ))

    # Manufacturing lead time → total schedule
    graph.add_edge(CouplingEdge(
        source_id="manufacturing_lead_time", target_id="total_schedule_delta",
        model=LinearModel(coefficient=1.0),
        physics_equation="total_schedule += mfg_lead_time",
        description="Manufacturing time feeds into total schedule",
    ))

    # Certification time → total schedule
    graph.add_edge(CouplingEdge(
        source_id="certification_time", target_id="total_schedule_delta",
        model=LinearModel(coefficient=1.0),
        physics_equation="total_schedule += cert_time",
        description="Certification time feeds into total schedule",
    ))

    # Power → manufacturing cost (if power node provided)
    if power_node_id and power_node_id in graph.nodes:
        graph.add_edge(CouplingEdge(
            source_id=power_node_id, target_id="manufacturing_cost",
            model=LinearModel(coefficient=rates["manufacturing_per_kg"] * 0.1),
            physics_equation="Higher power systems cost more to integrate",
            description="Power system complexity adds manufacturing cost",
        ))

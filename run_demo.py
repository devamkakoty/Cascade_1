"""
Quick CLI demo — runs both cascade modes and prints results.

Usage:
    python run_demo.py
"""

import numpy as np


def demo_system_cascade():
    """Demonstrate the Eviation windshield scenario."""
    from cascade_predict.subsystems import build_electric_aircraft
    from cascade_predict.graph import CascadeEngine

    print("=" * 70)
    print("  SYSTEM CASCADE — Eviation Windshield Scenario")
    print("=" * 70)
    print()
    print("Scenario: Engineer swaps windshield to a different material.")
    print("Thermal conductivity changes from 1.0 → 1.4 W/(m·K).")
    print("What breaks?")
    print()

    graph, components, constraints = build_electric_aircraft()
    engine = CascadeEngine(graph)

    # Change windshield thermal conductivity
    windshield = components["windshield"]
    deltas = windshield.get_graph_deltas({"thermal_conductivity": 1.4})
    trigger_node = list(deltas.keys())[0]
    trigger_delta = list(deltas.values())[0]

    result = engine.propagate(trigger_node, trigger_delta, mode="single_pass")
    summary = result.summary()

    print(f"--- CASCADE SUMMARY ---")
    print(f"  Parameters affected:  {summary['nodes_affected']}")
    print(f"  Subsystems hit:       {summary['subsystems_affected']}")
    print(f"  Cross-domain hops:    {summary['cross_domain_hops']}")
    print(f"  Cert violations:      {summary['violations']}")
    print()

    # Deduplicate steps
    seen = {}
    for step in result.steps:
        if step.target_node not in seen or abs(step.delta_output) > abs(seen[step.target_node].delta_output):
            seen[step.target_node] = step
    unique_steps = list(seen.values())

    print("--- CASCADE CHAIN ---")
    for step in unique_steps:
        node = graph.nodes[step.target_node]
        baseline = result.initial_state.get(step.target_node, step.old_value)
        pct = (step.delta_output / baseline * 100) if abs(baseline) > 1e-10 else 0

        flags = []
        if step.is_cross_domain:
            flags.append("CROSS-DOMAIN")
        if step.causes_violation:
            flags.append("VIOLATION")
        flag_str = f"  [{', '.join(flags)}]" if flags else ""

        print(f"  {step.source_node:30s} → {step.target_node:30s}  "
              f"Δ={step.delta_output:+10.4f} {node.unit:12s} ({pct:+.2f}%){flag_str}")

    if result.violations:
        print()
        print("--- CERTIFICATION VIOLATIONS ---")
        for v in result.violations:
            baseline = result.initial_state[v["node_id"]]
            print(f"  {v['regulatory_ref']:20s} | {v['node_id']:30s}")
            print(f"    Baseline: {baseline:.4f} {v['unit']} → After: {v['value']:.4f} {v['unit']}")
            print(f"    Limit:    {v['regulatory_limit']} {v['unit']}")

            paths = graph.find_paths(result.trigger_node, v["node_id"])
            if paths:
                shortest = min(paths, key=len)
                path_str = " → ".join(
                    [result.trigger_node] + [e.target_id for e in shortest]
                )
                print(f"    Path:     {path_str}")

    # Cleanup for next demo
    engine.reset_to_initial(result)

    # Now show a bigger change
    print()
    print("=" * 70)
    print("  What if k goes to 2.0? (even worse material choice)")
    print("=" * 70)
    graph2, components2, _ = build_electric_aircraft()
    engine2 = CascadeEngine(graph2)
    ws2 = components2["windshield"]
    deltas2 = ws2.get_graph_deltas({"thermal_conductivity": 2.0})
    result2 = engine2.propagate(list(deltas2.keys())[0], list(deltas2.values())[0], mode="single_pass")
    s2 = result2.summary()
    print(f"  Parameters affected: {s2['nodes_affected']}")
    print(f"  Violations: {s2['violations']}")
    if result2.violations:
        for v in result2.violations:
            print(f"    {v['regulatory_ref']}: {v['node_id']} = {v['value']:.2f} {v['unit']} "
                  f"(limit: {v['regulatory_limit']})")


def demo_battery_cascade():
    """Demonstrate cell-level thermal runaway cascade."""
    from cascade_predict.physics import CellParams
    from cascade_predict.simulation import CascadeSimulator

    print()
    print()
    print("=" * 70)
    print("  BATTERY THERMAL CASCADE — Cell-Level Propagation")
    print("=" * 70)

    params = CellParams()
    sim = CascadeSimulator(rows=4, cols=5, params=params)
    n_cells = sim.pack.n_cells

    print(f"\nPack: 4x5 hex layout, {n_cells} cells")

    rng = np.random.RandomState(42)
    soc = np.clip(rng.normal(0.8, 0.1, n_cells), 0.3, 1.0)

    print("Trigger: Cell 0 at 250°C")
    print("Running simulation (900s)...")

    result = sim.run_physics(
        trigger_cells=[0],
        soc_distribution=soc,
        trigger_temp=523.15,
        dt=0.5,
        t_end=900.0,
    )

    summary = sim.cascade_summary(result)
    print(f"\n--- RESULTS ---")
    print(f"  Cells affected:      {summary['cells_affected']}/{summary['total_cells']}")
    print(f"  Cascade fraction:    {summary['cascade_fraction']:.0%}")
    print(f"  Peak temperature:    {summary['max_temperature_C']:.0f}°C")
    print(f"  Avg prop delay:      {summary['avg_propagation_delay_s']:.1f}s")

    runaway_times = result["runaway_times"]
    order = np.argsort(runaway_times)
    print(f"\nRunaway order:")
    for rank, cell_id in enumerate(order):
        t = runaway_times[cell_id]
        if np.isfinite(t):
            print(f"  {rank+1:2d}. Cell {cell_id:2d} at t={t:6.1f}s  (SOC={soc[cell_id]:.2f})")
        else:
            break


if __name__ == "__main__":
    demo_system_cascade()
    demo_battery_cascade()

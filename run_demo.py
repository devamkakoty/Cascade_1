"""
Quick CLI demo — runs a cascade simulation and prints results.
No dependencies on streamlit/plotly, just numpy + matplotlib.

Usage:
    python run_demo.py
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from cascade_predict.physics import CellParams, PackGeometry
from cascade_predict.simulation import CascadeSimulator


def main():
    print("=" * 60)
    print("  Battery Cascade Prediction — Physics Demo")
    print("=" * 60)

    # Setup
    params = CellParams()  # uses calibrated defaults
    sim = CascadeSimulator(rows=4, cols=5, params=params)
    n_cells = sim.pack.n_cells

    print(f"\nPack: 4×5 hex layout, {n_cells} cells")
    print(f"Ambient: {params.t_ambient - 273.15:.0f}°C")

    # SOC distribution
    rng = np.random.RandomState(42)
    soc = np.clip(rng.normal(0.8, 0.1, n_cells), 0.3, 1.0)

    # Trigger cell 0
    trigger = [0]
    print(f"Trigger cell(s): {trigger}")
    print(f"Trigger temp: 250°C")

    # Run simulation
    print("\nRunning simulation (900s)...")
    result = sim.run_physics(
        trigger_cells=trigger,
        soc_distribution=soc,
        trigger_temp=523.15,
        dt=0.5,
        t_end=900.0,
    )

    # Summary
    summary = sim.cascade_summary(result)
    print(f"\n--- Results ---")
    print(f"Cells affected: {summary['cells_affected']}/{summary['total_cells']}")
    print(f"Cascade fraction: {summary['cascade_fraction']:.0%}")
    print(f"Peak temperature: {summary['max_temperature_C']:.0f}°C")
    print(f"Avg propagation delay: {summary['avg_propagation_delay_s']:.1f}s")

    # Runaway order
    runaway_times = result["runaway_times"]
    order = np.argsort(runaway_times)
    print(f"\nRunaway order:")
    for rank, cell_id in enumerate(order):
        t = runaway_times[cell_id]
        if np.isfinite(t):
            print(f"  {rank+1}. Cell {cell_id:2d} at t={t:6.1f}s  (SOC={soc[cell_id]:.2f})")
        else:
            break

    # Plot temperature curves
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    times = result["times"]
    temps_c = result["temperatures"] - 273.15

    for i in range(n_cells):
        color = "red" if i in trigger else ("orange" if np.isfinite(runaway_times[i]) else "steelblue")
        alpha = 0.9 if np.isfinite(runaway_times[i]) else 0.4
        ax1.plot(times, temps_c[:, i], color=color, alpha=alpha, linewidth=1.5)

    ax1.axhline(y=params.t_runaway - 273.15, color="red", linestyle="--", alpha=0.5, label="Runaway threshold")
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Temperature (°C)")
    ax1.set_title("Cell Temperature Evolution")
    ax1.legend()

    # Pack heatmap at final time
    pos = sim.pack.positions
    final_temps = temps_c[-1]
    scatter = ax2.scatter(
        pos[:, 0], pos[:, 1],
        c=final_temps, cmap="hot", s=300,
        edgecolors="black", linewidths=1.5, vmin=0, vmax=max(final_temps),
    )
    for i in range(n_cells):
        ax2.annotate(str(i), pos[i], ha="center", va="center", fontsize=8, color="white")
    plt.colorbar(scatter, ax=ax2, label="Temperature (°C)")
    ax2.set_title(f"Pack Temperature at t={times[-1]:.0f}s")
    ax2.set_aspect("equal")
    ax2.set_xticks([])
    ax2.set_yticks([])

    plt.tight_layout()
    plt.savefig("cascade_demo.png", dpi=150)
    print(f"\nPlot saved to cascade_demo.png")


if __name__ == "__main__":
    main()

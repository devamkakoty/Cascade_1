"""
Streamlit app for interactive battery cascade prediction.

Run with: streamlit run app.py
"""

import streamlit as st
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time

from cascade_predict.physics import CellParams, PackGeometry
from cascade_predict.simulation import CascadeSimulator


st.set_page_config(page_title="Battery Cascade Prediction", layout="wide")
st.title("Battery Cascade Prediction — Physics-Informed ML")
st.markdown(
    "Simulate thermal runaway cascade propagation in a battery pack. "
    "Click a cell to trigger it and watch the cascade unfold."
)

# --- Sidebar: Parameters ---
st.sidebar.header("Pack Configuration")
rows = st.sidebar.slider("Rows", 2, 8, 4)
cols = st.sidebar.slider("Columns", 2, 8, 5)

st.sidebar.header("Cell Parameters")
t_ambient_c = st.sidebar.slider("Ambient Temperature (°C)", 10, 50, 25)
soc_mean = st.sidebar.slider("Mean SOC", 0.3, 1.0, 0.8, step=0.05)
soc_spread = st.sidebar.slider("SOC Spread (±)", 0.0, 0.3, 0.1, step=0.05)

st.sidebar.header("Thermal Properties")
h_conv = st.sidebar.slider("Convection Coeff (W/m²K)", 1.0, 20.0, 2.0, step=0.5)
k_interface = st.sidebar.slider("Interface Conductivity (W/mK)", 1.0, 30.0, 10.0, step=1.0)

st.sidebar.header("Simulation")
t_end = st.sidebar.slider("Duration (s)", 120, 1200, 900, step=60)
dt = st.sidebar.select_slider("Time Step (s)", options=[0.1, 0.2, 0.5, 1.0], value=0.5)

# --- Build pack ---
params = CellParams(
    t_ambient=t_ambient_c + 273.15,
    h_conv=h_conv,
)

pack = PackGeometry(rows=rows, cols=cols, params=params)
n_cells = pack.n_cells

# --- Trigger cell selection ---
st.sidebar.header("Trigger Configuration")
trigger_mode = st.sidebar.radio("Trigger Mode", ["Single Cell", "Multiple Cells", "Random"])

if trigger_mode == "Single Cell":
    trigger_cell = st.sidebar.selectbox(
        "Trigger Cell ID", list(range(n_cells)), index=0
    )
    trigger_cells = [trigger_cell]
elif trigger_mode == "Multiple Cells":
    trigger_cells = st.sidebar.multiselect(
        "Trigger Cells", list(range(n_cells)), default=[0]
    )
elif trigger_mode == "Random":
    n_triggers = st.sidebar.slider("Number of triggers", 1, min(5, n_cells), 1)
    if st.sidebar.button("Randomize"):
        st.session_state["random_triggers"] = list(
            np.random.choice(n_cells, n_triggers, replace=False)
        )
    trigger_cells = st.session_state.get("random_triggers", [0])

trigger_temp_c = st.sidebar.slider("Trigger Temperature (°C)", 200, 400, 250, step=10)

# --- SOC distribution ---
rng = np.random.RandomState(42)
soc_distribution = np.clip(rng.normal(soc_mean, soc_spread, n_cells), 0.1, 1.0)

# --- Pack layout visualization ---
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Pack Layout")
    pos = pack.positions

    # Color cells by SOC
    colors = []
    for i in range(n_cells):
        if i in trigger_cells:
            colors.append("red")
        else:
            colors.append(f"rgb({int(50 + 200*(1-soc_distribution[i]))}, {int(50 + 200*soc_distribution[i])}, 100)")

    fig_layout = go.Figure()

    # Draw adjacency edges
    for i in range(n_cells):
        for j in range(i + 1, n_cells):
            if pack.adjacency[i, j]:
                fig_layout.add_trace(go.Scatter(
                    x=[pos[i, 0], pos[j, 0]],
                    y=[pos[i, 1], pos[j, 1]],
                    mode="lines",
                    line=dict(color="lightgray", width=1),
                    showlegend=False,
                    hoverinfo="skip",
                ))

    # Draw cells
    fig_layout.add_trace(go.Scatter(
        x=pos[:, 0], y=pos[:, 1],
        mode="markers+text",
        marker=dict(size=30, color=colors, line=dict(width=2, color="black")),
        text=[str(i) for i in range(n_cells)],
        textposition="middle center",
        textfont=dict(size=10, color="white"),
        hovertext=[
            f"Cell {i}<br>SOC: {soc_distribution[i]:.2f}<br>"
            f"{'TRIGGER' if i in trigger_cells else 'normal'}"
            for i in range(n_cells)
        ],
        hoverinfo="text",
        showlegend=False,
    ))

    fig_layout.update_layout(
        height=350,
        margin=dict(l=20, r=20, t=20, b=20),
        xaxis=dict(scaleanchor="y", scaleratio=1, showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
    )
    st.plotly_chart(fig_layout, use_container_width=True)

with col2:
    st.subheader("SOC Distribution")
    fig_soc = go.Figure(go.Bar(
        x=list(range(n_cells)),
        y=soc_distribution,
        marker_color=[colors[i] for i in range(n_cells)],
    ))
    fig_soc.update_layout(
        height=350,
        margin=dict(l=40, r=20, t=20, b=40),
        xaxis_title="Cell ID",
        yaxis_title="State of Charge",
        yaxis_range=[0, 1.1],
    )
    st.plotly_chart(fig_soc, use_container_width=True)


# --- Run simulation ---
if st.button("Run Cascade Simulation", type="primary", use_container_width=True):
    if not trigger_cells:
        st.error("Select at least one trigger cell.")
    else:
        sim = CascadeSimulator(rows=rows, cols=cols, params=params)
        sim.thermal.k_interface = k_interface

        with st.spinner("Running physics simulation..."):
            result = sim.run_physics(
                trigger_cells=trigger_cells,
                soc_distribution=soc_distribution,
                trigger_temp=trigger_temp_c + 273.15,
                dt=dt,
                t_end=t_end,
            )

        summary = sim.cascade_summary(result)

        # --- Metrics ---
        st.divider()
        st.subheader("Cascade Summary")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Cells Affected", f"{summary['cells_affected']}/{summary['total_cells']}")
        m2.metric("Cascade Fraction", f"{summary['cascade_fraction']:.0%}")
        m3.metric("Peak Temperature", f"{summary['max_temperature_C']:.0f}°C")
        m4.metric("Avg Propagation Delay", f"{summary['avg_propagation_delay_s']:.1f}s")

        # --- Temperature curves ---
        st.divider()
        st.subheader("Temperature Evolution")

        fig_temp = go.Figure()
        times = result["times"]
        temps_c = result["temperatures"] - 273.15

        # Color by runaway time
        runaway_times = result["runaway_times"]
        max_rt = np.max(runaway_times[np.isfinite(runaway_times)]) if np.any(np.isfinite(runaway_times)) else 1.0

        for i in range(n_cells):
            if i in trigger_cells:
                color = "red"
                width = 3
            elif np.isfinite(runaway_times[i]):
                frac = runaway_times[i] / max_rt
                r = int(255 * (1 - frac))
                g = int(100 * frac)
                color = f"rgb({r}, {g}, 50)"
                width = 2
            else:
                color = "lightblue"
                width = 1

            fig_temp.add_trace(go.Scatter(
                x=times, y=temps_c[:, i],
                mode="lines",
                name=f"Cell {i}",
                line=dict(color=color, width=width),
                hovertemplate=f"Cell {i}<br>T=%{{y:.1f}}°C<br>t=%{{x:.1f}}s",
            ))

        # Runaway threshold line
        fig_temp.add_hline(
            y=params.t_runaway - 273.15,
            line_dash="dash", line_color="red",
            annotation_text="Thermal Runaway Threshold",
        )

        fig_temp.update_layout(
            height=450,
            xaxis_title="Time (s)",
            yaxis_title="Temperature (°C)",
            legend=dict(orientation="h", yanchor="bottom", y=-0.3),
        )
        st.plotly_chart(fig_temp, use_container_width=True)

        # --- Cascade propagation heatmap ---
        st.subheader("Cascade Propagation Map")

        # Animate: select time step
        time_idx = st.slider(
            "Time step",
            0, len(times) - 1,
            value=min(len(times) - 1, len(times) // 2),
            format=f"t = %ds",
        )

        current_temps_c = temps_c[time_idx]
        t_min, t_max = float(temps_c.min()), float(temps_c.max())

        fig_heat = go.Figure()
        cell_colors = []
        for i in range(n_cells):
            normalized = (current_temps_c[i] - t_min) / max(t_max - t_min, 1)
            cell_colors.append(current_temps_c[i])

        fig_heat.add_trace(go.Scatter(
            x=pos[:, 0], y=pos[:, 1],
            mode="markers+text",
            marker=dict(
                size=45,
                color=cell_colors,
                colorscale="Hot",
                cmin=t_min,
                cmax=t_max,
                colorbar=dict(title="°C"),
                line=dict(width=2, color="black"),
            ),
            text=[f"{i}" for i in range(n_cells)],
            textposition="middle center",
            textfont=dict(size=9, color="white"),
            hovertext=[
                f"Cell {i}<br>T: {current_temps_c[i]:.1f}°C<br>"
                f"SOC: {soc_distribution[i]:.2f}<br>"
                f"Runaway: {'Yes' if np.isfinite(runaway_times[i]) and times[time_idx] >= runaway_times[i] else 'No'}"
                for i in range(n_cells)
            ],
            hoverinfo="text",
            showlegend=False,
        ))

        fig_heat.update_layout(
            height=400,
            title=f"Pack Temperature at t = {times[time_idx]:.1f}s",
            xaxis=dict(scaleanchor="y", scaleratio=1, showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            margin=dict(l=20, r=20, t=40, b=20),
        )
        st.plotly_chart(fig_heat, use_container_width=True)

        # --- Runaway timeline ---
        st.subheader("Runaway Timeline")
        finite_mask = np.isfinite(runaway_times)
        if np.any(finite_mask):
            order = np.argsort(runaway_times)
            timeline_cells = [i for i in order if np.isfinite(runaway_times[i])]

            fig_timeline = go.Figure(go.Bar(
                x=[runaway_times[i] for i in timeline_cells],
                y=[f"Cell {i}" for i in timeline_cells],
                orientation="h",
                marker_color=["red" if i in trigger_cells else "orange" for i in timeline_cells],
                text=[f"{runaway_times[i]:.1f}s" for i in timeline_cells],
                textposition="outside",
            ))
            fig_timeline.update_layout(
                height=max(200, 30 * len(timeline_cells)),
                xaxis_title="Time to Runaway (s)",
                margin=dict(l=80, r=40, t=20, b=40),
            )
            st.plotly_chart(fig_timeline, use_container_width=True)
        else:
            st.info("No cascade propagation — no cells reached thermal runaway threshold.")


# --- Footer ---
st.divider()
st.markdown(
    """
    **About**: This MVP simulates thermal runaway cascade propagation in lithium-ion
    battery packs using physics-based modeling (Arrhenius kinetics, conductive/convective
    heat transfer). The physics-informed neural network (PINN) learns from these
    simulations while enforcing physical constraints, enabling faster-than-realtime
    prediction for battery safety analysis.
    """
)

"""
Cascade Prediction — Interactive Dashboard.

Two modes:
  1. System Cascade: Change a component property → see the ripple effect
     across subsystems, which cert constraints get violated.
     Works with ANY template (aircraft, EV, marine, etc.)
  2. Battery Thermal: Cell-level thermal runaway cascade in a battery pack

Run with: streamlit run app.py
"""

import streamlit as st
import numpy as np
import plotly.graph_objects as go

st.set_page_config(page_title="Cascade Prediction", layout="wide")

tab_system, tab_battery = st.tabs([
    "System Cascade (Cross-Subsystem)",
    "Battery Thermal Cascade",
])

# Palette for auto-coloring subsystems (cycles if more than 10)
_PALETTE = [
    "rgba(255, 100, 100, 0.8)",  # red
    "rgba(100, 200, 255, 0.8)",  # blue
    "rgba(255, 200, 50, 0.8)",   # gold
    "rgba(100, 255, 150, 0.8)",  # green
    "rgba(200, 100, 255, 0.8)",  # purple
    "rgba(255, 150, 50, 0.8)",   # orange
    "rgba(180, 180, 180, 0.8)",  # gray
    "rgba(255, 50, 50, 0.8)",    # bright red
    "rgba(50, 200, 200, 0.8)",   # teal
    "rgba(200, 200, 100, 0.8)",  # olive
]


def _subsystem_color_map(subsystems: list[str]) -> dict[str, str]:
    """Auto-assign colors to subsystem labels."""
    return {s: _PALETTE[i % len(_PALETTE)] for i, s in enumerate(sorted(subsystems))}


# =====================================================================
# TAB 1: SYSTEM CASCADE (template-agnostic)
# =====================================================================
with tab_system:
    st.title("Design Change Cascade Prediction")
    st.markdown(
        "**Change one component property. See everything that breaks.**  \n"
        "Select a system template, pick a component, change a property, "
        "and watch the cascade ripple across subsystems."
    )

    from cascade_predict.templates import list_templates, get_template, get_failure_db
    from cascade_predict.graph import CascadeEngine

    # --- Template selector ---
    all_templates = list_templates()
    template_names = {t.template_id: f"{t.name} ({t.industry})" for t in all_templates}

    selected_template_id = st.sidebar.selectbox(
        "System Template",
        list(template_names.keys()),
        format_func=lambda x: template_names[x],
        key="tmpl_select",
    )
    tmpl = get_template(selected_template_id)
    st.sidebar.caption(tmpl.description)

    # Build system from template
    graph, components, constraints = tmpl.build()

    # --- Sidebar: Component selection ---
    st.sidebar.markdown("---")
    st.sidebar.header("Component Change")
    comp_names = {cid: c.name for cid, c in components.items()}
    selected_comp_id = st.sidebar.selectbox(
        "Select Component",
        list(comp_names.keys()),
        format_func=lambda x: comp_names[x],
    )
    comp = components[selected_comp_id]

    # Check failure DB for this component type
    failure_db = get_failure_db()
    comp_props = {p.name: p.value for p in comp.properties.values()}
    comp_warnings = failure_db.check_component(
        component_type=selected_comp_id,
        subsystem=comp.subsystem,
        properties=comp_props,
        product_type=tmpl.industry.lower(),
        tags=[selected_comp_id, comp.subsystem],
    )
    if comp_warnings:
        for w in comp_warnings:
            sev = w.failure.severity.value.upper()
            st.warning(
                f"**Historic Failure [{sev}]:** {w.failure.title}  \n"
                f"*{w.failure.root_cause[:200]}...*  \n"
                f"**Recommendation:** {w.recommendation}  \n"
                f"*Source: {w.failure.source} ({w.failure.date}) — {w.failure.reference}*"
            )

    # Show component properties that are linked to graph nodes
    editable_props = {
        name: prop for name, prop in comp.properties.items()
        if name in comp.property_to_node
    }

    if not editable_props:
        st.sidebar.warning("This component has no graph-linked properties to change.")
        selected_prop = None
    else:
        selected_prop = st.sidebar.selectbox(
            "Property to Change",
            list(editable_props.keys()),
            format_func=lambda x: f"{x} ({editable_props[x].value} {editable_props[x].unit})",
        )

    new_value = None
    if selected_prop:
        prop = editable_props[selected_prop]
        st.sidebar.markdown(f"**Current:** {prop.value} {prop.unit}")
        st.sidebar.markdown(f"*Source: {prop.source}*")

        st.sidebar.markdown("---")

        new_value = st.sidebar.number_input(
            f"New value ({prop.unit})",
            value=prop.value,
            step=abs(prop.value) * 0.05 if prop.value != 0 else 0.1,
            format="%.4f",
        )

        delta = new_value - prop.value
        if abs(delta) > 1e-10:
            pct = (delta / prop.value * 100) if prop.value != 0 else float("inf")
            st.sidebar.metric(
                "Change",
                f"{delta:+.4f} {prop.unit}",
                f"{pct:+.1f}%",
            )

    # --- Quick scenarios (loaded from template presets) ---
    if tmpl.presets:
        st.sidebar.markdown("---")
        st.sidebar.header("Quick Scenarios")
        preset_names = ["Custom (use controls above)"] + [p["name"] for p in tmpl.presets]
        scenario = st.sidebar.radio("Or try a preset:", preset_names, key="preset_radio")

        if scenario != "Custom (use controls above)":
            for p in tmpl.presets:
                if p["name"] == scenario:
                    selected_comp_id = p["component_id"]
                    selected_prop = p["property_name"]
                    new_value = p["new_value"]
                    comp = components[selected_comp_id]
                    break

    # --- Run cascade ---
    if st.button("Propagate Change", type="primary", use_container_width=True) and selected_prop and new_value is not None:
        # Rebuild fresh graph for each run
        graph, components, constraints = tmpl.build()
        comp = components[selected_comp_id]

        deltas = comp.get_graph_deltas({selected_prop: new_value})
        if not deltas:
            st.error("No graph-linked deltas for this property change.")
        else:
            engine = CascadeEngine(graph)
            trigger_node = list(deltas.keys())[0]
            trigger_delta = list(deltas.values())[0]
            result = engine.propagate(trigger_node, trigger_delta, mode="single_pass")
            summary = result.summary()

            # --- Metrics row ---
            st.divider()
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Parameters Affected", summary["nodes_affected"])
            m2.metric("Subsystems Hit", summary["subsystems_affected"])
            m3.metric("Cross-Domain Hops", summary["cross_domain_hops"])
            n_violations = summary["violations"]
            m4.metric("Violations", n_violations,
                       delta=f"{n_violations} cert issues" if n_violations > 0 else "Clean",
                       delta_color="inverse")

            # --- Cascade waterfall ---
            st.divider()
            st.subheader("Cascade Waterfall")
            st.markdown(
                "Each bar shows the change propagated to a parameter. "
                "Red = cert violation. Orange = cross-domain hop."
            )

            # Deduplicate: show only first (largest) delta per unique target node
            seen_targets = {}
            for step in result.steps:
                if step.target_node not in seen_targets:
                    seen_targets[step.target_node] = step
                else:
                    existing = seen_targets[step.target_node]
                    if abs(step.delta_output) > abs(existing.delta_output):
                        seen_targets[step.target_node] = step

            unique_steps = list(seen_targets.values())

            # Normalize deltas for visualization (% change from baseline)
            labels = []
            pct_changes = []
            colors = []
            hover_texts = []

            for step in unique_steps:
                baseline = result.initial_state.get(step.target_node, step.old_value)
                if abs(baseline) > 1e-10:
                    pct = (step.delta_output / baseline) * 100
                else:
                    pct = 0

                node = graph.nodes[step.target_node]
                label = step.target_node.replace("_", " ").title()
                labels.append(label)
                pct_changes.append(pct)

                if step.causes_violation:
                    colors.append("crimson")
                elif step.is_cross_domain:
                    colors.append("darkorange")
                else:
                    colors.append("steelblue")

                hover = (
                    f"<b>{label}</b><br>"
                    f"Subsystem: {step.target_subsystem}<br>"
                    f"Change: {step.delta_output:+.4f} {node.unit}<br>"
                    f"Before: {baseline:.4f} -> After: {step.new_value:.4f}<br>"
                    f"% Change: {pct:+.2f}%<br>"
                )
                if step.causes_violation:
                    hover += f"<b>VIOLATION: {step.violation_detail}</b><br>"
                if step.edge.physics_equation:
                    hover += f"Physics: {step.edge.physics_equation}"
                hover_texts.append(hover)

            fig_waterfall = go.Figure(go.Bar(
                x=pct_changes,
                y=labels,
                orientation="h",
                marker_color=colors,
                hovertext=hover_texts,
                hoverinfo="text",
            ))
            fig_waterfall.update_layout(
                height=max(350, 28 * len(labels)),
                xaxis_title="% Change from Baseline",
                margin=dict(l=200, r=40, t=20, b=40),
                yaxis=dict(autorange="reversed"),
            )
            fig_waterfall.add_vline(x=0, line_color="gray", line_width=1)
            st.plotly_chart(fig_waterfall, use_container_width=True)

            # --- Cascade flow (Sankey-style) ---
            st.divider()
            st.subheader("Cascade Flow")

            # Build Sankey from cascade steps
            all_node_ids = [result.trigger_node]
            for step in unique_steps:
                if step.source_node not in all_node_ids:
                    all_node_ids.append(step.source_node)
                if step.target_node not in all_node_ids:
                    all_node_ids.append(step.target_node)

            node_idx = {nid: i for i, nid in enumerate(all_node_ids)}

            # Auto-generate subsystem colors from the graph
            sub_colors = _subsystem_color_map(graph.subsystems())

            node_colors = []
            node_labels = []
            for nid in all_node_ids:
                node = graph.nodes.get(nid)
                if node:
                    node_colors.append(sub_colors.get(node.subsystem, "rgba(200,200,200,0.8)"))
                    node_labels.append(nid.replace("_", " ").title())
                else:
                    node_colors.append("rgba(200,200,200,0.8)")
                    node_labels.append(nid)

            sankey_sources = []
            sankey_targets = []
            sankey_values = []
            sankey_colors = []

            for step in unique_steps:
                if step.source_node in node_idx and step.target_node in node_idx:
                    sankey_sources.append(node_idx[step.source_node])
                    sankey_targets.append(node_idx[step.target_node])
                    sankey_values.append(max(abs(step.delta_output), 0.01))
                    if step.causes_violation:
                        sankey_colors.append("rgba(255, 0, 0, 0.5)")
                    elif step.is_cross_domain:
                        sankey_colors.append("rgba(255, 165, 0, 0.4)")
                    else:
                        sankey_colors.append("rgba(100, 149, 237, 0.3)")

            fig_sankey = go.Figure(go.Sankey(
                node=dict(
                    label=node_labels,
                    color=node_colors,
                    pad=15,
                    thickness=20,
                ),
                link=dict(
                    source=sankey_sources,
                    target=sankey_targets,
                    value=sankey_values,
                    color=sankey_colors,
                ),
            ))
            fig_sankey.update_layout(
                height=500,
                margin=dict(l=20, r=20, t=20, b=20),
            )
            st.plotly_chart(fig_sankey, use_container_width=True)

            # --- Violations detail ---
            if result.violations:
                st.divider()
                st.subheader("Certification Violations")
                for v in result.violations:
                    node = graph.nodes[v["node_id"]]
                    baseline = result.initial_state[v["node_id"]]
                    st.error(
                        f"**{v['regulatory_ref']}** — {v['node_id'].replace('_', ' ').title()}  \n"
                        f"Baseline: {baseline:.4f} {v['unit']} -> "
                        f"After cascade: **{v['value']:.4f} {v['unit']}**  \n"
                        f"Limit: {v['regulatory_limit']} {v['unit']}  \n"
                        f"Margin: **{v['margin']:.4f}** (negative = violated)"
                    )

                    # Show the path that led here
                    paths = graph.find_paths(result.trigger_node, v["node_id"])
                    if paths:
                        shortest = min(paths, key=len)
                        path_str = " -> ".join(
                            [result.trigger_node.replace("_", " ")] +
                            [e.target_id.replace("_", " ") for e in shortest]
                        )
                        st.markdown(f"*Cascade path:* `{path_str}`")

            # --- Historic failure warnings from cascade ---
            cascade_warnings = failure_db.check_cascade_result(
                affected_node_ids=result.affected_nodes,
                violated_node_ids=[v["node_id"] for v in result.violations],
            )
            if cascade_warnings:
                st.divider()
                st.subheader("Historic Failure Warnings")
                st.markdown(
                    "These past failures involved the same parameters affected by this cascade."
                )
                for w in cascade_warnings:
                    sev = w.failure.severity.value.upper()
                    sev_color = {"CRITICAL": "red", "HIGH": "orange", "MEDIUM": "blue"}.get(sev, "gray")
                    st.warning(
                        f"**[{sev}] {w.failure.title}**  \n"
                        f"{w.failure.root_cause[:300]}  \n\n"
                        f"**Why this is relevant:** {w.match_reason}  \n"
                        f"**Recommendation:** {w.recommendation}  \n"
                        f"*Product: {w.failure.product_type} | "
                        f"Source: {w.failure.source} ({w.failure.date})*"
                    )

            # --- Before/After comparison table ---
            st.divider()
            st.subheader("Parameter Comparison (Before -> After)")

            table_data = []
            for step in unique_steps:
                node = graph.nodes[step.target_node]
                baseline = result.initial_state.get(step.target_node, step.old_value)
                pct = (step.delta_output / baseline * 100) if abs(baseline) > 1e-10 else 0
                violated = "YES" if step.causes_violation else ""
                table_data.append({
                    "Parameter": step.target_node.replace("_", " ").title(),
                    "Subsystem": step.target_subsystem.upper(),
                    "Before": f"{baseline:.4f}",
                    "After": f"{step.new_value:.4f}",
                    "Delta": f"{step.delta_output:+.4f}",
                    "% Change": f"{pct:+.2f}%",
                    "Unit": node.unit,
                    "Violation": violated,
                })

            st.dataframe(table_data, use_container_width=True, hide_index=True)

            # Legend — auto-generated from subsystems in this template
            legend_parts = []
            for sub, color in sub_colors.items():
                legend_parts.append(f"<span style='color:{color}'>{sub}</span>")
            st.markdown(
                "**Subsystems:** " + " · ".join(legend_parts) + "  \n"
                "**Markers:** "
                "<span style='color:steelblue'>Same-domain</span> · "
                "<span style='color:darkorange'>Cross-domain hop</span> · "
                "<span style='color:crimson'>Certification violation</span>",
                unsafe_allow_html=True,
            )


# =====================================================================
# TAB 2: BATTERY THERMAL CASCADE
# =====================================================================
with tab_battery:
    st.title("Battery Thermal Runaway Cascade")
    st.markdown(
        "Cell-level thermal runaway propagation in a battery pack. "
        "Trigger a cell and watch the cascade spread."
    )

    from cascade_predict.physics import CellParams, PackGeometry
    from cascade_predict.simulation import CascadeSimulator

    # Sidebar
    st.sidebar.markdown("---")
    st.sidebar.header("Battery Pack")
    rows = st.sidebar.slider("Rows", 2, 8, 4, key="bat_rows")
    cols = st.sidebar.slider("Columns", 2, 8, 5, key="bat_cols")
    soc_mean = st.sidebar.slider("Mean SOC", 0.3, 1.0, 0.8, step=0.05, key="bat_soc")
    t_end = st.sidebar.slider("Duration (s)", 120, 1200, 900, step=60, key="bat_tend")

    params = CellParams()
    pack = PackGeometry(rows=rows, cols=cols, params=params)
    n_cells = pack.n_cells

    trigger_cell = st.sidebar.selectbox("Trigger Cell", list(range(n_cells)), key="bat_trigger")

    rng = np.random.RandomState(42)
    soc_distribution = np.clip(rng.normal(soc_mean, 0.1, n_cells), 0.1, 1.0)

    if st.button("Run Battery Cascade", type="primary", use_container_width=True, key="bat_run"):
        sim = CascadeSimulator(rows=rows, cols=cols, params=params)
        with st.spinner("Running thermal simulation..."):
            result = sim.run_physics(
                trigger_cells=[trigger_cell],
                soc_distribution=soc_distribution,
                trigger_temp=523.15,
                dt=0.5,
                t_end=t_end,
            )
        summary = sim.cascade_summary(result)

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Cells Affected", f"{summary['cells_affected']}/{summary['total_cells']}")
        m2.metric("Cascade Fraction", f"{summary['cascade_fraction']:.0%}")
        m3.metric("Peak Temp", f"{summary['max_temperature_C']:.0f}°C")
        m4.metric("Avg Delay", f"{summary['avg_propagation_delay_s']:.1f}s")

        # Temperature curves
        times = result["times"]
        temps_c = result["temperatures"] - 273.15
        runaway_times = result["runaway_times"]

        fig_temp = go.Figure()
        max_rt = np.max(runaway_times[np.isfinite(runaway_times)]) if np.any(np.isfinite(runaway_times)) else 1.0

        for i in range(n_cells):
            if i == trigger_cell:
                color, width = "red", 3
            elif np.isfinite(runaway_times[i]):
                frac = runaway_times[i] / max_rt
                color = f"rgb({int(255*(1-frac))}, {int(100*frac)}, 50)"
                width = 2
            else:
                color, width = "lightblue", 1

            fig_temp.add_trace(go.Scatter(
                x=times, y=temps_c[:, i], mode="lines",
                name=f"Cell {i}", line=dict(color=color, width=width),
            ))

        fig_temp.add_hline(y=params.t_runaway - 273.15, line_dash="dash",
                           line_color="red", annotation_text="Runaway Threshold")
        fig_temp.update_layout(height=400, xaxis_title="Time (s)", yaxis_title="Temperature (°C)")
        st.plotly_chart(fig_temp, use_container_width=True)

        # Runaway timeline
        finite_mask = np.isfinite(runaway_times)
        if np.any(finite_mask):
            order = np.argsort(runaway_times)
            timeline_cells = [i for i in order if np.isfinite(runaway_times[i])]
            fig_tl = go.Figure(go.Bar(
                x=[runaway_times[i] for i in timeline_cells],
                y=[f"Cell {i}" for i in timeline_cells],
                orientation="h",
                marker_color=["red" if i == trigger_cell else "orange" for i in timeline_cells],
                text=[f"{runaway_times[i]:.1f}s" for i in timeline_cells],
                textposition="outside",
            ))
            fig_tl.update_layout(
                height=max(200, 30 * len(timeline_cells)),
                xaxis_title="Time to Runaway (s)",
            )
            st.plotly_chart(fig_tl, use_container_width=True)

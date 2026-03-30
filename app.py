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
import streamlit.components.v1 as st_components
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
        "**Upload specs. Change a property. See everything that breaks.**"
    )

    from cascade_predict.templates import list_templates, get_template, get_failure_db
    from cascade_predict.graph import CascadeEngine
    from cascade_predict.spec_parser import (
        parse_csv, parse_text, parse_pdf, extract_pdf_text, ParameterOverride,
    )

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

    # =================================================================
    # STEP 1: Upload spec document (optional)
    # =================================================================
    _uploaded_overrides: list[ParameterOverride] = []

    with st.expander("Step 1: Upload Spec Document (optional)", expanded=False):
        st.markdown(
            "Upload a component spec to override default template values. "
            "Supports **CSV**, **PDF**, or **plain text** files."
        )

        upload_col, format_col = st.columns([2, 1])
        with upload_col:
            uploaded_file = st.file_uploader(
                "Upload spec document",
                type=["csv", "pdf", "txt", "text"],
                key="spec_upload",
                label_visibility="collapsed",
            )
        with format_col:
            st.markdown(
                "**CSV format:**\n"
                "```\n"
                "component,property,value,unit\n"
                "windshield,curvature,0.30,1/m\n"
                "battery_pack,mass,3800,kg\n"
                "```"
            )

        # Onshape URL input
        onshape_url = st.text_input(
            "Or paste an Onshape CAD URL:",
            placeholder="https://cad.onshape.com/documents/...",
            key="onshape_url_input",
        )

        if uploaded_file is not None:
            file_bytes = uploaded_file.read()
            file_name = uploaded_file.name.lower()

            if file_name.endswith(".csv"):
                _uploaded_overrides = parse_csv(file_bytes)
                if _uploaded_overrides:
                    st.success(f"Parsed **{len(_uploaded_overrides)}** parameters from CSV")
                else:
                    st.warning("No parameters found. Check CSV format: component, property, value, unit")

            elif file_name.endswith(".pdf"):
                with st.spinner("Extracting text from PDF..."):
                    pdf_text = extract_pdf_text(file_bytes)
                    _uploaded_overrides = parse_pdf(file_bytes)
                with st.expander("Extracted PDF text", expanded=False):
                    st.text(pdf_text[:3000])
                if _uploaded_overrides:
                    st.success(f"Found **{len(_uploaded_overrides)}** parameters in PDF")
                else:
                    st.info("No auto-detected parameters. You can enter values manually in the sidebar.")

            else:  # plain text
                _uploaded_overrides = parse_text(file_bytes)
                if _uploaded_overrides:
                    st.success(f"Found **{len(_uploaded_overrides)}** parameters in text")
                else:
                    st.info("No auto-detected parameters.")

            # Show extracted parameters and let user confirm/edit
            if _uploaded_overrides:
                st.markdown("**Extracted parameters** (matched to template components):")
                override_data = []
                for o in _uploaded_overrides:
                    matched = o.component_id in components
                    override_data.append({
                        "Component": o.component_id,
                        "Property": o.property_name,
                        "Value": o.value,
                        "Unit": o.unit,
                        "Confidence": o.confidence,
                        "Matched": "Yes" if matched else "No",
                    })
                st.dataframe(override_data, use_container_width=True, hide_index=True)

    # Apply uploaded overrides to component properties
    _overrides_applied = 0
    for o in _uploaded_overrides:
        if o.component_id in components:
            comp_obj = components[o.component_id]
            if o.property_name in comp_obj.properties:
                comp_obj.properties[o.property_name].value = o.value
                comp_obj.properties[o.property_name].source = f"uploaded ({o.confidence})"
                # Also update graph node if linked
                if o.property_name in comp_obj.property_to_node:
                    node_id = comp_obj.property_to_node[o.property_name]
                    if node_id in graph.nodes:
                        graph.nodes[node_id].value = o.value
                _overrides_applied += 1
    if _overrides_applied > 0:
        st.info(f"Applied **{_overrides_applied}** parameter overrides from uploaded document.")

    # --- Onshape CAD viewer ---
    _ONSHAPE_MODELS = {
        "electric_aircraft": {
            "Windshield Assembly": "https://cad.onshape.com/documents/1c2b19367ffb5d8a7c3953d3/w/d61a8106b16ff9792e6ec386/e/8a4b542a1512c1034282c522",
            "Full Aircraft Assembly": "https://cad.onshape.com/documents/aad2b820321cbbb01a2c1274/w/22c377380bdeb8d9d12abe4e/e/a5280709a215a12e106ece19",
        },
    }
    _cad_urls = dict(_ONSHAPE_MODELS.get(selected_template_id, {}))
    if onshape_url and "onshape.com" in onshape_url:
        _cad_urls["Uploaded CAD Model"] = onshape_url
    if _cad_urls:
        with st.expander("CAD Models (Onshape)", expanded=False):
            cad_tabs = st.tabs(list(_cad_urls.keys()))
            for tab, (label, url) in zip(cad_tabs, _cad_urls.items()):
                with tab:
                    st_components.iframe(url, height=500, scrolling=True)

    # =================================================================
    # STEP 2: Select component and vary parameter (sidebar)
    # =================================================================
    st.sidebar.markdown("---")
    st.sidebar.header("Step 2: Vary a Parameter")
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

    # --- Constraint relaxation ---
    if constraints:
        st.sidebar.markdown("---")
        with st.sidebar.expander("Relax Constraints", expanded=False):
            st.caption(
                "Toggle off or adjust limits for trade-study exploration. "
                "Relaxed constraints will not flag violations."
            )
            # Group constraints by unique (standard, section) to avoid
            # duplicate sliders for FAR/EASA pairs on the same parameter.
            _seen_params = {}
            for c in constraints:
                key = c.parameter_node_id
                if key not in _seen_params:
                    _seen_params[key] = c
            _constraint_relaxations = {}
            for c in _seen_params.values():
                col_toggle, col_label = st.columns([1, 4])
                enabled = col_toggle.checkbox(
                    "on", value=True, key=f"cst_en_{c.constraint_id}",
                    label_visibility="collapsed",
                )
                col_label.markdown(
                    f"**{c.standard} {c.section}** — {c.title}  \n"
                    f"{c.limit_type} {c.limit_value} {c.unit}"
                )
                if enabled:
                    new_limit = st.number_input(
                        f"Limit ({c.unit})",
                        value=c.limit_value,
                        step=abs(c.limit_value) * 0.05 if c.limit_value != 0 else 0.1,
                        format="%.4f",
                        key=f"cst_val_{c.constraint_id}",
                    )
                    _constraint_relaxations[c.parameter_node_id] = new_limit
                else:
                    # Disabled → remove the limit entirely
                    _constraint_relaxations[c.parameter_node_id] = None

    # --- Bayesian uncertainty toggle ---
    st.sidebar.markdown("---")
    st.sidebar.header("Uncertainty Analysis")
    _run_bayesian = st.sidebar.checkbox("Enable Bayesian (Monte Carlo)", value=False,
                                         key="bayesian_toggle")
    _mc_samples = 500
    if _run_bayesian:
        _mc_samples = st.sidebar.slider("MC Samples", 100, 2000, 500, step=100,
                                         key="mc_samples")
        st.sidebar.caption(
            "Runs the cascade N times with sampled edge sensitivities. "
            "Shows violation probabilities and confidence intervals."
        )

    # --- Run cascade ---
    if st.button("Propagate Change", type="primary", use_container_width=True) and selected_prop and new_value is not None:
        # Rebuild fresh graph for each run
        graph, components, constraints = tmpl.build()
        comp = components[selected_comp_id]

        # Apply constraint relaxations to graph nodes
        if constraints and _constraint_relaxations:
            for node_id, new_limit in _constraint_relaxations.items():
                if node_id in graph.nodes:
                    node = graph.nodes[node_id]
                    if new_limit is None:
                        # Constraint disabled — remove regulatory limit
                        node.regulatory_limit = None
                        node.bounds = (float("-inf"), float("inf"))
                    else:
                        node.regulatory_limit = new_limit
                        # Also update bounds upper if limit was raised
                        if node.bounds[1] != float("inf") and new_limit > node.bounds[1]:
                            node.bounds = (node.bounds[0], new_limit)

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
                product_type=tmpl.industry.lower(),
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

            # ==========================================================
            # BAYESIAN UNCERTAINTY ANALYSIS
            # ==========================================================
            if _run_bayesian:
                st.divider()
                st.subheader("Bayesian Uncertainty Analysis")
                st.markdown(
                    f"Monte Carlo propagation with **{_mc_samples} samples**. "
                    "Edge sensitivities are sampled from uncertainty distributions "
                    "to quantify confidence in the cascade predictions."
                )

                from cascade_predict.bayesian import BayesianCascadeEngine
                from cascade_predict.bayesian.uncertainty import (
                    build_default_aircraft_uncertainty,
                    build_default_ev_uncertainty,
                )

                _UNCERTAINTY_BUILDERS = {
                    "electric_aircraft": build_default_aircraft_uncertainty,
                    "ev_battery_pack": build_default_ev_uncertainty,
                }
                spec_builder = _UNCERTAINTY_BUILDERS.get(selected_template_id)
                if spec_builder is None:
                    st.info("No uncertainty spec defined for this template yet.")
                else:
                    spec = spec_builder()
                    with st.spinner(f"Running {_mc_samples} Monte Carlo samples..."):
                        bay_engine = BayesianCascadeEngine(
                            tmpl.build, spec, n_samples=_mc_samples, seed=42,
                        )
                        bay_result = bay_engine.propagate(
                            selected_comp_id, selected_prop, new_value,
                        )

                    bay_summary = bay_result.summary()

                    bm1, bm2, bm3 = st.columns(3)
                    bm1.metric("MC Samples", bay_summary["n_samples"])
                    bm2.metric("Likely Violations (>50%)", bay_summary["likely_violations"])
                    bm3.metric("Max P(violation)",
                               f"{bay_summary['max_violation_probability']:.0%}")

                    # --- Violation probability bars ---
                    active_violations = [
                        v for v in bay_result.violation_probabilities
                        if v.probability > 0.0
                    ]
                    if active_violations:
                        st.markdown("##### Violation Probabilities")
                        vp_labels = []
                        vp_probs = []
                        vp_colors = []
                        vp_hover = []
                        for v in active_violations:
                            label = v.node_id.replace("_", " ").title()
                            vp_labels.append(f"{label}\n({v.regulatory_ref})")
                            vp_probs.append(v.probability * 100)
                            if v.probability >= 0.8:
                                vp_colors.append("crimson")
                            elif v.probability >= 0.5:
                                vp_colors.append("darkorange")
                            elif v.probability >= 0.2:
                                vp_colors.append("gold")
                            else:
                                vp_colors.append("steelblue")
                            vp_hover.append(
                                f"<b>{label}</b><br>"
                                f"P(violation) = {v.probability:.1%}<br>"
                                f"Mean: {v.mean_value:.2f} {v.unit}<br>"
                                f"95th percentile: {v.p95_value:.2f} {v.unit}<br>"
                                f"Limit: {v.regulatory_limit} {v.unit}<br>"
                                f"Mean margin: {v.mean_margin:+.4f}"
                            )

                        fig_vp = go.Figure(go.Bar(
                            x=vp_probs, y=vp_labels, orientation="h",
                            marker_color=vp_colors,
                            hovertext=vp_hover, hoverinfo="text",
                            text=[f"{p:.0f}%" for p in vp_probs],
                            textposition="outside",
                        ))
                        fig_vp.add_vline(x=50, line_dash="dash",
                                         line_color="gray", line_width=1,
                                         annotation_text="50%")
                        fig_vp.update_layout(
                            height=max(200, 60 * len(active_violations)),
                            xaxis_title="Probability of Violation (%)",
                            xaxis=dict(range=[0, 110]),
                            margin=dict(l=200, r=60, t=20, b=40),
                        )
                        st.plotly_chart(fig_vp, use_container_width=True)

                    # --- Parameter distributions (box plots) ---
                    st.markdown("##### Parameter Distributions (90% CI)")
                    # Show distributions for nodes that changed meaningfully
                    dist_items = sorted(
                        bay_result.node_distributions.items(),
                        key=lambda x: abs(x[1].mean_delta),
                        reverse=True,
                    )[:12]  # top 12 by magnitude of change

                    if dist_items:
                        fig_box = go.Figure()
                        for nid, dist in dist_items:
                            label = nid.replace("_", " ").title()
                            # Normalize to % change from baseline for comparability
                            if abs(dist.baseline) > 1e-10:
                                pct_samples = (dist.samples - dist.baseline) / abs(dist.baseline) * 100
                            else:
                                pct_samples = dist.samples - dist.baseline
                            fig_box.add_trace(go.Box(
                                x=pct_samples, name=label,
                                boxpoints=False,
                                marker_color="steelblue",
                                line_color="steelblue",
                            ))

                        fig_box.update_layout(
                            height=max(300, 35 * len(dist_items)),
                            xaxis_title="% Change from Baseline (distribution across MC samples)",
                            margin=dict(l=200, r=40, t=20, b=40),
                            showlegend=False,
                        )
                        fig_box.add_vline(x=0, line_color="gray", line_width=1)
                        st.plotly_chart(fig_box, use_container_width=True)

                    # --- Uncertainty table ---
                    st.markdown("##### Detailed Uncertainty Summary")
                    unc_table = []
                    for nid, dist in dist_items:
                        node = graph.nodes.get(nid)
                        reg_limit = node.regulatory_limit if node else None
                        unc_table.append({
                            "Parameter": nid.replace("_", " ").title(),
                            "Baseline": f"{dist.baseline:.4f}",
                            "Mean": f"{dist.mean:.4f}",
                            "Std": f"{dist.std:.4f}",
                            "5th %ile": f"{dist.p5:.4f}",
                            "95th %ile": f"{dist.p95:.4f}",
                            "Unit": dist.unit,
                            "Limit": f"{reg_limit}" if reg_limit else "",
                        })
                    st.dataframe(unc_table, use_container_width=True, hide_index=True)


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

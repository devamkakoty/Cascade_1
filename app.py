"""
Cascade Prediction -- Interactive Dashboard.

Flow:
  1. Upload spec documents (CSV / PDF / text)
  2. Select template, component, property -- all in main area
  3. Run cascade -- results appear in navigable tabs
  4. Battery Thermal tab for cell-level cascade

Run with: streamlit run app.py
"""

import streamlit as st
import streamlit.components.v1 as st_components
import numpy as np
import plotly.graph_objects as go

st.set_page_config(page_title="Cascade Prediction", layout="wide")

# ── Palette ──────────────────────────────────────────────────────────
_PALETTE = [
    "rgba(255, 100, 100, 0.8)", "rgba(100, 200, 255, 0.8)",
    "rgba(255, 200, 50, 0.8)", "rgba(100, 255, 150, 0.8)",
    "rgba(200, 100, 255, 0.8)", "rgba(255, 150, 50, 0.8)",
    "rgba(180, 180, 180, 0.8)", "rgba(255, 50, 50, 0.8)",
    "rgba(50, 200, 200, 0.8)", "rgba(200, 200, 100, 0.8)",
]

def _subsystem_color_map(subsystems):
    return {s: _PALETTE[i % len(_PALETTE)] for i, s in enumerate(sorted(subsystems))}

_ONSHAPE_MODELS = {
    "electric_aircraft": {
        "Windshield Assembly": "https://cad.onshape.com/documents/1c2b19367ffb5d8a7c3953d3/w/d61a8106b16ff9792e6ec386/e/8a4b542a1512c1034282c522",
        "Full Aircraft Assembly": "https://cad.onshape.com/documents/aad2b820321cbbb01a2c1274/w/22c377380bdeb8d9d12abe4e/e/a5280709a215a12e106ece19",
    },
}

# ── Top-level tabs ───────────────────────────────────────────────────
tab_system, tab_battery = st.tabs([
    "System Cascade (Cross-Subsystem)",
    "Battery Thermal Cascade",
])

# =====================================================================
# TAB 1: SYSTEM CASCADE
# =====================================================================
with tab_system:
    st.title("Design Change Cascade Prediction")
    st.caption(
        "Upload a spec document, select a system template, change a property, "
        "and watch the cascade ripple across subsystems."
    )

    from cascade_predict.templates import list_templates, get_template, get_failure_db
    from cascade_predict.graph import CascadeEngine
    from cascade_predict.spec_parser import parse_csv, parse_pdf, parse_text, extract_pdf_text
    from cascade_predict.cad_parser import parse_cad_file, CADAnalysisResult
    from cascade_predict.components.cad_viewer import render_cad_viewer
    from cascade_predict.part_identifier import (
        SECTORS, get_sectors, get_parts_for_sector, get_part_profile,
        auto_suggest_part, build_custom_part_profile,
    )

    # ── STEP 1: Upload Documents ─────────────────────────────────────
    st.header("1  Upload Spec / CAD File")

    upload_col1, upload_col2 = st.columns(2)

    # --- Spec document upload ---
    with upload_col1:
        st.subheader("Spec Document")
        uploaded_spec = st.file_uploader(
            "CSV, PDF, or text with component specs",
            type=["csv", "pdf", "txt", "text"],
            key="spec_upload",
        )
    doc_overrides = []
    if uploaded_spec is not None:
        raw = uploaded_spec.read()
        fname = uploaded_spec.name.lower()
        if fname.endswith(".csv"):
            doc_overrides = parse_csv(raw)
        elif fname.endswith(".pdf"):
            doc_overrides = parse_pdf(raw)
            with st.expander("Extracted PDF text"):
                st.text(extract_pdf_text(raw))
        else:
            doc_overrides = parse_text(raw.decode("utf-8", errors="replace"))

        if doc_overrides:
            st.success(f"Extracted **{len(doc_overrides)}** parameter(s) from spec file.")
            for ov in doc_overrides:
                st.markdown(f"- `{ov.component_id}.{ov.property_name}` = **{ov.value}** {ov.unit}")
        else:
            st.info("No parameters auto-detected from spec. You can still select manually below.")

    # --- CAD file upload ---
    with upload_col2:
        st.subheader("CAD / 3D Model")
        uploaded_cad = st.file_uploader(
            "STL or STEP/STP file",
            type=["stl", "step", "stp"],
            key="cad_upload",
        )
        onshape_url = st.text_input(
            "Or paste an Onshape URL",
            placeholder="https://cad.onshape.com/documents/...",
            key="onshape_url_input",
        )

    # --- Onshape embed ---
    _has_onshape = False
    if onshape_url and onshape_url.strip():
        url = onshape_url.strip()
        if "onshape.com" in url:
            _has_onshape = True
            if url.startswith("http://"):
                url = "https://" + url[7:]
            elif not url.startswith("https://"):
                url = "https://" + url
            st.markdown("---")
            st.subheader("Onshape CAD Viewer")
            st_components.iframe(url, height=550, scrolling=True)
            st.success("Onshape model loaded. Proceed to Step 2 below to configure and run the cascade.")
        else:
            st.warning("Please paste a valid Onshape URL (must contain onshape.com).")

    cad_result = None  # type: CADAnalysisResult | None
    if uploaded_cad is not None:
        cad_raw = uploaded_cad.read()
        cad_result = parse_cad_file(uploaded_cad.name, cad_raw)

        st.markdown("---")
        st.subheader("3D Model Preview")
        if cad_result.file_type == "stl" and cad_result.raw_stl_bytes:
            render_cad_viewer(stl_bytes=cad_result.raw_stl_bytes, height=500)
        elif cad_result.vertices is not None and len(cad_result.vertices) > 0:
            render_cad_viewer(vertices=cad_result.vertices, height=500)

        if cad_result.parameters:
            with st.expander(f"Extracted Geometry ({len(cad_result.parameters)} parameters)", expanded=True):
                geo_table = []
                for p in cad_result.parameters:
                    geo_table.append({
                        "Parameter": p.name.replace("_", " ").title(),
                        "Value": f"{p.value:.4f}" if isinstance(p.value, float) and p.value != int(p.value) else f"{p.value:.1f}",
                        "Unit": p.unit,
                        "Category": p.category,
                        "Description": p.description,
                    })
                st.dataframe(geo_table, use_container_width=True, hide_index=True)

    # Show guidance if nothing uploaded yet
    if uploaded_spec is None and uploaded_cad is None and not _has_onshape:
        st.info("Upload a spec document, a CAD file (STL/STEP), or paste an Onshape URL above to get started. You can also skip directly to Step 3.")

    # ── STEP 2: Identify Part ────────────────────────────────────────
    st.markdown("---")
    st.header("2  Identify Your Part")

    # Auto-suggest if CAD was uploaded
    _suggestion = None
    if cad_result and cad_result.parameters:
        _suggestion = auto_suggest_part(cad_result.parameters)

    sector_options = get_sectors()
    default_sector_idx = 0
    if _suggestion:
        for i, (k, _) in enumerate(sector_options):
            if k == _suggestion[0]:
                default_sector_idx = i
                break

    col_sector, col_part = st.columns(2)
    with col_sector:
        selected_sector = st.selectbox(
            "Sector / Industry",
            [k for k, _ in sector_options],
            format_func=lambda x: dict(sector_options)[x],
            index=default_sector_idx,
            key="sector_select",
        )

    part_options = get_parts_for_sector(selected_sector)
    # Add "Other" option to every sector
    part_options_with_other = part_options + [("_other", "Other / Not Listed (describe below)")]
    default_part_idx = 0
    if _suggestion and _suggestion[0] == selected_sector:
        for i, (k, _) in enumerate(part_options):
            if k == _suggestion[1]:
                default_part_idx = i
                break

    with col_part:
        selected_part = st.selectbox(
            "Part Type",
            [k for k, _ in part_options_with_other],
            format_func=lambda x: dict(part_options_with_other)[x],
            index=default_part_idx,
            key="part_select",
        )

    # Handle custom / other part
    part_profile = None
    if selected_part == "_other":
        st.markdown("##### Describe Your Part")
        col_cname, col_cdesc = st.columns(2)
        with col_cname:
            custom_part_name = st.text_input("Part name", placeholder="e.g. Radar Mast, Keel Bracket...", key="custom_part_name")
        with col_cdesc:
            custom_part_desc = st.text_input("Brief description", placeholder="e.g. Welded steel bracket supporting the sonar dome", key="custom_part_desc")

        if cad_result and cad_result.parameters:
            part_profile = build_custom_part_profile(
                selected_sector, custom_part_name, custom_part_desc, cad_result.parameters,
            )
            st.info(f"Physics auto-inferred from geometry features for **{part_profile.part_label}**")
        elif custom_part_name:
            # No CAD uploaded, build minimal profile
            part_profile = build_custom_part_profile(
                selected_sector, custom_part_name, custom_part_desc, [],
            )
            st.info("Upload a CAD file in Step 1 for automatic physics inference, or configure manually in Step 3.")
    else:
        part_profile = get_part_profile(selected_sector, selected_part)
        if _suggestion and _suggestion[0] == selected_sector and _suggestion[1] == selected_part:
            st.success(f"Auto-detected: **{part_profile.part_label}** (confidence: {_suggestion[2]:.0%})")

    # Show part info and physics
    if part_profile:
        st.markdown(f"*{part_profile.description}*")

        with st.expander("Physics Models Triggered", expanded=True):
            for pm in part_profile.physics_models:
                st.markdown(f"- {pm}")

        # Map CAD geometry → cascade parameters
        if cad_result and cad_result.parameters and part_profile.mappings:
            st.markdown("##### Geometry to Cascade Mapping")
            cad_params = {p.name: p for p in cad_result.parameters}
            for mapping in part_profile.mappings:
                if mapping.cad_param in cad_params:
                    p = cad_params[mapping.cad_param]
                    mapped_value = p.value * mapping.unit_conversion
                    col_info, col_use = st.columns([4, 1])
                    col_info.markdown(
                        f"**{mapping.cad_param.replace('_', ' ').title()}** "
                        f"({p.value:.4f} {p.unit}) → "
                        f"`{mapping.component_id}.{mapping.property_name}` = **{mapped_value:.4f}**  \n"
                        f"_{mapping.description}_"
                    )
                    if col_use.checkbox("Use", key=f"partmap_{mapping.cad_param}", value=True):
                        from cascade_predict.spec_parser import ParameterOverride
                        doc_overrides.append(ParameterOverride(
                            component_id=mapping.component_id,
                            property_name=mapping.property_name,
                            value=mapped_value,
                            unit="",
                            source=f"CAD: {mapping.cad_param}",
                        ))

    # ── STEP 3: Template & Component Selection ───────────────────────
    st.markdown("---")
    st.header("3  Configure Cascade")

    all_templates = list_templates()
    template_names = {t.template_id: f"{t.name} ({t.industry})" for t in all_templates}

    # Auto-select template from part profile
    default_tmpl_idx = 0
    if part_profile:
        tmpl_keys = list(template_names.keys())
        if part_profile.template_id in tmpl_keys:
            default_tmpl_idx = tmpl_keys.index(part_profile.template_id)

    col_tmpl, col_comp = st.columns(2)
    with col_tmpl:
        selected_template_id = st.selectbox(
            "System Template",
            list(template_names.keys()),
            format_func=lambda x: template_names[x],
            index=default_tmpl_idx,
            key="tmpl_select",
        )
    tmpl = get_template(selected_template_id)
    graph, components, constraints = tmpl.build()

    # Auto-select component from part profile
    comp_names = {cid: c.name for cid, c in components.items()}
    # Add "Unknown / Custom" option
    comp_names_with_custom = dict(comp_names)
    comp_names_with_custom["_custom"] = "Unknown / Custom Component"
    default_comp_idx = 0
    if part_profile and part_profile.component_id in comp_names:
        comp_keys = list(comp_names_with_custom.keys())
        default_comp_idx = comp_keys.index(part_profile.component_id)

    with col_comp:
        selected_comp_id = st.selectbox(
            "Component",
            list(comp_names_with_custom.keys()),
            format_func=lambda x: comp_names_with_custom[x],
            index=default_comp_idx,
        )

    # Handle custom component
    if selected_comp_id == "_custom":
        from cascade_predict.subsystems.component import Component as CompClass, ComponentProperty
        st.markdown("##### Define Custom Component")
        col_ccomp_name, col_ccomp_sub = st.columns(2)
        with col_ccomp_name:
            custom_comp_name = st.text_input("Component name", placeholder="e.g. Sonar Dome", key="custom_comp_name")
        with col_ccomp_sub:
            # Let user pick which subsystem/graph node to attach to
            graph_node_ids = sorted([nid for nid in graph.nodes.keys()
                                      if nid not in {"material_cost", "manufacturing_cost", "tooling_cost",
                                                      "total_cost_delta", "manufacturing_lead_time",
                                                      "certification_time", "total_schedule_delta"}])
            target_node = st.selectbox(
                "Drives which system parameter?",
                graph_node_ids,
                format_func=lambda x: f"{x.replace('_', ' ').title()} ({graph.nodes[x].value:.2f} {graph.nodes[x].unit})",
                key="custom_target_node",
            )

        # Build a temporary component with a single editable property
        node = graph.nodes[target_node]
        comp = CompClass(
            component_id="_custom",
            name=custom_comp_name or "Custom Component",
            subsystem=node.subsystem,
            description="User-defined component",
        )
        comp.add_property(
            target_node, node.value, node.unit,
            graph_node_id=target_node,
            description=node.description,
            source="custom",
        )
        # Also add any doc_overrides as extra properties
        for ov in doc_overrides:
            if ov.property_name in [n for n in graph.nodes]:
                ovnode = graph.nodes[ov.property_name]
                comp.add_property(
                    ov.property_name, ov.value, ov.unit or ovnode.unit,
                    graph_node_id=ov.property_name,
                    description=f"From {ov.source}",
                    source=ov.source,
                )
        selected_comp_id = "_custom"
    else:
        comp = components[selected_comp_id]

    # Failure DB warnings
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
                f"*Source: {w.failure.source} ({w.failure.date}) -- {w.failure.reference}*"
            )

    # Property selection and new value
    editable_props = {
        name: prop for name, prop in comp.properties.items()
        if name in comp.property_to_node
    }

    if not editable_props:
        st.warning("This component has no graph-linked properties to change.")
        selected_prop = None
    else:
        col_prop, col_val = st.columns(2)
        with col_prop:
            selected_prop = st.selectbox(
                "Property to Change",
                list(editable_props.keys()),
                format_func=lambda x: f"{x} ({editable_props[x].value} {editable_props[x].unit})",
            )
        prop = editable_props[selected_prop]
        with col_val:
            new_value = st.number_input(
                f"New value ({prop.unit})",
                value=prop.value,
                step=abs(prop.value) * 0.05 if prop.value != 0 else 0.1,
                format="%.4f",
            )
            delta = new_value - prop.value
            if abs(delta) > 1e-10:
                pct = (delta / prop.value * 100) if prop.value != 0 else float("inf")
                st.metric("Change", f"{delta:+.4f} {prop.unit}", f"{pct:+.1f}%")

    # ── CAD Models ───────────────────────────────────────────────────
    if selected_template_id in _ONSHAPE_MODELS:
        with st.expander("CAD Models (Onshape)", expanded=False):
            cad_models = _ONSHAPE_MODELS[selected_template_id]
            cad_tabs = st.tabs(list(cad_models.keys()))
            for ctab, (label, url) in zip(cad_tabs, cad_models.items()):
                with ctab:
                    st_components.iframe(url, height=500, scrolling=True)

    # ── Quick Scenarios ──────────────────────────────────────────────
    if tmpl.presets:
        with st.expander("Quick Scenarios", expanded=False):
            preset_names = ["Custom (use controls above)"] + [p["name"] for p in tmpl.presets]
            scenario = st.radio("Try a preset:", preset_names, key="preset_radio")
            if scenario != "Custom (use controls above)":
                for p in tmpl.presets:
                    if p["name"] == scenario:
                        selected_comp_id = p["component_id"]
                        selected_prop = p["property_name"]
                        new_value = p["new_value"]
                        comp = components[selected_comp_id]
                        break

    # ── Constraint Relaxation ────────────────────────────────────────
    _constraint_relaxations = {}
    if constraints:
        with st.expander("Relax Constraints", expanded=False):
            st.caption("Toggle off or adjust limits for trade-study exploration.")
            _seen_params = {}
            for c in constraints:
                if c.parameter_node_id not in _seen_params:
                    _seen_params[c.parameter_node_id] = c
            for c in _seen_params.values():
                col_toggle, col_label = st.columns([1, 4])
                enabled = col_toggle.checkbox(
                    "on", value=True, key=f"cst_en_{c.constraint_id}",
                    label_visibility="collapsed",
                )
                col_label.markdown(
                    f"**{c.standard} {c.section}** -- {c.title}  \n"
                    f"{c.limit_type} {c.limit_value} {c.unit}"
                )
                if enabled:
                    new_limit = st.number_input(
                        f"Limit ({c.unit})", value=c.limit_value,
                        step=abs(c.limit_value) * 0.05 if c.limit_value != 0 else 0.1,
                        format="%.4f", key=f"cst_val_{c.constraint_id}",
                    )
                    _constraint_relaxations[c.parameter_node_id] = new_limit
                else:
                    _constraint_relaxations[c.parameter_node_id] = None

    # ── Bayesian Toggle ──────────────────────────────────────────────
    col_bay, col_mc = st.columns([1, 2])
    with col_bay:
        _run_bayesian = st.checkbox("Enable Bayesian (Monte Carlo)", value=False, key="bayesian_toggle")
    _mc_samples = 500
    if _run_bayesian:
        with col_mc:
            _mc_samples = st.slider("MC Samples", 100, 2000, 500, step=100, key="mc_samples")

    # ── Propagate Button ─────────────────────────────────────────────
    st.markdown("---")
    st.header("4  Run Cascade")
    if selected_prop:
        prop_obj = editable_props.get(selected_prop)
        if prop_obj and new_value is not None and abs(new_value - prop_obj.value) > 1e-10:
            pct_preview = ((new_value - prop_obj.value) / prop_obj.value * 100) if prop_obj.value != 0 else 0
            st.markdown(
                f"Ready to propagate: **{selected_prop}** on **{comp.name}** "
                f"({prop_obj.value:.4f} -> {new_value:.4f} {prop_obj.unit}, {pct_preview:+.1f}%)"
            )
        else:
            st.info("Change the property value above to see what breaks, then click Propagate.")
    run_clicked = st.button("Propagate Change", type="primary", use_container_width=True)

    # ═════════════════════════════════════════════════════════════════
    # RESULTS — only after clicking Propagate
    # ═════════════════════════════════════════════════════════════════
    if run_clicked and selected_prop and new_value is not None:
        # Rebuild fresh graph
        graph, components, constraints = tmpl.build()
        if selected_comp_id == "_custom":
            # Rebuild the custom component against the fresh graph
            node = graph.nodes[target_node]
            from cascade_predict.subsystems.component import Component as CompClass
            comp = CompClass("_custom", custom_comp_name or "Custom Component", node.subsystem)
            comp.add_property(target_node, node.value, node.unit, graph_node_id=target_node, source="custom")
            for ov in doc_overrides:
                if ov.property_name in graph.nodes:
                    ovn = graph.nodes[ov.property_name]
                    comp.add_property(ov.property_name, ov.value, ov.unit or ovn.unit,
                                      graph_node_id=ov.property_name, source=ov.source)
        else:
            comp = components[selected_comp_id]

        # Apply constraint relaxations
        if constraints and _constraint_relaxations:
            for node_id, new_limit in _constraint_relaxations.items():
                if node_id in graph.nodes:
                    node = graph.nodes[node_id]
                    if new_limit is None:
                        node.regulatory_limit = None
                        node.bounds = (float("-inf"), float("inf"))
                    else:
                        node.regulatory_limit = new_limit
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

            # Deduplicate steps
            seen_targets = {}
            for step in result.steps:
                if step.target_node not in seen_targets:
                    seen_targets[step.target_node] = step
                else:
                    if abs(step.delta_output) > abs(seen_targets[step.target_node].delta_output):
                        seen_targets[step.target_node] = step
            unique_steps = list(seen_targets.values())

            # Split steps into engineering vs cost/schedule
            _COST_SCHEDULE_NODES = {"material_cost", "manufacturing_cost", "tooling_cost",
                                     "total_cost_delta", "manufacturing_lead_time",
                                     "certification_time", "total_schedule_delta"}
            eng_steps = [s for s in unique_steps if s.target_node not in _COST_SCHEDULE_NODES]
            cost_steps = [s for s in unique_steps if s.target_node in _COST_SCHEDULE_NODES]

            st.divider()
            st.header("Cascade Results")

            # ── Result tabs ──────────────────────────────────────────
            result_tab_names = ["Overview", "Cascade Flow", "Violations", "Cost & Schedule", "Comparison"]
            if _run_bayesian:
                result_tab_names.append("Uncertainty")
            result_tabs = st.tabs(result_tab_names)

            # ── TAB: Overview ────────────────────────────────────────
            with result_tabs[0]:
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Parameters Affected", summary["nodes_affected"])
                m2.metric("Subsystems Hit", summary["subsystems_affected"])
                m3.metric("Cross-Domain Hops", summary["cross_domain_hops"])
                n_violations = summary["violations"]
                m4.metric("Violations", n_violations,
                           delta=f"{n_violations} cert issues" if n_violations > 0 else "Clean",
                           delta_color="inverse")

                st.subheader("Cascade Waterfall")
                st.caption("Each bar = change propagated. Red = violation. Orange = cross-domain hop.")

                labels, pct_changes, colors, hover_texts = [], [], [], []
                for step in eng_steps:
                    baseline = result.initial_state.get(step.target_node, step.old_value)
                    pct = (step.delta_output / baseline * 100) if abs(baseline) > 1e-10 else 0
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
                        f"<b>{label}</b><br>Subsystem: {step.target_subsystem}<br>"
                        f"Change: {step.delta_output:+.4f} {node.unit}<br>"
                        f"Before: {baseline:.4f} -> After: {step.new_value:.4f}<br>"
                        f"% Change: {pct:+.2f}%<br>"
                    )
                    if step.causes_violation:
                        hover += f"<b>VIOLATION: {step.violation_detail}</b><br>"
                    if step.edge.physics_equation:
                        hover += f"Physics: {step.edge.physics_equation}"
                    hover_texts.append(hover)

                fig_wf = go.Figure(go.Bar(
                    x=pct_changes, y=labels, orientation="h",
                    marker_color=colors, hovertext=hover_texts, hoverinfo="text",
                ))
                fig_wf.update_layout(
                    height=max(350, 28 * len(labels)),
                    xaxis_title="% Change from Baseline",
                    margin=dict(l=200, r=40, t=20, b=40),
                    yaxis=dict(autorange="reversed"),
                )
                fig_wf.add_vline(x=0, line_color="gray", line_width=1)
                st.plotly_chart(fig_wf, use_container_width=True)

            # ── TAB: Cascade Flow (Sankey) ───────────────────────────
            with result_tabs[1]:
                st.subheader("Cascade Flow (Engineering)")
                all_node_ids = [result.trigger_node]
                for step in eng_steps:
                    if step.source_node not in all_node_ids:
                        all_node_ids.append(step.source_node)
                    if step.target_node not in all_node_ids:
                        all_node_ids.append(step.target_node)
                node_idx = {nid: i for i, nid in enumerate(all_node_ids)}
                sub_colors = _subsystem_color_map(graph.subsystems())
                node_colors, node_labels = [], []
                for nid in all_node_ids:
                    node = graph.nodes.get(nid)
                    if node:
                        node_colors.append(sub_colors.get(node.subsystem, "rgba(200,200,200,0.8)"))
                        node_labels.append(nid.replace("_", " ").title())
                    else:
                        node_colors.append("rgba(200,200,200,0.8)")
                        node_labels.append(nid)
                s_src, s_tgt, s_val, s_col = [], [], [], []
                for step in eng_steps:
                    if step.source_node in node_idx and step.target_node in node_idx:
                        s_src.append(node_idx[step.source_node])
                        s_tgt.append(node_idx[step.target_node])
                        s_val.append(max(abs(step.delta_output), 0.01))
                        if step.causes_violation:
                            s_col.append("rgba(255, 0, 0, 0.5)")
                        elif step.is_cross_domain:
                            s_col.append("rgba(255, 165, 0, 0.4)")
                        else:
                            s_col.append("rgba(100, 149, 237, 0.3)")
                fig_sk = go.Figure(go.Sankey(
                    node=dict(label=node_labels, color=node_colors, pad=15, thickness=20),
                    link=dict(source=s_src, target=s_tgt, value=s_val, color=s_col),
                ))
                fig_sk.update_layout(height=500, margin=dict(l=20, r=20, t=20, b=20))
                st.plotly_chart(fig_sk, use_container_width=True)

            # ── TAB: Violations ──────────────────────────────────────
            with result_tabs[2]:
                if result.violations:
                    st.subheader("Certification Violations")
                    for v in result.violations:
                        node = graph.nodes[v["node_id"]]
                        baseline = result.initial_state[v["node_id"]]
                        st.error(
                            f"**{v['regulatory_ref']}** -- {v['node_id'].replace('_', ' ').title()}  \n"
                            f"Baseline: {baseline:.4f} {v['unit']} -> "
                            f"After cascade: **{v['value']:.4f} {v['unit']}**  \n"
                            f"Limit: {v['regulatory_limit']} {v['unit']}  \n"
                            f"Margin: **{v['margin']:.4f}** (negative = violated)"
                        )
                        paths = graph.find_paths(result.trigger_node, v["node_id"])
                        if paths:
                            shortest = min(paths, key=len)
                            path_str = " -> ".join(
                                [result.trigger_node.replace("_", " ")] +
                                [e.target_id.replace("_", " ") for e in shortest]
                            )
                            st.markdown(f"*Cascade path:* `{path_str}`")
                else:
                    st.success("No certification violations detected.")

                # Historic failure warnings
                cascade_warnings = failure_db.check_cascade_result(
                    affected_node_ids=result.affected_nodes,
                    violated_node_ids=[v["node_id"] for v in result.violations],
                    product_type=tmpl.industry.lower(),
                )
                if cascade_warnings:
                    st.subheader("Historic Failure Warnings")
                    st.caption("Past failures involving the same parameters affected by this cascade.")
                    for w in cascade_warnings:
                        sev = w.failure.severity.value.upper()
                        st.warning(
                            f"**[{sev}] {w.failure.title}**  \n"
                            f"{w.failure.root_cause[:300]}  \n\n"
                            f"**Why relevant:** {w.match_reason}  \n"
                            f"**Recommendation:** {w.recommendation}  \n"
                            f"*Product: {w.failure.product_type} | "
                            f"Source: {w.failure.source} ({w.failure.date})*"
                        )

            # ── TAB: Cost & Schedule ────────────────────────────────
            with result_tabs[3]:
                st.subheader("Cost & Schedule Impact")
                if cost_steps:
                    # Separate cost vs schedule
                    _cost_nodes = {"material_cost", "manufacturing_cost", "tooling_cost", "total_cost_delta"}
                    _sched_nodes = {"manufacturing_lead_time", "certification_time", "total_schedule_delta"}
                    cost_only = [s for s in cost_steps if s.target_node in _cost_nodes]
                    sched_only = [s for s in cost_steps if s.target_node in _sched_nodes]

                    # Cost summary metrics
                    total_cost = sum(s.delta_output for s in cost_only if s.target_node == "total_cost_delta")
                    total_weeks = sum(s.delta_output for s in sched_only if s.target_node == "total_schedule_delta")
                    mc1, mc2 = st.columns(2)
                    mc1.metric("Total Cost Impact", f"${total_cost:+,.0f}")
                    mc2.metric("Total Schedule Impact", f"{total_weeks:+.1f} weeks")

                    # Cost breakdown table
                    st.markdown("##### Cost Breakdown")
                    cost_table = []
                    for step in cost_only:
                        node = graph.nodes[step.target_node]
                        cost_table.append({
                            "Item": step.target_node.replace("_", " ").title(),
                            "Change": f"${step.delta_output:+,.0f}",
                            "After": f"${step.new_value:,.0f}",
                            "Description": node.description,
                        })
                    if cost_table:
                        st.dataframe(cost_table, use_container_width=True, hide_index=True)

                    # Cost bar chart
                    if cost_only:
                        cost_labels = [s.target_node.replace("_", " ").title() for s in cost_only]
                        cost_values = [s.delta_output for s in cost_only]
                        cost_colors = ["#e74c3c" if v > 0 else "#2ecc71" for v in cost_values]
                        fig_cost = go.Figure(go.Bar(
                            x=cost_values, y=cost_labels, orientation="h",
                            marker_color=cost_colors,
                            text=[f"${v:+,.0f}" for v in cost_values],
                            textposition="outside",
                        ))
                        fig_cost.update_layout(
                            height=max(200, 50 * len(cost_labels)),
                            xaxis_title="Cost Change ($)",
                            margin=dict(l=200, r=80, t=20, b=40),
                        )
                        fig_cost.add_vline(x=0, line_color="gray", line_width=1)
                        st.plotly_chart(fig_cost, use_container_width=True)

                    # Schedule breakdown
                    st.markdown("##### Schedule Breakdown")
                    sched_table = []
                    for step in sched_only:
                        node = graph.nodes[step.target_node]
                        sched_table.append({
                            "Item": step.target_node.replace("_", " ").title(),
                            "Change": f"{step.delta_output:+.1f} weeks",
                            "After": f"{step.new_value:.1f} weeks",
                            "Description": node.description,
                        })
                    if sched_table:
                        st.dataframe(sched_table, use_container_width=True, hide_index=True)

                    # Schedule bar chart
                    if sched_only:
                        sched_labels = [s.target_node.replace("_", " ").title() for s in sched_only]
                        sched_values = [s.delta_output for s in sched_only]
                        sched_colors = ["#e74c3c" if v > 0 else "#2ecc71" for v in sched_values]
                        fig_sched = go.Figure(go.Bar(
                            x=sched_values, y=sched_labels, orientation="h",
                            marker_color=sched_colors,
                            text=[f"{v:+.1f}w" for v in sched_values],
                            textposition="outside",
                        ))
                        fig_sched.update_layout(
                            height=max(200, 50 * len(sched_labels)),
                            xaxis_title="Schedule Change (weeks)",
                            margin=dict(l=200, r=80, t=20, b=40),
                        )
                        fig_sched.add_vline(x=0, line_color="gray", line_width=1)
                        st.plotly_chart(fig_sched, use_container_width=True)
                else:
                    st.info("No cost/schedule impact detected for this change.")

            # ── TAB: Comparison ──────────────────────────────────────
            with result_tabs[4]:
                st.subheader("Parameter Comparison (Before -> After)")
                table_data = []
                for step in eng_steps:
                    node = graph.nodes[step.target_node]
                    baseline = result.initial_state.get(step.target_node, step.old_value)
                    pct = (step.delta_output / baseline * 100) if abs(baseline) > 1e-10 else 0
                    table_data.append({
                        "Parameter": step.target_node.replace("_", " ").title(),
                        "Subsystem": step.target_subsystem.upper(),
                        "Before": f"{baseline:.4f}",
                        "After": f"{step.new_value:.4f}",
                        "Delta": f"{step.delta_output:+.4f}",
                        "% Change": f"{pct:+.2f}%",
                        "Unit": node.unit,
                        "Violation": "YES" if step.causes_violation else "",
                    })
                st.dataframe(table_data, use_container_width=True, hide_index=True)

                legend_parts = []
                for sub, color in sub_colors.items():
                    legend_parts.append(f"<span style='color:{color}'>{sub}</span>")
                st.markdown(
                    "**Subsystems:** " + " . ".join(legend_parts) + "  \n"
                    "**Markers:** "
                    "<span style='color:steelblue'>Same-domain</span> . "
                    "<span style='color:darkorange'>Cross-domain hop</span> . "
                    "<span style='color:crimson'>Certification violation</span>",
                    unsafe_allow_html=True,
                )

            # ── TAB: Uncertainty (Bayesian) ──────────────────────────
            if _run_bayesian:
                with result_tabs[5]:
                    st.subheader("Bayesian Uncertainty Analysis")
                    st.markdown(
                        f"Monte Carlo propagation with **{_mc_samples} samples**. "
                        "Edge sensitivities sampled from uncertainty distributions."
                    )
                    from cascade_predict.bayesian import BayesianCascadeEngine
                    from cascade_predict.bayesian.uncertainty import (
                        build_default_aircraft_uncertainty,
                        build_default_ev_uncertainty,
                    )
                    _UNC_BUILDERS = {
                        "electric_aircraft": build_default_aircraft_uncertainty,
                        "ev_battery_pack": build_default_ev_uncertainty,
                    }
                    spec_builder = _UNC_BUILDERS.get(selected_template_id)
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
                        bm3.metric("Max P(violation)", f"{bay_summary['max_violation_probability']:.0%}")

                        # Violation probability bars
                        active_violations = [
                            v for v in bay_result.violation_probabilities if v.probability > 0.0
                        ]
                        if active_violations:
                            st.markdown("##### Violation Probabilities")
                            vp_labels, vp_probs, vp_colors, vp_hover = [], [], [], []
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
                                    f"95th pct: {v.p95_value:.2f} {v.unit}<br>"
                                    f"Limit: {v.regulatory_limit} {v.unit}<br>"
                                    f"Mean margin: {v.mean_margin:+.4f}"
                                )
                            fig_vp = go.Figure(go.Bar(
                                x=vp_probs, y=vp_labels, orientation="h",
                                marker_color=vp_colors, hovertext=vp_hover, hoverinfo="text",
                                text=[f"{p:.0f}%" for p in vp_probs], textposition="outside",
                            ))
                            fig_vp.add_vline(x=50, line_dash="dash", line_color="gray",
                                             line_width=1, annotation_text="50%")
                            fig_vp.update_layout(
                                height=max(200, 60 * len(active_violations)),
                                xaxis_title="Probability of Violation (%)",
                                xaxis=dict(range=[0, 110]),
                                margin=dict(l=200, r=60, t=20, b=40),
                            )
                            st.plotly_chart(fig_vp, use_container_width=True)

                        # Parameter distributions
                        st.markdown("##### Parameter Distributions (90% CI)")
                        dist_items = sorted(
                            bay_result.node_distributions.items(),
                            key=lambda x: abs(x[1].mean_delta), reverse=True,
                        )[:12]
                        if dist_items:
                            fig_box = go.Figure()
                            for nid, dist in dist_items:
                                label = nid.replace("_", " ").title()
                                if abs(dist.baseline) > 1e-10:
                                    pct_samples = (dist.samples - dist.baseline) / abs(dist.baseline) * 100
                                else:
                                    pct_samples = dist.samples - dist.baseline
                                fig_box.add_trace(go.Box(
                                    x=pct_samples, name=label, boxpoints=False,
                                    marker_color="steelblue", line_color="steelblue",
                                ))
                            fig_box.update_layout(
                                height=max(300, 35 * len(dist_items)),
                                xaxis_title="% Change from Baseline",
                                margin=dict(l=200, r=40, t=20, b=40),
                                showlegend=False,
                            )
                            fig_box.add_vline(x=0, line_color="gray", line_width=1)
                            st.plotly_chart(fig_box, use_container_width=True)

                        # Uncertainty table
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
    st.caption(
        "Cell-level thermal runaway propagation in a battery pack. "
        "Trigger a cell and watch the cascade spread."
    )

    from cascade_predict.physics import CellParams, PackGeometry
    from cascade_predict.simulation import CascadeSimulator

    st.header("Pack Configuration")
    col_r, col_c, col_s, col_d = st.columns(4)
    with col_r:
        rows = st.slider("Rows", 2, 8, 4, key="bat_rows")
    with col_c:
        cols = st.slider("Columns", 2, 8, 5, key="bat_cols")
    with col_s:
        soc_mean = st.slider("Mean SOC", 0.3, 1.0, 0.8, step=0.05, key="bat_soc")
    with col_d:
        t_end = st.slider("Duration (s)", 120, 1200, 900, step=60, key="bat_tend")

    params = CellParams()
    pack = PackGeometry(rows=rows, cols=cols, params=params)
    n_cells = pack.n_cells

    trigger_cell = st.selectbox("Trigger Cell", list(range(n_cells)), key="bat_trigger")

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
        bsummary = sim.cascade_summary(result)

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Cells Affected", f"{bsummary['cells_affected']}/{bsummary['total_cells']}")
        m2.metric("Cascade Fraction", f"{bsummary['cascade_fraction']:.0%}")
        m3.metric("Peak Temp", f"{bsummary['max_temperature_C']:.0f} C")
        m4.metric("Avg Delay", f"{bsummary['avg_propagation_delay_s']:.1f}s")

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
        fig_temp.update_layout(height=400, xaxis_title="Time (s)", yaxis_title="Temperature (C)")
        st.plotly_chart(fig_temp, use_container_width=True)

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

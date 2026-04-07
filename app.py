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
tab_system, tab_battery, tab_about = st.tabs([
    "System Cascade (Cross-Subsystem)",
    "Battery Thermal Cascade",
    "How It Works",
])

# =====================================================================
# TAB 1: SYSTEM CASCADE
# =====================================================================
with tab_system:
    from cascade_predict.templates import list_templates, get_template, get_failure_db
    from cascade_predict.graph import CascadeEngine
    from cascade_predict.spec_parser import parse_csv, parse_pdf, parse_text, extract_pdf_text
    from cascade_predict.cad_parser import parse_cad_file, CADAnalysisResult
    from cascade_predict.components.cad_viewer import render_cad_viewer
    from cascade_predict.part_identifier import (
        SECTORS, get_sectors, get_parts_for_sector, get_part_profile,
        auto_suggest_part, build_custom_part_profile,
    )
    from cascade_predict.drawing_reader.parser import parse_drawing, DrawingParseResult

    # ── Session state: separate results "page" ──────────────────────
    if "cascade_inputs" not in st.session_state:
        st.session_state.cascade_inputs = None

    # ═════════════════════════════════════════════════════════════════
    # RESULTS PAGE — shown when cascade has been run
    # ═════════════════════════════════════════════════════════════════
    if st.session_state.cascade_inputs is not None:
        _ci = st.session_state.cascade_inputs

        # Back button
        _back_cols = st.columns([1, 5])
        with _back_cols[0]:
            if st.button("← Back to Configuration", key="back_to_config"):
                st.session_state.cascade_inputs = None
                st.rerun()

        # Rebuild graph + compute results from stored inputs
        if _ci["using_auto"]:
            from cascade_predict.auto_assembler import auto_assemble_graph, PartGeometry
            _assembled = auto_assemble_graph(
                geometry=_ci["geo"], material_key=_ci["sel_mat"],
                sector=_ci["sector"],
                part_name=_ci["part_name"],
                has_power_system=_ci["power_kw"] > 0,
                power_kw=_ci["power_kw"],
                speed_m_s=_ci["speed_ms"],
                baseline_range=_ci["range_km"],
            )
            graph = _assembled.graph
            components = {c.component_id: c for c in _assembled.components}
            constraints = _assembled.constraints
            comp = _assembled.components[0]
        else:
            tmpl = get_template(_ci["template_id"])
            graph, components, constraints = tmpl.build()

        if not _ci["using_auto"] and _ci["comp_id"] == "_custom":
            node = graph.nodes[_ci["target_node"]]
            from cascade_predict.subsystems.component import Component as CompClass
            comp = CompClass("_custom", _ci["comp_name"], node.subsystem)
            comp.add_property(_ci["target_node"], node.value, node.unit,
                              graph_node_id=_ci["target_node"], source="custom")
        elif not _ci["using_auto"]:
            comp = components[_ci["comp_id"]]

        selected_comp_id = _ci["comp_id"]
        selected_prop = _ci["prop_name"]
        new_value = _ci["new_value"]
        selected_sector = _ci["sector"]
        selected_template_id = _ci["template_id"]
        _change_cause = _ci["change_cause"]
        _change_severity = _ci["change_severity"]
        _regulatory_involved = _ci["regulatory_involved"]
        _estimated_cost = _ci["estimated_cost"]
        _cost_method = _ci["cost_method"]
        _run_bayesian = _ci["run_bayesian"]
        _mc_samples = _ci["mc_samples"]
        _using_auto = _ci["using_auto"]
        prop = comp.properties[selected_prop]

        # Apply constraint relaxations
        if constraints and _ci.get("constraint_relaxations"):
            for node_id, new_limit in _ci["constraint_relaxations"].items():
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
            st.stop()

        engine = CascadeEngine(graph)
        trigger_node = list(deltas.keys())[0]
        trigger_delta = list(deltas.values())[0]
        result = engine.propagate(trigger_node, trigger_delta, mode="single_pass")
        summary = result.summary(total_graph_nodes=len(graph.nodes))

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
        editable_props = {
            name: p for name, p in comp.properties.items()
            if name in comp.property_to_node
        }

        st.markdown(
            "<div style='background:#b8452a; padding:12px 20px; border-radius:8px; "
            "margin:10px 0 20px 0;'>"
            "<h2 style='color:white; margin:0;'>Cascade Prediction Results</h2>"
            f"<p style='color:#ffe8d6; margin:4px 0 0 0;'>"
            f"Change: <b>{selected_prop.replace('_',' ').title()}</b> on <b>{comp.name}</b> "
            f"({prop.value:.4f} → {new_value:.4f} {prop.unit})</p>"
            "</div>",
            unsafe_allow_html=True,
        )

        # ── Cost Variance Prediction ─────────────────────────────
        from cascade_predict.cost_predictor import (
            get_predictor, ChangeRequest, CostPrediction,
            compute_propagation_score,
        )
        _predictor = get_predictor()
        # Auto-calculate cost from cascade if user didn't provide one
        if _cost_method == "auto" and cost_steps:
            _total_cost_step = [s for s in cost_steps if s.target_node == "total_cost_delta"]
            if _total_cost_step:
                _estimated_cost = max(1000, abs(_total_cost_step[0].delta_output))
        _change_req = ChangeRequest(
            change_type=comp.subsystem if comp else "structural",
            affected_subsystems=list(set(s.target_subsystem for s in unique_steps)),
            change_cause=_change_cause,
            severity=_change_severity,
            regulatory_involved=_regulatory_involved,
            estimated_cost=_estimated_cost,
            propagation_score=summary["propagation_score"],
            n_nodes_affected=summary["nodes_affected"],
            n_subsystems_affected=summary["subsystems_affected"],
            n_violations=summary["violations"],
            n_cross_domain_hops=summary["cross_domain_hops"],
            cascade_depth=summary["cascade_depth"],
            max_pct_change=summary["max_pct_change"],
            sector=selected_sector,
            duration_days=60,
        )
        _cost_pred = _predictor.predict(_change_req)

        # ── Result tabs ──────────────────────────────────────────
        result_tab_names = ["Overview", "Cascade Flow", "Violations",
                            "Cost & Schedule", "Risk Assessment", "Comparison"]
        if _run_bayesian:
            result_tab_names.append("Uncertainty")
        result_tabs = st.tabs(result_tab_names)

        # ── TAB: Overview ────────────────────────────────────────
        with result_tabs[0]:
            m1, m2, m3, m4, m5 = st.columns(5)
            m1.metric("Parameters Affected", summary["nodes_affected"])
            m2.metric("Subsystems Hit", summary["subsystems_affected"])
            m3.metric("Cross-Domain Hops", summary["cross_domain_hops"])
            n_violations = summary["violations"]
            m4.metric("Violations", n_violations,
                       delta=f"{n_violations} cert issues" if n_violations > 0 else "Clean",
                       delta_color="inverse")
            _tier_colors = {1: "normal", 2: "normal", 3: "inverse", 4: "inverse"}
            m5.metric("Risk Tier", f"T{_cost_pred.risk_tier}",
                      delta=_cost_pred.risk_label,
                      delta_color=_tier_colors.get(_cost_pred.risk_tier, "normal"))

            # Propagation score bar
            st.markdown(
                f"**Propagation Score:** `{summary['propagation_score']:.2f}` / 1.00 "
                f"— {'Low' if summary['propagation_score'] < 0.3 else 'Moderate' if summary['propagation_score'] < 0.6 else 'High'} cascade severity"
            )
            st.progress(min(1.0, summary["propagation_score"]))

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
            with st.expander("How to read this graph", expanded=False):
                st.markdown("""
**Sankey Diagram — Reading Guide**

- **Each node (vertical bar)** represents an engineering parameter (e.g. hull thickness, total mass, drag force).
- **Each flow (link)** represents a physics coupling — energy, force, or material relationship that transmits a design change from one parameter to another.
- **Flow width** is proportional to the magnitude of the change transmitted.
- **Flow left to right** shows the propagation direction: the trigger parameter is on the far left, downstream effects flow rightward.

**Color coding:**
- **Blue flows** — same-subsystem propagation (e.g. structural → structural)
- **Orange flows** — cross-domain hop (e.g. structural → thermal). These are the most important to watch — they indicate the change has jumped to a different engineering discipline.
- **Red flows** — the downstream parameter **violates a certification limit** (regulatory, safety, or design constraint).

**Node colors** correspond to subsystems (structural, thermal, electrical, performance, cost, etc.). The legend appears below the Comparison table.

**What to look for:**
1. **Wide red flows** = large violations triggered by your change
2. **Many orange hops** = the change cascades across multiple disciplines
3. **Fan-out points** = a single parameter driving many downstream effects (high coupling)
""")

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
                product_type=tmpl.industry.lower() if tmpl else selected_sector,
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

        # ── TAB: Risk Assessment ─────────────────────────────────
        with result_tabs[4]:
            st.subheader("Risk Assessment & Cost Prediction")

            # Risk tier banner
            _tier_banner = {
                1: ("success", "Tier 1 — Fast-track. Low cost overrun risk. Proceed with standard approval."),
                2: ("info", "Tier 2 — Standard Review. Moderate overrun risk. Standard review process recommended."),
                3: ("warning", "Tier 3 — Senior Review. High overrun risk. Senior engineering review required."),
                4: ("error", "Tier 4 — Deep Analysis. Very high overrun risk. Detailed cost & schedule analysis needed."),
            }
            _banner_fn = {"success": st.success, "info": st.info, "warning": st.warning, "error": st.error}
            _btype, _bmsg = _tier_banner[_cost_pred.risk_tier]
            _banner_fn[_btype](_bmsg)

            rc1, rc2, rc3, rc4 = st.columns(4)
            rc1.metric("Predicted Overrun", f"{_cost_pred.predicted_variance_pct:.1f}%")
            rc2.metric("Estimated Cost", f"${_estimated_cost:,.0f}")
            rc3.metric("Predicted Actual", f"${_cost_pred.predicted_actual_cost:,.0f}")
            rc4.metric("Model Confidence (R²)", f"{_cost_pred.confidence:.2f}")

            # Feature importance
            if _cost_pred.feature_importance:
                st.markdown("##### What drives the overrun prediction?")
                fi_labels = [k.replace("_", " ").title() for k in _cost_pred.feature_importance.keys()]
                fi_values = list(_cost_pred.feature_importance.values())
                fig_fi = go.Figure(go.Bar(
                    x=fi_values, y=fi_labels, orientation="h",
                    marker_color="#d4725c",
                ))
                fig_fi.update_layout(
                    height=max(200, 35 * len(fi_labels)),
                    xaxis_title="Feature Importance",
                    margin=dict(l=200, r=40, t=20, b=40),
                    yaxis=dict(autorange="reversed"),
                )
                st.plotly_chart(fig_fi, use_container_width=True)

            # Propagation details
            st.markdown("##### Cascade Severity Breakdown")
            ps_col1, ps_col2, ps_col3, ps_col4 = st.columns(4)
            ps_col1.metric("Propagation Score", f"{summary['propagation_score']:.2f}")
            ps_col2.metric("Max Parameter Change", f"{summary['max_pct_change']:.1f}%")
            ps_col3.metric("Cascade Depth", summary["cascade_depth"])
            ps_col4.metric("Cross-Domain Hops", summary["cross_domain_hops"])

            # Feedback capture
            st.markdown("---")
            st.markdown("##### Feedback (after implementation)")
            st.caption("Record actual cost to improve future predictions.")
            fb_col1, fb_col2 = st.columns(2)
            with fb_col1:
                _actual_cost = st.number_input(
                    "Actual cost ($) — fill in after implementation",
                    value=0.0, min_value=0.0,
                    step=1000.0, format="%.0f",
                    key="actual_cost_feedback",
                )
            with fb_col2:
                if _actual_cost > 0 and _estimated_cost > 0:
                    _actual_variance = (_actual_cost - _estimated_cost) / _estimated_cost * 100
                    _pred_error = _actual_variance - _cost_pred.predicted_variance_pct
                    st.metric("Actual Variance", f"{_actual_variance:+.1f}%",
                              delta=f"Prediction error: {_pred_error:+.1f}%")

        # ── TAB: Comparison ──────────────────────────────────────
        with result_tabs[5]:
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
            with result_tabs[6]:
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



        st.stop()



    st.title("Design Change Cascade Prediction")
    st.caption(
        "Upload a spec document, select a system template, change a property, "
        "and watch the cascade ripple across subsystems."
    )

    # ── STEP 1: Upload Documents ─────────────────────────────────────
    st.header("1  Upload Spec / CAD / Drawing")

    upload_col1, upload_col2, upload_col3 = st.columns(3)

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

    # --- 2D Drawing upload ---
    with upload_col3:
        st.subheader("2D Drawing")
        uploaded_drawing = st.file_uploader(
            "PNG, JPEG, or PDF engineering drawing",
            type=["png", "jpg", "jpeg", "tiff", "tif", "bmp", "pdf"],
            key="drawing_upload",
        )

    # --- Sample Library (no upload needed) ---
    from cascade_predict.sample_library import get_sample_library, get_sample_by_name
    with st.expander("Or pick from the Sample Library (no file needed)", expanded=False):
        _samples = get_sample_library()
        _sample_names = ["(none)"] + [s.name for s in _samples]
        _sample_descs = {s.name: f"**{s.sector.title()}** — {s.description}" for s in _samples}
        _sel_sample = st.selectbox(
            "Select a sample CAD model",
            _sample_names,
            key="sample_select",
        )
        if _sel_sample != "(none)":
            st.markdown(_sample_descs[_sel_sample])

    # Load sample as if it was uploaded
    _sample_cad_data = None
    _sample_cad_name = None
    if _sel_sample != "(none)" and uploaded_cad is None:
        sample = get_sample_by_name(_sel_sample)
        if sample.file_type == "stl":
            _sample_cad_data = sample.data
            _sample_cad_name = sample.name.split("(")[0].strip().replace(" ", "_") + ".stl"

    # --- Process 2D Drawing ---
    drawing_result = None  # type: DrawingParseResult | None
    if uploaded_drawing is not None:
        drawing_raw = uploaded_drawing.read()
        drawing_result = parse_drawing(drawing_raw, uploaded_drawing.name)

        st.markdown("---")
        st.subheader("2D Drawing Analysis")

        # Show the uploaded image
        if not uploaded_drawing.name.lower().endswith(".pdf"):
            st.image(drawing_raw, caption=uploaded_drawing.name, use_container_width=True)

        if drawing_result.dimensions:
            st.success(f"Extracted **{len(drawing_result.dimensions)}** dimension(s) from drawing.")
            dim_table = []
            for d in drawing_result.dimensions:
                tol_str = ""
                if d.tolerance_plus != 0 or d.tolerance_minus != 0:
                    tol_str = f"+{d.tolerance_plus}/{d.tolerance_minus}"
                dim_table.append({
                    "Type": d.dim_type.title(),
                    "Value": d.value,
                    "Unit": d.unit,
                    "Tolerance": tol_str,
                    "Label": d.label.replace("_", " ").title() if d.label else "",
                    "Confidence": f"{d.confidence:.0%}",
                })
            st.dataframe(dim_table, use_container_width=True, hide_index=True)
        else:
            st.info("No dimensions auto-detected. Ensure the drawing has clear dimension callouts.")

        if drawing_result.notes:
            with st.expander(f"Drawing Notes ({len(drawing_result.notes)})"):
                for n in drawing_result.notes:
                    st.markdown(f"- **[{n.category}]** {n.value or n.text} (conf: {n.confidence:.0%})")

        if drawing_result.material:
            st.info(f"Detected material: **{drawing_result.material}**")

        if not drawing_result.ocr_available:
            st.warning("OCR (Tesseract) not available. Install `tesseract-ocr` and `pytesseract` for better results.")

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
    # Use uploaded CAD or sample library
    _cad_source_raw = None
    _cad_source_name = None
    if uploaded_cad is not None:
        _cad_source_raw = uploaded_cad.read()
        _cad_source_name = uploaded_cad.name
    elif _sample_cad_data is not None:
        _cad_source_raw = _sample_cad_data
        _cad_source_name = _sample_cad_name
        st.success(f"Loaded sample: **{_sel_sample}**")

    if _cad_source_raw is not None:
        cad_result = parse_cad_file(_cad_source_name, _cad_source_raw)

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
    if uploaded_spec is None and uploaded_cad is None and uploaded_drawing is None and not _has_onshape and _sample_cad_data is None:
        st.info("Upload a spec document, a CAD file (STL/STEP), a 2D drawing, paste an Onshape URL, or pick from the Sample Library above. You can also skip directly to Step 3.")

    # ── STEP 2: Identify Part ────────────────────────────────────────
    st.markdown("---")
    st.header("2  Identify Your Part")

    # Auto-suggest if CAD was uploaded
    _suggestion = None
    if cad_result and cad_result.parameters:
        _suggestion = auto_suggest_part(cad_result.parameters)

    # Use 2D drawing sector as fallback hint
    _drawing_sector = None
    if drawing_result and drawing_result.sector:
        _drawing_sector = drawing_result.sector

    sector_options = get_sectors()
    default_sector_idx = 0
    if _suggestion:
        for i, (k, _) in enumerate(sector_options):
            if k == _suggestion[0]:
                default_sector_idx = i
                break
    elif _drawing_sector:
        for i, (k, _) in enumerate(sector_options):
            if k == _drawing_sector:
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

    from cascade_predict.system_map import get_subtypes, get_system_map, get_zone_by_id
    from cascade_predict.graph_assembler import (
        assemble_graph as auto_assemble_graph,
        PartGeometry,
        geometry_from_cad_params,
        geometry_from_drawing,
    )
    from cascade_predict.physics_models.universal import list_materials, lookup_material

    # ── Change Context (drives cost/risk prediction) ─────────────────
    with st.expander("Change Context", expanded=True):
        st.caption("Fill in what you know — the system infers the rest.")

        ctx_col1, ctx_col2, ctx_col3 = st.columns(3)
        with ctx_col1:
            _change_cause = st.selectbox(
                "Why is this change happening?",
                ["design_optimization", "customer_requirement", "integration_issue",
                 "supplier_change", "regulatory", "field_failure",
                 "cost_reduction", "performance_improvement", "not_sure"],
                format_func=lambda x: x.replace("_", " ").title(),
                key="change_cause",
            )
        with ctx_col2:
            _change_severity = st.selectbox(
                "How big is this change?",
                ["medium", "high", "low", "not_sure"],
                format_func=lambda x: {"high": "Major — affects multiple systems",
                                        "medium": "Moderate — affects nearby systems",
                                        "low": "Minor — localized change",
                                        "not_sure": "Not sure yet"}.get(x, x.title()),
                key="change_severity",
            )
        with ctx_col3:
            _regulatory_involved = st.checkbox(
                "Might need re-certification?",
                value=False,
                key="regulatory_involved",
                help="Check if this change could affect regulatory compliance (FAR, DNV, ISO, etc.)",
            )
        if _change_severity == "not_sure":
            _change_severity = "medium"

        # ── System Sub-Type & Visual Zone Selector ───────────────
        st.markdown("---")
        st.markdown("##### Where does this part sit in the system?")

        _subtypes = get_subtypes(selected_sector)
        _subtype_col, _zone_col = st.columns([1, 2])
        with _subtype_col:
            _selected_subtype = st.selectbox(
                "System type",
                [k for k, _ in _subtypes],
                format_func=lambda x: dict(_subtypes).get(x, x),
                key="system_subtype",
            )
        _sys_map = get_system_map(_selected_subtype)

        # Zone selection with visual layout
        _selected_zone = None
        _zone_data = None
        if _sys_map:
            with _zone_col:
                _zone_options = [("not_sure", "I'm not sure / not listed")] + [
                    (z.zone_id, f"{z.label} — {z.description}") for z in _sys_map.zones
                ]
                _selected_zone_id = st.selectbox(
                    "Select zone",
                    [k for k, _ in _zone_options],
                    format_func=lambda x: dict(_zone_options).get(x, x),
                    key="system_zone",
                )
                if _selected_zone_id != "not_sure":
                    _zone_data = get_zone_by_id(_sys_map, _selected_zone_id)

            # Draw the system map
            _map_fig = go.Figure()

            for z in _sys_map.zones:
                _is_selected = z.zone_id == _selected_zone_id
                _fill_color = "#d4725c" if _is_selected else z.color
                _line_width = 3 if _is_selected else 1
                _line_color = "#8b0000" if _is_selected else "#888"

                _map_fig.add_shape(
                    type="rect",
                    x0=z.x - z.width/2, y0=1 - z.y - z.height/2,
                    x1=z.x + z.width/2, y1=1 - z.y + z.height/2,
                    fillcolor=_fill_color,
                    line=dict(color=_line_color, width=_line_width),
                    opacity=0.85,
                )
                _map_fig.add_annotation(
                    x=z.x, y=1 - z.y,
                    text=f"<b>{z.label}</b>",
                    showarrow=False,
                    font=dict(size=10, color="#2b2b2b"),
                )

            _map_fig.update_layout(
                title=dict(text=f"{_sys_map.label}", font=dict(size=14)),
                height=350,
                xaxis=dict(range=[-0.05, 1.05], showgrid=False, zeroline=False,
                           showticklabels=False),
                yaxis=dict(range=[-0.05, 1.05], showgrid=False, zeroline=False,
                           showticklabels=False, scaleanchor="x"),
                margin=dict(l=10, r=10, t=40, b=10),
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
            )
            st.plotly_chart(_map_fig, use_container_width=True, key="system_map_chart")

            # Show zone details when selected
            if _zone_data:
                zd = _zone_data
                st.success(
                    f"**{zd.label}** — {zd.description}  \n"
                    f"Location: {zd.location_type.replace('_', ' ').title()} · "
                    f"Loads: {', '.join(lt.replace('_', ' ') for lt in zd.load_types)} · "
                    f"Temp: {zd.operating_temp_c}°C · "
                    f"Environment: {zd.environment}"
                )
                if zd.standards:
                    st.caption(f"Applicable standards: {', '.join(zd.standards)}")

        # Pull values from zone or use defaults
        if _zone_data:
            _part_location = _zone_data.location_type
            _load_type = _zone_data.load_types[0] if _zone_data.load_types else "combined"
            _operating_temp = _zone_data.operating_temp_c
            _auto_interfaces = _zone_data.interfaces
        else:
            _SECTOR_DEFAULTS = {
                "aerospace": ("internal_structural_primary", "cyclic_fatigue", 40.0),
                "naval": ("external_exposed", "cyclic_fatigue", 35.0),
                "automotive_ev": ("internal_structural_primary", "vibration", 45.0),
                "robotics": ("internal_structural_primary", "cyclic_fatigue", 30.0),
            }
            _def = _SECTOR_DEFAULTS.get(selected_sector, ("internal_structural_primary", "combined", 25.0))
            _part_location, _load_type, _operating_temp = _def
            _SECTOR_DEFAULT_INTERFACES = {
                "aerospace": ["structural_frame", "electrical_harness", "thermal_management", "control_system"],
                "naval": ["structural_frame", "cooling_system", "electrical_harness", "external_environment"],
                "automotive_ev": ["structural_frame", "cooling_system", "electrical_harness", "software_controls"],
                "robotics": ["structural_frame", "electrical_harness", "control_system", "sensors"],
            }
            _auto_interfaces = _SECTOR_DEFAULT_INTERFACES.get(selected_sector, ["structural_frame"])

        # Show overridable details
        with st.expander("Override auto-detected environment (optional)", expanded=False):
            _ov_col1, _ov_col2, _ov_col3 = st.columns(3)
            with _ov_col1:
                _operating_temp = st.number_input(
                    "Operating temp (°C)", value=_operating_temp,
                    min_value=-60.0, max_value=500.0, step=5.0, key="operating_temp",
                )
            with _ov_col2:
                _load_type = st.selectbox(
                    "Primary loads",
                    ["static", "cyclic_fatigue", "impact", "thermal_cycling",
                     "pressure", "vibration", "torsion", "combined"],
                    index=["static", "cyclic_fatigue", "impact", "thermal_cycling",
                           "pressure", "vibration", "torsion", "combined"].index(
                        _load_type if _load_type in ["static", "cyclic_fatigue", "impact",
                        "thermal_cycling", "pressure", "vibration", "torsion", "combined"]
                        else "combined"
                    ),
                    format_func=lambda x: x.replace("_", " ").title(),
                    key="load_type_override",
                )
            with _ov_col3:
                _part_location = st.selectbox(
                    "Location type",
                    ["external_exposed", "external_submerged", "external_pressurized",
                     "internal_structural_primary", "internal_structural_secondary",
                     "internal_non_structural", "interface_boundary", "moving_joint",
                     "high_vibration_zone"],
                    index=max(0, ["external_exposed", "external_submerged", "external_pressurized",
                     "internal_structural_primary", "internal_structural_secondary",
                     "internal_non_structural", "interface_boundary", "moving_joint",
                     "high_vibration_zone"].index(_part_location)
                     if _part_location in ["external_exposed", "external_submerged",
                     "external_pressurized", "internal_structural_primary",
                     "internal_structural_secondary", "internal_non_structural",
                     "interface_boundary", "moving_joint", "high_vibration_zone"] else 3),
                    format_func=lambda x: x.replace("_", " ").title(),
                    key="location_override",
                )

        # ── Supplier Details ─────────────────────────────────────
        st.markdown("##### Supplier Information")
        st.caption("Optional — helps estimate lead time, cost impact, and supply chain risk.")
        sup_col1, sup_col2, sup_col3 = st.columns(3)
        with sup_col1:
            _current_supplier = st.text_input(
                "Current supplier",
                placeholder="e.g. Arconic, Timet, local shop...",
                key="current_supplier",
            )
        with sup_col2:
            _new_supplier = st.text_input(
                "New / proposed supplier (if changing)",
                placeholder="Leave blank if same supplier",
                key="new_supplier",
            )
        with sup_col3:
            _supplier_change_reason = st.selectbox(
                "Supplier change reason",
                ["no_change", "cost_reduction", "lead_time", "quality_issue",
                 "sole_source_risk", "capacity", "customer_mandate", "other"],
                format_func=lambda x: x.replace("_", " ").title() if x != "no_change" else "No supplier change",
                key="supplier_reason",
            )
        _is_supplier_change = _change_cause == "supplier_change" or _new_supplier.strip() != ""

        # ── Connected Systems (auto-from zone or sector) ─────────
        st.markdown("##### Connected Systems")
        st.caption("Auto-selected from zone. Adjust if needed.")
        _ALL_INTERFACES = [
            "structural_frame", "cooling_system", "electrical_harness",
            "hydraulic_lines", "control_system", "propulsion",
            "sensors", "human_operator", "external_environment",
            "adjacent_parts", "mounting_hardware", "seals_gaskets",
            "thermal_management", "software_controls", "fuel_system",
        ]
        _interfaces = st.multiselect(
            "Interfaces / connections",
            _ALL_INTERFACES,
            default=[i for i in _auto_interfaces if i in _ALL_INTERFACES],
            format_func=lambda x: x.replace("_", " ").title(),
            key="part_interfaces",
            help="Pre-filled from zone selection. Add or remove as needed.",
        )

        # ── Cost Estimation (auto-calculated) ────────────────────
        st.markdown("##### Cost Estimate")
        _cost_method = st.radio(
            "How to estimate change cost?",
            ["auto", "manual"],
            format_func=lambda x: {
                "auto": "Auto-estimate from material, mass, and sector rates",
                "manual": "I have my own estimate",
            }[x],
            horizontal=True,
            key="cost_method",
        )
        if _cost_method == "manual":
            _estimated_cost = st.number_input(
                "Your estimated change cost ($)",
                value=10000.0, min_value=0.0,
                step=1000.0, format="%.0f",
                key="estimated_cost",
            )
        else:
            _SECTOR_BASE_COST = {
                "aerospace": 25000.0, "naval": 15000.0,
                "automotive_ev": 5000.0, "robotics": 3000.0,
            }
            _estimated_cost = _SECTOR_BASE_COST.get(selected_sector, 10000.0)
            st.caption(f"Base estimate: ${_estimated_cost:,.0f} (will be refined after cascade runs)")

    all_templates = list_templates()
    template_names = {t.template_id: f"{t.name} ({t.industry})" for t in all_templates}
    # Add auto-assemble option
    template_names["_auto"] = "Auto-Assemble from Geometry (no template needed)"

    # Auto-select template from part profile, or auto-assemble if geometry available
    default_tmpl_idx = 0
    _has_geometry = (cad_result and cad_result.parameters) or (drawing_result and drawing_result.dimensions)
    if _has_geometry and not part_profile:
        # Default to auto-assemble when we have geometry but no known part
        default_tmpl_idx = list(template_names.keys()).index("_auto")
    elif part_profile:
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

    # ── AUTO-ASSEMBLE path ──────────────────────────────────────────
    _using_auto = selected_template_id == "_auto"
    if _using_auto:
        st.info("Auto-assembling cascade graph from geometry + material. No hand-coded template needed.")

        # Determine geometry
        _geo = PartGeometry()
        if cad_result and cad_result.parameters:
            _geo = geometry_from_cad_params(cad_result.parameters)
        elif drawing_result and drawing_result.dimensions:
            _geo = geometry_from_drawing(drawing_result.dimensions)

        # Material selection
        _materials = list_materials()
        _mat_names = {k: f"{display} (ρ={dens} kg/m³)" for k, display, dens in _materials}

        # Try to auto-detect material from drawing
        _default_mat_idx = 0
        if drawing_result and drawing_result.material:
            _detected_mat = lookup_material(drawing_result.material)
            if _detected_mat:
                for i, (k, _, _) in enumerate(_materials):
                    if lookup_material(k) == _detected_mat:
                        _default_mat_idx = i
                        break

        col_mat, col_perf = st.columns(2)
        with col_mat:
            _sel_mat = st.selectbox(
                "Material",
                list(_mat_names.keys()),
                format_func=lambda x: _mat_names[x],
                index=_default_mat_idx,
                key="auto_material",
            )
        with col_perf:
            _part_name = st.text_input(
                "Part name",
                value=part_profile.part_label if part_profile else "Part",
                key="auto_part_name",
            )

        # Optional: power/range context
        with st.expander("Performance Context (optional)", expanded=False):
            _col_pw, _col_sp, _col_rg = st.columns(3)
            with _col_pw:
                _power_kw = st.number_input("System power (kW)", value=0.0, min_value=0.0, key="auto_power")
            with _col_sp:
                _speed_ms = st.number_input("Operating speed (m/s)", value=0.0, min_value=0.0, key="auto_speed")
            with _col_rg:
                _range_km = st.number_input("Baseline range (km)", value=0.0, min_value=0.0, key="auto_range")

        # Assemble
        _assembled = auto_assemble_graph(
            geometry=_geo,
            material_key=_sel_mat,
            sector=selected_sector,
            part_name=_part_name,
            has_power_system=_power_kw > 0,
            power_kw=_power_kw,
            speed_m_s=_speed_ms,
            baseline_range=_range_km,
        )
        graph = _assembled.graph
        components = {c.component_id: c for c in _assembled.components}
        constraints = _assembled.constraints

        with st.expander("Auto-Assembled Graph Summary", expanded=True):
            st.text(_assembled.summary)
            st.markdown(f"**Subsystems:** {', '.join(graph.subsystems())}")
            st.markdown(f"**Nodes:** {len(graph.nodes)} | **Edges:** {len(graph.edges)}")

        # Skip to property selection (no component/template choice needed)
        comp = _assembled.components[0]
        selected_comp_id = comp.component_id
        tmpl = None  # no template object for auto-assembled

    else:
        # ── TEMPLATE path (existing behavior) ──────────────────────
        tmpl = get_template(selected_template_id)
        graph, components, constraints = tmpl.build()

    if not _using_auto:
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
          st.markdown("##### Describe Your Component")
          col_ccomp_name, col_ccomp_desc = st.columns(2)
          with col_ccomp_name:
              custom_comp_name = st.text_input(
                  "Component name",
                  placeholder="e.g. Sonar Dome, Gearbox Housing, Cable Tray...",
                  key="custom_comp_name",
              )
          with col_ccomp_desc:
              custom_comp_desc_text = st.text_input(
                  "What does it do? (brief)",
                  placeholder="e.g. Protects sonar array from seawater, supports 200kg radar...",
                  key="custom_comp_desc_text",
              )

          # Smart parameter matching — guess from name/description, don't force selection
          graph_node_ids = sorted([nid for nid in graph.nodes.keys()
                                    if nid not in {"material_cost", "manufacturing_cost", "tooling_cost",
                                                    "total_cost_delta", "manufacturing_lead_time",
                                                    "certification_time", "total_schedule_delta"}])

          # Auto-suggest: find nodes whose names overlap with user's description
          _search_text = (custom_comp_name + " " + custom_comp_desc_text).lower()
          _keyword_scores = {}
          for nid in graph_node_ids:
              node = graph.nodes[nid]
              score = 0
              for word in nid.split("_"):
                  if len(word) > 2 and word in _search_text:
                      score += 1
              for word in node.description.lower().split():
                  if len(word) > 3 and word in _search_text:
                      score += 0.5
              _keyword_scores[nid] = score
          _sorted_nodes = sorted(graph_node_ids, key=lambda x: -_keyword_scores.get(x, 0))
          _best_match = _sorted_nodes[0] if _keyword_scores.get(_sorted_nodes[0], 0) > 0 else graph_node_ids[0]

          st.markdown("##### Which parameter does this affect?")
          st.caption("Best guess is pre-selected. Change if needed, or describe below if none fit.")
          _default_idx = _sorted_nodes.index(_best_match)
          target_node = st.selectbox(
              "Closest matching parameter",
              _sorted_nodes,
              format_func=lambda x: f"{x.replace('_', ' ').title()} — {graph.nodes[x].description} ({graph.nodes[x].value:.2f} {graph.nodes[x].unit})",
              index=_default_idx,
              key="custom_target_node",
          )

          # Free-text fallback for when nothing fits
          _custom_param_desc = st.text_input(
              "Or describe the parameter (if none of the above match)",
              placeholder="e.g. I'm changing the mounting bolt torque from 50Nm to 80Nm...",
              key="custom_param_freetext",
          )
          if _custom_param_desc:
              st.info("Free-text parameters will be matched to the closest graph node. "
                      "For best results, also select the nearest match above.")

          # Build a temporary component with a single editable property
          node = graph.nodes[target_node]
          comp = CompClass(
              component_id="_custom",
              name=custom_comp_name or "Custom Component",
              subsystem=node.subsystem,
              description=custom_comp_desc_text or "User-defined component",
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
    st.markdown("##### What are you changing?")
    editable_props = {
        name: prop for name, prop in comp.properties.items()
        if name in comp.property_to_node
    }

    if not editable_props:
        st.warning("This component has no graph-linked properties to change.")
        selected_prop = None
    else:
        # Build descriptions for each property
        _prop_descs = {}
        for pname, pobj in editable_props.items():
            gnode_id = comp.property_to_node.get(pname, "")
            gnode = graph.nodes.get(gnode_id)
            desc = gnode.description if gnode and gnode.description else ""
            _prop_descs[pname] = desc

        col_prop, col_val = st.columns(2)
        with col_prop:
            selected_prop = st.selectbox(
                "Property to change",
                list(editable_props.keys()),
                format_func=lambda x: (
                    f"{x.replace('_', ' ').title()} "
                    f"({editable_props[x].value} {editable_props[x].unit})"
                    + (f" — {_prop_descs[x]}" if _prop_descs.get(x) else "")
                ),
            )
        prop = editable_props[selected_prop]
        with col_val:
            # Ensure value and step are both float to avoid Streamlit type error
            _prop_val = float(prop.value)
            _prop_step = abs(_prop_val) * 0.05 if _prop_val != 0 else 0.1
            new_value = st.number_input(
                f"New value ({prop.unit})",
                value=_prop_val,
                step=_prop_step,
                format="%.4f",
            )
            delta = new_value - prop.value
            if abs(delta) > 1e-10:
                pct = (delta / prop.value * 100) if prop.value != 0 else float("inf")
                st.metric("Change", f"{delta:+.4f} {prop.unit}", f"{pct:+.1f}%")

        # Describe a change not in the list
        with st.expander("My change isn't listed above", expanded=False):
            _unlisted_desc = st.text_area(
                "Describe what you want to change",
                placeholder="e.g. I'm changing the bolt pattern from 4x M10 to 6x M8, "
                            "or switching adhesive from epoxy to polyurethane...",
                key="unlisted_change",
                height=80,
            )
            if _unlisted_desc:
                st.info(
                    "For now, pick the **closest matching property** above and adjust the value. "
                    "The cascade will propagate from that parameter. Your description is noted "
                    "and will improve future parameter matching."
                )

    # ── CAD Models ───────────────────────────────────────────────────
    if selected_template_id in _ONSHAPE_MODELS:
        with st.expander("CAD Models (Onshape)", expanded=False):
            cad_models = _ONSHAPE_MODELS[selected_template_id]
            cad_tabs = st.tabs(list(cad_models.keys()))
            for ctab, (label, url) in zip(cad_tabs, cad_models.items()):
                with ctab:
                    st_components.iframe(url, height=500, scrolling=True)

    # ── Quick Scenarios ──────────────────────────────────────────────
    if tmpl and tmpl.presets:
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
    # STORE INPUTS & SWITCH TO RESULTS PAGE
    # ═════════════════════════════════════════════════════════════════
    if run_clicked and selected_prop and new_value is not None:
        st.session_state.cascade_inputs = {
            "template_id": selected_template_id,
            "sector": selected_sector,
            "using_auto": _using_auto,
            "comp_id": selected_comp_id,
            "comp_name": comp.name if comp else "Custom",
            "prop_name": selected_prop,
            "new_value": new_value,
            "change_cause": _change_cause,
            "change_severity": _change_severity,
            "regulatory_involved": _regulatory_involved,
            "estimated_cost": _estimated_cost,
            "cost_method": _cost_method,
            "run_bayesian": _run_bayesian,
            "mc_samples": _mc_samples,
            "constraint_relaxations": _constraint_relaxations,
        }
        # Store auto-assemble params if needed
        if _using_auto:
            st.session_state.cascade_inputs.update({
                "geo": _geo, "sel_mat": _sel_mat, "part_name": _part_name,
                "power_kw": _power_kw, "speed_ms": _speed_ms, "range_km": _range_km,
            })
        # Store custom comp params if needed
        if selected_comp_id == "_custom":
            st.session_state.cascade_inputs["target_node"] = target_node
        st.rerun()


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

# =====================================================================
# TAB 3: HOW IT WORKS
# =====================================================================
with tab_about:
    st.title("How It Works")
    st.caption("Overview of the prediction methodology and data sources powering this platform.")

    about_tab1, about_tab2, about_tab3, about_tab4 = st.tabs([
        "Methodology", "Data Sources", "Supported Sectors", "Architecture",
    ])

    # ── Methodology ──────────────────────────────────────────────────
    with about_tab1:
        st.subheader("Deterministic Cascade Propagation")
        st.markdown("""
This platform predicts the downstream engineering impact of a design change
using **deterministic physics-based graph propagation** — no LLMs, no
machine learning in the cascade path.

**How it works:**

1. **Directed Graph** — Every system is modeled as a directed acyclic graph (DAG).
   Nodes are engineering parameters (mass, stress, temperature, drag, cost, etc.).
   Edges encode physics coupling equations.

2. **Breadth-First Propagation** — When you change a parameter, the engine walks
   the graph layer by layer. Each edge applies a parameterized physics equation
   (e.g. `Δmass = ρ × A × Δt` for thickness→mass) to compute the downstream delta.

3. **Constraint Checking** — After propagation, every node is checked against
   regulatory limits (e.g. DNV, FAR 25, ISO 10218). Violations are flagged with
   the specific standard and section.

4. **Cost & Schedule Roll-up** — Engineering deltas feed into cost models
   (material, manufacturing, tooling) and schedule models (lead time, certification).

**Key properties:**
- Fully reproducible — same input always produces same output
- Transparent — every edge has a named physics equation you can inspect
- Fast — single-pass DAG traversal, sub-second for typical systems
""")

        st.subheader("Universal Physics Library")
        st.markdown("""
The platform includes a library of **parameterized physics models** that work
across sectors. These are classical engineering equations — not trained models:

| Category | Examples |
|----------|----------|
| **Mass** | Plate mass (ρ×A×t), tube mass, volume-based |
| **Structural** | Section modulus, bending stress, fatigue life (S-N) |
| **Thermal** | Fourier conduction, heat capacity, resistive heating |
| **Fluid** | Drag force, power-to-overcome-drag, fuel consumption |
| **Performance** | Range estimation, power balance |

Each model takes material properties from a built-in database
(12 materials including steels, aluminum alloys, titanium, composites, copper, Inconel).
""")

        st.subheader("Graph Auto-Assembly")
        st.markdown("""
For parts without a pre-built template, the platform can **auto-assemble** a
cascade graph from:

- **Geometry** — extracted from CAD files (STL/STEP) or 2D drawings (via OCR)
- **Material** — selected from the material database
- **Sector** — determines safety factors, temperature limits, and constraint standards

The assembler classifies the part shape (plate, cylinder, tube, etc.), selects
applicable physics models, and wires them into a DAG with sector-appropriate
constraints — all without human intervention or AI generation.
""")

    # ── Data Sources ─────────────────────────────────────────────────
    with about_tab2:
        st.subheader("Data Sources & Pipeline")
        st.markdown("""
The platform supports multiple data input channels, from immediate use
to future integration:

**Available Now:**
- **CAD Geometry** (STL, STEP) — automatic dimension extraction
- **2D Engineering Drawings** (PNG, JPEG, PDF) — OCR-based dimension and tolerance extraction
- **Spec Documents** (CSV, PDF, text) — parameter override extraction
- **Sample Library** — 9 pre-built models across all sectors for quick testing

**In Pipeline:**
- **Engineering Change Records (ECRs)** — historical change-impact data used to
  calibrate coupling coefficients via least-squares fitting. The platform includes
  synthetic ECR generators for naval, aerospace, automotive EV, and robotics sectors.
- **Simulation Surrogates** — response surfaces fitted from FEA/CFD/modal analysis
  results. Linear and quadratic surrogate models can be dropped into the graph
  as physics model replacements where analytical equations are insufficient.
- **Operational Sensor Data** — real-time or historical sensor streams mapped to
  graph nodes. Drift detection identifies when in-service parameters deviate from
  design values, triggering cascade re-evaluation.
""")

        st.subheader("Bayesian Uncertainty")
        st.markdown("""
Optional Monte Carlo mode samples edge sensitivities from uncertainty
distributions, producing violation *probabilities* instead of point estimates.
Useful for trade studies where manufacturing tolerances or material variability
matter.
""")

        st.subheader("Cost Variance Prediction & Risk Tiering")
        st.markdown("""
An ML layer (Random Forest) predicts **how much actual cost will deviate from
the estimate**, based on change characteristics and cascade results. Features
include propagation score, violation count, cross-domain hops, change cause,
severity, and regulatory involvement.

**Risk Tiers:**

| Tier | Predicted Overrun | Action |
|------|------------------|--------|
| 1 | < 15% | Fast-track |
| 2 | 15–30% | Standard review |
| 3 | 30–45% | Senior review |
| 4 | > 45% | Deep analysis |

The feedback loop captures actual cost after implementation
to improve future predictions.
""")

    # ── Supported Sectors ────────────────────────────────────────────
    with about_tab3:
        st.subheader("Sectors & Templates")

        col_s1, col_s2 = st.columns(2)
        with col_s1:
            st.markdown("""
**Naval / Marine**
- Hull plate, deck stiffener, propulsion, sonar dome
- DNV GL structural rules
- Templates: hull thickness cascade

**Aerospace**
- Fuselage panel, wing spar, engine mount, landing gear
- FAR 25 / CS-25 certification
- Templates: electric aircraft cross-subsystem
""")
        with col_s2:
            st.markdown("""
**Automotive EV**
- Battery cell, cooling plate, BMS, motor, chassis
- UN R100, ISO 6469 safety
- Templates: EV battery pack cascade

**Robotics**
- 6-DOF arm links, joint actuators, end effectors, sensors
- ISO 10218, ISO 9283, IEC 60034
- Templates: robotic arm cascade (33 nodes, 37 edges)
""")

        st.info("Custom parts can use the **Auto-Assemble** path — no template required.")

    # ── Architecture ─────────────────────────────────────────────────
    with about_tab4:
        st.subheader("System Architecture")
        st.markdown("""
```
┌─────────────────────────────────────────────────────────┐
│                    Streamlit UI Layer                     │
│  Upload → Identify → Configure → Propagate → Results     │
├─────────────────────────────────────────────────────────┤
│                   Cascade Engine                         │
│  BFS propagation · Constraint checking · Cost roll-up    │
├──────────────┬──────────────┬───────────────────────────┤
│ Graph Builder│ Physics Lib  │  Material DB              │
│ Templates or │ 20+ models   │  12 materials             │
│ Auto-Assemble│ parameterized│  with full properties     │
├──────────────┴──────────────┴───────────────────────────┤
│                    Input Parsers                          │
│  CAD (STL/STEP) · 2D Drawing (OCR) · Spec (CSV/PDF)     │
├─────────────────────────────────────────────────────────┤
│                  Data Pipeline (future)                   │
│  ECR fitting · Simulation surrogates · Sensor streams    │
└─────────────────────────────────────────────────────────┘
```

**Key design decisions:**
- No LLMs in the propagation path — deterministic, auditable results
- Physics models are parameterized by material, not hard-coded per application
- Auto-assembly enables new parts without writing templates
- Sector-specific constraints loaded from standards databases
""")

"""
Seed failure records — real-world-ish historic failure data.

These would normally come from a company's failure tracking system,
warranty database, or test reports. For MVP, hardcoded examples
that demonstrate the matching and warning system.
"""

from cascade_predict.knowledge.failure_db import FailureRecord, FailureDatabase, Severity


def build_default_failure_db() -> FailureDatabase:
    """Build a failure database with seed records across industries."""
    db = FailureDatabase()
    db.add_many(_get_seed_records())
    return db


def _get_seed_records() -> list[FailureRecord]:
    return [
        # ============================================================
        # ELECTRICAL — FUSES
        # ============================================================
        FailureRecord(
            failure_id="FAIL-2W-001",
            title="Inline fuse thermal failure on two-wheeler EV",
            component_type="fuse",
            failure_mode="thermal_overload",
            root_cause=(
                "Inline fuse rated at 30A continuous was used in a two-wheeler "
                "powertrain drawing sustained 35-42A during hill climb and "
                "high-speed cruise. Fuse housing was enclosed with no airflow, "
                "causing cumulative thermal buildup. Fuse element degraded over "
                "~200 thermal cycles and failed open during a hill climb, "
                "cutting motor power mid-ride."
            ),
            description=(
                "Two-wheeler EV project Alpha used a standard automotive inline "
                "blade fuse for main powertrain protection. The fuse was rated "
                "for 30A continuous but the duty cycle involved repeated 35-42A "
                "draws of 3-5 min duration. Enclosed under-seat mounting provided "
                "no convective cooling. Fuse passed initial qualification at 25°C "
                "ambient but failed in field during summer operation (45°C ambient)."
            ),
            product_type="two_wheeler",
            subsystem="electrical",
            operating_conditions="sustained 35-42A, enclosed mounting, 45°C ambient",
            environment="tropical",
            affected_parameters=["battery_capacity", "mission_energy"],
            affected_properties=["current_rating", "fuse_type"],
            severity=Severity.CRITICAL,
            consequence="Field failure, safety recall of 340 units, 4-month delay",
            corrective_action=(
                "Replace inline fuse with semiconductor eFuse (electronic fuse) "
                "rated for 60A continuous with active thermal management. "
                "Alternatively, use a bolt-down fuse with exposed terminals "
                "for convective cooling and derate by 25% for enclosed mounting."
            ),
            source="Project Alpha post-mortem",
            date="2023-08",
            reference="ALPHA-FR-2023-008",
            trigger_conditions={
                "current_rating": {"max": 30, "unit": "A"},
            },
            tags=["fuse", "inline_fuse", "thermal", "overcurrent", "two_wheeler",
                  "powertrain", "enclosed_mounting", "derating"],
        ),

        FailureRecord(
            failure_id="FAIL-2W-002",
            title="Fuse holder contact resistance increase from vibration",
            component_type="fuse",
            failure_mode="vibration_fatigue",
            root_cause=(
                "Blade fuse in spring-clip holder experienced micro-fretting "
                "from two-wheeler frame vibration. Contact resistance increased "
                "from 0.5 mOhm to 12 mOhm over 8000 km, causing localized "
                "heating at the fuse terminals."
            ),
            description=(
                "Standard ATO fuse holder mounted on two-wheeler frame. "
                "Vibration profile (50-200 Hz, 2-5g) caused progressive "
                "contact degradation. Not caught in standard vibration test "
                "which only ran 24h vs real-world cumulative exposure."
            ),
            product_type="two_wheeler",
            subsystem="electrical",
            operating_conditions="frame-mounted, 50-200 Hz vibration, 2-5g",
            environment="urban",
            severity=Severity.HIGH,
            consequence="Hot terminal, melted housing in 3 field units",
            corrective_action=(
                "Use bolt-down fuse connection or soldered fuse link. "
                "If blade fuse required, add secondary retention clip and "
                "validate with extended vibration test (500h minimum)."
            ),
            source="Field warranty analysis",
            date="2023-11",
            reference="ALPHA-FR-2023-014",
            tags=["fuse", "vibration", "contact_resistance", "two_wheeler",
                  "connector", "fretting"],
        ),

        # ============================================================
        # BATTERY — CELL / PACK
        # ============================================================
        FailureRecord(
            failure_id="FAIL-BAT-001",
            title="NMC cell capacity fade accelerated by high-temp cycling",
            component_type="battery_cell",
            failure_mode="accelerated_degradation",
            root_cause=(
                "NMC811 cells cycled between 10-90% SOC at cell temperatures "
                "consistently above 40°C showed 15% capacity fade at 500 cycles "
                "vs expected 8% at 25°C. Thermal management system was undersized "
                "— chiller capacity 30% below required for sustained fast charging "
                "in hot climate."
            ),
            description=(
                "EV sedan battery pack in tropical market. Customers used "
                "DC fast charging 3-4x per week. Combined with 40°C+ ambient "
                "and solar soaking, cell temperatures during charging reached "
                "48-52°C regularly. Pack warranty claims started at 18 months."
            ),
            product_type="ev_sedan",
            subsystem="pack_thermal",
            operating_conditions="fast charging 3-4x/week, 40°C+ ambient",
            environment="tropical",
            affected_parameters=["max_cell_temp", "chiller_capacity", "battery_capacity"],
            affected_properties=["cooling_capacity", "thermal_conductivity"],
            severity=Severity.HIGH,
            consequence="Warranty cost $4.2M, customer satisfaction impact",
            corrective_action=(
                "Upsize chiller by 40% for tropical markets. Add pre-conditioning "
                "routine that cools pack to 30°C before fast charge. Implement "
                "BMS thermal curtailment at 42°C cell temp (reduce charge rate)."
            ),
            source="Warranty data analysis",
            date="2024-03",
            reference="EV-BATT-2024-003",
            trigger_conditions={
                "max_cell_temp": {"max": 40, "unit": "deg_C"},
                "chiller_capacity": {"min": 6.0, "unit": "kW"},
            },
            tags=["battery", "thermal", "degradation", "fast_charging",
                  "hot_climate", "chiller", "NMC811"],
        ),

        FailureRecord(
            failure_id="FAIL-BAT-002",
            title="Cell tab weld failure from thermal expansion cycling",
            component_type="battery_cell",
            failure_mode="weld_fatigue",
            root_cause=(
                "Ultrasonic welds between cell tabs and busbar experienced "
                "thermal fatigue from repeated expansion/contraction cycles. "
                "Tab temperature swing of 25°C per charge/discharge cycle "
                "caused progressive crack growth in weld nugget. Cells with "
                "higher internal resistance generated more heat, accelerating "
                "failure at those specific joints."
            ),
            description=(
                "Module-level failure in EV pack. 3 of 96 cells in a module "
                "developed high-resistance tab connections after 800 cycles. "
                "Higher resistance → more heat → faster degradation feedback loop."
            ),
            product_type="ev_sedan",
            subsystem="module",
            operating_conditions="daily cycling, 2C peak discharge",
            affected_parameters=["cell_resistance", "cell_heat_gen", "total_heat_gen"],
            affected_properties=["internal_resistance"],
            severity=Severity.HIGH,
            consequence="Module replacement under warranty, 12 packs affected",
            corrective_action=(
                "Increase weld nugget size by 20%. Add thermal pad between "
                "tab and busbar to reduce peak temperature at joint. Monitor "
                "cell resistance trend in BMS — flag cells with >15% resistance "
                "increase for predictive maintenance."
            ),
            source="Root cause analysis",
            date="2024-01",
            reference="EV-MOD-2024-001",
            trigger_conditions={
                "internal_resistance": {"max": 30, "unit": "mOhm"},
            },
            tags=["battery", "welding", "tab", "thermal_cycling",
                  "resistance_growth", "module"],
        ),

        # ============================================================
        # THERMAL — INSULATION / HVAC
        # ============================================================
        FailureRecord(
            failure_id="FAIL-TH-001",
            title="Cabin insulation R-value degradation from moisture ingress",
            component_type="insulation",
            failure_mode="moisture_degradation",
            root_cause=(
                "Cabin insulation blankets absorbed moisture through damaged "
                "vapor barrier, reducing effective R-value from 2.5 to 1.4 "
                "m²·K/W. This increased cabin heat load by ~40%, causing HVAC "
                "to run at max capacity continuously, draining battery 12% "
                "faster on hot days."
            ),
            description=(
                "Electric aircraft operating in humid coastal environment. "
                "Vapor barrier punctured during maintenance. Moisture wicking "
                "into fiberglass blankets went undetected for 6 weeks. Pilots "
                "reported 'HVAC always running' and reduced range."
            ),
            product_type="electric_aircraft",
            subsystem="thermal",
            operating_conditions="humid coastal, frequent rain, tropical",
            environment="tropical",
            affected_parameters=[
                "fuselage_insulation_rvalue", "cabin_heat_load",
                "hvac_cooling_capacity", "mission_energy",
            ],
            affected_properties=["insulation_rvalue"],
            severity=Severity.MEDIUM,
            consequence="Range reduction, increased energy consumption, maintenance cost",
            corrective_action=(
                "Add moisture detection sensors in insulation bays. "
                "Specify closed-cell foam insulation (moisture-resistant) "
                "for humid environments. Include insulation R-value check "
                "in 500-hour maintenance interval."
            ),
            source="Maintenance finding report",
            date="2024-06",
            reference="AC-MX-2024-012",
            tags=["insulation", "moisture", "thermal", "hvac", "range_impact",
                  "maintenance"],
        ),

        FailureRecord(
            failure_id="FAIL-TH-002",
            title="Windshield delamination from thermal stress",
            component_type="windshield",
            failure_mode="delamination",
            root_cause=(
                "High thermal conductivity windshield material (k=1.8 W/mK) "
                "created large temperature gradient across laminate layers "
                "when cabin was cooled while exterior was hot. Differential "
                "thermal expansion caused edge delamination after 400 thermal "
                "cycles."
            ),
            description=(
                "Polycarbonate/glass laminate windshield on electric aircraft. "
                "Material selected for optical clarity had higher thermal "
                "conductivity than previous design. The resulting higher heat "
                "transfer rate also caused larger through-thickness gradients "
                "during rapid HVAC cooldown."
            ),
            product_type="electric_aircraft",
            subsystem="thermal",
            operating_conditions="hot exterior + cold cabin, rapid cooldown",
            affected_parameters=["windshield_k", "cabin_heat_load"],
            affected_properties=["thermal_conductivity"],
            severity=Severity.MEDIUM,
            consequence="Windshield replacement, 2 aircraft grounded",
            corrective_action=(
                "Limit windshield thermal conductivity to k < 1.5 W/(m·K) "
                "or add edge stress relief layer. Consider low-e coating to "
                "reduce solar transmittance rather than changing substrate material."
            ),
            source="Engineering investigation",
            date="2023-12",
            reference="AC-STR-2023-019",
            trigger_conditions={
                "thermal_conductivity": {"max": 1.5, "unit": "W/(m·K)"},
            },
            tags=["windshield", "thermal_stress", "delamination",
                  "material_selection", "thermal_conductivity"],
        ),

        # ============================================================
        # AERODYNAMIC — WINDSHIELD CURVATURE
        # ============================================================
        FailureRecord(
            failure_id="FAIL-AERO-001",
            title="Windshield curvature change caused drag exceedance and range shortfall",
            component_type="windshield",
            failure_mode="drag_exceedance",
            root_cause=(
                "Windshield redesign increased panel curvature from 0.12 to 0.38 1/m "
                "to improve bird-strike resistance. The higher-curvature panels disrupted "
                "the forward fuselage boundary layer, increasing parasite drag by 4.2%. "
                "Combined with the 3.6 kg mass increase from thicker edge sections, "
                "mission range dropped 18 nm below the minimum certification requirement. "
                "Drag increase was not predicted by panel-level CFD which used isolated "
                "geometry without fuselage integration effects."
            ),
            description=(
                "Electric aircraft development program. Windshield supplier proposed "
                "higher-curvature design for improved structural margin. Panel-level "
                "analysis showed benefits but full-aircraft wind tunnel test revealed "
                "unexpected drag penalty from flow separation at windshield-fuselage "
                "junction. Required windshield redesign and 3-month schedule slip."
            ),
            product_type="electric_aircraft",
            subsystem="aerodynamic",
            operating_conditions="cruise, M=0.28, ISA conditions",
            environment="standard",
            affected_parameters=["windshield_curvature", "cruise_ld", "windshield_mass",
                                 "range_nm", "mission_energy"],
            affected_properties=["curvature"],
            severity=Severity.HIGH,
            consequence="Range shortfall, windshield redesign, 3-month delay",
            corrective_action=(
                "Limit windshield curvature to < 0.30 1/m unless validated with "
                "full-aircraft CFD or wind tunnel. Include fuselage junction fillet "
                "in aero assessment. Run integrated drag audit at each design gate."
            ),
            source="Wind tunnel test correlation",
            date="2024-04",
            reference="AC-AERO-2024-005",
            trigger_conditions={
                "curvature": {"max": 0.30, "unit": "1/m"},
            },
            tags=["windshield", "curvature", "drag", "aerodynamic", "range",
                  "boundary_layer", "flow_separation"],
        ),

        # ============================================================
        # STRUCTURAL — WEIGHT GROWTH
        # ============================================================
        FailureRecord(
            failure_id="FAIL-STR-001",
            title="MTOW exceedance from cascading weight growth",
            component_type="battery_pack",
            failure_mode="weight_growth",
            root_cause=(
                "Battery pack was 8% heavier than spec due to additional "
                "thermal interface material added during development. This "
                "cascaded: heavier pack → higher MTOW → stronger wing needed "
                "→ heavier wing → even higher MTOW. Spiral converged at +180 kg "
                "over original MTOW target, requiring Type Certificate amendment."
            ),
            description=(
                "Electric aircraft development program. Initial battery mass "
                "estimate was 3500 kg, actual came in at 3780 kg. Cascade "
                "through structural sizing added another 180 kg. Final MTOW "
                "exceeded type certificate by 4.2%."
            ),
            product_type="electric_aircraft",
            subsystem="mass",
            affected_parameters=["battery_mass", "oew", "mtow",
                                 "wing_root_bending", "wing_mass"],
            severity=Severity.CRITICAL,
            consequence="Type Certificate amendment needed, 8-month schedule delay",
            corrective_action=(
                "Build 5% mass contingency into battery pack estimates. "
                "Run cascade analysis at every design review gate to catch "
                "weight spiral early. Set hard mass budgets per subsystem "
                "with formal change control."
            ),
            source="Program lessons learned",
            date="2024-02",
            reference="AC-PM-2024-002",
            tags=["weight_growth", "mass_spiral", "mtow", "certification",
                  "battery", "structural"],
        ),

        # ============================================================
        # CONNECTOR / WIRING
        # ============================================================
        FailureRecord(
            failure_id="FAIL-EL-001",
            title="High-voltage connector arcing from thermal expansion gap",
            component_type="connector",
            failure_mode="arcing",
            root_cause=(
                "HV connector between battery modules used aluminum contacts. "
                "Thermal cycling caused differential expansion with copper busbar, "
                "creating micro-gap. At 400V, gap arced intermittently under "
                "high current draw, eventually welding contacts and causing "
                "permanent high-resistance joint."
            ),
            description=(
                "EV battery pack HV distribution. Connector passed qualification "
                "at room temperature but failed in field after thermal cycling "
                "(-10°C to +55°C pack temperatures). 6 vehicles affected."
            ),
            product_type="ev_sedan",
            subsystem="electrical",
            operating_conditions="thermal cycling -10 to 55°C, 400V, 200A peak",
            affected_parameters=["pack_voltage", "cell_resistance"],
            severity=Severity.CRITICAL,
            consequence="Safety recall, pack replacement in 6 vehicles",
            corrective_action=(
                "Use same-metal contacts (copper-to-copper) for HV connections. "
                "Add spring-loaded contact retention. Validate with 1000 thermal "
                "cycles at full voltage and current before qualification sign-off."
            ),
            source="Safety investigation",
            date="2024-05",
            reference="EV-SAF-2024-007",
            tags=["connector", "arcing", "high_voltage", "thermal_cycling",
                  "battery", "safety"],
        ),
    ]

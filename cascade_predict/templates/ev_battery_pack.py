"""
Template: EV Battery Pack System (Automotive)

Industry: Automotive
Reference scenario: Cell chemistry upgrade cascading through pack design,
BMS, thermal management, vehicle integration, and homologation.

Subsystems: cell, module, pack_thermal, bms, vehicle_integration, regulatory
"""

from cascade_predict.graph.dependency_graph import (
    DependencyGraph,
    SubsystemNode,
    CouplingEdge,
)
from cascade_predict.subsystems.component import Component, CertConstraint
from cascade_predict.templates import TemplateInfo, register_template
from cascade_predict.physics_models import LinearModel
from cascade_predict.physics_models.thermal import (
    JouleHeating,
    HeatGenScaling,
    ThermalResistanceToTemp,
    ConvectiveCooling,
    HVACSizing,
    COPPowerDraw,
    ThermalMargin,
)
from cascade_predict.physics_models.electrical import (
    CellEnergy,
    SpecificEnergyFromMass,
    PowerBalance,
    ChargeTime,
)
from cascade_predict.physics_models.mass import (
    DirectMassSum,
    CountBasedMass,
    PackLevelSpecificEnergy,
)
from cascade_predict.physics_models.fluid import (
    ModuleCountScaling,
    WeightToConsumption,
    ConsumptionToRange,
)


def build_ev_battery_pack():
    """
    Build an EV battery pack system model.

    Models a typical 75 kWh automotive battery pack with:
    - Cell-level properties (chemistry, capacity, impedance, thermal)
    - Module construction (cell count, interconnects, cooling)
    - Pack thermal management (liquid cooling loop)
    - BMS constraints (voltage windows, current limits)
    - Vehicle integration (mass, volume, range)
    - Regulatory (UN ECE R100, IEC 62619)

    Returns:
        graph, components, constraints — same interface as every template
    """
    g = DependencyGraph()
    components = {}
    constraints = []

    # ================================================================
    # COMPONENTS
    # ================================================================

    # --- Battery Cell ---
    cell = Component(
        "battery_cell", "Li-ion NMC Cell (21700)", "cell",
        description="Cylindrical 21700 NMC811 cell",
    )
    cell.add_property("nominal_capacity", 5.0, "Ah",
                       graph_node_id="cell_capacity",
                       description="Nominal cell capacity",
                       source="cell datasheet")
    cell.add_property("nominal_voltage", 3.6, "V",
                       graph_node_id="cell_voltage",
                       description="Nominal voltage",
                       source="cell datasheet")
    cell.add_property("internal_resistance", 25.0, "mOhm",
                       graph_node_id="cell_resistance",
                       description="DC internal resistance at 25C, 50% SOC",
                       source="cell datasheet")
    cell.add_property("mass", 0.070, "kg",
                       graph_node_id="cell_mass",
                       description="Cell mass",
                       source="cell datasheet")
    cell.add_property("specific_energy", 257.0, "Wh/kg",
                       graph_node_id="cell_specific_energy",
                       description="Cell-level gravimetric energy density",
                       source="cell datasheet")
    cell.add_property("max_charge_rate", 2.0, "C",
                       graph_node_id="cell_max_charge_rate",
                       description="Max continuous charge rate",
                       source="cell datasheet")
    cell.add_property("thermal_runaway_onset", 155.0, "deg_C",
                       graph_node_id="cell_tr_onset",
                       description="Thermal runaway onset temperature",
                       source="ARC test data")
    components["battery_cell"] = cell

    # --- Module ---
    module = Component(
        "battery_module", "Battery Module", "module",
        description="12s8p module with liquid cold plate",
    )
    module.add_property("cells_series", 12.0, "count",
                         graph_node_id="module_series",
                         description="Cells in series per module",
                         source="module design")
    module.add_property("cells_parallel", 8.0, "count",
                         graph_node_id="module_parallel",
                         description="Cells in parallel per module",
                         source="module design")
    module.add_property("mass_overhead", 1.8, "kg",
                         graph_node_id="module_overhead_mass",
                         description="Non-cell mass (busbars, housing, cold plate)",
                         source="module BOM")
    components["battery_module"] = module

    # --- Cooling System ---
    cooling = Component(
        "cooling_system", "Pack Thermal Management", "pack_thermal",
        description="Liquid glycol cooling loop with chiller",
    )
    cooling.add_property("coolant_flow_rate", 10.0, "L/min",
                          graph_node_id="coolant_flow",
                          description="Coolant volumetric flow rate",
                          source="thermal design")
    cooling.add_property("chiller_capacity", 5.0, "kW",
                          graph_node_id="chiller_capacity",
                          description="Chiller thermal rejection capacity",
                          source="vendor spec")
    cooling.add_property("interface_resistance", 0.5, "K/W",
                          graph_node_id="thermal_interface_resistance",
                          description="Cell-to-coolant thermal resistance",
                          source="thermal sim")
    components["cooling_system"] = cooling

    # --- BMS ---
    bms = Component(
        "bms", "Battery Management System", "bms",
        description="Master + slave BMS with active balancing",
    )
    bms.add_property("max_pack_voltage", 403.0, "V",
                      graph_node_id="bms_max_voltage",
                      description="Maximum allowed pack voltage",
                      source="BMS config")
    bms.add_property("max_discharge_current", 300.0, "A",
                      graph_node_id="bms_max_current",
                      description="Max continuous discharge current",
                      source="BMS config")
    components["bms"] = bms

    # ================================================================
    # GRAPH NODES
    # ================================================================

    # Cell-level
    g.add_node(SubsystemNode("cell_capacity", "cell", 5.0, "Ah", "Cell nominal capacity"))
    g.add_node(SubsystemNode("cell_voltage", "cell", 3.6, "V", "Cell nominal voltage"))
    g.add_node(SubsystemNode("cell_resistance", "cell", 25.0, "mOhm", "Cell internal resistance"))
    g.add_node(SubsystemNode("cell_mass", "cell", 0.070, "kg", "Cell mass"))
    g.add_node(SubsystemNode("cell_specific_energy", "cell", 257.0, "Wh/kg", "Cell specific energy"))
    g.add_node(SubsystemNode("cell_energy", "cell", 18.0, "Wh", "Cell energy (V × Ah)"))
    g.add_node(SubsystemNode("cell_max_charge_rate", "cell", 2.0, "C", "Max charge C-rate"))
    g.add_node(SubsystemNode("cell_tr_onset", "cell", 155.0, "deg_C", "Thermal runaway onset"))
    g.add_node(SubsystemNode("cell_heat_gen", "cell", 2.5, "W",
                              "Cell heat generation at 1C discharge"))

    # Module-level
    g.add_node(SubsystemNode("module_series", "module", 12.0, "count", "Cells in series"))
    g.add_node(SubsystemNode("module_parallel", "module", 8.0, "count", "Cells in parallel"))
    g.add_node(SubsystemNode("module_voltage", "module", 43.2, "V", "Module voltage"))
    g.add_node(SubsystemNode("module_capacity", "module", 40.0, "Ah", "Module capacity"))
    g.add_node(SubsystemNode("module_energy", "module", 1.728, "kWh", "Module energy"))
    g.add_node(SubsystemNode("module_cell_count", "module", 96.0, "count", "Cells per module"))
    g.add_node(SubsystemNode("module_mass", "module", 8.52, "kg", "Module total mass"))
    g.add_node(SubsystemNode("module_overhead_mass", "module", 1.8, "kg", "Module non-cell mass"))

    # Pack-level
    n_modules = 44  # ~75 kWh pack
    g.add_node(SubsystemNode("pack_modules", "module", float(n_modules), "count", "Modules in pack"))
    g.add_node(SubsystemNode("pack_energy", "vehicle_integration", 76.0, "kWh",
                              "Pack total energy"))
    g.add_node(SubsystemNode("pack_voltage", "module", 403.0, "V", "Pack nominal voltage"))
    g.add_node(SubsystemNode("pack_mass", "vehicle_integration", 450.0, "kg", "Pack total mass"))
    g.add_node(SubsystemNode("pack_specific_energy", "vehicle_integration", 169.0, "Wh/kg",
                              "Pack-level specific energy"))

    # Thermal
    g.add_node(SubsystemNode("total_heat_gen", "pack_thermal", 110.0, "W",
                              "Total pack heat generation at 1C"))
    g.add_node(SubsystemNode("coolant_flow", "pack_thermal", 10.0, "L/min",
                              "Coolant flow rate"))
    g.add_node(SubsystemNode("chiller_capacity", "pack_thermal", 5.0, "kW",
                              "Chiller capacity"))
    g.add_node(SubsystemNode("thermal_interface_resistance", "pack_thermal", 0.5, "K/W",
                              "Cell-to-coolant thermal resistance"))
    g.add_node(SubsystemNode("max_cell_temp", "pack_thermal", 35.0, "deg_C",
                              "Max cell temperature at sustained 1C",
                              bounds=(0.0, 60.0),
                              regulatory_limit=60.0,
                              regulatory_ref="IEC 62619"))
    g.add_node(SubsystemNode("thermal_delta_t", "pack_thermal", 5.0, "deg_C",
                              "Max cell-to-cell temperature spread",
                              bounds=(0.0, 10.0)))
    g.add_node(SubsystemNode("chiller_power", "pack_thermal", 1.5, "kW",
                              "Chiller electrical power consumption"))

    # BMS
    g.add_node(SubsystemNode("bms_max_voltage", "bms", 403.0, "V", "BMS max voltage limit"))
    g.add_node(SubsystemNode("bms_max_current", "bms", 300.0, "A", "BMS max current limit"))
    g.add_node(SubsystemNode("max_charge_power", "bms", 120.0, "kW",
                              "Max DC charge power"))
    g.add_node(SubsystemNode("charge_time_10_80", "bms", 35.0, "min",
                              "10-80% charge time",
                              bounds=(10.0, 120.0)))

    # Vehicle integration
    g.add_node(SubsystemNode("vehicle_curb_weight", "vehicle_integration", 1850.0, "kg",
                              "Vehicle curb weight"))
    g.add_node(SubsystemNode("vehicle_range", "vehicle_integration", 400.0, "km",
                              "WLTP range",
                              bounds=(200.0, 800.0)))
    g.add_node(SubsystemNode("vehicle_energy_consumption", "vehicle_integration", 19.0, "kWh/100km",
                              "WLTP energy consumption"))

    # Regulatory
    g.add_node(SubsystemNode("nail_penetration_margin", "regulatory", 0.30, "fraction",
                              "Nail penetration test margin",
                              bounds=(0.0, 1.0),
                              regulatory_ref="UN ECE R100"))

    # ================================================================
    # COUPLING EDGES — physics models provide the equations
    # ================================================================

    n_cells_total = 96 * n_modules  # 12s × 8p × 44 modules = 4224 cells

    # -- Electrical: Cell energy = V × Ah --
    g.add_edge(CouplingEdge("cell_voltage", "cell_energy",
        model=CellEnergy(other_quantity=5.0)))     # Ah
    g.add_edge(CouplingEdge("cell_capacity", "cell_energy",
        model=CellEnergy(other_quantity=3.6)))     # V

    # -- Thermal: Cell resistance → heat generation (P = I²R) --
    g.add_edge(CouplingEdge("cell_resistance", "cell_heat_gen",
        model=JouleHeating(current=5.0)))          # 1C = 5A

    # -- Electrical: Cell mass → specific energy --
    g.add_edge(CouplingEdge("cell_mass", "cell_specific_energy",
        model=SpecificEnergyFromMass(energy_wh=18.0, baseline_mass=0.070)))

    # -- Module: cell → module scaling --
    g.add_edge(CouplingEdge("cell_voltage", "module_voltage",
        model=LinearModel(12.0, "Module V = n_series × V_cell", "V_mod = 12 × V_cell")))
    g.add_edge(CouplingEdge("cell_capacity", "module_capacity",
        model=LinearModel(8.0, "Module Ah = n_parallel × Ah_cell", "Q_mod = 8 × Q_cell")))

    # -- Module energy --
    g.add_edge(CouplingEdge("module_voltage", "module_energy",
        model=LinearModel(0.040, "Module energy = V × Ah / 1000")))
    g.add_edge(CouplingEdge("module_capacity", "module_energy",
        model=LinearModel(0.0432, "Module energy = V × Ah / 1000")))

    # -- Module cell count --
    g.add_edge(CouplingEdge("module_series", "module_cell_count",
        model=LinearModel(8.0, "Total cells = series × parallel")))
    g.add_edge(CouplingEdge("module_parallel", "module_cell_count",
        model=LinearModel(12.0, "Total cells = series × parallel")))

    # -- Mass: Module mass --
    g.add_edge(CouplingEdge("cell_mass", "module_mass",
        model=CountBasedMass(count=96)))
    g.add_edge(CouplingEdge("module_overhead_mass", "module_mass",
        model=DirectMassSum()))

    # -- Mass: Pack scaling --
    g.add_edge(CouplingEdge("module_energy", "pack_energy",
        model=ModuleCountScaling(module_count=n_modules)))
    g.add_edge(CouplingEdge("module_mass", "pack_mass",
        model=ModuleCountScaling(module_count=n_modules)))

    # -- Mass: Pack specific energy --
    g.add_edge(CouplingEdge("pack_energy", "pack_specific_energy",
        model=PackLevelSpecificEnergy(baseline_energy_kwh=76.0, baseline_mass_kg=450.0, source_is_energy=True)))
    g.add_edge(CouplingEdge("pack_mass", "pack_specific_energy",
        model=PackLevelSpecificEnergy(baseline_energy_kwh=76.0, baseline_mass_kg=450.0, source_is_energy=False)))

    # -- Thermal: Heat generation scales with cell count --
    g.add_edge(CouplingEdge("cell_heat_gen", "total_heat_gen",
        model=HeatGenScaling(n_elements=n_cells_total)))

    # -- Thermal: heat gen → cell temperature --
    g.add_edge(CouplingEdge("total_heat_gen", "max_cell_temp",
        model=LinearModel(0.012, "Heat to cell temp via thermal resistance network", "T_cell = T_amb + Q × R_eff")))
    g.add_edge(CouplingEdge("thermal_interface_resistance", "max_cell_temp",
        model=ThermalResistanceToTemp(heat_load=5.0)))
    g.add_edge(CouplingEdge("coolant_flow", "max_cell_temp",
        model=ConvectiveCooling(sensitivity=-0.8)))

    # -- Thermal: cell temperature spread --
    g.add_edge(CouplingEdge("total_heat_gen", "thermal_delta_t",
        model=LinearModel(0.003, "Heat generation increases cell-to-cell spread")))
    g.add_edge(CouplingEdge("coolant_flow", "thermal_delta_t",
        model=ConvectiveCooling(sensitivity=-0.3)))

    # -- Thermal: Chiller sizing --
    g.add_edge(CouplingEdge("total_heat_gen", "chiller_capacity",
        model=HVACSizing(margin_factor=0.0015)))  # kW of chiller per W of pack heat
    g.add_edge(CouplingEdge("chiller_capacity", "chiller_power",
        model=COPPowerDraw(cop=3.3)))

    # -- Thermal: Safety margin --
    g.add_edge(CouplingEdge("max_cell_temp", "nail_penetration_margin",
        model=ThermalMargin(sensitivity=-0.0065)))
    g.add_edge(CouplingEdge("cell_tr_onset", "nail_penetration_margin",
        model=ThermalMargin(sensitivity=0.0065)))

    # -- Electrical: BMS pack voltage --
    g.add_edge(CouplingEdge("module_voltage", "pack_voltage",
        model=ModuleCountScaling(module_count=n_modules, divisor=12.0)))

    # -- Electrical: Charge power --
    g.add_edge(CouplingEdge("pack_voltage", "max_charge_power",
        model=PowerBalance(other_quantity=300.0, source_is_voltage=True)))
    g.add_edge(CouplingEdge("bms_max_current", "max_charge_power",
        model=PowerBalance(other_quantity=403.0, source_is_voltage=False)))

    # -- Electrical: Charge time --
    g.add_edge(CouplingEdge("pack_energy", "charge_time_10_80",
        model=ChargeTime(sensitivity=0.60)))
    g.add_edge(CouplingEdge("max_charge_power", "charge_time_10_80",
        model=ChargeTime(sensitivity=-0.18)))

    # -- Mass/Vehicle: Pack → vehicle weight --
    g.add_edge(CouplingEdge("pack_mass", "vehicle_curb_weight",
        model=DirectMassSum()))
    g.add_edge(CouplingEdge("chiller_power", "vehicle_energy_consumption",
        model=LinearModel(0.5, "Chiller parasitic load")))
    g.add_edge(CouplingEdge("vehicle_curb_weight", "vehicle_energy_consumption",
        model=WeightToConsumption(kwh_per_100km_per_kg=0.005)))

    # -- Range --
    g.add_edge(CouplingEdge("pack_energy", "vehicle_range",
        model=LinearModel(5.26, "More energy → more range", "R = E / consumption × 100")))
    g.add_edge(CouplingEdge("vehicle_energy_consumption", "vehicle_range",
        model=ConsumptionToRange(sensitivity=-22.2)))

    # ================================================================
    # CERT CONSTRAINTS
    # ================================================================
    constraints.append(CertConstraint(
        "iec_62619_temp", "IEC", "62619", "Cell temperature limit",
        "Cell temperature must not exceed 60°C during normal operation.",
        "max_cell_temp", 60.0, "max", "deg_C",
    ))
    constraints.append(CertConstraint(
        "un_ece_r100_safety", "UN ECE", "R100", "Electrical safety",
        "Battery must pass nail penetration with positive margin.",
        "nail_penetration_margin", 0.0, "min", "fraction",
    ))
    constraints.append(CertConstraint(
        "oem_range_min", "OEM", "Requirement", "Minimum range",
        "Vehicle must achieve minimum 350 km WLTP range.",
        "vehicle_range", 350.0, "min", "km",
    ))

    return g, components, constraints


TEMPLATE_INFO = TemplateInfo(
    template_id="ev_battery_pack",
    name="EV Battery Pack",
    description=(
        "75 kWh automotive battery pack with cell, module, thermal management, "
        "BMS, and vehicle integration coupling. Reference scenario: cell chemistry "
        "upgrade cascading through thermal, mass, charging, and range."
    ),
    industry="Automotive",
    build_fn=build_ev_battery_pack,
    icon="battery",
    presets=[
        {
            "name": "Cell upgrade: 5.0 -> 5.8 Ah (higher energy cell)",
            "component_id": "battery_cell",
            "property_name": "nominal_capacity",
            "new_value": 5.8,
        },
        {
            "name": "Cell resistance increase: 25 -> 35 mOhm (aging)",
            "component_id": "battery_cell",
            "property_name": "internal_resistance",
            "new_value": 35.0,
        },
        {
            "name": "Cooling degradation: R_th 0.5 -> 0.9 K/W",
            "component_id": "cooling_system",
            "property_name": "interface_resistance",
            "new_value": 0.9,
        },
        {
            "name": "Cheaper cell: TR onset 155 -> 130 deg_C",
            "component_id": "battery_cell",
            "property_name": "thermal_runaway_onset",
            "new_value": 130.0,
        },
    ],
)

register_template(TEMPLATE_INFO)

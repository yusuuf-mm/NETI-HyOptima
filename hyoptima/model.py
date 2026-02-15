"""
HyOptima Core Optimization Model

This module defines the Pyomo optimization model for hybrid energy systems.
It implements the mathematical formulation for minimizing total system cost
while meeting demand and policy constraints aligned with Nigeria's Energy
Transition Plan.

Mathematical Formulation:
========================
    minimize: investment_cost + fuel_cost + grid_cost + penalty_cost
    
    subject to:
        - Power balance at each time period
        - Generator capacity limits
        - Renewable availability limits
        - Battery state-of-charge dynamics
        - Emission constraints (optional)
        - Reliability requirements
        - Policy constraints (optional)

Decision Variables:
==================
    Sizing:
        - solar_capacity: Solar PV capacity (kW)
        - gas_capacity: Gas generator capacity (kW)
        - battery_capacity: Battery storage capacity (kWh)
    
    Operational (per time period t):
        - solar_gen[t]: Solar generation (kW)
        - gas_gen[t]: Gas generation (kW)
        - grid_import[t]: Grid import (kW)
        - battery_charge[t]: Battery charging (kW)
        - battery_discharge[t]: Battery discharging (kW)
        - soc[t]: State of charge (kWh)
        - unserved[t]: Unserved energy (kW)
        - gas_on[t]: Gas generator on/off status (binary)

Author: NETI-HyOptima Team
"""

from pyomo.environ import (
    ConcreteModel, Set, Param, Var, Objective, Constraint,
    NonNegativeReals, Binary, minimize, value, Suffix, inequality
)
from pyomo.opt import SolverFactory, TerminationCondition
from typing import Dict, Optional, Tuple, Any, List
import numpy as np
import logging

from .parameters import EconomicParameters, TechnicalParameters, PolicyParameters
from .profiles import LoadProfile, SolarProfile

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class HyOptimaModel:
    """
    HyOptima Hybrid Energy Optimization Model.
    
    This class builds and solves the optimization problem for sizing
    and operating a hybrid energy system with solar, gas, and storage.
    
    The model is designed to support Nigeria's Energy Transition Plan
    by optimizing for cost, emissions, and reliability simultaneously.
    
    Attributes:
        load_profile: Electrical demand profile
        solar_profile: Solar availability profile
        economic: Economic parameters (costs)
        technical: Technical parameters (efficiencies, limits)
        policy: Policy constraints (emissions, targets)
        name: Model identifier
        model: Pyomo ConcreteModel instance
        results: Optimization results dictionary
    
    Example:
        >>> load = LoadProfile.generate_synthetic(peak_demand=300)
        >>> solar = SolarProfile.from_capacity_factor(capacity_factor=0.20)
        >>> model = HyOptimaModel(load, solar)
        >>> model.build_model()
        >>> # Solve using HyOptimaSolver
    """
    
    def __init__(
        self,
        load_profile: LoadProfile,
        solar_profile: SolarProfile,
        economic_params: Optional[EconomicParameters] = None,
        technical_params: Optional[TechnicalParameters] = None,
        policy_params: Optional[PolicyParameters] = None,
        name: str = "HyOptima_v0"
    ):
        """
        Initialize the HyOptima model.
        
        Args:
            load_profile: Electrical demand profile
            solar_profile: Solar availability profile
            economic_params: Economic parameters (costs). Defaults to Nigerian market estimates.
            technical_params: Technical parameters (efficiencies, limits). Defaults to typical values.
            policy_params: Policy constraints (emissions, targets). Defaults to ETP-aligned values.
            name: Model name identifier
        """
        self.load_profile = load_profile
        self.solar_profile = solar_profile
        self.economic = economic_params or EconomicParameters()
        self.technical = technical_params or TechnicalParameters()
        self.policy = policy_params or PolicyParameters()
        self.name = name
        
        self.model: Optional[ConcreteModel] = None
        self.results: Optional[Dict] = None
        
        logger.info(f"Initialized HyOptimaModel: {name}")
        logger.info(f"Load profile: {load_profile.demand_duration} hours, peak {load_profile.peak_demand:.1f} kW")
        logger.info(f"Solar profile: CF = {solar_profile.capacity_factor:.1%}")
    
    def build_model(self) -> ConcreteModel:
        """
        Build the Pyomo optimization model.
        
        This method constructs the complete optimization model including:
        - Sets (time periods, energy sources)
        - Parameters (demand, costs, technical specs)
        - Decision variables (capacity and operational)
        - Objective function (total cost minimization)
        - Constraints (physical and policy)
        
        Returns:
            ConcreteModel: The built Pyomo model ready for solving
        """
        logger.info("Building optimization model...")
        
        m = ConcreteModel(name=self.name)
        
        # ===================
        # SETS
        # ===================
        
        # Time periods
        T = len(self.load_profile.demand)
        m.T = Set(initialize=range(T), doc="Time periods (hours)")
        
        # Energy sources (for future extensibility)
        m.SOURCES = Set(initialize=['solar', 'gas', 'grid'], doc="Energy sources")
        
        logger.info(f"Model has {T} time periods")
        
        # ===================
        # PARAMETERS
        # ===================
        
        # Demand profile
        m.demand = Param(
            m.T,
            initialize={t: self.load_profile.demand[t] for t in range(T)},
            within=NonNegativeReals,
            doc="Electricity demand (kW)"
        )
        
        # Solar availability profile
        m.solar_availability = Param(
            m.T,
            initialize={t: self.solar_profile.availability[t] for t in range(T)},
            within=NonNegativeReals,
            doc="Solar availability factor (0-1)"
        )
        
        # --- Economic Parameters ---
        m.solar_capex = Param(
            initialize=self.economic.solar_capex,
            within=NonNegativeReals,
            doc="Solar PV CAPEX ($/kW)"
        )
        m.gas_capex = Param(
            initialize=self.economic.gas_capex,
            within=NonNegativeReals,
            doc="Gas generator CAPEX ($/kW)"
        )
        m.battery_capex = Param(
            initialize=self.economic.battery_capex,
            within=NonNegativeReals,
            doc="Battery CAPEX ($/kWh)"
        )
        
        # Annualized CAPEX using Capital Recovery Factor (CRF)
        m.solar_capex_annualized = Param(
            initialize=self.economic.get_annualized_cost(
                self.economic.solar_capex, self.economic.solar_lifetime
            ),
            within=NonNegativeReals,
            doc="Annualized solar CAPEX ($/kW/year)"
        )
        m.gas_capex_annualized = Param(
            initialize=self.economic.get_annualized_cost(
                self.economic.gas_capex, self.economic.gas_lifetime
            ),
            within=NonNegativeReals,
            doc="Annualized gas CAPEX ($/kW/year)"
        )
        m.battery_capex_annualized = Param(
            initialize=self.economic.get_annualized_cost(
                self.economic.battery_capex, self.economic.battery_lifetime
            ),
            within=NonNegativeReals,
            doc="Annualized battery CAPEX ($/kWh/year)"
        )
        
        m.gas_fuel_cost = Param(
            initialize=self.economic.gas_fuel_cost,
            within=NonNegativeReals,
            doc="Gas fuel cost ($/kWh)"
        )
        m.grid_tariff = Param(
            initialize=self.economic.grid_tariff,
            within=NonNegativeReals,
            doc="Grid import tariff ($/kWh)"
        )
        m.unserved_penalty = Param(
            initialize=self.economic.unserved_energy_penalty,
            within=NonNegativeReals,
            doc="Unserved energy penalty ($/kWh)"
        )
        m.emission_penalty = Param(
            initialize=self.economic.emission_penalty,
            within=NonNegativeReals,
            doc="Carbon price ($/ton CO2)"
        )
        
        # --- Technical Parameters ---
        m.battery_eff_charge = Param(
            initialize=self.technical.battery_charge_efficiency,
            within=NonNegativeReals,
            doc="Battery charge efficiency"
        )
        m.battery_eff_discharge = Param(
            initialize=self.technical.battery_discharge_efficiency,
            within=NonNegativeReals,
            doc="Battery discharge efficiency"
        )
        m.battery_c_rate = Param(
            initialize=self.technical.battery_c_rate,
            within=NonNegativeReals,
            doc="Battery C-rate (max charge/discharge rate)"
        )
        m.battery_min_soc_frac = Param(
            initialize=self.technical.battery_min_soc,
            within=NonNegativeReals,
            doc="Minimum SOC fraction"
        )
        m.battery_max_soc_frac = Param(
            initialize=self.technical.battery_max_soc,
            within=NonNegativeReals,
            doc="Maximum SOC fraction"
        )
        m.max_solar = Param(
            initialize=self.technical.max_solar_capacity,
            within=NonNegativeReals,
            doc="Maximum solar capacity (kW)"
        )
        m.max_gas = Param(
            initialize=self.technical.max_gas_capacity,
            within=NonNegativeReals,
            doc="Maximum gas capacity (kW)"
        )
        m.max_battery = Param(
            initialize=self.technical.max_battery_capacity,
            within=NonNegativeReals,
            doc="Maximum battery capacity (kWh)"
        )
        m.emission_factor_gas = Param(
            initialize=self.technical.emission_factor_gas,
            within=NonNegativeReals,
            doc="Gas emission factor (kg CO2/kWh)"
        )
        m.gas_ramp_rate = Param(
            initialize=self.technical.gas_ramp_rate,
            within=NonNegativeReals,
            doc="Gas generator ramp rate (fraction of capacity per hour)"
        )
        m.target_reliability = Param(
            initialize=self.technical.target_reliability,
            within=NonNegativeReals,
            doc="Target reliability (fraction)"
        )
        
        # Off-grid mode (no grid import available - typical for Nigerian rural mini-grids)
        m.grid_available = Param(
            initialize=0,  # Default: off-grid mode
            within=Binary,
            doc="Grid availability (1=available, 0=not available)"
        )
        
        # ===================
        # DECISION VARIABLES
        # ===================
        
        # --- Capacity Sizing Variables ---
        m.solar_capacity = Var(
            within=NonNegativeReals,
            bounds=(0, value(m.max_solar)),
            doc="Solar PV capacity (kW)"
        )
        
        m.gas_capacity = Var(
            within=NonNegativeReals,
            bounds=(0, value(m.max_gas)),
            doc="Gas generator capacity (kW)"
        )
        
        m.battery_capacity = Var(
            within=NonNegativeReals,
            bounds=(0, value(m.max_battery)),
            doc="Battery capacity (kWh)"
        )
        
        # --- Operational Variables (per time period) ---
        m.solar_gen = Var(
            m.T,
            within=NonNegativeReals,
            doc="Solar generation (kW)"
        )
        
        m.gas_gen = Var(
            m.T,
            within=NonNegativeReals,
            doc="Gas generation (kW)"
        )
        
        m.grid_import = Var(
            m.T,
            within=NonNegativeReals,
            doc="Grid import (kW)"
        )
        
        m.battery_charge = Var(
            m.T,
            within=NonNegativeReals,
            doc="Battery charging power (kW)"
        )
        
        m.battery_discharge = Var(
            m.T,
            within=NonNegativeReals,
            doc="Battery discharging power (kW)"
        )
        
        m.soc = Var(
            m.T,
            within=NonNegativeReals,
            doc="State of charge (kWh)"
        )
        
        m.unserved = Var(
            m.T,
            within=NonNegativeReals,
            doc="Unserved energy (kW)"
        )
        
        # --- Binary Variables (for MILP) ---
        m.gas_on = Var(
            m.T,
            within=Binary,
            doc="Gas generator on/off status"
        )
        
        # Battery mode binary: 1 = charging, 0 = discharging
        m.battery_charging = Var(
            m.T,
            within=Binary,
            doc="Battery charging mode (1=charging, 0=discharging/idle)"
        )
        
        # ===================
        # OBJECTIVE FUNCTION
        # ===================
        
        def total_cost_rule(model):
            """
            Minimize total system cost.
            
            Total Cost = Annualized Investment Cost + Fuel Cost + Grid Cost + Carbon Cost + Penalty Cost
            
            CAPEX is annualized using Capital Recovery Factor (CRF):
            CRF = r(1+r)^n / ((1+r)^n - 1)
            Annualized CAPEX = CAPEX × CRF
            """
            # Annualized investment cost using CRF
            # Note: For a single-day optimization, we scale by (1/365) to get daily cost
            daily_factor = 1.0 / 365.0
            
            investment_cost = (
                model.solar_capacity * model.solar_capex_annualized +
                model.gas_capacity * model.gas_capex_annualized +
                model.battery_capacity * model.battery_capex_annualized
            ) * daily_factor
            
            # Operating cost: fuel for gas generator
            fuel_cost = sum(
                model.gas_gen[t] * model.gas_fuel_cost 
                for t in model.T
            )
            
            # Grid import cost
            grid_cost = sum(
                model.grid_import[t] * model.grid_tariff 
                for t in model.T
            )
            
            # Carbon cost (emission penalty)
            carbon_cost = sum(
                model.gas_gen[t] * model.emission_factor_gas * model.emission_penalty / 1000.0
                for t in model.T
            )
            
            # Penalty for unserved energy (reliability incentive)
            penalty_cost = sum(
                model.unserved[t] * model.unserved_penalty 
                for t in model.T
            )
            
            return investment_cost + fuel_cost + grid_cost + carbon_cost + penalty_cost
        
        m.TotalCost = Objective(rule=total_cost_rule, sense=minimize)
        
        # ===================
        # CONSTRAINTS
        # ===================
        
        # --- Power Balance Constraint ---
        def power_balance_rule(model, t):
            """
            Power balance at each time period.
            
            Supply = Demand
            solar_gen + gas_gen + grid_import + battery_discharge - battery_charge + unserved = demand
            """
            return (
                model.solar_gen[t] + 
                model.gas_gen[t] + 
                model.grid_import[t] + 
                model.battery_discharge[t] - 
                model.battery_charge[t] + 
                model.unserved[t] == 
                model.demand[t]
            )
        
        m.PowerBalance = Constraint(m.T, rule=power_balance_rule)
        
        # --- Solar Generation Constraint ---
        def solar_generation_limit_rule(model, t):
            """
            Solar generation limited by capacity and availability.
            
            solar_gen[t] <= solar_capacity * solar_availability[t]
            """
            return model.solar_gen[t] <= model.solar_capacity * model.solar_availability[t]
        
        m.SolarLimit = Constraint(m.T, rule=solar_generation_limit_rule)
        
        # --- Gas Generation Constraints (Big-M Linearization) ---
        # The bilinear constraint gas_gen[t] <= gas_capacity * gas_on[t] is reformulated
        # using Big-M method into two linear constraints:
        # 1. gas_gen[t] <= M * gas_on[t]  (if gas_on=0, gas_gen=0)
        # 2. gas_gen[t] <= gas_capacity   (generation limited by capacity)
        
        def gas_generation_on_off_rule(model, t):
            """
            Gas generation is zero when generator is off.
            gas_gen[t] <= M * gas_on[t]
            where M is the maximum possible gas capacity.
            """
            return model.gas_gen[t] <= model.max_gas * model.gas_on[t]
        
        m.GasOnOffLimit = Constraint(m.T, rule=gas_generation_on_off_rule)
        
        def gas_generation_capacity_rule(model, t):
            """
            Gas generation limited by installed capacity.
            gas_gen[t] <= gas_capacity
            """
            return model.gas_gen[t] <= model.gas_capacity
        
        m.GasCapacityLimit = Constraint(m.T, rule=gas_generation_capacity_rule)
        
        # --- Battery Charge Limit (Linearized) ---
        # The constraint battery_charge[t] <= battery_capacity * c_rate is bilinear
        # We linearize by using max_battery * c_rate as an upper bound
        def battery_charge_limit_rule(model, t):
            """
            Battery charging limited by C-rate times capacity.
            Linearized: battery_charge[t] <= max_battery * c_rate (upper bound)
            """
            return model.battery_charge[t] <= model.max_battery * model.battery_c_rate
        
        m.BatteryChargeLimit = Constraint(m.T, rule=battery_charge_limit_rule)
        
        # --- Battery Discharge Limit (Linearized) ---
        def battery_discharge_limit_rule(model, t):
            """
            Battery discharging limited by C-rate times capacity.
            Linearized: battery_discharge[t] <= max_battery * c_rate (upper bound)
            """
            return model.battery_discharge[t] <= model.max_battery * model.battery_c_rate
        
        m.BatteryDischargeLimit = Constraint(m.T, rule=battery_discharge_limit_rule)
        
        # --- Battery SOC Dynamics (Linearized) ---
        # For Phase 1, we simplify the SOC dynamics to avoid bilinear terms.
        # Initial SOC is 0 - no free energy. Battery must be charged from solar/gas.
        def soc_dynamics_rule(model, t):
            """
            State of charge dynamics.
            
            SOC[t] = SOC[t-1] + charge * eff_charge - discharge / eff_discharge
            
            For t=0, initial SOC is 0 (no free energy).
            """
            if t == 0:
                # Initial condition: start at 0 (no free energy)
                return model.soc[t] == (
                    model.battery_charge[t] * model.battery_eff_charge -
                    model.battery_discharge[t] / model.battery_eff_discharge
                )
            else:
                return model.soc[t] == (
                    model.soc[t-1] +
                    model.battery_charge[t] * model.battery_eff_charge -
                    model.battery_discharge[t] / model.battery_eff_discharge
                )
        
        m.SOCDynamics = Constraint(m.T, rule=soc_dynamics_rule)
        
        # --- Battery SOC Limits (Linked to Capacity Decision) ---
        # SOC must be within bounds of INSTALLED capacity, not max theoretical capacity
        def soc_min_limit_rule(model, t):
            """
            State of charge must stay above minimum fraction of installed capacity.
            SOC[t] >= min_soc_frac * battery_capacity
            """
            return model.soc[t] >= model.battery_min_soc_frac * model.battery_capacity
        
        m.SOCMinLimit = Constraint(m.T, rule=soc_min_limit_rule)
        
        def soc_max_limit_rule(model, t):
            """
            State of charge must stay below maximum fraction of installed capacity.
            SOC[t] <= max_soc_frac * battery_capacity
            """
            return model.soc[t] <= model.battery_max_soc_frac * model.battery_capacity
        
        m.SOCMaxLimit = Constraint(m.T, rule=soc_max_limit_rule)
        
        # --- Battery Charge/Discharge Limits (Linked to Capacity) ---
        def battery_charge_capacity_limit_rule(model, t):
            """
            Battery charging limited by C-rate times INSTALLED capacity.
            battery_charge[t] <= battery_capacity * c_rate
            """
            return model.battery_charge[t] <= model.battery_capacity * model.battery_c_rate
        
        m.BatteryChargeCapacityLimit = Constraint(m.T, rule=battery_charge_capacity_limit_rule)
        
        def battery_discharge_capacity_limit_rule(model, t):
            """
            Battery discharging limited by C-rate times INSTALLED capacity.
            battery_discharge[t] <= battery_capacity * c_rate
            """
            return model.battery_discharge[t] <= model.battery_capacity * model.battery_c_rate
        
        m.BatteryDischargeCapacityLimit = Constraint(m.T, rule=battery_discharge_capacity_limit_rule)
        
        # --- Battery Mutual Exclusivity (Charge XOR Discharge) ---
        # Using Big-M method to prevent simultaneous charge and discharge
        # charge[t] <= M * battery_charging[t]
        # discharge[t] <= M * (1 - battery_charging[t])
        def battery_charge_exclusivity_rule(model, t):
            """
            Battery can only charge when in charging mode.
            battery_charge[t] <= max_battery * c_rate * battery_charging[t]
            """
            return model.battery_charge[t] <= model.max_battery * model.battery_c_rate * model.battery_charging[t]
        
        m.BatteryChargeExclusivity = Constraint(m.T, rule=battery_charge_exclusivity_rule)
        
        def battery_discharge_exclusivity_rule(model, t):
            """
            Battery can only discharge when not in charging mode.
            battery_discharge[t] <= max_battery * c_rate * (1 - battery_charging[t])
            """
            return model.battery_discharge[t] <= model.max_battery * model.battery_c_rate * (1 - model.battery_charging[t])
        
        m.BatteryDischargeExclusivity = Constraint(m.T, rule=battery_discharge_exclusivity_rule)
        
        # --- Gas Generator Ramp Rate Constraint ---
        # Limit how quickly gas generation can change between consecutive hours
        def gas_ramp_up_rule(model, t):
            """
            Gas generation ramp-up limit.
            gas_gen[t] - gas_gen[t-1] <= ramp_rate * gas_capacity
            """
            if t == 0:
                return Constraint.Skip  # No previous period for t=0
            return model.gas_gen[t] - model.gas_gen[t-1] <= model.gas_ramp_rate * model.gas_capacity
        
        m.GasRampUp = Constraint(m.T, rule=gas_ramp_up_rule)
        
        def gas_ramp_down_rule(model, t):
            """
            Gas generation ramp-down limit.
            gas_gen[t-1] - gas_gen[t] <= ramp_rate * gas_capacity
            """
            if t == 0:
                return Constraint.Skip  # No previous period for t=0
            return model.gas_gen[t-1] - model.gas_gen[t] <= model.gas_ramp_rate * model.gas_capacity
        
        m.GasRampDown = Constraint(m.T, rule=gas_ramp_down_rule)
        
        # --- Reliability Constraint ---
        def reliability_constraint_rule(model):
            """
            Total unserved energy limited by reliability target.
            
            sum(unserved) <= (1 - target_reliability) * sum(demand)
            
            This ensures the system meets the specified reliability level.
            """
            total_demand = sum(model.demand[t] for t in model.T)
            max_unserved = total_demand * (1 - model.target_reliability)
            return sum(model.unserved[t] for t in model.T) <= max_unserved
        
        m.Reliability = Constraint(rule=reliability_constraint_rule)
        
        # --- Grid Import Constraint (Off-grid Mode) ---
        def grid_limit_rule(model, t):
            """
            Limit grid import based on availability.
            For off-grid mode (grid_available=0), no grid import is allowed.
            This is typical for Nigerian rural mini-grids.
            """
            return model.grid_import[t] <= model.grid_available * model.demand[t]
        
        m.GridLimit = Constraint(m.T, rule=grid_limit_rule)
        
        # --- Dual Variables (for shadow prices) ---
        # Enable dual variable reporting for explainability
        m.dual = Suffix(direction=Suffix.IMPORT)
        
        self.model = m
        logger.info("Model built successfully")
        
        return m
    
    def get_model_summary(self) -> str:
        """
        Return a summary of the model structure.
        
        Returns:
            Formatted string with model summary
        """
        if self.model is None:
            return "Model not built yet. Call build_model() first."
        
        m = self.model
        
        summary = f"""
        {'='*60}
        HyOptima Model Summary: {self.name}
        {'='*60}
        
        SETS
        ----
          Time periods (T): {len(m.T)} hours
          Energy sources: {list(m.SOURCES)}
        
        DECISION VARIABLES
        ------------------
          Sizing:
            - solar_capacity (kW): bounds [0, {value(m.max_solar)}]
            - gas_capacity (kW): bounds [0, {value(m.max_gas)}]
            - battery_capacity (kWh): bounds [0, {value(m.max_battery)}]
          
          Operational (per hour):
            - solar_gen, gas_gen, grid_import (kW)
            - battery_charge, battery_discharge (kW)
            - soc (kWh), unserved (kW)
            - gas_on (binary)
        
        OBJECTIVE
        ---------
          Minimize: Total Cost
            = Investment + Fuel + Grid + Penalty
        
        CONSTRAINTS
        -----------
          - PowerBalance: Supply = Demand (each hour)
          - SolarLimit: solar_gen <= capacity * availability
          - GasLimit: gas_gen <= capacity * gas_on
          - BatteryChargeLimit: charge <= capacity * C_rate
          - BatteryDischargeLimit: discharge <= capacity * C_rate
          - SOCDynamics: SOC[t] = SOC[t-1] + charge - discharge
          - SOCLimit: min_soc <= SOC <= max_soc
          - Reliability: unserved <= (1 - target) * demand
        
        INPUT PROFILES
        --------------
          Load:
            - Duration: {self.load_profile.demand_duration} hours
            - Peak demand: {self.load_profile.peak_demand:.1f} kW
            - Total energy: {self.load_profile.total_energy:.1f} kWh
            - Load factor: {self.load_profile.load_factor:.2%}
          
          Solar:
            - Capacity factor: {self.solar_profile.capacity_factor:.2%}
            - Peak hours: {self.solar_profile.peak_hours:.1f} hours
            - Daylight hours: {self.solar_profile.daylight_hours}
        
        ECONOMIC PARAMETERS
        -------------------
          Solar CAPEX: ${value(m.solar_capex)}/kW
          Gas CAPEX: ${value(m.gas_capex)}/kW
          Battery CAPEX: ${value(m.battery_capex)}/kWh
          Gas fuel cost: ${value(m.gas_fuel_cost)}/kWh
          Grid tariff: ${value(m.grid_tariff)}/kWh
          Unserved penalty: ${value(m.unserved_penalty)}/kWh
        
        TECHNICAL PARAMETERS
        --------------------
          Battery efficiency: {value(m.battery_eff_charge):.0%} (charge), {value(m.battery_eff_discharge):.0%} (discharge)
          Battery C-rate: {value(m.battery_c_rate)}
          SOC limits: {value(m.battery_min_soc_frac):.0%} - {value(m.battery_max_soc_frac):.0%}
          Target reliability: {value(m.target_reliability):.1%}
        
        {'='*60}
        """
        return summary
    
    def get_variable_bounds(self) -> Dict[str, Tuple]:
        """Get bounds for all decision variables."""
        if self.model is None:
            return {}
        
        return {
            "solar_capacity": (0, value(self.model.max_solar)),
            "gas_capacity": (0, value(self.model.max_gas)),
            "battery_capacity": (0, value(self.model.max_battery)),
        }
    
    def validate_inputs(self) -> List[str]:
        """
        Validate input data for common issues.
        
        Returns:
            List of warning messages (empty if all valid)
        """
        warnings = []
        
        # Check demand profile
        if np.any(self.load_profile.demand < 0):
            warnings.append("Negative demand values detected")
        
        if np.all(self.load_profile.demand == 0):
            warnings.append("All demand values are zero")
        
        # Check solar profile
        if np.any(self.solar_profile.availability < 0):
            warnings.append("Negative solar availability values detected")
        
        if np.any(self.solar_profile.availability > 1):
            warnings.append("Solar availability values exceed 1.0")
        
        # Check consistency
        if len(self.load_profile.demand) != len(self.solar_profile.availability):
            warnings.append("Load and solar profiles have different lengths")
        
        # Check parameters
        if self.economic.unserved_energy_penalty <= self.economic.grid_tariff:
            warnings.append("Unserved penalty is lower than grid tariff - may lead to load shedding")
        
        return warnings

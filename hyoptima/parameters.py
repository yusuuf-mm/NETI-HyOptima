"""
Economic and Technical Parameters for HyOptima Model

This module defines the economic and technical parameters used in the
hybrid energy optimization model, aligned with Nigerian market conditions
and the Energy Transition Plan.

The parameters are based on:
- IEA World Energy Outlook cost curves
- IRENA renewable cost databases
- Nigerian electricity tariff structures
- Manufacturer specifications for equipment

Author: NETI-HyOptima Team
"""

from dataclasses import dataclass, field
from typing import Dict, Optional, List
import numpy as np


@dataclass
class EconomicParameters:
    """
    Economic parameters for energy system optimization.
    
    All costs are in USD unless otherwise specified.
    Default values are based on Nigerian market estimates and IEA cost curves.
    
    Attributes:
        solar_capex: Solar PV capital cost ($/kW)
        solar_opex: Solar PV fixed O&M cost ($/kW/year)
        solar_lifetime: Solar PV system lifetime (years)
        gas_capex: Gas generator capital cost ($/kW)
        gas_opex: Gas generator fixed O&M cost ($/kW/year)
        gas_fuel_cost: Gas fuel cost ($/kWh)
        gas_lifetime: Gas generator lifetime (years)
        battery_capex: Battery storage capital cost ($/kWh)
        battery_opex: Battery storage fixed O&M cost ($/kWh/year)
        battery_lifetime: Battery storage lifetime (years)
        grid_tariff: Grid electricity import tariff ($/kWh)
        grid_export_price: Grid electricity export price ($/kWh)
        unserved_energy_penalty: Penalty for unmet demand ($/kWh)
        emission_penalty: Carbon price ($/ton CO2)
        discount_rate: Discount rate for NPV calculations
    
    Example:
        >>> params = EconomicParameters(solar_capex=800, gas_fuel_cost=0.08)
        >>> annualized = params.get_annualized_cost(800, 25)
    """
    
    # Solar PV costs (based on IEA 2023 estimates for emerging markets)
    solar_capex: float = 800.0  # $/kW installed capacity
    solar_opex: float = 15.0    # $/kW/year fixed O&M
    solar_lifetime: int = 25    # years
    
    # Gas generator costs (Nigerian market estimates)
    gas_capex: float = 500.0    # $/kW installed capacity
    gas_opex: float = 25.0      # $/kW/year fixed O&M
    gas_fuel_cost: float = 0.08 # $/kWh fuel cost (LPG/Natural Gas)
    gas_lifetime: int = 20       # years
    
    # Battery storage costs (Li-ion, declining trend)
    battery_capex: float = 300.0  # $/kWh capacity
    battery_opex: float = 5.0     # $/kWh/year
    battery_lifetime: int = 10    # years (calendar life)
    
    # Grid costs (Nigerian electricity tariffs)
    grid_tariff: float = 0.12    # $/kWh import cost (commercial tariff)
    grid_export_price: float = 0.08  # $/kWh export revenue (feed-in tariff)
    
    # Penalty costs
    unserved_energy_penalty: float = 1.0  # $/kWh (high penalty for reliability)
    emission_penalty: float = 50.0  # $/ton CO2 (carbon price reference)
    
    # Discount rate for NPV calculations
    discount_rate: float = 0.10  # 10% typical for Nigeria (country risk adjusted)
    
    def get_annualized_cost(self, capex: float, lifetime: int) -> float:
        """
        Calculate annualized capital cost using Capital Recovery Factor (CRF).
        
        CRF = r(1+r)^n / ((1+r)^n - 1)
        
        Args:
            capex: Capital expenditure ($)
            lifetime: Asset lifetime (years)
        
        Returns:
            Annualized cost ($/year)
        """
        r = self.discount_rate
        if r == 0:
            return capex / lifetime
        crf = (r * (1 + r)**lifetime) / ((1 + r)**lifetime - 1)
        return capex * crf
    
    def get_lcoe_contribution(
        self, 
        capex: float, 
        opex: float, 
        lifetime: int, 
        annual_generation: float
    ) -> float:
        """
        Calculate LCOE contribution from a generation asset.
        
        LCOE = (Annualized CAPEX + OPEX) / Annual Generation
        
        Args:
            capex: Capital cost per unit capacity
            opex: Annual O&M cost per unit capacity
            lifetime: Asset lifetime (years)
            annual_generation: Annual energy generation per unit capacity
        
        Returns:
            LCOE contribution ($/kWh)
        """
        annualized_capex = self.get_annualized_cost(capex, lifetime)
        if annual_generation > 0:
            return (annualized_capex + opex) / annual_generation
        return float('inf')
    
    def to_dict(self) -> Dict:
        """Convert parameters to dictionary."""
        return {
            "solar_capex": self.solar_capex,
            "solar_opex": self.solar_opex,
            "solar_lifetime": self.solar_lifetime,
            "gas_capex": self.gas_capex,
            "gas_opex": self.gas_opex,
            "gas_fuel_cost": self.gas_fuel_cost,
            "gas_lifetime": self.gas_lifetime,
            "battery_capex": self.battery_capex,
            "battery_opex": self.battery_opex,
            "battery_lifetime": self.battery_lifetime,
            "grid_tariff": self.grid_tariff,
            "grid_export_price": self.grid_export_price,
            "unserved_energy_penalty": self.unserved_energy_penalty,
            "emission_penalty": self.emission_penalty,
            "discount_rate": self.discount_rate,
        }


@dataclass
class TechnicalParameters:
    """
    Technical parameters for energy system components.
    
    Default values based on typical Nigerian installations and manufacturer specs.
    
    Attributes:
        solar_efficiency: Solar panel efficiency
        solar_degradation: Annual degradation rate
        solar_inverter_efficiency: Inverter efficiency
        gas_min_load: Minimum load factor for gas generator
        gas_max_efficiency: Maximum thermal efficiency
        gas_ramp_rate: Maximum ramp rate per hour
        battery_charge_efficiency: Battery charging efficiency
        battery_discharge_efficiency: Battery discharging efficiency
        battery_min_soc: Minimum state of charge
        battery_max_soc: Maximum state of charge
        battery_c_rate: Maximum charge/discharge rate (C-rate)
        max_solar_capacity: Maximum solar capacity limit
        max_gas_capacity: Maximum gas capacity limit
        max_battery_capacity: Maximum battery capacity limit
        emission_factor_gas: Gas emission factor (kg CO2/kWh)
        emission_factor_diesel: Diesel emission factor (kg CO2/kWh)
        emission_factor_grid: Grid emission factor (kg CO2/kWh)
        target_reliability: Target reliability level
        max_unserved_fraction: Maximum unserved energy fraction
    """
    
    # Solar PV technical specs
    solar_efficiency: float = 0.18  # Panel efficiency (typical for commercial panels)
    solar_degradation: float = 0.005  # Annual degradation rate (0.5%/year)
    solar_inverter_efficiency: float = 0.96  # Inverter efficiency
    
    # Gas generator technical specs
    gas_min_load: float = 0.3  # Minimum load factor (30%)
    gas_max_efficiency: float = 0.40  # Maximum thermal efficiency
    gas_ramp_rate: float = 0.5  # Max ramp rate per hour (50% of capacity)
    
    # Battery technical specs (Li-ion)
    battery_charge_efficiency: float = 0.95
    battery_discharge_efficiency: float = 0.95
    battery_min_soc: float = 0.1  # Minimum state of charge (10%)
    battery_max_soc: float = 0.95  # Maximum state of charge (95%)
    battery_c_rate: float = 0.5  # Max charge/discharge rate (C/2)
    
    # System constraints (for optimization bounds)
    max_solar_capacity: float = 10000.0  # kW (10 MW max for microgrid)
    max_gas_capacity: float = 5000.0     # kW (5 MW max)
    max_battery_capacity: float = 20000.0  # kWh (20 MWh max)
    
    # Emission factors (kg CO2/kWh) - based on IPCC and IEA data
    emission_factor_gas: float = 0.45  # Natural gas
    emission_factor_diesel: float = 0.70  # Diesel generator
    emission_factor_grid: float = 0.55  # Nigerian grid average (gas-dominated)
    
    # Reliability parameters
    target_reliability: float = 0.99  # 99% reliability target
    max_unserved_fraction: float = 0.01  # Max 1% unserved energy
    
    def get_round_trip_efficiency(self) -> float:
        """Calculate battery round-trip efficiency."""
        return self.battery_charge_efficiency * self.battery_discharge_efficiency
    
    def to_dict(self) -> Dict:
        """Convert parameters to dictionary."""
        return {
            "solar_efficiency": self.solar_efficiency,
            "solar_degradation": self.solar_degradation,
            "solar_inverter_efficiency": self.solar_inverter_efficiency,
            "gas_min_load": self.gas_min_load,
            "gas_max_efficiency": self.gas_max_efficiency,
            "gas_ramp_rate": self.gas_ramp_rate,
            "battery_charge_efficiency": self.battery_charge_efficiency,
            "battery_discharge_efficiency": self.battery_discharge_efficiency,
            "battery_min_soc": self.battery_min_soc,
            "battery_max_soc": self.battery_max_soc,
            "battery_c_rate": self.battery_c_rate,
            "emission_factor_gas": self.emission_factor_gas,
            "emission_factor_diesel": self.emission_factor_diesel,
            "emission_factor_grid": self.emission_factor_grid,
            "target_reliability": self.target_reliability,
        }


@dataclass
class PolicyParameters:
    """
    Policy constraints aligned with Nigeria Energy Transition Plan (ETP).
    
    The Nigeria ETP sets ambitious targets for 2030 and 2060 net-zero:
    - 30 GW of renewable capacity by 2030
    - 5 million solar home systems
    - Gas as transition fuel
    - $1.9 trillion investment required
    
    Attributes:
        emission_cap_annual: Annual emission cap (tons CO2/year)
        emission_reduction_target: Emission reduction target (fraction)
        renewable_fraction_min: Minimum renewable fraction
        solar_capacity_target: Solar capacity target (kW)
        max_capex_budget: Maximum capital expenditure budget ($)
        planning_horizon: Planning horizon (years)
        target_year: Target year for policy goals
        gas_transition_phase: Phase of gas transition strategy
    """
    
    # Emission targets (aligned with Nigeria ETP)
    emission_cap_annual: float = 10000.0  # tons CO2/year (example for state-level)
    emission_reduction_target: float = 0.20  # 20% reduction by 2030
    
    # Renewable targets
    renewable_fraction_min: float = 0.30  # 30% renewable by 2030
    solar_capacity_target: float = 30000.0  # kW state-level target
    
    # Investment limits
    max_capex_budget: float = 10_000_000.0  # $10M max investment
    
    # Timeline
    planning_horizon: int = 25  # years (2025-2050)
    target_year: int = 2030
    
    # Gas transition strategy
    gas_transition_phase: str = "transition"  # "transition", "phase-out", "none"
    gas_max_fraction: float = 0.50  # Maximum gas fraction during transition
    
    def get_emission_intensity_target(self, baseline_emission: float) -> float:
        """
        Calculate target emission intensity.
        
        Args:
            baseline_emission: Baseline emission level (tons CO2/year)
        
        Returns:
            Target emission level (tons CO2/year)
        """
        return baseline_emission * (1 - self.emission_reduction_target)
    
    def is_renewable_target_met(self, renewable_fraction: float) -> bool:
        """Check if renewable target is met."""
        return renewable_fraction >= self.renewable_fraction_min
    
    def to_dict(self) -> Dict:
        """Convert parameters to dictionary."""
        return {
            "emission_cap_annual": self.emission_cap_annual,
            "emission_reduction_target": self.emission_reduction_target,
            "renewable_fraction_min": self.renewable_fraction_min,
            "solar_capacity_target": self.solar_capacity_target,
            "max_capex_budget": self.max_capex_budget,
            "planning_horizon": self.planning_horizon,
            "target_year": self.target_year,
            "gas_transition_phase": self.gas_transition_phase,
            "gas_max_fraction": self.gas_max_fraction,
        }


# Predefined parameter sets for common Nigerian scenarios

def get_parameters_by_scenario(scenario: str) -> Dict:
    """
    Get predefined parameter sets for common Nigerian scenarios.
    
    Args:
        scenario: Scenario name ('rural_minigrid', 'urban_industrial', 
                  'commercial', 'residential_estate')
    
    Returns:
        Dictionary with economic, technical, and policy parameters
    """
    scenarios = {
        "rural_minigrid": {
            "economic": EconomicParameters(
                solar_capex=900,  # Higher due to remote location
                gas_capex=600,
                gas_fuel_cost=0.10,  # Higher fuel transport cost
                battery_capex=350,
                unserved_energy_penalty=0.5,  # Lower penalty for rural
            ),
            "technical": TechnicalParameters(
                max_solar_capacity=1000.0,  # 1 MW max
                max_gas_capacity=500.0,     # 500 kW max
                max_battery_capacity=2000.0,  # 2 MWh max
                target_reliability=0.95,  # 95% acceptable for rural
            ),
            "policy": PolicyParameters(
                renewable_fraction_min=0.50,  # Higher renewable target
                max_capex_budget=2_000_000.0,  # $2M budget
            ),
        },
        "urban_industrial": {
            "economic": EconomicParameters(
                solar_capex=750,  # Lower due to economies of scale
                gas_capex=450,
                gas_fuel_cost=0.06,  # Pipeline gas available
                battery_capex=280,
                unserved_energy_penalty=2.0,  # High penalty for industrial
            ),
            "technical": TechnicalParameters(
                max_solar_capacity=10000.0,  # 10 MW
                max_gas_capacity=5000.0,     # 5 MW
                max_battery_capacity=20000.0,  # 20 MWh
                target_reliability=0.999,  # 99.9% for industrial
            ),
            "policy": PolicyParameters(
                renewable_fraction_min=0.30,
                max_capex_budget=20_000_000.0,  # $20M budget
            ),
        },
        "commercial": {
            "economic": EconomicParameters(
                solar_capex=800,
                gas_capex=500,
                gas_fuel_cost=0.08,
                battery_capex=300,
                unserved_energy_penalty=1.5,
            ),
            "technical": TechnicalParameters(
                max_solar_capacity=2000.0,
                max_gas_capacity=1000.0,
                max_battery_capacity=4000.0,
                target_reliability=0.99,
            ),
            "policy": PolicyParameters(
                renewable_fraction_min=0.35,
                max_capex_budget=5_000_000.0,
            ),
        },
        "residential_estate": {
            "economic": EconomicParameters(
                solar_capex=850,
                gas_capex=550,
                gas_fuel_cost=0.09,
                battery_capex=320,
                unserved_energy_penalty=0.8,
            ),
            "technical": TechnicalParameters(
                max_solar_capacity=500.0,
                max_gas_capacity=250.0,
                max_battery_capacity=1000.0,
                target_reliability=0.98,
            ),
            "policy": PolicyParameters(
                renewable_fraction_min=0.40,
                max_capex_budget=1_000_000.0,
            ),
        },
    }
    
    return scenarios.get(scenario, {
        "economic": EconomicParameters(),
        "technical": TechnicalParameters(),
        "policy": PolicyParameters(),
    })

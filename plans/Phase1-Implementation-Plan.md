# NETI-HyOptima: Phase 1 Implementation Plan

## Overview

This document provides the detailed implementation plan for Phase 1 of NETI-HyOptima: building the core energy optimization model (HyOptima v0). This is the mathematical heart of the platform - a working energy optimization run in Python with no frontend, database, or cloud infrastructure.

---

## 1. Repository Structure to Create

```
neti-hyoptima/
|
+-- README.md                          # Project overview (exists, needs expansion)
+-- requirements.txt                   # Python dependencies
+-- .gitignore                         # Git ignore patterns
|
+-- notebooks/                         # Exploration & experiments
|   +-- phase1_core_model.ipynb        # Working demonstration notebook
|   +-- data_exploration.ipynb         # Data profiling notebook
|
+-- data/
|   +-- raw/                           # Untouched datasets (future)
|   +-- processed/                     # Cleaned inputs (future)
|   +-- synthetic/                     # Generated test data
|       +-- .gitkeep
|
+-- hyoptima/                          # THE ACTUAL ENGINE
|   +-- __init__.py                    # Package initialization
|   +-- model.py                       # Pyomo model definition
|   +-- parameters.py                  # Economic + tech assumptions
|   +-- profiles.py                    # Load & solar generation profiles
|   +-- solver.py                      # Run optimization
|   +-- utils.py                       # Helper functions
|
+-- results/
|   +-- figures/                       # Generated plots
|   +-- runs/                          # Optimization run outputs
|   +-- .gitkeep
|
+-- docs/
|   +-- phase1_notes.md                # Phase 1 documentation
|   +-- mathematical_derivation.md     # Math model derivation
|   +-- data_sources.md                # Data source documentation
|
+-- tests/                             # Unit tests
    +-- __init__.py
    +-- test_model.py
    +-- test_solver.py
```

---

## 2. File Contents Specification

### 2.1 requirements.txt

```
# NETI-HyOptima Core Dependencies
# Phase 1: Mathematical Optimization Engine

# Optimization
pyomo>=6.6.0
highspy>=1.5.0
pulp>=2.7.0

# Data Processing
numpy>=1.24.0
pandas>=2.0.0

# Visualization
matplotlib>=3.7.0
plotly>=5.15.0

# Machine Learning (Phase 2+)
scikit-learn>=1.3.0
xgboost>=1.7.0

# API Development (Phase 4+)
fastapi>=0.100.0
uvicorn>=0.23.0
pydantic>=2.0.0

# Testing
pytest>=7.4.0
pytest-cov>=4.1.0

# Utilities
python-dotenv>=1.0.0
tqdm>=4.65.0
```

### 2.2 .gitignore

```
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# Jupyter Notebook
.ipynb_checkpoints/

# Environment
.env
.venv
env/
venv/
ENV/

# IDE
.vscode/
.idea/
*.swp
*.swo

# Data (large files)
data/raw/*
!data/raw/.gitkeep
results/runs/*
!results/runs/.gitkeep

# OS
.DS_Store
Thumbs.db

# Testing
.pytest_cache/
.coverage
htmlcov/
```

### 2.3 hyoptima/__init__.py

```python
"""
NETI-HyOptima: Nigeria Energy Transition Intelligence Platform
HyOptima Core Engine - Hybrid Energy Optimization

This package contains the core optimization engine for hybrid energy systems.
"""

__version__ = "0.1.0"
__author__ = "NETI-HyOptima Team"

from .model import HyOptimaModel
from .solver import HyOptimaSolver
from .parameters import EconomicParameters, TechnicalParameters
from .profiles import LoadProfile, SolarProfile

__all__ = [
    "HyOptimaModel",
    "HyOptimaSolver", 
    "EconomicParameters",
    "TechnicalParameters",
    "LoadProfile",
    "SolarProfile",
]
```

### 2.4 hyoptima/parameters.py

```python
"""
Economic and Technical Parameters for HyOptima Model

This module defines the economic and technical parameters used in the
hybrid energy optimization model, aligned with Nigerian market conditions
and the Energy Transition Plan.
"""

from dataclasses import dataclass
from typing import Dict, Optional


@dataclass
class EconomicParameters:
    """
    Economic parameters for energy system optimization.
    
    All costs are in USD unless otherwise specified.
    Default values are based on Nigerian market estimates and IEA cost curves.
    """
    
    # Solar PV costs
    solar_capex: float = 800.0  # $/kW installed capacity
    solar_opex: float = 15.0    # $/kW/year fixed O&M
    solar_lifetime: int = 25    # years
    
    # Gas generator costs
    gas_capex: float = 500.0    # $/kW installed capacity
    gas_opex: float = 25.0      # $/kW/year fixed O&M
    gas_fuel_cost: float = 0.08 # $/kWh fuel cost
    gas_lifetime: int = 20       # years
    
    # Battery storage costs
    battery_capex: float = 300.0  # $/kWh capacity
    battery_opex: float = 5.0     # $/kWh/year
    battery_lifetime: int = 10    # years
    
    # Grid costs (if applicable)
    grid_tariff: float = 0.12    # $/kWh import cost
    grid_export_price: float = 0.08  # $/kWh export revenue
    
    # Penalty costs
    unserved_energy_penalty: float = 1.0  # $/kWh (high penalty for reliability)
    emission_penalty: float = 50.0  # $/ton CO2 (carbon price)
    
    # Discount rate for NPV calculations
    discount_rate: float = 0.10  # 10% typical for Nigeria
    
    def get_annualized_cost(self, capex: float, lifetime: int) -> float:
        """Calculate annualized capital cost using CRF."""
        r = self.discount_rate
        if r == 0:
            return capex / lifetime
        crf = (r * (1 + r)**lifetime) / ((1 + r)**lifetime - 1)
        return capex * crf


@dataclass
class TechnicalParameters:
    """
    Technical parameters for energy system components.
    
    Default values based on typical Nigerian installations and manufacturer specs.
    """
    
    # Solar PV technical specs
    solar_efficiency: float = 0.18  # Panel efficiency
    solar_degradation: float = 0.005  # Annual degradation rate
    solar_inverter_efficiency: float = 0.96
    
    # Gas generator technical specs
    gas_min_load: float = 0.3  # Minimum load factor (30%)
    gas_max_efficiency: float = 0.40  # Maximum thermal efficiency
    gas_ramp_rate: float = 0.5  # Max ramp rate per hour (50% of capacity)
    
    # Battery technical specs
    battery_charge_efficiency: float = 0.95
    battery_discharge_efficiency: float = 0.95
    battery_min_soc: float = 0.1  # Minimum state of charge (10%)
    battery_max_soc: float = 0.95  # Maximum state of charge (95%)
    battery_c_rate: float = 0.5  # Max charge/discharge rate (C/2)
    
    # System constraints
    max_solar_capacity: float = 10000.0  # kW (10 MW max for microgrid)
    max_gas_capacity: float = 5000.0     # kW (5 MW max)
    max_battery_capacity: float = 20000.0  # kWh (20 MWh max)
    
    # Emission factors (kg CO2/kWh)
    emission_factor_gas: float = 0.45
    emission_factor_diesel: float = 0.70
    emission_factor_grid: float = 0.55  # Nigerian grid average
    
    # Reliability parameters
    target_reliability: float = 0.99  # 99% reliability target
    max_unserved_fraction: float = 0.01  # Max 1% unserved energy


@dataclass
class PolicyParameters:
    """
    Policy constraints aligned with Nigeria Energy Transition Plan.
    """
    
    # Emission targets
    emission_cap_annual: float = 10000.0  # tons CO2/year
    emission_reduction_target: float = 0.20  # 20% reduction by 2030
    
    # Renewable targets
    renewable_fraction_min: float = 0.30  # 30% renewable by 2030
    solar_capacity_target: float = 30000.0  # kW state-level target
    
    # Investment limits
    max_capex_budget: float = 10_000_000.0  # $10M max investment
    
    # Timeline
    planning_horizon: int = 25  # years (2025-2050)
    target_year: int = 2030
```

### 2.5 hyoptima/profiles.py

```python
"""
Load and Solar Generation Profiles for HyOptima Model

This module provides functions to generate and manipulate load profiles
and solar generation profiles for Nigerian energy systems.
"""

import numpy as np
from typing import List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class LoadProfile:
    """
    Represents an electrical load profile over time.
    
    Default profiles are based on typical Nigerian consumption patterns
    for residential, commercial, and mixed communities.
    """
    
    demand: np.ndarray  # Hourly demand in kW
    time_resolution: int = 1  # Hours per time step
    
    @classmethod
    def generate_synthetic(
        cls,
        peak_demand: float = 500.0,  # kW
        profile_type: str = "residential",
        hours: int = 24,
        noise_level: float = 0.05
    ) -> "LoadProfile":
        """
        Generate a synthetic load profile based on typical patterns.
        
        Args:
            peak_demand: Peak demand in kW
            profile_type: Type of load profile (residential, commercial, mixed)
            hours: Number of hours to generate
            noise_level: Random noise level (fraction of demand)
        
        Returns:
            LoadProfile with synthetic demand data
        """
        t = np.arange(hours)
        
        if profile_type == "residential":
            # Nigerian residential: morning peak, evening peak, low night
            base = 0.3 + 0.2 * np.sin(2 * np.pi * (t - 6) / 24)
            morning_peak = 0.4 * np.exp(-((t - 7)**2) / 4)
            evening_peak = 0.5 * np.exp(-((t - 20)**2) / 4)
            profile = base + morning_peak + evening_peak
            
        elif profile_type == "commercial":
            # Commercial: high during business hours (8-18)
            business_hours = np.where((t >= 8) & (t <= 18), 0.8, 0.2)
            lunch_dip = 0.2 * np.exp(-((t - 13)**2) / 2)
            profile = business_hours - lunch_dip
            
        elif profile_type == "mixed":
            # Mixed community: combination of patterns
            residential = 0.3 + 0.2 * np.sin(2 * np.pi * (t - 6) / 24)
            commercial = np.where((t >= 8) & (t <= 18), 0.4, 0.1)
            evening = 0.3 * np.exp(-((t - 20)**2) / 4)
            profile = residential + commercial + evening
            
        else:
            # Default: flat profile with slight variation
            profile = 0.5 + 0.1 * np.sin(2 * np.pi * t / 24)
        
        # Normalize and scale
        profile = profile / profile.max() * peak_demand
        
        # Add noise
        if noise_level > 0:
            noise = np.random.normal(0, noise_level * peak_demand, hours)
            profile = np.maximum(profile + noise, 0)
        
        return cls(demand=profile, time_resolution=1)
    
    @property
    def total_energy(self) -> float:
        """Total energy demand in kWh."""
        return float(np.sum(self.demand) * self.time_resolution)
    
    @property
    def peak_demand(self) -> float:
        """Peak demand in kW."""
        return float(np.max(self.demand))
    
    @property
    def load_factor(self) -> float:
        """Load factor (average/peak)."""
        if self.peak_demand == 0:
            return 0
        return float(np.mean(self.demand) / self.peak_demand)


@dataclass
class SolarProfile:
    """
    Represents solar generation profile (normalized irradiance).
    
    Profiles are based on Nigerian solar resource data (NASA POWER).
    Nigeria has excellent solar potential: 4-6 kWh/m2/day average.
    """
    
    availability: np.ndarray  # Normalized availability (0-1)
    time_resolution: int = 1  # Hours per time step
    
    @classmethod
    def generate_synthetic(
        cls,
        sunrise_hour: int = 6,
        sunset_hour: int = 18,
        peak_irradiance: float = 1.0,
        hours: int = 24,
        cloud_factor: float = 0.85,
        noise_level: float = 0.03
    ) -> "SolarProfile":
        """
        Generate a synthetic solar availability profile.
        
        Args:
            sunrise_hour: Hour of sunrise
            sunset_hour: Hour of sunset
            peak_irradiance: Peak normalized irradiance (0-1)
            hours: Number of hours to generate
            cloud_factor: Cloud cover reduction factor
            noise_level: Random noise level
        
        Returns:
            SolarProfile with synthetic availability data
        """
        t = np.arange(hours)
        
        # Daylight hours only
        daylight = np.zeros(hours)
        day_mask = (t >= sunrise_hour) & (t <= sunset_hour)
        
        # Bell curve for solar intensity during day
        day_length = sunset_hour - sunrise_hour
        midday = (sunrise_hour + sunset_hour) / 2
        intensity = np.exp(-((t - midday)**2) / (2 * (day_length/3)**2))
        
        daylight = np.where(day_mask, intensity, 0)
        
        # Apply cloud factor and noise
        profile = daylight * peak_irradiance * cloud_factor
        
        if noise_level > 0:
            noise = np.random.normal(0, noise_level, hours)
            profile = np.clip(profile + noise * daylight, 0, 1)
        
        return cls(availability=profile, time_resolution=1)
    
    @classmethod
    def from_capacity_factor(
        cls,
        capacity_factor: float = 0.20,
        hours: int = 24
    ) -> "SolarProfile":
        """
        Generate profile from annual capacity factor.
        
        Args:
            capacity_factor: Annual capacity factor (typical Nigeria: 18-22%)
            hours: Number of hours to generate
        
        Returns:
            SolarProfile scaled to match capacity factor
        """
        # Generate base profile
        profile = cls.generate_synthetic(hours=hours, noise_level=0)
        
        # Scale to match capacity factor
        # CF = actual_energy / (capacity * hours)
        # For normalized profile: mean should equal CF
        current_mean = np.mean(profile.availability)
        if current_mean > 0:
            scale = capacity_factor / current_mean
            profile.availability = np.clip(profile.availability * scale, 0, 1)
        
        return profile
    
    @property
    def capacity_factor(self) -> float:
        """Calculate capacity factor from profile."""
        return float(np.mean(self.availability))
    
    @property
    def peak_hours(self) -> int:
        """Number of peak sun hours (equivalent full-power hours)."""
        return int(np.sum(self.availability))


def generate_nigeria_scenarios(
    location: str = "kano",
    season: str = "dry"
) -> Tuple[LoadProfile, SolarProfile]:
    """
    Generate location-specific profiles for Nigerian cities.
    
    Args:
        location: Nigerian city name
        season: Season (dry/wet)
    
    Returns:
        Tuple of (LoadProfile, SolarProfile)
    """
    # Location-specific parameters
    location_params = {
        "kano": {"peak_demand": 600, "solar_cf": 0.22},
        "lagos": {"peak_demand": 800, "solar_cf": 0.18},
        "abuja": {"peak_demand": 500, "solar_cf": 0.20},
        "port_harcourt": {"peak_demand": 700, "solar_cf": 0.16},
        "bauchi": {"peak_demand": 300, "solar_cf": 0.21},
    }
    
    params = location_params.get(location, {"peak_demand": 500, "solar_cf": 0.20})
    
    # Season adjustments
    if season == "wet":
        solar_cf = params["solar_cf"] * 0.85  # Reduced solar in wet season
    else:
        solar_cf = params["solar_cf"]
    
    load = LoadProfile.generate_synthetic(
        peak_demand=params["peak_demand"],
        profile_type="mixed"
    )
    
    solar = SolarProfile.from_capacity_factor(capacity_factor=solar_cf)
    
    return load, solar
```

### 2.6 hyoptima/model.py

```python
"""
HyOptima Core Optimization Model

This module defines the Pyomo optimization model for hybrid energy systems.
It implements the mathematical formulation for minimizing total system cost
while meeting demand and policy constraints.

Mathematical Formulation:
    minimize: investment_cost + fuel_cost + penalty_cost
    
    subject to:
        - Power balance at each time period
        - Generator capacity limits
        - Renewable availability limits
        - Battery state-of-charge dynamics
        - Emission constraints (optional)
        - Reliability requirements
"""

from pyomo.environ import (
    ConcreteModel, Set, Param, Var, Objective, Constraint,
    NonNegativeReals, Binary, minimize, value, Suffix
)
from typing import Dict, Optional, Tuple
import numpy as np

from .parameters import EconomicParameters, TechnicalParameters, PolicyParameters
from .profiles import LoadProfile, SolarProfile


class HyOptimaModel:
    """
    HyOptima Hybrid Energy Optimization Model.
    
    This class builds and solves the optimization problem for sizing
    and operating a hybrid energy system with solar, gas, and storage.
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
            economic_params: Economic parameters (costs)
            technical_params: Technical parameters (efficiencies, limits)
            policy_params: Policy constraints (emissions, targets)
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
        
    def build_model(self) -> ConcreteModel:
        """
        Build the Pyomo optimization model.
        
        Returns:
            ConcreteModel: The built Pyomo model
        """
        m = ConcreteModel(name=self.name)
        
        # ===================
        # SETS
        # ===================
        
        # Time periods
        T = len(self.load_profile.demand)
        m.T = Set(initialize=range(T), doc="Time periods (hours)")
        
        # Energy sources
        m.SOURCES = Set(initialize=['solar', 'gas', 'grid'], doc="Energy sources")
        
        # ===================
        # PARAMETERS
        # ===================
        
        # Demand
        m.demand = Param(
            m.T,
            initialize={t: self.load_profile.demand[t] for t in range(T)},
            doc="Electricity demand (kW)"
        )
        
        # Solar availability
        m.solar_availability = Param(
            m.T,
            initialize={t: self.solar_profile.availability[t] for t in range(T)},
            doc="Solar availability factor (0-1)"
        )
        
        # Economic parameters
        m.solar_capex = Param(initialize=self.economic.solar_capex, doc="Solar CAPEX ($/kW)")
        m.gas_capex = Param(initialize=self.economic.gas_capex, doc="Gas CAPEX ($/kW)")
        m.battery_capex = Param(initialize=self.economic.battery_capex, doc="Battery CAPEX ($/kWh)")
        m.gas_fuel_cost = Param(initialize=self.economic.gas_fuel_cost, doc="Gas fuel cost ($/kWh)")
        m.grid_tariff = Param(initialize=self.economic.grid_tariff, doc="Grid tariff ($/kWh)")
        m.unserved_penalty = Param(initialize=self.economic.unserved_energy_penalty, doc="Unserved energy penalty ($/kWh)")
        
        # Technical parameters
        m.battery_eff_charge = Param(initialize=self.technical.battery_charge_efficiency, doc="Battery charge efficiency")
        m.battery_eff_discharge = Param(initialize=self.technical.battery_discharge_efficiency, doc="Battery discharge efficiency")
        m.max_solar = Param(initialize=self.technical.max_solar_capacity, doc="Max solar capacity (kW)")
        m.max_gas = Param(initialize=self.technical.max_gas_capacity, doc="Max gas capacity (kW)")
        m.max_battery = Param(initialize=self.technical.max_battery_capacity, doc="Max battery capacity (kWh)")
        m.emission_factor_gas = Param(initialize=self.technical.emission_factor_gas, doc="Gas emission factor (kg CO2/kWh)")
        
        # ===================
        # DECISION VARIABLES
        # ===================
        
        # Capacity sizing variables
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
        
        # Operational variables (per time period)
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
            doc="Battery charging (kW)"
        )
        
        m.battery_discharge = Var(
            m.T,
            within=NonNegativeReals,
            doc="Battery discharging (kW)"
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
        
        # Binary variable for gas generator on/off (optional for MILP)
        m.gas_on = Var(
            m.T,
            within=Binary,
            doc="Gas generator on/off status"
        )
        
        # ===================
        # OBJECTIVE FUNCTION
        # ===================
        
        def total_cost_rule(model):
            """Minimize total system cost."""
            # Investment cost (annualized)
            investment = (
                model.solar_capacity * model.solar_capex +
                model.gas_capacity * model.gas_capex +
                model.battery_capacity * model.battery_capex
            )
            
            # Fuel cost (operational)
            fuel = sum(model.gas_gen[t] * model.gas_fuel_cost for t in model.T)
            
            # Grid import cost
            grid_cost = sum(model.grid_import[t] * model.grid_tariff for t in model.T)
            
            # Unserved energy penalty
            penalty = sum(model.unserved[t] * model.unserved_penalty for t in model.T)
            
            return investment + fuel + grid_cost + penalty
        
        m.TotalCost = Objective(rule=total_cost_rule, sense=minimize)
        
        # ===================
        # CONSTRAINTS
        # ===================
        
        def power_balance_rule(model, t):
            """Power balance at each time period."""
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
        
        def solar_generation_limit_rule(model, t):
            """Solar generation limited by capacity and availability."""
            return model.solar_gen[t] <= model.solar_capacity * model.solar_availability[t]
        
        m.SolarLimit = Constraint(m.T, rule=solar_generation_limit_rule)
        
        def gas_generation_limit_rule(model, t):
            """Gas generation limited by capacity and on/off status."""
            return model.gas_gen[t] <= model.gas_capacity * model.gas_on[t]
        
        m.GasLimit = Constraint(m.T, rule=gas_generation_limit_rule)
        
        def battery_charge_limit_rule(model, t):
            """Battery charging limited by capacity and C-rate."""
            return model.battery_charge[t] <= model.battery_capacity * self.technical.battery_c_rate
        
        m.BatteryChargeLimit = Constraint(m.T, rule=battery_charge_limit_rule)
        
        def battery_discharge_limit_rule(model, t):
            """Battery discharging limited by capacity and C-rate."""
            return model.battery_discharge[t] <= model.battery_capacity * self.technical.battery_c_rate
        
        m.BatteryDischargeLimit = Constraint(m.T, rule=battery_discharge_limit_rule)
        
        def soc_dynamics_rule(model, t):
            """State of charge dynamics."""
            if t == 0:
                # Initial SOC: start at 50% capacity
                return model.soc[t] == 0.5 * model.battery_capacity + \
                       model.battery_charge[t] * model.battery_eff_charge - \
                       model.battery_discharge[t] / model.battery_eff_discharge
            else:
                return model.soc[t] == model.soc[t-1] + \
                       model.battery_charge[t] * model.battery_eff_charge - \
                       model.battery_discharge[t] / model.battery_eff_discharge
        
        m.SOCDynamics = Constraint(m.T, rule=soc_dynamics_rule)
        
        def soc_limit_rule(model, t):
            """State of charge within battery capacity."""
            return inequality(
                self.technical.battery_min_soc * model.battery_capacity,
                model.soc[t],
                self.technical.battery_max_soc * model.battery_capacity
            )
        
        m.SOCLimit = Constraint(m.T, rule=soc_limit_rule)
        
        def reliability_constraint_rule(model):
            """Total unserved energy limited by reliability target."""
            total_demand = sum(model.demand[t] for t in model.T)
            max_unserved = total_demand * (1 - self.technical.target_reliability)
            return sum(model.unserved[t] for t in model.T) <= max_unserved
        
        m.Reliability = Constraint(rule=reliability_constraint_rule)
        
        # Add dual variables for shadow prices (for explainability)
        m.dual = Suffix(direction=Suffix.IMPORT)
        
        self.model = m
        return m
    
    def get_model_summary(self) -> str:
        """Return a summary of the model structure."""
        if self.model is None:
            return "Model not built yet."
        
        summary = f"""
        HyOptima Model Summary: {self.name}
        ========================================
        
        Sets:
          - T: {len(self.model.T)} time periods
          - SOURCES: {list(self.model.SOURCES)}
        
        Variables:
          - Sizing: solar_capacity, gas_capacity, battery_capacity
          - Operational: solar_gen, gas_gen, grid_import, battery_charge/discharge, soc, unserved
          - Binary: gas_on (generator status)
        
        Objective:
          - Minimize: investment + fuel + grid + penalty costs
        
        Constraints:
          - Power balance (demand = supply)
          - Solar generation limit
          - Gas generation limit
          - Battery charge/discharge limits
          - SOC dynamics and limits
          - Reliability constraint
        
        Parameters:
          - Demand profile: peak = {self.load_profile.peak_demand:.1f} kW
          - Solar profile: CF = {self.solar_profile.capacity_factor:.2%}
          - Economic: solar=${self.economic.solar_capex}/kW, gas=${self.economic.gas_capex}/kW
        """
        return summary
```

### 2.7 hyoptima/solver.py

```python
"""
HyOptima Solver Module

This module handles the execution of the optimization model and
extraction of results.
"""

from typing import Dict, Optional, Any
import numpy as np
from pyomo.environ import SolverFactory, value, ComponentMap
import time

from .model import HyOptimaModel


class HyOptimaSolver:
    """
    Solver for HyOptima optimization model.
    
    Handles solver configuration, execution, and result extraction.
    """
    
    def __init__(
        self,
        solver_name: str = "highs",
        solver_options: Optional[Dict] = None,
        tee: bool = True
    ):
        """
        Initialize the solver.
        
        Args:
            solver_name: Name of the solver (highs, cbc, glpk, gurobi)
            solver_options: Dictionary of solver options
            tee: Whether to print solver output
        """
        self.solver_name = solver_name
        self.solver_options = solver_options or {}
        self.tee = tee
        self.solver = None
        self.results = None
        self.solve_time = 0.0
        
    def solve(self, model: HyOptimaModel) -> Dict[str, Any]:
        """
        Solve the HyOptima model.
        
        Args:
            model: HyOptimaModel instance with built Pyomo model
        
        Returns:
            Dictionary containing optimization results
        """
        if model.model is None:
            model.build_model()
        
        # Create solver
        self.solver = SolverFactory(self.solver_name)
        
        # Set solver options
        for key, val in self.solver_options.items():
            self.solver.options[key] = val
        
        # Solve
        start_time = time.time()
        self.results = self.solver.solve(model.model, tee=self.tee)
        self.solve_time = time.time() - start_time
        
        # Extract results
        return self._extract_results(model)
    
    def _extract_results(self, model: HyOptimaModel) -> Dict[str, Any]:
        """
        Extract optimization results from solved model.
        
        Args:
            model: Solved HyOptimaModel instance
        
        Returns:
            Dictionary with capacity decisions, costs, and dispatch
        """
        m = model.model
        
        # Check termination condition
        termination = self.results.solver.termination_condition.name
        
        results = {
            "status": termination,
            "solve_time": self.solve_time,
            "solver": self.solver_name,
            
            # Capacity decisions
            "solar_capacity_kw": value(m.solar_capacity),
            "gas_capacity_kw": value(m.gas_capacity),
            "battery_capacity_kwh": value(m.battery_capacity),
            
            # Costs
            "total_cost": value(m.TotalCost),
            
            # Time series
            "dispatch": {
                "solar_gen": [value(m.solar_gen[t]) for t in m.T],
                "gas_gen": [value(m.gas_gen[t]) for t in m.T],
                "grid_import": [value(m.grid_import[t]) for t in m.T],
                "battery_charge": [value(m.battery_charge[t]) for t in m.T],
                "battery_discharge": [value(m.battery_discharge[t]) for t in m.T],
                "soc": [value(m.soc[t]) for t in m.T],
                "unserved": [value(m.unserved[t]) for t in m.T],
                "demand": [value(m.demand[t]) for t in m.T],
            },
            
            # Metrics
            "metrics": {},
            
            # Shadow prices (for explainability)
            "shadow_prices": {},
        }
        
        # Calculate metrics
        results["metrics"] = self._calculate_metrics(results, model)
        
        # Extract shadow prices
        results["shadow_prices"] = self._extract_shadow_prices(m)
        
        return results
    
    def _calculate_metrics(
        self,
        results: Dict,
        model: HyOptimaModel
    ) -> Dict[str, float]:
        """Calculate performance metrics from results."""
        dispatch = results["dispatch"]
        
        total_demand = sum(dispatch["demand"])
        total_solar = sum(dispatch["solar_gen"])
        total_gas = sum(dispatch["gas_gen"])
        total_grid = sum(dispatch["grid_import"])
        total_unserved = sum(dispatch["unserved"])
        
        # Renewable fraction
        total_generation = total_solar + total_gas + total_grid
        renewable_fraction = total_solar / total_generation if total_generation > 0 else 0
        
        # Reliability
        reliability = 1 - (total_unserved / total_demand) if total_demand > 0 else 0
        
        # Emissions
        emissions_kg = total_gas * model.technical.emission_factor_gas
        
        # Battery utilization
        battery_capacity = results["battery_capacity_kwh"]
        avg_soc = np.mean(dispatch["soc"])
        battery_utilization = avg_soc / battery_capacity if battery_capacity > 0 else 0
        
        # LCOE approximation
        total_energy = total_demand
        total_cost = results["total_cost"]
        lcoe = total_cost / total_energy if total_energy > 0 else 0
        
        return {
            "total_demand_kwh": total_demand,
            "total_solar_kwh": total_solar,
            "total_gas_kwh": total_gas,
            "total_grid_kwh": total_grid,
            "total_unserved_kwh": total_unserved,
            "renewable_fraction": renewable_fraction,
            "reliability": reliability,
            "emissions_kg": emissions_kg,
            "battery_utilization": battery_utilization,
            "lcoe_approx": lcoe,
        }
    
    def _extract_shadow_prices(self, model) -> Dict[str, Any]:
        """Extract shadow prices (dual values) for explainability."""
        shadow_prices = {}
        
        try:
            # Power balance shadow prices (marginal cost of demand)
            for t in model.T:
                constraint = model.PowerBalance[t]
                if constraint in model.dual:
                    shadow_prices[f"power_balance_{t}"] = model.dual[constraint]
        except Exception:
            pass
        
        return shadow_prices
    
    def print_summary(self, results: Dict) -> str:
        """Print a formatted summary of results."""
        summary = f"""
        ============================================
        HyOptima Optimization Results
        ============================================
        
        Status: {results['status']}
        Solve Time: {results['solve_time']:.2f} seconds
        Solver: {results['solver']}
        
        --- Capacity Decisions ---
        Solar Capacity: {results['solar_capacity_kw']:.1f} kW
        Gas Capacity: {results['gas_capacity_kw']:.1f} kW
        Battery Capacity: {results['battery_capacity_kwh']:.1f} kWh
        
        --- Costs ---
        Total System Cost: ${results['total_cost']:,.2f}
        
        --- Performance Metrics ---
        Total Demand: {results['metrics']['total_demand_kwh']:.1f} kWh
        Solar Generation: {results['metrics']['total_solar_kwh']:.1f} kWh
        Gas Generation: {results['metrics']['total_gas_kwh']:.1f} kWh
        Grid Import: {results['metrics']['total_grid_kwh']:.1f} kWh
        Unserved Energy: {results['metrics']['total_unserved_kwh']:.1f} kWh
        
        Renewable Fraction: {results['metrics']['renewable_fraction']:.1%}
        Reliability: {results['metrics']['reliability']:.2%}
        Emissions: {results['metrics']['emissions_kg']:.1f} kg CO2
        Approximate LCOE: ${results['metrics']['lcoe_approx']:.3f}/kWh
        
        ============================================
        """
        return summary
```

### 2.8 hyoptima/utils.py

```python
"""
Utility functions for HyOptima.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Optional


def plot_dispatch_results(
    results: Dict,
    save_path: Optional[str] = None,
    show: bool = True
) -> plt.Figure:
    """
    Plot the dispatch results from optimization.
    
    Args:
        results: Results dictionary from solver
        save_path: Path to save the figure
        show: Whether to display the figure
    
    Returns:
        matplotlib Figure object
    """
    dispatch = results["dispatch"]
    hours = range(len(dispatch["demand"]))
    
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    
    # Plot 1: Generation Stack
    ax1 = axes[0]
    ax1.stackplot(
        hours,
        dispatch["solar_gen"],
        dispatch["gas_gen"],
        dispatch["grid_import"],
        dispatch["battery_discharge"],
        labels=["Solar", "Gas", "Grid", "Battery Discharge"],
        colors=["gold", "brown", "gray", "green"],
        alpha=0.7
    )
    ax1.plot(hours, dispatch["demand"], "k--", linewidth=2, label="Demand")
    ax1.set_xlabel("Hour")
    ax1.set_ylabel("Power (kW)")
    ax1.set_title("Energy Dispatch Profile")
    ax1.legend(loc="upper right")
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Battery State of Charge
    ax2 = axes[1]
    ax2.fill_between(
        hours,
        dispatch["soc"],
        alpha=0.5,
        color="green",
        label="State of Charge"
    )
    ax2.set_xlabel("Hour")
    ax2.set_ylabel("SOC (kWh)")
    ax2.set_title("Battery State of Charge")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Solar Availability vs Generation
    ax3 = axes[2]
    capacity = results["solar_capacity_kw"]
    solar_available = [capacity * dispatch["solar_gen"][t] / max(dispatch["solar_gen"][t], 0.001) 
                       if dispatch["solar_gen"][t] > 0 else 0 
                       for t in hours]
    ax3.bar(hours, dispatch["solar_gen"], color="gold", alpha=0.7, label="Solar Generation")
    ax3.set_xlabel("Hour")
    ax3.set_ylabel("Power (kW)")
    ax3.set_title("Solar Generation")
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    
    if show:
        plt.show()
    
    return fig


def calculate_lcoe(
    total_cost: float,
    total_energy: float,
    discount_rate: float = 0.10,
    lifetime: int = 25
) -> float:
    """
    Calculate Levelized Cost of Energy.
    
    Args:
        total_cost: Total system cost ($)
        total_energy: Annual energy production (kWh)
        discount_rate: Discount rate
        lifetime: System lifetime (years)
    
    Returns:
        LCOE in $/kWh
    """
    if total_energy == 0:
        return float('inf')
    
    # Simplified LCOE (for single year)
    # Full LCOE would sum over lifetime with degradation
    return total_cost / total_energy


def format_currency(value: float, currency: str = "$") -> str:
    """Format a value as currency."""
    return f"{currency}{value:,.2f}"


def format_percentage(value: float) -> str:
    """Format a value as percentage."""
    return f"{value:.1%}"
```

---

## 3. Notebook Structure (phase1_core_model.ipynb)

The Jupyter notebook should follow this structure:

### Cell 1: Setup and Imports
```python
# NETI-HyOptima Phase 1: Core Model Demonstration
# This notebook demonstrates the HyOptima optimization engine

import sys
sys.path.append('..')

import numpy as np
import matplotlib.pyplot as plt

from hyoptima import (
    HyOptimaModel,
    HyOptimaSolver,
    EconomicParameters,
    TechnicalParameters,
    LoadProfile,
    SolarProfile,
)
from hyoptima.utils import plot_dispatch_results
```

### Cell 2: Generate Synthetic Data
```python
# Generate synthetic load and solar profiles for a Nigerian community
# Example: Bauchi state rural community

load_profile = LoadProfile.generate_synthetic(
    peak_demand=300,  # 300 kW peak demand
    profile_type="mixed",
    hours=24,
    noise_level=0.05
)

solar_profile = SolarProfile.from_capacity_factor(
    capacity_factor=0.20,  # 20% capacity factor (typical for Nigeria)
    hours=24
)

print(f"Load Profile: Peak = {load_profile.peak_demand:.1f} kW, "
      f"Total = {load_profile.total_energy:.1f} kWh")
print(f"Solar Profile: Capacity Factor = {solar_profile.capacity_factor:.1%}")
```

### Cell 3: Configure Parameters
```python
# Economic parameters (Nigeria-specific)
economic_params = EconomicParameters(
    solar_capex=800,      # $/kW
    gas_capex=500,        # $/kW
    battery_capex=300,    # $/kWh
    gas_fuel_cost=0.08,   # $/kWh
    unserved_energy_penalty=1.0,  # High penalty for reliability
)

# Technical parameters
technical_params = TechnicalParameters(
    battery_charge_efficiency=0.95,
    battery_discharge_efficiency=0.95,
    target_reliability=0.99,  # 99% reliability
)

print("Parameters configured for Nigerian context")
```

### Cell 4: Build and Solve Model
```python
# Create and build the optimization model
model = HyOptimaModel(
    load_profile=load_profile,
    solar_profile=solar_profile,
    economic_params=economic_params,
    technical_params=technical_params,
    name="Bauchi_Community_v0"
)

model.build_model()
print(model.get_model_summary())
```

### Cell 5: Run Optimization
```python
# Solve the model
solver = HyOptimaSolver(solver_name="highs", tee=True)
results = solver.solve(model)

# Print results
print(solver.print_summary(results))
```

### Cell 6: Visualize Results
```python
# Plot dispatch results
fig = plot_dispatch_results(results, save_path="../results/figures/phase1_dispatch.png")
```

### Cell 7: Analysis and Insights
```python
# Analyze results
print("\n--- Key Insights ---")
print(f"Optimal Solar: {results['solar_capacity_kw']:.1f} kW")
print(f"Optimal Gas: {results['gas_capacity_kw']:.1f} kW")
print(f"Optimal Battery: {results['battery_capacity_kwh']:.1f} kWh")
print(f"\nRenewable Fraction: {results['metrics']['renewable_fraction']:.1%}")
print(f"Reliability Achieved: {results['metrics']['reliability']:.2%}")
print(f"Approximate LCOE: ${results['metrics']['lcoe_approx']:.3f}/kWh")
```

---

## 4. Testing Strategy

### Unit Tests (tests/test_model.py)

```python
"""
Unit tests for HyOptima model.
"""

import pytest
import numpy as np
from hyoptima import (
    HyOptimaModel,
    HyOptimaSolver,
    LoadProfile,
    SolarProfile,
    EconomicParameters,
    TechnicalParameters,
)


class TestLoadProfile:
    """Tests for LoadProfile class."""
    
    def test_synthetic_generation(self):
        """Test synthetic load profile generation."""
        profile = LoadProfile.generate_synthetic(peak_demand=100)
        assert len(profile.demand) == 24
        assert profile.peak_demand == pytest.approx(100, rel=0.1)
    
    def test_total_energy(self):
        """Test total energy calculation."""
        profile = LoadProfile.generate_synthetic(peak_demand=100)
        assert profile.total_energy > 0


class TestSolarProfile:
    """Tests for SolarProfile class."""
    
    def test_synthetic_generation(self):
        """Test synthetic solar profile generation."""
        profile = SolarProfile.generate_synthetic()
        assert len(profile.availability) == 24
        assert all(0 <= v <= 1 for v in profile.availability)
    
    def test_capacity_factor(self):
        """Test capacity factor calculation."""
        profile = SolarProfile.from_capacity_factor(capacity_factor=0.20)
        assert profile.capacity_factor == pytest.approx(0.20, rel=0.1)


class TestHyOptimaModel:
    """Tests for HyOptimaModel class."""
    
    @pytest.fixture
    def basic_model(self):
        """Create a basic model for testing."""
        load = LoadProfile.generate_synthetic(peak_demand=100)
        solar = SolarProfile.generate_synthetic()
        return HyOptimaModel(load_profile=load, solar_profile=solar)
    
    def test_model_build(self, basic_model):
        """Test model building."""
        m = basic_model.build_model()
        assert m is not None
        assert hasattr(m, 'T')
        assert hasattr(m, 'TotalCost')
    
    def test_model_solve(self, basic_model):
        """Test model solving."""
        basic_model.build_model()
        solver = HyOptimaSolver(solver_name="highs", tee=False)
        results = solver.solve(basic_model)
        
        assert results['status'] in ['optimal', 'feasible']
        assert results['solar_capacity_kw'] >= 0
        assert results['gas_capacity_kw'] >= 0
        assert results['battery_capacity_kwh'] >= 0
```

---

## 5. Documentation Files

### docs/phase1_notes.md

```markdown
# Phase 1 Development Notes

## Objective
Build and validate the core HyOptima optimization engine for hybrid energy systems.

## Current Status
- [x] Define mathematical model
- [x] Implement Pyomo model structure
- [ ] Test with synthetic data
- [ ] Validate results
- [ ] Document findings

## Key Decisions

### Solver Choice
Using HiGHS solver because:
- Open source and free
- Fast for LP/MILP problems
- Works well in Google Colab
- No license required

### Model Scope (v0)
- Single location (microgrid)
- 24-hour horizon
- Solar + Gas + Battery
- No grid connection (island mode)
- No uncertainty (deterministic)

## Next Steps
1. Run optimization with various demand profiles
2. Test sensitivity to cost parameters
3. Add emission constraints
4. Extend to multiple days
```

---

## 6. Execution Checklist

Before proceeding to Phase 2, ensure:

- [ ] Repository structure created
- [ ] requirements.txt installed
- [ ] hyoptima/ package implemented
- [ ] Notebook runs end-to-end
- [ ] Optimization produces valid results
- [ ] Results are interpretable
- [ ] Documentation complete

---

## 7. Switch to Code Mode

To implement this plan, switch to Code mode to create:

1. `requirements.txt`
2. `.gitignore`
3. `hyoptima/__init__.py`
4. `hyoptima/parameters.py`
5. `hyoptima/profiles.py`
6. `hyoptima/model.py`
7. `hyoptima/solver.py`
8. `hyoptima/utils.py`
9. `notebooks/phase1_core_model.ipynb`
10. `tests/test_model.py`
11. `docs/phase1_notes.md`

---

*Document Version: 1.0*
*Created: February 2026*
*Status: Ready for Implementation in Code Mode*
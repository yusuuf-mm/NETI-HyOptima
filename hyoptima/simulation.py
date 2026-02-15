"""
Monte Carlo Simulation Layer for HyOptima

This module provides uncertainty analysis capabilities for the HyOptima
optimization engine. It wraps the deterministic optimizer in a Monte Carlo
framework to compute probability distributions of outcomes.

Key Features:
- Scenario sampling from uncertainty distributions
- Parallel scenario execution
- Statistical analysis (mean, variance, percentiles)
- Risk metrics (CVaR, worst-case scenarios)

Uncertainties Modeled:
- Solar irradiance variability
- Demand forecast error
- Fuel price volatility
- Equipment availability

Author: NETI-HyOptima Team
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Callable, Any
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import logging
from tqdm import tqdm
import json

from hyoptima.model import HyOptimaModel
from hyoptima.parameters import EconomicParameters, TechnicalParameters
from hyoptima.profiles import LoadProfile, SolarProfile
from hyoptima.solver import HyOptimaSolver

logger = logging.getLogger(__name__)


@dataclass
class UncertaintyDistribution:
    """
    Defines an uncertainty distribution for a parameter.
    
    Attributes:
        name: Parameter name
        base_value: Base (expected) value
        distribution_type: 'normal', 'uniform', 'triangular', 'lognormal'
        std_dev: Standard deviation (for normal/lognormal)
        min_value: Minimum value (for uniform/triangular)
        max_value: Maximum value (for uniform/triangular)
        mode: Mode value (for triangular)
    """
    name: str
    base_value: float
    distribution_type: str = "normal"
    std_dev: float = 0.1
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    mode: Optional[float] = None
    
    def sample(self, n_samples: int, seed: Optional[int] = None) -> np.ndarray:
        """
        Generate random samples from the distribution.
        
        Args:
            n_samples: Number of samples to generate
            seed: Random seed for reproducibility
        
        Returns:
            Array of sampled values
        """
        if seed is not None:
            np.random.seed(seed)
        
        if self.distribution_type == "normal":
            samples = np.random.normal(self.base_value, self.std_dev, n_samples)
        
        elif self.distribution_type == "uniform":
            min_v = self.min_value if self.min_value is not None else self.base_value * (1 - self.std_dev)
            max_v = self.max_value if self.max_value is not None else self.base_value * (1 + self.std_dev)
            samples = np.random.uniform(min_v, max_v, n_samples)
        
        elif self.distribution_type == "triangular":
            min_v = self.min_value if self.min_value is not None else self.base_value * 0.8
            max_v = self.max_value if self.max_value is not None else self.base_value * 1.2
            mode_v = self.mode if self.mode is not None else self.base_value
            samples = np.random.triangular(min_v, mode_v, max_v, n_samples)
        
        elif self.distribution_type == "lognormal":
            # Lognormal: mean = exp(mu + sigma^2/2), std = sqrt(exp(2*mu + sigma^2)*(exp(sigma^2)-1))
            # Solve for mu and sigma from base_value and std_dev
            mu = np.log(self.base_value**2 / np.sqrt(self.base_value**2 + self.std_dev**2))
            sigma = np.sqrt(np.log(1 + self.std_dev**2 / self.base_value**2))
            samples = np.random.lognormal(mu, sigma, n_samples)
        
        else:
            raise ValueError(f"Unknown distribution type: {self.distribution_type}")
        
        # Ensure non-negative for physical quantities
        samples = np.maximum(samples, 0)
        
        return samples


@dataclass
class SimulationConfig:
    """
    Configuration for Monte Carlo simulation.
    
    Attributes:
        n_scenarios: Number of scenarios to run
        seed: Random seed for reproducibility
        n_workers: Number of parallel workers (0 = sequential)
        verbose: Print progress information
    """
    n_scenarios: int = 100
    seed: int = 42
    n_workers: int = 0  # 0 = sequential, -1 = all cores
    verbose: bool = True


@dataclass
class SimulationResult:
    """
    Results from a single scenario run.
    
    Attributes:
        scenario_id: Scenario identifier
        solar_capacity: Optimal solar capacity (kW)
        gas_capacity: Optimal gas capacity (kW)
        battery_capacity: Optimal battery capacity (kWh)
        total_cost: Total system cost ($)
        reliability: Achieved reliability (fraction)
        emissions: Total emissions (kg CO2)
        lcoe: Levelized cost of energy ($/kWh)
        unserved_energy: Total unserved energy (kWh)
        parameters: Sampled parameter values
    """
    scenario_id: int
    solar_capacity: float
    gas_capacity: float
    battery_capacity: float
    total_cost: float
    reliability: float
    emissions: float
    lcoe: float
    unserved_energy: float
    parameters: Dict[str, float]


class MonteCarloSimulator:
    """
    Monte Carlo simulation framework for HyOptima.
    
    This class wraps the deterministic optimizer and runs multiple
    scenarios with sampled parameter values to compute probability
    distributions of outcomes.
    
    Example:
        >>> from hyoptima.simulation import MonteCarloSimulator, UncertaintyDistribution
        >>> 
        >>> # Define uncertainties
        >>> uncertainties = [
        ...     UncertaintyDistribution("solar_cf", 0.20, "normal", std_dev=0.03),
        ...     UncertaintyDistribution("fuel_cost", 0.08, "triangular", min_value=0.06, max_value=0.12),
        ... ]
        >>> 
        >>> # Run simulation
        >>> simulator = MonteCarloSimulator(uncertainties)
        >>> results = simulator.run(load_profile, solar_profile, n_scenarios=100)
        >>> 
        >>> # Analyze results
        >>> print(results.summary())
    """
    
    def __init__(
        self,
        uncertainties: List[UncertaintyDistribution],
        config: Optional[SimulationConfig] = None,
    ):
        """
        Initialize Monte Carlo simulator.
        
        Args:
            uncertainties: List of uncertainty distributions
            config: Simulation configuration
        """
        self.uncertainties = {u.name: u for u in uncertainties}
        self.config = config or SimulationConfig()
        self.results: List[SimulationResult] = []
        
        logger.info(f"Initialized MonteCarloSimulator with {len(uncertainties)} uncertainties")
    
    def _sample_parameters(self, n_samples: int) -> List[Dict[str, float]]:
        """
        Sample parameter values from uncertainty distributions.
        
        Args:
            n_samples: Number of samples to generate
        
        Returns:
            List of parameter dictionaries
        """
        # Set random seed
        np.random.seed(self.config.seed)
        
        # Sample each parameter
        samples = {}
        for name, dist in self.uncertainties.items():
            samples[name] = dist.sample(n_samples)
        
        # Convert to list of dictionaries
        param_list = []
        for i in range(n_samples):
            params = {name: values[i] for name, values in samples.items()}
            param_list.append(params)
        
        return param_list
    
    def _run_single_scenario(
        self,
        scenario_id: int,
        params: Dict[str, float],
        load_profile: LoadProfile,
        solar_profile: SolarProfile,
        economic_params: Optional[EconomicParameters] = None,
        technical_params: Optional[TechnicalParameters] = None,
    ) -> Optional[SimulationResult]:
        """
        Run a single optimization scenario.
        
        Args:
            scenario_id: Scenario identifier
            params: Sampled parameter values
            load_profile: Load profile
            solar_profile: Solar profile
            economic_params: Base economic parameters
            technical_params: Base technical parameters
        
        Returns:
            SimulationResult or None if optimization failed
        """
        try:
            # Create modified parameters
            economic = economic_params or EconomicParameters()
            technical = technical_params or TechnicalParameters()
            
            # Apply sampled parameter values
            if "solar_cf" in params:
                # Modify solar profile based on sampled capacity factor
                solar_profile = SolarProfile.from_capacity_factor(
                    capacity_factor=params["solar_cf"]
                )
            
            if "fuel_cost" in params:
                economic = EconomicParameters(
                    solar_capex=economic.solar_capex,
                    gas_capex=economic.gas_capex,
                    battery_capex=economic.battery_capex,
                    gas_fuel_cost=params["fuel_cost"],
                    grid_tariff=economic.grid_tariff,
                    unserved_energy_penalty=economic.unserved_energy_penalty,
                    emission_penalty=economic.emission_penalty,
                    discount_rate=economic.discount_rate,
                )
            
            if "demand_multiplier" in params:
                # Scale demand profile
                load_profile = LoadProfile.from_array(
                    demand=np.array([d * params["demand_multiplier"] for d in load_profile.demand])
                )
            
            if "solar_capex" in params:
                economic = EconomicParameters(
                    solar_capex=params["solar_capex"],
                    gas_capex=economic.gas_capex,
                    battery_capex=economic.battery_capex,
                    gas_fuel_cost=economic.gas_fuel_cost,
                    grid_tariff=economic.grid_tariff,
                    unserved_energy_penalty=economic.unserved_energy_penalty,
                    emission_penalty=economic.emission_penalty,
                    discount_rate=economic.discount_rate,
                )
            
            if "carbon_price" in params:
                economic = EconomicParameters(
                    solar_capex=economic.solar_capex,
                    gas_capex=economic.gas_capex,
                    battery_capex=economic.battery_capex,
                    gas_fuel_cost=economic.gas_fuel_cost,
                    grid_tariff=economic.grid_tariff,
                    unserved_energy_penalty=economic.unserved_energy_penalty,
                    emission_penalty=params["carbon_price"],
                    discount_rate=economic.discount_rate,
                )
            
            # Build and solve model
            model = HyOptimaModel(
                load_profile=load_profile,
                solar_profile=solar_profile,
                economic_params=economic,
                technical_params=technical,
            )
            model.build_model()
            
            solver = HyOptimaSolver(solver_name="highs", tee=False)
            results = solver.solve(model)
            
            # Extract results
            from pyomo.environ import value
            
            total_demand = sum(load_profile.demand)
            total_emissions = sum(
                value(model.model.gas_gen[t]) * technical.emission_factor_gas
                for t in model.model.T
            )
            
            result = SimulationResult(
                scenario_id=scenario_id,
                solar_capacity=results["solar_capacity_kw"],
                gas_capacity=results["gas_capacity_kw"],
                battery_capacity=results["battery_capacity_kwh"],
                total_cost=results["total_cost"],
                reliability=results["metrics"]["reliability"],
                emissions=total_emissions,
                lcoe=results["total_cost"] / total_demand if total_demand > 0 else 0,
                unserved_energy=results["metrics"]["total_unserved_kwh"],
                parameters=params,
            )
            
            return result
            
        except Exception as e:
            logger.warning(f"Scenario {scenario_id} failed: {e}")
            return None
    
    def run(
        self,
        load_profile: LoadProfile,
        solar_profile: SolarProfile,
        economic_params: Optional[EconomicParameters] = None,
        technical_params: Optional[TechnicalParameters] = None,
    ) -> List[SimulationResult]:
        """
        Run Monte Carlo simulation.
        
        Args:
            load_profile: Base load profile
            solar_profile: Base solar profile
            economic_params: Base economic parameters
            technical_params: Base technical parameters
        
        Returns:
            List of SimulationResult objects
        """
        logger.info(f"Running {self.config.n_scenarios} scenarios...")
        
        # Sample parameters
        param_samples = self._sample_parameters(self.config.n_scenarios)
        
        # Run scenarios
        self.results = []
        
        if self.config.n_workers == 0:
            # Sequential execution
            iterator = range(self.config.n_scenarios)
            if self.config.verbose:
                iterator = tqdm(iterator, desc="Running scenarios")
            
            for i in iterator:
                result = self._run_single_scenario(
                    i,
                    param_samples[i],
                    load_profile,
                    solar_profile,
                    economic_params,
                    technical_params,
                )
                if result is not None:
                    self.results.append(result)
        
        else:
            # Parallel execution
            n_workers = self.config.n_workers
            if n_workers < 0:
                import os
                n_workers = os.cpu_count()
            
            with ThreadPoolExecutor(max_workers=n_workers) as executor:
                futures = []
                for i in range(self.config.n_scenarios):
                    future = executor.submit(
                        self._run_single_scenario,
                        i,
                        param_samples[i],
                        load_profile,
                        solar_profile,
                        economic_params,
                        technical_params,
                    )
                    futures.append(future)
                
                if self.config.verbose:
                    futures = tqdm(futures, desc="Running scenarios")
                
                for future in futures:
                    result = future.result()
                    if result is not None:
                        self.results.append(result)
        
        logger.info(f"Completed {len(self.results)} scenarios successfully")
        
        return self.results
    
    def summary(self) -> Dict[str, Any]:
        """
        Generate summary statistics from simulation results.
        
        Returns:
            Dictionary with summary statistics
        """
        if not self.results:
            return {"error": "No results available"}
        
        # Extract arrays
        solar = np.array([r.solar_capacity for r in self.results])
        gas = np.array([r.gas_capacity for r in self.results])
        battery = np.array([r.battery_capacity for r in self.results])
        cost = np.array([r.total_cost for r in self.results])
        reliability = np.array([r.reliability for r in self.results])
        emissions = np.array([r.emissions for r in self.results])
        lcoe = np.array([r.lcoe for r in self.results])
        
        def stats(arr):
            return {
                "mean": float(np.mean(arr)),
                "std": float(np.std(arr)),
                "min": float(np.min(arr)),
                "max": float(np.max(arr)),
                "p5": float(np.percentile(arr, 5)),
                "p25": float(np.percentile(arr, 25)),
                "p50": float(np.percentile(arr, 50)),
                "p75": float(np.percentile(arr, 75)),
                "p95": float(np.percentile(arr, 95)),
            }
        
        summary = {
            "n_scenarios": len(self.results),
            "solar_capacity_kw": stats(solar),
            "gas_capacity_kw": stats(gas),
            "battery_capacity_kwh": stats(battery),
            "total_cost": stats(cost),
            "reliability": stats(reliability),
            "emissions_kg": stats(emissions),
            "lcoe": stats(lcoe),
        }
        
        return summary
    
    def technology_adoption(self, threshold_kw: float = 1.0) -> Dict[str, Any]:
        """
        Compute technology selection frequency across scenarios.
        
        This metric answers: "How often is each technology chosen?"
        
        This is more valuable than average LCOE for planning because:
        - Planners need to know what is consistently chosen
        - Shows robustness of technology selection
        - Identifies technologies that are marginal vs essential
        
        Args:
            threshold_kw: Minimum capacity to count as "selected" (default 1 kW)
        
        Returns:
            Dictionary with adoption probabilities and statistics
        """
        if not self.results:
            return {"error": "No results available"}
        
        n = len(self.results)
        
        # Count selections
        solar_selected = sum(1 for r in self.results if r.solar_capacity >= threshold_kw)
        gas_selected = sum(1 for r in self.results if r.gas_capacity >= threshold_kw)
        battery_selected = sum(1 for r in self.results if r.battery_capacity >= threshold_kw)
        
        # Compute adoption probabilities
        solar_prob = solar_selected / n
        gas_prob = gas_selected / n
        battery_prob = battery_selected / n
        
        # Conditional statistics (when technology IS selected)
        solar_when_selected = [r.solar_capacity for r in self.results if r.solar_capacity >= threshold_kw]
        gas_when_selected = [r.gas_capacity for r in self.results if r.gas_capacity >= threshold_kw]
        battery_when_selected = [r.battery_capacity for r in self.results if r.battery_capacity >= threshold_kw]
        
        def safe_stats(arr, name):
            if not arr:
                return {"mean": 0, "std": 0, "min": 0, "max": 0, "count": 0}
            return {
                "mean": float(np.mean(arr)),
                "std": float(np.std(arr)),
                "min": float(np.min(arr)),
                "max": float(np.max(arr)),
                "count": len(arr),
            }
        
        # Technology combinations
        solar_only = sum(1 for r in self.results 
                        if r.solar_capacity >= threshold_kw 
                        and r.gas_capacity < threshold_kw)
        gas_only = sum(1 for r in self.results 
                      if r.gas_capacity >= threshold_kw 
                      and r.solar_capacity < threshold_kw)
        hybrid = sum(1 for r in self.results 
                    if r.solar_capacity >= threshold_kw 
                    and r.gas_capacity >= threshold_kw)
        
        return {
            "adoption_probability": {
                "solar": {
                    "probability": solar_prob,
                    "selected_count": solar_selected,
                    "total_scenarios": n,
                },
                "gas": {
                    "probability": gas_prob,
                    "selected_count": gas_selected,
                    "total_scenarios": n,
                },
                "battery": {
                    "probability": battery_prob,
                    "selected_count": battery_selected,
                    "total_scenarios": n,
                },
            },
            "capacity_when_selected": {
                "solar_kw": safe_stats(solar_when_selected, "solar"),
                "gas_kw": safe_stats(gas_when_selected, "gas"),
                "battery_kwh": safe_stats(battery_when_selected, "battery"),
            },
            "system_configurations": {
                "solar_only": {
                    "count": solar_only,
                    "probability": solar_only / n,
                },
                "gas_only": {
                    "count": gas_only,
                    "probability": gas_only / n,
                },
                "hybrid_solar_gas": {
                    "count": hybrid,
                    "probability": hybrid / n,
                },
            },
            "threshold_kw": threshold_kw,
        }
    
    def risk_metrics(self) -> Dict[str, float]:
        """
        Compute risk metrics from simulation results.
        
        Returns:
            Dictionary with risk metrics
        """
        if not self.results:
            return {"error": "No results available"}
        
        costs = np.array([r.total_cost for r in self.results])
        
        # Value at Risk (VaR) - 95% confidence
        var_95 = np.percentile(costs, 95)
        
        # Conditional Value at Risk (CVaR) - Expected Shortfall
        cvar_95 = np.mean(costs[costs >= var_95])
        
        # Worst case
        worst_case = np.max(costs)
        
        # Best case
        best_case = np.min(costs)
        
        # Range
        cost_range = worst_case - best_case
        
        # Coefficient of variation
        cv = np.std(costs) / np.mean(costs) if np.mean(costs) > 0 else 0
        
        return {
            "var_95": float(var_95),
            "cvar_95": float(cvar_95),
            "worst_case": float(worst_case),
            "best_case": float(best_case),
            "range": float(cost_range),
            "coefficient_of_variation": float(cv),
        }
    
    def to_dataframe(self) -> pd.DataFrame:
        """
        Convert results to pandas DataFrame.
        
        Returns:
            DataFrame with all scenario results
        """
        if not self.results:
            return pd.DataFrame()
        
        data = []
        for r in self.results:
            row = {
                "scenario_id": r.scenario_id,
                "solar_capacity_kw": r.solar_capacity,
                "gas_capacity_kw": r.gas_capacity,
                "battery_capacity_kwh": r.battery_capacity,
                "total_cost": r.total_cost,
                "reliability": r.reliability,
                "emissions_kg": r.emissions,
                "lcoe": r.lcoe,
                "unserved_energy_kwh": r.unserved_energy,
            }
            # Add sampled parameters
            row.update(r.parameters)
            data.append(row)
        
        return pd.DataFrame(data)
    
    def save_results(self, filepath: str) -> None:
        """
        Save results to JSON file.
        
        Args:
            filepath: Path to save file
        """
        data = {
            "config": {
                "n_scenarios": self.config.n_scenarios,
                "seed": self.config.seed,
            },
            "summary": self.summary(),
            "risk_metrics": self.risk_metrics(),
            "technology_adoption": self.technology_adoption(),
            "scenarios": [
                {
                    "scenario_id": r.scenario_id,
                    "solar_capacity_kw": r.solar_capacity,
                    "gas_capacity_kw": r.gas_capacity,
                    "battery_capacity_kwh": r.battery_capacity,
                    "total_cost": r.total_cost,
                    "reliability": r.reliability,
                    "emissions_kg": r.emissions,
                    "lcoe": r.lcoe,
                    "parameters": r.parameters,
                }
                for r in self.results
            ],
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Results saved to {filepath}")


def create_default_uncertainties() -> List[UncertaintyDistribution]:
    """
    Create default uncertainty distributions for Nigerian energy system.
    
    Prioritized by impact on planning decisions:
    1. Fuel cost volatility - MAJOR (dominant uncertainty in Nigeria)
    2. Carbon price - MAJOR (policy transition risk)
    3. Demand forecast - MODERATE
    4. Solar variability - MINOR (but useful for robustness)
    
    Returns:
        List of UncertaintyDistribution objects
    """
    return [
        # FUEL COST: Primary uncertainty in Nigeria
        # Gas prices highly volatile, range $0.04-$0.15/kWh
        # This is the dominant risk factor for energy planning
        UncertaintyDistribution(
            name="fuel_cost",
            base_value=0.08,
            distribution_type="triangular",
            min_value=0.04,   # Low gas price scenario
            max_value=0.15,   # High gas price shock
            mode=0.08,        # Most likely
        ),
        
        # CARBON PRICE: Policy transition risk
        # Nigeria ETP implies future carbon pricing
        # Range: $0 (no policy) to $100/ton (aggressive)
        UncertaintyDistribution(
            name="carbon_price",
            base_value=50.0,   # $50/ton reference
            distribution_type="triangular",
            min_value=0.0,     # No carbon price
            max_value=100.0,   # High carbon price
            mode=30.0,         # Most likely moderate price
        ),
        
        # DEMAND MULTIPLIER: Forecast uncertainty
        # ±15% reflects demand growth uncertainty
        UncertaintyDistribution(
            name="demand_multiplier",
            base_value=1.0,
            distribution_type="normal",
            std_dev=0.15,
        ),
        
        # SOLAR CAPACITY FACTOR: Weather variability
        # 20% ± 20% (less critical than fuel economics)
        UncertaintyDistribution(
            name="solar_cf",
            base_value=0.20,
            distribution_type="normal",
            std_dev=0.04,
        ),
    ]


def create_conservative_uncertainties() -> List[UncertaintyDistribution]:
    """
    Create conservative uncertainty distributions for risk-averse planning.
    
    Wider ranges for stress testing.
    
    Returns:
        List of UncertaintyDistribution objects
    """
    return [
        # Fuel cost with extreme scenarios
        UncertaintyDistribution(
            name="fuel_cost",
            base_value=0.10,
            distribution_type="uniform",
            min_value=0.02,
            max_value=0.20,
        ),
        
        # Carbon price with policy uncertainty
        UncertaintyDistribution(
            name="carbon_price",
            base_value=50.0,
            distribution_type="uniform",
            min_value=0.0,
            max_value=150.0,
        ),
        
        # Demand with growth uncertainty
        UncertaintyDistribution(
            name="demand_multiplier",
            base_value=1.0,
            distribution_type="triangular",
            min_value=0.7,
            max_value=1.5,
            mode=1.0,
        ),
        
        # Solar with inter-annual variability
        UncertaintyDistribution(
            name="solar_cf",
            base_value=0.18,
            distribution_type="uniform",
            min_value=0.12,
            max_value=0.25,
        ),
    ]

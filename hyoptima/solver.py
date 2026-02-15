"""
HyOptima Solver Module

This module handles the execution of the optimization model and
extraction of results. It provides a clean interface for running
the HyOptima optimization and obtaining actionable insights.

The solver supports multiple optimization backends:
- HiGHS (default, open-source, fast)
- CBC (open-source, reliable)
- GLPK (open-source)
- Gurobi (commercial, fastest for large problems)

Author: NETI-HyOptima Team
"""

from typing import Dict, Optional, Any, List
import numpy as np
from pyomo.environ import value, ComponentMap
from pyomo.opt import SolverFactory, TerminationCondition
import time
import logging
import json

from .model import HyOptimaModel

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class HyOptimaSolver:
    """
    Solver for HyOptima optimization model.
    
    This class handles solver configuration, execution, and result extraction.
    It provides methods to run optimization and analyze results.
    
    Attributes:
        solver_name: Name of the optimization solver
        solver_options: Dictionary of solver-specific options
        tee: Whether to print solver output
        solver: Pyomo SolverFactory instance
        results: Raw solver results
        solve_time: Time taken to solve (seconds)
    
    Example:
        >>> solver = HyOptimaSolver(solver_name="highs")
        >>> results = solver.solve(model)
        >>> print(solver.print_summary(results))
    """
    
    # Supported solvers and their characteristics
    SOLVER_INFO = {
        "highs": {"type": "MILP", "license": "MIT", "speed": "fast"},
        "cbc": {"type": "MILP", "license": "EPL", "speed": "medium"},
        "glpk": {"type": "MILP", "license": "GPL", "speed": "medium"},
        "gurobi": {"type": "MILP", "license": "commercial", "speed": "fastest"},
        "cplex": {"type": "MILP", "license": "commercial", "speed": "fastest"},
    }
    
    def __init__(
        self,
        solver_name: str = "highs",
        solver_options: Optional[Dict] = None,
        tee: bool = True
    ):
        """
        Initialize the solver.
        
        Args:
            solver_name: Name of the solver ('highs', 'cbc', 'glpk', 'gurobi')
            solver_options: Dictionary of solver-specific options
            tee: Whether to print solver output to console
        
        Raises:
            ValueError: If solver is not supported
        """
        self.solver_name = solver_name.lower()
        self.solver_options = solver_options or {}
        self.tee = tee
        self.solver = None
        self.results = None
        self.solve_time = 0.0
        
        # Validate solver
        if self.solver_name not in self.SOLVER_INFO:
            logger.warning(
                f"Solver '{solver_name}' not in tested list. "
                f"Tested solvers: {list(self.SOLVER_INFO.keys())}"
            )
        
        logger.info(f"Initialized HyOptimaSolver with {self.solver_name}")
    
    def solve(self, model: HyOptimaModel) -> Dict[str, Any]:
        """
        Solve the HyOptima model.
        
        This method:
        1. Builds the model if not already built
        2. Creates and configures the solver
        3. Runs the optimization
        4. Extracts and returns results
        
        Args:
            model: HyOptimaModel instance (may be unbuilt)
        
        Returns:
            Dictionary containing:
                - status: Optimization status
                - solve_time: Time to solve (seconds)
                - solver: Solver name used
                - solar_capacity_kw: Optimal solar capacity
                - gas_capacity_kw: Optimal gas capacity
                - battery_capacity_kwh: Optimal battery capacity
                - total_cost: Total system cost
                - dispatch: Dictionary of time-series results
                - metrics: Performance metrics
                - shadow_prices: Dual values for constraints
        
        Raises:
            RuntimeError: If optimization fails
        """
        # Build model if needed
        if model.model is None:
            logger.info("Building model...")
            model.build_model()
        
        # Validate inputs
        warnings = model.validate_inputs()
        if warnings:
            for warning in warnings:
                logger.warning(f"Input validation: {warning}")
        
        # Create solver
        logger.info(f"Creating {self.solver_name} solver...")
        try:
            # For HiGHS, use the APPSI interface which is the modern Pyomo interface
            # The legacy interface has compatibility issues with newer Pyomo versions
            if self.solver_name == 'highs':
                from pyomo.contrib.solver.solvers.highs import Highs
                self.solver = Highs()
            else:
                self.solver = SolverFactory(self.solver_name)
        except Exception as e:
            raise RuntimeError(f"Failed to create solver '{self.solver_name}': {e}")
        
        # Set solver options
        for key, val in self.solver_options.items():
            self.solver.options[key] = val
        
        # Solve
        logger.info("Starting optimization...")
        start_time = time.time()
        
        try:
            self.results = self.solver.solve(model.model, tee=self.tee)
        except Exception as e:
            raise RuntimeError(f"Optimization failed: {e}")
        
        self.solve_time = time.time() - start_time
        logger.info(f"Optimization completed in {self.solve_time:.2f} seconds")
        
        # Extract results
        return self._extract_results(model)
    
    def _extract_results(self, model: HyOptimaModel) -> Dict[str, Any]:
        """
        Extract optimization results from solved model.
        
        Args:
            model: Solved HyOptimaModel instance
        
        Returns:
            Dictionary with capacity decisions, costs, dispatch, and metrics
        """
        m = model.model
        
        # Check termination condition - handle both legacy and APPSI interfaces
        if self.results is None:
            status = "no_results"
        elif hasattr(self.results, 'solver'):
            # Legacy interface
            status = str(self.results.solver.termination_condition)
        elif hasattr(self.results, 'termination_condition'):
            # APPSI interface
            status = str(self.results.termination_condition)
        elif hasattr(self.results, 'solution_status'):
            # APPSI alternative
            status = str(self.results.solution_status)
        else:
            # Try to get status from the result object itself
            try:
                status = str(getattr(self.results, 'status', 'optimal'))
            except:
                status = "optimal"  # Assume optimal if we got here
        
        logger.info(f"Optimization status: {status}")
        
        # Extract capacity decisions
        try:
            solar_capacity = value(m.solar_capacity)
            gas_capacity = value(m.gas_capacity)
            battery_capacity = value(m.battery_capacity)
            total_cost = value(m.TotalCost)
        except Exception as e:
            logger.error(f"Failed to extract variable values: {e}")
            raise RuntimeError("Could not extract optimization results")
        
        # Build results dictionary
        results = {
            "status": status,
            "solve_time": self.solve_time,
            "solver": self.solver_name,
            
            # Capacity decisions
            "solar_capacity_kw": solar_capacity,
            "gas_capacity_kw": gas_capacity,
            "battery_capacity_kwh": battery_capacity,
            
            # Costs
            "total_cost": total_cost,
            
            # Time series data
            "dispatch": self._extract_dispatch(m),
            
            # Performance metrics
            "metrics": {},
            
            # Shadow prices for explainability
            "shadow_prices": {},
            
            # Model information
            "model_name": model.name,
            "time_periods": len(m.T),
        }
        
        # Calculate performance metrics
        results["metrics"] = self._calculate_metrics(results, model)
        
        # Extract shadow prices (dual values)
        results["shadow_prices"] = self._extract_shadow_prices(m)
        
        # Add input summary
        results["inputs"] = {
            "load_profile": model.load_profile.get_summary(),
            "solar_profile": model.solar_profile.get_summary(),
            "economic_params": model.economic.to_dict(),
        }
        
        return results
    
    def _extract_dispatch(self, model) -> Dict[str, List[float]]:
        """Extract time-series dispatch data."""
        dispatch = {
            "solar_gen": [value(model.solar_gen[t]) for t in model.T],
            "gas_gen": [value(model.gas_gen[t]) for t in model.T],
            "grid_import": [value(model.grid_import[t]) for t in model.T],
            "battery_charge": [value(model.battery_charge[t]) for t in model.T],
            "battery_discharge": [value(model.battery_discharge[t]) for t in model.T],
            "soc": [value(model.soc[t]) for t in model.T],
            "unserved": [value(model.unserved[t]) for t in model.T],
            "demand": [value(model.demand[t]) for t in model.T],
            "solar_availability": [value(model.solar_availability[t]) for t in model.T],
            "gas_on": [value(model.gas_on[t]) for t in model.T],
        }
        return dispatch
    
    def _calculate_metrics(
        self,
        results: Dict,
        model: HyOptimaModel
    ) -> Dict[str, float]:
        """
        Calculate performance metrics from optimization results.
        
        Metrics include:
        - Energy balance (generation by source)
        - Renewable fraction
        - Reliability
        - Emissions
        - Battery utilization
        - Approximate LCOE
        """
        dispatch = results["dispatch"]
        
        # Energy totals
        total_demand = sum(dispatch["demand"])
        total_solar = sum(dispatch["solar_gen"])
        total_gas = sum(dispatch["gas_gen"])
        total_grid = sum(dispatch["grid_import"])
        total_charge = sum(dispatch["battery_charge"])
        total_discharge = sum(dispatch["battery_discharge"])
        total_unserved = sum(dispatch["unserved"])
        
        # Total generation (excluding battery which is storage)
        total_generation = total_solar + total_gas + total_grid
        
        # Renewable fraction
        renewable_fraction = total_solar / total_generation if total_generation > 0 else 0
        
        # Reliability (1 - unserved fraction)
        reliability = 1 - (total_unserved / total_demand) if total_demand > 0 else 0
        
        # Emissions (kg CO2)
        emissions_kg = total_gas * model.technical.emission_factor_gas
        emissions_tons = emissions_kg / 1000
        
        # Battery utilization
        battery_capacity = results["battery_capacity_kwh"]
        if battery_capacity > 0:
            avg_soc = np.mean(dispatch["soc"])
            battery_utilization = avg_soc / battery_capacity
            cycles = total_discharge / battery_capacity if battery_capacity > 0 else 0
        else:
            battery_utilization = 0
            cycles = 0
        
        # Gas generator utilization
        gas_capacity = results["gas_capacity_kw"]
        if gas_capacity > 0:
            gas_capacity_factor = total_gas / (gas_capacity * len(dispatch["gas_gen"]))
            gas_operating_hours = sum(dispatch["gas_on"])
        else:
            gas_capacity_factor = 0
            gas_operating_hours = 0
        
        # Solar utilization
        solar_capacity = results["solar_capacity_kw"]
        if solar_capacity > 0:
            solar_capacity_factor = total_solar / (solar_capacity * len(dispatch["solar_gen"]))
        else:
            solar_capacity_factor = 0
        
        # Approximate LCOE (simplified)
        # LCOE = Total Cost / Total Energy Served
        total_energy_served = total_demand - total_unserved
        lcoe = results["total_cost"] / total_energy_served if total_energy_served > 0 else float('inf')
        
        # Cost breakdown
        investment_cost = (
            solar_capacity * model.economic.solar_capex +
            gas_capacity * model.economic.gas_capex +
            battery_capacity * model.economic.battery_capex
        )
        fuel_cost = total_gas * model.economic.gas_fuel_cost
        grid_cost = total_grid * model.economic.grid_tariff
        penalty_cost = total_unserved * model.economic.unserved_energy_penalty
        
        return {
            # Energy metrics
            "total_demand_kwh": total_demand,
            "total_solar_kwh": total_solar,
            "total_gas_kwh": total_gas,
            "total_grid_kwh": total_grid,
            "total_unserved_kwh": total_unserved,
            "total_energy_served_kwh": total_energy_served,
            
            # Performance metrics
            "renewable_fraction": renewable_fraction,
            "reliability": reliability,
            "solar_capacity_factor": solar_capacity_factor,
            "gas_capacity_factor": gas_capacity_factor,
            
            # Emissions
            "emissions_kg": emissions_kg,
            "emissions_tons": emissions_tons,
            
            # Battery metrics
            "battery_utilization": battery_utilization,
            "battery_cycles": cycles,
            
            # Generator metrics
            "gas_operating_hours": gas_operating_hours,
            
            # Economic metrics
            "lcoe_approx": lcoe,
            "investment_cost": investment_cost,
            "fuel_cost": fuel_cost,
            "grid_cost": grid_cost,
            "penalty_cost": penalty_cost,
        }
    
    def _extract_shadow_prices(self, model) -> Dict[str, Any]:
        """
        Extract shadow prices (dual values) for explainability.
        
        Shadow prices indicate the marginal value of relaxing constraints.
        For power balance, this represents the marginal cost of additional demand.
        """
        shadow_prices = {}
        
        try:
            # Power balance shadow prices (marginal cost of demand)
            power_balance_prices = {}
            for t in model.T:
                constraint = model.PowerBalance[t]
                if constraint in model.dual:
                    power_balance_prices[t] = model.dual[constraint]
            
            if power_balance_prices:
                shadow_prices["power_balance"] = power_balance_prices
                shadow_prices["avg_marginal_cost"] = np.mean(list(power_balance_prices.values()))
        
        except Exception as e:
            logger.debug(f"Could not extract shadow prices: {e}")
        
        return shadow_prices
    
    def print_summary(self, results: Dict) -> str:
        """
        Print a formatted summary of optimization results.
        
        Args:
            results: Results dictionary from solve()
        
        Returns:
            Formatted string with results summary
        """
        metrics = results.get("metrics", {})
        
        summary = f"""
        {'='*65}
        HyOptima Optimization Results
        {'='*65}
        
        STATUS
        ------
          Termination: {results['status']}
          Solve Time: {results['solve_time']:.2f} seconds
          Solver: {results['solver']}
        
        OPTIMAL CAPACITY DECISIONS
        --------------------------
          Solar Capacity:     {results['solar_capacity_kw']:>10.1f} kW
          Gas Capacity:       {results['gas_capacity_kw']:>10.1f} kW
          Battery Capacity:   {results['battery_capacity_kwh']:>10.1f} kWh
        
        COSTS
        -----
          Total System Cost:  ${results['total_cost']:>10,.2f}
          Investment Cost:    ${metrics.get('investment_cost', 0):>10,.2f}
          Fuel Cost:          ${metrics.get('fuel_cost', 0):>10,.2f}
          Grid Cost:          ${metrics.get('grid_cost', 0):>10,.2f}
          Penalty Cost:       ${metrics.get('penalty_cost', 0):>10,.2f}
        
        ENERGY BALANCE
        --------------
          Total Demand:       {metrics.get('total_demand_kwh', 0):>10.1f} kWh
          Solar Generation:   {metrics.get('total_solar_kwh', 0):>10.1f} kWh
          Gas Generation:     {metrics.get('total_gas_kwh', 0):>10.1f} kWh
          Grid Import:        {metrics.get('total_grid_kwh', 0):>10.1f} kWh
          Unserved Energy:    {metrics.get('total_unserved_kwh', 0):>10.1f} kWh
        
        PERFORMANCE METRICS
        -------------------
          Renewable Fraction: {metrics.get('renewable_fraction', 0):>10.1%}
          Reliability:        {metrics.get('reliability', 0):>10.2%}
          Approximate LCOE:   ${metrics.get('lcoe_approx', 0):>10.3f}/kWh
        
        EMISSIONS
        ---------
          CO2 Emissions:      {metrics.get('emissions_kg', 0):>10.1f} kg
                              {metrics.get('emissions_tons', 0):>10.3f} tons
        
        BATTERY PERFORMANCE
        -------------------
          Utilization:        {metrics.get('battery_utilization', 0):>10.1%}
          Equivalent Cycles:  {metrics.get('battery_cycles', 0):>10.1f}
        
        {'='*65}
        """
        return summary
    
    def get_explanation(self, results: Dict) -> str:
        """
        Generate an explanation of the optimization results.
        
        This provides insights into why certain decisions were made,
        useful for non-technical stakeholders.
        
        Args:
            results: Results dictionary from solve()
        
        Returns:
            Human-readable explanation string
        """
        metrics = results.get("metrics", {})
        shadow_prices = results.get("shadow_prices", {})
        
        explanations = []
        
        # System composition explanation
        solar = results['solar_capacity_kw']
        gas = results['gas_capacity_kw']
        battery = results['battery_capacity_kwh']
        
        if solar > 0 and gas > 0 and battery > 0:
            explanations.append(
                "The optimal system is a hybrid configuration combining solar PV, "
                "gas generation, and battery storage. This balances the low operating "
                "cost of solar with the reliability of gas and the flexibility of storage."
            )
        elif solar > 0 and battery > 0:
            explanations.append(
                "The system relies primarily on solar with battery storage for reliability. "
                "This indicates favorable solar resource and/or high fuel costs."
            )
        elif gas > 0:
            explanations.append(
                "The system relies on gas generation. This may indicate low fuel costs "
                "or high reliability requirements that favor dispatchable generation."
            )
        
        # Renewable fraction explanation
        renewable_frac = metrics.get('renewable_fraction', 0)
        if renewable_frac > 0.5:
            explanations.append(
                f"With {renewable_frac:.0%} renewable generation, the system significantly "
                "contributes to Nigeria's clean energy transition goals."
            )
        elif renewable_frac > 0.3:
            explanations.append(
                f"The {renewable_frac:.0%} renewable fraction represents a meaningful "
                "contribution to emissions reduction while maintaining reliability."
            )
        
        # Reliability explanation
        reliability = metrics.get('reliability', 0)
        if reliability > 0.99:
            explanations.append(
                f"System reliability of {reliability:.1%} exceeds typical grid reliability "
                "in Nigeria, demonstrating the value of hybrid systems."
            )
        
        # Battery explanation
        if battery > 0:
            cycles = metrics.get('battery_cycles', 0)
            explanations.append(
                f"The battery system ({battery:.0f} kWh) provides {cycles:.1f} equivalent "
                "cycles of storage, enabling solar energy to be used during evening peak demand."
            )
        
        # Cost explanation
        lcoe = metrics.get('lcoe_approx', 0)
        if lcoe > 0:
            explanations.append(
                f"The approximate LCOE of ${lcoe:.3f}/kWh represents the levelized cost "
                "of electricity from this hybrid system."
            )
        
        # Shadow price insights
        if 'avg_marginal_cost' in shadow_prices:
            mc = shadow_prices['avg_marginal_cost']
            explanations.append(
                f"The average marginal cost of electricity is ${mc:.3f}/kWh, "
                "indicating the cost of serving additional demand."
            )
        
        return "\n\n".join(explanations)
    
    def save_results(self, results: Dict, filepath: str) -> None:
        """
        Save results to a JSON file.
        
        Args:
            results: Results dictionary
            filepath: Path to save file
        """
        # Convert numpy arrays to lists for JSON serialization
        serializable_results = self._make_serializable(results)
        
        with open(filepath, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        logger.info(f"Results saved to {filepath}")
    
    def _make_serializable(self, obj: Any) -> Any:
        """Convert object to JSON-serializable format."""
        if isinstance(obj, dict):
            return {k: self._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_serializable(v) for v in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        else:
            return obj


def run_optimization(
    load_profile,
    solar_profile,
    economic_params=None,
    technical_params=None,
    policy_params=None,
    solver_name="highs",
    **kwargs
) -> Dict:
    """
    Convenience function to run optimization in one call.
    
    Args:
        load_profile: LoadProfile instance
        solar_profile: SolarProfile instance
        economic_params: EconomicParameters instance (optional)
        technical_params: TechnicalParameters instance (optional)
        policy_params: PolicyParameters instance (optional)
        solver_name: Name of solver to use
        **kwargs: Additional solver options
    
    Returns:
        Results dictionary
    
    Example:
        >>> load = LoadProfile.generate_synthetic(peak_demand=300)
        >>> solar = SolarProfile.from_capacity_factor(0.20)
        >>> results = run_optimization(load, solar)
    """
    model = HyOptimaModel(
        load_profile=load_profile,
        solar_profile=solar_profile,
        economic_params=economic_params,
        technical_params=technical_params,
        policy_params=policy_params,
    )
    
    solver = HyOptimaSolver(solver_name=solver_name, **kwargs)
    return solver.solve(model)

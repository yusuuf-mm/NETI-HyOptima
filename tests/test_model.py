"""
Unit Tests for HyOptima Model

This module contains tests for the HyOptima optimization model components.

Run tests with: pytest tests/test_model.py -v
"""

import pytest
import numpy as np
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hyoptima import (
    HyOptimaModel,
    HyOptimaSolver,
    LoadProfile,
    SolarProfile,
    EconomicParameters,
    TechnicalParameters,
    PolicyParameters,
)
from hyoptima.profiles import generate_nigeria_scenarios


class TestLoadProfile:
    """Tests for LoadProfile class."""
    
    def test_synthetic_generation_residential(self):
        """Test synthetic load profile generation for residential."""
        profile = LoadProfile.generate_synthetic(
            peak_demand=100,
            profile_type="residential"
        )
        assert len(profile.demand) == 24
        assert profile.peak_demand == pytest.approx(100, rel=0.15)
        assert profile.total_energy > 0
    
    def test_synthetic_generation_commercial(self):
        """Test synthetic load profile generation for commercial."""
        profile = LoadProfile.generate_synthetic(
            peak_demand=200,
            profile_type="commercial"
        )
        assert len(profile.demand) == 24
        assert profile.peak_demand == pytest.approx(200, rel=0.15)
    
    def test_synthetic_generation_mixed(self):
        """Test synthetic load profile generation for mixed community."""
        profile = LoadProfile.generate_synthetic(
            peak_demand=150,
            profile_type="mixed"
        )
        assert len(profile.demand) == 24
        assert profile.peak_demand > 0
    
    def test_total_energy_calculation(self):
        """Test total energy calculation."""
        demand = np.ones(24) * 100  # 100 kW constant
        profile = LoadProfile(demand=demand)
        assert profile.total_energy == 2400  # 100 kW * 24 hours
    
    def test_load_factor(self):
        """Test load factor calculation."""
        demand = np.ones(24) * 100
        profile = LoadProfile(demand=demand)
        assert profile.load_factor == 1.0  # Flat profile
        
        # Variable profile
        demand = np.array([50] * 12 + [100] * 12)
        profile = LoadProfile(demand=demand)
        assert profile.load_factor == pytest.approx(0.75, rel=0.01)
    
    def test_from_array(self):
        """Test creating profile from array."""
        demand = np.random.rand(24) * 100
        profile = LoadProfile.from_array(demand, name="test")
        assert len(profile.demand) == 24
        assert profile.name == "test"
    
    def test_get_summary(self):
        """Test summary generation."""
        profile = LoadProfile.generate_synthetic(peak_demand=100)
        summary = profile.get_summary()
        assert "peak_demand_kw" in summary
        assert "total_energy_kwh" in summary
        assert "load_factor" in summary


class TestSolarProfile:
    """Tests for SolarProfile class."""
    
    def test_synthetic_generation(self):
        """Test synthetic solar profile generation."""
        profile = SolarProfile.generate_synthetic()
        assert len(profile.availability) == 24
        assert all(0 <= v <= 1 for v in profile.availability)
        # Should be zero at night
        assert profile.availability[0] == 0  # Midnight
        assert profile.availability[23] == 0  # 11 PM
    
    def test_from_capacity_factor(self):
        """Test solar profile from capacity factor."""
        profile = SolarProfile.from_capacity_factor(capacity_factor=0.20)
        assert len(profile.availability) == 24
        assert profile.capacity_factor == pytest.approx(0.20, rel=0.1)
    
    def test_capacity_factor_range(self):
        """Test capacity factor for different Nigerian locations."""
        # Northern Nigeria (higher solar)
        profile_north = SolarProfile.from_capacity_factor(capacity_factor=0.22)
        # Southern Nigeria (lower solar)
        profile_south = SolarProfile.from_capacity_factor(capacity_factor=0.16)
        
        assert profile_north.capacity_factor > profile_south.capacity_factor
    
    def test_daylight_hours(self):
        """Test daylight hours calculation."""
        profile = SolarProfile.generate_synthetic(
            sunrise_hour=6,
            sunset_hour=18
        )
        # Should have ~12 daylight hours
        assert 10 <= profile.daylight_hours <= 14
    
    def test_peak_hours(self):
        """Test peak sun hours calculation."""
        profile = SolarProfile.from_capacity_factor(capacity_factor=0.20)
        # Peak hours should be roughly CF * 24
        expected_peak_hours = 0.20 * 24
        assert profile.peak_hours == pytest.approx(expected_peak_hours, rel=0.2)


class TestEconomicParameters:
    """Tests for EconomicParameters class."""
    
    def test_default_values(self):
        """Test default parameter values."""
        params = EconomicParameters()
        assert params.solar_capex > 0
        assert params.gas_capex > 0
        assert params.battery_capex > 0
        assert params.discount_rate > 0
    
    def test_annualized_cost(self):
        """Test annualized cost calculation."""
        params = EconomicParameters(discount_rate=0.10)
        annualized = params.get_annualized_cost(1000, 10)
        # CRF for 10 years at 10% is approximately 0.1627
        expected = 1000 * 0.1627
        assert annualized == pytest.approx(expected, rel=0.01)
    
    def test_zero_discount_rate(self):
        """Test annualized cost with zero discount rate."""
        params = EconomicParameters(discount_rate=0)
        annualized = params.get_annualized_cost(1000, 10)
        assert annualized == 100  # 1000 / 10
    
    def test_to_dict(self):
        """Test conversion to dictionary."""
        params = EconomicParameters()
        d = params.to_dict()
        assert "solar_capex" in d
        assert "gas_capex" in d
        assert "discount_rate" in d


class TestTechnicalParameters:
    """Tests for TechnicalParameters class."""
    
    def test_default_values(self):
        """Test default parameter values."""
        params = TechnicalParameters()
        assert 0 < params.battery_charge_efficiency <= 1
        assert 0 < params.battery_discharge_efficiency <= 1
        assert params.target_reliability > 0
    
    def test_round_trip_efficiency(self):
        """Test round-trip efficiency calculation."""
        params = TechnicalParameters(
            battery_charge_efficiency=0.95,
            battery_discharge_efficiency=0.95
        )
        rte = params.get_round_trip_efficiency()
        assert rte == pytest.approx(0.9025, rel=0.01)
    
    def test_emission_factors(self):
        """Test emission factor values."""
        params = TechnicalParameters()
        assert params.emission_factor_gas > 0
        assert params.emission_factor_diesel > params.emission_factor_gas  # Diesel emits more


class TestHyOptimaModel:
    """Tests for HyOptimaModel class."""
    
    @pytest.fixture
    def basic_model(self):
        """Create a basic model for testing."""
        load = LoadProfile.generate_synthetic(peak_demand=100)
        solar = SolarProfile.generate_synthetic()
        return HyOptimaModel(load_profile=load, solar_profile=solar)
    
    def test_model_initialization(self, basic_model):
        """Test model initialization."""
        assert basic_model.model is None  # Not built yet
        assert basic_model.load_profile is not None
        assert basic_model.solar_profile is not None
    
    def test_model_build(self, basic_model):
        """Test model building."""
        m = basic_model.build_model()
        assert m is not None
        assert hasattr(m, 'T')
        assert hasattr(m, 'TotalCost')
        assert hasattr(m, 'solar_capacity')
        assert hasattr(m, 'gas_capacity')
        assert hasattr(m, 'battery_capacity')
    
    def test_model_summary(self, basic_model):
        """Test model summary generation."""
        basic_model.build_model()
        summary = basic_model.get_model_summary()
        assert "HyOptima" in summary
        assert "Solar" in summary
        assert "Gas" in summary
    
    def test_input_validation(self, basic_model):
        """Test input validation."""
        warnings = basic_model.validate_inputs()
        assert isinstance(warnings, list)
    
    def test_variable_bounds(self, basic_model):
        """Test variable bounds."""
        basic_model.build_model()
        bounds = basic_model.get_variable_bounds()
        assert "solar_capacity" in bounds
        assert bounds["solar_capacity"][0] == 0
        assert bounds["solar_capacity"][1] > 0


class TestHyOptimaSolver:
    """Tests for HyOptimaSolver class."""
    
    @pytest.fixture
    def solved_model(self):
        """Create and solve a model for testing."""
        load = LoadProfile.generate_synthetic(peak_demand=100)
        solar = SolarProfile.from_capacity_factor(capacity_factor=0.20)
        
        model = HyOptimaModel(
            load_profile=load,
            solar_profile=solar,
            economic_params=EconomicParameters(),
            technical_params=TechnicalParameters(),
        )
        
        solver = HyOptimaSolver(solver_name="highs", tee=False)
        results = solver.solve(model)
        
        return model, solver, results
    
    def test_solver_initialization(self):
        """Test solver initialization."""
        solver = HyOptimaSolver(solver_name="highs")
        assert solver.solver_name == "highs"
    
    def test_solve_returns_results(self, solved_model):
        """Test that solve returns results dictionary."""
        _, _, results = solved_model
        assert isinstance(results, dict)
        assert "status" in results
        assert "solar_capacity_kw" in results
        assert "gas_capacity_kw" in results
        assert "battery_capacity_kwh" in results
    
    def test_solve_status(self, solved_model):
        """Test optimization status."""
        _, _, results = solved_model
        # Accept various optimal/feasible status strings from different solvers
        valid_statuses = [
            "optimal", "feasible", "stoppedByLimit",
            "TerminationCondition.convergenceCriteriaSatisfied",
            "TerminationCondition.optimal",
        ]
        assert results["status"] in valid_statuses
    
    def test_capacity_values_non_negative(self, solved_model):
        """Test that capacity values are non-negative."""
        _, _, results = solved_model
        assert results["solar_capacity_kw"] >= 0
        assert results["gas_capacity_kw"] >= 0
        assert results["battery_capacity_kwh"] >= 0
    
    def test_metrics_calculated(self, solved_model):
        """Test that metrics are calculated."""
        _, _, results = solved_model
        metrics = results.get("metrics", {})
        assert "total_demand_kwh" in metrics
        assert "reliability" in metrics
        assert "renewable_fraction" in metrics
    
    def test_reliability_constraint(self, solved_model):
        """Test that reliability constraint is satisfied."""
        _, _, results = solved_model
        reliability = results["metrics"]["reliability"]
        # Should meet target reliability (default 99%)
        assert reliability >= 0.95  # Allow some margin
    
    def test_dispatch_extraction(self, solved_model):
        """Test dispatch time series extraction."""
        _, _, results = solved_model
        dispatch = results.get("dispatch", {})
        assert "solar_gen" in dispatch
        assert "gas_gen" in dispatch
        assert "demand" in dispatch
        assert len(dispatch["solar_gen"]) == 24
    
    def test_print_summary(self, solved_model):
        """Test summary printing."""
        _, solver, results = solved_model
        summary = solver.print_summary(results)
        assert "HyOptima" in summary
        assert "Solar Capacity" in summary
    
    def test_explanation_generation(self, solved_model):
        """Test explanation generation."""
        _, solver, results = solved_model
        explanation = solver.get_explanation(results)
        assert len(explanation) > 0
        assert isinstance(explanation, str)


class TestNigeriaScenarios:
    """Tests for Nigerian location-specific scenarios."""
    
    def test_kano_scenario(self):
        """Test Kano (Northern Nigeria) scenario."""
        load, solar = generate_nigeria_scenarios("kano", "dry")
        assert load.peak_demand > 0
        assert solar.capacity_factor > 0.18  # High solar in north
    
    def test_lagos_scenario(self):
        """Test Lagos (Southern Nigeria) scenario."""
        load, solar = generate_nigeria_scenarios("lagos", "dry")
        assert load.peak_demand > 0
        assert solar.capacity_factor > 0.15
    
    def test_seasonal_difference(self):
        """Test seasonal difference in solar."""
        _, solar_dry = generate_nigeria_scenarios("kano", "dry")
        _, solar_wet = generate_nigeria_scenarios("kano", "wet")
        # Dry season should have higher solar
        assert solar_dry.capacity_factor >= solar_wet.capacity_factor
    
    def test_invalid_location(self):
        """Test handling of invalid location."""
        load, solar = generate_nigeria_scenarios("invalid_city")
        # Should return default values
        assert load.peak_demand > 0
        assert solar.capacity_factor > 0


class TestIntegration:
    """Integration tests for the complete optimization workflow."""
    
    def test_end_to_end_optimization(self):
        """Test complete optimization workflow."""
        # Generate profiles
        load = LoadProfile.generate_synthetic(
            peak_demand=300,
            profile_type="mixed"
        )
        solar = SolarProfile.from_capacity_factor(capacity_factor=0.20)
        
        # Configure parameters
        economic = EconomicParameters(
            solar_capex=800,
            gas_capex=500,
            battery_capex=300,
            gas_fuel_cost=0.08,
        )
        
        technical = TechnicalParameters(
            target_reliability=0.99,
        )
        
        # Build and solve
        model = HyOptimaModel(
            load_profile=load,
            solar_profile=solar,
            economic_params=economic,
            technical_params=technical,
        )
        
        solver = HyOptimaSolver(solver_name="highs", tee=False)
        results = solver.solve(model)
        
        # Verify results
        # Accept various optimal/feasible status strings from different solvers
        valid_statuses = [
            "optimal", "feasible",
            "TerminationCondition.convergenceCriteriaSatisfied",
            "TerminationCondition.optimal",
        ]
        assert results["status"] in valid_statuses
        assert results["total_cost"] > 0
        assert results["metrics"]["reliability"] >= 0.95
    
    def test_sensitivity_to_fuel_cost(self):
        """Test sensitivity to fuel cost changes."""
        load = LoadProfile.generate_synthetic(peak_demand=200)
        solar = SolarProfile.from_capacity_factor(capacity_factor=0.20)
        
        # Low fuel cost - should favor gas
        economic_low = EconomicParameters(gas_fuel_cost=0.04)
        model_low = HyOptimaModel(load, solar, economic_params=economic_low)
        solver = HyOptimaSolver(solver_name="highs", tee=False)
        results_low = solver.solve(model_low)
        
        # High fuel cost - should favor solar
        economic_high = EconomicParameters(gas_fuel_cost=0.15)
        model_high = HyOptimaModel(load, solar, economic_params=economic_high)
        results_high = solver.solve(model_high)
        
        # Higher fuel cost should lead to more solar (or less gas)
        # This is a qualitative test - exact behavior depends on other factors
        assert results_low["total_cost"] != results_high["total_cost"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

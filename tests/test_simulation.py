"""
Unit tests for HyOptima Monte Carlo Simulation Module

Tests cover:
- UncertaintyDistribution sampling
- SimulationConfig validation
- MonteCarloSimulator execution
- Statistical analysis
- Risk metrics calculation
"""

import pytest
import numpy as np
from typing import List

from hyoptima.simulation import (
    UncertaintyDistribution,
    SimulationConfig,
    SimulationResult,
    MonteCarloSimulator,
    create_default_uncertainties,
)
from hyoptima.profiles import LoadProfile, SolarProfile
from hyoptima.parameters import EconomicParameters, TechnicalParameters


class TestUncertaintyDistribution:
    """Tests for UncertaintyDistribution class."""
    
    def test_normal_distribution(self):
        """Test normal distribution sampling."""
        dist = UncertaintyDistribution(
            name="test_param",
            base_value=100.0,
            distribution_type="normal",
            std_dev=10.0,
        )
        
        samples = dist.sample(1000, seed=42)
        
        assert len(samples) == 1000
        assert np.isclose(np.mean(samples), 100.0, atol=2.0)
        assert np.isclose(np.std(samples), 10.0, atol=1.0)
        assert all(samples >= 0)  # Non-negative enforced
    
    def test_uniform_distribution(self):
        """Test uniform distribution sampling."""
        dist = UncertaintyDistribution(
            name="test_param",
            base_value=100.0,
            distribution_type="uniform",
            min_value=80.0,
            max_value=120.0,
        )
        
        samples = dist.sample(1000, seed=42)
        
        assert len(samples) == 1000
        assert all(samples >= 80.0)
        assert all(samples <= 120.0)
        assert np.isclose(np.mean(samples), 100.0, atol=5.0)
    
    def test_triangular_distribution(self):
        """Test triangular distribution sampling."""
        dist = UncertaintyDistribution(
            name="test_param",
            base_value=100.0,
            distribution_type="triangular",
            min_value=50.0,
            max_value=150.0,
            mode=100.0,
        )
        
        samples = dist.sample(1000, seed=42)
        
        assert len(samples) == 1000
        assert all(samples >= 50.0)
        assert all(samples <= 150.0)
        # Mode should be near the peak
        assert np.isclose(np.mean(samples), 100.0, atol=10.0)
    
    def test_lognormal_distribution(self):
        """Test lognormal distribution sampling."""
        dist = UncertaintyDistribution(
            name="test_param",
            base_value=100.0,
            distribution_type="lognormal",
            std_dev=20.0,
        )
        
        samples = dist.sample(1000, seed=42)
        
        assert len(samples) == 1000
        assert all(samples >= 0)  # Lognormal is always positive
        # Mean should be approximately base_value
        assert np.isclose(np.mean(samples), 100.0, atol=20.0)
    
    def test_reproducibility_with_seed(self):
        """Test that seed produces reproducible results."""
        dist = UncertaintyDistribution(
            name="test_param",
            base_value=100.0,
            distribution_type="normal",
            std_dev=10.0,
        )
        
        samples1 = dist.sample(100, seed=42)
        samples2 = dist.sample(100, seed=42)
        
        assert np.array_equal(samples1, samples2)
    
    def test_invalid_distribution_type(self):
        """Test that invalid distribution type raises error."""
        dist = UncertaintyDistribution(
            name="test_param",
            base_value=100.0,
            distribution_type="invalid",
        )
        
        with pytest.raises(ValueError, match="Unknown distribution type"):
            dist.sample(10)


class TestSimulationConfig:
    """Tests for SimulationConfig dataclass."""
    
    def test_default_values(self):
        """Test default configuration values."""
        config = SimulationConfig()
        
        assert config.n_scenarios == 100
        assert config.seed == 42
        assert config.n_workers == 0
        assert config.verbose is True
    
    def test_custom_values(self):
        """Test custom configuration values."""
        config = SimulationConfig(
            n_scenarios=500,
            seed=123,
            n_workers=4,
            verbose=False,
        )
        
        assert config.n_scenarios == 500
        assert config.seed == 123
        assert config.n_workers == 4
        assert config.verbose is False


class TestSimulationResult:
    """Tests for SimulationResult dataclass."""
    
    def test_result_creation(self):
        """Test creating a simulation result."""
        result = SimulationResult(
            scenario_id=1,
            solar_capacity=100.0,
            gas_capacity=50.0,
            battery_capacity=200.0,
            total_cost=1000.0,
            reliability=0.99,
            emissions=500.0,
            lcoe=0.15,
            unserved_energy=10.0,
            parameters={"solar_cf": 0.20, "fuel_cost": 0.08},
        )
        
        assert result.scenario_id == 1
        assert result.solar_capacity == 100.0
        assert result.gas_capacity == 50.0
        assert result.battery_capacity == 200.0
        assert result.total_cost == 1000.0
        assert result.reliability == 0.99
        assert result.emissions == 500.0
        assert result.lcoe == 0.15
        assert result.unserved_energy == 10.0
        assert result.parameters["solar_cf"] == 0.20


class TestMonteCarloSimulator:
    """Tests for MonteCarloSimulator class."""
    
    @pytest.fixture
    def basic_uncertainties(self):
        """Create basic uncertainty distributions for testing."""
        return [
            UncertaintyDistribution("solar_cf", 0.20, "normal", std_dev=0.02),
            UncertaintyDistribution("fuel_cost", 0.08, "triangular", min_value=0.06, max_value=0.10, mode=0.08),
        ]
    
    @pytest.fixture
    def profiles(self):
        """Create test profiles."""
        load = LoadProfile.generate_synthetic(peak_demand=100)
        solar = SolarProfile.from_capacity_factor(capacity_factor=0.20)
        return load, solar
    
    def test_initialization(self, basic_uncertainties):
        """Test simulator initialization."""
        simulator = MonteCarloSimulator(basic_uncertainties)
        
        assert len(simulator.uncertainties) == 2
        assert "solar_cf" in simulator.uncertainties
        assert "fuel_cost" in simulator.uncertainties
        assert simulator.config.n_scenarios == 100
    
    def test_initialization_with_config(self, basic_uncertainties):
        """Test simulator initialization with custom config."""
        config = SimulationConfig(n_scenarios=50, seed=123)
        simulator = MonteCarloSimulator(basic_uncertainties, config)
        
        assert simulator.config.n_scenarios == 50
        assert simulator.config.seed == 123
    
    def test_parameter_sampling(self, basic_uncertainties):
        """Test parameter sampling."""
        config = SimulationConfig(n_scenarios=10, seed=42)
        simulator = MonteCarloSimulator(basic_uncertainties, config)
        
        samples = simulator._sample_parameters(10)
        
        assert len(samples) == 10
        assert all("solar_cf" in s for s in samples)
        assert all("fuel_cost" in s for s in samples)
        # Check values are within reasonable bounds
        for s in samples:
            assert 0.10 <= s["solar_cf"] <= 0.30  # Normal distribution
            assert 0.06 <= s["fuel_cost"] <= 0.10  # Triangular bounds
    
    def test_run_simulation(self, basic_uncertainties, profiles):
        """Test running Monte Carlo simulation."""
        load, solar = profiles
        config = SimulationConfig(n_scenarios=5, seed=42, verbose=False)
        simulator = MonteCarloSimulator(basic_uncertainties, config)
        
        results = simulator.run(load, solar)
        
        assert len(results) == 5
        assert all(isinstance(r, SimulationResult) for r in results)
        assert all(r.solar_capacity >= 0 for r in results)
        assert all(r.gas_capacity >= 0 for r in results)
        assert all(r.battery_capacity >= 0 for r in results)
    
    def test_summary_statistics(self, basic_uncertainties, profiles):
        """Test summary statistics calculation."""
        load, solar = profiles
        config = SimulationConfig(n_scenarios=10, seed=42, verbose=False)
        simulator = MonteCarloSimulator(basic_uncertainties, config)
        simulator.run(load, solar)
        
        summary = simulator.summary()
        
        # Check structure
        assert "solar_capacity_kw" in summary
        assert "gas_capacity_kw" in summary
        assert "battery_capacity_kwh" in summary
        assert "total_cost" in summary
        
        # Check statistics
        for key in ["solar_capacity_kw", "gas_capacity_kw", "battery_capacity_kwh", "total_cost"]:
            assert "mean" in summary[key]
            assert "std" in summary[key]
            assert "min" in summary[key]
            assert "max" in summary[key]
            assert "p5" in summary[key]
            assert "p95" in summary[key]
    
    def test_risk_metrics(self, basic_uncertainties, profiles):
        """Test risk metrics calculation."""
        load, solar = profiles
        config = SimulationConfig(n_scenarios=10, seed=42, verbose=False)
        simulator = MonteCarloSimulator(basic_uncertainties, config)
        simulator.run(load, solar)
        
        risk = simulator.risk_metrics()
        
        # Check structure
        assert "var_95" in risk
        assert "cvar_95" in risk
        assert "worst_case" in risk
        assert "best_case" in risk
        assert "coefficient_of_variation" in risk
        
        # Check relationships
        assert risk["worst_case"] >= risk["cvar_95"]
        assert risk["cvar_95"] >= risk["var_95"]
        assert risk["best_case"] <= risk["var_95"]
        assert risk["coefficient_of_variation"] >= 0
    
    def test_to_dataframe(self, basic_uncertainties, profiles):
        """Test conversion to DataFrame."""
        load, solar = profiles
        config = SimulationConfig(n_scenarios=5, seed=42, verbose=False)
        simulator = MonteCarloSimulator(basic_uncertainties, config)
        simulator.run(load, solar)
        
        df = simulator.to_dataframe()
        
        assert len(df) == 5
        assert "scenario_id" in df.columns
        assert "solar_capacity_kw" in df.columns
        assert "gas_capacity_kw" in df.columns
        assert "battery_capacity_kwh" in df.columns
        assert "total_cost" in df.columns
    
    def test_save_and_load_results(self, basic_uncertainties, profiles, tmp_path):
        """Test saving and loading results."""
        load, solar = profiles
        config = SimulationConfig(n_scenarios=5, seed=42, verbose=False)
        simulator = MonteCarloSimulator(basic_uncertainties, config)
        simulator.run(load, solar)
        
        # Save results (JSON format)
        output_file = tmp_path / "simulation_results.json"
        simulator.save_results(str(output_file))
        
        assert output_file.exists()
        
        # Check file content
        import json
        with open(output_file, 'r') as f:
            data = json.load(f)
        
        assert "config" in data
        assert "summary" in data
        assert "scenarios" in data
        assert len(data["scenarios"]) == 5


class TestDefaultUncertainties:
    """Tests for default uncertainty creation."""
    
    def test_create_default_uncertainties(self):
        """Test creating default uncertainties."""
        uncertainties = create_default_uncertainties()
        
        assert len(uncertainties) == 4  # fuel_cost, carbon_price, demand_multiplier, solar_cf
        names = [u.name for u in uncertainties]
        assert "solar_cf" in names
        assert "fuel_cost" in names
        assert "demand_multiplier" in names
        assert "carbon_price" in names
    
    def test_default_uncertainty_values(self):
        """Test default uncertainty values are reasonable."""
        uncertainties = create_default_uncertainties()
        
        for u in uncertainties:
            if u.name == "solar_cf":
                assert u.base_value == 0.20
                assert u.distribution_type == "normal"
            elif u.name == "fuel_cost":
                assert u.base_value == 0.08
                assert u.distribution_type == "triangular"
            elif u.name == "demand_multiplier":
                assert u.base_value == 1.0
                assert u.distribution_type == "normal"


class TestSimulationWithCustomParameters:
    """Tests for simulation with custom economic/technical parameters."""
    
    @pytest.fixture
    def uncertainties(self):
        """Create uncertainty distributions."""
        return [
            UncertaintyDistribution("solar_cf", 0.20, "normal", std_dev=0.03),
            UncertaintyDistribution("fuel_cost", 0.08, "triangular", min_value=0.05, max_value=0.12),
        ]
    
    @pytest.fixture
    def profiles(self):
        """Create test profiles."""
        load = LoadProfile.generate_synthetic(peak_demand=200)
        solar = SolarProfile.from_capacity_factor(capacity_factor=0.20)
        return load, solar
    
    def test_custom_economic_parameters(self, uncertainties, profiles):
        """Test simulation with custom economic parameters."""
        load, solar = profiles
        economic = EconomicParameters(
            solar_capex=800,
            gas_capex=600,
            battery_capex=350,
            gas_fuel_cost=0.10,
            grid_tariff=0.15,
            unserved_energy_penalty=2.0,
            emission_penalty=100.0,
            discount_rate=0.10,
        )
        
        config = SimulationConfig(n_scenarios=5, seed=42, verbose=False)
        simulator = MonteCarloSimulator(uncertainties, config)
        results = simulator.run(load, solar, economic_params=economic)
        
        assert len(results) == 5
        # Higher costs should lead to different capacity decisions
    
    def test_custom_technical_parameters(self, uncertainties, profiles):
        """Test simulation with custom technical parameters."""
        load, solar = profiles
        technical = TechnicalParameters(
            solar_efficiency=0.18,
            battery_charge_efficiency=0.92,
            battery_discharge_efficiency=0.92,
            emission_factor_gas=0.25,
            target_reliability=0.95,
        )
        
        config = SimulationConfig(n_scenarios=5, seed=42, verbose=False)
        simulator = MonteCarloSimulator(uncertainties, config)
        results = simulator.run(load, solar, technical_params=technical)
        
        assert len(results) == 5


class TestSimulationEdgeCases:
    """Tests for edge cases in simulation."""
    
    def test_single_scenario(self):
        """Test running a single scenario."""
        uncertainties = [
            UncertaintyDistribution("solar_cf", 0.20, "normal", std_dev=0.01),
        ]
        load = LoadProfile.generate_synthetic(peak_demand=50)
        solar = SolarProfile.from_capacity_factor(capacity_factor=0.20)
        
        config = SimulationConfig(n_scenarios=1, seed=42, verbose=False)
        simulator = MonteCarloSimulator(uncertainties, config)
        results = simulator.run(load, solar)
        
        assert len(results) == 1
    
    def test_zero_uncertainty(self):
        """Test with zero uncertainty (deterministic)."""
        uncertainties = [
            UncertaintyDistribution("solar_cf", 0.20, "normal", std_dev=0.0),
        ]
        load = LoadProfile.generate_synthetic(peak_demand=50)
        solar = SolarProfile.from_capacity_factor(capacity_factor=0.20)
        
        config = SimulationConfig(n_scenarios=3, seed=42, verbose=False)
        simulator = MonteCarloSimulator(uncertainties, config)
        results = simulator.run(load, solar)
        
        # All results should be identical with zero uncertainty
        assert len(results) == 3
        capacities = [r.solar_capacity for r in results]
        # With zero std_dev, all solar_cf values should be the same
        # So results should be very similar (may have small numerical differences)
        assert np.std(capacities) < 1.0  # Very small variation
    
    def test_wide_uncertainty_range(self):
        """Test with wide uncertainty range."""
        uncertainties = [
            UncertaintyDistribution("fuel_cost", 0.08, "uniform", min_value=0.01, max_value=0.20),
        ]
        load = LoadProfile.generate_synthetic(peak_demand=50)
        solar = SolarProfile.from_capacity_factor(capacity_factor=0.20)
        
        config = SimulationConfig(n_scenarios=10, seed=42, verbose=False)
        simulator = MonteCarloSimulator(uncertainties, config)
        results = simulator.run(load, solar)
        
        assert len(results) == 10
        # With wide fuel cost range, we should see variation in gas capacity
        gas_capacities = [r.gas_capacity for r in results]
        assert np.std(gas_capacities) > 0  # Should have variation


class TestIntegration:
    """Integration tests for simulation module."""
    
    def test_full_simulation_workflow(self):
        """Test complete simulation workflow."""
        # Create uncertainties
        uncertainties = create_default_uncertainties()
        
        # Create profiles
        load = LoadProfile.generate_synthetic(peak_demand=150, profile_type="mixed")
        solar = SolarProfile.from_capacity_factor(capacity_factor=0.20)
        
        # Run simulation
        config = SimulationConfig(n_scenarios=20, seed=42, verbose=False)
        simulator = MonteCarloSimulator(uncertainties, config)
        results = simulator.run(load, solar)
        
        # Get summary
        summary = simulator.summary()
        
        # Get risk metrics
        risk = simulator.risk_metrics()
        
        # Verify results
        assert len(results) == 20
        assert all(r.reliability >= 0.95 for r in results)  # High reliability
        
        # Summary should have reasonable values
        assert summary["solar_capacity_kw"]["mean"] > 0
        assert summary["gas_capacity_kw"]["mean"] >= 0
        assert summary["battery_capacity_kwh"]["mean"] >= 0
        
        # Risk metrics should be consistent
        assert risk["worst_case"] >= risk["best_case"]
        assert risk["coefficient_of_variation"] >= 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

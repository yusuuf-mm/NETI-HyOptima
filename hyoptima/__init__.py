"""
NETI-HyOptima: Nigeria Energy Transition Intelligence Platform
HyOptima Core Engine - Hybrid Energy Optimization

This package contains the core optimization engine for hybrid energy systems,
designed to support Nigeria's Energy Transition Plan through actionable,
bankable investment decisions.

The HyOptima engine optimizes energy systems across cost, emissions, and 
reliability dimensions under Nigeria's Energy Transition Plan constraints.

Modules:
    - model: Pyomo optimization model definition
    - solver: Optimization execution and result extraction
    - parameters: Economic and technical parameters
    - profiles: Load and solar generation profiles
    - utils: Utility functions for visualization and analysis
"""

__version__ = "0.1.0"
__author__ = "NETI-HyOptima Team"
__description__ = "Hybrid Energy Optimization Engine for Nigeria's Energy Transition"

from .model import HyOptimaModel
from .solver import HyOptimaSolver
from .parameters import EconomicParameters, TechnicalParameters, PolicyParameters
from .profiles import LoadProfile, SolarProfile, generate_nigeria_scenarios

__all__ = [
    # Core model and solver
    "HyOptimaModel",
    "HyOptimaSolver",
    # Parameters
    "EconomicParameters",
    "TechnicalParameters", 
    "PolicyParameters",
    # Profiles
    "LoadProfile",
    "SolarProfile",
    "generate_nigeria_scenarios",
]

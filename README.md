# NETI-HyOptima

**Nigeria Energy Transition Intelligence Platform**

NETI-HyOptima is a decision intelligence platform for Nigeria's energy transition—not an academic model, not a dashboard, but a **computational policy environment** that converts the Nigeria Energy Transition Plan into actionable, bankable investment decisions.

---

## Overview

NETI-HyOptima integrates machine learning, operations research, and simulation to provide:

- **Hybrid Energy Optimization**: Optimal sizing and dispatch of solar, gas, and battery systems
- **Policy Alignment**: Constraints aligned with Nigeria's Energy Transition Plan (ETP)
- **Decision Intelligence**: Explainable optimization results for policymakers and investors
- **Scenario Analysis**: Compare locations, costs, and policy scenarios

### Core Components

| Component | Description |
|-----------|-------------|
| **HyOptima Engine** | MILP optimization for hybrid energy systems |
| **NEXUS Layer** | Agentic execution interface (Phase 2) |
| **Policy Intelligence** | ETP alignment and transition tracking |

---

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/your-org/neti-hyoptima.git
cd neti-hyoptima

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Basic Usage

```python
from hyoptima import (
    HyOptimaModel, HyOptimaSolver,
    LoadProfile, SolarProfile,
    EconomicParameters, TechnicalParameters
)

# Generate synthetic profiles for a Nigerian community
load = LoadProfile.generate_synthetic(peak_demand=300, profile_type="mixed")
solar = SolarProfile.from_capacity_factor(capacity_factor=0.20)

# Configure parameters
economic = EconomicParameters(solar_capex=800, gas_fuel_cost=0.08)
technical = TechnicalParameters(target_reliability=0.99)

# Build and solve optimization model
model = HyOptimaModel(load, solar, economic, technical)
solver = HyOptimaSolver(solver_name="highs")
results = solver.solve(model)

# View results
print(solver.print_summary(results))
```

---

## Project Structure

```
neti-hyoptima/
├── hyoptima/                 # Core optimization engine
│   ├── model.py              # Pyomo MILP model
│   ├── solver.py             # Optimization execution
│   ├── parameters.py         # Economic & technical parameters
│   ├── profiles.py           # Load & solar profile generation
│   └── utils.py              # Visualization & analysis
├── notebooks/                # Jupyter notebooks
│   └── phase1_core_model.ipynb
├── tests/                    # Unit tests
├── data/                     # Data storage
├── results/                  # Optimization outputs
├── docs/                     # Documentation
└── plans/                    # Planning documents
```

---

## Mathematical Model

HyOptima solves a Mixed-Integer Linear Program (MILP):

**Objective**: Minimize total system cost
```
min: Investment + Fuel + Grid + Penalty
```

**Constraints**:
- Power balance: supply = demand (each hour)
- Solar limit: generation ≤ capacity × availability
- Gas limit: generation ≤ capacity × on/off status
- Battery dynamics: SOC[t] = SOC[t-1] + charge - discharge
- Reliability: unserved ≤ (1 - target) × demand

---

## Nigeria-Specific Features

### Location Profiles

Pre-configured profiles for Nigerian cities:

```python
from hyoptima.profiles import generate_nigeria_scenarios

# Generate profiles for Kano (Northern Nigeria)
load, solar = generate_nigeria_scenarios("kano", season="dry")
```

| Location | Solar CF (Dry) | Solar CF (Wet) | Typical Demand |
|----------|---------------|----------------|----------------|
| Kano | 22% | 18% | 600 kW |
| Lagos | 18% | 14% | 800 kW |
| Abuja | 20% | 16% | 500 kW |
| Bauchi | 21% | 17% | 300 kW |

### Economic Parameters

Default costs based on Nigerian market estimates:

| Parameter | Value | Source |
|-----------|-------|--------|
| Solar CAPEX | $800/kW | IEA 2023 |
| Gas CAPEX | $500/kW | Market estimate |
| Battery CAPEX | $300/kWh | Li-ion trend |
| Gas fuel cost | $0.08/kWh | LPG/NG Nigeria |

---

## Development Phases

| Phase | Focus | Status |
|-------|-------|--------|
| **Phase 1** | Core optimization model | ✅ Complete |
| **Phase 2** | Simulation & uncertainty | 🔄 Planned |
| **Phase 3** | Data engineering | 📋 Planned |
| **Phase 4** | Backend API | 📋 Planned |
| **Phase 5** | Policy intelligence | 📋 Planned |
| **Phase 6** | Frontend dashboard | 📋 Planned |

---

## Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=hyoptima --cov-report=html
```

---

## Documentation

- [Comprehensive Documentation](plans/NETI-HyOptima-Comprehensive-Documentation.md)
- [Phase 1 Implementation Plan](plans/Phase1-Implementation-Plan.md)
- [Phase 1 Development Notes](docs/phase1_notes.md)

---

## Requirements

- Python 3.9+
- Pyomo >= 6.6.0
- HiGHS solver (default) or CBC

---

## License

MIT License - See LICENSE file for details.

---

## Citation

If you use NETI-HyOptima in your research, please cite:

```bibtex
@software{neti_hyoptima_2026,
  title = {NETI-HyOptima: Nigeria Energy Transition Intelligence Platform},
  author = {NETI-HyOptima Team},
  year = {2026},
  description = {Decision intelligence platform for Nigeria's energy transition}
}
```

---

## Contact

For questions or collaboration opportunities, please open an issue or contact the development team.

---

*NETI-HyOptima: Converting Nigeria's Energy Transition Plan into actionable investment decisions.*

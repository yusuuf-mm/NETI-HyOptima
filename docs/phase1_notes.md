# Phase 1 Development Notes

## Objective

Build and validate the core HyOptima optimization engine for hybrid energy systems. This phase focuses on the mathematical correctness of the optimization model without frontend, database, or cloud infrastructure.

## Current Status

- [x] Define mathematical model
- [x] Implement Pyomo model structure
- [x] Create parameter classes
- [x] Create profile generation functions
- [x] Implement solver module
- [ ] Test with synthetic data
- [ ] Validate results
- [ ] Document findings

## Key Decisions

### Solver Choice

Using **HiGHS** solver because:
- Open source and free (MIT license)
- Fast for LP/MILP problems
- Works well in Google Colab
- No license required
- Active development and good documentation

Alternative solvers:
- **CBC**: Good open-source alternative, slightly slower
- **GLPK**: Reliable but slower for larger problems
- **Gurobi**: Fastest but commercial license required

### Model Scope (v0)

The initial model focuses on:
- Single location (microgrid)
- 24-hour horizon
- Solar + Gas + Battery
- Grid connection optional
- No uncertainty (deterministic)

**Not included in v0:**
- Multi-day optimization
- Stochastic programming
- Capacity expansion planning
- Transmission constraints
- Demand response

### Mathematical Formulation

The model solves a Mixed-Integer Linear Program (MILP):

```
minimize: Total Cost = Investment + Fuel + Grid + Penalty

subject to:
    Power Balance: supply = demand (each hour)
    Solar Limit: solar_gen <= capacity * availability
    Gas Limit: gas_gen <= capacity * on/off status
    Battery Dynamics: SOC[t] = SOC[t-1] + charge - discharge
    SOC Limits: min_soc <= SOC <= max_soc
    Reliability: unserved <= (1 - target) * demand
```

### Parameter Values

Default parameters are based on Nigerian market estimates:

| Parameter             | Value     | Source                      |
| --------------------- | --------- | --------------------------- |
| Solar CAPEX           | $800/kW   | IEA 2023, emerging markets  |
| Gas CAPEX             | $500/kW   | Nigerian market estimates   |
| Battery CAPEX         | $300/kWh  | Li-ion, declining trend     |
| Gas fuel cost         | $0.08/kWh | LPG/Natural gas in Nigeria  |
| Grid tariff           | $0.12/kWh | Commercial tariff           |
| Solar capacity factor | 18-22%    | NASA POWER data for Nigeria |

## Testing Strategy

### Unit Tests

1. **Profile Generation**
   - Test synthetic load profile generation
   - Test solar profile from capacity factor
   - Test location-specific profiles

2. **Model Building**
   - Test model construction
   - Test constraint generation
   - Test variable bounds

3. **Optimization**
   - Test solver execution
   - Test result extraction
   - Test metric calculations

### Integration Tests

1. **End-to-end optimization**
   - Generate profiles
   - Build model
   - Solve
   - Extract results
   - Verify feasibility

2. **Edge cases**
   - Zero demand
   - Very high demand
   - No solar availability
   - High reliability requirements

## Validation Approach

### Sanity Checks

1. **Cost Monotonicity**
   - Higher demand should increase cost
   - Higher fuel cost should reduce gas usage
   - Lower solar cost should increase solar capacity

2. **Physical Feasibility**
   - All demand must be met (or unserved)
   - Battery SOC must stay within limits
   - Generation cannot exceed capacity

3. **Economic Reasonableness**
   - LCOE should be in realistic range ($0.10-0.30/kWh)
   - Capacity factors should be reasonable
   - Costs should match input parameters

### Benchmark Scenarios

1. **Solar-only system**
   - High solar, no gas, large battery
   - Test battery sizing logic

2. **Gas-only system**
   - No solar, gas only
   - Test generator sizing

3. **Hybrid system**
   - Balanced solar + gas + battery
   - Test trade-offs

## Known Limitations (v0)

1. **Single day optimization**
   - Does not capture seasonal variation
   - Battery initial SOC assumption matters

2. **No uncertainty**
   - Deterministic optimization
   - May underestimate capacity needs

3. **Simplified costs**
   - Not annualized properly
   - No O&M costs in objective

4. **No policy constraints**
   - Emission limits not enforced
   - Renewable targets not included

## Next Steps

1. Run optimization with various demand profiles
2. Test sensitivity to cost parameters
3. Add emission constraints
4. Extend to multiple days
5. Add stochastic scenarios (Phase 2)

## References

1. Nigeria Energy Transition Plan (2022)
2. IEA World Energy Outlook (2023)
3. IRENA Renewable Cost Database
4. NASA POWER Data for Nigeria
5. Pyomo Documentation

---

*Last Updated: February 2026*
*Phase 1 Status: Implementation Complete, Testing In Progress*

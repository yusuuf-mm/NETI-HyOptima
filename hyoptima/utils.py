"""
Utility Functions for HyOptima

This module provides visualization and analysis utilities for the
HyOptima optimization results.

Features:
- Dispatch visualization
- Cost breakdown charts
- Sensitivity analysis plots
- Result export functions

Author: NETI-HyOptima Team
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Optional, List, Tuple
import json
import os


def plot_dispatch_results(
    results: Dict,
    save_path: Optional[str] = None,
    show: bool = True,
    figsize: Tuple[int, int] = (14, 10)
) -> plt.Figure:
    """
    Plot the dispatch results from optimization.
    
    Creates a multi-panel figure showing:
    1. Generation stack with demand overlay
    2. Battery state of charge
    3. Solar availability and generation
    4. Cost breakdown (if metrics available)
    
    Args:
        results: Results dictionary from solver
        save_path: Path to save the figure (optional)
        show: Whether to display the figure
        figsize: Figure size (width, height)
    
    Returns:
        matplotlib Figure object
    
    Example:
        >>> fig = plot_dispatch_results(results, save_path="dispatch.png")
    """
    dispatch = results["dispatch"]
    hours = range(len(dispatch["demand"]))
    
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    
    # --- Plot 1: Generation Stack ---
    ax1 = axes[0, 0]
    ax1.stackplot(
        hours,
        dispatch["solar_gen"],
        dispatch["gas_gen"],
        dispatch["grid_import"],
        dispatch["battery_discharge"],
        labels=["Solar", "Gas", "Grid Import", "Battery Discharge"],
        colors=["#FFD700", "#8B4513", "#808080", "#228B22"],
        alpha=0.7
    )
    ax1.plot(hours, dispatch["demand"], "k--", linewidth=2, label="Demand", marker='o', markersize=3)
    ax1.set_xlabel("Hour", fontsize=10)
    ax1.set_ylabel("Power (kW)", fontsize=10)
    ax1.set_title("Energy Dispatch Profile", fontsize=12, fontweight='bold')
    ax1.legend(loc="upper right", fontsize=8)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, len(hours)-1)
    
    # --- Plot 2: Battery State of Charge ---
    ax2 = axes[0, 1]
    battery_capacity = results["battery_capacity_kwh"]
    ax2.fill_between(
        hours,
        dispatch["soc"],
        alpha=0.5,
        color="#228B22",
        label="State of Charge"
    )
    if battery_capacity > 0:
        ax2.axhline(y=battery_capacity, color='r', linestyle='--', alpha=0.5, label=f"Capacity ({battery_capacity:.0f} kWh)")
    ax2.set_xlabel("Hour", fontsize=10)
    ax2.set_ylabel("SOC (kWh)", fontsize=10)
    ax2.set_title("Battery State of Charge", fontsize=12, fontweight='bold')
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, len(hours)-1)
    
    # --- Plot 3: Solar Generation vs Availability ---
    ax3 = axes[1, 0]
    solar_capacity = results["solar_capacity_kw"]
    
    # Calculate available solar
    solar_available = [solar_capacity * dispatch["solar_availability"][h] for h in hours]
    
    ax3.bar(hours, solar_available, color="#FFEB3B", alpha=0.5, label="Available Solar")
    ax3.bar(hours, dispatch["solar_gen"], color="#FFD700", alpha=0.9, label="Solar Generation")
    ax3.set_xlabel("Hour", fontsize=10)
    ax3.set_ylabel("Power (kW)", fontsize=10)
    ax3.set_title("Solar Generation vs Availability", fontsize=12, fontweight='bold')
    ax3.legend(fontsize=8)
    ax3.grid(True, alpha=0.3, axis='y')
    ax3.set_xlim(-0.5, len(hours)-0.5)
    
    # --- Plot 4: Cost Breakdown or Energy Mix ---
    ax4 = axes[1, 1]
    metrics = results.get("metrics", {})
    
    if metrics:
        # Energy mix pie chart
        energy_sources = {
            "Solar": metrics.get("total_solar_kwh", 0),
            "Gas": metrics.get("total_gas_kwh", 0),
            "Grid": metrics.get("total_grid_kwh", 0),
        }
        # Remove zero values
        energy_sources = {k: v for k, v in energy_sources.items() if v > 0}
        
        if energy_sources:
            colors = {"Solar": "#FFD700", "Gas": "#8B4513", "Grid": "#808080"}
            ax4.pie(
                energy_sources.values(),
                labels=energy_sources.keys(),
                colors=[colors[k] for k in energy_sources.keys()],
                autopct='%1.1f%%',
                startangle=90
            )
            ax4.set_title("Energy Mix", fontsize=12, fontweight='bold')
        else:
            ax4.text(0.5, 0.5, "No energy data", ha='center', va='center')
    else:
        ax4.text(0.5, 0.5, "No metrics available", ha='center', va='center')
    
    plt.tight_layout()
    
    # Save if path provided
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Figure saved to {save_path}")
    
    if show:
        plt.show()
    
    return fig


def plot_capacity_comparison(
    results_list: List[Dict],
    labels: List[str],
    save_path: Optional[str] = None,
    show: bool = True
) -> plt.Figure:
    """
    Compare capacity decisions across multiple scenarios.
    
    Args:
        results_list: List of results dictionaries
        labels: Labels for each scenario
        save_path: Path to save figure
        show: Whether to display figure
    
    Returns:
        matplotlib Figure object
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = np.arange(len(labels))
    width = 0.25
    
    solar = [r["solar_capacity_kw"] for r in results_list]
    gas = [r["gas_capacity_kw"] for r in results_list]
    battery = [r["battery_capacity_kwh"] / 10 for r in results_list]  # Scale for visibility
    
    bars1 = ax.bar(x - width, solar, width, label='Solar (kW)', color='#FFD700')
    bars2 = ax.bar(x, gas, width, label='Gas (kW)', color='#8B4513')
    bars3 = ax.bar(x + width, battery, width, label='Battery (kWh / 10)', color='#228B22')
    
    ax.set_xlabel('Scenario', fontsize=11)
    ax.set_ylabel('Capacity', fontsize=11)
    ax.set_title('Capacity Comparison Across Scenarios', fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax.annotate(f'{height:.0f}',
                           xy=(bar.get_x() + bar.get_width() / 2, height),
                           xytext=(0, 3),
                           textcoords="offset points",
                           ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    
    if show:
        plt.show()
    
    return fig


def plot_sensitivity(
    parameter_values: List[float],
    results_list: List[Dict],
    parameter_name: str,
    metric: str = "total_cost",
    save_path: Optional[str] = None,
    show: bool = True
) -> plt.Figure:
    """
    Plot sensitivity analysis results.
    
    Args:
        parameter_values: List of parameter values tested
        results_list: List of results for each parameter value
        parameter_name: Name of the parameter being varied
        metric: Metric to plot ('total_cost', 'solar_capacity_kw', etc.)
        save_path: Path to save figure
        show: Whether to display figure
    
    Returns:
        matplotlib Figure object
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    metric_values = [r[metric] for r in results_list]
    
    ax.plot(parameter_values, metric_values, 'b-o', linewidth=2, markersize=8)
    ax.set_xlabel(parameter_name, fontsize=11)
    ax.set_ylabel(metric.replace('_', ' ').title(), fontsize=11)
    ax.set_title(f'Sensitivity Analysis: {metric.replace("_", " ").title()} vs {parameter_name}', 
                 fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
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
    
    This is a simplified LCOE calculation. For a complete analysis,
    use the annualized costs from EconomicParameters.
    
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
    
    return total_cost / total_energy


def calculate_npv(
    annual_cash_flows: List[float],
    discount_rate: float = 0.10
) -> float:
    """
    Calculate Net Present Value.
    
    Args:
        annual_cash_flows: List of annual cash flows (year 0 = initial investment)
        discount_rate: Discount rate
    
    Returns:
        NPV in $
    """
    npv = 0
    for t, cf in enumerate(annual_cash_flows):
        npv += cf / ((1 + discount_rate) ** t)
    return npv


def format_currency(value: float, currency: str = "$") -> str:
    """Format a value as currency."""
    if abs(value) >= 1e6:
        return f"{currency}{value/1e6:.2f}M"
    elif abs(value) >= 1e3:
        return f"{currency}{value/1e3:.2f}K"
    else:
        return f"{currency}{value:.2f}"


def format_percentage(value: float) -> str:
    """Format a value as percentage."""
    return f"{value:.1%}"


def export_results_to_csv(
    results: Dict,
    filepath: str,
    include_dispatch: bool = True
) -> None:
    """
    Export results to CSV files.
    
    Args:
        results: Results dictionary
        filepath: Base path for exports
        include_dispatch: Whether to export dispatch time series
    """
    import pandas as pd
    
    # Export summary
    summary_data = {
        "Metric": [
            "Solar Capacity (kW)",
            "Gas Capacity (kW)",
            "Battery Capacity (kWh)",
            "Total Cost ($)",
            "Renewable Fraction",
            "Reliability",
            "LCOE ($/kWh)",
            "Emissions (kg CO2)",
        ],
        "Value": [
            results["solar_capacity_kw"],
            results["gas_capacity_kw"],
            results["battery_capacity_kwh"],
            results["total_cost"],
            results["metrics"].get("renewable_fraction", 0),
            results["metrics"].get("reliability", 0),
            results["metrics"].get("lcoe_approx", 0),
            results["metrics"].get("emissions_kg", 0),
        ]
    }
    
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(f"{filepath}_summary.csv", index=False)
    
    # Export dispatch if requested
    if include_dispatch and "dispatch" in results:
        dispatch_df = pd.DataFrame(results["dispatch"])
        dispatch_df.to_csv(f"{filepath}_dispatch.csv", index=False)
    
    print(f"Results exported to {filepath}_*.csv")


def generate_report(results: Dict, template: str = "standard") -> str:
    """
    Generate a text report from optimization results.
    
    Args:
        results: Results dictionary
        template: Report template ('standard', 'detailed', 'summary')
    
    Returns:
        Formatted report string
    """
    metrics = results.get("metrics", {})
    
    if template == "summary":
        return f"""
NETI-HyOptima Optimization Summary
==================================
Status: {results['status']}
Total Cost: ${results['total_cost']:,.2f}

Capacity:
  Solar: {results['solar_capacity_kw']:.1f} kW
  Gas: {results['gas_capacity_kw']:.1f} kW  
  Battery: {results['battery_capacity_kwh']:.1f} kWh

Performance:
  Renewable: {metrics.get('renewable_fraction', 0):.1%}
  Reliability: {metrics.get('reliability', 0):.1%}
  LCOE: ${metrics.get('lcoe_approx', 0):.3f}/kWh
"""
    
    elif template == "detailed":
        return f"""
NETI-HyOptima Detailed Optimization Report
===========================================

OPTIMIZATION STATUS
-------------------
Termination: {results['status']}
Solver: {results['solver']}
Solve Time: {results['solve_time']:.2f} seconds

CAPACITY DECISIONS
------------------
Solar PV Capacity:     {results['solar_capacity_kw']:>10.1f} kW
Gas Generator Capacity: {results['gas_capacity_kw']:>10.1f} kW
Battery Capacity:      {results['battery_capacity_kwh']:>10.1f} kWh

COST ANALYSIS
-------------
Total System Cost:     ${results['total_cost']:>10,.2f}
Investment Cost:       ${metrics.get('investment_cost', 0):>10,.2f}
Fuel Cost:             ${metrics.get('fuel_cost', 0):>10,.2f}
Grid Import Cost:      ${metrics.get('grid_cost', 0):>10,.2f}
Unserved Penalty:      ${metrics.get('penalty_cost', 0):>10,.2f}

ENERGY BALANCE
--------------
Total Demand:          {metrics.get('total_demand_kwh', 0):>10.1f} kWh
Solar Generation:      {metrics.get('total_solar_kwh', 0):>10.1f} kWh
Gas Generation:        {metrics.get('total_gas_kwh', 0):>10.1f} kWh
Grid Import:           {metrics.get('total_grid_kwh', 0):>10.1f} kWh
Unserved Energy:       {metrics.get('total_unserved_kwh', 0):>10.1f} kWh

PERFORMANCE INDICATORS
----------------------
Renewable Fraction:    {metrics.get('renewable_fraction', 0):>10.1%}
Reliability:           {metrics.get('reliability', 0):>10.2%}
Solar Capacity Factor: {metrics.get('solar_capacity_factor', 0):>10.1%}
Gas Capacity Factor:   {metrics.get('gas_capacity_factor', 0):>10.1%}

EMISSIONS
---------
CO2 Emissions:         {metrics.get('emissions_kg', 0):>10.1f} kg
                       {metrics.get('emissions_tons', 0):>10.3f} tons

BATTERY PERFORMANCE
-------------------
Utilization:           {metrics.get('battery_utilization', 0):>10.1%}
Equivalent Cycles:     {metrics.get('battery_cycles', 0):>10.1f}

ECONOMIC METRICS
----------------
Approximate LCOE:      ${metrics.get('lcoe_approx', 0):>10.3f}/kWh

===========================================
Generated by NETI-HyOptima Platform
"""
    
    else:  # standard
        return f"""
NETI-HyOptima Optimization Report
=================================

Status: {results['status']}
Solver: {results['solver']} ({results['solve_time']:.2f}s)

Capacity Decisions:
  Solar:   {results['solar_capacity_kw']:.1f} kW
  Gas:     {results['gas_capacity_kw']:.1f} kW
  Battery: {results['battery_capacity_kwh']:.1f} kWh

Total Cost: ${results['total_cost']:,.2f}

Performance:
  Renewable Fraction: {metrics.get('renewable_fraction', 0):.1%}
  Reliability: {metrics.get('reliability', 0):.1%}
  LCOE: ${metrics.get('lcoe_approx', 0):.3f}/kWh
  Emissions: {metrics.get('emissions_kg', 0):.0f} kg CO2
"""


def compare_scenarios(
    results_list: List[Dict],
    labels: List[str]
) -> Dict[str, List]:
    """
    Compare multiple optimization scenarios.
    
    Args:
        results_list: List of results dictionaries
        labels: Labels for each scenario
    
    Returns:
        Dictionary with comparison metrics
    """
    comparison = {
        "scenario": labels,
        "solar_capacity_kw": [],
        "gas_capacity_kw": [],
        "battery_capacity_kwh": [],
        "total_cost": [],
        "renewable_fraction": [],
        "reliability": [],
        "lcoe": [],
        "emissions_kg": [],
    }
    
    for results in results_list:
        metrics = results.get("metrics", {})
        comparison["solar_capacity_kw"].append(results["solar_capacity_kw"])
        comparison["gas_capacity_kw"].append(results["gas_capacity_kw"])
        comparison["battery_capacity_kwh"].append(results["battery_capacity_kwh"])
        comparison["total_cost"].append(results["total_cost"])
        comparison["renewable_fraction"].append(metrics.get("renewable_fraction", 0))
        comparison["reliability"].append(metrics.get("reliability", 0))
        comparison["lcoe"].append(metrics.get("lcoe_approx", 0))
        comparison["emissions_kg"].append(metrics.get("emissions_kg", 0))
    
    return comparison


def print_comparison_table(comparison: Dict) -> str:
    """
    Print a formatted comparison table.
    
    Args:
        comparison: Comparison dictionary from compare_scenarios
    
    Returns:
        Formatted table string
    """
    scenarios = comparison["scenario"]
    n_scenarios = len(scenarios)
    
    # Header
    header = f"{'Metric':<25}" + "".join([f"{s:>15}" for s in scenarios])
    separator = "-" * len(header)
    
    lines = [separator, header, separator]
    
    # Metrics
    metrics = [
        ("Solar Capacity (kW)", "solar_capacity_kw"),
        ("Gas Capacity (kW)", "gas_capacity_kw"),
        ("Battery (kWh)", "battery_capacity_kwh"),
        ("Total Cost ($)", "total_cost"),
        ("Renewable Fraction", "renewable_fraction"),
        ("Reliability", "reliability"),
        ("LCOE ($/kWh)", "lcoe"),
        ("Emissions (kg CO2)", "emissions_kg"),
    ]
    
    for label, key in metrics:
        values = comparison[key]
        if "fraction" in key or "reliability" in key:
            row = f"{label:<25}" + "".join([f"{v:>14.1%}" for v in values])
        elif "cost" in key.lower() or "lcoe" in key:
            row = f"{label:<25}" + "".join([f"{v:>14,.0f}" for v in values])
        else:
            row = f"{label:<25}" + "".join([f"{v:>14.1f}" for v in values])
        lines.append(row)
    
    lines.append(separator)
    
    return "\n".join(lines)

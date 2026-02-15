"""
Load and Solar Generation Profiles for HyOptima Model

This module provides functions to generate and manipulate load profiles
and solar generation profiles for Nigerian energy systems.

The profiles are based on:
- NASA POWER data for Nigerian solar irradiance
- NBS (National Bureau of Statistics) for demand patterns
- Typical Nigerian consumption patterns by sector

Nigeria Solar Resource:
- Average daily irradiance: 4-6 kWh/m²/day
- Capacity factor range: 16-22%
- Peak sun hours: 4-6 hours/day

Author: NETI-HyOptima Team
"""

import numpy as np
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass
import json


@dataclass
class LoadProfile:
    """
    Represents an electrical load profile over time.
    
    Default profiles are based on typical Nigerian consumption patterns
    for residential, commercial, and mixed communities.
    
    Attributes:
        demand: Hourly demand array (kW)
        time_resolution: Hours per time step
        name: Profile identifier
        metadata: Additional profile information
    
    Properties:
        total_energy: Total energy demand (kWh)
        peak_demand: Peak demand (kW)
        load_factor: Load factor (average/peak)
    """
    
    demand: np.ndarray  # Hourly demand in kW
    time_resolution: int = 1  # Hours per time step
    name: str = "default"
    metadata: Dict = None
    
    def __post_init__(self):
        """Initialize metadata if not provided."""
        if self.metadata is None:
            self.metadata = {}
    
    @classmethod
    def generate_synthetic(
        cls,
        peak_demand: float = 500.0,  # kW
        profile_type: str = "residential",
        hours: int = 24,
        noise_level: float = 0.05,
        name: str = "synthetic"
    ) -> "LoadProfile":
        """
        Generate a synthetic load profile based on typical Nigerian patterns.
        
        Args:
            peak_demand: Peak demand in kW
            profile_type: Type of load profile
                - 'residential': Morning and evening peaks
                - 'commercial': Business hours peak
                - 'industrial': Flat with slight variation
                - 'mixed': Combination of patterns
            hours: Number of hours to generate
            noise_level: Random noise level (fraction of demand)
            name: Profile identifier
        
        Returns:
            LoadProfile with synthetic demand data
        
        Example:
            >>> profile = LoadProfile.generate_synthetic(peak_demand=300, profile_type="mixed")
            >>> print(f"Peak: {profile.peak_demand:.1f} kW")
        """
        t = np.arange(hours)
        
        if profile_type == "residential":
            # Nigerian residential: morning peak (6-8am), evening peak (7-10pm)
            # Low night demand, moderate afternoon
            base = 0.25 * np.ones(hours)  # Base load (25% of peak)
            
            # Morning peak (6-9am)
            morning_peak = 0.35 * np.exp(-((t - 7)**2) / 2)
            
            # Afternoon plateau (12-4pm)
            afternoon = 0.15 * np.where((t >= 12) & (t <= 16), 1, 0)
            
            # Evening peak (7-10pm) - highest
            evening_peak = 0.45 * np.exp(-((t - 20)**2) / 4)
            
            profile = base + morning_peak + afternoon + evening_peak
            
        elif profile_type == "commercial":
            # Commercial: high during business hours (8am-6pm)
            # Peak around midday
            base = 0.15 * np.ones(hours)  # Night base load
            
            # Business hours ramp
            business = np.where((t >= 8) & (t <= 18), 0.7, 0)
            
            # Lunch dip (1-2pm)
            lunch_dip = 0.1 * np.exp(-((t - 13)**2) / 0.5)
            
            # Evening wind-down
            evening = 0.15 * np.exp(-((t - 19)**2) / 2)
            
            profile = base + business - lunch_dip + evening
            
        elif profile_type == "industrial":
            # Industrial: relatively flat with slight variations
            # 24-hour operations typical
            base = 0.7 * np.ones(hours)
            
            # Slight day increase
            day_increase = 0.2 * np.where((t >= 7) & (t <= 19), 1, 0)
            
            # Shift change dips
            shift_dip = 0.1 * (np.exp(-((t - 6)**2) / 1) + np.exp(-((t - 18)**2) / 1))
            
            profile = base + day_increase - shift_dip
            
        elif profile_type == "mixed":
            # Mixed community: combination of residential and commercial
            # Common in Nigerian towns and cities
            
            # Residential component
            residential = 0.2 + 0.15 * np.sin(2 * np.pi * (t - 6) / 24)
            evening_res = 0.25 * np.exp(-((t - 20)**2) / 4)
            
            # Commercial component
            commercial = np.where((t >= 8) & (t <= 18), 0.35, 0.05)
            
            # Morning ramp
            morning = 0.15 * np.exp(-((t - 7)**2) / 2)
            
            profile = residential + evening_res + commercial + morning
            
        else:
            # Default: sinusoidal variation
            profile = 0.5 + 0.3 * np.sin(2 * np.pi * (t - 6) / 24)
        
        # Normalize and scale to peak demand
        profile = profile / np.max(profile) * peak_demand
        
        # Add random noise for realism
        if noise_level > 0:
            np.random.seed(42)  # Reproducibility
            noise = np.random.normal(0, noise_level * peak_demand, hours)
            profile = np.maximum(profile + noise, 0)
        
        return cls(
            demand=profile,
            time_resolution=1,
            name=name,
            metadata={
                "type": profile_type,
                "peak_demand_target": peak_demand,
                "noise_level": noise_level,
                "synthetic": True,
            }
        )
    
    @classmethod
    def from_array(cls, demand: np.ndarray, name: str = "imported", **metadata) -> "LoadProfile":
        """
        Create LoadProfile from numpy array.
        
        Args:
            demand: Demand array (kW)
            name: Profile identifier
            **metadata: Additional metadata
        
        Returns:
            LoadProfile instance
        """
        return cls(demand=demand, name=name, metadata=metadata)
    
    @classmethod
    def from_csv(cls, filepath: str, demand_column: str = "demand") -> "LoadProfile":
        """
        Load profile from CSV file.
        
        Args:
            filepath: Path to CSV file
            demand_column: Name of demand column
        
        Returns:
            LoadProfile instance
        """
        import pandas as pd
        df = pd.read_csv(filepath)
        demand = df[demand_column].values
        return cls(demand=demand, name=filepath.split("/")[-1])
    
    @property
    def total_energy(self) -> float:
        """Total energy demand in kWh."""
        return float(np.sum(self.demand) * self.time_resolution)
    
    @property
    def peak_demand(self) -> float:
        """Peak demand in kW."""
        return float(np.max(self.demand))
    
    @property
    def min_demand(self) -> float:
        """Minimum demand in kW."""
        return float(np.min(self.demand))
    
    @property
    def mean_demand(self) -> float:
        """Mean demand in kW."""
        return float(np.mean(self.demand))
    
    @property
    def load_factor(self) -> float:
        """Load factor (average/peak)."""
        if self.peak_demand == 0:
            return 0
        return float(self.mean_demand / self.peak_demand)
    
    @property
    def demand_duration(self) -> int:
        """Duration of profile in hours."""
        return len(self.demand)
    
    def get_summary(self) -> Dict:
        """Get summary statistics."""
        return {
            "name": self.name,
            "hours": self.demand_duration,
            "total_energy_kwh": self.total_energy,
            "peak_demand_kw": self.peak_demand,
            "min_demand_kw": self.min_demand,
            "mean_demand_kw": self.mean_demand,
            "load_factor": self.load_factor,
        }
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "demand": self.demand.tolist(),
            "time_resolution": self.time_resolution,
            "name": self.name,
            "metadata": self.metadata,
        }


@dataclass
class SolarProfile:
    """
    Represents solar generation profile (normalized availability).
    
    Profiles are based on Nigerian solar resource data (NASA POWER).
    Nigeria has excellent solar potential: 4-6 kWh/m²/day average.
    
    Attributes:
        availability: Normalized availability factor (0-1) per hour
        time_resolution: Hours per time step
        name: Profile identifier
        metadata: Additional profile information
    
    Properties:
        capacity_factor: Average capacity factor
        peak_hours: Number of peak sun hours
    """
    
    availability: np.ndarray  # Normalized availability (0-1)
    time_resolution: int = 1  # Hours per time step
    name: str = "default"
    metadata: Dict = None
    
    def __post_init__(self):
        """Initialize metadata if not provided."""
        if self.metadata is None:
            self.metadata = {}
    
    @classmethod
    def generate_synthetic(
        cls,
        sunrise_hour: int = 6,
        sunset_hour: int = 18,
        peak_irradiance: float = 1.0,
        hours: int = 24,
        cloud_factor: float = 0.85,
        noise_level: float = 0.03,
        name: str = "synthetic"
    ) -> "SolarProfile":
        """
        Generate a synthetic solar availability profile.
        
        Models solar irradiance using a Gaussian curve during daylight hours,
        with optional cloud cover and noise effects.
        
        Args:
            sunrise_hour: Hour of sunrise (default 6am for Nigeria)
            sunset_hour: Hour of sunset (default 6pm for Nigeria)
            peak_irradiance: Peak normalized irradiance (0-1)
            hours: Number of hours to generate
            cloud_factor: Cloud cover reduction factor (0-1)
            noise_level: Random noise level
            name: Profile identifier
        
        Returns:
            SolarProfile with synthetic availability data
        
        Example:
            >>> profile = SolarProfile.generate_synthetic(cloud_factor=0.9)
            >>> print(f"Capacity factor: {profile.capacity_factor:.1%}")
        """
        t = np.arange(hours)
        
        # Daylight mask
        day_mask = (t >= sunrise_hour) & (t <= sunset_hour)
        
        # Gaussian curve for solar intensity during day
        # Peak at solar noon (middle of daylight period)
        day_length = sunset_hour - sunrise_hour
        midday = (sunrise_hour + sunset_hour) / 2
        
        # Standard deviation controls the width of the bell curve
        sigma = day_length / 4
        
        # Base irradiance curve
        intensity = np.exp(-((t - midday)**2) / (2 * sigma**2))
        
        # Apply daylight mask
        daylight = np.where(day_mask, intensity, 0)
        
        # Apply cloud factor
        profile = daylight * peak_irradiance * cloud_factor
        
        # Add noise (only during daylight)
        if noise_level > 0:
            np.random.seed(42)
            noise = np.random.normal(0, noise_level, hours)
            profile = np.clip(profile + noise * daylight, 0, 1)
        
        return cls(
            availability=profile,
            time_resolution=1,
            name=name,
            metadata={
                "sunrise": sunrise_hour,
                "sunset": sunset_hour,
                "cloud_factor": cloud_factor,
                "synthetic": True,
            }
        )
    
    @classmethod
    def from_capacity_factor(
        cls,
        capacity_factor: float = 0.20,
        hours: int = 24,
        name: str = "from_cf"
    ) -> "SolarProfile":
        """
        Generate profile from annual capacity factor.
        
        The capacity factor is the ratio of actual energy produced to
        maximum possible energy if operating at full capacity.
        
        Typical Nigerian capacity factors:
        - Northern Nigeria (Kano, Sokoto): 20-22%
        - Southern Nigeria (Lagos, Port Harcourt): 16-18%
        - Central Nigeria (Abuja): 18-20%
        
        Args:
            capacity_factor: Annual capacity factor (typical Nigeria: 0.16-0.22)
            hours: Number of hours to generate
            name: Profile identifier
        
        Returns:
            SolarProfile scaled to match capacity factor
        """
        # Generate base profile
        profile = cls.generate_synthetic(hours=hours, noise_level=0, name=name)
        
        # Scale to match capacity factor
        # CF = mean(availability) for normalized profile
        current_mean = np.mean(profile.availability)
        if current_mean > 0:
            scale = capacity_factor / current_mean
            profile.availability = np.clip(profile.availability * scale, 0, 1)
        
        profile.metadata["target_capacity_factor"] = capacity_factor
        
        return profile
    
    @classmethod
    def from_location(
        cls,
        latitude: float,
        longitude: float,
        hours: int = 24,
        name: str = "location"
    ) -> "SolarProfile":
        """
        Generate profile based on geographic location.
        
        Uses approximate solar geometry to estimate irradiance.
        For accurate data, use NASA POWER API in production.
        
        Args:
            latitude: Location latitude (Nigeria: 4-14°N)
            longitude: Location longitude (Nigeria: 3-15°E)
            hours: Number of hours to generate
            name: Profile identifier
        
        Returns:
            SolarProfile for the location
        """
        # Approximate sunrise/sunset based on latitude
        # Nigeria is near equator, so roughly 12-hour days
        base_daylight = 12
        latitude_effect = 0.5 * np.sin(np.radians(latitude - 10))  # Small variation
        
        sunrise = int(6 - latitude_effect)
        sunset = int(18 + latitude_effect)
        
        # Capacity factor estimate based on latitude
        # Northern Nigeria has higher solar resource
        if latitude > 10:  # Northern Nigeria
            cf = 0.20 + 0.01 * (latitude - 10)
        else:  # Southern Nigeria
            cf = 0.18 - 0.005 * (10 - latitude)
        
        profile = cls.from_capacity_factor(
            capacity_factor=min(cf, 0.22),
            hours=hours,
            name=name
        )
        
        profile.metadata.update({
            "latitude": latitude,
            "longitude": longitude,
            "estimated_sunrise": sunrise,
            "estimated_sunset": sunset,
        })
        
        return profile
    
    @classmethod
    def from_csv(cls, filepath: str, availability_column: str = "availability") -> "SolarProfile":
        """
        Load profile from CSV file.
        
        Args:
            filepath: Path to CSV file
            availability_column: Name of availability column
        
        Returns:
            SolarProfile instance
        """
        import pandas as pd
        df = pd.read_csv(filepath)
        availability = df[availability_column].values
        return cls(availability=availability, name=filepath.split("/")[-1])
    
    @property
    def capacity_factor(self) -> float:
        """Calculate capacity factor from profile."""
        return float(np.mean(self.availability))
    
    @property
    def peak_hours(self) -> float:
        """Number of peak sun hours (equivalent full-power hours)."""
        return float(np.sum(self.availability))
    
    @property
    def peak_availability(self) -> float:
        """Peak availability value."""
        return float(np.max(self.availability))
    
    @property
    def daylight_hours(self) -> int:
        """Number of hours with non-zero availability."""
        return int(np.sum(self.availability > 0))
    
    def get_summary(self) -> Dict:
        """Get summary statistics."""
        return {
            "name": self.name,
            "hours": len(self.availability),
            "capacity_factor": self.capacity_factor,
            "peak_hours": self.peak_hours,
            "peak_availability": self.peak_availability,
            "daylight_hours": self.daylight_hours,
        }
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "availability": self.availability.tolist(),
            "time_resolution": self.time_resolution,
            "name": self.name,
            "metadata": self.metadata,
        }


def generate_nigeria_scenarios(
    location: str = "kano",
    season: str = "dry",
    peak_demand: float = None
) -> Tuple[LoadProfile, SolarProfile]:
    """
    Generate location-specific profiles for Nigerian cities.
    
    This function provides realistic load and solar profiles based on
    Nigerian geographic and demographic data.
    
    Args:
        location: Nigerian city name
            - 'kano': Northern Nigeria, high solar, commercial center
            - 'lagos': Southern Nigeria, lower solar, industrial hub
            - 'abuja': Central Nigeria, moderate solar, government center
            - 'port_harcourt': Niger Delta, lower solar, oil & gas
            - 'bauchi': Northeast, high solar, agricultural
            - 'sokoto': Northwest, highest solar in Nigeria
        season: Season ('dry' or 'wet')
            - Dry (Nov-Mar): Higher solar, different demand patterns
            - Wet (Apr-Oct): Lower solar, higher cooling demand
        peak_demand: Override default peak demand (kW)
    
    Returns:
        Tuple of (LoadProfile, SolarProfile)
    
    Example:
        >>> load, solar = generate_nigeria_scenarios("kano", "dry")
        >>> print(f"Peak demand: {load.peak_demand:.1f} kW")
        >>> print(f"Solar CF: {solar.capacity_factor:.1%}")
    """
    # Location-specific parameters based on Nigerian data
    location_params = {
        "kano": {
            "peak_demand": 600,
            "solar_cf_dry": 0.22,
            "solar_cf_wet": 0.18,
            "load_type": "mixed",
            "latitude": 12.0,
        },
        "lagos": {
            "peak_demand": 800,
            "solar_cf_dry": 0.18,
            "solar_cf_wet": 0.14,
            "load_type": "industrial",
            "latitude": 6.5,
        },
        "abuja": {
            "peak_demand": 500,
            "solar_cf_dry": 0.20,
            "solar_cf_wet": 0.16,
            "load_type": "commercial",
            "latitude": 9.1,
        },
        "port_harcourt": {
            "peak_demand": 700,
            "solar_cf_dry": 0.16,
            "solar_cf_wet": 0.12,
            "load_type": "industrial",
            "latitude": 4.8,
        },
        "bauchi": {
            "peak_demand": 300,
            "solar_cf_dry": 0.23,
            "solar_cf_wet": 0.19,
            "load_type": "residential",
            "latitude": 10.5,
        },
        "sokoto": {
            "peak_demand": 250,
            "solar_cf_dry": 0.24,  # Highest in Nigeria
            "solar_cf_wet": 0.20,
            "load_type": "residential",
            "latitude": 13.0,
        },
    }
    
    # Get location parameters or default
    params = location_params.get(
        location,
        {"peak_demand": 500, "solar_cf_dry": 0.20, "solar_cf_wet": 0.16, "load_type": "mixed", "latitude": 9.0}
    )
    
    # Determine capacity factor based on season
    if season == "wet":
        solar_cf = params["solar_cf_wet"]
        cloud_factor = 0.75  # More clouds in wet season
    else:
        solar_cf = params["solar_cf_dry"]
        cloud_factor = 0.90
    
    # Override peak demand if provided
    demand = peak_demand if peak_demand is not None else params["peak_demand"]
    
    # Generate profiles
    load = LoadProfile.generate_synthetic(
        peak_demand=demand,
        profile_type=params["load_type"],
        name=f"{location}_{season}_load"
    )
    
    solar = SolarProfile.from_capacity_factor(
        capacity_factor=solar_cf,
        name=f"{location}_{season}_solar"
    )
    
    # Add location metadata
    load.metadata.update({
        "location": location,
        "season": season,
        "country": "Nigeria",
    })
    
    solar.metadata.update({
        "location": location,
        "season": season,
        "country": "Nigeria",
        "latitude": params["latitude"],
    })
    
    return load, solar


def create_multi_day_profile(
    base_profile: LoadProfile,
    days: int = 7,
    daily_variation: float = 0.1
) -> LoadProfile:
    """
    Extend a single-day profile to multiple days with variation.
    
    Args:
        base_profile: Single-day LoadProfile
        days: Number of days to extend
        daily_variation: Daily variation factor (0-1)
    
    Returns:
        Multi-day LoadProfile
    """
    np.random.seed(42)
    
    single_day = base_profile.demand
    hours_per_day = len(single_day)
    
    multi_day = []
    for day in range(days):
        # Add daily variation
        variation = 1 + np.random.normal(0, daily_variation)
        day_profile = single_day * variation
        multi_day.extend(day_profile)
    
    return LoadProfile(
        demand=np.array(multi_day),
        name=f"{base_profile.name}_{days}day",
        metadata={
            **base_profile.metadata,
            "days": days,
            "daily_variation": daily_variation,
        }
    )

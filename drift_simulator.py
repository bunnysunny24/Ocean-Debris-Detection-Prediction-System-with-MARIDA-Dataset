"""
Physics-Based Drift Modeling Module
Ocean current and wind-based debris drift prediction using Parcels-like integration
"""

import numpy as np
from typing import Tuple, List
import json


# ============================================================================
# OCEAN CURRENT & WIND DATA INTEGRATION
# ============================================================================

class OceanCurrentData:
    """Load and interpolate ocean current data (CMEMS format)."""
    
    def __init__(self, u_data=None, v_data=None, lat_grid=None, lon_grid=None, time_grid=None):
        """
        Args:
            u_data: Zonal velocity component (time, lat, lon)
            v_data: Meridional velocity component (time, lat, lon)
            lat_grid: Latitude coordinates
            lon_grid: Longitude coordinates
            time_grid: Time coordinates
        """
        self.u_data = u_data
        self.v_data = v_data
        self.lat_grid = lat_grid
        self.lon_grid = lon_grid
        self.time_grid = time_grid
    
    def get_velocity_at(self, lat, lon, time_idx=0):
        """
        Interpolate velocity at given lat/lon/time.
        
        Args:
            lat: Latitude
            lon: Longitude
            time_idx: Time index
        
        Returns:
            (u, v) velocity components
        """
        if self.u_data is None:
            return 0.0, 0.0
        
        # Simple nearest-neighbor for now (would use scipy.interpolate in production)
        lat_idx = np.argmin(np.abs(self.lat_grid - lat))
        lon_idx = np.argmin(np.abs(self.lon_grid - lon))
        
        u = self.u_data[time_idx, lat_idx, lon_idx]
        v = self.v_data[time_idx, lat_idx, lon_idx]
        
        return u, v


class WindData:
    """Load and interpolate wind data (ERA5 format)."""
    
    def __init__(self, u10_data=None, v10_data=None, lat_grid=None, lon_grid=None, time_grid=None):
        """
        Args:
            u10_data: 10m zonal wind (time, lat, lon)
            v10_data: 10m meridional wind (time, lat, lon)
            lat_grid: Latitude coordinates
            lon_grid: Longitude coordinates
            time_grid: Time coordinates
        """
        self.u10_data = u10_data
        self.v10_data = v10_data
        self.lat_grid = lat_grid
        self.lon_grid = lon_grid
        self.time_grid = time_grid
    
    def get_wind_at(self, lat, lon, time_idx=0):
        """
        Interpolate wind at given lat/lon/time.
        
        Args:
            lat: Latitude
            lon: Longitude
            time_idx: Time index
        
        Returns:
            (u10, v10) wind components at 10m
        """
        if self.u10_data is None:
            return 0.0, 0.0
        
        lat_idx = np.argmin(np.abs(self.lat_grid - lat))
        lon_idx = np.argmin(np.abs(self.lon_grid - lon))
        
        u10 = self.u10_data[time_idx, lat_idx, lon_idx]
        v10 = self.v10_data[time_idx, lat_idx, lon_idx]
        
        return u10, v10


# ============================================================================
# DRIFT SIMULATION ENGINE
# ============================================================================

class DebrisParticle:
    """Single debris particle for Lagrangian tracking."""
    
    def __init__(self, particle_id, lat, lon, timestamp, debris_type='plastic'):
        """
        Args:
            particle_id: Unique identifier
            lat: Initial latitude
            lon: Initial longitude
            timestamp: Initial timestamp
            debris_type: Type of debris (plastic, foam, algae, etc.)
        """
        self.id = particle_id
        self.lat = lat
        self.lon = lon
        self.timestamp = timestamp
        self.debris_type = debris_type
        
        # Trajectory history
        self.trajectory = [(lat, lon, timestamp)]
    
    def update_position(self, u_current, v_current, u_wind, v_wind, 
                       dt_hours=1.0, leeway_coeff=0.03):
        """
        Update particle position using Euler integration.
        
        Args:
            u_current: Zonal ocean current (m/s)
            v_current: Meridional ocean current (m/s)
            u_wind: Zonal wind at 10m (m/s)
            v_wind: Meridional wind at 10m (m/s)
            dt_hours: Time step in hours
            leeway_coeff: Leeway coefficient for wind drag (typical: 0.02-0.04)
        
        Note:
            Advection equation:
            dx/dt = u_current + leeway_coeff * u_wind
            dy/dt = v_current + leeway_coeff * v_wind
        """
        # Convert time step to seconds
        dt_seconds = dt_hours * 3600
        
        # Total velocity = ocean current + wind-driven leeway
        u_total = u_current + leeway_coeff * u_wind
        v_total = v_current + leeway_coeff * v_wind
        
        # Degrees per meter at equator (~0.00001)
        DEG_PER_METER = 1.0 / 111320.0
        
        # Update position (Euler integration)
        dlat = v_total * dt_seconds * DEG_PER_METER
        dlon = (u_total * dt_seconds * DEG_PER_METER) / np.cos(np.radians(self.lat))
        
        self.lat += dlat
        self.lon += dlon
        self.timestamp += dt_hours
        
        # Store in trajectory
        self.trajectory.append((self.lat, self.lon, self.timestamp))
    
    def get_trajectory(self):
        """Return full trajectory as list of (lat, lon, time) tuples."""
        return self.trajectory


class DriftSimulator:
    """Simulate debris drift using ocean currents and wind data."""
    
    def __init__(self, ocean_currents=None, wind_data=None, leeway_coeff=0.03):
        """
        Args:
            ocean_currents: OceanCurrentData instance
            wind_data: WindData instance
            leeway_coeff: Leeway coefficient for wind effect
        """
        self.ocean_currents = ocean_currents
        self.wind_data = wind_data
        self.leeway_coeff = leeway_coeff
    
    def simulate_drift(self, initial_positions: List[Tuple[float, float]], 
                      debris_types: List[str] = None,
                      duration_hours: int = 24,
                      dt_hours: float = 1.0) -> List[DebrisParticle]:
        """
        Simulate drift for multiple debris particles.
        
        Args:
            initial_positions: List of (lat, lon) tuples
            debris_types: List of debris type strings
            duration_hours: Total simulation time in hours
            dt_hours: Time step in hours
        
        Returns:
            List of DebrisParticle objects with trajectories
        """
        particles = []
        
        for idx, (lat, lon) in enumerate(initial_positions):
            debris_type = debris_types[idx] if debris_types else 'plastic'
            particle = DebrisParticle(idx, lat, lon, 0.0, debris_type)
            
            # Simulate trajectory
            num_steps = int(duration_hours / dt_hours)
            for step in range(num_steps):
                time_idx = min(step, 23)  # Assume 24-hour forecast
                
                # Get forcing fields
                u_curr, v_curr = self.ocean_currents.get_velocity_at(
                    particle.lat, particle.lon, time_idx
                ) if self.ocean_currents else (0, 0)
                
                u_wind, v_wind = self.wind_data.get_wind_at(
                    particle.lat, particle.lon, time_idx
                ) if self.wind_data else (0, 0)
                
                # Update particle position
                particle.update_position(
                    u_curr, v_curr, u_wind, v_wind,
                    dt_hours=dt_hours,
                    leeway_coeff=self.leeway_coeff
                )
            
            particles.append(particle)
        
        return particles


# ============================================================================
# DRIFT TRAJECTORY ANALYSIS
# ============================================================================

class TrajectoryAnalyzer:
    """Analyze and visualize debris trajectories."""
    
    @staticmethod
    def get_drift_distance(particle: DebrisParticle) -> float:
        """
        Calculate total drift distance in km.
        
        Args:
            particle: DebrisParticle instance
        
        Returns:
            Total drift distance in kilometers
        """
        trajectory = particle.get_trajectory()
        if len(trajectory) < 2:
            return 0.0
        
        total_dist = 0.0
        for i in range(1, len(trajectory)):
            lat1, lon1, _ = trajectory[i-1]
            lat2, lon2, _ = trajectory[i]
            
            # Haversine formula
            dlat = np.radians(lat2 - lat1)
            dlon = np.radians(lon2 - lon1)
            a = (np.sin(dlat/2)**2 +
                 np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * 
                 np.sin(dlon/2)**2)
            c = 2 * np.arcsin(np.sqrt(a))
            total_dist += 6371 * c  # Earth radius in km
        
        return total_dist
    
    @staticmethod
    def get_displacement(particle: DebrisParticle) -> Tuple[float, float]:
        """
        Calculate straight-line displacement from start to end.
        
        Args:
            particle: DebrisParticle instance
        
        Returns:
            (distance_km, bearing_degrees)
        """
        trajectory = particle.get_trajectory()
        start_lat, start_lon, _ = trajectory[0]
        end_lat, end_lon, _ = trajectory[-1]
        
        # Haversine distance
        dlat = np.radians(end_lat - start_lat)
        dlon = np.radians(end_lon - start_lon)
        a = (np.sin(dlat/2)**2 +
             np.cos(np.radians(start_lat)) * np.cos(np.radians(end_lat)) * 
             np.sin(dlon/2)**2)
        c = 2 * np.arcsin(np.sqrt(a))
        distance = 6371 * c
        
        # Bearing
        y = np.sin(dlon) * np.cos(np.radians(end_lat))
        x = (np.cos(np.radians(start_lat)) * np.sin(np.radians(end_lat)) -
             np.sin(np.radians(start_lat)) * np.cos(np.radians(end_lat)) * np.cos(dlon))
        bearing = np.degrees(np.arctan2(y, x)) % 360
        
        return distance, bearing
    
    @staticmethod
    def export_trajectory_geojson(particles: List[DebrisParticle], 
                                  filepath: str = 'drift_trajectories.geojson'):
        """
        Export particle trajectories as GeoJSON.
        
        Args:
            particles: List of DebrisParticle objects
            filepath: Output GeoJSON file path
        """
        features = []
        
        for particle in particles:
            trajectory = particle.get_trajectory()
            coords = [[lon, lat] for lat, lon, _ in trajectory]
            
            feature = {
                'type': 'Feature',
                'properties': {
                    'id': particle.id,
                    'debris_type': particle.debris_type,
                    'distance_km': TrajectoryAnalyzer.get_drift_distance(particle),
                    'duration_hours': trajectory[-1][2]
                },
                'geometry': {
                    'type': 'LineString',
                    'coordinates': coords
                }
            }
            features.append(feature)
        
        geojson = {
            'type': 'FeatureCollection',
            'features': features
        }
        
        with open(filepath, 'w') as f:
            json.dump(geojson, f, indent=2)
        
        print(f"✓ Exported {len(particles)} trajectories to {filepath}")


# ============================================================================
# HEATMAP GENERATION
# ============================================================================

def generate_drift_heatmap(particles: List[DebrisParticle],
                          lat_bounds: Tuple[float, float],
                          lon_bounds: Tuple[float, float],
                          resolution: int = 100) -> np.ndarray:
    """
    Generate a 2D heatmap of debris density from trajectories.
    
    Args:
        particles: List of DebrisParticle objects
        lat_bounds: (min_lat, max_lat)
        lon_bounds: (min_lon, max_lon)
        resolution: Grid resolution (pixels)
    
    Returns:
        (resolution, resolution) heatmap array
    """
    heatmap = np.zeros((resolution, resolution))
    
    lat_edges = np.linspace(lat_bounds[0], lat_bounds[1], resolution + 1)
    lon_edges = np.linspace(lon_bounds[0], lon_bounds[1], resolution + 1)
    
    for particle in particles:
        trajectory = particle.get_trajectory()
        for lat, lon, _ in trajectory:
            lat_idx = np.searchsorted(lat_edges, lat) - 1
            lon_idx = np.searchsorted(lon_edges, lon) - 1
            
            if 0 <= lat_idx < resolution and 0 <= lon_idx < resolution:
                heatmap[lat_idx, lon_idx] += 1
    
    # Normalize
    if heatmap.max() > 0:
        heatmap = heatmap / heatmap.max()
    
    return heatmap

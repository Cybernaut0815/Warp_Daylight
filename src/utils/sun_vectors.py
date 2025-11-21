"""
Helper functions to generate sun direction vectors using Ladybug
"""
import numpy as np
from typing import List, Tuple, Optional, Union
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

try:
    from ladybug.sunpath import Sunpath
    from ladybug.location import Location
    LADYBUG_AVAILABLE = True
except ImportError:
    LADYBUG_AVAILABLE = False


def _convert_timezone_to_offset(timezone: Union[str, int, float], reference_datetime: datetime = None) -> float:
    """
    Convert timezone string to numeric UTC offset in hours.
    
    Args:
        timezone: Either a timezone string (e.g., 'America/New_York') or numeric offset
        reference_datetime: Reference datetime for DST calculation (default: now)
    
    Returns:
        Numeric UTC offset in hours (e.g., -5.0 for EST)
    """
    # If already numeric, return as is
    if isinstance(timezone, (int, float)):
        return float(timezone)
    
    # Convert timezone string to numeric offset
    try:
        if reference_datetime is None:
            reference_datetime = datetime.now()
        
        # Create a timezone-aware datetime
        tz = ZoneInfo(timezone)
        aware_datetime = reference_datetime.replace(tzinfo=tz)
        
        # Get the UTC offset for the reference datetime (handles DST)
        offset = aware_datetime.utcoffset()
        if offset is None:
            raise ValueError(f"Could not determine UTC offset for timezone: {timezone}")
        
        return offset.total_seconds() / 3600.0  # Convert to hours
    except ValueError as exc:
        # If conversion fails, try to parse as float
        try:
            return float(timezone)
        except ValueError:
            raise ValueError(
                f"Invalid timezone: {timezone}. "
                "Expected a timezone string (e.g., 'America/New_York') or numeric offset (e.g., -5.0)"
            ) from exc


def get_sun_vectors_from_ladybug(
    latitude: float,
    longitude: float,
    start_datetime: datetime,
    end_datetime: datetime,
    time_step_minutes: int = 60,
    timezone: Union[str, int, float] = 'UTC',
    coordinate_system: str = 'y_up'
) -> Tuple[np.ndarray, List[datetime]]:
    """
    Get sun direction vectors for a time period using Ladybug.
    
    Args:
        latitude: Location latitude in degrees (-90 to 90)
        longitude: Location longitude in degrees (-180 to 180)
        start_datetime: Start date and time
        end_datetime: End date and time
        time_step_minutes: Time step in minutes (default: 60 = hourly)
        timezone: Timezone string (e.g., 'America/New_York') or numeric UTC offset (e.g., -5.0)
        coordinate_system: 'z_up' for Ladybug's default or 'y_up' to convert (default: 'y_up')
    
    Returns:
        light_directions: Array of light direction vectors FROM sun (N, 3)
        timestamps: List of datetime objects corresponding to each vector
    
    Example:
        >>> from datetime import datetime
        >>> # Get sun vectors for a summer day in New York
        >>> light_dirs, times = get_sun_vectors_from_ladybug(
        ...     latitude=40.7128,
        ...     longitude=-74.0060,
        ...     start_datetime=datetime(2024, 6, 21, 6, 0),
        ...     end_datetime=datetime(2024, 6, 21, 20, 0),
        ...     time_step_minutes=30
        ... )
    """
    if not LADYBUG_AVAILABLE:
        raise ImportError(
            "Ladybug is not installed. Install it with:\n"
            "  pip install ladybug-core"
        )
    
    # Convert timezone to numeric offset (Ladybug expects numeric UTC offset)
    timezone_offset = _convert_timezone_to_offset(timezone, start_datetime)
    
    # Create location
    location = Location(
        latitude=latitude,
        longitude=longitude,
        time_zone=timezone_offset
    )
    
    # Create sunpath calculator
    sunpath = Sunpath.from_location(location)
    
    # Generate time series
    current_time = start_datetime
    timestamps = []
    sun_positions = []
    
    time_delta_minutes = time_step_minutes
    
    while current_time <= end_datetime:
        # Get sun position for this time
        sun = sunpath.calculate_sun_from_date_time(current_time)
        
        # Only include times when sun is above horizon
        if sun.altitude > 0:
            # Get sun vector (Ladybug uses different coordinate system)
            # Ladybug: azimuth (0=North, 90=East), altitude (degrees above horizon)
            # Convert to Cartesian coordinates
            
            # Azimuth: 0° = North, 90° = East, 180° = South, 270° = West
            # Altitude: 0° = horizon, 90° = zenith
            
            altitude_rad = np.radians(sun.altitude)
            azimuth_rad = np.radians(sun.azimuth)
            
            # Convert to Cartesian (standard convention: Y=North, X=East, Z=Up)
            x = np.sin(azimuth_rad) * np.cos(altitude_rad)  # East
            y = np.cos(azimuth_rad) * np.cos(altitude_rad)  # North
            z = np.sin(altitude_rad)  # Up
            
            sun_position = np.array([x, y, z])
            sun_position = sun_position / np.linalg.norm(sun_position)  # Normalize
            
            # Light direction is FROM the sun (negative of sun position)
            light_direction = -sun_position
            
            sun_positions.append(light_direction)
            timestamps.append(current_time)
        
        # Increment time
        current_time += timedelta(minutes=time_delta_minutes)
    
    if len(sun_positions) == 0:
        raise ValueError(
            f"No sun positions found above horizon between {start_datetime} and {end_datetime}. "
            "Check your location and time range."
        )
    
    light_directions = np.array(sun_positions, dtype=np.float32)
    
    # Convert coordinate system if requested
    # Ladybug uses: X=East, Y=North, Z=Up
    # Y-up system uses: X=East, Y=Up, Z=North
    if coordinate_system == 'y_up':
        # Convert from Z-up to Y-up: (x, y, z) -> (x, z, y)
        light_directions = light_directions[:, [0, 2, 1]]
    
    return light_directions, timestamps


def get_yearly_sun_vectors(
    latitude: float,
    longitude: float,
    year: Optional[int] = None,
    time_step_hours: int = 1,
    timezone: Union[str, int, float] = 'UTC',
    coordinate_system: str = 'y_up'
) -> Tuple[np.ndarray, List[datetime]]:
    """
    Get sun vectors for an entire year (only includes times when sun is above horizon).
    
    Args:
        latitude: Location latitude in degrees
        longitude: Location longitude in degrees
        year: Year to analyze (default: current year)
        time_step_hours: Time step in hours (default: 1 = hourly)
        timezone: Timezone string (e.g., 'America/New_York') or numeric UTC offset (e.g., -5.0)
        coordinate_system: 'z_up' for Ladybug's default or 'y_up' to convert (default: 'y_up')
    
    Returns:
        light_directions: Array of light direction vectors FROM sun
        timestamps: List of datetime objects
    """
    if year is None:
        year = datetime.now().year
    
    start_datetime = datetime(year, 1, 1, 0, 0)
    end_datetime = datetime(year, 12, 31, 23, 59)
    
    return get_sun_vectors_from_ladybug(
        latitude=latitude,
        longitude=longitude,
        start_datetime=start_datetime,
        end_datetime=end_datetime,
        time_step_minutes=time_step_hours * 60,
        timezone=timezone,
        coordinate_system=coordinate_system
    )


def get_sun_vectors_for_typical_day(
    latitude: float,
    longitude: float,
    month: Optional[int] = None,
    day: Optional[int] = None,
    year: Optional[int] = None,
    time_step_minutes: int = 15,
    timezone: Union[str, int, float] = 'UTC',
    coordinate_system: str = 'y_up'
) -> Tuple[np.ndarray, List[datetime]]:
    """
    Get sun vectors for a typical day (e.g., summer solstice, winter solstice).
    
    Args:
        latitude: Location latitude in degrees
        longitude: Location longitude in degrees
        month: Month (1-12), default: current month
        day: Day of month, default: current day
        year: Year, default: current year
        time_step_minutes: Time step in minutes
        timezone: Timezone string (e.g., 'America/New_York') or numeric UTC offset (e.g., -5.0)
        coordinate_system: 'z_up' for Ladybug's default or 'y_up' to convert (default: 'y_up')
    
    Returns:
        light_directions: Array of light direction vectors FROM sun
        timestamps: List of datetime objects
        
    Example typical days:
        - Summer solstice: month=6, day=21
        - Winter solstice: month=12, day=21
        - Spring equinox: month=3, day=20
        - Fall equinox: month=9, day=22
    """
    # Use current date if not provided
    now = datetime.now()
    if year is None:
        year = now.year
    if month is None:
        month = now.month
    if day is None:
        day = now.day
    
    start_datetime = datetime(year, month, day, 0, 0)
    end_datetime = datetime(year, month, day, 23, 59)
    
    return get_sun_vectors_from_ladybug(
        latitude=latitude,
        longitude=longitude,
        start_datetime=start_datetime,
        end_datetime=end_datetime,
        time_step_minutes=time_step_minutes,
        timezone=timezone,
        coordinate_system=coordinate_system
    )


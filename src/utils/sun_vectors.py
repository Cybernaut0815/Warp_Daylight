"""
Sun direction vector generation using the Ladybug library.

Converts geographic coordinates and time ranges into arrays of unit-length
light direction vectors (pointing FROM the sun) suitable for GPU raycasting.
"""
import math
import numpy as np
from functools import lru_cache
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

# TODO(roadmap): Replace Ladybug modules for retrieving sun vectors.
#   The Ladybug dependency could be swapped for a lighter solar-position
#   library (e.g. pvlib or a pure-numpy implementation) to reduce the
#   install footprint and allow easier packaging.
try:
    from ladybug.sunpath import Sunpath
    from ladybug.location import Location
    LADYBUG_AVAILABLE = True
except ImportError:
    LADYBUG_AVAILABLE = False


def _convert_timezone_to_offset(
    timezone: str | int | float,
    reference_datetime: datetime | None = None,
) -> float:
    """Convert a timezone identifier to a numeric UTC offset in hours.

    Parameters
    ----------
    timezone:
        IANA timezone string (e.g. ``'America/New_York'``) or a numeric
        UTC offset (e.g. ``-5.0``).
    reference_datetime:
        Used to resolve DST.  Defaults to ``datetime.now()``.

    Returns
    -------
    float
        UTC offset in hours (e.g. ``-5.0`` for EST).
    """
    if isinstance(timezone, (int, float)):
        return float(timezone)

    try:
        if reference_datetime is None:
            reference_datetime = datetime.now()

        tz = ZoneInfo(timezone)
        aware_datetime = reference_datetime.replace(tzinfo=tz)
        offset = aware_datetime.utcoffset()
        if offset is None:
            raise ValueError(f"Could not determine UTC offset for timezone: {timezone}")

        return offset.total_seconds() / 3600.0
    except ValueError as exc:
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
    timezone: str | int | float = "UTC",
    coordinate_system: str = "y_up",
) -> tuple[np.ndarray, list[datetime]]:
    """Generate sun direction vectors for a time range using Ladybug.

    Only hours where the sun is **above the horizon** (altitude > 0) are
    included.

    Parameters
    ----------
    latitude:
        Location latitude in degrees (-90 to 90).
    longitude:
        Location longitude in degrees (-180 to 180).
    start_datetime / end_datetime:
        Time window to evaluate.
    time_step_minutes:
        Resolution in minutes (default 60 = hourly).
    timezone:
        IANA name or numeric UTC offset.
    coordinate_system:
        ``'z_up'`` for Ladybug default, ``'y_up'`` (Maya / USD) to swap
        Y and Z.

    Returns
    -------
    light_directions:
        (N, 3) float32 — unit-length vectors pointing FROM the sun.
    timestamps:
        Corresponding ``datetime`` objects.
    """
    if not LADYBUG_AVAILABLE:
        raise ImportError(
            "Ladybug is not installed. Install it with:\n"
            "  pip install ladybug-core"
        )

    timezone_offset = _convert_timezone_to_offset(timezone, start_datetime)

    location = Location(
        latitude=latitude,
        longitude=longitude,
        time_zone=timezone_offset,
    )

    # TODO(roadmap): Replace Ladybug Sunpath usage — a custom or lighter
    #   solar-position calculator would be plugged in here.
    sunpath = Sunpath.from_location(location)

    current_time = start_datetime
    time_delta = timedelta(minutes=time_step_minutes)
    deg2rad = math.pi / 180.0

    altitudes: list[float] = []
    azimuths: list[float] = []
    timestamps: list[datetime] = []

    while current_time <= end_datetime:
        sun = sunpath.calculate_sun_from_date_time(current_time)
        if sun.altitude > 0:
            altitudes.append(sun.altitude * deg2rad)
            azimuths.append(sun.azimuth * deg2rad)
            timestamps.append(current_time)
        current_time += time_delta

    if len(altitudes) == 0:
        raise ValueError(
            f"No sun positions found above horizon between {start_datetime} "
            f"and {end_datetime}. Check your location and time range."
        )

    alt = np.array(altitudes, dtype=np.float32)
    az = np.array(azimuths, dtype=np.float32)

    cos_alt = np.cos(alt)
    x = -(np.sin(az) * cos_alt)
    y = -(np.cos(az) * cos_alt)
    z = -np.sin(alt)

    light_directions = np.column_stack((x, y, z))

    if coordinate_system == "y_up":
        light_directions = light_directions[:, [0, 2, 1]]

    return light_directions, timestamps


def get_yearly_sun_vectors(
    latitude: float,
    longitude: float,
    year: int | None = None,
    time_step_hours: int = 1,
    timezone: str | int | float = "UTC",
    coordinate_system: str = "y_up",
) -> tuple[np.ndarray, list[datetime]]:
    """Sun vectors for an entire year (only above-horizon positions).

    Results are LRU-cached internally so repeated identical calls are free.

    Parameters
    ----------
    latitude / longitude:
        Location in degrees.
    year:
        Defaults to the current year.
    time_step_hours:
        Resolution in hours (default 1).
    timezone:
        IANA name or numeric UTC offset.
    coordinate_system:
        ``'y_up'`` or ``'z_up'``.

    Returns
    -------
    light_directions:
        (N, 3) float32 vectors FROM the sun.
    timestamps:
        Corresponding ``datetime`` objects.
    """
    if year is None:
        year = datetime.now().year

    dirs, ts = _get_yearly_sun_vectors_cached(
        latitude, longitude, year, time_step_hours,
        timezone if isinstance(timezone, (int, float)) else str(timezone),
        coordinate_system,
    )
    return dirs.copy(), list(ts)


@lru_cache(maxsize=32)
def _get_yearly_sun_vectors_cached(
    latitude: float,
    longitude: float,
    year: int,
    time_step_hours: int,
    timezone: str | float,
    coordinate_system: str,
) -> tuple[np.ndarray, tuple]:
    """Cached inner implementation — returns immutable tuple of timestamps."""
    start_datetime = datetime(year, 1, 1, 0, 0)
    end_datetime = datetime(year, 12, 31, 23, 59)

    dirs, ts = get_sun_vectors_from_ladybug(
        latitude=latitude,
        longitude=longitude,
        start_datetime=start_datetime,
        end_datetime=end_datetime,
        time_step_minutes=time_step_hours * 60,
        timezone=timezone,
        coordinate_system=coordinate_system,
    )
    return dirs, tuple(ts)


def get_sun_vectors_for_typical_day(
    latitude: float,
    longitude: float,
    month: int | None = None,
    day: int | None = None,
    year: int | None = None,
    time_step_minutes: int = 15,
    timezone: str | int | float = "UTC",
    coordinate_system: str = "y_up",
) -> tuple[np.ndarray, list[datetime]]:
    """Sun vectors for a single representative day.

    Useful for quick checks (solstices, equinoxes, etc.).

    Parameters
    ----------
    latitude / longitude:
        Location in degrees.
    month / day / year:
        Date to evaluate.  Each defaults to today's value.
    time_step_minutes:
        Resolution in minutes (default 15).
    timezone:
        IANA name or numeric UTC offset.
    coordinate_system:
        ``'y_up'`` or ``'z_up'``.

    Returns
    -------
    light_directions:
        (N, 3) float32 vectors FROM the sun.
    timestamps:
        Corresponding ``datetime`` objects.

    Example typical days::

        Summer solstice:  month=6,  day=21
        Winter solstice:  month=12, day=21
        Spring equinox:   month=3,  day=20
        Autumn equinox:   month=9,  day=22
    """
    now = datetime.now()
    year = year or now.year
    month = month or now.month
    day = day or now.day

    start_datetime = datetime(year, month, day, 0, 0)
    end_datetime = datetime(year, month, day, 23, 59)

    return get_sun_vectors_from_ladybug(
        latitude=latitude,
        longitude=longitude,
        start_datetime=start_datetime,
        end_datetime=end_datetime,
        time_step_minutes=time_step_minutes,
        timezone=timezone,
        coordinate_system=coordinate_system,
    )

import numpy as np


def haversine(lat0, lon0, lat1, lon1):
    """Great-circle distance in km between two points (scalar or array)."""
    R = 6371.0
    lat0, lon0, lat1, lon1 = map(np.radians, [lat0, lon0, lat1, lon1])
    dlat = lat1 - lat0
    dlon = lon1 - lon0
    a = np.sin(dlat/2)**2 + np.cos(lat0)*np.cos(lat1)*np.sin(dlon/2)**2
    return 2*R*np.arcsin(np.sqrt(a))


def haversine_vectorized(lat1, lon1, lat2, lon2):
    """Vectorized haversine for meshgrid-shaped arrays, returns km."""
    R = 6371.0
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2.0)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2.0)**2
    return 2*R*np.arctan2(np.sqrt(a), np.sqrt(1 - a))


def coriolis(lat):
    """Coriolis parameter [s-1] at given latitude (degrees)."""
    return 2*7.2921e-5*np.sin(np.radians(lat))


def _bearing(lon1, lat1, lon2, lat2):
    """
    Forward bearing in mathematical convention (degrees, 0=East, 90=North)
    from point 1 to point 2. Inputs in degrees.
    """
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    x = np.sin(dlon)*np.cos(lat2)
    y = np.cos(lat1)*np.sin(lat2) - np.sin(lat1)*np.cos(lat2)*np.cos(dlon)
    # meteorological bearing (0=North, clockwise) → mathematical (0=East, CCW)
    met_bearing = (np.degrees(np.arctan2(x, y)) + 360) % 360
    return 90 - met_bearing


def storm_motion(lons, lats, times):
    """
    Compute storm translation speed [m/s] and direction [math degrees, 0=East]
    from track positions and times (np.datetime64 array).

    Returns arrays of the same length as input; last value is repeated from
    second-to-last.
    """
    lons = np.asarray(lons, dtype=float)
    lats = np.asarray(lats, dtype=float)
    times = np.asarray(times, dtype='datetime64[ns]')
    n = len(lons)

    speed = np.zeros(n)
    direction = np.zeros(n)

    for i in range(n - 1):
        dt_s = float((times[i+1] - times[i]).astype('timedelta64[s]').astype(float))
        dist_km = haversine(lats[i], lons[i], lats[i+1], lons[i+1])
        speed[i] = (dist_km * 1000) / dt_s if dt_s > 0 else 0.0
        direction[i] = _bearing(lons[i], lats[i], lons[i+1], lats[i+1])

    speed[-1] = speed[-2]
    direction[-1] = direction[-2]
    return speed, direction

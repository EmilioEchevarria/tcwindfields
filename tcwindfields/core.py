import warnings
import numpy as np
import pandas as pd
import xarray as xr
from scipy.interpolate import interp1d
from tqdm import tqdm

from ._utils import storm_motion, coriolis
from ._fields import cle15_wind_profile, wind_field_2d
from ._holland import beta_model, calc_pressure_profile, pressure_field_2d


def interpolate_track(times, lons, lats, vmax, pmin, rmax, interval='1h'):
    """
    Interpolate TC track arrays to a regular time interval.

    Parameters
    ----------
    times : array-like of np.datetime64
    lons : array-like, degrees E
    lats : array-like, degrees N  (negative for Southern Hemisphere)
    vmax : array-like, m/s
    pmin : array-like, hPa
    rmax : array-like, km
    interval : str
        Pandas-compatible duration string, e.g. '1h', '20min', '30min'.

    Returns
    -------
    dict with keys: times, lons, lats, vmax, pmin, rmax
        All arrays interpolated to the requested interval.
    """
    times = np.asarray(times, dtype='datetime64[ns]')
    lons = np.asarray(lons, dtype=float)
    lats = np.asarray(lats, dtype=float)
    vmax = np.asarray(vmax, dtype=float)
    pmin = np.asarray(pmin, dtype=float)
    rmax = np.asarray(rmax, dtype=float)

    dt_s = pd.Timedelta(interval).total_seconds()
    t0 = times[0]
    t_seconds = (times - t0).astype('timedelta64[s]').astype(float)
    t_end = t_seconds[-1]
    t_new = np.arange(0, t_end + dt_s, dt_s)

    def _interp(x):
        return interp1d(t_seconds, x, kind='linear', fill_value='extrapolate')(t_new)

    times_new = t0 + (t_new * 1e9).astype('timedelta64[ns]')

    return {
        'times': times_new,
        'lons': _interp(lons),
        'lats': _interp(lats),
        'vmax': _interp(vmax),
        'pmin': _interp(pmin),
        'rmax': _interp(rmax),
    }


def compute_tc_fields(times, lons, lats, vmax, pmin, rmax,
                       lons_grid, lats_grid,
                       interp_interval='20min',
                       verbose=True,
                       **cle15_kwargs):
    """
    Compute 2-D tropical cyclone wind and pressure fields for a full track.

    Parameters
    ----------
    times : array-like of np.datetime64
    lons : array-like, degrees E
    lats : array-like, degrees N  (negative = Southern Hemisphere)
    vmax : array-like, m/s  (10-min mean surface wind)
    pmin : array-like, hPa  (minimum central pressure)
    rmax : array-like, km   (radius of maximum wind)
    lons_grid : 1-D array-like, degrees E
    lats_grid : 1-D array-like, degrees N
    interp_interval : str or None
        Interpolate track to this time step before computing (e.g. '20min',
        '1h').  Pass None to skip interpolation and use input data as-is.
    verbose : bool
        Show tqdm progress bar.
    **cle15_kwargs
        Optional overrides for CLE15 model parameters:
        Cdvary, C_d, w_cool, CkCdvary, CkCd, eye_adj, alpha_eye.

    Returns
    -------
    xr.Dataset
        Variables: uwnd (m/s), vwnd (m/s), pres (Pa)
        Dimensions: time, lat, lon
    """
    times = np.asarray(times, dtype='datetime64[ns]')
    lons = np.asarray(lons, dtype=float)
    lats = np.asarray(lats, dtype=float)
    vmax = np.asarray(vmax, dtype=float)
    pmin = np.asarray(pmin, dtype=float)
    rmax = np.asarray(rmax, dtype=float)
    lons_grid = np.asarray(lons_grid, dtype=float)
    lats_grid = np.asarray(lats_grid, dtype=float)

    # Drop any rows with NaN in critical fields
    valid = ~(np.isnan(lons) | np.isnan(lats) | np.isnan(vmax)
              | np.isnan(rmax) | np.isnan(pmin))
    if not np.all(valid):
        n_dropped = int((~valid).sum())
        warnings.warn(f"Dropping {n_dropped} time step(s) with NaN values.")
        times, lons, lats = times[valid], lons[valid], lats[valid]
        vmax, pmin, rmax = vmax[valid], pmin[valid], rmax[valid]

    if interp_interval is not None:
        track = interpolate_track(times, lons, lats, vmax, pmin, rmax,
                                   interval=interp_interval)
        times = track['times']
        lons, lats = track['lons'], track['lats']
        vmax, pmin, rmax = track['vmax'], track['pmin'], track['rmax']

    spd_ms, dir_math = storm_motion(lons, lats, times)
    beta = beta_model(times, lats, pmin, spd_ms)

    nt = len(times)
    nlat = len(lats_grid)
    nlon = len(lons_grid)

    uwnd_out = np.zeros((nt, nlat, nlon), dtype=np.float32)
    vwnd_out = np.zeros((nt, nlat, nlon), dtype=np.float32)
    pres_out = np.zeros((nt, nlat, nlon), dtype=np.float32)

    iterator = tqdm(range(nt)) if verbose else range(nt)
    for T in iterator:
        # CLE15 wind profile
        rr_km, VV_ms, _ = cle15_wind_profile(vmax[T], rmax[T], lats[T],
                                               **cle15_kwargs)

        U, V = wind_field_2d(rr_km, VV_ms, lons[T], lats[T],
                              spd_ms[T], dir_math[T],
                              lons_grid, lats_grid)
        uwnd_out[T] = U.astype(np.float32)
        vwnd_out[T] = V.astype(np.float32)

        # Holland pressure profile
        rr_p, Pr_p = calc_pressure_profile(lats[T], pmin[T], vmax[T],
                                            rmax[T], beta[T])
        P2d = pressure_field_2d(lons[T], lats[T], rr_p, Pr_p,
                                 lons_grid, lats_grid)
        pres_out[T] = (P2d * 100).astype(np.float32)  # hPa → Pa

    ds = xr.Dataset(
        {
            'uwnd': (['time', 'lat', 'lon'], uwnd_out,
                     {'units': 'm s-1', 'long_name': 'zonal wind'}),
            'vwnd': (['time', 'lat', 'lon'], vwnd_out,
                     {'units': 'm s-1', 'long_name': 'meridional wind'}),
            'pres': (['time', 'lat', 'lon'], pres_out,
                     {'units': 'Pa', 'long_name': 'surface pressure'}),
        },
        coords={
            'time': times,
            'lat': lats_grid,
            'lon': lons_grid,
        },
    )
    return ds

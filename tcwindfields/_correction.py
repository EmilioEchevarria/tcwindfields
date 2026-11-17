"""
Statistical land correction (optional: compute_tc_fields(..., terrain=..., land_correction=True)).

A factor f on the wind over land, predicted by gradient-boosted decision trees (LightGBM) from 19 inputs computed
from the track and the terrain at each point. Trained on station winds near TCs (GHCNh 2000-2025). 
Applied with the land weight, so the wind over open sea is unchanged:
    wind = physics x (1 + w (f - 1)) x (1 + w (PEAK_FACTOR - 1))
PEAK_FACTOR is an empirical factor that adjusts the storm maximum at each point.
"""
import pathlib

import numpy as np
import pandas as pd

from ._utils import haversine_vectorized
from ._terrain import _lsm_bilinear

MODEL_FILE = pathlib.Path(__file__).resolve().parent.parent/'assets'/'land_correction.txt'
PEAK_FACTOR = 1.181  # median observed / corrected storm maximum at land stations (set by the fit)
F_LIMITS = (0.2, 3.0)
INPUTS = ['r_rmax', 'r_km', 'azi_strong', 'tc_vmax', 'tc_speed', 'abs_lat', 't_overland_h', 'tc_d2l', 'rmax',
          'ln_wind', 'z0_up', 'water_up', 'coast_km', 'z_dem', 'std5', 'tpi1', 'sx1_up', 'm_topo', 'm_terrain']
_BOOSTER = []


def _booster():
    if not _BOOSTER:
        import lightgbm as lgb
        _BOOSTER.append(lgb.Booster(model_file=str(MODEL_FILE)))
    return _BOOSTER[0]

def bearing_met(lat0, lon0, lat1, lon1):
    """Bearing (deg clockwise from north) from point 0 to point 1."""
    p0, p1, dl = np.radians(lat0), np.radians(lat1), np.radians(np.asarray(lon1, float) - lon0)
    return np.degrees(np.arctan2(np.sin(dl)*np.cos(p1), np.cos(p0)*np.sin(p1) - np.sin(p0)*np.cos(p1)*np.cos(dl))) % 360

def track_land(lons, lats, times, box_deg=10.0, cap_km=1500.0):
    """
    Along the track, from the ERA5 land-sea mask: distance (km) from the centre to land (0 when the centre is over
    land: mask >= 0.5) and hours since the centre was last over water (since the start if never).
    """
    lons, lats = np.asarray(lons, float), np.asarray(lats, float)
    t = np.asarray(times, dtype='datetime64[s]').astype(float)
    land = _lsm_bilinear(lats, lons) >= 0.5
    d2l = np.zeros(len(lats))
    for i in np.flatnonzero(~land):
        g = np.arange(-box_deg, box_deg + 1e-9, 0.25)
        la, lo = np.meshgrid(lats[i] + g, lons[i] + g/np.cos(np.radians(min(abs(lats[i]), 80.0))), indexing='ij')
        ok = (np.abs(la) <= 90) & (_lsm_bilinear(np.clip(la, -90, 90), lo) >= 0.5)
        d2l[i] = min(haversine_vectorized(lats[i], lons[i], la[ok], lo[ok]).min(), cap_km) if ok.any() else cap_km
    last_wet = np.maximum.accumulate(np.where(~land, t, -np.inf))
    t_over = np.where(np.isfinite(last_wet), t - last_wet, t - t[0])/3600.0
    return d2l, t_over

def storm_geometry(lon_c, lat_c, heading_deg, lons, lats):
    """Distance (km) of the points from the centre and their bearing from the motion (0 ahead, 90 right; SH mirrored)."""
    r_km = haversine_vectorized(lat_c, lon_c, lats, lons)
    azi_rel = (bearing_met(lat_c, lon_c, lats, lons) - heading_deg) % 360
    return r_km, np.where(np.asarray(lat_c) < 0, (-azi_rel) % 360, azi_rel)

def inputs(r_km, azi_strong, tc, lf, wind, ter):
    """
    Inputs of the correction at points (1-D arrays of the same length): distance (km) and bearing from the motion
    (storm_geometry); tc: dict of vmax, rmax, speed, lat, d2l, t_overland (track values, scalars or arrays);
    lf: land_factors() at the points; wind: physics wind (m/s); ter: dict of coast_km, z_dem, std5, tpi1 at the points.
    """
    n = len(np.atleast_1d(r_km))
    b = lambda v: np.broadcast_to(np.asarray(v, float), (n,))
    return pd.DataFrame({'r_rmax': b(r_km)/b(tc['rmax']), 'r_km': b(r_km), 'azi_strong': b(azi_strong),
                         'tc_vmax': b(tc['vmax']), 'tc_speed': b(tc['speed']), 'abs_lat': np.abs(b(tc['lat'])),
                         't_overland_h': b(tc['t_overland']), 'tc_d2l': b(tc['d2l']), 'rmax': b(tc['rmax']),
                         'ln_wind': np.log(np.maximum(b(wind), 0.5)),
                         'z0_up': b(lf['z0_up']), 'water_up': b(lf['water_up']), 'coast_km': b(ter['coast_km']),
                         'z_dem': b(ter['z_dem']), 'std5': b(ter['std5']), 'tpi1': b(ter['tpi1']),
                         'sx1_up': b(lf['sx1_up']), 'm_topo': b(lf['m_topo']),
                         'm_terrain': b(lf['m_rough'])*b(lf['m_fetch'])})

def factor(X):
    """Correction factor f (observed / physics) at the rows of X (inputs()), limited to F_LIMITS."""
    return np.clip(_booster().predict(X[INPUTS]), *F_LIMITS)

def corrected_speed(spd, lf, terrain, lons2d, lats2d, lon_c, lat_c, heading_deg, tc):
    """
    Physics wind speed spd (lat, lon) times the land correction and the peak factor at one time step (land weight;
    points with no land weight or wind <= 0.5 m/s unchanged). tc: track values at this step (vmax, rmax, speed, lat,
    d2l, t_overland).
    """
    sel = (lf['w'] > 0) & (spd > 0.5)
    if not sel.any():
        return spd
    r_km, azi = storm_geometry(lon_c, lat_c, heading_deg, lons2d[sel], lats2d[sel])
    X = inputs(r_km, azi, tc, {k: v[sel] for k, v in lf.items()}, spd[sel],
               {k: terrain[k][sel] for k in ('coast_km', 'z_dem', 'std5', 'tpi1')})
    w = lf['w'][sel]
    out = spd.copy()
    out[sel] = spd[sel]*(1 + w*(factor(X) - 1))*(1 + w*(PEAK_FACTOR - 1))
    return out

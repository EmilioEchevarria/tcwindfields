import numpy as np
from ._utils import haversine_vectorized, coriolis
from ._cle15 import ER11E04_nondim_rmaxinput


_CLE15_DEFAULTS = dict(Cdvary=0, C_d=1.5e-3, w_cool=2e-3, CkCdvary=0, CkCd=1,
                        eye_adj=0, alpha_eye=0.15)


def cle15_wind_profile(vmax_ms, rmax_km, lat, **kwargs):
    """
    Compute CLE15 (Chavas, Lin & Emanuel 2015) radial wind profile.

    Parameters
    ----------
    vmax_ms : float, m/s
    rmax_km : float, km
    lat : float, degrees (sign gives hemisphere)

    Returns
    -------
    rr_km : 1-D array, km
    VV_ms : 1-D array, m/s
    r0_km : float, km
    """
    p = {**_CLE15_DEFAULTS, **kwargs}
    fcor = coriolis(lat)
    if fcor == 0:
        fcor = 1e-5  # equatorial fallback

    rr_m, VV_ms, r0_m, _, _ = ER11E04_nondim_rmaxinput(
        vmax_ms, rmax_km*1000, fcor,
        p['Cdvary'], p['C_d'], p['w_cool'],
        p['CkCdvary'], p['CkCd'],
        p['eye_adj'], p['alpha_eye'],
    )
    return rr_m/1000, VV_ms, r0_m/1000


def wind_field_2d(rr_km, VV_ms, tc_lon, tc_lat, storm_spd_ms, storm_ang_math,
                   lons_grid, lats_grid):
    """
    Project a radial wind profile onto a 2-D Cartesian grid, adding storm-motion asymmetry.

    Parameters
    ----------
    rr_km : 1-D array, km
    VV_ms : 1-D array, m/s
    tc_lon, tc_lat : float, degrees
    storm_spd_ms : float, m/s (translation speed)
    storm_ang_math : float, degrees (mathematical convention: 0=East, 90=North)
    lons_grid, lats_grid : 1-D arrays, degrees

    Returns
    -------
    U, V : 2-D arrays, m/s (shape = [len(lats_grid), len(lons_grid)])
    """
    valid = rr_km != 0
    rr = rr_km[valid]
    vv = VV_ms[valid]
    max_rr = np.nanmax(rr)

    # Background wind asymmetry (0.55 * storm speed, 20 deg left of motion)
    # This was done following Wang, Lin and Gori (2022): 
    # "Investigation of Tropical Cyclone Wind Models With Application to Storm Tide Simulations"
    wnd_back = 0.55*storm_spd_ms
    theta_back_rad = np.deg2rad(storm_ang_math - 20)
    u_back = np.cos(theta_back_rad)*wnd_back
    v_back = np.sin(theta_back_rad)*wnd_back

    lons_mg, lats_mg = np.meshgrid(lons_grid, lats_grid)
    dists = haversine_vectorized(tc_lat, tc_lon, lats_mg, lons_mg)

    in_storm = dists < max_rr
    idx = np.searchsorted(rr, dists[in_storm])
    idx = np.clip(idx, 0, len(vv)-1)

    winds_grid = np.zeros_like(dists)
    winds_grid[in_storm] = vv[idx]

    # Tangential direction (CCW unit vector from TC center)
    dlons = lons_mg - tc_lon
    dlats = lats_mg - tc_lat
    thetas = np.arctan2(dlats, dlons)
    theta_tang = thetas + np.pi/2

    decay = np.clip((1 - dists/max_rr)**2, 0, 1)

    # Cyclone rotation: CCW in NH (lat>0), CW in SH (lat<0)
    # Using: U = u_back*decay + sign(lat)*winds*cos(theta_tang)
    rot = np.sign(tc_lat) if tc_lat != 0 else 1.0
    U = u_back*decay + rot*winds_grid*np.cos(theta_tang)
    V = v_back*decay + rot*winds_grid*np.sin(theta_tang)

    U[winds_grid < 0.01] = 0
    V[winds_grid < 0.01] = 0
    return U, V

import numpy as np
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter1d
from ._utils import haversine, coriolis


def beta_model(times, lats, pres_hpa, storm_spd_ms):
    """
    Empirical Holland B parameter.

    Parameters
    ----------
    times : array of np.datetime64
    lats : array, degrees
    pres_hpa : array, hPa
    storm_spd_ms : array, m/s

    Returns
    -------
    beta : array
    """
    times = np.asarray(times, dtype='datetime64[ns]')
    pres_hpa = np.asarray(pres_hpa, dtype=float)
    lats = np.asarray(lats, dtype=float)
    storm_spd_ms = np.asarray(storm_spd_ms, dtype=float)

    eP = 1013.0
    dP = pres_hpa - eP
    dP[dP < 1] = 1
    x = 0.6*(1 - dP/215)

    dPdt = np.zeros(len(pres_hpa))
    for i in range(1, len(pres_hpa)):
        dt_h = float((times[i] - times[i-1]).astype('timedelta64[s]').astype(float))/3600.0
        dPdt[i-1] = (pres_hpa[i] - pres_hpa[i-1])/dt_h if dt_h > 0 else 0.0
    dPdt[-1] = dPdt[-2]

    beta = (-4.4e-5*(dP**2) + 0.01*dP + 0.03*dPdt
            - 0.014*np.abs(lats) + 0.15*(storm_spd_ms**x) + 1)
    return beta


def calc_pressure_profile(lat, pmin_hpa, vmax_ms, rmax_km, beta):
    """
    Holland radial pressure profile from TC center to outer radius.

    Returns
    -------
    rr_km : 1-D array, km
    Pr_hpa : 1-D array, hPa
    """
    p_env = 1010.0
    f = coriolis(lat)
    Ma = (rmax_km*1000*vmax_ms) + 0.5*f*(rmax_km*1000)**2
    r0_km = np.sqrt((2*Ma)/np.abs(f))/1000
    if r0_km < rmax_km:
        r0_km = rmax_km

    r_arr = np.linspace(0, rmax_km, 51)
    Pr = np.zeros(len(r_arr))
    for i in range(len(r_arr)):
        if r_arr[i] == 0:
            Pr[i] = pmin_hpa
        else:
            Pr[i] = pmin_hpa + (p_env - pmin_hpa)*np.exp(-(rmax_km/r_arr[i])**beta)

    def _outer(r, p_rmax, r_max, r0):
        k = 3
        return 1010 - (1010 - p_rmax)*np.exp(-k*(r - r_max)/(r0 - r_max))

    dr = np.diff(r_arr).mean()
    rr_extended = np.arange(0, r0_km+1e-9, dr)
    idx0 = int(np.nanargmin(np.abs(rr_extended - r_arr[-1]))) + 1
    Pr_extended = np.zeros(len(rr_extended))
    Pr_extended[:idx0] = Pr
    Pr_extended[idx0:] = _outer(rr_extended[idx0:], Pr[-1], rmax_km, r0_km)
    Pr_extended[max(0, idx0-3):idx0+3] = gaussian_filter1d(
        Pr_extended[max(0, idx0-3):idx0+3], sigma=1)

    return rr_extended, Pr_extended


def pressure_field_2d(lon_c, lat_c, rr_km, Pr_hpa, lons_grid, lats_grid):
    """
    Interpolate radial pressure profile onto a 2-D grid.

    Returns 2-D array of pressure in hPa with shape [len(lats_grid), len(lons_grid)].
    """
    lon_grid, lat_grid = np.meshgrid(lons_grid, lats_grid)
    dist = haversine(lat_c, lon_c, lat_grid, lon_grid)
    f_interp = interp1d(rr_km, Pr_hpa, bounds_error=False, fill_value=1010.0)
    return f_interp(dist)

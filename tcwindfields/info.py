def input_units():
    """Print the required units for all inputs to compute_tc_fields."""
    print("""
tcwindfields.compute_tc_fields -- required input units
======================================================

Track arrays (one value per time step):
  times     : np.datetime64  (e.g. np.datetime64('2006-04-18T12:00'))
  lons      : degrees East   (float, -180 to 360)
  lats      : degrees North  (float, negative for Southern Hemisphere)
  vmax      : m/s            (10-min mean maximum surface wind speed)
  pmin      : hPa            (minimum central pressure; e.g. 910.0)
  rmax      : km             (radius of maximum wind)

Output grid (1-D arrays):
  lons_grid : degrees East
  lats_grid : degrees North

Optional keyword:
  interp_interval : str  pandas duration string, e.g. '20min', '1h', '30min'
                         Set to None to use track data without interpolation.

Outputs (xr.Dataset):
  uwnd  : m/s   zonal wind        (time, lat, lon)
  vwnd  : m/s   meridional wind   (time, lat, lon)
  pres  : Pa    surface pressure  (time, lat, lon)

CLE15 model default parameters (can be overridden via **kwargs):
  Cdvary   = 0       (0 = fixed C_d; 1 = Donelan et al. 2004 variable)
  C_d      = 1.5e-3  (drag coefficient, used when Cdvary=0)
  w_cool   = 2e-3    (radiative subsidence rate, m/s)
  CkCdvary = 0       (0 = fixed Ck/Cd; 1 = Chavas et al. 2015 quadratic fit)
  CkCd     = 1.0     (enthalpy-to-momentum exchange coefficient ratio)
  eye_adj  = 0       (0 = standard ER11 eye profile)
  alpha_eye= 0.15    (eye adjustment exponent, used when eye_adj=1)
""")

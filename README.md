# tcwindfields

Parametric 2D wind and pressure fields for tropical cyclone tracks.

Given a TC track (position, intensity, size), `tcwindfields` produces gridded
`uwnd`, `vwnd` [m/s] and `pres` [Pa] fields as an `xarray.Dataset` ready to
save as NetCDF.

**Wind models:** Chavas, Lin & Emanuel (2015) CLE15 (default), or Holland (1980) 
(selectable via `wind_model='cle15'` / `wind_model='holland'`).

**Pressure model:** Holland (1980) radial profile with the Willoughby & Rahn (2004) empirical B parameter.

<p align="center">
  <img src="https://raw.githubusercontent.com/EmilioEchevarria/tcwindfields/main/assets/TC_Alfred_winds.gif" width="70%"/>
</p>

---

## Installation

```bash
pip install tcwindfields
```

**Dependencies:** numpy, scipy, shapely, xarray, pandas, tqdm

---

## Example: TC Alfred (2025)

```python
import urllib.request, pathlib
import numpy as np
import xarray as xr
import tcwindfields as tcwf

# --- Download IBTrACS (last 3 years, ~10 MB) ---
url  = ("https://www.ncei.noaa.gov/data/international-best-track-archive-for-"
        "climate-stewardship-ibtracs/v04r01/access/netcdf/"
        "IBTrACS.last3years.v04r01.nc")
dest = pathlib.Path("IBTrACS.last3years.v04r01.nc")
if not dest.exists():
    urllib.request.urlretrieve(url, dest)

# --- Extract Alfred ---
ds_ib   = xr.open_dataset(dest)
names   = np.array(ds_ib['name'].values.astype(str))
seasons = np.array(ds_ib['season'].values.astype(int))

idx    = int(np.where((names == 'ALFRED') & (seasons == 2025))[0][0])
ds_tc  = ds_ib.isel(storm=idx)

lons_tc = np.array(ds_tc['lon'])
lats_tc = np.array(ds_tc['lat'])
time_tc = np.array(ds_tc['time'])
vmax_tc = np.array(ds_tc['bom_wind']) / 1.94384 # knots --> m/s
pmin_tc = np.array(ds_tc['bom_pres'])           # hPa
rmax_tc = np.array(ds_tc['bom_rmw'])  * 1.852   # n mi  --> km

# Fill RMW gaps with Willoughby & Rahn (2004)
rmax_tc = tcwf.fill_rmax_gaps(rmax_tc, vmax_tc, lats_tc)

# Drop rows with missing position / intensity
valid = ~(np.isnan(lons_tc) | np.isnan(lats_tc) | np.isnan(vmax_tc) | np.isnan(pmin_tc))
lons_tc, lats_tc, time_tc = lons_tc[valid], lats_tc[valid], time_tc[valid]
vmax_tc, pmin_tc, rmax_tc = vmax_tc[valid], pmin_tc[valid], rmax_tc[valid]

# Filter to dates of interest
t0 = np.datetime64('2025-02-28T12:00:00')
t1 = np.datetime64('2025-03-08T18:00:00')
mask = (time_tc >= t0) & (time_tc <= t1)
lons_tc, lats_tc, time_tc = lons_tc[mask], lats_tc[mask], time_tc[mask]
vmax_tc, pmin_tc, rmax_tc = vmax_tc[mask], pmin_tc[mask], rmax_tc[mask]

# --- Compute 2D fields ---
lons_grid = np.arange(149.0, 160.0, 0.05)
lats_grid = np.arange(-33.0, -21.0, 0.05)

ds_wnd = tcwf.compute_tc_fields(
    time_tc, lons_tc, lats_tc, vmax_tc, pmin_tc, rmax_tc,
    lons_grid, lats_grid,
    interp_interval='1h',
)
ds_wnd.to_netcdf('Alfred_2025_TC_winds.nc')
```

---

### --> tcwf.compute_tc_fields(...)

Main function. Returns an `xr.Dataset` with dimensions `(time, lat, lon)`.

#### Input variables:

| Parameter | Type | Units | Description |
|-----------|------|-------|-------------|
| `times` | `np.datetime64` array | Track times |
| `lons` | float array | degrees E | TC centre longitude |
| `lats` | float array | degrees N | TC centre latitude (negative = SH) |
| `vmax` | float array | m/s | Maximum surface wind speed |
| `pmin` | float array | hPa | Minimum central pressure |
| `rmax` | float array | km | Radius of maximum wind |
| `lons_grid` | 1-D float array | degrees E | Output grid |
| `lats_grid` | 1-D float array | degrees N | Output grid |
| `interp_interval` | str or None | e.g. `'20min'`, `'1h'`; `None` = no interpolation |
| `wind_model` | str | `'cle15'` (default) or `'holland'` |
| `terrain` | `xr.Dataset` or None (default) | Output of `tcwf.terrain_features` for the same grid |
| `land_correction` | bool | With `terrain`: also apply a statistical land correction (default `False`: physics only) |

#### `Output variables:` 

`uwnd` (m/s), `vwnd` (m/s), `pres` (Pa).


### --> tcwf.terrain_features(lons_grid, lats_grid, terrain_dir, n_procs=1)

With `terrain=...`, `compute_tc_fields` changes the wind over land

- **Surface roughness**: rougher ground (forest, towns) slows the 10m wind more than smooth ground (grass, water). The roughness is taken from the land cover upwind (ESA WorldCover 2021).
- **Land fetch**: the more land the wind has crossed within 50 km upwind, the lower the wind (ERA5 land-sea mask).
- **Elevation** (Copernicus DEM, 90m): speed-up on slopes facing the wind and on crests, slow-down in valleys, channelling along valleys, shelter behind steep upwind slopes, lower winds in rugged terrain.
- **Inflow**: the wind turns towards the centre, by 37° inland and 22° with water upwind.

The coefficients were fitted on station winds (GHCNh) near tropical cyclones, 2000-2025.

`terrain_features` reads the tiles around every grid point once. It is slow, so save the result and reuse it for every storm on that grid. It needs `rasterio` (`pip install tcwindfields[terrain]`) and the tiles in `terrain_dir/worldcover/` and `terrain_dir/cop_dem_90m/` (use `tcwf.download_terrain_tiles` to download them). It stops if a tile with land is missing (`tcwf.missing_tiles` lists them), since a missing tile would be read as sea.

```python
tcwf.download_terrain_tiles(lons_grid, lats_grid, 'terrain_dir')
terr = tcwf.terrain_features(lons_grid, lats_grid, 'terrain_dir', n_procs=8)
terr.to_netcdf('terrain_grid.nc')
ds = tcwf.compute_tc_fields(time_tc, lons_tc, lats_tc, vmax_tc, pmin_tc, rmax_tc,
                            lons_grid, lats_grid, terrain=terr)
```


## Statistical land correction (optional): 
### --> compute_tc_fields(..., terrain=terr, land_correction=True)

On top of the physics above, a factor on the wind over land learned from station observations: gradient-boosted decision trees (LightGBM, model in `assets/land_correction.txt`) trained on GHCNh station winds near TCs (2000-2025). Its 19 inputs come from the track (dist from the centre, bearing, Vmax, Rmax, forward speed, lat, time it's been over land, dist2land, wind) and the terrain (upwind roughness and water, dist2coast, elevation, ruggedness, shelter, the hill and roughness factors). It corrects storm-scale errors over land that the physics does not. The wind over open sea is unchanged. It needs `lightgbm`, and a terrain file from `terrain_features` 


### --> tcwf.fill_rmax_gaps(rmax_km, vmax_ms, lats)

Fill `NaN` values in a `rmax` array using the Willoughby & Rahn (2004)
empirical formula. Observed values are kept unchanged.


### --> tcwf.interpolate_track(times, lons, lats, vmax, pmin, rmax, interval='1h')

Interpolate track arrays to a regular time step. Returns a dict with the same keys. If `interp_interval` is not None, `compute_tc_fields` will use it.


### --> tcwf.input_units()

Print a summary of all required input units and default model parameters.

---

## References

- Chavas, D. R., Lin, N., & Emanuel, K. (2015). A model for the complete radial structure of the tropical cyclone wind field. Part I: Comparison with observed structure. *Journal of the Atmospheric Sciences*, 72(9).3647-3662.
- Emanuel, K., & Rotunno, R. (2011). Self-stratification of tropical cyclone outflow. Part I: Implications for storm structure. *Journal of the Atmospheric Sciences*, 68(10), 2236–2249.
- Emanuel, K. (2004). Tropical cyclone energetics and structure. *Atmospheric Turbulence and Mesoscale Meteorology*, 8, 165-191.
- Holland, G. J. (1980). An analytic model of the wind and pressure profiles in hurricanes. *Monthly Weather Review*, 108(8), 1212-1218.
- Willoughby, H. E., & Rahn, M. E. (2004). Parametric representation of the primary hurricane vortex. Part I: Observations and evaluation of the Holland (1980) model. *Monthly Weather Review*, 132(12), 3033-3048.
- Jackson, P. S., & Hunt, J. C. R. (1975). Turbulent wind flow over a low hill. *Quarterly Journal of the Royal Meteorological Society*, 101(430), 929-955.
- Troen, I., & Petersen, E. L. (1989). *European Wind Atlas*. Risø National Laboratory (WAsP).
- FEMA (2021). *Hazus Hurricane Model Technical Manual* (sea-land roughness transition).
- Zanaga, D., et al. (2022). ESA WorldCover 10 m 2021 v200. doi:10.5281/zenodo.7254221.

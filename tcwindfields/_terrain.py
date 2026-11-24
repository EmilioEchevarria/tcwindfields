"""
Land effects on the 10m wind

  roughness   upwind, distance-weighted roughness length z0 from the land cover (ESA WorldCover 2021)
  land fetch  lower winds the more land the wind has crossed within 50 km upwind (ERA5 land-sea mask)
  topography  speed-up on slopes facing the wind and on crests, slow-down in valleys, channelling along
              valleys (linear potential flow over the Copernicus DEM, Jackson & Hunt 1975), shelter
              behind steep upwind slopes, and lower winds in rugged terrain
  inflow      the wind turns towards the centre: 37 deg inland, 22 deg with water upwind

All quantities are taken in the upwind direction (8 sectors of 45 deg, interpolated between the two nearest).
"""

import pathlib
import urllib.request
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache, partial

import numpy as np
import xarray as xr
from tqdm import tqdm

# ESA WorldCover 2021 v200 classes and roughness lengths z0 (m), literature values (HAZUS land cover table, FEMA 2021; water = z0_sea). 
wc_codes = (10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 100)
wc_names = ('tree', 'shrub', 'grass', 'crop', 'built', 'bare', 'snow', 'water', 'wetland', 'mangrove', 'moss')
z0_class = {'tree': 0.6, 'shrub': 0.12, 'grass': 0.06, 'crop': 0.05, 'built': 1.0, 'bare': 0.01,
            'snow': 0.001, 'water': None, 'wetland': 0.05, 'mangrove': 0.5, 'moss': 0.01}
z0_sea = 0.0013  # m, marine roughness (HAZUS)
h_bl = 600.0  # m, height where the wind is unchanged from sea to land
ring_edges_km = (0.0, 0.1, 0.25, 0.5, 1.0, 2.0, 3.0, 5.0, 10.0)
fetch_x0_m, fetch_xmax_km = 10.0, 20.0  # distance weights ln(x/x0)/ln(xmax/x0) (~60% within 1 km)
land_fetch_km, land_fetch_gamma = 50.0, -0.2824  # wind x exp(gamma x land fraction within 50 km upwind)
topo_coef = {'s_along': 2.8849, 's_across': 0.8860, 'sx_pos': -0.03323, 'rug': -0.04859}  # fitted with gamma
topo_cap = (0.5, 1.71)  # limits of the topographic factor (upper": flow separation, AS/NZS 1170.2)
topo_z_eff, topo_r_avg, topo_half_km = 400.0, 500.0, 10.0    # m, m, km
inflow_sea, inflow_land = 22.0, 37.0  # deg, inflow angle with water upwind / inland
topo_axes = (0.0, 45.0, 90.0, 135.0)  # wind axes
km_per_deg = 111.2

_wc_url = 'https://esa-worldcover.s3.eu-central-1.amazonaws.com/v200/2021/map/ESA_WorldCover_10m_2021_v200_{tile}_Map.tif'
_dem_url = 'https://copernicus-dem-90m.s3.amazonaws.com/{name}/{name}.tif'
lsm_file = pathlib.Path(__file__).resolve().parent.parent/'assets'/'era5_land_sea_mask.nc'
_no_tile = 'no_tile.txt'  # tiles that do not exist on the server (ocean), written by download_terrain_tiles

_lut = np.full(256, wc_names.index('water'), dtype=np.int64)  # no data / unknown codes = water (ocean)
for _k, _c in enumerate(wc_codes):
    _lut[_c] = _k


# Tiles
def worldcover_tile(lat0, lon0):
    """ESA WorldCover 3x3 degree tile name from its lower-left corner, e.g. (-48, 36) --> 'S48E036'."""
    return f"{'N' if lat0 >= 0 else 'S'}{abs(lat0):02d}{'E' if lon0 >= 0 else 'W'}{abs(lon0):03d}"

def cop_dem_tile(lat0, lon0):
    """Copernicus DEM GLO-90 1x1 degree tile name from its south-west corner."""
    return (f"Copernicus_DSM_COG_30_{'N' if lat0 >= 0 else 'S'}{abs(lat0):02d}_00_"
            f"{'E' if lon0 >= 0 else 'W'}{abs(lon0):03d}_00_DEM")

def _wc_file(terrain_dir, lat0, lon0):
    return pathlib.Path(terrain_dir)/'worldcover'/f'ESA_WorldCover_10m_2021_v200_{worldcover_tile(lat0, lon0)}_Map.tif'

def _dem_file(terrain_dir, lat0, lon0):
    return pathlib.Path(terrain_dir)/'cop_dem_90m'/f'{cop_dem_tile(lat0, lon0)}.tif'

def _grid_box(lons_grid, lats_grid, buffer_km=12.0):
    """Lat/lon box of the grid plus the 10-km terrain boxes around its points."""
    dlat = buffer_km/km_per_deg
    la0, la1 = np.min(lats_grid) - dlat, np.max(lats_grid) + dlat
    dlon = buffer_km/(km_per_deg*np.cos(np.radians(min(max(abs(la0), abs(la1)), 85.0))))
    return la0, la1, np.min(lons_grid) - dlon, np.max(lons_grid) + dlon

def _tiles(lons_grid, lats_grid):
    """(folder, file name, url, lat0, lon0, size in deg) of every tile the grid needs."""
    la0, la1, lo0, lo1 = _grid_box(lons_grid, lats_grid)
    out = []
    for a in range(int(np.floor(la0/3)*3), int(np.floor(la1/3)*3) + 1, 3):
        for b in range(int(np.floor(lo0/3)*3), int(np.floor(lo1/3)*3) + 1, 3):
            bw = (b + 180) % 360 - 180
            u = _wc_url.format(tile=worldcover_tile(a, bw))
            out.append(('worldcover', u.rsplit('/', 1)[1], u, a, bw, 3))
    for a in range(int(np.floor(la0)), int(np.floor(la1)) + 1):
        for b in range(int(np.floor(lo0)), int(np.floor(lo1)) + 1):
            bw = (b + 180) % 360 - 180
            u = _dem_url.format(name=cop_dem_tile(a, bw))
            out.append(('cop_dem_90m', u.rsplit('/', 1)[1], u, a, bw, 1))
    return out

def terrain_tiles(lons_grid, lats_grid):
    """URLs of the WorldCover and Copernicus DEM tiles needed for a grid (tiles over the open ocean do not exist)."""
    urls = {'worldcover': [], 'cop_dem_90m': []}
    for sub, _, u, *_ in _tiles(lons_grid, lats_grid):
        urls[sub].append(u)
    return urls

def missing_tiles(lons_grid, lats_grid, terrain_dir):
    """
    Tiles the grid needs that are not in terrain_dir although the ERA5 land-sea mask has land in them
    """
    la0, la1, lo0, lo1 = _grid_box(lons_grid, lats_grid)
    m = xr.open_dataset(lsm_file)['lsm']
    lsm, la_g, lo_g = m.values, m['latitude'].values, m['longitude'].values
    out = []
    for sub, name, _, a, b, size in _tiles(lons_grid, lats_grid):
        d = pathlib.Path(terrain_dir)/sub
        if (d/name).exists():
            continue
        f_no = d/_no_tile
        if f_no.exists() and name in f_no.read_text().split():
            continue
        ya = (la_g >= max(a, la0)) & (la_g <= min(a + size, la1))
        xb = ((lo_g - b) % 360 <= size) & ((lo_g - lo0) % 360 <= lo1 - lo0)
        if ya.any() and xb.any() and np.nanmax(lsm[np.ix_(ya, xb)]) > 0:
            out.append(f'{sub}/{name}')
    return out

def download_terrain_tiles(lons_grid, lats_grid, terrain_dir):
    """
    Download the tiles of terrain_tiles() that are not yet in terrain_dir/worldcover and terrain_dir/cop_dem_90m.
    Tiles that do not exist on the server (ocean) are listed in <folder>/no_tile.txt.
    """
    for sub, name, u, *_ in _tiles(lons_grid, lats_grid):
        d = pathlib.Path(terrain_dir)/sub
        d.mkdir(parents=True, exist_ok=True)
        f = d/name
        if f.exists():
            continue
        try:
            print(f"downloading {name} ...")
            urllib.request.urlretrieve(u, f)
        except urllib.error.HTTPError as e:  # 404: ocean tile, does not exist
            if e.code != 404:
                raise
            with open(d/_no_tile, 'a') as fo:
                fo.write(name + '\n')

@lru_cache(maxsize=64)
def _open(path):
    import rasterio
    return rasterio.open(path)

def read_box(lat, lon, half_km, tile_file, step, fill, res_default):
    """
    Raster values on a regular lat/lon grid covering +-half_km around (lat, lon), mosaicked from the step x step
    degree tiles tile_file(lat0, lon0) (nearest pixel; missing tiles = fill; dateline-safe).
    Returns values (ny, nx), lat centres (north to south), lon centres.
    """
    from rasterio.windows import Window
    dlat = half_km/km_per_deg
    dlon = half_km/(km_per_deg*np.cos(np.radians(min(abs(lat), 85.0))))
    f0 = tile_file(int(np.floor(lat/step)*step), int(np.floor(lon/step)*step))
    res_x, res_y = _open(str(f0)).res if f0.exists() else (res_default, res_default)
    j0, j1 = int(np.floor((lon - dlon)/res_x)), int(np.ceil((lon + dlon)/res_x))
    i0, i1 = int(np.floor((lat - dlat)/res_y)), int(np.ceil((lat + dlat)/res_y))
    lon_c = (np.arange(j0, j1) + 0.5)*res_x
    lat_c = (np.arange(i1, i0, -1) - 0.5)*res_y
    lat_c = lat_c[np.abs(lat_c) < 90]
    out = None
    for a in range(int(np.floor(lat_c.min()/step)*step), int(np.floor(lat_c.max()/step)*step) + 1, step):
        for b in range(int(np.floor(lon_c.min()/step)*step), int(np.floor(lon_c.max()/step)*step) + 1, step):
            bw = (b + 180) % 360 - 180
            f = tile_file(a, bw)
            if not f.exists():
                continue
            src = _open(str(f))
            t = src.transform
            r = np.floor((t.f - lat_c)/(-t.e)).astype(int)
            c = np.floor((lon_c - (t.c + b - bw))/t.a).astype(int)
            iv = np.flatnonzero((r >= 0) & (r < src.height))
            jv = np.flatnonzero((c >= 0) & (c < src.width))
            if iv.size == 0 or jv.size == 0:
                continue
            r0, r1, c0, c1 = r[iv].min(), r[iv].max(), c[jv].min(), c[jv].max()
            for k in range(3):  # some occasional read errors (not sure why): reopen and retry
                try:
                    sub = src.read(1, window=Window(c0, r0, c1 - c0 + 1, r1 - r0 + 1))
                    break
                except Exception:
                    if k == 2:
                        raise
                    _open.cache_clear()
                    src = _open(str(f))
            if src.nodata is not None:
                sub = np.where(sub == src.nodata, fill, sub)
            if out is None:
                out = np.full((lat_c.size, lon_c.size), fill, dtype=sub.dtype)
            out[np.ix_(iv, jv)] = sub[np.ix_(r[iv] - r0, c[jv] - c0)]
    if out is None:
        out = np.full((lat_c.size, lon_c.size), fill)
    return out, lat_c, lon_c


# Terrain at one point
def _polar_index(lat_c, lon_c, lat, lon, edges_km, n_sectors):
    """Sector and ring of each pixel (ring = -1 outside the last ring) and distance (km)."""
    dy = ((lat_c - lat)*km_per_deg).astype(np.float32)
    dx = ((lon_c - lon)*km_per_deg*np.cos(np.radians(lat))).astype(np.float32)
    dist = np.hypot(dy[:, None], dx[None, :])
    bear = np.degrees(np.arctan2(dx[None, :], dy[:, None])) % 360
    width = 360/n_sectors
    sector = (((bear + width/2) % 360)//width).astype(np.int64) % n_sectors
    ring = np.searchsorted(np.asarray(edges_km, np.float32), dist, side='right') - 1
    ring[dist >= edges_km[-1]] = -1
    return sector, ring, dist

def _landcover_cells(lat, lon, terrain_dir, edges_km=ring_edges_km, n_sectors=8):
    """WorldCover class fractions (sector, ring, class) and pixel counts (sector, ring)."""
    vals, lat_c, lon_c = read_box(lat, lon, edges_km[-1], partial(_wc_file, terrain_dir), step=3, fill=80, res_default=1/12000)
    sector, ring, _ = _polar_index(lat_c, lon_c, lat, lon, edges_km, n_sectors)
    ok = ring >= 0
    n_r, n_c = len(edges_km) - 1, len(wc_codes)
    idx = (sector[ok]*n_r + ring[ok])*n_c + _lut[vals[ok]]
    counts = np.bincount(idx, minlength=n_sectors*n_r*n_c).reshape(n_sectors, n_r, n_c)
    n = counts.sum(axis=2)
    return counts/np.maximum(n, 1)[..., None], n

def _dem_cells(lat, lon, terrain_dir, edges_km=ring_edges_km, n_sectors=8):
    """By (sector, ring), elevation relative to the point: pixel count, mean, std and max slope angle (deg)."""
    z, lat_c, lon_c = read_box(lat, lon, edges_km[-1], partial(_dem_file, terrain_dir), step=1, fill=0.0, res_default=1/1200)
    z = z.astype(np.float64)
    sector, ring, dist = _polar_index(lat_c, lon_c, lat, lon, edges_km, n_sectors)
    i_pt, j_pt = np.argmin(np.abs(lat_c - lat)), np.argmin(np.abs(lon_c - lon))
    own = np.zeros(z.shape, dtype=bool)
    own[i_pt, j_pt] = True
    ok = ring >= 0
    n_r = len(edges_km) - 1
    m = n_sectors*n_r
    cell = sector[ok]*n_r + ring[ok]
    dz, d = z[ok] - z[i_pt, j_pt], dist[ok].astype(np.float64)
    n = np.bincount(cell, minlength=m)
    mean = np.bincount(cell, weights=dz, minlength=m)/np.maximum(n, 1)
    var = np.bincount(cell, weights=dz**2, minlength=m)/np.maximum(n, 1) - mean**2
    far = ~own[ok]
    amax = np.full(m, -np.inf)
    np.maximum.at(amax, cell[far], np.degrees(np.arctan2(dz[far], d[far]*1000)))
    empty = n == 0
    out = []
    for v in (mean.astype(np.float32), np.sqrt(np.maximum(var, 0)).astype(np.float32), amax.astype(np.float32)):
        v[empty | ~np.isfinite(v)] = np.nan
        out.append(v.reshape(n_sectors, n_r))
    return (n.reshape(n_sectors, n_r), *out, float(z[i_pt, j_pt]))

def _detrend_taper(z, alpha=0.25):
    """z minus its best-fit plane, tapered to 0 at the edges (Tukey window)."""
    ny, nx = z.shape
    yy, xx = np.mgrid[0:ny, 0:nx]
    A = np.column_stack([np.ones(z.size), xx.ravel(), yy.ravel()])
    c = np.linalg.lstsq(A, z.ravel(), rcond=None)[0]
    r = z - (A @ c).reshape(ny, nx)

    def tukey(n):
        w = np.ones(n)
        m = int(alpha*n/2)
        if m > 0:
            w[:m] = 0.5*(1 - np.cos(np.pi*np.arange(m)/m))
            w[-m:] = w[:m][::-1]
        return w
    return r*np.outer(tukey(ny), tukey(nx))

def speedup_fields(z, dx_m, dy_m, axes_deg=topo_axes, z_eff=topo_z_eff):
    """
    Surface speed-up of potential flow over the terrain, for winds along each axis (bearing): 
    s_along = F^-1[(k.n)^2/|k| h^ exp(-|k| z_eff)] (+ hills, - valleys), s_across with the
    across-wind wavenumber (+ ridges along the wind, - valleys along the wind: channelling)
    """
    ny, nx = z.shape
    py, px = int(2**np.ceil(np.log2(2*ny))), int(2**np.ceil(np.log2(2*nx)))
    h = np.zeros((py, px))
    h[:ny, :nx] = _detrend_taper(np.asarray(z, float))
    H = np.fft.fft2(h)
    kx = 2*np.pi*np.fft.fftfreq(px, d=dx_m)[None, :]
    ky = 2*np.pi*np.fft.fftfreq(py, d=dy_m)[:, None]
    kk = np.hypot(kx, ky)
    damp = np.exp(-kk*z_eff)*np.where(kk > 0, 1/np.where(kk > 0, kk, 1), 0.0)
    sa, sc = np.zeros((len(axes_deg), ny, nx)), np.zeros((len(axes_deg), ny, nx))
    for j, th in enumerate(np.radians(axes_deg)):
        kn = kx*np.sin(th) + ky*np.cos(th)
        kt = kx*np.cos(th) - ky*np.sin(th)
        sa[j] = np.fft.ifft2(kn**2*damp*H).real[:ny, :nx]
        sc[j] = np.fft.ifft2(kt**2*damp*H).real[:ny, :nx]
    return sa, sc

def _point_speedup(lat, lon, terrain_dir):
    """s_along, s_across (n_axes) averaged within topo_r_avg of the point."""
    z, lat_c, lon_c = read_box(lat, lon, topo_half_km, partial(_dem_file, terrain_dir), step=1, fill=0.0, res_default=1/1200)
    z = np.where(np.isfinite(z), z, 0.0).astype(float)[::-1]
    lat_c = lat_c[::-1]
    dy = abs(lat_c[1] - lat_c[0])*111.195*1000
    dx = abs(lon_c[1] - lon_c[0])*111.195*1000*np.cos(np.radians(lat))
    i, j = int(np.argmin(np.abs(lat_c - lat))), int(np.argmin(np.abs(((lon_c - lon) + 180) % 360 - 180)))
    sa, sc = speedup_fields(z, dx, dy)
    yy, xx = np.mgrid[0:z.shape[0], 0:z.shape[1]]
    m = np.hypot((yy - i)*dy, (xx - j)*dx) <= max(topo_r_avg, 0.6*max(dx, dy))
    return sa[:, m].mean(-1), sc[:, m].mean(-1)

def _fetch_weights(edges_km):
    """Weight of each distance ring in the upwind average (transition ln(x/x0)/ln(xmax/x0), as HAZUS)."""
    e = np.asarray(edges_km, dtype=float)*1000
    F = np.clip(np.log(np.maximum(e, fetch_x0_m)/fetch_x0_m)/np.log(fetch_xmax_km*1000/fetch_x0_m), 0, 1)
    w = np.diff(F)
    return w/w.sum()

def _one_point(args):
    """
    Upwind features of one point (sector-wise), its land weight, DEM elevation, height above the mean within 1 km
    (TPI) and whether the point itself is on land.
    """
    lat, lon, terrain_dir = args
    frac, n = _landcover_cells(lat, lon, terrain_dir)
    dn, mu, sd, amax, z_dem = _dem_cells(lat, lon, terrain_dir)
    sa, sc = _point_speedup(lat, lon, terrain_dir)
    edges = np.asarray(ring_edges_km)
    outer = edges[1:]
    w = _fetch_weights(edges)
    iw = wc_names.index('water')
    ln_z0c = np.log([z0_sea if z0_class[k] is None else z0_class[k] for k in wc_names])
    ln_z0 = ((frac*ln_z0c).sum(-1)*w).sum(-1)  # (sector)
    water_up = (frac[..., iw]*w).sum(-1)
    land_point = frac[:, 0, iw].mean() < 0.5
    land10 = 1 - (frac[..., iw]*n).sum()/max(n.sum(), 1)
    land_weight = 1.0 if land_point else land10
    sel = outer <= 1.0 + 1e-9  # Sx within 1 km
    with np.errstate(all='ignore'):
        sx1 = np.nanmax(np.where(np.isfinite(amax[:, sel]), amax[:, sel], -np.inf), axis=1)
    sx1[~np.isfinite(sx1)] = np.nan
    sel = outer <= 5.0 + 1e-9  # relief std within 5 km (all directions)
    dnf, muf, sdf = dn[:, sel].astype(float), np.nan_to_num(mu[:, sel]), np.nan_to_num(sd[:, sel])
    N = dnf.sum()
    m1 = (muf*dnf).sum()/max(N, 1)
    std5 = np.sqrt(max(((sdf**2 + muf**2)*dnf).sum()/max(N, 1) - m1**2, 0)) if N > 0 else np.nan
    sel = outer <= 1.0 + 1e-9  # TPI within 1 km: point elevation minus the mean around it
    N1 = dn[:, sel].sum()
    tpi1 = -(np.nan_to_num(mu[:, sel])*dn[:, sel]).sum()/max(N1, 1) if N1 > 0 else np.nan
    sec = np.arange(8) % len(topo_axes)
    return ln_z0, water_up, land_weight, sx1, std5, sa[sec], sc[sec], z_dem, tpi1, land_point

@lru_cache(maxsize=1)
def _lsm():
    """ERA5 land-sea mask (lsm_file): latitudes (ascending), longitudes, field."""
    m = xr.open_dataset(lsm_file)['lsm'].load()
    return m['latitude'].values.astype(float), m['longitude'].values.astype(float), m.values.astype(float)

def _lsm_bilinear(plat, plon):
    """ERA5 land-sea mask (lsm_file) bilinearly interpolated at points."""
    la_g, lo_g, field = _lsm()
    fy = (np.asarray(plat, float) - la_g[0])/(la_g[1] - la_g[0])
    j0 = np.clip(np.floor(fy).astype(int), 0, len(la_g) - 2)
    wy = np.clip(fy - j0, 0.0, 1.0)
    fx = ((np.asarray(plon, float) - lo_g[0]) % 360.0)/(lo_g[1] - lo_g[0])
    i0 = np.floor(fx).astype(int) % len(lo_g)
    i1, wx = (i0 + 1) % len(lo_g), fx - np.floor(fx)
    return (1 - wy)*((1 - wx)*field[j0, i0] + wx*field[j0, i1]) + wy*((1 - wx)*field[j0 + 1, i0] + wx*field[j0 + 1, i1])

def _lsm_coast_km(lat, lon, land_point, max_km=300.0, step_km=5.0):
    """Distance (km) to the sea (land-sea mask < 0.5) along 8 directions, 0 on water points, max_km + step if none."""
    dists = np.arange(step_km, max_km + 0.1*step_km, step_km)
    prof = _lsm_profile(lat, lon, dists, rays=3).mean(axis=2)  # (point, sector, dist)
    sea = prof < 0.5
    first = np.where(sea.any(-1), dists[np.argmax(sea, axis=-1)], max_km + step_km)
    return np.where(land_point, first.min(axis=1), 0.0)

def _lsm_land_fraction(lat, lon, radius_km=land_fetch_km, step_km=5.0):
    """Mean land fraction (ERA5 land-sea mask) within radius_km upwind of each point: (point, sector)."""
    dists = np.arange(step_km, radius_km + 0.1*step_km, step_km)
    return _lsm_profile(lat, lon, dists).mean(axis=(2, 3))

def _lsm_profile(lat, lon, dists, n_sectors=8, rays=5):
    """Land-sea mask along `rays` bearings across each upwind sector, at distances dists (km): (point, sector, ray, dist)."""
    width = 360.0/n_sectors
    brg = np.arange(n_sectors)[:, None]*width + (np.arange(rays) + 0.5)/rays*width - width/2
    th = np.deg2rad(brg)[None, :, :, None]
    dl = (np.asarray(dists, float)/6371.0)[None, None, None, :]
    p1, l1 = np.deg2rad(np.asarray(lat, float))[:, None, None, None], np.deg2rad(np.asarray(lon, float))[:, None, None, None]
    p2 = np.arcsin(np.sin(p1)*np.cos(dl) + np.cos(p1)*np.sin(dl)*np.cos(th))
    l2 = l1 + np.arctan2(np.sin(th)*np.sin(dl)*np.cos(p1), np.cos(dl) - np.sin(p1)*np.sin(p2))
    return _lsm_bilinear(np.rad2deg(p2).ravel(), np.rad2deg(l2).ravel()).reshape(p2.shape)

def terrain_features(lons_grid, lats_grid, terrain_dir, n_procs=1):
    """
    Upwind terrain features of a regular grid, for compute_tc_fields(..., terrain=...). 
    It is slow! It reads 10km boxes of 10m land cover and 90m DEM around grid points)
    Run it once per grid and save it (to_netcdf).

    terrain_dir: folder with worldcover/ (ESA WorldCover 2021 v200 tiles) and cop_dem_90m/ (Copernicus GLO-90)
    Returns xr.Dataset (lat, lon[, sector]): ln_z0, water_up, land_up, s_along, s_across, sx1 (sector, wind FROM), std5, land_weight.
    """
    miss = missing_tiles(lons_grid, lats_grid, terrain_dir)
    if miss:
        raise FileNotFoundError(f"{len(miss)} terrain tiles with land are missing in {terrain_dir} (e.g. {miss[:3]}): "
                                "download them with tcwf.download_terrain_tiles(lons_grid, lats_grid, terrain_dir)")
    lons_grid, lats_grid = np.asarray(lons_grid, float), np.asarray(lats_grid, float)
    LO, LA = np.meshgrid(lons_grid, lats_grid)
    lat, lon = LA.ravel(), (LO.ravel() + 180) % 360 - 180
    order = np.lexsort((np.floor(lon/3), np.floor(lat/3)))  # by tile, so workers re-use open tiles
    args = [(lat[i], lon[i], str(terrain_dir)) for i in order]
    if n_procs > 1:
        _open.cache_clear()
        with ProcessPoolExecutor(n_procs) as ex:
            res = list(tqdm(ex.map(_one_point, args, chunksize=8), total=len(args), desc='Terrain'))
    else:
        res = [_one_point(a) for a in tqdm(args, desc='Terrain')]
    res = [res[i] for i in np.argsort(order)]
    shp = LO.shape
    g3, g2 = ('lat', 'lon', 'sector'), ('lat', 'lon')
    ds = xr.Dataset({'ln_z0': (g3, np.stack([r[0] for r in res]).reshape(shp + (8,))),
                     'water_up': (g3, np.stack([r[1] for r in res]).reshape(shp + (8,))),
                     'land_weight': (g2, np.array([r[2] for r in res]).reshape(shp)),
                     'sx1': (g3, np.stack([r[3] for r in res]).reshape(shp + (8,))),
                     'std5': (g2, np.array([r[4] for r in res]).reshape(shp)),
                     's_along': (g3, np.stack([r[5] for r in res]).reshape(shp + (8,))),
                     's_across': (g3, np.stack([r[6] for r in res]).reshape(shp + (8,))),
                     'land_up': (g3, _lsm_land_fraction(lat, lon).reshape(shp + (8,))),
                     'z_dem': (g2, np.array([r[7] for r in res]).reshape(shp)),
                     'tpi1': (g2, np.array([r[8] for r in res]).reshape(shp)),
                     'coast_km': (g2, _lsm_coast_km(lat, lon, np.array([r[9] for r in res])).reshape(shp))},
                    coords={'lat': lats_grid, 'lon': lons_grid, 'sector': np.arange(8)*45.0})
    attrs = {'ln_z0': 'ln of the upwind, distance-weighted roughness length (m)',
             'water_up': 'upwind, distance-weighted water fraction',
             'land_weight': 'weight of the land effects: 1 on land, land fraction within 10 km over water',
             'sx1': 'steepest upwind slope angle within 1 km (deg; > 0 sheltered)',
             'std5': 'std of the elevation within 5 km (m; ruggedness)',
             's_along': 'potential-flow speed-up, along the wind', 's_across': 'potential-flow speed-up, across',
             'land_up': f'land fraction within {land_fetch_km:g} km upwind (ERA5 land-sea mask)',
             'z_dem': 'DEM elevation at the point (m)', 'tpi1': 'elevation minus the mean within 1 km (m; > 0 hilltop)',
             'coast_km': 'distance to the sea (km; ERA5 land-sea mask, 0 on water points)'}
    for k, v in attrs.items():
        ds[k].attrs['long_name'] = v
    ds['sector'].attrs = {'units': 'degree', 'long_name': 'upwind sector centre (wind FROM)'}
    return ds


# Applying the land effects at a time step
def roughness_multiplier(z0_land, z0_sea=z0_sea, h=h_bl, z=10.0):
    """Wind at z over roughness z0_land relative to the marine wind (z0_sea), wind at h unchanged (log law)."""
    z0_land = np.asarray(z0_land, dtype=float)
    return (np.log(z/z0_land)/np.log(h/z0_land))*(np.log(h/z0_sea)/np.log(z/z0_sea))

def _at_direction(x, wdir_from):
    """Values of x (lat, lon, sector) for the wind wdir_from (lat, lon), between the two nearest sectors."""
    n = x.shape[-1]
    sf = (wdir_from % 360)/(360/n)
    s0 = np.floor(sf).astype(int) % n
    a = sf - np.floor(sf)
    x0 = np.take_along_axis(x, s0[..., None], -1)[..., 0]
    x1 = np.take_along_axis(x, ((s0 + 1) % n)[..., None], -1)[..., 0]
    return (1 - a)*x0 + a*x1

def land_factors(U, V, terrain, tc_lat):
    """
    Land effects at a time step (arrays shaped like U): wind direction turned inward (wdir, FROM), land weight (w),
    multipliers (m_rough, m_fetch, m_topo), and the upwind z0 (m), water fraction and Sx for that direction.
    terrain: dict of the arrays of terrain_features (lat, lon[, sector]).
    """
    wdir = np.degrees(np.arctan2(-U, -V)) % 360  # meteorological, wind from direction
    w = terrain['land_weight']
    inflow = inflow_sea + (inflow_land - inflow_sea)*(1 - np.clip(_at_direction(terrain['water_up'], wdir), 0, 1))
    d = (wdir - w*inflow*np.sign(tc_lat)) % 360 # NH: turned anticlockwise (towards the centre)
    z0_up = np.exp(_at_direction(terrain['ln_z0'], d))
    sx1_up = _at_direction(terrain['sx1'], d)
    ln_topo = (topo_coef['s_along']*np.nan_to_num(_at_direction(terrain['s_along'], d))
               + topo_coef['s_across']*np.nan_to_num(_at_direction(terrain['s_across'], d))
               + topo_coef['sx_pos']*np.maximum(np.nan_to_num(sx1_up), 0)/10.0
               + topo_coef['rug']*np.nan_to_num(terrain['std5'])/100.0)
    return {'wdir': d, 'w': w, 'm_rough': roughness_multiplier(z0_up),
            'm_fetch': np.exp(land_fetch_gamma*np.nan_to_num(_at_direction(terrain['land_up'], d))),
            'm_topo': np.clip(np.exp(ln_topo), *topo_cap), 'z0_up': z0_up,
            'water_up': _at_direction(terrain['water_up'], d), 'sx1_up': sx1_up}

def land_effects(U, V, terrain, tc_lat, correction=None):
    """
    Wind (U, V: lat x lon, m/s) with the land effects: the direction turned inward (inflow)
    and the speed times roughness x land fetch x topography (weighted by the land weight).
    correction: optional function (speed, land_factors) --> speed, applied after (statistical land correction).
    Returns U, V and the upwind roughness length z0 (m, for the wnd gust).
    """
    lf = land_factors(U, V, terrain, tc_lat)
    spd = np.hypot(U, V)*(1 + lf['w']*(lf['m_rough']*lf['m_fetch']*lf['m_topo'] - 1))
    if correction is not None:
        spd = correction(spd, lf)
    rad = np.radians(lf['wdir'])
    return -spd*np.sin(rad), -spd*np.cos(rad), lf['z0_up']

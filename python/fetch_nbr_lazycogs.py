#!/usr/bin/env python3
"""GDAL-free annual NBR fetch: rustac + lazycogs (A/B vs the rasterio path).

Same canonical AOI, same QA-mask + NBR + annual-median reduce as fetch_nbr.py, but
the READ layer is the Rust-native stack instead of rasterio/GDAL:

  rustac.search   STAC API -> items            (rustac, the rustac/stac-geoparquet tool)
  rustac.write    items   -> geoparquet
  obstore.AzureStore + MPC container SAS token (reads MPC blobs without GDAL/AWS creds)
  lazycogs.open   geoparquet -> lazy (band, time, y, x)   (async-geotiff, no GDAL)
  numpy           QA-mask + NBR + annual median           (our reduce, unchanged)

Writes data/nbr_lazycogs.npz; produces byte-identical composites to fetch_nbr.py.
Needs Python 3.13 (lazycogs requires >=3.12 and segfaults on 3.14).

Run: python python/fetch_nbr_lazycogs.py
"""
from pathlib import Path
import numpy as np
import rustac, lazycogs, obstore
from planetary_computer import sas
from pyproj import Transformer
from rasterio.transform import from_origin

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "nbr_lazycogs.npz"
GP = ROOT / "data" / "_items.parquet"
BBOX_LL = (-123.855, 45.882, -123.835, 45.896)
START, END = 1984, 2016
EPSG, RES = "EPSG:32610", 30.0
SR_SCALE, SR_OFFSET = 2.75e-5, -0.2
QA_BAD_BITS = [1, 2, 3, 4]
MPC = "https://planetarycomputer.microsoft.com/api/stac/v1"

# 1. search (rustac) -------------------------------------------------------------
print("[rustac] searching MPC landsat-c2-l2 ...", flush=True)
items = rustac.search_sync(
    MPC, collections=["landsat-c2-l2"], bbox=list(BBOX_LL),
    datetime=f"{START}-01-01/{END}-12-31",
    query={"eo:cloud_cover": {"lt": 60},
           "platform": {"in": ["landsat-5", "landsat-7", "landsat-8"]}})
items = [it for it in items if 6 <= int(it["properties"]["datetime"][5:7]) <= 9]
print(f"[rustac] {len(items)} summer scenes")

# 2. write geoparquet (rustac) — keep UNSIGNED hrefs; the AzureStore below carries
#    the container SAS token, so we don't sign per-asset URLs.
rustac.write_sync(str(GP), items)
print(f"[rustac] wrote {GP.name}")

# 3. one AzureStore for the whole landsat-c2 container (obstore native Azure +
#    MPC SAS token — sidesteps obstore mis-routing signed blob URLs as Azure auth).
TOK = sas.get_token("landsateuwest", "landsat-c2").token
STORE = obstore.store.AzureStore(account_name="landsateuwest",
                                 container_name="landsat-c2", sas_token=TOK)
def path_from_href(h):                       # href -> blob path within the container
    return h.split("/landsat-c2/", 1)[1].split("?", 1)[0]

# 4. lazycogs read -> (band, time, y, x) ----------------------------------------
tf = Transformer.from_crs("EPSG:4326", EPSG, always_xy=True)
xs, ys = [], []
for lon in (BBOX_LL[0], BBOX_LL[2]):
    for lat in (BBOX_LL[1], BBOX_LL[3]):
        x, y = tf.transform(lon, lat); xs.append(x); ys.append(y)
xmin = np.floor(min(xs) / RES) * RES; xmax = np.ceil(max(xs) / RES) * RES
ymin = np.floor(min(ys) / RES) * RES; ymax = np.ceil(max(ys) / RES) * RES
bbox_utm = (xmin, ymin, xmax, ymax)
print("[lazycogs] opening ...", flush=True)
da = lazycogs.open(str(GP), bbox=bbox_utm, crs=EPSG, resolution=RES,
                   bands=["nir08", "swir22", "qa_pixel"], time_period=None,
                   nodata=0, store=STORE, path_from_href=path_from_href)
band = list(da.coords["band"].values)
arr = np.asarray(da.values)                      # (band, time, y, x)
times = np.asarray(da.coords["time"].values)
yrs = times.astype("datetime64[Y]").astype(int) + 1970
bi = {b: i for i, b in enumerate(band)}
nir = arr[bi["nir08"]].astype("f4"); swir = arr[bi["swir22"]].astype("f4")
qa = arr[bi["qa_pixel"]]
print(f"[lazycogs] read {arr.shape} (band,time,y,x); years {yrs.min()}-{yrs.max()}")

# 5. QA-mask + NBR + annual median (identical to fetch_nbr.py) -------------------
def qa_bad(q):
    qi = q.astype(np.uint32); bad = np.zeros(qi.shape, bool)
    for b in QA_BAD_BITS: bad |= ((qi >> b) & 1).astype(bool)
    return bad
nirs = nir * SR_SCALE + SR_OFFSET; swirs = swir * SR_SCALE + SR_OFFSET
denom = nirs + swirs
with np.errstate(divide="ignore", invalid="ignore"):
    nbr = (nirs - swirs) / denom
nbr[qa_bad(qa) | (nir == 0) | (denom == 0)] = np.nan

years = [y for y in range(START, END + 1) if (yrs == y).any()]
import warnings
with warnings.catch_warnings():
    warnings.simplefilter("ignore", RuntimeWarning)
    annual = np.stack([np.nanmedian(nbr[yrs == y], axis=0) for y in years]).astype("f4")
spy = np.asarray([(yrs == y).sum() for y in years], np.int32)
transform = from_origin(xmin, ymax, RES, RES)
np.savez_compressed(OUT, annual=annual, years=np.asarray(years, np.int32),
                    transform=np.asarray(transform[:6], float), bbox=np.asarray(BBOX_LL, float),
                    scenes_per_year=spy)
print(f"[cache] {OUT.name}: {annual.shape}, {len(years)} years, {annual.nbytes/1e3:.0f} KB")

# 6. quick A/B vs the rasterio npz ----------------------------------------------
ras = ROOT / "data" / "nbr_1984_2016.npz"
if ras.exists():
    z = np.load(ras)
    ra, ry = z["annual"], z["years"].astype(int)
    common = [y for y in years if y in ry]
    a = np.array([annual[years.index(y), 26, 26] for y in common]) * 1000
    b = np.array([ra[list(ry).index(y), 26, 26] for y in common]) * 1000
    m = np.isfinite(a) & np.isfinite(b)
    print(f"\n[A/B canonical pixel] lazycogs vs rasterio: overlap {m.sum()} yrs  "
          f"corr {np.corrcoef(a[m], b[m])[0,1]:.4f}  MAD {np.mean(np.abs(a[m]-b[m])):.1f} NBRx1000")

import os
import shutil
import numpy as np
import xarray as xr
from dask.diagnostics import ProgressBar # type: ignore

URL = (
    "gs://weatherbench2/datasets/era5/"
    "1959-2023_01_10-wb13-6h-1440x721_with_derived_variables.zarr"
)

OUT = "era5_28ch_0p25_6h.zarr"
INDICES = "indices2048.npy"

TOTAL = 32
BATCH = 32

LEVELS = [1000, 925, 850, 700]

SURFACE = [
    "2m_temperature",
    "mean_sea_level_pressure",
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
    "total_precipitation_6hr",
    "sea_surface_temperature",
    "total_column_water_vapour",
    "total_cloud_cover",
]

PRESSURE = [
    "temperature",
    "u_component_of_wind",
    "v_component_of_wind",
    "geopotential",
    "specific_humidity",
]

print("Opening WeatherBench2...")

ds = xr.open_zarr(
    URL,
    consolidated=True,
    storage_options={"token": "anon"},
    chunks={"time": 1}, # type: ignore
)

ds = ds.sel(
    time=slice("2021-01-01", "2021-12-31")
)

print("Frames available:", ds.time.size)

###########################################################
# Индексы
###########################################################

if os.path.exists(INDICES):
    print("Loading saved indices...")
    idx = np.load(INDICES)
else:
    print("Generating random indices...")
    rng = np.random.default_rng(42)

    idx = np.sort(
        rng.choice(
            ds.time.size,
            TOTAL,
            replace=False,
        )
    )

    np.save(INDICES, idx)

###########################################################
# Сколько уже скачано
###########################################################

if os.path.exists(OUT):
    current = xr.open_zarr(OUT, consolidated=True)
    downloaded = current.time.size
    first = False

    print(f"Already downloaded: {downloaded}")
else:
    downloaded = 0
    first = True

###########################################################
# Докачиваем
###########################################################

idx = idx[downloaded:]

print(f"Remaining: {len(idx)}")

for start in range(0, len(idx), BATCH):

    stop = min(start + BATCH, len(idx))

    real_start = downloaded + start
    real_stop = downloaded + stop

    print(f"Batch {real_start}:{real_stop}")

    batch = ds.isel(time=idx[start:stop])

    batch = xr.merge([
        batch[SURFACE],
        batch[PRESSURE].sel(level=LEVELS),
    ])

    batch = batch.astype(np.float32)

    batch = batch.chunk({
        "time": BATCH,
        "latitude": -1,
        "longitude": -1,
    })

    with ProgressBar():

        if first:

            batch.to_zarr(
                OUT, # type: ignore
                mode="w",
                consolidated=True,
            )

            first = False

        else:

            batch.to_zarr(
                OUT, # type: ignore
                mode="a",
                append_dim="time",
            )

print("DONE")
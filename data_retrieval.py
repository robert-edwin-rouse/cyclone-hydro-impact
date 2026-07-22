"""

"""

import cdsapi
import xarray as xr
from dask.diagnostics import ProgressBar
from dask.distributed import Client, LocalCluster
import os
import config


# =============================================================================
# Download ERA5 data by month and year
# =============================================================================
client = cdsapi.Client()
data_dir = config.era5_data_dir

for year in config.years:
    for month in config.months:
        filename = os.path.join(data_dir, f"era5_pressure_{year}_{month}.nc")
        try:
            xr.open_dataset(filename,
                engine="h5netcdf",
                chunks="auto",
            )
        except:
            request = {
                "product_type": "reanalysis",
                "variable": config.pressure_variables,
                "year": [str(year)],
                "month": [str(month)],
                "day": config.days,
                "time": config.hours,
                "pressure_level": config.pressure_levels,
                "data_format": "netcdf",
                "download_format": "unarchived",
                # "area": config.lat_lon
            }
            client.retrieve(config.pressure_dataset, request).download(filename)


for year in config.years:
    for month in config.months:
        filename = os.path.join(data_dir, f"era5_sst_{year}_{month}.nc")
        try:
            xr.open_dataset(filename,
                engine="h5netcdf",
                chunks="auto",
            )
        except:
            request = {
                "product_type": "reanalysis",
                "variable": config.sst_variables,
                "year": [str(year)],
                "month": [str(month)],
                "day": config.days,
                "time": config.hours,
                "data_format": "netcdf",
                "download_format": "unarchived",
                # "area": config.lat_lon
            }
            client.retrieve(config.surface_dataset, request).download(filename)


for year in config.years:
    for month in config.months:
        filename = os.path.join(data_dir, f"era5_rain_{year}_{month}.nc")
        try:
            xr.open_dataset(filename,
                engine="h5netcdf",
                chunks="auto",
            )
        except:
            request = {
                "product_type": "reanalysis",
                "variable": config.rain_variables,
                "year": [str(year)],
                "month": [str(month)],
                "day": config.days,
                "time": config.hours,
                "data_format": "netcdf",
                "download_format": "unarchived",
                # "area": config.lat_lon
            }
            client.retrieve(config.surface_dataset, request).download(filename)


# =============================================================================
# Concatenate all yearly data
# =============================================================================
ds_all = xr.open_mfdataset(
    "data/era5/era5_pressure_*.nc",
    engine="h5netcdf",
    combine="by_coords",
    parallel=True,
    coords="minimal",
    data_vars="all",
    chunks={"valid_time": 1460},
)

delayed_write = ds_all.to_netcdf(config.pressure_path, compute=False)
print("Streaming dataset to disk...")
with ProgressBar():
    delayed_write.compute()
ds_all.close()
print(f"Wrote combined dataset to: {config.pressure_path}")


ds_all = xr.open_mfdataset(
    "data/era5/era5_sst_*.nc",
    engine="h5netcdf",
    combine="by_coords",
    parallel=True,
    coords="minimal",
    data_vars="all",
    chunks={"valid_time": 1460},
)

delayed_write = ds_all.to_netcdf(config.sst_path, compute=False)
print("Streaming dataset to disk...")
with ProgressBar():
    delayed_write.compute()
ds_all.close()
print(f"Wrote combined dataset to: {config.sst_path}")


ds_all = xr.open_mfdataset(
    "data/era5/era5_rain_*.nc",
    engine="h5netcdf",
    combine="by_coords",
    parallel=True,
    coords="minimal",
    data_vars="all",
    chunks={"valid_time": 1460},
)

delayed_write = ds_all.to_netcdf(config.rain_path, compute=False)
print("Streaming dataset to disk...")
with ProgressBar():
    delayed_write.compute()
ds_all.close()
print(f"Wrote combined dataset to: {config.rain_path}")

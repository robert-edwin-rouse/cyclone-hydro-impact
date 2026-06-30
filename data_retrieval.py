"""

"""

import cdsapi
import xarray as xr
import os
import config


# =============================================================================
# Download ERA5 data by month and year
# =============================================================================
client = cdsapi.Client()
data_dir = config.era5_data_dir

for year in config.years:
    for month in config.months:
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
            "area": config.lat_lon
        }
        filename = os.path.join(data_dir, f"era5_pressure_{year}_{month}.nc")
        client.retrieve(config.pressure_dataset, request).download(filename)


for year in config.years:
    for month in config.months:
        request = {
            "product_type": "reanalysis",
            "variable": config.surface_variables,
            "year": [str(year)],
            "month": [str(month)],
            "day": config.days,
            "time": config.hours,
            "data_format": "netcdf",
            "download_format": "unarchived",
            "area": config.lat_lon
        }
        filename = os.path.join(data_dir, f"era5_surface_{year}_{month}.nc")
        client.retrieve(config.surface_dataset, request).download(filename)


# =============================================================================
# Concatenate all yearly data
# =============================================================================
pressure_monthly_files = sorted(data_dir.glob("era5_pressure_*.nc"))
print(f" Found {len(pressure_monthly_files)} monthly files to combine.")

ds_all = xr.open_mfdataset(
    [str(p) for p in pressure_monthly_files],
    engine="h5netcdf",
    combine="by_coords",
    parallel=True,
    coords="minimal",
    data_vars="all",
    chunks="auto",
)
ds_all.to_netcdf(config.pressure_path)
ds_all.close()
print(f"Wrote combined dataset to: {config.pressure_path}")

surface_monthly_files = sorted(data_dir.glob("era5_surface_*.nc"))
print(f" Found {len(surface_monthly_files)} monthly files to combine.")

ds_all = xr.open_mfdataset(
    [str(p) for p in surface_monthly_files],
    engine="h5netcdf",
    combine="by_coords",
    parallel=True,
    coords="minimal",
    data_vars="all",
    chunks="auto",
)
ds_all.to_netcdf(config.surface_path)
ds_all.close()
print(f"Wrote combined dataset to: {config.surface_path}")

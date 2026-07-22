#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 22 23:18:57 2026

@author: robert
"""

import xarray as xr
from dask.distributed import Client, LocalCluster
from dask.diagnostics import ProgressBar
import config
import hdf5plugin

def combine_era5():
    cluster = LocalCluster(
        n_workers=16,
        threads_per_worker=1,
        memory_limit='7.5GB')
    client = Client(cluster)
    print('Dask Dashboard running at: {client.dashboard_link}')
    
    target_chunks = {
        'valid_time': 72,
        # 'pressure_level': -1,
        'latitude': -1,
        'longitude': -1}
    
    ds_all = xr.open_mfdataset(
        "data/era5/era5_pressure_*.nc",
        engine="h5netcdf",
        combine="by_coords",
        parallel=True,
        coords="minimal",
        data_vars="all",
        chunks={"valid_time": 365},
    )
    
    ds_all = ds_all.chunk(target_chunks)
    
    with ProgressBar():
        ds_all.to_zarr(config.pressure_zarr_path,
                        mode="w",
                        consolidated=True,
                        compute=True)
    ds_all.close()
    client.close()
    cluster.close()
    print(f"Wrote combined dataset to: {config.pressure_zarr_path}")

if __name__ == 'main':
    combine_era5()
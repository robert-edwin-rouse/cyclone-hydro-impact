'''
Text pending.
'''

import xarray as xr
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import dask.array as darray
import config
import os
import shutil


# =============================================================================
# Load in ERA5 data
# =============================================================================
pressure_data = xr.open_dataset('haiyan_pressure.nc',
                                # config.pressure_path,
                                engine='h5netcdf',
                                chunks={'valid_time': 20})
pressure_vars = pressure_data[config.pressure_var_codes]
pressure_array = pressure_vars.to_array(dim='var')
pressure_array = pressure_array.transpose('valid_time', 'latitude',
                                          'longitude', 'var', 'pressure_level')
pressure_array = pressure_array.stack(channel=('var', 'pressure_level'))

surface_array = xr.open_dataset('haiyan_surface.nc',
                                # config.surface_path,
                                engine='h5netcdf',
                                chunks={'valid_time': 20})


# =============================================================================
# Transform SST using land-temperature mask
# =============================================================================
sst_mask = (np.isnan(surface_array.sst.isel(valid_time=0))).astype(int)[0]
sst_mask = (sst_mask.expand_dims(valid_time=surface_array.valid_time, axis=0).
            expand_dims(channel=[-2], axis=-1).
            transpose('valid_time', 'latitude', 'longitude', 'channel'))

sst = surface_array.sst.fillna(surface_array.t2m)[0]
sst = (sst.expand_dims(channel=[-1], axis=-1).
       transpose('valid_time', 'latitude', 'longitude', 'channel'))


# =============================================================================
# Chunk data
# =============================================================================
t_chunks, y_chunks, x_chunks, _ = pressure_array.chunks
n_channel = pressure_array.sizes['channel'] + sst.sizes['channel'] + \
    sst_mask.sizes['channel']
chunk_dict = {'valid_time': t_chunks,
              'latitude':   y_chunks,
              'longitude':  x_chunks,
              'channel':    n_channel, }

pressure_array = pressure_array.astype('float32')
sst_mask = sst_mask.astype('float32')
sst = sst.astype('float32')

n_pressure_ch = pressure_array.sizes['channel']
pressure_array = pressure_array.assign_coords(
    channel=np.arange(n_pressure_ch, dtype=np.int64))

sst_mask = sst_mask.assign_coords(channel=np.array([-2], dtype=np.int64))
sst = sst.assign_coords(channel=np.array([-1], dtype=np.int64))

feature_array = xr.concat([pressure_array, sst_mask, sst],
                          dim='channel', coords='minimal',
                          join='exact', compat='override',).chunk(chunk_dict)


# =============================================================================
# Load in IBTRACS data, define radii, and lifestage classifications
# =============================================================================
df = pd.read_csv('ibtracs.since1980.list.v04r01.csv')
df = df[['SID', 'ISO_TIME', 'LAT', 'LON', 'USA_STATUS',
                          'USA_WIND', 'USA_PRES', 'USA_SSHS', 'USA_R34_NE',
                          'USA_R34_SE', 'USA_R34_SW', 'USA_R34_NW']]
df = df.drop(0)
df['ISO_TIME'] = pd.to_datetime(df['ISO_TIME'], errors='coerce')
df['USA_SSHS'] = pd.to_numeric(df['USA_SSHS'], errors='coerce')
df['USA_SSHS'] = df['USA_SSHS'].fillna(-1)
df['LAT'] = pd.to_numeric(df['LAT'])
df['LON'] = pd.to_numeric(df['LON'])
df["LON"] = np.where(df["LON"] < 0, df["LON"] + 360, df["LON"])
df['USA_R34_NE'] = pd.to_numeric(df['USA_R34_NE'], errors='coerce').fillna(0)
df['USA_R34_SE'] = pd.to_numeric(df['USA_R34_SE'], errors='coerce').fillna(0)
df['USA_R34_SW'] = pd.to_numeric(df['USA_R34_SW'], errors='coerce').fillna(0)
df['USA_R34_NW'] = pd.to_numeric(df['USA_R34_NW'], errors='coerce').fillna(0)
df['Effective_Radius'] = df[['USA_R34_NE', 'USA_R34_SE',
                             'USA_R34_SW', 'USA_R34_NW']].max(axis=1)
square_res = df['Effective_Radius'].max() *\
    config.nm_to_km * config.grid_res
df['Lat_Min'] = df['LAT'] - square_res/2
df['Lat_Max'] = df['LAT'] + square_res/2
df['Lon_Min'] = df['LON'] - square_res/2
df['Lon_Max'] = df['LON'] + square_res/2
df = df.dropna(subset=['ISO_TIME']).sort_values(['SID', 'ISO_TIME'])


def lifestage(sequence, labels, threshold):
    values = np.asarray(sequence)
    result = []
    last_threshold_cross = -1
    for i, val in enumerate(values):
        if val >= threshold:
            last_threshold_cross = i
    if last_threshold_cross == -1:
        return [labels[1]] * len(values)
    result = []
    seen_above_threshold = False
    for i, val in enumerate(values):
        if val >= threshold:
            seen_above_threshold = True
            result.append(labels[2])
        elif not seen_above_threshold:
            result.append(labels[1])
        elif i > last_threshold_cross:
            result.append(labels[0])
        else:
            result.append(labels[2])
    return result


cyclones = []
for sid, grp in df.groupby('SID'):
    if 1 not in grp.USA_SSHS.values:
        grp['Classification'] = config.lifestages[0]
    if 1 in grp.USA_SSHS.values:
        grp['Classification'] = lifestage(grp['USA_SSHS'],
                                          config.lifestages[1:4], 1)
    cyclones.append(grp)
cyclones = pd.concat(cyclones, ignore_index=True)
cyclones = cyclones[cyclones['Classification'] != 'Storm - Nondeveloping']


# =============================================================================
# Create cyclone lifestage image masks
# =============================================================================
times = pd.to_datetime(surface_array.valid_time.values)
cyclones = cyclones[cyclones['ISO_TIME'].isin(times)]

def drop_both_overlapping(cyclones_df):
    df = cyclones_df.copy()
    to_drop = set()

    for time, grp in df.groupby('ISO_TIME'):
        idxs = grp.index.values
        if len(idxs) <= 1:
            continue

        lat_min = grp['Lat_Min'].values
        lat_max = grp['Lat_Max'].values
        lon_min = grp['Lon_Min'].values
        lon_max = grp['Lon_Max'].values

        lat_overlap = (lat_min[:, None] < lat_max[None, :]) & \
            (lat_max[:, None] > lat_min[None, :])
        lon_overlap = (lon_min[:, None] < lon_max[None, :]) & \
            (lon_max[:, None] > lon_min[None, :])
        overlap = lat_overlap & lon_overlap
        np.fill_diagonal(overlap, False)
        rows_with_overlap = np.where(overlap.any(axis=1))[0]
        for r in rows_with_overlap:
            to_drop.add(int(idxs[r]))

    return df.drop(index=list(to_drop))


cyclones = drop_both_overlapping(cyclones)


# =============================================================================
# Build input-output sample dataset from cyclone bounding boxes
# =============================================================================
precipitation = surface_array['tp'][0]
precipitation = precipitation.astype('float32')
precipitation = (
    precipitation.expand_dims(channel=[0], axis=-1)
    .transpose('valid_time', 'latitude', 'longitude', 'channel')
)

pressure_dask = pressure_array.data 
precip_dask = precipitation.data

time_vals = pressure_array["valid_time"].values
lat_vals = pressure_array["latitude"].values
lon_vals = pressure_array["longitude"].values

target_size = 46

def _index_bounds(values, lo, hi):
    if values[0] > values[-1]:
        values2 = -values
        lo2, hi2 = -hi, -lo
        lo_idx = int(np.searchsorted(values2, lo2, side="left"))
        hi_idx = int(np.searchsorted(values2, hi2, side="right"))
        return lo_idx, hi_idx
    else:
        lo_idx = int(np.searchsorted(values, lo, side="left"))
        hi_idx = int(np.searchsorted(values, hi, side="right"))
        return lo_idx, hi_idx

def _crop_or_pad(arr):
    # arr here is a Dask array slice with shape (h, w, c)
    h, w = arr.shape[:2]

    # --- Step 1: Crop dimensions that are larger than target_size ---
    if h > target_size:
        y0 = (h - target_size) // 2
        arr = arr[y0:y0 + target_size, :, :]
        h = target_size

    if w > target_size:
        x0 = (w - target_size) // 2
        arr = arr[:, x0:x0 + target_size, :]
        w = target_size

    # --- Step 2: Pad dimensions that are smaller than target_size ---
    if h < target_size or w < target_size:
        pad_h = max(0, target_size - h)
        pad_w = max(0, target_size - w)

        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left

        arr = da.pad(
            arr,
            ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)),
            mode="constant",
            constant_values=0.0
        )

    return arr

input_arrays = []
output_arrays = []
sample_times = []
sample_ids = []

for sample_id, row in cyclones.iterrows():
    storm_time = pd.Timestamp(row["ISO_TIME"])

    t_idx = int(np.where(time_vals == np.datetime64(storm_time))[0][0])

    lat_lo = float(min(row["Lat_Min"], row["Lat_Max"]))
    lat_hi = float(max(row["Lat_Min"], row["Lat_Max"]))
    lon_lo = float(min(row["Lon_Min"], row["Lon_Max"]))
    lon_hi = float(max(row["Lon_Min"], row["Lon_Max"]))

    lat0, lat1 = _index_bounds(lat_vals, lat_lo, lat_hi)
    lon0, lon1 = _index_bounds(lon_vals, lon_lo, lon_hi)

    lat0 = int(np.clip(lat0, 0, len(lat_vals) - 1))
    lat1 = int(np.clip(lat1, lat0 + 1, len(lat_vals)))
    lon0 = int(np.clip(lon0, 0, len(lon_vals) - 1))
    lon1 = int(np.clip(lon1, lon0 + 1, len(lon_vals)))

    # Slice the dask arrays lazily (no computation occurs here)
    input_box = pressure_dask[t_idx, lat0:lat1, lon0:lon1, :]
    output_box = precip_dask[t_idx, lat0:lat1, lon0:lon1, :]

    # Pass the lazy arrays to our updated dask-compatible crop/pad function
    input_arrays.append(_crop_or_pad(input_box))
    output_arrays.append(_crop_or_pad(output_box))
    sample_times.append(storm_time)
    sample_ids.append(int(sample_id))

# Stack the lazy Dask arrays along a new sample dimension (axis=0)
input_array = da.stack(input_arrays, axis=0)
output_array = da.stack(output_arrays, axis=0)


# Convert tracking metadata into numpy arrays for splitting
sample_times_np = np.array(sample_times)
sample_ids_np = np.array(sample_ids)

# --- Define Splits from config ---
# Assuming config defines proportions like: config.train_pct = 0.70, config.val_pct = 0.15
n_samples = len(sample_ids)
train_end = int(n_samples * config.train_set_percent)
val_end = train_end + int(n_samples * config.valid_set_percent)

# Split lazy input/output Dask arrays (No data is loaded into memory yet)
train_input = input_array[:train_end]
val_input = input_array[train_end:val_end]
test_input = input_array[val_end:]

train_output = output_array[:train_end]
val_output = output_array[train_end:val_end]
test_output = output_array[val_end:]

# Split corresponding metadata
train_times, train_ids = sample_times_np[:train_end], sample_ids_np[:train_end]
val_times, val_ids = sample_times_np[train_end:val_end], sample_ids_np[train_end:val_end]
test_times, test_ids = sample_times_np[val_end:], sample_ids_np[val_end:]


# Define axes to reduce over: (sample/batch, height, width)
reduce_axes = (0, 1, 2)

# Lazily define the mean and standard deviation calculations
mean_in_lazy = train_input.mean(axis=reduce_axes)
max_in_lazy = train_input.max(axis=reduce_axes)
min_in_lazy = train_input.min(axis=reduce_axes)
range_in_lazy = max_in_lazy - min_in_lazy

mean_out_lazy = train_output.mean(axis=reduce_axes)
max_out_lazy = train_output.max(axis=reduce_axes)
min_out_lazy = train_output.min(axis=reduce_axes)
range_out_lazy = max_out_lazy - min_out_lazy

# Trigger compute ONLY on these tiny 1D arrays (extremely fast and memory-safe)
mean_in, range_in = da.compute(mean_in_lazy, range_in_lazy)
mean_out, range_out = da.compute(mean_out_lazy, range_out_lazy)

# Guard against division by zero for invariant channels
range_in = np.where(range_in == 0, 1.0, range_in)
range_out = np.where(range_out == 0, 1.0, range_out)

# --- Save Normalization Parameters to NetCDF ---
norm_ds = xr.Dataset(
    data_vars={
        "input_mean": (["input_channel"], mean_in),
        "input_range": (["input_channel"], range_in),
        "output_mean": (["output_channel"], mean_out),
        "output_range": (["output_channel"], range_out),
    },
    coords={
        "input_channel": np.arange(mean_in.shape[0]),
        "output_channel": np.arange(mean_out.shape[0]),
    }
)
norm_ds.to_netcdf("normalization_params.nc")


# Apply normalization lazily
train_input_norm = (train_input - mean_in) / range_in
val_input_norm = (val_input - mean_in) / range_in
test_input_norm = (test_input - mean_in) / range_in

train_output_norm = (train_output - mean_out) / range_out
val_output_norm = (val_output - mean_out) / range_out
test_output_norm = (test_output - mean_out) / range_out


def save_split_to_zarr(input_dask, output_dask, times, ids, path):
    """
    Wraps normalized lazy Dask arrays and metadata into an Xarray Dataset
    and writes it lazily to a Zarr store.
    """
    dataset = xr.Dataset(
        data_vars={
            "inputs": (["sample", "y", "x", "input_channel"], input_dask),
            "outputs": (["sample", "y", "x", "output_channel"], output_dask),
            "sample_id": (["sample"], ids),
        },
        coords={
            "sample": np.arange(len(ids)),
            "time": (["sample"], times),
            "y": np.arange(target_size),
            "x": np.arange(target_size),
            "input_channel": np.arange(input_dask.shape[-1]),
            "output_channel": np.arange(output_dask.shape[-1]),
        }
    )
    
    # Write to Zarr lazily, utilizing Dask under the hood
    dataset.to_zarr(path, mode="w", consolidated=True)
    print(f"Successfully wrote {path}")

# Write splits to disk
save_split_to_zarr(train_input_norm, train_output_norm, train_times, train_ids, "train_dataset.zarr")
save_split_to_zarr(val_input_norm, val_output_norm, val_times, val_ids, "val_dataset.zarr")
save_split_to_zarr(test_input_norm, test_output_norm, test_times, test_ids, "test_dataset.zarr")

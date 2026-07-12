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
pressure_data = xr.open_dataset('era5_pressure_data.nc',
                                # config.pressure_path,
                                engine='h5netcdf',
                                chunks={'valid_time': 20})
pressure_vars = pressure_data[config.pressure_var_codes]
pressure_array = pressure_vars.to_array(dim='var')
pressure_array = pressure_array.transpose('valid_time', 'latitude',
                                          'longitude', 'var', 'pressure_level')
pressure_array = pressure_array.stack(channel=('var', 'pressure_level'))

surface_array = xr.open_dataset('era5_surface_data.nc',
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

pressure_np = pressure_array.compute().astype("float32").values
precip_np = precipitation.compute().astype("float32").values

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
    h, w = arr.shape[:2]

    if h >= target_size and w >= target_size:
        y0 = (h - target_size) // 2
        x0 = (w - target_size) // 2
        return arr[y0:y0 + target_size, x0:x0 + target_size, :]

    # pad_h = max(0, target_size - h)
    # pad_w = max(0, target_size - w)

    # pad_top = pad_h // 2
    # pad_bottom = pad_h - pad_top
    # pad_left = pad_w // 2
    # pad_right = pad_w - pad_left

    # return np.pad(
    #     arr,
    #     ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)),
    #     mode="constant",
    # )

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

    input_box = pressure_np[t_idx, lat0:lat1, lon0:lon1, :]
    output_box = precip_np[t_idx, lat0:lat1, lon0:lon1, :]

    input_arrays.append(_crop_or_pad(input_box))
    output_arrays.append(_crop_or_pad(output_box))
    sample_times.append(storm_time)
    sample_ids.append(int(sample_id))

input_array = np.stack(input_arrays, axis=0)
output_array = np.stack(output_arrays, axis=0)

# =============================================================================
# Split into train/valid/test, normalise inputs from training statistics,
# cache the statistics, and write the datasets to zarr.
# =============================================================================

def _split_indices(num_items, train_frac, valid_frac, test_frac):
    n_train = int(num_items * train_frac)
    n_valid = int(num_items * valid_frac)
    n_test = num_items - n_train - n_valid
    if n_test < 0:
        raise ValueError(
            f"Invalid split fractions: {train_frac}, {valid_frac}, {test_frac}"
        )
    idx = np.arange(num_items)
    return idx[:n_train], idx[n_train:n_train + n_valid], idx[n_train + n_valid:]


def _compute_normalisation_stats(train_inputs):
    mean = train_inputs.mean(axis=(0, 1, 2)).astype(np.float32)
    range_ = train_inputs.max(axis=(0, 1, 2)) - train_inputs.min(axis=(0, 1, 2))
    range_ = np.where(range_ == 0, 1.0, range_).astype(np.float32)
    return mean, range_


def save_ds_splits_to_zarr(train, valid, test, base_dir):
    os.makedirs(base_dir, exist_ok=True)

    for filename, split_ds in [
        ("train_data.zarr", train),
        ("valid_data.zarr", valid),
        ("test_data.zarr", test),
    ]:
        path = os.path.join(base_dir, filename)
        if os.path.exists(path):
            shutil.rmtree(path)

        chunk_spec = {
            "sample": max(1, min(20, split_ds.sizes["sample"])),
            "latitude": split_ds.sizes["latitude"],
            "longitude": split_ds.sizes["longitude"],
        }
        if "channel" in split_ds.dims:
            chunk_spec["channel"] = split_ds.sizes["channel"]
        if "target_channel" in split_ds.dims:
            chunk_spec["target_channel"] = split_ds.sizes["target_channel"]

        split_ds.chunk(chunk_spec).to_zarr(path, mode="w")


num_samples = input_array.shape[0]
train_idx, valid_idx, test_idx = _split_indices(
    num_samples,
    config.train_set_percent,
    config.valid_set_percent,
    config.test_set_percent,
)

train_inputs = input_array[train_idx].astype(np.float32)
train_mean, train_range = _compute_normalisation_stats(train_inputs)

normalised_inputs = (
    input_array.astype(np.float32) - train_mean[None, None, None, :]
) / train_range[None, None, None, :]

normalisation_cache = {
    "mean": train_mean,
    "range": train_range,
}

stats_path = os.path.join(config.data_dir, "normalisation_stats.npz")
np.savez_compressed(
    stats_path,
    mean=train_mean,
    range=train_range,)

sample_ids_arr = np.array(sample_ids, dtype=np.int32)
sample_times_arr = np.array(sample_times, dtype="datetime64[ns]")

split_inputs = [
    normalised_inputs[train_idx],
    normalised_inputs[valid_idx],
    normalised_inputs[test_idx],
]
split_outputs = [
    output_array[train_idx],
    output_array[valid_idx],
    output_array[test_idx],
]
split_sample_ids = [
    sample_ids_arr[train_idx],
    sample_ids_arr[valid_idx],
    sample_ids_arr[test_idx],
]
split_sample_times = [
    sample_times_arr[train_idx],
    sample_times_arr[valid_idx],
    sample_times_arr[test_idx],
]

splits = []
for split_input, split_output, split_sample_id, split_sample_time in zip(
    split_inputs, split_outputs, split_sample_ids, split_sample_times
):
    split_output = split_output.reshape(split_output.shape[0], split_output.shape[1], split_output.shape[2], 1)
    split_ds = xr.Dataset(
        data_vars={
            "inputs": (("sample", "latitude", "longitude", "channel"), split_input),
            "precipitation": (("sample", "latitude", "longitude", "target_channel"), split_output),
        },
        coords={
            "sample": split_sample_id,
            "valid_time": ("sample", split_sample_time),
            "latitude": np.arange(split_input.shape[1], dtype=np.int32),
            "longitude": np.arange(split_input.shape[2], dtype=np.int32),
            "channel": np.arange(split_input.shape[3], dtype=np.int32),
            "target_channel": np.array([0], dtype=np.int32),
        },
    )
    splits.append(split_ds)

train_ds, valid_ds, test_ds = splits
save_ds_splits_to_zarr(train_ds, valid_ds, test_ds, config.data_dir)

print("Saved train/valid/test datasets to", config.data_dir)
print("Normalisation stats cached to", stats_path)

"""
This module defines resources, both computational and data, the relevant
features, and training configurations / hyperparameters.
"""

import torch
import torch.optim as optim
import os


# =============================================================================
# Project Initialisation
# =============================================================================
base_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(base_dir, "data")
era5_data_dir = os.path.join(base_dir, "data", "era5")

os.makedirs(data_dir, exist_ok=True)
os.makedirs(era5_data_dir, exist_ok=True)

cds_api_url = "https://cds.climate.copernicus.eu/api"
# cds_api_key = "########################"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =============================================================================
# Data Acquisition & Feature Selection
# =============================================================================
lat_lon = [70, -260, 0, -100]

years = [x for x in range(2010, 2016)]
months = [x for x in range(1, 13)]
days = [x for x in range(1, 32)]
hours = [6*x for x in range(0, 4)]

surface_dataset = "reanalysis-era5-single-levels"
surface_variables = ["model_bathymetry",
                     "sea_surface_temperature",
                     "2m_temperature",
                     "total_precipitation",]
surface_var_codes = ["sst", "t2m", "tp"]
surface_path = "era5_surface_data.nc"

pressure_dataset = "reanalysis-era5-pressure-levels"
pressure_levels = [1000, 750, 500]
pressure_variables = ["relative_humidity",
                      "temperature",
                      "u_component_of_wind",
                      "v_component_of_wind",
                      "vorticity"]
pressure_var_codes = ["r", "t", "u", "v", "vo"]
pressure_path = "era5_pressure_data.nc"


# =============================================================================
# Data Labelling & Output Configuration
# =============================================================================
nm_to_km = 1.852
grid_res = 1/125
area_growth_factor = 0.25
lifestages = ['Storm - Nondeveloping',
              'Cyclolysis',
              'Cyclogenesis',
              'Active Cyclone',]
train_set_percent = 0.75
valid_set_percent = 0.05
test_set_percent = 0.2


# =============================================================================
# Training & Model Configuration
# =============================================================================
optimiser = optim.AdamW
epochs = 64
batch_size = 8
num_workers = 0
learning_rate = 0.0001
weight_decay = 1e-4
dropout = 0.1

model_path = 'cyclone-hydro-ml.pt'

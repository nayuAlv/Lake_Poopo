import numpy as np
import xarray as xr
import pandas as pd


def area_weighted_mean(ds: xr.Dataset, var: str) -> xr.DataArray:
    """
    Compute the area-weighted spatial mean over (lat, lon) of `var`.
    Weights use cos(latitude in radians).
    """
    lat = ds["latitude"]
    weights = np.cos(np.deg2rad(lat))
    weights.name = "weights"
    weighted = ds[var].weighted(weights)
    return weighted.mean(dim=("latitude", "longitude"))

def era5_monthly_to_mm(ds, var, unit_convertion):
    """
    ERA5 monthly-means: values are m/day (daily mean for the month).
    To get monthly accumulation in mm:
      value [m/day] * days_in_month * 1000 [mm/m] = mm/month
    """
    weights = np.cos(np.deg2rad(ds["latitude"]))
    #weights = weights / weights.sum()
    weighted_mean = (ds[var].weighted(weights).mean(dim=("latitude", "longitude")))
    df = weighted_mean.to_dataframe().reset_index()
    days = pd.to_datetime(df["valid_time"]).dt.days_in_month
    df[f"{var}_mm"] = df[var] * days * unit_convertion        
    return df
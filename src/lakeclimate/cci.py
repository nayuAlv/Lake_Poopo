
import datetime
import numpy as np
import pandas as pd
import xarray as xr

# version dataset (2.1 is the version published in July 2024)
version = '2.1'

def build_lake_mask(maskfile, lake_id):

    mask_xr = xr.open_dataset(maskfile)
    mask = mask_xr['CCI_lakeid'].values
    mask_ind  = np.where(mask == lake_id)
    minx = np.min(mask_ind[1][:]) - 1
    maxx = np.max(mask_ind[1][:]) + 1
    miny = np.min(mask_ind[0][:]) - 1
    maxy = np.max(mask_ind[0][:]) + 1

    mask_lake = mask[miny:maxy+1, minx:maxx+1]
    mask_lake[mask_lake!=lake_id] = 0
    mask_lake[mask_lake == lake_id] = 1
    mask_xr.close()
    
    return mask_lake, miny, maxy, minx, maxx


def extract_variable_timeseries(lake_id, varname, date_range, maskfile, version = version):
    """
    Return a DataFrame with columns ['date', varname] for one variable.
    """

    # test if dates are in the temporal coverage
    mindate = datetime.datetime.strptime(date_range[0], '%Y-%m-%d')
    maxdate = datetime.datetime.strptime(date_range[1], '%Y-%m-%d')
    mindate = max([mindate, datetime.datetime(1992,9,26)])
    maxdate = min([maxdate, datetime.datetime(2022,12,31)]) 

    mask_lake,  miny, maxy, minx, maxx = build_lake_mask(maskfile, lake_id)
    date_vec, data_vec = [], []   
    for data_date in np.arange(mindate.toordinal(), maxdate.toordinal()+1):
        current_date = datetime.datetime.fromordinal(data_date)
        date_str = current_date.strftime("%Y%m%d")
        path = 'dap2://data.cci.ceda.ac.uk/thredds/dodsC/esacci/lakes/data/lake_products/L3S/v2.1/merged_product/'
        path += f'{current_date.year}/{current_date.month:02}/'
        path += f'ESACCI-LAKES-L3S-LK_PRODUCTS-MERGED-{date_str}-fv{version}.0.nc?{varname}'
        dataset = xr.open_dataset(path, engine="pydap")
        # extract data in the defined zones
        dataset = dataset.isel(lat=slice(miny, maxy+1), lon=slice(minx, maxx+1))
        filval = dataset[varname].encoding.get('_FillValue', np.nan)
        data = dataset[varname][0,:,:].values.copy()
        dataset.close()
        units  = dataset[varname].units
        data[data == filval] = np.nan
        data[mask_lake == 0] = np.nan
        if np.isnan(data).all() :
            continue
        date_vec.append(date_str)
        data_vec.append(np.nanmean(data))
        df = pd.DataFrame({'date': date_vec, varname: data_vec})
        df['date'] = pd.to_datetime(df['date'], format='%Y%m%d')
    return df, units

def daily_to_monthly(df: pd.DataFrame, var: str, min_obs_per_month=3) -> pd.DataFrame:
    """ Convert a sparse dataframe of a given daily variable into a
        monthly dataframe.
    """
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    monthly_counts = df.groupby(["year","month"]).size()
    complete_months = monthly_counts[monthly_counts >= min_obs_per_month].index
    df_midx = pd.MultiIndex.from_arrays([df["year"], df["month"]])
    monthly_df = (
    df[df_midx.isin(complete_months)]
    .groupby(["year","month"])[var]
    .mean()
    .reset_index()
    )
    return monthly_df

def monthly_to_annual(df: pd.DataFrame, var: str, min_obs_per_year=11) -> pd.DataFrame:
    """Convert a monthly dataframe into an annual one.
    """
    annual_counts = df.groupby(["year"]).size()
    complete_years = annual_counts[annual_counts >= min_obs_per_year].index
    annual_df = (
    df[df["year"].isin(complete_years)]
    .groupby("year")[var]
    .mean()
    .reset_index()
    )
    return annual_df
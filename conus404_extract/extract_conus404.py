import calendar
import os

import dask
import dask.distributed
import numpy as np
import pandas as pd
import xarray as xr
from dask.distributed import Client


def extract_conus404(stations, nc_data, out_data, workers=8, overwrite=False, bounds=None,
                     debug=False):
    station_list = pd.read_csv(stations)
    if 'LAT' in station_list.columns:
        station_list = station_list.rename(columns={'STAID': 'fid', 'LAT': 'latitude', 'LON': 'longitude'})
    station_list.index = station_list['fid']

    client = Client(n_workers=workers,
                    memory_limit='256GB')

    if bounds:
        w, s, e, n = bounds
        station_list = station_list[(station_list['latitude'] < n) & (station_list['latitude'] >= s)]
        station_list = station_list[(station_list['longitude'] < e) & (station_list['longitude'] >= w)]
    else:
        ln = station_list.shape[0]
        w, s, e, n = (-125.0, 25.0, -67.0, 53.0)
        station_list = station_list[(station_list['latitude'] < n) & (station_list['latitude'] >= s)]
        station_list = station_list[(station_list['longitude'] < e) & (station_list['longitude'] >= w)]
        print('dropped {} stations outside NLDAS-2 extent'.format(ln - station_list.shape[0]))

    print(f'{len(station_list)} stations to write')

    station_list = station_list.rename(columns={'latitude': 'lat', 'longitude': 'lon'})
    indexer = station_list[['lat', 'lon']].to_xarray()
    fids = np.unique(indexer.fid.values).tolist()

    dates = [(year, month, calendar.monthrange(year, month)[-1])
             for year in range(1990, 2020) for month in range(1, 13)]
    variables = ['T2', 'TD2', 'QVAPOR', 'U10', 'V10', 'PSFC', 'ACSWDNLSM']
    ds = xr.open_zarr(nc_data, consolidated=True)
    ds = ds[variables]

    if debug:
        for date in dates:
            get_month_met(ds, fids, date, out_data, overwrite, )
        return
    else:
        delayed_results = [dask.delayed(get_month_met)(ds, fids, dt, out_data, overwrite)
                           for dt in dates]
        dask.compute(*delayed_results)


def get_month_met(ds_subset, staids, date_, out_data, overwrite):
    year, month, month_end = date_
    ds_subset = ds_subset.sel(time=slice(f'{year}-{month}-01', f'{year}-{month}-{month_end}'))
    feature_ids = [x[0] for x in staids]
    ds_subset = ds_subset.sel(feature_id=xr.DataArray(feature_ids, dims='station'))
    ds_subset = ds_subset.assign_coords(feature_id=feature_ids)
    date_string = f'{year}{month}'
    for staid in staids:
        feature_id, station_id = staid
        dst_dir = os.path.join(out_data, 'monthly', station_id)
        if not os.path.exists(dst_dir):
            os.mkdir(dst_dir)
        _file = os.path.join(dst_dir, '{}_{}.csv'.format(station_id, date_string))

        if not os.path.exists(_file) or overwrite:
            df_station = ds_subset.sel(feature_id=feature_id)['streamflow'].to_dataframe()
            df_station = df_station.groupby(df_station.index.get_level_values('time')).first()
            df_station['dt'] = [i.strftime('%Y%m%d%H') for i in df_station.index]
            df_station.to_csv(_file, index=False)
            print(_file)


if __name__ == '__main__':
    r = '/caldera/hovenweep/projects/usgs/water'
    d = os.path.join(r, 'wymtwsc', 'dketchum')
    c404 = os.path.join(d, 'conus404')
    dads = os.path.join(d, 'dads')
    ghcn = os.path.join(d, 'climate', 'ghcn')

    zarr_store = os.path.join(r, 'impd/hytest/conus404/conus404_hourly.zarr')

    # sites = ghcn_CANUSA_stations_mgrs.csv'
    sites = 'madis_29OCT2024.csv'

    csv_files = os.path.join(c404, 'station_data')
    p_files = '/data/ssd2/nldas2/parquet/'

    extract_conus404(sites, zarr_store, csv_files, workers=24)

# ========================= EOF ====================================================================

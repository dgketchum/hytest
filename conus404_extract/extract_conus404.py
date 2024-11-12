import calendar
import os
import concurrent.futures
import numpy as np
import pandas as pd
import xarray as xr
import xoak
import dask
from dask.diagnostics import ProgressBar
from dask.distributed import Client, LocalCluster


def extract_conus404(stations, nc_data, out_data, workers=8, overwrite=False, bounds=None,
                     debug=False):
    station_list = pd.read_csv(stations)
    if 'LAT' in station_list.columns:
        station_list = station_list.rename(columns={'STAID': 'fid', 'LAT': 'latitude', 'LON': 'longitude'})
    station_list.index = station_list['fid']

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

    print(station_list[station_list.index.duplicated()])

    station_list = station_list.head(n=100)
    station_list = station_list.to_xarray()
    fids = np.unique(station_list.fid.values).tolist()

    dates = [(year, month, calendar.monthrange(year, month)[-1])
             for year in range(1990, 2020) for month in range(1, 13)]
    variables = ['T2', 'TD2', 'QVAPOR', 'U10', 'V10', 'PSFC', 'ACSWDNLSM']

    ds = xr.open_zarr(nc_data, consolidated=True)
    ds = ds[variables]

    ds.xoak.set_index(['lat', 'lon'], 'sklearn_geo_balltree')
    with ProgressBar(), dask.config.set(scheduler='processes'):
        ds = ds.xoak.sel(lat=station_list.latitude, lon=station_list.longitude)
    ds = xr.merge([station_list, ds])

    if debug:
        for date in dates:
            get_month_met(ds, fids, date, out_data, overwrite)
        return
    else:
        with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:

            futures = [executor.submit(get_month_met, ds, fids, dt, out_data, overwrite)
                       for dt in dates]
            concurrent.futures.wait(futures)


def get_month_met(ds_subset, fids, date_, out_data, overwrite):
    year, month, month_end = date_
    ds_subset = ds_subset.sel(time=slice(f'{year}-{month}-01', f'{year}-{month}-{month_end}'))
    date_string = f'{year}{month}'
    all_df = ds_subset.to_dataframe()
    for fid in fids:

        dst_dir = os.path.join(out_data, fid)
        if not os.path.exists(dst_dir):
            os.mkdir(dst_dir)
        _file = os.path.join(dst_dir, f'{fid}_{date_string}.csv')

        if not os.path.exists(_file) or overwrite:
            df_station = all_df.loc[slice(fid), slice(None)].copy()
            df_station = df_station.groupby(df_station.index.get_level_values('time')).first()
            df_station['dt'] = [i.strftime('%Y%m%d%H') for i in df_station.index]
            df_station.to_csv(_file, index=False)
            print(os.path.basename(_file))


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
    extract_conus404(sites, zarr_store, csv_files, workers=24, debug=False)

# ========================= EOF ====================================================================

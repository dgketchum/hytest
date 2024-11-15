import calendar
import logging
import os
import concurrent.futures
import numpy as np
import pandas as pd
import xarray as xr
import xoak
import dask
from dask.diagnostics import ProgressBar
from dask.distributed import Client, LocalCluster


def extract_conus404(stations, nc_data, out_data, workers=8, overwrite=False, bounds=None, mode='multi',
                     start_yr=2000, end_yr=2023):
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

    dates = [(year, month, calendar.monthrange(year, month)[-1])
             for year in range(start_yr, end_yr + 1) for month in range(1, 13)]

    if mode == 'debug':
        for date in dates:
            get_month_met(nc_data, station_list, date, out_data, overwrite)
        return

    elif mode == 'multi':
        with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(get_month_met, nc_data, station_list, dt, out_data, overwrite)
                       for dt in dates]
            concurrent.futures.wait(futures)

    elif mode == 'dask':
        cluster = LocalCluster(n_workers=workers, memory_limit='32GB', threads_per_worker=1,
                               silence_logs=logging.ERROR)
        client = Client(cluster)
        print("Dask cluster started with dashboard at:", client.dashboard_link)
        station_list = client.scatter(station_list)
        tasks = [dask.delayed(get_month_met)(nc_data, station_list, date, out_data, overwrite) for date in dates]
        dask.compute(*tasks)
        client.close()


def get_month_met(nc_data_, station_list_, date_, out_data, overwrite):
    """"""
    year, month, month_end = date_
    date_string = '{}-{}'.format(year, str(month).rjust(2, '0'))

    variables = ['T2', 'TD2', 'QVAPOR', 'U10', 'V10', 'PSFC', 'ACSWDNLSM']
    fids = station_list_.index.to_list()
    station_list_ = station_list_.to_xarray()

    print(f'read zarr {date_string} ')
    ds = xr.open_zarr(nc_data_, consolidated=True, chunks='auto')
    print(f'select time {date_string} ')
    ds = ds.sel(time=slice(f'{year}-{month}-01', f'{year}-{month}-{month_end}'))
    ds = ds[variables]
    print(f'index stations {date_string} ')
    ds.xoak.set_index(['lat', 'lon'], 'sklearn_geo_balltree')
    ds = ds.xoak.sel(lat=station_list_.latitude, lon=station_list_.longitude)
    ds = xr.merge([station_list_, ds])

    all_df = ds.to_dataframe()
    print(f'write {date_string} from dataframe...')

    try:
        ct = 0
        for fid in fids:
            dst_dir = os.path.join(out_data, fid)
            if not os.path.exists(dst_dir):
                os.mkdir(dst_dir)
            _file = os.path.join(dst_dir, f'{fid}_{date_string}.parquet')
            if not os.path.exists(_file) or overwrite:
                df_station = all_df.loc[slice(fid), slice(None)].copy()
                df_station = df_station.groupby(df_station.index.get_level_values('time')).first()
                df_station['dt'] = [i.strftime('%Y%m%d%H') for i in df_station.index]
                df_station.to_parquet(_file, index=False)
                ct += 1
        if ct % 1000 == 0.:
            print(f'{ct} for {date_string}')
    except Exception as exc:
        print(f'{date_string}: {exc}')

    del ds


def get_quadrants(b):
    mid_longitude = (b[0] + b[2]) / 2
    mid_latitude = (b[1] + b[3]) / 2
    quadrant_nw = (b[0], mid_latitude, mid_longitude, b[3])
    quadrant_ne = (mid_longitude, mid_latitude, b[2], b[3])
    quadrant_sw = (b[0], b[1], mid_longitude, mid_latitude)
    quadrant_se = (mid_longitude, b[1], b[2], mid_latitude)
    quadrants = [quadrant_nw, quadrant_ne, quadrant_sw, quadrant_se]
    return quadrants


if __name__ == '__main__':
    r = '/caldera/hovenweep/projects/usgs/water'
    d = os.path.join(r, 'wymtwsc', 'dketchum')
    c404 = os.path.join(d, 'conus404')
    dads = os.path.join(d, 'dads')
    ghcn = os.path.join(d, 'climate', 'ghcn')

    zarr_store = os.path.join(r, 'impd/hytest/conus404/conus404_hourly.zarr')
    sites = os.path.join(dads, 'met', 'stations', 'madis_29OCT2024.csv')
    csv_files = os.path.join(c404, 'station_data')

    bounds = (-125.0, 25.0, -67.0, 53.0)
    quadrants = get_quadrants(bounds)
    sixteens = [get_quadrants(q) for q in quadrants]
    sixteens = [x for xs in sixteens for x in xs]

    for e, sector in enumerate(sixteens, start=1):

        print(f'\n\n\n\n Sector {e} of {len(sixteens)} \n\n\n\n')

        extract_conus404(sites, zarr_store, csv_files, workers=36, mode='dask')

# ========================= EOF ====================================================================

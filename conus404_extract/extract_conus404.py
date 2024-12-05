import calendar
import logging
import os
import intake
import concurrent.futures
import numpy as np
import pandas as pd
import xarray as xr
import dask
import zarr
from dask.diagnostics import ProgressBar
from dask.distributed import Client, LocalCluster
import warnings
import pyproj
import cartopy.crs as ccrs

warnings.filterwarnings("ignore",
                        message="The return type of `Dataset.dims` will be changed to return a set of dimension names in future")


def extract_conus404(stations, out_data, workers=8, overwrite=False, bounds=None, mode='multi',
                     start_yr=2000, end_yr=2022, output_target='uncorrected'):
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
    print(f'sample stations for the selected region:\n {station_list.sample(n=5)}')

    dates = [(year, month, calendar.monthrange(year, month)[-1])
             for year in range(start_yr, end_yr + 1) for month in range(1, 13)]

    if mode == 'debug':
        for date in dates:
            get_month_met(station_list, date, out_data, overwrite, bounds, output_target)

    elif mode == 'multi':
        with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
            futures = [
                executor.submit(get_month_met, station_list, dt, out_data, overwrite, bounds, output_target)
                for dt in dates]
            concurrent.futures.wait(futures)

    elif mode == 'dask':
        cluster = LocalCluster(n_workers=workers, memory_limit='32GB', threads_per_worker=1,
                               silence_logs=logging.ERROR)
        client = Client(cluster)
        print("Dask cluster started with dashboard at:", client.dashboard_link)
        station_list = client.scatter(station_list)
        tasks = [dask.delayed(get_month_met)(station_list, date, out_data, overwrite, bounds, output_target)
                 for date in dates]
        dask.compute(*tasks)
        client.close()


def get_month_met(station_list_, date_, out_data, overwrite, bounds_=None, output_mode='uncorrected'):
    """"""
    import xoak
    year, month, month_end = date_

    # dataset 1979 to 2022-10-01
    if year == 2022 and month > 9:
        return
    date_string = '{}-{}'.format(year, str(month).rjust(2, '0'))

    fids = station_list_.index.to_list()

    hytest_cat = intake.open_catalog(
        "https://raw.githubusercontent.com/hytest-org/hytest/main/dataset_catalog/hytest_intake_catalog.yml")
    cat = hytest_cat['conus404-catalog']
    if output_mode == 'uncorrected':
        # model output, uncorrected
        dataset = 'conus404-hourly-onprem-hw'
        variables = ['T2', 'TD2', 'QVAPOR', 'U10', 'V10', 'PSFC', 'ACSWDNLSM']
        print('using uncorrected data')
    elif output_mode == 'ba':
        variables = ['RAINRATE', 'T2D']
        # bias-adjusted for precip and temp
        dataset = 'conus404-hourly-ba-onprem-hw'
        print('using bias-adjusted data')
    else:
        raise ValueError('output_mode not recognized')

    ds = cat[dataset].to_dask()
    # extract crs meta before continuing to modify ds
    bounds_proj = projected_coords(row=None, _bounds=bounds, buffer=10000.)
    ds = ds.sel(time=slice(f'{year}-{month}-01', f'{year}-{month}-{month_end}'))
    ds = ds[variables]
    if bounds_ is not None:
        ds = ds.sel(y=slice(bounds_proj[1], bounds_proj[3]),
                    x=slice(bounds_proj[0], bounds_proj[2]))
    if output_mode == 'uncorrected':
        station_list_ = station_list_.to_xarray()
        # Add 'within_bounds' column (lat/lon)
        station_list_['within_bounds'] = (
                (station_list_.latitude >= ds.lat.min()) &
                (station_list_.latitude <= ds.lat.max()) &
                (station_list_.longitude >= ds.lon.min()) &
                (station_list_.longitude <= ds.lon.max()))
        ds.xoak.set_index(['lat', 'lon'], 'sklearn_geo_balltree')
        ds = ds.xoak.sel(lat=station_list_.latitude, lon=station_list_.longitude, tolerance=4000)
    else:
        station_list_[['xs', 'ys']] = station_list_.apply(projected_coords, axis=1, result_type='expand')
        station_list_ = station_list_.to_xarray()
        # Add 'within_bounds' column (x/y)
        station_list_['within_bounds'] = (
                (station_list_.ys >= ds.y.min()) &
                (station_list_.ys <= ds.y.max()) &
                (station_list_.xs >= ds.x.min()) &
                (station_list_.xs <= ds.x.max()))
        try:
            ds = ds.sel(y=station_list_.ys, x=station_list_.xs, method='nearest', tolerance=4000)
        except KeyError as exc:
            print(f"KeyError: {exc}")
            print(bounds_proj)
            print(bounds)
            print("Problematic y values:", np.array(station_list_.ys)[~np.isin(station_list_.ys, ds.y)])
            print(ds)
            return

    ds = xr.merge([station_list_, ds])
    all_df = ds.to_dataframe()
    print(f'to data frame {date_string}')

    try:
        ct = 0
        for enum, fid in enumerate(fids, start=1):
            print(f'{enum}: {ct} of {len(fids)}')
            dst_dir = os.path.join(out_data, fid)
            if not os.path.exists(dst_dir):
                os.mkdir(dst_dir)
            _file = os.path.join(dst_dir, f'{fid}_{date_string}.parquet')
            if not os.path.exists(_file) or overwrite:
                print(f'extract {fid} for {date_string}')
                df_station = all_df.loc[slice(fid), slice(None)].copy()
                df_station = df_station.groupby(df_station.index.get_level_values('time')).first()
                df_station['dt'] = [i.strftime('%Y%m%d%H') for i in df_station.index]
                df_station.to_parquet(_file, index=False)
                ct += 1
        if ct % 1000 == 0.:
            print(f'{ct} of {len(fids)} for {date_string}')
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


def projected_coords(row, _bounds=None, buffer=None):
    globe = ccrs.Globe(ellipse='sphere', semimajor_axis=6370000, semiminor_axis=6370000)
    lcc = ccrs.LambertConformal(globe=globe,
                                central_longitude=-97.9000015258789,
                                central_latitude=39.100006103515625,
                                standard_parallels=[30.0, 50.0])
    lcc_wkt = lcc.to_wkt()
    source_crs = 'epsg:4326'
    transformer = pyproj.Transformer.from_crs(source_crs, lcc_wkt)
    if _bounds is not None:
        west, south, east, north = _bounds
        sw_x, sw_y = transformer.transform(south, west)
        ne_x, ne_y = transformer.transform(north, east)
        if buffer:
            return sw_x - buffer, sw_y - buffer, ne_x + buffer, ne_y + buffer
        else:
            return sw_x, sw_y, ne_x, ne_y
    else:
        x, y = transformer.transform( row['latitude'], row['longitude'])
        return x, y



if __name__ == '__main__':
    r = '/caldera/hovenweep/projects/usgs/water'
    d = os.path.join(r, 'wymtwsc', 'dketchum')

    if not os.path.isdir(d):
        home = os.path.expanduser('~')
        d = os.path.join(home, 'data', 'IrrigationGIS')

    c404 = os.path.join(d, 'conus404')
    dads = os.path.join(d, 'dads')
    ghcn = os.path.join(d, 'climate', 'ghcn')

    zarr_store = os.path.join(r, 'impd/hytest/conus404/conus404_hourly.zarr')
    sites = os.path.join(dads, 'met', 'stations', 'madis_29OCT2024.csv')
    # sites = os.path.join(ghcn, 'stations', 'ghcn_CANUSA_stations_mgrs.csv')

    model_target = 'ba'
    if model_target == 'ba':
        csv_files = os.path.join(c404, 'station_data_ba')
    else:
        csv_files = os.path.join(c404, 'station_data')

    bounds = (-125.0, 25.0, -67.0, 53.0)
    quadrants = get_quadrants(bounds)
    sixteens = [get_quadrants(q) for q in quadrants]
    sixteens = [x for xs in sixteens for x in xs]

    for e, sector in enumerate(sixteens, start=1):
        print(f'\n\n\n\n Sector {e} of {len(sixteens)} \n\n\n\n')

        if e < 3:
            continue

        extract_conus404(sites, csv_files, workers=18, mode='dask', bounds=sector, output_target=model_target)

# ========================= EOF ====================================================================

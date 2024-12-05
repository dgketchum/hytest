import concurrent.futures
import os
import json
import shutil
from datetime import datetime

import pandas as pd
import numpy as np


def process_and_concat_csv(stations, root, start_date, end_date, outdir, workers, missing_file=None,
                           debug=False):
    start = pd.to_datetime(start_date)
    end = pd.to_datetime(end_date)
    expected_index = pd.date_range(start=start, end=end, freq='h')

    station_list = pd.read_csv(stations)
    if 'LAT' in station_list.columns:
        station_list = station_list.rename(columns={'STAID': 'fid', 'LAT': 'latitude', 'LON': 'longitude'})
    w, s, e, n = (-125.0, 25.0, -67.0, 53.0)
    station_list = station_list[(station_list['latitude'] < n) & (station_list['latitude'] >= s)]
    station_list = station_list[(station_list['longitude'] < e) & (station_list['longitude'] >= w)]

    station_list = station_list.sample(frac=1)
    subdirs = station_list['fid'].to_list()

    print(f'{len(subdirs)} directories to check')

    if missing_file:
        for sd in subdirs:
            conus404_parquet(root, sd, expected_index, outdir)

    if debug:
        for subdir in subdirs:
            conus404_parquet(root, subdir, expected_index, outdir)
    else:
        with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
            executor.map(conus404_parquet, [root] * len(subdirs), subdirs,
                         [expected_index] * len(subdirs), [outdir] * len(subdirs))

def conus404_parquet(root_, subdir_, expected_index_, outdir_):
    subdir_path = os.path.join(root_, subdir_)
    out_file = os.path.join(outdir_, f'{subdir_}.parquet.gzip')

    if os.path.isdir(subdir_path):
        csv_files_ = [f for f in os.listdir(subdir_path) if f.endswith('.parquet')]
        # if os.path.exists(out_file) and csv_files_:
        #     shutil.rmtree(subdir_path)
        #     print(f'{os.path.basename(out_file)} exists, removing {len(csv_files)} csv files')
        #     return

        dtimes = [f.split('_')[-1].replace('.csv', '') for f in csv_files_]
        rm_files = csv_files_.copy()
        required_years_ = sorted(list(set([i.year for i in expected_index_])))

        if len(dtimes) < len(required_years_):
            missing = [m for m in required_years_ if m not in dtimes]
            if len(missing) > 0:
                print(f'{subdir_} missing {len(missing)} months: {np.random.choice(missing, size=5, replace=False)}')
                return

        dfs = []
        for file in csv_files_:
            c = pd.read_parquet(os.path.join(subdir_path, file))
            dfs.append(c)
        df = pd.concat(dfs)
        df = df.drop_duplicates(subset='dt', keep='first')
        df = df.set_index('dt').sort_index()
        missing = len(expected_index_) - df.shape[0]
        if missing > 15:
            print(f'{subdir_} is missing {missing} records')
            return

        df['dt'] = df.index
        df.to_parquet(out_file, compression='gzip')
        # shutil.rmtree(subdir_path)
        return
        
    else:
        if os.path.exists(out_file):
            print(f'{os.path.basename(out_file)} exists, skipping')
        else:
            print(f'{subdir_} not found')


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

    model_target = 'uncorrected'
    if model_target == 'ba':
        csv_files = os.path.join(c404, 'station_data_ba')
        p_files = os.path.join(c404, 'parquet_ba')
    else:
        csv_files = os.path.join(c404, 'station_data')
        p_files = os.path.join(c404, 'parquet')

    process_and_concat_csv(sites, csv_files, start_date='2000-01-01', end_date='2022-09-30', outdir=p_files,
                           workers=12, missing_file=None, debug=False)

# ========================= EOF ====================================================================

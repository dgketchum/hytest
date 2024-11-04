import calendar
import concurrent
import os
from datetime import datetime

# import boto3
# import botocore
# import fsspec
import geopandas as gpd
import numpy as np
import xarray as xr
from scipy.spatial import cKDTree


# from dask.distributed import Client


def list_s3_bucket_contents(bucket_name='noaa-nwm-retro-v2-zarr-pds'):
    s3 = boto3.client('s3', config=botocore.client.Config(signature_version=botocore.UNSIGNED))

    paginator = s3.get_paginator('list_objects_v2')
    for page in paginator.paginate(Bucket=bucket_name):
        for obj in page.get('Contents', []):
            print(obj['Key'])


def download_nwm_data(metadata, out_data, workers=8, overwrite=False, debug=False, bounds=None, location=None):
    gages = gpd.read_file(metadata)
    gages['latitude'] = gages['geometry'].y
    gages['longitude'] = gages['geometry'].x
    ln = gages.shape[0]
    print(f'{ln} gages')

    if bounds:
        w, s, e, n = bounds
        gages = gages[(gages['latitude'] < n) & (gages['latitude'] >= s)]
        gages = gages[(gages['longitude'] < e) & (gages['longitude'] >= w)]
        print('dropped {} stations outside bounds'.format(ln - gages.shape[0]))

    else:
        w, s, e, n = (-125.0, 25.0, -67.0, 53.0)
        gages = gages[(gages['latitude'] < n) & (gages['latitude'] >= s)]
        gages = gages[(gages['longitude'] < e) & (gages['longitude'] >= w)]
        print('dropped {} stations outside NLDAS-2 extent'.format(ln - gages.shape[0]))

    if location:
        ds = xr.open_dataset(location)
    else:
        client = Client(n_workers=8)
        url = 's3://noaa-nwm-retro-v2-zarr-pds'
        ds = xr.open_zarr(fsspec.get_mapper(url, anon=True), consolidated=True)

    ds_latitudes = ds['latitude'].values
    ds_longitudes = ds['longitude'].values
    feature_ids = ds['feature_id'].values
    tree = cKDTree(np.c_[ds_latitudes, ds_longitudes])
    distances, indices = tree.query(np.c_[gages['latitude'], gages['longitude']])

    nearest_feature_ids = feature_ids[indices]
    nearest_feature_lats = ds_latitudes[indices]
    nearest_feature_lons = ds_longitudes[indices]
    gages['feature_ids'] = nearest_feature_ids
    gages['feature_lats'] = nearest_feature_lats
    gages['feature_lons'] = nearest_feature_lons
    gages['distance_m'] = gages.apply(lambda r: haversine_distance(r['latitude'], r['longitude'],
                                                                   r['feature_lats'], r['feature_lons']), axis=1)
    out_csv = os.path.join(out_data, 'nearest.csv')
    gages.to_csv(out_csv)

    gage_ids = gages['staid'].values

    staids = [(nearest_feature_ids[i], gage_ids[i]) for i in range(len(gage_ids))]
    ds_subset = ds.sel(feature_id=xr.DataArray(nearest_feature_ids, dims='station'))
    ds_subset = ds_subset.assign_coords(feature_id=nearest_feature_ids)

    print(f'{len(staids)} stations to process')
    print(ds_subset)

    for year in range(1990, 2020):

        for month in range(1, 13):

            month_start = datetime(year, month, 1)
            month_end = calendar.monthrange(year, month)[-1]
            date_string = month_start.strftime('%Y%m')
            mds = ds_subset.sel(time=slice(f'{year}-{month}-01', f'{year}-{month}-{month_end}'))

            if debug:
                for staid in staids:
                    process_fid(staid, mds, date_string, out_data, overwrite)
            else:
                with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
                    futures = [executor.submit(process_fid, staid, mds, date_string, out_data, overwrite)
                               for staid in staids]
                    concurrent.futures.wait(futures)


def process_fid(fid, ds, yearmo, out_data, overwrite):
    feature_id, station_id = fid
    dst_dir = os.path.join(out_data, station_id)
    if not os.path.exists(dst_dir):
        os.mkdir(dst_dir)
    _file = os.path.join(dst_dir, '{}_{}.csv'.format(station_id, yearmo))

    if not os.path.exists(_file) or overwrite:
        df_station = ds.sel(feature_id=feature_id)['streamflow'].to_dataframe()
        df_station = df_station.groupby(df_station.index.get_level_values('time')).first()
        df_station['dt'] = [i.strftime('%Y%m%d%H') for i in df_station.index]
        df_station.to_csv(_file, index=False)
        print(_file)


def haversine_distance(lat1_, lon1_, lat2_, lon2_):
    R = 6371
    dlat = np.radians(lat2_ - lat1_)
    dlon = np.radians(lon2_ - lon1_)
    a = np.sin(dlat / 2) * np.sin(dlat / 2) + \
        np.cos(np.radians(lat1_)) * np.cos(np.radians(lat2_)) * \
        np.sin(dlon / 2) * np.sin(dlon / 2)
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    distance = R * c
    return distance


if __name__ == '__main__':
    home = os.path.expanduser('~')

    d = '/media/research/IrrigationGIS'
    if not os.path.isdir(d):
        home = os.path.expanduser('~')
        d = os.path.join(home, 'data', 'IrrigationGIS')

    out_dir = os.path.join(home, 'nwm_hydrographs')
    metadata_ = os.path.join(home, 'gage_info', 'selected-streamflow-data-meta-merged.geojson')
    # download_nwm_data(metadata_, out_dir, workers=8, debug=True, bounds=(-115.0, 42.0, -112.0, 47.0))

    metadata_ = 'selected-streamflow-data-meta-merged.geojson'
    out_dir = os.path.join(d, 'usgs_gages', 'nwm')
    loc = os.path.join(home, 'Downloads', 'nwmv21_nwis.nc')
    download_nwm_data(metadata_, out_dir, location=loc, bounds=(-115.0, 42.0, -112.0, 47.0))

# ========================= EOF ====================================================================

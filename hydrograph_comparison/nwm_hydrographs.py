import calendar
import os

import dask
import fsspec
import geopandas as gpd
import numpy as np
import pandas as pd
import xarray as xr
from dask.distributed import Client
from scipy.spatial import cKDTree

COLS = ['staid', 'feature_id', 'latitude', 'longitude', 'distance_km', 'gnis_name']
URL = 's3://noaa-nwm-retro-v2-zarr-pds'


def extract_nwm_data(metadata, out_data, nwm_fabric, workers=8, overwrite=False, debug=False, bounds=None,
                     complete=None):
    """
    This function takes a list of USGS gages, finds the closest National Water Model stream segment, and
    extracts the daily streamflow NWM model output at that reach, month to month. This should be run in an
    AWS EC2 instance (us-west-2). This is intended to follow the use of prepped_usgs_nwm.py, which access pre-processed
    time series of the same model output that has been clipped and rechunked for streamflow extraction. This codes
    uses the entire NWM output and is thus much slower. Further, the results of this function should be inspected in
    'usgs_nwm_matches_unverified.csv' or  'usgs_nwm_matches_unverified_subsel.csv', as the proximity selection
    is not guarateed accurate, owing to the 'nwm_fabric' using single coordinate pairs to locate a stream segment
    that represents a line feature, and the selection of flowlines themselves by NHD and NWM.

    Args:
        metadata: Geojson or shapefile of sought gages.
        out_data: Ouput directory.
        nwm_fabric: NWM fabric has GNIS names th
        workers:
        overwrite:
        debug:
        bounds:
        complete: List of gages that were already extracted in usgs_nwm_index.py.

    Returns:

    """
    nwm = pd.read_csv(nwm_fabric, index_col='feature_id')

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

    client = Client(n_workers=workers)
    ds = xr.open_zarr(fsspec.get_mapper(URL, anon=True), consolidated=True)

    ds_latitudes = ds['latitude'].values
    ds_longitudes = ds['longitude'].values
    feature_ids = ds['feature_id'].values
    tree = cKDTree(np.c_[ds_latitudes, ds_longitudes])
    distances, indices = tree.query(np.c_[gages['latitude'], gages['longitude']])

    nearest_feature_ids = feature_ids[indices]
    nearest_feature_lats = ds_latitudes[indices]
    nearest_feature_lons = ds_longitudes[indices]
    gages['feature_id'] = nearest_feature_ids
    gages['feature_lats'] = nearest_feature_lats
    gages['feature_lons'] = nearest_feature_lons
    gages['distance_km'] = gages.apply(lambda r: haversine_distance(r['latitude'], r['longitude'],
                                                                    r['feature_lats'], r['feature_lons']), axis=1)

    gages['gnis_name'] = [nwm.loc[i, 'gnis_name'] for i in gages['feature_id']]
    print(gages.columns)
    gages = gages[COLS]

    out_csv = os.path.join(out_data, 'usgs_nwm_matches_unverified.csv')
    gages.to_csv(out_csv)
    print(f'{gages.shape[0]} gages matched by proximity')

    if complete:
        sub_idx = [i for i, r in gages.iterrows() if r['staid'] not in complete]
        print(len(sub_idx))
        gages = gages.loc[sub_idx]
        out_csv = os.path.join(out_data, 'usgs_nwm_matches_unverified_subsel.csv')
        gages.to_csv(out_csv)

    gage_ids = gages['staid'].values
    feature_ids = gages['feature_id'].values
    staids = [(f, g) for f, g in zip(feature_ids, gage_ids)]

    print(f'{len(staids)} stations to process')

    ds = None
    dates = [(year, month, calendar.monthrange(year, month)[-1])
             for year in range(1990, 2020) for month in range(1, 13)]

    if debug:
        get_month_flows(staids, dates[0], out_data, overwrite)
        return
    else:
        delayed_results = [dask.delayed(get_month_flows)(staids, dt, out_data, overwrite)
                           for dt in dates]
        dask.compute(*delayed_results)


def get_month_flows(staids, dates, out_data, overwrite):
    year, month, month_end = dates
    ds = xr.open_zarr(fsspec.get_mapper(URL, anon=True), consolidated=True)
    ds_subset = ds.sel(time=slice(f'{year}-{month}-01', f'{year}-{month}-{month_end}'))
    feature_ids = [x[0] for x in staids]
    ds_subset = ds_subset.sel(feature_id=xr.DataArray(feature_ids, dims='station'))
    ds_subset = ds_subset.assign_coords(feature_id=feature_ids)
    date_string = f'{year}{month}'
    for staid in staids:
        feature_id, station_id = staid
        dst_dir = os.path.join(out_data, station_id)
        if not os.path.exists(dst_dir):
            os.mkdir(dst_dir)
        _file = os.path.join(dst_dir, '{}_{}.csv'.format(station_id, date_string))

        if not os.path.exists(_file) or overwrite:
            df_station = ds_subset.sel(feature_id=feature_id)['streamflow'].to_dataframe()
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

    out_dir = os.path.join(d, 'nwm', 'hydrographs', 'monthly')
    csv = os.path.join(d, 'nwm', 'nwm_hydrofabric.csv')
    usgs_prepped = os.path.join(d, 'nwm', 'hydrographs', 'usgs_nwm_ID_matched_subselection.csv')
    proc_gages = pd.read_csv(usgs_prepped,  dtype={'staid': str})['staid'].tolist()

    metadata_ = os.path.join(d, 'nwm', 'selected-streamflow-data-meta-merged.geojson')
    extract_nwm_data(metadata_, out_dir, complete=proc_gages, bounds=None, nwm_fabric=csv, debug=True)

# ========================= EOF ====================================================================

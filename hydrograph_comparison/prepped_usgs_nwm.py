import concurrent
import os

import geopandas as gpd
import numpy as np
import pandas as pd
import xarray as xr
from scipy.spatial import cKDTree

COLS = ['SOURCE_FEA', 'FLComID', 'feature_ids', 'latitude', 'longitude',
        'distance_km', 'gnis_name']


def get_prepped_usgs_nwm(metadata, out_data, nc, nwm_fabric, workers=8, overwrite=False, debug=False,
                         bounds=None, subselection=None):
    """Extract NWM hydrographs from prepped nc data found at:
    https://www.sciencebase.gov/catalog/item/612e264ed34e40dd9c091228
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

    ds = xr.open_dataset(nc)

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
    gages['distance_km'] = gages.apply(lambda r: haversine_distance(r['latitude'], r['longitude'],
                                                                    r['feature_lats'], r['feature_lons']), axis=1)

    gages['gnis_name'] = [nwm.loc[i, 'gnis_name'] for i in gages['feature_ids']]
    gages = gages[COLS]
    gages = gages.rename(columns={'SOURCE_FEA': 'staid'})
    gages = gages[gages['FLComID'] == gages['feature_ids']]
    out_csv = os.path.join(out_data, 'usgs_nwm_ID_matched.csv')
    gages.to_csv(out_csv)

    gage_ids = gages['staid'].values
    feature_ids = gages['feature_ids'].values
    staids = [(f, g) for f, g in zip(feature_ids, gage_ids)]

    if subselection:
        staids = [t for t in staids if t[1] in subselection]
        sub_idx = [i for i, r in gages.iterrows() if r['staid'] in subselection]
        gages = gages.loc[sub_idx]
        out_csv = os.path.join(out_data, 'usgs_nwm_ID_matched_subselection.csv')
        gages.to_csv(out_csv, index=False)

    if debug:
        for staid in staids:
            process_fid(staid, ds, out_data, overwrite)
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(process_fid, staid, ds, out_data, overwrite)
                       for staid in staids]
            concurrent.futures.wait(futures)


def process_fid(fid, ds, out_data, overwrite):
    feature_id, station_id = fid

    _file = os.path.join(out_data, 'blodgett_prepped', '{}.csv'.format(station_id))

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

    metadata_ = 'selected-streamflow-data-meta-merged.geojson'
    gages = gpd.read_file(metadata_)['staid'].tolist()

    metadata_ = os.path.join(d, 'usgs_gages', 'GageLoc', 'GageLoc.shp')
    out_dir = os.path.join(d, 'nwm', 'hydrographs')
    nc_ = os.path.join(d, 'nwm', 'output', 'nwmv21_nwis.nc')
    csv = os.path.join(d, 'nwm', 'nwm_hydrofabric.csv')
    get_prepped_usgs_nwm(metadata_, out_dir, nc_, csv, bounds=None, debug=False,
                         workers=8, subselection=gages)

# ========================= EOF ====================================================================

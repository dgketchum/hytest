import os

import geopandas as gpd
import pandas as pd


def write_unmatched(gages, matches, outshp):
    gages = gpd.read_file(gages, dtype={'staid': str})
    gages.index = gages['staid']
    nhm_poi = pd.read_csv(matches, index_col='staid', dtype={'staid': str})
    idx = [i for i in gages.index if i in nhm_poi.index]
    gages.loc[idx, 'usgs_poi'] = nhm_poi.loc[idx, 'is_poi']
    gages.drop(columns=['staid'], inplace=True)
    gages.to_file(outshp)


if __name__ == '__main__':

    if __name__ == '__main__':
        home = os.path.expanduser('~')

        d = '/media/research/IrrigationGIS'
        if not os.path.isdir(d):
            home = os.path.expanduser('~')
            d = os.path.join(home, 'data', 'IrrigationGIS')

        metadata_ = os.path.join(d, 'nwm', 'selected-streamflow-data-meta-merged.geojson')
        station_matches = os.path.join(d, 'nhm', 'station_list_annotated.csv')
        out_shape = os.path.join(d, 'nhm', 'matched_nhm_poi.shp')

        write_unmatched(metadata_, station_matches, out_shape)

# ========================= EOF ====================================================================

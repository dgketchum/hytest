import os

import dask
import dataretrieval.nwis as nwis
import geopandas as gpd
import holoviews as hv
import numpy as np
import pandas as pd
import xarray as xr
import zarr
import hvplot.xarray

from dask.distributed import Client
from zarr.convenience import consolidate_metadata

hv.extension('bokeh')
os.environ['USE_PYGEOS'] = '0'

START = '2023-01-01'
END = '2023-12-31'


def prepare_streamflow_data(metadata, outfile):
    gages = gpd.read_file(metadata)
    gages = gages.iloc[:2]
    gages['lat'] = gages['geometry'].y
    gages['lon'] = gages['geometry'].x

    date_range = pd.date_range(START, END, freq='D')
    dt_idx = pd.DatetimeIndex(date_range)

    dummy = xr.Dataset(
        {'streamflow': (('time', 'gage_id'), np.ones((len(dt_idx), len(gages))) * np.nan)},
        coords={'time': dt_idx,
                'gage_id': gages['staid'],
                'lat': gages['lat'],
                'lon': gages['lon']})

    store = zarr.DirectoryStore(outfile)

    dummy.to_zarr(
        store,
        consolidated=True,
        compute=False,
        encoding={'streamflow': {'_FillValue': np.nan},
                  'lat': {'_FillValue': np.nan},
                  'lon': {'_FillValue': np.nan}},
        mode='w')

    for j, (i, r) in enumerate(gages.iterrows()):
        write_one_gage(j, r, outfile)

    # client = Client()
    # results = dask.compute(*[dask.delayed(write_one_gage)(j, r, outfile) for
    #                          j, (i, r) in enumerate(gages.iterrows())], retries=10)
    # _ = consolidate_metadata(outfile)
    # client.close()


def write_one_gage(n, gage, outfile):
    obs = nwis.get_record(sites=gage['staid'], service='dv', start=START, end=END)
    try:
        obs = obs[['00060_Mean']]
    except KeyError as e:
        print(e, 'empty: ', obs.empty)
        return None

    obs[obs['00060_Mean'] < 0.0] = np.nan
    obs.rename(columns={'00060_Mean': 'streamflow'}, inplace=True)
    idx = pd.date_range(obs.index[0], obs.index[-1], freq='D')
    obs = obs.reindex(idx)

    if np.all(np.isnan(obs['streamflow'])):
        print(f'{gage['staid']} is all nan')
        return None

    obs['time'] = obs.index

    ds = xr.Dataset(
        {'streamflow': ('time', obs['streamflow'])},
        coords={'time': obs.index,
                'gage_id': gage['staid']})

    ds = ds.expand_dims('gage_id')

    mf = '{:.1f}'.format(ds.streamflow.mean().item())
    print(f'xarray {gage['staid']} mean flow: {mf}')

    ds.to_zarr(
        outfile,
        region={'time': slice(0, len(obs.index)),
                'gage_id': slice(n, n + 1)})


def verify(metadata, outfile, html_outfile):
    dst = xr.open_dataset(outfile, engine='zarr', chunks={}, backend_kwargs=dict(consolidated=True))
    gages = gpd.read_file(metadata)
    gage_ids = gages['staid'].to_list()

    for sid in gage_ids:
        flow = dst.sel(gage_id=sid).streamflow.values
        if np.isnan(flow).all():
            print(f"No data found for gage ID {sid}.")

        plot = dst['streamflow'].sel(gage_id=sid).hvplot(x='time', y='streamflow', grid=True)
        hv.save(plot, filename=html_outfile.format(sid))

if __name__ == "__main__":
    metadata_ = 'selected-streamflow-data-meta-merged.geojson'
    out_file_ = 'nwis.zarr'
    plot_file_ = 'hydrographs/hydrograph_{}.html'
    prepare_streamflow_data(metadata_, out_file_)
    verify(metadata_, out_file_, plot_file_)

# ========================= EOF ====================================================================

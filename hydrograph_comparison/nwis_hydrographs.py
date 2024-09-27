import os

import dask
import dataretrieval.nwis as nwis
import geopandas as gpd
import holoviews as hv
import numpy as np
import pandas as pd
import xarray as xr
import zarr
# doesn't appear but is used
import hvplot.xarray
import matplotlib.pyplot as plt

from dask.distributed import Client
from zarr.convenience import consolidate_metadata

hv.extension('bokeh')
os.environ['USE_PYGEOS'] = '0'

START = '2023-01-01'
END = '2023-12-31'


def prepare_streamflow_data(metadata, outfile):
    gages = gpd.read_file(metadata)
    gages = gages.iloc[:100]
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


def write_one_gage(n, gage, outfile):
    try:
        obs = nwis.get_record(sites=gage['staid'], service='dv', start=START, end=END)
    except ValueError:
        return
    try:
        obs = obs[['00060_Mean']].copy()
    except KeyError as e:
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
    ds.to_zarr(
        outfile,
        region={'time': slice(0, len(obs.index)),
                'gage_id': slice(n, n + 1)})
    print(gage['staid'])


def verify(metadata, outfile, html_outfile, write_html=True):
    dst = xr.open_dataset(outfile, engine='zarr', chunks={}, backend_kwargs=dict(consolidated=True))
    gages = gpd.read_file(metadata)
    gage_ids = gages['staid'].to_list()

    for sid in gage_ids:
        try:
            flow = dst.sel(gage_id=sid).streamflow.values
            if np.isnan(flow).all():
                # print(f"No data found for gage ID {sid}.")
                continue

            if write_html:
                plot = dst['streamflow'].sel(gage_id=sid).hvplot(x='time', y='streamflow', grid=True)
                hv.save(plot, filename=html_outfile.format(sid))
            else:
                fig, ax = plt.subplots(figsize=(10, 6))
                dst['streamflow'].sel(gage_id=sid).plot(ax=ax)
                ax.set_title(f'Hydrograph for Gage {sid}')
                ax.grid(True)
                plt.savefig(html_outfile.format(sid).replace('.html', '.png'))
                plt.close(fig)

        except KeyError as e:
            # print(e, sid, 'error')
            continue


if __name__ == "__main__":
    metadata_ = 'selected-streamflow-data-meta-merged.geojson'
    out_file_ = 'nwis.zarr'
    plot_file_ = 'hydrographs/hydrograph_{}.html'
    prepare_streamflow_data(metadata_, out_file_)
    # verify(metadata_, out_file_, plot_file_, write_html=False)

# ========================= EOF ====================================================================

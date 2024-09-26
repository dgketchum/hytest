import os

import numpy as np
import xarray as xr
from pygeohydro import NWIS
from zarr.convenience import consolidate_metadata
import dask
from dask.distributed import Client

os.environ['USE_PYGEOS'] = '0'


def prepare_streamflow_data(gages, outfile, start_time='1979-01-01', stop_time='2023-12-31'):
    date_range = (start_time, stop_time)

    nwis = NWIS()
    observed = nwis.get_streamflow(gages[0:2], date_range, to_xarray=True)

    observed = (observed
                .rename_dims({'station_id': 'gage_id'})
                .rename({'discharge': 'streamflow', 'station_id': 'gage_id'})
                )

    source_dataset = observed
    template = (xr.zeros_like(source_dataset)
    .chunk()
    .isel(gage_id=0, drop=True)
    .expand_dims(gage_id=len(gages), axis=-1)
    .assign_coords({'gage_id': gages})
    .chunk({
        'time': len(observed.time),
        'gage_id': 1}
    )
    )

    template.to_zarr(
        outfile,
        compute=False,
        encoding={
            'station_nm': dict(_FillValue=None, dtype='<U64'),
            'alt_datum_cd': dict(_FillValue=None, dtype='<U6'),
            'alt_acy_va': dict(_FillValue=-2147483647, dtype=np.int32),
            'alt_va': dict(_FillValue=9.96921e+36, dtype=np.float32),
            'dec_lat_va': dict(_FillValue=None, dtype=np.float32),
            'dec_long_va': dict(_FillValue=None, dtype=np.float32),
            'streamflow': dict(_FillValue=9.96921e+36, dtype=np.float32)
        },
        consolidated=True,
        mode='w'
    )

    n_timesteps = len(observed.time)
    time_steps = observed.time.values
    client = Client()
    results = dask.compute(*[dask.delayed(write_one_gage)(i, gages, date_range, time_steps, n_timesteps, outfile)
                             for i in range(len(gages))], retries=10)
    _ = consolidate_metadata(outfile)
    client.close()


def write_one_gage(n, gages, date_range, time_steps, n_timesteps, outfile):
    site_id = gages[n]
    try:
        nwis = NWIS()

        _obs = nwis.get_streamflow(site_id, date_range, to_xarray=True).interp(time=time_steps)
        _obs = _obs.rename_dims({'station_id': 'gage_id'}).rename({'station_id': 'gage_id', 'discharge': 'streamflow'})
        _obs['station_nm'] = xr.DataArray(data=_obs['station_nm'].values.astype('<U64'), dims='gage_id')
        _obs['alt_datum_cd'] = xr.DataArray(data=_obs['alt_datum_cd'].values.astype('<U6'), dims='gage_id')

        _obs.to_zarr(
            outfile,
            region={
                'time': slice(0, n_timesteps),
                'gage_id': slice(n, n + 1)
            }
        )
        return n
    except Exception as e:
        pass


# Main execution flow
if __name__ == "__main__":
    os.environ['USE_PYGEOS'] = '0'

# ========================= EOF ====================================================================

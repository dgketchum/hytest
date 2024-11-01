import boto3
import botocore
import dask
import fsspec
import geopandas as gpd
import xarray as xr
from dask.distributed import Client


def list_s3_bucket_contents(bucket_name='noaa-nwm-retro-v2-zarr-pds'):

    s3 = boto3.client('s3', config=botocore.client.Config(signature_version=botocore.UNSIGNED))

    paginator = s3.get_paginator('list_objects_v2')
    for page in paginator.paginate(Bucket=bucket_name):
        for obj in page.get('Contents', []):
            print(obj['Key'])


def download_nwm_data(metadata, t_chunk=672, id_chunk=1000):
    gages = gpd.read_file(metadata)
    gages = gages.iloc[:100]
    gages['lat'] = gages['geometry'].y
    gages['lon'] = gages['geometry'].x

    client = Client(n_workers=8)

    # url = 's3://noaa-nwm-retrospective-3-0-pds'
    url = 's3://hytest/tutorials/evaluation/nwm'
    # url = 's3://noaa-nwm-retro-v2-zarr-pds'

    ds = xr.open_zarr(fsspec.get_mapper(url, anon=True), consolidated=True)

    idx = ((ds.latitude > 44.0) & (ds.latitude < 48.0) &
           (ds.longitude > -114.5) & (ds.longitude < -113.5))

    with dask.config.set(**{'array.slicing.split_large_chunks': False}):
        ds_out = ds[['streamflow']].isel(feature_id=idx).sel(time=slice('2023-01-01', '2023-12-31'))

    def gchunks(ds_chunk, chunks):
        group_chunks = {}
        for var in ds_chunk.variables:
            group_chunks[var] = []
            for di in ds_chunk[var].dims:
                if di in chunks.keys():
                    group_chunks[var].append(min(chunks[di], len(ds_chunk[di])))
                else:
                    group_chunks[var].append(len(ds_chunk[di]))
            ds_chunk[var] = ds_chunk[var].chunk(tuple(group_chunks[var]))
            group_chunks[var] = {'chunks': tuple(group_chunks[var])}
        return group_chunks

    encoding = gchunks(ds_out, {'time': t_chunk, 'feature_id': id_chunk})

    ds_out.to_zarr('nwm.zarr', mode='w', encoding=encoding)
    a = 1


if __name__ == '__main__':
    metadata_ = 'selected-streamflow-data-meta-merged.geojson'
    date_range_ = ('2023-01-01', '2023-12-31')
    download_nwm_data(metadata_)

# ========================= EOF ====================================================================

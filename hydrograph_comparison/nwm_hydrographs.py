import xarray as xr
import pandas as pd
from datetime import datetime
import s3fs


# https://noaa-nwm-retrospective-3-0-pds.s3.amazonaws.com/CONUS/netcdf/CHRTOUT/1979/197902010100.CHRTOUT_DOMAIN1

def download_subset_from_s3(station_ids, date_range):
    s3 = s3fs.S3FileSystem(anon=True)
    base_path = "noaa-nwm-retrospective-3-0-pds/CONUS/netcdf/CHRTOUT/"

    start_date, end_date = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])
    date_range = pd.date_range(start_date, end_date, freq='H')

    s3 = s3fs.S3FileSystem(anon=True, client_kwargs={'endpoint_url': 'https://s3.us-east-1.amazonaws.com'})

    file_paths = []
    for date in date_range:
        date_str = date.strftime('%Y/%Y%m%d%H00')
        if int(date.strftime('%Y%m%d%H00')) < 197902010100:
            continue

        s3_path = f"{base_path}{date_str}.CHRTOUT_DOMAIN1"

        file_paths.append(s3_path)
        ds = xr.open_mfdataset(file_paths,
                               combine='by_coords',
                               mask_and_scale=True,
                               decode_cf=True,
                               chunks='auto')


if __name__ == '__main__':
    station_ids = pd.read_csv('vetted_gages.csv')['staid'].to_list()
    date_range = ('1979-01-01', '2023-12-31')
    subset_data = download_subset_from_s3(station_ids, date_range)
    print(subset_data)

# ========================= EOF ====================================================================

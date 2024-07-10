import fsspec
from tqdm import tqdm
import xarray as xr
import os


fs = fsspec.filesystem('')
flist = fs.glob('/datadrive/kauai/transmissions/ooi_lf/*.nc')

for file in tqdm(flist):

    fnew = f'{file[:-25]}ooi_lf_tc/{file[38:]}'


    if os.path.exists(fnew):
        print(f'skipping {file}')
        continue
    ds = xr.open_dataset(file)
    ds.transmission.encoding = {'units': 'nanoseconds since 1970-01-01'}
    
    ds.to_netcdf(fnew, mode='w')

import xarray as xr
import dask
from dask.distributed import LocalCluster, Client

if __name__ == '__main__':
    dask.config.set({'temporary_directory': '/datadrive/tmp'})

    cluster = LocalCluster(n_workers=8)
    print(cluster.dashboard_link)
    client = Client(cluster)

    ds = xr.open_mfdataset('/datadrive/kauai/transmissions/64k_sensors/real/*.nc') + \
        1j*xr.open_mfdataset('/datadrive/kauai/transmissions/64k_sensors/imag/*.nc')

    ds_chunk = ds.set_index({'time':['longtime', 'shorttime']}).unstack().chunk({'longtime':26, 'transmission':10, 'shorttime':13640})

    store = '/datadrive/kauai/transmissions/64k_sensors/zarr_store.zarr/'
    ds_chunk.to_zarr(store, encoding={'transmission':{'dtype':int}})
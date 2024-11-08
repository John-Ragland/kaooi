'''
set of helper functions to open different KB datasets
'''
import fsspec
import xarray as xr
import os
from tqdm import tqdm
import ujson
from kerchunk.hdf import SingleHdf5ToZarr
from kerchunk.combine import MultiZarrToZarr
import numpy as np
from concurrent.futures import ProcessPoolExecutor

def open_ooi_bb(
    dataset_dir: str = '/datadrive/kauai/transmissions/ooi_bb_tc/',
    verbose: bool = False,
    compute: bool = False,
    mapping_dir: str = None,
    nodes : list = ['LJ01C', 'PC01A', 'PC03A', 'LJ01D', 'LJ01A']
):
    '''
    open OOI Broadband dataset - curre  ntly consists of ['LJ01C','PC01A','PC03A']

    Parameters
    ----------
    dataset_dir : str
        directory where data is stored
    verbose : bool
        print verbose output
    compute : bool
        if true, mapping is computed regardless of if there exists a mapping saved to disc
    mapping_dir : str
        directory where mapping is stored. Default (None) is to dataset_dir/kerchunk_map.json

    Returns
    -------
    ds : xr.Dataset
        dataset of OOI broadband data
    '''

    nodes = ['LJ01C', 'PC01A', 'PC03A', 'LJ01D', 'LJ01A']

    if mapping_dir is None:
        mapping_dir = f'{dataset_dir}/kerchunk_map.json'

    if (not os.path.exists(mapping_dir)) or compute:
        if verbose:
            print(f'mapping dataset to {dataset_dir}')
        # create mapping if it doesn't exits

        fs = fsspec.filesystem('')

        flist = []
        for node in nodes:
            flist += fs.glob(f'{dataset_dir}{node}/*.nc')

        mappings = []

        with ProcessPoolExecutor() as executor:
            mappings = list(tqdm(executor.map(__process_file, flist), total=len(flist)))

        mzz = MultiZarrToZarr(mappings, concat_dims=['transmission'], identical_dims=['time']).translate()

        with open(mapping_dir, 'wb') as f:
            f.write(ujson.dumps(mzz).encode())
    else:
        if verbose:
            print(f'mapping already exists and not computed')

    # open dataset

    so = {'fo': mapping_dir}

    ds = xr.open_dataset(
        "reference://", engine="zarr", backend_kwargs={"consolidated": False, "storage_options": so}
    ).chunk({'transmission': 1})

    return ds


def open_ooi_lf(
    dataset_dir: str = '/datadrive/kauai/transmissions/ooi_lf/',
    verbose: bool = False,
    compute: bool = False,
    mapping_dir: str = None,
):
    '''
    open OOI Low Frequency dataset

    Parameters
    ----------
    dataset_dir : str
        directory where data is stored
    verbose : bool
        print verbose output
    compute : bool
        if true, mapping is computed regardless of if there exists a mapping saved to disc
    mapping_dir : str
        directory where mapping is stored. Default (None) is to dataset_dir/kerchunk_map.json

    Returns
    -------
    ds : xr.Dataset
        dataset of OOI broadband data
    '''

    if mapping_dir is None:
        mapping_dir = f'{dataset_dir}/kerchunk_map.json'

    if (not os.path.exists(mapping_dir)) or compute:
        if verbose:
            print(f'mapping dataset to {dataset_dir}')
        # create mapping if it doesn't exits

        fs = fsspec.filesystem('')

        flist = fs.glob(f'{dataset_dir}*.nc')

        mappings = []
        with ProcessPoolExecutor() as executor:
            mappings = list(tqdm(executor.map(__process_file, flist), total=len(flist)))
        mzz = MultiZarrToZarr(mappings, concat_dims=['transmission'], identical_dims=['time']).translate()
        
        with open(mapping_dir, 'wb') as f:
            f.write(ujson.dumps(mzz).encode())
    else:
        if verbose:
            print(f'mapping already exists and not computed')

    # read json file
    with open(mapping_dir, 'rb') as f:
        mzz_read = ujson.load(f)

    # open dataset
    so = {'fo': mzz_read}

    ds = xr.open_dataset(
        "reference://", engine="zarr", backend_kwargs={"consolidated": False, "storage_options": so}
    ).chunk({'transmission': 1})

    # dumb way to fix issues
    #problem_idx = [42, 63, 102, 104, 124, 131, 133, 134, 135, 136, 137, 138, 139, 152, 257] + list(np.arange(302, 332))
    #good_idx = list(set(np.arange(ds.sizes['transmission'])) - set(problem_idx))

    #ds_good = ds.isel({'transmission': good_idx})
    return ds


def __process_file(f):
    h5chunks = SingleHdf5ToZarr(f)
    try:
        return h5chunks.translate()
    except Exception as e:
        raise Exception(f'error computing mapping for {f}') from e
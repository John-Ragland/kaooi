'''
tools for simulating acoustic propagation across ocean basins
'''
from geopy.distance import great_circle
import xarray as xr
import pyproj
import oceans.sw_extras as sw
import numpy as np
from tqdm import tqdm

def compute_distances(latitudes, longitudes, target_coordinate):
    '''
    compute_distances - given vector of latitudes and longitudes, and a target coordinate, return the great circle distance between each point and the target

    Parameters
    ----------
    latitudes : list
        list of latitudes
    longitudes : list   
        list of longitudes
    target_coordinate : tuple
        (lat, lon) of target coordinate

    Returns
    -------
    distances : list
        list of distances between each point and the target
    '''
    distances = []
    for lat, lon in zip(latitudes, longitudes):
        coord1 = (lat, lon)
        distance = great_circle(coord1, target_coordinate).kilometers 
        distances.append(distance)
    return distances


def compute_gc_coords(lat1, lon1, lat2, lon2, num_points):
    '''
    compute_gc_coords - given two coordinates, compute the great circle path between them

    Parameters
    ----------
    lat1 : float
        latitude of source
    lon1 : float
        longitude of source
    lat2 : float
        latitude of reciever
    lon2 : float
        longitude of reciever
    num_points : int
        number of points to sample along the great circle

    Returns
    -------
    lats : np.array
        latitudes of points along great circle
    lons : np.array
        longitudes of points along great circle
    ranges : np.array
        ranges of points along great circle

    '''
    g = pyproj.Geod(ellps='WGS84')

    # calculate lat lon points along great circle
    lonlats = g.npts(lon1, lat1, lon2, lat2, num_points)

    # npts doesn't include start/end points, so prepend/append them
    lonlats.insert(0, (lon1, lat1))
    lonlats.append((lon2, lat2))

    lats = []
    lons = []
    for point in lonlats:
        lons.append(point[0])
        lats.append(point[1])
    
    # compute ranges
    ranges = compute_distances(lats, lons, (lat1, lon1))

    return np.array(lats), np.array(lons), np.array(ranges)


def get_bathymetry_slice(Tx_coords, Rx_coords, num_range_points, bathy_grid='/datadrive/bathymetry_data/GEBCO_2023.nc'):
    '''
    given source and reciever cooridinates in lat/lon, number of range points, return
        a slice of bathymetry data, along the great circle that connects them

    Parameters
    ----------
    Tx_coords : tuple
        (lat, lon) of source
    Rx_coords : tuple
        (lat, lon) of reciever
    num_range_points : int
        number of points to sample along the great circle
    bathy_grid : str
        path to bathymetry data. Built around (2023 GEBCO grid)[https://www.gebco.net/data_and_products/gridded_bathymetry_data/gebco_2023/]
    
    Returns
    -------
    depth_slice : xr.DataArray
        slice of depth along great circle
    '''

    # open gridded bathymetry data
    ds = xr.open_dataset(bathy_grid).chunk({'lon': 1000, 'lat': 1000})

    # unpack coordinates
    lat1, lon1 = Tx_coords
    lat2, lon2 = Rx_coords
    
    lats, lons, range_coord = compute_gc_coords(lat1, lon1, lat2, lon2, num_range_points)
    # interpolate grid along great_circle
    depth_interp = (-1 * ds['elevation']).interp(
        lat=xr.DataArray(lats, dims='range'), lon=xr.DataArray(lons, dims='range')).assign_coords({'range':range_coord})

    return depth_interp

def compute_stitched_ssp(woa_dir, ssp_dir):
    '''
    take woa data and stitch together monthly and annual profiles
    '''
    # open datasets
    fn = f'{woa_dir}monthly_2015_2022_0.25/temp/*.nc'
    temp = xr.open_mfdataset(fn, decode_times=False)['t_an'].load()

    fn = f'{woa_dir}monthly_2015_2022_0.25/sal/*.nc'
    sal = xr.open_mfdataset(fn, decode_times=False)['s_an'].load()

    fn = f'{woa_dir}woa23_B5C2_t00_04.nc'
    temp_an = xr.open_dataset(fn, decode_times=False)['t_an'].drop_vars('time').load()

    fn = f'{woa_dir}woa23_B5C2_s00_04.nc'
    sal_an = xr.open_dataset(fn, decode_times=False)['s_an'].drop_vars('time').load()

    # stitch together annual and monthly profiles
    temp_an_time = xr.concat([temp_an[0, :, :, :].loc[1550:].expand_dims(dim='time')] * 12, dim='time')
    temp_an_time = temp_an_time.assign_coords({'time': temp.time})

    sal_an_time = xr.concat([sal_an[0, :, :, :].loc[1550:].expand_dims(dim='time')] * 12, dim='time')
    sal_an_time = sal_an_time.assign_coords({'time': sal.time})

    temp_stitch = xr.concat((temp, temp_an_time), dim='depth')
    sal_stitch = xr.concat((sal, sal_an_time), dim='depth')

    ssp = sw.soundspeed(sal_stitch, temp_stitch, temp_stitch.depth)

    ssp.to_netcdf(ssp_dir)
    return


def get_ssp_slice(Tx_coords, Rx_coords, num_range_points, ssp_dir='/datadrive/woa/ssp_stitched.nc', fillna=False):
    '''
    get_ssp_slice - given source and reciever cooridinates in lat/lon, number of range points, return
        a slice of sound speed profile data, along the great circle that connects them
    
    SSP data is from WOA monthly profiles. Data below 1500 is stitched from anual woa profiles

    World ocean data can be downloaded with script ::TODO::

    Parameters
    ----------
    Tx_coords : tuple
        (lat, lon) of source
    Rx_coords : tuple
        (lat, lon) of reciever
    num_range_points : int
        number of points to sample along the great circle
    ssp_dir : str
        path to sound speed profile data. Built around (WOA data)[https://www.nodc.noaa.gov/OC5/woa18/]
    fillna : bool
        if true, replace nan (land) with pressure only sound speed
    '''

    # unpack coordinates
    lat1, lon1 = Tx_coords
    lat2, lon2 = Rx_coords
    lats, lons, range_coord = compute_gc_coords(lat1, lon1, lat2, lon2, num_range_points)

    # open ssp
    ssp = xr.open_dataarray(ssp_dir, decode_times=False).chunk({'depth': 100, 'lat': 1000, 'lon': 1000})
    ssp_interp = ssp.interp(
        lat=xr.DataArray(lats, dims='range'),
        lon=xr.DataArray(lons, dims='range'),
    )

    # check that start and end points are not adjacent to nan
    idx_lat1 = np.argmin(np.abs(ssp.lat.values - lats[0]))
    idx_lon1 = np.argmin(np.abs(ssp.lon.values - lons[0]))
    idx_lat2 = np.argmin(np.abs(ssp.lat.values - lats[-1]))
    idx_lon2 = np.argmin(np.abs(ssp.lon.values - lons[-1]))

    coord1_nan = np.isnan(ssp.isel({'lat':slice(idx_lat1-1, idx_lat1+1), 'lon':slice(idx_lon1-1, idx_lon1+1)}))
    coord2_nan = np.isnan(ssp.isel({'lat':slice(idx_lat2-1, idx_lat2+1), 'lon':slice(idx_lon2-1, idx_lon2+1)}))

    # if start or end points are adjacent to nan, use
    # nearest neighbor interpolation for start and end points

    if coord1_nan.any():
        ssp_lat1 = ssp.isel({'lat':idx_lat1, 'lon':idx_lon1}).expand_dims('range')
        ssp_interp = xr.concat((ssp_lat1, ssp_interp.isel({'range':slice(1,None)})), dim='range')
    elif coord2_nan.any():
        ssp_lat2 = ssp.isel({'lat':idx_lat2, 'lon':idx_lon2}).expand_dims('range')
        ssp_interp = xr.concat((ssp_interp.isel({'range':slice(None,-1)}), ssp_lat2), dim='range')

    ssp_interp = ssp_interp.assign_coords({'range': range_coord})

    # add depth of 10000 m
    ssp_interp = xr.concat((ssp_interp, xr.ones_like(ssp_interp.isel({'depth': -1}))*np.nan), dim='depth')
    depth_coords = ssp_interp.depth.values
    depth_coords[-1] = 10000
    ssp_interp = ssp_interp.assign_coords({'depth': depth_coords})

    # add range of 5 $\times$ max range
    ssp_interp = xr.concat((ssp_interp, xr.ones_like(ssp_interp.isel({'range': -1}))*np.nan), dim='range')
    range_coords = ssp_interp.range.values
    range_coords[-1] = 5 * range_coords[-2]
    ssp_interp = ssp_interp.assign_coords({'range': range_coords})
    
    # if filled is true, replace nan (land) with pressure only sound speed
    if fillna:
        ssp_depth = sw.soundspeed(
            xr.ones_like(ssp_interp) * 34.717712, xr.ones_like(ssp_interp) * 1.44, ssp_interp.depth # average pressure and salinity at last depth of woa
        )
        ssp_interp = ssp_interp.fillna(ssp_depth)

    return ssp_interp


def write_bty(bathy_slice : xr.DataArray, filename : str, npts : int = None, interp : str = 'L'):
    '''
    given a bathymetry slice (computed with ::get_bathymetry_slice::), write a .bty file

    Parameters
    ----------
    bathy_slice : xr.DataArray
        slice of depth along great circle. Can be computed with ::get_bathymetry_slice::
    filename : str
        name of file to write to. Should not include file extension
    npts : int
        number of points to include in bty file. Default (None) includes all points
        in bathy_slice
    '''
    if npts is not None:
        bathy_slice = bathy_slice.interp(
            range = np.linspace(
                bathy_slice.range.min(),
                bathy_slice.range.max(),
                npts),
            method='linear'
        )
    else:
        npts = len(bathy_slice)

    # compute and load bathy slice
    bathy_slice = bathy_slice.compute()
    with open(f'{filename}.bty', 'w') as file:
        file.write(f"'{interp}'\n")
        file.write(f"{npts}\n")
        for k in range(len(bathy_slice)):
            file.write(f"{bathy_slice.range.values[k]}\t{bathy_slice.values[k]}\n")

def write_ssp(ssp_slice : xr.DataArray, filename : str, npts : int = None, interp : str = 'L'):
    '''
    given a ssp slice (computed with ::get_ssp_slice::), write a .ssp file

    Parameters
    ----------
    ssp_slice : xr.DataArray
        slice of sound speed profile along great circle. Can be computed with ::get_ssp_slice::
        should have dimensions ['depth', 'range']. Range should be in km, depth in meters. 
        There shouldn't be any other dimensions
    filename : str
        name of file to write to. Should not include file extension
    npts : int
        number of range points to include in ssp file. Default (None) includes all range points
        in ssp_slice
    interp : str
        interpolation type. Default is 'L' for linear
    '''

    # check to make sure ssp only has depth and range dimensions
    dims = {'range','depth'}
    ssp_dims = set(ssp_slice.dims)
    if dims != ssp_dims:
        raise ValueError(f"ssp_slice should only dimensions ['range', 'depth'], but have dimensions [{ssp_slice.dims}]")

    if npts is not None:
        ssp_slice = ssp_slice.interp(
            range = np.linspace(
                ssp_slice.range.min(),
                ssp_slice.range.max(),
                npts),
            method='linear'
        )
    else:
        npts = ssp_slice.sizes['range']
    
    # compute and load ssp slice
    ssp_slice = ssp_slice.compute()

    with open(f'{filename}.ssp', 'w') as file:
        file.write(f"{npts}\n")

        # write range values (in km)
        for k in range(ssp_slice.sizes['range']):
            file.write(f"{ssp_slice.range.values[k]}\t")
        file.write('\n')
        
        # write ssp values (in m/s)
        ssp_matrix = ssp_slice.transpose('depth', 'range').values
        np.savetxt(file, ssp_matrix, delimiter='\t', fmt='%1.4f')

def write_env(filename, ssp_slice):
    '''
    write_env - write .env file

    Parameters
    ----------

    '''

    title = 'TEST'
    freq = 75
    nmedia = 1
    interp_type = 'QVW'
    bottom_depth = 5000    

    bottom_speed = 1600
    source_depth = 500
    Rx_depth_num = 51
    Rx_depth1 = 0
    Rx_depth2 = 5000
    Rx_range_num = 1000
    Rx_range1 = 0
    Rx_range2 = 1000
    run_type = 'R'
    n_beams = 50
    alpha1 = -20
    alpha2 = 20

    with open(f'{filename}.env', 'w') as file:
        file.write(f"'{title}'\t\t!TITLE\n")
        file.write(f"{freq}\t\t!FREQUENCY (Hz)\n")
        file.write(f"{nmedia}\t\t!NMEDIA\n")
        file.write(f"'{interp_type}'\t\t! SSP INTERPOLATION TYPE\n")
        file.write(f"{0}\t{0}\t{bottom_depth}\n")

        for k in range(ssp_slice.sizes['depth']):
            # print first ssp to file

            depth_single = ssp_slice.isel(range=0).depth.values[k]
            ssp_single = ssp_slice.isel(range=0).values[k]

            num_digits_depth = len(str(int(depth_single)))
            num_digits_ssp = len(str(int(ssp_single)))

            spacing=4
            file.write(f"{depth_single:>{num_digits_depth+spacing}.1f}\t{ssp_single:>{num_digits_ssp+spacing}.2f}\t/\n")

        file.write(f"'A*'\t0\n")
        file.write(f"{bottom_depth}\t{bottom_speed}\t{0}\t{1}\t/\n")
        file.write(f'{1}\t\t!NSD\n')
        file.write(f'{source_depth}\t\t/ ! Source depth (!:NSD) (m)\n')
        file.write(f'{Rx_depth_num}\t\t!NRD\n')
        file.write(f'{Rx_depth1}\t{Rx_depth2}\t/\t\t!RD(1:NRD) (m)\n')
        file.write(f'{Rx_range_num}\t\t!NR\n')
        file.write(f'{Rx_range1}\t{Rx_range2}\t\t/!R(1:NR ) (km)\n')
        file.write(f"'{run_type}'\t\t!'R/C/I/S'\n")
        file.write(f"{n_beams}\t\t!NBeams\n")
        file.write(f"{alpha1}\t{alpha2}\t\t!Alpha1, Alpha2 (degrees)\n")
        file.write(f"{0}\t{Rx_depth2}\t{Rx_range2}\t\t!STEP (m), ZBox (m), RBOX (km)\n")

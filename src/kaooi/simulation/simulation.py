'''
tools for simulating acoustic propagation across ocean basins
'''
from geopy.distance import great_circle
import xarray as xr
import pyproj
import oceans.sw_extras as sw

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
    lonlats : list
        list of (lon, lat) coordinates along the great circle
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

    return lats, lons, ranges


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


def get_ssp_slice(Tx_coords, Rx_coords, num_range_points, woa_dir='/datadrive/woa/'):
    '''
    get_ssp_slice - given source and reciever cooridinates in lat/lon, number of range points, return
        a slice of sound speed profile data, along the great circle that connects them
    
    SSP data is from WOA monthly profiles. Data below 1500 is stitched from anual woa profiles

    World ocean data can be downloaded with script ::TODO::

    '''

    # unpack coordinates
    lat1, lon1 = Tx_coords
    lat2, lon2 = Rx_coords
    lats, lons, range_coord = compute_gc_coords(lat1, lon1, lat2, lon2, num_range_points)

    # open datasets
    fn = f'{woa_dir}monthly_2015_2022_0.25/temp/*.nc'
    temp = xr.open_mfdataset(fn, decode_times=False)['t_an']

    fn = f'{woa_dir}monthly_2015_2022_0.25/sal/*.nc'
    sal = xr.open_mfdataset(fn, decode_times=False)['s_an']

    fn = f'{woa_dir}woa23_B5C2_t00_04.nc'
    temp_an = xr.open_dataset(fn, decode_times=False)['t_an'].drop_vars('time')

    fn = f'{woa_dir}woa23_B5C2_s00_04.nc'
    sal_an = xr.open_dataset(fn, decode_times=False)['s_an'].drop_vars('time')

    # stitch together annual and monthly profiles
    temp_an_time = xr.concat([temp_an[0, :, :, :].loc[1550:].expand_dims(dim='time')] * 12, dim='time')
    temp_an_time = temp_an_time.assign_coords({'time': temp.time})

    sal_an_time = xr.concat([sal_an[0, :, :, :].loc[1550:].expand_dims(dim='time')] * 12, dim='time')
    sal_an_time = sal_an_time.assign_coords({'time': sal.time})

    temp_stitch = xr.concat((temp, temp_an_time), dim='depth')
    sal_stitch = xr.concat((sal, sal_an_time), dim='depth')

    ssp = sw.soundspeed(sal_stitch, temp_stitch, temp_stitch.depth)

    ssp_interp = ssp.interp(lat=xr.DataArray(lats, dims='range'), lon=xr.DataArray(lons, dims='range'))

    ssp_interp = ssp_interp.assign_coords({'range': range_coord})

    return ssp_interp
#######################################################
# Useful tools
#######################################################

import numpy as np
import sys


# Constants
deg2rad = np.pi/180.0
rad2deg = 1./deg2rad
# Radius of Earth
rEarth = 6.371e6


#true_scale_lat = 70
#re = 6378.273
#e = 0.081816153


def polar_xy_to_lonlat(x, y, true_scale_lat=70, re=6378.137, e=0.08181919, hemi_direction=-1):
    """Convert from Polar Stereographic (x, y) coordinates to
    geodetic longitude and latitude.
    Args:
        x (float): X coordinate(s) in km
        y (float): Y coordinate(s) in km
        true_scale_lat (float): true-scale latitude in degrees
        hemisphere ('north' or 'south'): Northern or Southern hemisphere
        re (float): Earth radius in km
        e (float): Earth eccentricity
    Returns:
        If x and y are scalars then the result is a
        two-element list containing [longitude, latitude].
        If x and y are numpy arrays then the result will be a two-element
        list where the first element is a numpy array containing
        the longitudes and the second element is a numpy array containing
        the latitudes.
    """

#     hemisphere = validate_hemisphere(hemisphere)
#     hemi_direction = _hemi_direction(hemisphere)

    e2 = e * e
    slat = true_scale_lat * np.pi / 180
    rho = np.sqrt(x ** 2 + y ** 2)

    if abs(true_scale_lat - 90.) < 1e-5:
        t = rho * np.sqrt((1 + e) ** (1 + e) * (1 - e) ** (1 - e)) / (2 * re)
    else:
        cm = np.cos(slat) / np.sqrt(1 - e2 * (np.sin(slat) ** 2))
        t = np.tan((np.pi / 4) - (slat / 2)) / \
            ((1 - e * np.sin(slat)) / (1 + e * np.sin(slat))) ** (e / 2)
        t = rho * t / (re * cm)

    chi = (np.pi / 2) - 2 * np.arctan(t)
    lat = chi + \
        ((e2 / 2) + (5 * e2 ** 2 / 24) + (e2 ** 3 / 12)) * np.sin(2 * chi) + \
        ((7 * e2 ** 2 / 48) + (29 * e2 ** 3 / 240)) * np.sin(4 * chi) + \
        (7 * e2 ** 3 / 120) * np.sin(6 * chi)
    lat = hemi_direction * lat * 180 / np.pi
    lon = np.arctan2(hemi_direction * x, -hemi_direction * y)
    lon = hemi_direction * lon * 180 / np.pi
    lon = lon + np.less(lon, 0) * 360
    return [lon, lat]


def polar_lonlat_to_xy(longitude, latitude, true_scale_lat=70, re=6378.137, e=0.08181919, hemi_direction=-1):
    """Convert from geodetic longitude and latitude to Polar Stereographic
    (X, Y) coordinates in km.
    Args:
        longitude (float): longitude or longitude array in degrees
        latitude (float): latitude or latitude array in degrees (positive)
        true_scale_lat (float): true-scale latitude in degrees
        re (float): Earth radius in km
        e (float): Earth eccentricity
        hemisphere ('north' or 'south'): Northern or Southern hemisphere
    Returns:
        If longitude and latitude are scalars then the result is a
        two-element list containing [X, Y] in km.
        If longitude and latitude are numpy arrays then the result will be a
        two-element list where the first element is a numpy array containing
        the X coordinates and the second element is a numpy array containing
        the Y coordinates.
    """

#     hemisphere = validate_hemisphere(hemisphere)
#     hemi_direction = _hemi_direction(hemisphere)

    lat = abs(latitude) * np.pi / 180
    lon = longitude * np.pi / 180
    slat = true_scale_lat * np.pi / 180

    e2 = e * e

    # Snyder (1987) p. 161 Eqn 15-9
    t = np.tan(np.pi / 4 - lat / 2) / \
        ((1 - e * np.sin(lat)) / (1 + e * np.sin(lat))) ** (e / 2)

    if abs(90 - true_scale_lat) < 1e-5:
        # Snyder (1987) p. 161 Eqn 21-33
        rho = 2 * re * t / np.sqrt((1 + e) ** (1 + e) * (1 - e) ** (1 - e))
    else:
        # Snyder (1987) p. 161 Eqn 21-34
        tc = np.tan(np.pi / 4 - slat / 2) / \
            ((1 - e * np.sin(slat)) / (1 + e * np.sin(slat))) ** (e / 2)
        mc = np.cos(slat) / np.sqrt(1 - e2 * (np.sin(slat) ** 2))
        rho = re * mc * t / tc

    x = rho * hemi_direction * np.sin(hemi_direction * lon)
    y = -rho * hemi_direction * np.cos(hemi_direction * lon)
    return [x, y]











# Convert longitude and latitude to polar stereographic projection used by BEDMAP2. Adapted from polarstereo_fwd.m in the MITgcm Matlab toolbox for Bedmap.
def polar_stereo (lon, lat, a=6378137., e=0.08181919, lat_c=-71, lon0=0):

    # Deep copies of arrays in case they are reused
    lon = np.copy(lon)
    lat = np.copy(lat)

    if lat_c < 0:
        # Southern hemisphere
        pm = -1
    else:
        # Northern hemisphere
        pm = 1

    # Prepare input
    lon = lon*pm*deg2rad
    lat = lat*pm*deg2rad
    lat_c = lat_c*pm*deg2rad
    lon0 = lon0*pm*deg2rad

    # Calculations
    t = np.tan(np.pi/4 - lat/2)/((1 - e*np.sin(lat))/(1 + e*np.sin(lat)))**(e/2)
    t_c = np.tan(np.pi/4 - lat_c/2)/((1 - e*np.sin(lat_c))/(1 + e*np.sin(lat_c)))**(e/2)
    m_c = np.cos(lat_c)/np.sqrt(1 - (e*np.sin(lat_c))**2)
    rho = a*m_c*t/t_c
    x = pm*rho*np.sin(lon - lon0)
    y = -pm*rho*np.cos(lon - lon0)

    return x, y

# Determine the x and y coordinates based on whether the user wants polar stereographic or not.
def get_x_y (lon, lat, pster=False):
    if pster:
        x, y = polar_stereo(lon, lat)
    else:
        x = lon
        y = lat
    return x, y



# Helper function for read_binary and write_binary. Given a precision (32 or 64) and endian-ness ('big' or 'little'), construct the python data type string.
def set_dtype (prec, endian):

    if endian == 'big':
        dtype = '>'
    elif endian == 'little':
        dtype = '<'
    else:
        print('Error (set_dtype): invalid endianness')
        sys.exit()
    if prec == 32:
        dtype += 'f4'
    elif prec == 64:
        dtype += 'f8'
    else:
        print('Error (set_dtype): invalid precision')
        sys.exit()
    return dtype


# Write an array ("data"), of any dimension, to a binary file ("file_path"). Optional keyword arguments ("prec" and "endian") are as in function read_binary.
def write_binary (data, file_path, prec=32, endian='big'):

    print(('Writing ' + file_path))

    if isinstance(data, np.ma.MaskedArray):
        # Need to remove the mask
        data = data.data

    dtype = set_dtype(prec, endian)    
    # Make sure data is in the right precision
    data = data.astype(dtype)

    # Write to file
    id = open(file_path, 'w')
    data.tofile(id)
    id.close()


# Interpolate a topography field "data" (eg bathymetry, ice shelf draft, mask) to grid cells. We want the area-averaged value over each grid cell. So it's not enough to just interpolate to a point (because the source data might be much higher resolution than the new grid) or to average all points within the cell (because the source data might be lower or comparable resolution). Instead, interpolate to a finer grid within each grid cell (default 10x10) and then average over these points.

# Arguments:
# x, y: 1D arrays with x and y coordinates of source data (polar stereographic for BEDMAP2, lon and lat for GEBCO)
# data: 2D array of source data
# x_interp, y_interp: 2D arrays with x and y coordinates of the EDGES of grid cells - the output array will be 1 smaller in each dimension

# Optional keyword argument:
# n_subgrid: dimension of finer grid within each grid cell (default 10, i.e. 10 x 10 points per grid cell)

# Output: data on centres of new grid

def interp_topo (x, y, data, x_interp, y_interp, n_subgrid=10):

    from scipy.interpolate import RectBivariateSpline

    # x_interp and y_interp are the edges of the grid cells, so the number of cells is 1 less
    num_j = y_interp.shape[0]
    num_i = x_interp.shape[1]
    data_interp = np.empty([num_j, num_i])
    
    # pre-process for bathymetry data
    data[np.isnan(data)] = 0
    
    
    # RectBivariateSpline needs (y,x) not (x,y) - this can really mess you up when BEDMAP2 is square!!
    interpolant = RectBivariateSpline(y, x, data)

    # Loop over grid cells (can't find a vectorised way to do this without overflowing memory)
    for j in range(num_j-1):
        for i in range(num_i-1):
            # Make a finer grid within this grid cell (regular in x and y)
            # First identify the boundaries so that x and y are strictly increasing
            if x_interp[j,i] < x_interp[j,i+1]:
                x_start = x_interp[j,i]
                x_end = x_interp[j,i+1]
            else:
                x_start = x_interp[j,i+1]
                x_end = x_interp[j,i]
            if y_interp[j,i] < y_interp[j+1,i]:
                y_start = y_interp[j,i]
                y_end = y_interp[j+1,i]
            else:
                y_start = y_interp[j+1,i]
                y_end = y_interp[j,i]
            # Define edges of the sub-cells
            x_edges = np.linspace(x_start, x_end, num=n_subgrid+1)
            y_edges = np.linspace(y_start, y_end, num=n_subgrid+1)
            # Calculate centres of the sub-cells
            x_vals = 0.5*(x_edges[1:] + x_edges[:-1])
            y_vals = 0.5*(y_edges[1:] + y_edges[:-1])
            # Interpolate to the finer grid, then average over those points to estimate the mean value of the original field over the entire grid cell
            data_interp[j,i] = np.mean(interpolant(y_vals, x_vals))
    
    # post-process for bathymetry data
    data_interp[data_interp>0] = 0
    
    return data_interp



# Helper function for masking functions below
# depth_dependent only has an effect if the mask is 2D.
def apply_mask (data, mask, time_dependent=False, depth_dependent=False):

    if depth_dependent and len(mask.shape)==2:
        # Tile a 2D mask in the depth dimension
        grid_dim = [data.shape[-1], data.shape[-2], data.shape[-3]]
        mask = xy_to_xyz(mask, grid_dim)
    if time_dependent:
        # Tile the mask in the time dimension
        mask = add_time_dim(mask, data.shape[0])

    if len(mask.shape) != len(data.shape):
        print('Error (apply_mask): invalid dimensions of data')
        sys.exit()

    data = np.ma.masked_where(mask, data)
    return data


# Mask land out of an array.

# Arguments:
# data: array of data to mask, assumed to be 2D unless time_dependent or depth_dependent say otherwise
# grid: Grid object

# Optional keyword arguments:
# gtype: as in function Grid.get_hfac
# time_dependent: as in function apply_mask
# depth_dependent: as in function apply_mask

def mask_land (data, grid, gtype='t', time_dependent=False, depth_dependent=False):

    return apply_mask(data, grid.get_land_mask(gtype=gtype), time_dependent=time_dependent, depth_dependent=depth_dependent)


# Mask land and ice shelves out of an array, just leaving the open ocean.
def mask_land_ice (data, grid, gtype='t', time_dependent=False, depth_dependent=False):

    return apply_mask(data, grid.get_land_mask(gtype=gtype)+grid.get_ice_mask(gtype=gtype), time_dependent=time_dependent, depth_dependent=depth_dependent)


# Mask land and open ocean out of an array, just leaving the ice shelves.
def mask_except_ice (data, grid, gtype='t', time_dependent=False, depth_dependent=False):

    return apply_mask(data, np.invert(grid.get_ice_mask(gtype=gtype)), time_dependent=time_dependent, depth_dependent=depth_dependent)


# Mask everything except FRIS out of an array.
def mask_except_fris (data, grid, gtype='t', time_dependent=False, depth_dependent=False):

    return apply_mask(data, np.invert(grid.get_ice_mask(shelf='fris', gtype=gtype)), time_dependent=time_dependent, depth_dependent=depth_dependent)


# Apply the 3D hfac mask. Dry cells are masked out; partial cells are untouched.

# Arguments:
# data: array of data to mask, assumed to be 3D unless time_dependent=True
# grid: Grid object

# Optional keyword arguments:
# gtype: as in function Grid.get_hfac
# time_dependent: as in function apply_mask

def mask_3d (data, grid, gtype='t', time_dependent=False):

    return apply_mask(data, grid.get_hfac(gtype=gtype)==0, time_dependent=time_dependent)












# Find all the factors of the integer n.
def factors (n):

    factors = []
    for i in range(1, n+1):
        if n % i == 0:
            factors.append(i)
    return factors


# Given a path to a directory, make sure it ends with /
def real_dir (dir_path):

    if not dir_path.endswith('/'):
        dir_path += '/'
    return dir_path


# Given an array representing a mask (as above) and 2D arrays of longitude and latitude, mask out the points between the given lat/lon bounds.
def mask_box (data, lon, lat, xmin=None, xmax=None, ymin=None, ymax=None, mask_val=0):

    # Set any bounds which aren't already set
    if xmin is None:
        xmin = np.amin(lon)
    if xmax is None:
        xmax = np.amax(lon)
    if ymin is None:
        ymin = np.amin(lat)
    if ymax is None:
        ymax = np.amax(lat)
    index = (lon >= xmin)*(lon <= xmax)*(lat >= ymin)*(lat <= ymax)
    data[index] = mask_val
    return data


# Mask out the points above or below the line segment bounded by the given points.
def mask_line (data, lon, lat, p_start, p_end, direction, mask_val=0):

    limit = (p_end[1] - p_start[1])/float(p_end[0] - p_start[0])*(lon - p_start[0]) + p_start[1]
    west_bound = min(p_start[0], p_end[0])
    east_bound = max(p_start[0], p_end[0])
    if direction == 'above':
        index = (lat >= limit)*(lon >= west_bound)*(lon <= east_bound)
    elif direction == 'below':
        index = (lat <= limit)*(lon >= west_bound)*(lon <= east_bound)
    else:
        print(('Error (mask_line): invalid direction ' + direction))
        sys.exit()
    data[index] = mask_val
    return data


# Interface to mask_line: mask points above line segment (to the north)
def mask_above_line (data, lon, lat, p_start, p_end, mask_val=0):

    return mask_line(data, lon, lat, p_start, p_end, 'above', mask_val=mask_val)


# Interface to mask_line: mask points below line segment (to the south)
def mask_below_line (data, lon, lat, p_start, p_end, mask_val=0):

    return mask_line(data, lon, lat, p_start, p_end, 'below', mask_val=mask_val)


# Like mask_box, but only mask out ice shelf points within the given box.
def mask_iceshelf_box (omask, imask, lon, lat, xmin=None, xmax=None, ymin=None, ymax=None, mask_val=0, option='land'):

    # Set any bounds which aren't already set
    if xmin is None:
        xmin = np.amin(lon)
    if xmax is None:
        xmax = np.amax(lon)
    if ymin is None:
        ymin = np.amin(lat)
    if ymax is None:
        ymax = np.amax(lat)
    index = (lon >= xmin)*(lon <= xmax)*(lat >= ymin)*(lat <= ymax)*(imask == 1)
    if option == 'land':
        # Turn ice shelf points into land
        mask = omask
    elif option == 'ocean':
        # Turn ice shelf points into open ocean
        mask = imask
    else:
        print(('Error (mask_iceshelf_box): Invalid option ' + option))
        sys.exit()
    mask[index] = mask_val
    return mask


# Split and rearrange the given array along the given index in the longitude axis (last axis). This is useful when converting from longitude ranges (0, 360) to (-180, 180) if the longitude array needs to be strictly increasing for later interpolation.
def split_longitude (array, split):

    return np.concatenate((array[...,split:], array[...,:split]), axis=-1)


# Return the root mean squared difference between the two arrays (assumed to be the same size), summed over all entries.
def rms (array1, array2):

    return np.sqrt(np.sum((array1 - array2)**2))


# Work out whether the given year is a leap year.
def is_leap_year (year):
    return year%4 == 0 and (year%100 != 0 or year%400 == 0)


# Return the number of days in the given month (indexed 1-12) of the given year.
def days_per_month (month, year, allow_leap=True):

    # Days per month in non-leap years
    days = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]

    # Special case for February in leap years
    if month == 2 and is_leap_year(year) and allow_leap:
        return days[month-1]+1
    else:
        return days[month-1]


# Make sure the given field isn't time-dependent, based on the expected number of dimensions.
def check_time_dependent (var, num_dim=3):

    if len(var.shape) == num_dim+1:
        print('Error (check_time_dependent): variable cannot be time dependent.')
        sys.exit()


# Calculate hFacC, hFacW, or hFacS (depending on value of gtype) without knowing the full grid, i.e. just from the bathymetry and ice shelf draft on the tracer grid.
def calc_hfac (bathy, draft, z_edges, hFacMin=0.1, hFacMinDr=20., gtype='t'):

    if gtype == 'u':
        # Need to get bathy and draft on the western edge of each cell
        # Choose the shallowest bathymetry from the adjacent tracer cells
        bathy = np.concatenate((np.expand_dims(bathy[:,0],1), np.maximum(bathy[:,:-1], bathy[:,1:])), axis=1)
        # Choose the deepest ice shelf draft from the adjacent tracer cells
        draft = np.concatenate((np.expand_dims(draft[:,0],1), np.minimum(draft[:,:-1], draft[:,1:])), axis=1)
        # Now correct for negative wct
        draft = np.maximum(draft, bathy)
    elif gtype == 'v':
        # Need to get bathy and draft on the southern edge of each cell
        bathy = np.concatenate((np.expand_dims(bathy[0,:],0), np.maximum(bathy[:-1,:], bathy[1:,:])), axis=0)
        draft = np.concatenate((np.expand_dims(draft[0,:],0), np.minimum(draft[:-1,:], draft[1:,:])), axis=0)
        draft = np.maximum(draft, bathy)        

    # Calculate a few grid variables
    z_above = z_edges[:-1]
    z_below = z_edges[1:]
    dz = np.abs(z_edges[1:] - z_edges[:-1])
    nz = dz.size
    ny = bathy.shape[0]
    nx = bathy.shape[1]    
    
    # Tile all arrays to be 3D
    bathy = xy_to_xyz(bathy, [nx, ny, nz])
    draft = xy_to_xyz(draft, [nx, ny, nz])
    dz = z_to_xyz(dz, [nx, ny, ny])
    z_above = z_to_xyz(z_above, [nx, ny, nz])
    z_below = z_to_xyz(z_below, [nx, ny, nz])
    
    # Start out with all cells closed
    hfac = np.zeros([nz, ny, nx])
    # Find fully open cells
    index = (z_below >= bathy)*(z_above <= draft)
    hfac[index] = 1
    # Find partial cells due to bathymetry alone
    index = (z_below < bathy)*(z_above <= draft)
    hfac[index] = (z_above[index] - bathy[index])/dz[index]
    # Find partial cells due to ice shelf draft alone
    index = (z_below >= bathy)*(z_above > draft)
    hfac[index] = (draft[index] - z_below[index])/dz[index]
    # Find partial cells which are intersected by both
    index = (z_below < bathy)*(z_above > draft)
    hfac[index] = (draft[index] - bathy[index])/dz[index]

    # Now apply hFac limitations
    hfac_limit = np.maximum(hFacMin, np.minimum(hFacMinDr/dz, 1))    
    index = hfac < hfac_limit/2
    hfac[index] = 0
    index = (hfac >= hfac_limit/2)*(hfac < hfac_limit)
    hfac[index] = hfac_limit[index]

    return hfac


# Calculate bathymetry or ice shelf draft from hFacC.
def bdry_from_hfac (option, hfac, z_edges):

    nz = hfac.shape[0]
    ny = hfac.shape[1]
    nx = hfac.shape[2]
    dz = z_edges[:-1]-z_edges[1:]

    bdry = np.zeros([ny, nx])
    bdry[:,:] = np.nan
    if option == 'bathy':
        # Loop from bottom to top
        k_vals = list(range(nz-1, -1, -1))
    elif option == 'draft':
        # Loop from top to bottom
        k_vals = list(range(nz))
    else:
        print(('Error (bdry_from_hfac): invalid option ' + option))
        sys.exit()
    for k in k_vals:
        hfac_tmp = hfac[k,:]
        # Identify wet cells with no boundary assigned yet
        index = (hfac_tmp!=0)*np.isnan(bdry)
        if option == 'bathy':
            bdry[index] = z_edges[k] - dz[k]*hfac_tmp[index]
        elif option == 'draft':
            bdry[index] = z_edges[k] - dz[k]*(1-hfac_tmp[index])
    # Anything still NaN is land mask and should be zero
    index = np.isnan(bdry)
    bdry[index] = 0

    return bdry


# Modify the given bathymetry or ice shelf draft to make it reflect what the model will actually see, based on hFac constraints.
def model_bdry (option, bathy, draft, z_edges, hFacMin=0.1, hFacMinDr=20.):

    # First calculate the hFacC
    hfac = calc_hfac(bathy, draft, z_edges, hFacMin=hFacMin, hFacMinDr=hFacMinDr)
    # Now calculate the new boundary
    return bdry_from_hfac(option, hfac, z_edges)


# Determine if a string is an integer.
def str_is_int (s):
    try:
        int(s)
        return True
    except ValueError:
        return False


# Find the Cartesian distance between two lon-lat points.
# This also works if one of point0, point1 is a 2D array of many points.
def dist_btw_points (point0, point1):
    [lon0, lat0] = point0
    [lon1, lat1] = point1
    dx = rEarth*np.cos((lat0+lat1)/2*deg2rad)*(lon1-lon0)*deg2rad
    dy = rEarth*(lat1-lat0)*deg2rad
    return np.sqrt(dx**2 + dy**2)





# Given an axis with values in the centre of each cell, find the locations of the boundaries of each cell (extrapolating for the outer boundaries).
def axis_edges (x):
    x_bound = 0.5*(x[:-1]+x[1:])
    x_bound = np.concatenate(([2*x_bound[0]-x_bound[1]], x_bound, [2*x_bound[-1]-x_bound[-2]]))
    return x_bound


# Given an array (or two), find the min and max value (unless these are already defined), and pad with the given percentage (default 2%) of the difference between them.
def choose_range (x1, x2=None, xmin=None, xmax=None, pad=0.02):

    xmin_set = xmin is not None
    xmax_set = xmax is not None

    if not xmin_set:
        xmin = np.amin(x1)
        if x2 is not None:
            xmin = min(xmin, np.amin(x2))
    if not xmax_set:
        xmax = np.amax(x1)
        if x2 is not None:
            xmax = max(xmax, np.amax(x2))
            
    delta = pad*(xmax-xmin)
    if not xmin_set:
        xmin -= delta
    if not xmax_set:
        xmax += delta
    return xmin, xmax


# Figure out if a field is depth-dependent, given the last two dimensions being lat and lon, and the possibility of time-dependency.
def is_depth_dependent (data, time_dependent=False):
    return (time_dependent and len(data.shape)==4) or (not time_dependent and len(data.shape)==3)


# Mask everything outside the given bounds. The array must include latitude and longitude dimensions; depth and time are optional.
def mask_outside_box (data, grid, gtype='t', xmin=None, xmax=None, ymin=None, ymax=None, time_dependent=False):
    depth_dependent = is_depth_dependent(data, time_dependent=time_dependent)
    lon, lat = grid.get_lon_lat(gtype=gtype)
    if depth_dependent:
        lon = xy_to_xyz(lon, grid)
        lat = xy_to_xyz(lat, grid)
    if time_dependent:
        lon = add_time_dim(lon, data.shape[0])
        lat = add_time_dim(lat, data.shape[0])
    if xmin is None:
        xmin = np.amin(lon)
    if xmax is None:
        xmax = np.amax(lon)
    if ymin is None:
        ymin = np.amin(lat)
    if ymax is None:
        ymax = np.amax(lat)
    index = np.invert((lon >= xmin)*(lon <= xmax)*(lat >= ymin)*(lat <= ymax))
    return np.ma.masked_where(index, data)


# Given a field with a periodic boundary (in longitude), wrap it on either end so we can interpolate with  no gaps in the middle. If is_lon, add/subtract 360 from these values, if needed, to make sure it's monotonic.
def wrap_periodic (data, is_lon=False):

    # Add 1 column to the beginning and 1 to the end of the longitude dimension
    new_shape = list(data.shape[:-1]) + [data.shape[-1]+2]
    data_wrap = np.empty(new_shape)
    # Copy the middle
    data_wrap[...,1:-1] = data
    # Wrap the edges from either end
    data_wrap[...,0] = data[...,-1]
    data_wrap[...,-1] = data[...,0]
    if is_lon:
        if np.amin(np.diff(data, axis=-1)) < 0:
            print('Error (wrap_periodic): longitude array is not monotonic')
            sys.exit()
        # Add/subtract 360, if needed
        data_wrap[...,0] -= 360
        data_wrap[...,-1] += 360
    return data_wrap


# Given an array of one year of data where the first dimension is time, convert from daily averages to monthly averages.
# If you want to consider leap years, pass the year argument. The default is a year with no leap (1979).
# If there is more than one record per day, set the per_day argument.
def daily_to_monthly (data, year=1979, per_day=1):

    if data.shape[0]//per_day not in [365, 366]:
        print('Error (daily_to_monthly): The first dimension is not time, or else this is not one year of data.')
        sys.exit()
    new_shape = [12] + list(data.shape[1:])
    if isinstance(data, np.ma.MaskedArray):
        data_monthly = np.ma.empty(new_shape)
    else:
        data_monthly = np.empty(new_shape)
    t = 0
    for month in range(12):
        nt = days_per_month(month+1, year)*per_day
        data_monthly[month,...] = np.mean(data[t:t+nt,...], axis=0)
        t += nt
    return data_monthly


# Given a set of titles, find the common parts from the beginning and the end of each title. Trim them and return the master beginning title (trimmed of unnecessary prepositions) as well as the trimmed individual titles.
# For example, the list
# ['Basal mass balance of Pine Island Glacier Ice Shelf',
#  'Basal mass balance of Dotson and Crosson Ice Shelves',
#  'Basal mass balance of Thwaites Ice Shelf']
# would return the master title 'Basal mass balance' and the trimmed titles
# ['Pine Island Glacier', 'Dotson and Crosson', 'Thwaites']
def trim_titles (titles):

    # First replace "shelves" with "shelf" (ignore s so not case sensitive)
    for n in range(len(titles)):
        titles[n] = titles[n].replace('helves', 'helf')
    # Trim the common starts and ends, saving the starts
    title_start = ''
    found = True
    while found:
        found = False
        if all([s[0]==titles[0][0] for s in titles]):
            found = True
            title_start += titles[0][0]
            titles = [s[1:] for s in titles]
        if all([s[-1]==titles[0][-1] for s in titles]):
            found = True
            titles = [s[:-1] for s in titles]
    # Trim any white space
    title_start = title_start.strip()
    # Remove prepositions
    for s in [' of', ' in', ' from']:
        if title_start.endswith(s):
            title_start = title_start[:title_start.index(s)]
    return title_start, titles


# Smooth the given data with a moving average of the given window, and trim and/or shift the time axis too if it's given. The data can be of any number of dimensions; the smoothing will happen on the first dimension.
def moving_average (data, window, time=None, keep_edges=False):

    if window == 0:
        if time is not None:
            return data, time
        else:
            return data

    centered = window%2==1
    if centered:
        radius = (window-1)//2
    else:
        radius = window//2

    # Will have to trim each end by one radius
    t_first = radius
    t_last = data.shape[0] - radius  # First one not selected, as per python convention
    # Need to set up an array of zeros of the same shape as a single time index of data
    shape = [1]
    for t in range(1, len(data.shape)):
        shape.append(data.shape[t])
    zero_base = np.zeros(shape)
    # Do the smoothing in two steps
    data_cumsum = np.ma.concatenate((zero_base, np.ma.cumsum(data, axis=0)), axis=0)
    if centered:
        data_smoothed = (data_cumsum[t_first+radius+1:t_last+radius+1,...] - data_cumsum[t_first-radius:t_last-radius,...])/(2*radius+1)
    else:
        data_smoothed = (data_cumsum[t_first+radius:t_last+radius,...] - data_cumsum[t_first-radius:t_last-radius,...])/(2*radius)
    if keep_edges:
        # Add the edges back on, smoothing them as much as we can with smaller windows.
        if centered:
            data_smoothed_full = np.ma.empty(data.shape)
            data_smoothed_full[t_first:t_last,...] = data_smoothed
            for n in range(radius):
                # Edges at beginning
                data_smoothed_full[n,...] = np.mean(data[:2*n+1,...], axis=0)
                # Edges at end
                data_smoothed_full[-(n+1),...] = np.mean(data[-(2*n+1):,...], axis=0)
            data_smoothed = data_smoothed_full
        else:
            print('Error (moving_average): have not yet coded keep_edges=False for even windows. Want to figure it out?')
            sys.exit()
    if time is not None and not keep_edges:
        if centered:
            time_trimmed = time[radius:time.size-radius]
        else:
            # Need to shift time array half an index forward
            # This will work whether it's datetime or numerical values
            time1 = time[radius-1:time.size-radius-1]
            time2 = time[radius:time.size-radius]
            if isinstance(time[0], int):
                time_trimmed = time1 + (time2-time1)/2.
            else:
                time_trimmed = time1 + (time2-time1)//2 # Can't have a float for datetime            
        return data_smoothed, time_trimmed
    else:
        return data_smoothed


# Return the index of the given start year in the array of Datetime objects.
def index_year_start (time, year0):
    years = np.array([t.year for t in time])
    return np.where(years==year0)[0][0]

# Return the first index after the given end year in the array of Datetime objects.
def index_year_end (time, year0):
    years = np.array([t.year for t in time])
    if years[-1] == year0:
        return years.size
    else:
        return np.where(years>year0)[0][0]

# Do both at once
def index_period (time, year_start, year_end):
    return index_year_start(time, year_start), index_year_end(time, year_end)


# Helper function to make a 2D mask 3D, with masking of bathymetry and optional depth bounds (zmin=deep, zmax=shallow, both negative in metres)
def mask_2d_to_3d (mask, grid, zmin=None, zmax=None):

    if zmin is None:
        zmin = grid.z[-1]
    if zmax is None:
        zmax = grid.z[0]
    mask = xy_to_xyz(mask, grid)
    # Mask out closed cells
    mask *= grid.hfac!=0
    # Mask out everything outside of depth bounds
    z_3d = z_to_xyz(grid.z, grid)
    mask *= (z_3d >= zmin)*(z_3d <= zmax)
    return mask


# Helper function to average 1 year of monthly data from a variable (starting with time index index t0), of any dimension (as long as time is first), with proper monthly weighting for the given calendar (360-day, noleap, or standard - if standard need to provide the year).
def average_12_months (data, t0, calendar='standard', year=None):

    if calendar == 'standard' and year is None:
        print('Error (average_12_months): must provide year')
    if calendar in ['360-day', '360_day']:
        days = None
    else:
        if calendar == 'noleap':
            # Dummy year
            year = 1979
        days = np.array([days_per_month(n, year) for n in np.arange(1,12+1)])
    return np.ma.average(data[t0:t0+12,...], axis=0, weights=days)


# Calculate the depth of the maximum value of the 3D field at each x-y point.
def depth_of_max (data, grid, gtype='t'):

    z_3d = z_to_xyz(grid.z, grid)
    data = mask_3d(data, grid, gtype=gtype)
    # Calculate the maximum value at each point and tile to make 3D
    max_val = np.amax(data, axis=0)
    max_val = xy_to_xyz(max_val, grid)    
    # Get a mask of 1s and 0s which is 1 in the locations where the value equals the maximum in that water column
    max_mask = (data==max_val).astype(float)
    # Make sure there's exactly one such point in each water column
    if np.amax(np.sum(max_mask, axis=0)) > 1:
        # Loop over any problem points
        indices = np.argwhere(np.sum(max_mask,axis=0)>1)
        for index in indices:
            [j,i] = index
        # Choose the shallowest one
        k = np.argmax(max_mask[:,j,i])
        max_mask[:,j,i] = 0
        max_mask[k,j,i] = 1
    # Select z at these points and collapse the vertical dimension
    return np.sum(z_3d*max_mask, axis=-3)


# Calculate the shallowest depth of the given isoline, below the given depth z0.
# Regions where the entire water column is below the given isoline will be set to the seafloor depth; regions where it is entirely above the isoline will trigger an error.
def depth_of_isoline (data, z, val0, z0=None):

    [nz, ny, nx] = data.shape
    if len(z.shape) == 1:
        # Make z 3D
        z = z_to_xyz(z, [nx, ny, nz])
    if z0 is None:
        z0 = 0
    # Get data and depth below each level
    z_bottom = z[-1,:]
    z_below = np.ma.concatenate((z[1:,:], z_bottom[None,:]), axis=0)
    data_bottom = np.ma.masked_where(True, data[-1,:])
    data_below = np.ma.concatenate((data[1:,:], data_bottom[None,:]), axis=0)
    # Find points where the isoline is crossed, in either direction
    mask1 = (data < val0)*(data_below >= val0)*(z <= z0)
    mask2 = (data >= val0)*(data_below < val0)*(z <= z0)
    mask = (mask1.astype(bool) + mask2.astype(bool)).astype(float)
    # Find points where the entire water column below z0 is below or above val0
    mask_below = (np.amax(np.ma.masked_where(z > z0, data), axis=0) < val0)
    mask_above = (np.amin(np.ma.masked_where(z > z0, data), axis=0) > val0)
    # Find the seafloor depth at each point
    bathy = np.amin(np.ma.masked_where(data.mask, z), axis=0)
    # And the land mask
    land_mask = np.sum(np.invert(data.mask), axis=0) == 0
    # Make sure there's at most one point in each water column
    if np.amax(np.sum(mask, axis=0)) > 1:
        # Loop over any problem points
        indices = np.argwhere(np.sum(mask, axis=0)>1)
        for index in indices:
            [j,i] = index
            # Choose the shallowest one
            k = np.argmax(mask[:,j,i])
            mask[:,j,i] = 0
            mask[k,j,i] = 1
    # Select data and depth at these points and collapse the vertical dimension
    data_cross = np.sum(data*mask, axis=0)
    data_below_cross = np.sum(data_below*mask, axis=0)
    z_cross = np.sum(z*mask, axis=0)
    z_below_cross = np.sum(z_below*mask, axis=0)
    # Now interpolate to the given isotherm
    depth_iso = (z_cross - z_below_cross)/(data_cross - data_below_cross)*(val0 - data_cross) + z_cross
    # Deal with regions where there is no such isotherm
    # Mask out the land mask
    depth_iso = np.ma.masked_where(land_mask, depth_iso)
    # Mask out regions shallower than z0
    depth_iso = np.ma.masked_where(bathy > z0, depth_iso)
    # Set to seafloor depth where the entire water column is below val0
    depth_iso[mask_below] = bathy[mask_below]
    # Mask where the entire water column is above val0
    depth_iso = np.ma.masked_where(mask_above, depth_iso)
    return depth_iso
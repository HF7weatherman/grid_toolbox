from pathlib import Path
import xarray as xr
import numpy as np
import healpy as hp
import easygems.healpix as egh

from constants import EARTH_RADIUS

# ------------------------------------------------------------------------------
# Basic HEALPix functionality
# ---------------------------
def _extract_hp_params(var: xr.DataArray) -> tuple[int, np.ndarray]:
    """
    Extracts HEALPix parameters from the given DataArray.

    Parameters
    ----------
    var : xr.DataArray
        The input data array containing HEALPix data.

    Returns
    -------
    tuple[int, np.array]
        A tuple containing the nside parameter and the ring index array.
    """
    nside = egh.get_nside(var)
    nest = egh.get_nest(var)
    if nest:
        ring_index = _nest2ring_index(var, nside)
    else:
        ring_index = None
    return nside, ring_index


def _nest2ring_index(var: xr.DataArray, nside: int) -> np.ndarray:
    """
    Convert nested indices to ring indices for a given variable.

    Parameters:
    -----------
    var : xr.DataArray
        An xr.DataArray containing the nested indices of the HEALPix map in its
        'cell' coordinate.
    nside : int
        The nside parameter defining the resolution of the HEALPix map.

    Returns:
    --------
    numpy.ndarray
        An array of ring indices corresponding to the nested indices.
    """
    return np.array([hp.ring2nest(nside, i) for i in var.cell.values])


def _ring2nest_index(var: xr.DataArray, nside: int) -> np.ndarray:
    """
    Convert ring indices to nested indices for a given variable.

    Parameters:
    -----------
    var : xr.DataArray
        An xr.DataArray containing the ring indices of the HEALPix map in its
        'cell' coordinate.
    nside : int
        The nside parameter defining the resolution of the HEALPix map.

    Returns:
    --------
    numpy.ndarray
        An array of nested indices corresponding to the ring indices.
    """
    return np.array([hp.nest2ring(nside, i) for i in np.arange(len(var))])


# ------------------------------------------------------------------------------
# Remapping from the healpix grid to regular or rectilinear lat-lon grids
# -----------------------------------------------------------------------
def remap_nn_hp2latlon(
        var_hp: xr.DataArray,
        lats: tuple[int, int, int],
        lons: tuple[int, int, int],
        supersampling: dict={"lon": 1, "lat": 1},
        ) -> xr.DataArray:
    """
    Remap a HEALPix grid to a regular or rectilinear latitude-longitude grid
    using nearest neighbor interpolation.

    Parameters
    ----------
    var_hp : xr.DataArray
        The input data array on a Healpix grid.
    lats : tuple[int, int, int]
        A tuple specifying the latitude range and resolution as
        (start, end, num_points).
    lons : tuple[int, int, int]
        A tuple specifying the longitude range and resolution as
        (start, end, num_points).
    supersampling : dict, optional
        A dictionary specifying the supersampling factors for longitude and
        latitude. Default is {"lon": 1, "lat": 1}.

    Returns
    -------
    xr.DataArray
        The remapped data array on a regular or rectilinear latitude-longitude
        grid.
    """
    idx = _get_nn_lon_lat_index(
        egh.get_nside(var_hp),
        np.linspace(lons[0], lons[1], lons[2]*supersampling['lon']),
        np.linspace(lats[0], lats[1], lats[2]*supersampling['lat'])
    )
    return var_hp.drop(['lat', 'lon']).isel(cell=idx).coarsen(
        supersampling).mean(skipna=False)


def _get_nn_lon_lat_index(
        nside: int,
        lons: np.ndarray,
        lats: np.ndarray
        ) -> xr.DataArray:
    """
    Calculate the nearest neighbor HEALPix index for a set of longitudes and
    latitudes and a given nside.

    Parameters
    ----------
    nside : int
        The nside parameter for the HEALPix map.
    lons : array-like
        Array of longitudes.
    lats : array-like
        Array of latitudes.

    Returns
    -------
    xr.DataArray
        DataArray containing the nearest neighbor indices for the given
        longitudes and latitudes.
    """
    lons2, lats2 = np.meshgrid(lons, lats)
    return xr.DataArray(
        hp.ang2pix(nside, lons2, lats2, nest=True, lonlat=True),
        coords=[("lat", lats), ("lon", lons)],
    )


# ------------------------------------------------------------------------------
# Derivatives on the HEALPix grid
# -------------------------------
def compute_gradient_on_hp(
        var: xr.DataArray,
        ) -> tuple[np.ndarray, np.ndarray]:
    """
    Calculates the cartesian gradient of a 2D scalar field on the HEALPix grid.

    Parameters:
    -----------
    var : xr.DataArray
        Input data array representing the 2D scalar field on the HEALPix grid.

    Returns:
    --------
    tuple[np.array, np.array]
        A tuple containing:
        - dvar_dx: Gradient of the scalar field in the x-direction (longitude).
        - dvar_dy: Gradient of the scalar field in the y-direction (latitude).
    """
    nside, ring_index = _extract_hp_params(var)
    dvar_dphi, dvar_dtheta = _compute_hder_hp(
        var.isel(cell=ring_index).values, nside
        )
    return _compute_gradient_on_hp(dvar_dphi, dvar_dtheta, nside)


def compute_laplacian_on_hp(var: xr.DataArray) -> np.ndarray:
    """
    Calculates the cartesian Laplacian of a 2D scalar field on a HEALPix grid.

    Parameters
    ----------
    var : xr.DataArray
        Input 2D scalar field defined on a HEALPix grid. The DataArray should
        have a 'cell' dimension corresponding to the HEALPix cells and a 'lat'
        coordinate for the latitudes of the cells.

    Returns
    -------
    np.array
        The cartesian Laplacian of the input scalar field, reordered to the
        nested indexing scheme.

    Notes
    -----
    The function first preprocesses the input data to obtain the HEALPix nside
    and ring index. It then computes the first and second derivatives of the
    input field with respect to theta and phi. The Laplacian is calculated using
    these derivatives and the tangent of the latitude. Finally, the result is
    reordered to the nested indexing scheme.

    References
    ----------
    - HEALPix: Hierarchical Equal Area isoLatitude Pixelation of a sphere.
    - EARTH_RADIUS is a predefined constant representing the Earth's radius in
    the same units as the input data.
    """
    nside, ring_index = _extract_hp_params(var)
    dvar_dphi, dvar_dtheta = _compute_hder_hp(
        var.isel(cell=ring_index).values, nside
        )
    return _compute_laplacian_on_hp(
        var, dvar_dphi, dvar_dtheta, nside, ring_index
        )
    

def compute_gradient_and_laplacian_on_hp(
        var: xr.DataArray
        ) -> tuple[tuple[np.ndarray, np.ndarray], np.ndarray]:
    """
    Computes both the cartesian gradient and the cartesian Laplacian of a
    variable on a HEALPix grid.

    Parameters
    ----------
    var : xr.DataArray
        Input data array representing the variable for which the gradient and
        Laplacian are to be computed.

    Returns
    -------
    tuple
        A tuple containing the cartesian gradient and cartesian Laplacian of
        the input variable, both reordered to the nested indexing scheme.
    """
    nside, ring_index = _extract_hp_params(var)
    dvar_dphi, dvar_dtheta = _compute_hder_hp(
        var.isel(cell=ring_index).values, nside
        )
    gradient = _compute_gradient_on_hp(dvar_dphi, dvar_dtheta, nside)
    laplacian = _compute_laplacian_on_hp(
        var, dvar_dphi, dvar_dtheta, nside, ring_index
        )
    return gradient, laplacian


def _compute_gradient_on_hp(
        dvar_dphi: xr.DataArray,
        dvar_dtheta: xr.DataArray,
        nside: int,
        ) -> tuple[np.ndarray, np.ndarray]:
    """
    Computes the cartesian gradient components from spherical gradient
    components on a HEALPix grid.

    Parameters
    ----------
    dvar_dphi : xr.DataArray
        The spherical gradient component of the variable with respect to
        longitude (phi).
    dvar_dtheta : xr.DataArray
        The spherical gradient component of the variable with respect to
        latitude (theta).
    nside : int
        The nside parameter of the HEALPix grid.

    Returns
    -------
    tuple[np.array, np.array]
        A tuple containing the cartesian gradients in the x and y directions,
        respectively,  mapped to the nested indexing scheme of the HEALPix grid.
    """
    dvar_dy = -dvar_dtheta/EARTH_RADIUS
    dvar_dx = dvar_dphi/EARTH_RADIUS
    nest_index = _ring2nest_index(dvar_dx, nside)
    return [dvar_dx[nest_index], dvar_dy[nest_index]]


def _compute_laplacian_on_hp(
        var: xr.DataArray,
        dvar_dphi: np.ndarray,
        dvar_dtheta: np.ndarray,
        nside: int,
        ring_index: int,
        ) -> np.ndarray:
    """
    Computes the cartesian Laplacian of a variable on a HEALPix grid.

    Parameters
    ----------
    var : xr.DataArray
        Input data array representing the variable for which the Laplacian is to
        be computed.
    dvar_dphi : np.array
        The spherical derivative of the variable with respect to longitude (phi).
    dvar_dtheta : np.array
        The spherical derivative of the variable with respect to latitude (theta).
    nside : int
        The nside parameter of the HEALPix map, which determines the resolution
        of the map.
    ring_index : int
        The ring index array for the HEALPix map.

    Returns
    -------
    np.array
        The cartesian Laplacian of the input variable, reordered to the nested
        indexing scheme.
    """
    d2var_dphi2, _ = _compute_hder_hp(dvar_dphi, nside)
    _, d2var_dtheta2 = _compute_hder_hp(dvar_dtheta, nside)
    dvar_dtheta_tanlat = dvar_dtheta * \
        np.tan(np.deg2rad(var.lat.isel(cell=ring_index).values))
    laplacian = -(d2var_dtheta2 + dvar_dtheta_tanlat - d2var_dphi2)/\
        (EARTH_RADIUS**2)
    nest_index = _ring2nest_index(laplacian, nside)
    return laplacian[nest_index]


def compute_hor_wind_conv_on_hp(
        ua: xr.DataArray,
        va: xr.DataArray
        ) -> xr.DataArray:
    """
    Computes the horizontal wind convergence on a HEALPix grid.

    Parameters
    ----------
    ua : xr.DataArray
        Zonal wind component.
    va : xr.DataArray
        Meridional wind component.

    Returns
    -------
    np.array
        The horizontal wind convergence.
    """
    nside, ring_index = _extract_hp_params(ua)
    return _compute_hor_wind_conv_on_hp(
        ua.isel(cell=ring_index).values,
        va.isel(cell=ring_index).values,
        ua.lat.isel(cell=ring_index).values,
        nside
    )


def _compute_hor_wind_conv_on_hp(
        ua: xr.DataArray,
        va: xr.DataArray,
        lat: xr.DataArray,
        nside: int
        ) -> np.ndarray:
    """
    Computes the horizontal wind convergence on a HEALPix grid.

    Parameters
    ----------
    ua : xr.DataArray
        Zonal wind component.
    va : xr.DataArray
        Meridional wind component.
    lat : xr.DataArray
        Latitude values.
    nside : int
        The nside parameter of the HEALPix map, which determines the resolution
        of the map.

    Returns
    -------
    np.array
        The horizontal wind convergence.
    """
    dua_dphi, _ = _compute_hder_hp(ua, nside)
    _, dva_dtheta  = _compute_hder_hp(va, nside)
    va_tanlat = va * np.tan(np.deg2rad(lat))
    convergence = -(dua_dphi - dva_dtheta - va_tanlat)/EARTH_RADIUS
    nest_index = _ring2nest_index(convergence, nside)
    return convergence[nest_index]


def _compute_hder_hp(
        var: np.ndarray,
        nside: int
        ) -> tuple[np.ndarray, np.ndarray]:
    """
    Computes the horizontal derivatives of a variable (1 vertical level, 1 time)
    using spherical harmonics.

    Parameters
    ----------
    var : numpy.ndarray
        Input array representing the variable for which the horizontal
        derivatives are to be computed.
    nside : int
        The nside parameter of the HEALPix map, which determines the resolution
        of the map.

    Returns
    -------
    tuple of numpy.ndarray
        A tuple containing two arrays:
        - dvar_dphi (numpy.ndarray): The derivative of the variable with respect
            to longitude (phi).
        - dvar_dtheta (numpy.ndarray): The derivative of the variable with
            respect to latitude (theta).

    Notes
    -----
    This function uses the HEALPix library to perform spherical harmonic
    transformations.
    """
    var_alm = hp.sphtfunc.map2alm(var)
    der_arr = hp.sphtfunc.alm2map_der1(var_alm, nside)
    return der_arr[2, :], der_arr[1, :]


# ------------------------------------------------------------------------------
# Derivatives on regular or rectilinear lat-lon grids
# ---------------------------------------------------
def absolute_gradient(
        gradient: tuple[xr.DataArray, xr.DataArray]
        ) -> xr.DataArray:
    """
    Computes the absolute gradient from the given gradient components.

    Parameters
    ----------
    gradient : tuple[xr.DataArray, xr.DataArray]
        A tuple containing the gradient components (dvar_dx, dvar_dy).

    Returns
    -------
    xr.DataArray
        The absolute gradient.
    """
    return np.sqrt(gradient[0]**2 + gradient[1]**2)


def compute_gradient_on_latlon(
        var: xr.DataArray
        ) -> tuple[xr.DataArray, xr.DataArray]:
    """
    Computes the cartesian gradient of a variable on regular or rectilinear
    lat-lon grids.

    Parameters
    ----------
    var_latlon : xr.DataArray
        The input data array on a regular or rectilinear lat-lon grid.

    Returns
    -------
    tuple[xr.DataArray, xr.DataArray]
        A tuple containing:
        - dvar_dx: Cartesian gradient of the variable in the longitude direction.
        - dvar_dy: Cartesian gradient of the variable in the latitude direction.
    """
    var = _deg2rad_coordinates(var)
    dvar_dphi, dvar_dlambda = _compute_hder_on_latlon(var)
    return _compute_gradient_on_latlon(dvar_dphi, dvar_dlambda)
    

def compute_laplacian_on_latlon(var: xr.DataArray) -> xr.DataArray:
    """
    Computes the cartesian Laplacian of a variable on regular or rectilinear
    lat-lon grids.

    Parameters
    ----------
    var_latlon : xr.DataArray
        The input data array on a regular or rectilinear lat-lon grid.

    Returns
    -------
    xr.DataArray
        The cartesian Laplacian of the input variable.
    """
    var = _deg2rad_coordinates(var)
    dvar_dphi, dvar_dlambda = _compute_hder_on_latlon(var)
    return _compute_laplacian_on_latlon(var, dvar_dphi, dvar_dlambda)


def compute_gradient_and_laplacian_on_latlon(
        var: xr.DataArray
        ) -> tuple[tuple[xr.DataArray, xr.DataArray], xr.DataArray]:
    """
    Computes both the cartesian gradient and the cartesian Laplacian of a
    variable on regular or rectilinear lat-lon grids.

    Parameters
    ----------
    var_latlon : xr.DataArray
        The input data array on a regular or rectilinear lat-lon grid.

    Returns
    -------
    tuple[tuple[xr.DataArray, xr.DataArray], xr.DataArray]
        A tuple containing:
        - gradient: A tuple with the cartesian gradient components
                    (dvar_dx, dvar_dy).
        - laplacian: The cartesian Laplacian of the input variable.
    """
    var = _deg2rad_coordinates(var)
    dvar_dphi, dvar_dlambda = _compute_hder_on_latlon(var)
    gradient = _compute_gradient_on_latlon(dvar_dphi, dvar_dlambda)
    laplacian = _compute_laplacian_on_latlon(var, dvar_dphi, dvar_dlambda)
    return gradient, laplacian


def _compute_gradient_on_latlon(
        dvar_dphi: xr.DataArray,
        dvar_dlambda: xr.DataArray
        ) -> tuple[xr.DataArray, xr.DataArray]:
    """
    Computes the cartesian gradient components from spherical gradient
    components.

    Parameters
    ----------
    dvar_dphi : xr.DataArray
        The spherical gradient component with respect to longitude.
    dvar_dtheta : xr.DataArray
        The spherical gradient component with respect to latitude.

    Returns
    -------
    tuple[xr.DataArray, xr.DataArray]
        A tuple containing the cartesian gradient components (dvar_dx, dvar_dy).
    """
    return [dvar_dphi/EARTH_RADIUS, dvar_dlambda/EARTH_RADIUS]


def _compute_hder_on_latlon(
        var: xr.DataArray
        ) -> tuple[xr.DataArray, xr.DataArray]:
    """
    Computes the spherical horizontal derivatives on regular or rectilinear
    lat-lon grids.

    Parameters
    ----------
    var_latlon : xr.DataArray
        The input data array on a regular or rectilinear lat-lon grid.

    Returns
    -------
    tuple[xr.DataArray, xr.DataArray]
        A tuple containing the spherical horizontal derivatives
        (dvar_dphi, dvar_dtheta).
    """
    dvar_dphi = var.differentiate('lon_rad') * 1/np.cos(var['lat_rad'])
    dvar_dlambda = var.differentiate('lat_rad')
    return dvar_dphi, dvar_dlambda


def _compute_laplacian_on_latlon(
        var: xr.DataArray,
        dvar_dphi: xr.DataArray,
        dvar_dlambda: xr.DataArray,
        ) -> xr.DataArray:
    """
    Computes the cartesian Laplacian on regular or rectilinear lat-lon grids.

    Parameters
    ----------
    var_latlon : xr.DataArray
        The input data array on a regular or rectilinear lat-lon grid.
    dvar_dphi : xr.DataArray
        The spherical gradient component with respect to longitude.
    dvar_dtheta : xr.DataArray
        The spherical gradient component with respect to latitude.

    Returns
    -------
    xr.DataArray
        The cartesian Laplacian of the input variable.
    """
    d2var_dphi2 = dvar_dphi.differentiate('lon_rad') * 1/np.cos(var['lat_rad'])
    d2var_dlambda2 = dvar_dlambda.differentiate('lat_rad')
    dvar_dtheta_tanlat = dvar_dlambda * np.tan(var['lat_rad'])
    return -(d2var_dlambda2 + dvar_dtheta_tanlat - d2var_dphi2)/\
        (EARTH_RADIUS**2)


def compute_hor_wind_conv_on_latlon(
        ua: xr.DataArray,
        va: xr.DataArray,
        ) -> tuple[xr.DataArray, xr.DataArray]:
    """
    Computes the cartesian gradient of a variable on regular or rectilinear
    lat-lon grids.

    Parameters
    ----------
    var_latlon : xr.DataArray
        The input data array on a regular or rectilinear lat-lon grid.

    Returns
    -------
    tuple[xr.DataArray, xr.DataArray]
        A tuple containing:
        - dvar_dx: Cartesian gradient of the variable in the longitude direction.
        - dvar_dy: Cartesian gradient of the variable in the latitude direction.
    """
    ua = _deg2rad_coordinates(ua)
    va = _deg2rad_coordinates(va)
    dua_dphi, _ = _compute_hder_on_latlon(ua)
    _, dva_dlambda = _compute_hder_on_latlon(va)
    va_tanlat = va * np.tan(va['lat_rad'])
    convergence = -(dua_dphi + dva_dlambda - va_tanlat)/EARTH_RADIUS
    return convergence


def _deg2rad_coordinates(var_latlon: xr.DataArray) -> xr.DataArray:
    """
    Converts the coordinates of a variable from degrees to radians.

    Parameters
    ----------
    var_latlon : xr.DataArray
        The input data array with coordinates in degrees.

    Returns
    -------
    xr.DataArray
        The input data array with additional coordinates in radians.
    """
    return var_latlon.assign_coords({
        "lon_rad": (np.deg2rad(var_latlon['lon'])),
        "lat_rad": (np.deg2rad(var_latlon['lat'])),
        })


# ------------------------------------------------------------------------------
# Helper functions
# ------------------------------------------------------------------------------
def get_ocean_fraction_surface(inpath: Path, hp_zoom: int=9) -> xr.Dataset:
    """
    Get the ocean fraction surface data on the HEALPix grid from a NetCDF file.

    Parameters
    ----------
    inpath : Path or str
        The input path to the directory containing the NetCDF file.
    hp_zoom : int, optional
        The HEALPix zoom level of the data, by default 9.

    Returns
    -------
    xarray.Dataset
        The ocean fraction surface dataset on the HEALPix grid with attached
        coordinates.

    Notes
    -----
    The function expects the NetCDF file to be named in the format
    'ocean_fraction_surface_hpz{hp_zoom}.nc' within the provided directory.
    """
    return xr.open_dataset(
        str(inpath/Path(f'ocean_fraction_surface_hpz{hp_zoom}.nc'))
        ).pipe(egh.attach_coords)
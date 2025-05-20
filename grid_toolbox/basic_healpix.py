import xarray as xr
import numpy as np
import healpy as hp
import easygems.healpix as egh
from typing import Tuple

# ------------------------------------------------------------------------------
# Basic HEALPix functionality
# ---------------------------
def _extract_hp_params(var: xr.DataArray) -> Tuple[int, np.ndarray]:
    """
    Extracts HEALPix parameters from the given DataArray.

    Parameters
    ----------
    var : xr.DataArray
        The input data array containing HEALPix data.

    Returns
    -------
    Tuple[int, np.array]
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
    return np.array([hp.nest2ring(nside, i) for i in var.cell.values])


# ------------------------------------------------------------------------------
# Remapping from the healpix grid to regular or rectilinear lat-lon grids
# -----------------------------------------------------------------------
def remap_nn_hp2latlon(
        var_hp: xr.DataArray,
        lats: Tuple[int, int, int],
        lons: Tuple[int, int, int],
        supersampling: dict={"lon": 1, "lat": 1},
        ) -> xr.DataArray:
    """
    Remap a HEALPix grid to a regular or rectilinear latitude-longitude grid
    using nearest neighbor interpolation.

    Parameters
    ----------
    var_hp : xr.DataArray
        The input data array on a Healpix grid.
    lats : Tuple[int, int, int]
        A tuple specifying the latitude range and resolution as
        (start, end, num_points).
    lons : Tuple[int, int, int]
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
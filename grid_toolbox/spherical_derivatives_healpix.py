import xarray as xr
import numpy as np
import healpy as hp
from typing import Tuple

from grid_toolbox.constants import EARTH_RADIUS
from grid_toolbox.basic_healpix import _extract_hp_params, _ring2nest_index

def compute_gradient_on_hp(
        var: xr.DataArray,
        ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculates the cartesian gradient of a 2D scalar field on the HEALPix grid.

    Parameters:
    -----------
    var : xr.DataArray
        Input data array representing the 2D scalar field on the HEALPix grid.

    Returns:
    --------
    Tuple[np.array, np.array]
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
        ) -> Tuple[Tuple[np.ndarray, np.ndarray], np.ndarray]:
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
    Tuple
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
        ) -> Tuple[np.ndarray, np.ndarray]:
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
    Tuple[np.array, np.array]
        A tuple containing the cartesian gradients in the x and y directions,
        respectively,  mapped to the nested indexing scheme of the HEALPix grid.
    """
    dvar_dy = -dvar_dtheta/EARTH_RADIUS
    dvar_dx = dvar_dphi/EARTH_RADIUS
    nest_index = _ring2nest_index(dvar_dx, nside)
    return (dvar_dx[nest_index], dvar_dy[nest_index])


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
        ) -> Tuple[np.ndarray, np.ndarray]:
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
    Tuple of numpy.ndarray
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
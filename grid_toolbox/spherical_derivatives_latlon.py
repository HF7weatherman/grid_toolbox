import xarray as xr
import numpy as np
from typing import Tuple

from constants import EARTH_RADIUS

# ------------------------------------------------------------------------------
# Derivatives on regular or rectilinear lat-lon grids
# ---------------------------------------------------
def absolute_gradient(
        gradient: Tuple[xr.DataArray, xr.DataArray]
        ) -> xr.DataArray:
    """
    Computes the absolute gradient from the given gradient components.

    Parameters
    ----------
    gradient : Tuple[xr.DataArray, xr.DataArray]
        A tuple containing the gradient components (dvar_dx, dvar_dy).

    Returns
    -------
    xr.DataArray
        The absolute gradient.
    """
    return np.sqrt(gradient[0]**2 + gradient[1]**2)


def compute_gradient_on_latlon(
        var: xr.DataArray
        ) -> Tuple[xr.DataArray, xr.DataArray]:
    """
    Computes the cartesian gradient of a variable on regular or rectilinear
    lat-lon grids.

    Parameters
    ----------
    var_latlon : xr.DataArray
        The input data array on a regular or rectilinear lat-lon grid.

    Returns
    -------
    Tuple[xr.DataArray, xr.DataArray]
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
        ) -> Tuple[Tuple[xr.DataArray, xr.DataArray], xr.DataArray]:
    """
    Computes both the cartesian gradient and the cartesian Laplacian of a
    variable on regular or rectilinear lat-lon grids.

    Parameters
    ----------
    var_latlon : xr.DataArray
        The input data array on a regular or rectilinear lat-lon grid.

    Returns
    -------
    Tuple[Tuple[xr.DataArray, xr.DataArray], xr.DataArray]
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
        ) -> Tuple[xr.DataArray, xr.DataArray]:
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
    Tuple[xr.DataArray, xr.DataArray]
        A tuple containing the cartesian gradient components (dvar_dx, dvar_dy).
    """
    return (dvar_dphi/EARTH_RADIUS, dvar_dlambda/EARTH_RADIUS)


def _compute_hder_on_latlon(
        var: xr.DataArray
        ) -> Tuple[xr.DataArray, xr.DataArray]:
    """
    Computes the spherical horizontal derivatives on regular or rectilinear
    lat-lon grids.

    Parameters
    ----------
    var_latlon : xr.DataArray
        The input data array on a regular or rectilinear lat-lon grid.

    Returns
    -------
    Tuple[xr.DataArray, xr.DataArray]
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
        ) -> Tuple[xr.DataArray, xr.DataArray]:
    """
    Computes the cartesian gradient of a variable on regular or rectilinear
    lat-lon grids.

    Parameters
    ----------
    var_latlon : xr.DataArray
        The input data array on a regular or rectilinear lat-lon grid.

    Returns
    -------
    Tuple[xr.DataArray, xr.DataArray]
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
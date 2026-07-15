import xarray as xr
import numpy as np
from typing import Tuple

from grid_toolbox.constants import EARTH_RADIUS

# ------------------------------------------------------------------------------
# Simple Cartesian derivatives on spherical coordinates (lat-lon grids)
# ---------------------------------------------------------------------
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
    return (dvar_dphi, dvar_dlambda)
    

def compute_laplacian_on_latlon(
        var: xr.DataArray,
        components: bool=False,
        ) -> xr.DataArray:
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
    laplacian_components = _compute_laplacian_components_on_latlon(
        var, dvar_dphi, dvar_dlambda,
        )
    if components:
        return laplacian_components
    else:
        return laplacian_components[0] + laplacian_components[1]


def compute_gradient_and_laplacian_on_latlon(
        var: xr.DataArray,
        components_laplacian: bool=False,
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
    gradient = (dvar_dphi, dvar_dlambda)
    laplacian_components = _compute_laplacian_components_on_latlon(
        var, dvar_dphi, dvar_dlambda,
        )
    if components_laplacian:
        laplacian = laplacian_components
    else:
        laplacian = laplacian_components[0] + laplacian_components[1]
    return gradient, laplacian


# ------------------------------------------------------------------------------
# Complex Cartesian derivative matrices on spherical coordinates (lat-lon grids)
# ------------------------------------------------------------------------------ß
def compute_2d_jacobian_on_latlon(
        f1: xr.DataArray,
        f2: xr.DataArray,
        ) -> xr.DataArray:
    """
    Computes the cartesian Jacobian of two variables on regular or rectilinear
    lat-lon grids.

    Parameters
    ----------
    f1: xr.DataArray
        The first input data array on a regular or rectilinear lat-lon grid.
    f2: xr.DataArray
        The second input data array on a regular or rectilinear lat-lon grid.

    Returns
    -------
    xr.DataArray
        The cartesian Jacobian of the two input variables.
    """
    f1_name = f1.name if f1.name is not None else 'f1'
    f2_name = f2.name if f2.name is not None else 'f2'

    f1 = _deg2rad_coordinates(f1)
    f2 = _deg2rad_coordinates(f2)
    df1_dphi, df1_dlambda = _compute_hder_on_latlon(f1)
    df2_dphi, df2_dlambda = _compute_hder_on_latlon(f2)
    jacobian_latlon = xr.merge([
        df1_dphi.rename(f'd{f1_name}_dx'),
        df1_dlambda.rename(f'd{f1_name}_dy'),
        df2_dphi.rename(f'd{f2_name}_dx'),
        df2_dlambda.rename(f'd{f2_name}_dy')
        ])
    return jacobian_latlon


def compute_2d_covariant_hessian_on_latlon(
        f: xr.DataArray,
        ) -> xr.DataArray:
    """
    Computes the cartesian Hessian of a variable on regular or rectilinear
    lat-lon grids.

    Parameters
    ----------
    f: xr.DataArray
        The input data array on a regular or rectilinear lat-lon grid.

    Returns
    -------
    xr.DataArray
        The cartesian Hessian of the input variable.
    """
    var_name = f.name if f.name is not None else 'f'
    f = _deg2rad_coordinates(f)
    df_dphi, df_dlambda = _compute_hder_on_latlon(f)
    d2f_dphi2, _ = _compute_hder_on_latlon(df_dphi)
    d2f_dlambda_dphi, d2f_dlambda2 = _compute_hder_on_latlon(df_dlambda)

    d2f_dxx = d2f_dphi2 - (df_dlambda * np.tan(f['lat_rad']) / EARTH_RADIUS)
    d2f_dyy = d2f_dlambda2
    d2f_dxy = d2f_dlambda_dphi + (\
        (df_dlambda * np.tan(f['lat_rad'])) / \
        (EARTH_RADIUS * np.cos(f['lat_rad']))
        )
    hessian_latlon = xr.merge([
        d2f_dxx.rename(f'd{var_name}_dxx'),
        d2f_dyy.rename(f'd{var_name}_dyy'),
        d2f_dxy.rename(f'd{var_name}_dxy')
        ])
    return hessian_latlon


# ------------------------------------------------------------------------------
# Horizontal wind convergence on regular or rectilinear lat-lon grids
# ------------------------------------------------------------------------------
def compute_hor_wind_conv_on_latlon(
        u: xr.DataArray,
        v: xr.DataArray,
        ) -> xr.DataArray:
    """
    Computes the cartesian gradient of a variable on regular or rectilinear
    lat-lon grids.

    Parameters
    ----------
    var_latlon : xr.DataArray
        The input data array on a regular or rectilinear lat-lon grid.

    Returns
    -------
    xr.DataArray
        convegence: Cartesian convergence of a horizontal flow field.
    """
    components = _compute_hor_wind_conv_components_on_latlon(u, v)
    return components['conv_u'] + components['conv_v']


def compute_hor_wind_conv_components_on_latlon(
        u: xr.DataArray,
        v: xr.DataArray,
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
        - convergence_ua: Cartesian convergence of zonal flow component
        - convergence_va: Cartesian convergence of meridional flow component
    """
    return _compute_hor_wind_conv_components_on_latlon(u, v)


def _compute_hor_wind_conv_components_on_latlon(
        u: xr.DataArray,
        v: xr.DataArray,
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
        - convergence_ua: Cartesian convergence of zonal flow component
        - convergence_va: Cartesian convergence of meridional flow component
    """
    u = _deg2rad_coordinates(u)
    v = _deg2rad_coordinates(v)
    du_dphi, _ = _compute_hder_on_latlon(u)
    _, dv_dlambda = _compute_hder_on_latlon(v)
    v_tanlat = v * np.tan(v['lat_rad'])
    convergence_u = -du_dphi
    convergence_v = -(dv_dlambda - v_tanlat/EARTH_RADIUS)
    return xr.merge(
        [convergence_u.rename('conv_u'), convergence_v.rename('conv_v')]
    )


# ------------------------------------------------------------------------------
# Low-level functions
# -------------------
def _compute_laplacian_components_on_latlon(
        f: xr.DataArray,
        df_dphi: xr.DataArray,
        df_dlambda: xr.DataArray,
        ) -> xr.DataArray:
    """
    Computes the cartesian Laplacian on regular or rectilinear lat-lon grids.

    Parameters
    ----------
    f : xr.DataArray
        The input data array on a regular or rectilinear lat-lon grid.
    df_dphi : xr.DataArray
        The spherical gradient component with respect to longitude.
    df_dlambda : xr.DataArray
        The spherical gradient component with respect to latitude.

    Returns
    -------
    xr.DataArray
        The cartesian Laplacian of the input variable.
    """
    d2f_dphi2, _ = _compute_hder_on_latlon(df_dphi)
    _, d2f_dlambda2 = _compute_hder_on_latlon(df_dlambda)
    df_dtheta_tanlat = (df_dlambda / EARTH_RADIUS) * np.tan(f['lat_rad'])

    return (d2f_dphi2, d2f_dlambda2 - df_dtheta_tanlat)


def _compute_hder_on_latlon(
        f: xr.DataArray
        ) -> Tuple[xr.DataArray, xr.DataArray]:
    """
    Computes the spherical horizontal derivatives on regular or rectilinear
    lat-lon grids.

    Parameters
    ----------
    f : xr.DataArray
        The input data array on a regular or rectilinear lat-lon grid.

    Returns
    -------
    Tuple[xr.DataArray, xr.DataArray]
        A tuple containing the spherical horizontal derivatives
        (df_dphi, df_dlambda).
    """
    df_dphi = f.differentiate('lon_rad') / np.cos(f['lat_rad'])
    df_dlambda = f.differentiate('lat_rad')
    return df_dphi/EARTH_RADIUS, df_dlambda/EARTH_RADIUS


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
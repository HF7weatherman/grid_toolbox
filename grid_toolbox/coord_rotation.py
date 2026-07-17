import numpy as np
import xarray as xr

from typing import Tuple, Dict

from grid_toolbox.spherical_derivatives_latlon import \
    compute_2d_jacobian_on_latlon, compute_2d_covariant_hessian_on_latlon


# ------------------------------------------------------------------------------
# Top-level functions
# -------------------
def rotate_vector_by_wind(
        vector: Dict[str, xr.DataArray],
        wind: Dict[str, xr.DataArray],
        ) -> Tuple[xr.DataArray, xr.DataArray]:
    rot_angle = np.arctan2(wind['v'], wind['u']).rename('rotation_angle')
    return _rotate_vector(vector, rot_angle)


def calc_wind_rotated_divergence(
        vector: Dict[str, xr.DataArray],
        wind: Dict[str, xr.DataArray],
        ) -> Tuple[xr.DataArray, xr.DataArray]:
    rot_angle = np.arctan2(wind['v'], wind['u']).rename('rotation_angle')
    return _calc_rotated_divergence(vector, rot_angle)


def calc_wind_rotated_laplacian(
        field: xr.DataArray,
        wind: Dict[str, xr.DataArray],
        ) -> Tuple[xr.DataArray, xr.DataArray]:
    rot_angle = np.arctan2(wind['v'], wind['u']).rename('rotation_angle')
    return _calc_rotated_laplacian(field, rot_angle)


# ------------------------------------------------------------------------------
# Low-level functions
# -------------------
def _rotate_vector(
        vector: Dict[str, xr.DataArray],
        angle: xr.DataArray
        ) -> Tuple[xr.DataArray, xr.DataArray]:
    cos_angle = np.cos(angle)
    sin_angle = np.sin(angle)
    x_rot = cos_angle * vector['x'] + sin_angle * vector['y']
    y_rot = -sin_angle * vector['x'] + cos_angle * vector['y']
    return (x_rot, y_rot)


def _calc_rotated_divergence(
        vector: Dict[str, xr.DataArray],
        angle: xr.DataArray,
        ) -> Tuple[xr.DataArray, xr.DataArray]:
    jacobian = compute_2d_jacobian_on_latlon(
        vector['x'].rename('u'),
        vector['y'].rename('v')
        )
    cos_angle = np.cos(angle)
    sin_angle = np.sin(angle)

    dur_dr = \
        cos_angle**2 * jacobian['du_dx'] + \
        sin_angle**2 * jacobian['dv_dy'] + \
        cos_angle * sin_angle * (jacobian['du_dy'] + jacobian['dv_dx'])
    dus_ds = \
        sin_angle**2 * jacobian['du_dx'] + \
        cos_angle**2 * jacobian['dv_dy'] - \
        cos_angle * sin_angle * (jacobian['du_dy'] + jacobian['dv_dx'])
    return (dur_dr, dus_ds)
    

def _calc_rotated_laplacian(
        field: xr.DataArray,
        angle: xr.DataArray,
        ) -> Tuple[xr.DataArray, xr.DataArray]:
    hessian = compute_2d_covariant_hessian_on_latlon(field)
    cos_angle = np.cos(angle)
    sin_angle = np.sin(angle)

    vname = field.name if field.name is not None else 'f'
    dur_dr = \
        cos_angle**2 * hessian[f'd{vname}_dxx'] + \
        sin_angle**2 * hessian[f'd{vname}_dyy'] + \
        2 * cos_angle * sin_angle * hessian[f'd{vname}_dxy']
    dus_ds = \
        sin_angle**2 * hessian[f'd{vname}_dxx'] + \
        cos_angle**2 * hessian[f'd{vname}_dyy'] - \
        2 * cos_angle * sin_angle * hessian[f'd{vname}_dxy']
    return (dur_dr, dus_ds)
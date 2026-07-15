import numpy as np
import xarray as xr

from grid_toolbox.spherical_derivatives_latlon import \
    compute_2d_jacobian_on_latlon, compute_2d_covariant_hesse_on_latlon

def rotate_vector_by_wind(
        vector: dict[str, xr.DataArray],
        wind: dict[str, xr.DataArray],
        ) -> tuple[xr.DataArray, xr.DataArray]:
    rot_angle = np.arctan2(wind['v'], wind['u']).rename('rotation_angle')
    return _rotate_vector(vector, rot_angle)


def _rotate_vector(
        vector: dict[str, xr.DataArray],
        angle: xr.DataArray
        ) -> tuple[xr.DataArray, xr.DataArray]:
    cos_angle = np.cos(angle)
    sin_angle = np.sin(angle)
    x_rot = cos_angle * vector['x'] + sin_angle * vector['y']
    y_rot = -sin_angle * vector['x'] + cos_angle * vector['y']
    return (x_rot, y_rot)


def calc_wind_rotated_divergence(
        vector: dict[str, xr.DataArray],
        wind: dict[str, xr.DataArray],
        ) -> tuple[xr.DataArray, xr.DataArray]:
    rot_angle = np.arctan2(wind['v'], wind['u']).rename('rotation_angle')
    return _calc_rotated_divergence(vector, rot_angle)
    

def _calc_rotated_divergence(
        vector: dict[str, xr.DataArray],
        angle: xr.DataArray,
        ) -> tuple[xr.DataArray, xr.DataArray]:
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


#-------------------------------------------------------------------------------
def calc_wind_rotated_laplacian(
        field: xr.DataArray,
        wind: dict[str, xr.DataArray],
        ) -> tuple[xr.DataArray, xr.DataArray]:
    rot_angle = np.arctan2(wind['v'], wind['u']).rename('rotation_angle')
    return _calc_rotated_laplacian(field, rot_angle)
    

def _calc_rotated_laplacian(
        field: xr.DataArray,
        angle: xr.DataArray,
        ) -> tuple[xr.DataArray, xr.DataArray]:
    hessian = compute_2d_covariant_hesse_on_latlon(field)
    cos_angle = np.cos(angle)
    sin_angle = np.sin(angle)

    dur_dr = \
        cos_angle**2 * hessian['dvar_dxx'] + \
        sin_angle**2 * hessian['dvar_dyy'] + \
        2 * cos_angle * sin_angle * hessian['dvar_dxy']
    dus_ds = \
        sin_angle**2 * hessian['dvar_dxx'] + \
        cos_angle**2 * hessian['dvar_dyy'] - \
        2 * cos_angle * sin_angle * hessian['dvar_dxy']
    return (dur_dr, dus_ds)
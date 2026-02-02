"""Compute grid areas."""

import numpy as np
import xarray as xr

from grid_toolbox.constants import EARTH_RADIUS

_LAT_LON_PRECISION = 1E-5 # 1 meter at the equator

def get_cells_area(variable: xr.DataArray) -> xr.DataArray:
    """Return the area of each cell on the grid in meters squares."""
    _check_grid_dimensions(variable)
    lat = np.radians(variable.lat)
    lon = np.radians(variable.lon)
    lat, lon = xr.broadcast(lat, lon)
    dlat, dlon = _get_grid_spacing(lat, lon)
    cells_area = (EARTH_RADIUS**2)*np.cos(lat) * dlat * dlon
    return cells_area

def _check_grid_dimensions(variable):
    grid_coordinates = set(variable.coords.keys()) 
    expected_coordinates = {"lon", "lat"}
    if not grid_coordinates.intersection(expected_coordinates):
        msg = ("Invalid coordinates detected. Expecting DataArray with only "
               "'lat' and 'lon' coordinates")
        raise AttributeError(msg)

def _get_grid_spacing(lat, lon):
    dlons = lon.diff("lon").values
    dlats = lat.diff("lat").values
    dlon_var = dlons.std()
    dlat_var = dlats.std()      
    if (dlon_var > _LAT_LON_PRECISION) or (dlat_var > _LAT_LON_PRECISION):
        msg = "Grid must be equally spaced in lon and lat."
        raise ValueError(msg)
    dlon = dlons.mean()
    dlat = dlats.mean()
    return dlat, dlon

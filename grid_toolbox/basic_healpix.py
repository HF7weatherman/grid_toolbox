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


def guess_gridn(da: xr.DataArray) -> str:
    """
    Try to guess the name of the spatial coordinate name from a list of frequent
    options. Developed by Lukas Brunner, UHH."""
    dims = list(da.dims)
    gridn = []
    if 'values' in dims:
        gridn.append('values')
    if 'value' in dims:
        gridn.append('value')
    if 'cell' in dims:
        gridn.append('cell')
    if 'x' in dims:
        gridn.append('x')
    if len(gridn) == 1:
        return gridn[0]

    raise ValueError(
        'gridn needs to be set manually to one of: {}'.format(', '.join(dims))
        )


def rechunk_along_griddim(data: xr.Dataset) -> xr.Dataset:
    """
    Rechunks an xarray Dataset along its grid dimension.

    This function determines the grid dimension name using the internal
    `_guess_gridn` function and rechunks the dataset so that all data
    along this dimension is contained in a single chunk.

    Parameters
    ----------
    data : xr.Dataset
        The input xarray Dataset to rechunk.

    Returns
    -------
    xr.Dataset
        The rechunked xarray Dataset, with the grid dimension in a single chunk.
    """
    gridn = guess_gridn(data)
    return data.chunk({gridn: -1})


def coarsen_hp_grid_xr(
        da: xr.DataArray,
        z_out: int,
        method: str='mean',
        gridn=None
        ) -> xr.DataArray:
    """
    Thin xarray wrapper for `aggregate_grid'. Developed by Lukas Brunner, UHH
    """
    npix_out = hp.nside2npix(2**z_out)
    if gridn is None:  # try to guess grid name from frequent options
        gridn = guess_gridn(da)
            
    return xr.apply_ufunc(
        _coarsen_hp_grid,
        da, z_out,
        input_core_dims=[[gridn], []],
        dask = "parallelized",
        vectorize=True,
        output_core_dims=[['cell']],
        kwargs={'method': method},
        dask_gufunc_kwargs = {"output_sizes": {"cell": npix_out}},
        output_dtypes=["f8"],
    )


def _coarsen_hp_grid(
        arr: np.ndarray,
        z_out: int,
        method: str='mean'
        ) -> np.ndarray:
    """Spatially aggregate to a coarser grid. Developed by Lukas Brunner, UHH.

    Parameters
    ----------
    arr : np.ndarray, shape (M,)
        The length of the array M has to be M = 12 * (2**zoom)**2
    z_out : int
        Healpix zoom level of the output grid. Needs to be smaller than the input zoom level.
    method : str, optional by default 'mean'
        Spatial aggregation method. 
        - 'mean': Mean of sub-grid cells -> conservative regridding
                  This is equivalent to `healpy.ud_grade`
        - 'std': Standard deviation of sub-grid cells
        - 'min': Minimum of sub-grid cells
        - 'max': Maximum of sub-grid cells

    Returns
    -------
    np.ndarray, shape (N < M,)

    Info
    ----
    Zoom levels in healpix (Hierarchical Equal Area isoLatitude Pixelization of a sphere)
    https://healpy.readthedocs.io/en/latest/index.html
    https://healpix.jpl.nasa.gov/index.shtml

    Zoom level 0 divides the globe into 12 grid cells, each zoom level
    increase increases the number of cells by 4 and half the resolution
    in kilometer

    nside = 2**zoom
    nr. cells = 12 * nside**2

    | zoom | nside | res. (km) | nr. cells  |
    | ---- | ----- | --------- | ---------- |
         0 |     1 |    6519.6 | 12
         1 |     2 |    3259.8 | 48
         2 |     4 |    1629.9 | 192
         3 |     8 |     815.0 | 768
         4 |    16 |     407.5 | 3,072
         5 |    32 |     203.7 | 12,288
         6 |    64 |     101.9 | 49,152
         7 |   128 |      50.9 | 196,608
         8 |   256 |      25.5 | 786,432
         9 |   512 |      12.7 | 3,145,728
        10 |  1024 |       6.4 | 12,582,912
        11 |  2048 |       3.2 | 50,331,648
        12 |  4096 |       1.6 | 201,326,592
    """
    npix_in = arr.size
    npix_out = hp.nside2npix(2**z_out)    
    if npix_out > npix_in:
        error = 'Output zoom level needs to be smaller than input zoom level'
        raise ValueError(error)
    elif npix_out == npix_in:
        note = 'Output zoom level is the same as input zoom level. Thus, ' + \
            'there is no need for coarsening and the input data is returned.'
        print(note)
        return arr

    ratio = npix_in / npix_out
    if not ratio.is_integer():  # this should never happen
        raise ValueError(f'{ratio=}')
    else:
        ratio = int(ratio)
    
    if method == 'mean':
        return arr.reshape(npix_out, ratio).mean(axis=-1)
    if method == 'std':
        return arr.reshape(npix_out, ratio).std(axis=-1)
    if method == 'min':
        return arr.reshape(npix_out, ratio).min(axis=-1)
    if method == 'max':
        return arr.reshape(npix_out, ratio).max(axis=-1)
        
    raise ValueError(f'{method=}')


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
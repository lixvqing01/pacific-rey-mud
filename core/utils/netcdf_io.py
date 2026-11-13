"""Read NetCDF3/4 files through backends supporting Unicode file paths."""
from pathlib import Path
import xarray as xr


def open_dataset(path, **kwargs):
    """Choose SciPy for NetCDF3 and h5netcdf for HDF5-backed NetCDF4.

    Backend selection changes file access only; xarray retains its default
    masking, scaling and coordinate decoding. Explicit engine requests win.
    """
    if 'engine' not in kwargs:
        with Path(path).open('rb') as stream:
            magic = stream.read(8)
        if magic.startswith(b'CDF'):
            kwargs['engine'] = 'scipy'
        elif magic == b'\x89HDF\r\n\x1a\n':
            kwargs['engine'] = 'h5netcdf'
    return xr.open_dataset(path, **kwargs)

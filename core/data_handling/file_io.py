
import os
import xarray as xr
from core.utils.netcdf_io import open_dataset
import pandas as pd
from core.utils.logger_config import logger


def get_nc_variables(file_path):
    """
    Opens a single NetCDF file and returns its coordinate and data variable names.

    Args:
        file_path (str): Path to the NetCDF file.

    Returns:
        dict: {'coords': [...], 'data_vars': [...]} or None if error.
    """

    if not os.path.isfile(file_path):
        logger.error(f"Error: NetCDF file not found: {file_path}")
        return None

    try:
        with open_dataset(file_path) as ds:
            coords = list(ds.coords.keys())
            data_vars = list(ds.data_vars.keys())
            logger.debug(f"Variables in {os.path.basename(file_path)} - Coords: {coords}, Data Vars: {data_vars}")
            return {'coords': coords, 'data_vars': data_vars}
    except Exception as e:
        logger.error(f"Error opening or reading NetCDF file {file_path}: {e}")
        return None


def guess_nc_variables(coords: list, data_vars: list, ds: xr.Dataset = None) -> dict:
    """
    Tries to guess the main data, latitude, and longitude variables based on
    common names, standard names, and units if available.

    Args:
        coords (list): List of coordinate variable names.
        data_vars (list): List of data variable names.
        ds (xr.Dataset, optional): The dataset itself, to access attributes. Defaults to None.

    Returns:
        dict: {'data_var': guessed_data_var, 'lat_var': guessed_lat_var, 'lon_var': guessed_lon_var}
              Values can be None if no guess could be made.
    """
    guessed = {'data_var': None, 'lat_var': None, 'lon_var': None}
    all_vars = {**{str(c): 'coord' for c in coords}, **{str(d): 'data' for d in data_vars}}
    all_vars_lower = {k.lower(): k for k in all_vars}


    coord_candidates = {}
    if ds:
        for var_name_orig, var_type in all_vars.items():
            try:
                var = ds[var_name_orig]
                attrs = {k.lower(): v for k, v in var.attrs.items()}


                if 'standard_name' in attrs and 'latitude' in attrs['standard_name'].lower(): coord_candidates[var_name_orig] = coord_candidates.get(var_name_orig, 0) + 3
                if 'units' in attrs and 'degrees_north' in attrs['units'].lower(): coord_candidates[var_name_orig] = coord_candidates.get(var_name_orig, 0) + 2
                if 'axis' in attrs and attrs['axis'].lower() == 'y': coord_candidates[var_name_orig] = coord_candidates.get(var_name_orig, 0) + 1
                if var_name_orig.lower() in ['lat', 'latitude', 'y']: coord_candidates[var_name_orig] = coord_candidates.get(var_name_orig, 0) + 1


                if 'standard_name' in attrs and 'longitude' in attrs['standard_name'].lower(): coord_candidates[var_name_orig] = coord_candidates.get(var_name_orig, 0) + 3
                if 'units' in attrs and 'degrees_east' in attrs['units'].lower(): coord_candidates[var_name_orig] = coord_candidates.get(var_name_orig, 0) + 2
                if 'axis' in attrs and attrs['axis'].lower() == 'x': coord_candidates[var_name_orig] = coord_candidates.get(var_name_orig, 0) + 1
                if var_name_orig.lower() in ['lon', 'longitude', 'long', 'x']: coord_candidates[var_name_orig] = coord_candidates.get(var_name_orig, 0) + 1

            except Exception as e:
                 logger.debug(f"Could not access attributes for variable '{var_name_orig}' during guessing: {e}")
                 pass


    best_lat_score = 0
    best_lon_score = 0
    lat_names_lower = ['lat', 'latitude', 'y']
    lon_names_lower = ['lon', 'longitude', 'long', 'x']

    potential_lats = {}
    potential_lons = {}

    for var_name_orig, score in coord_candidates.items():
        var_lower = var_name_orig.lower()

        if any(name in var_lower for name in lat_names_lower) or \
           (ds and ds[var_name_orig].attrs.get('standard_name', '').lower() == 'latitude') or \
           (ds and ds[var_name_orig].attrs.get('axis', '').lower() == 'y'):
            potential_lats[var_name_orig] = score

        if any(name in var_lower for name in lon_names_lower) or \
           (ds and ds[var_name_orig].attrs.get('standard_name', '').lower() == 'longitude') or \
           (ds and ds[var_name_orig].attrs.get('axis', '').lower() == 'x'):
           potential_lons[var_name_orig] = score

    if potential_lats:
        guessed['lat_var'] = max(potential_lats, key=potential_lats.get)
    if potential_lons:

        lon_candidates_filtered = {k: v for k, v in potential_lons.items() if k != guessed['lat_var']}
        if lon_candidates_filtered:
            guessed['lon_var'] = max(lon_candidates_filtered, key=lon_candidates_filtered.get)


    if not guessed['lat_var']:
        for name in lat_names_lower:
            if name in all_vars_lower:
                guessed['lat_var'] = all_vars_lower[name]
                break
    if not guessed['lon_var']:
        for name in lon_names_lower:

            if name in all_vars_lower and all_vars_lower[name] != guessed['lat_var']:
                guessed['lon_var'] = all_vars_lower[name]
                break


    data_candidates = {}
    guessed_coord_dims = set()
    if ds and guessed['lat_var'] and guessed['lon_var']:
        try:
            guessed_coord_dims.add(ds[guessed['lat_var']].dims[0])
            guessed_coord_dims.add(ds[guessed['lon_var']].dims[0])
        except Exception: pass

    for var_name_orig in data_vars:
        score = 0
        var_name_lower = var_name_orig.lower()

        if var_name_orig == guessed['lat_var'] or var_name_orig == guessed['lon_var']:
             continue


        common_data_names = ['z', 'value', 'data', 'depth', 'elevation', 'band1', 'beta']
        if var_name_lower in common_data_names: score += 1


        if ds and guessed_coord_dims:
            try:
                 var_dims = set(ds[var_name_orig].dims)

                 if guessed_coord_dims.issubset(var_dims): score += 3

                 elif len(guessed_coord_dims.intersection(var_dims)) == 1: score += 1

                 if len(var_dims) > 1 : score +=1
            except Exception: pass

        data_candidates[var_name_orig] = score


    if data_candidates:
        guessed['data_var'] = max(data_candidates, key=data_candidates.get)
    elif data_vars:
        potential_data_vars = [dv for dv in data_vars if dv != guessed['lat_var'] and dv != guessed['lon_var']]
        if potential_data_vars:
            guessed['data_var'] = potential_data_vars[0]
        elif data_vars:
             guessed['data_var'] = data_vars[0]


    logger.debug(f"Guessed variables: Lat='{guessed['lat_var']}', Lon='{guessed['lon_var']}', Data='{guessed['data_var']}'")

    return guessed


import xarray as xr
from typing import Literal
import numpy as np
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import StandardScaler


from core.utils.logger_config import logger


InterpolationMethod = Literal["linear", "nearest", "cubic"]

def interpolate_grid(
    dataset: xr.Dataset,
    method: InterpolationMethod = "linear",
    variables: list[str] | None = None,
    land_mask: xr.DataArray | None = None
) -> xr.Dataset:
    """Fill missing grid values using the selected interpolation method.

    variables selects data variables; None selects all. An aligned land_mask
    excludes land cells. Returns a copy of the dataset with filled values."""
    logger.info(f"Interpolating dataset; method: '{method}'...")


    if variables is None:
        target_variables = list(dataset.data_vars)
    else:
        target_variables = [v for v in variables if v in dataset.data_vars]

    if not target_variables:
        logger.warning("No interpolation variables found; returning the input dataset.")
        return dataset

    logger.debug(f"Variables selected for interpolation: {target_variables}")


    possible_lat_names = ['latitude', 'lat', 'Latitude', 'Lat']
    possible_lon_names = ['longitude', 'lon', 'Longitude', 'Lon']

    lat_dim = None
    lon_dim = None


    for name in possible_lat_names:
        if name in dataset.dims or name in dataset.coords:
            lat_dim = name
            break


    for name in possible_lon_names:
        if name in dataset.dims or name in dataset.coords:
            lon_dim = name
            break

    if lat_dim is None or lon_dim is None:
        logger.error(f"Latitude or longitude dimension missing; detected dimensions: lat={lat_dim}, lon={lon_dim}")
        logger.debug(f"Dataset dimensions: {list(dataset.dims)}")
        logger.debug(f"Dataset coordinates: {list(dataset.coords)}")
        raise ValueError("Interpolation requires latitude and longitude dimensions.")

    interp_dims = [lat_dim, lon_dim]
    logger.debug(f"Interpolation dimensions: {interp_dims}")

    interpolated_ds = dataset.copy()

    for var_name in target_variables:
        try:


            var_data = interpolated_ds[var_name]


            if not np.issubdtype(var_data.dtype, np.number):
                logger.warning(f"Variable '{var_name}' is not numeric; skipping interpolation.")
                continue


            ocean_data = var_data.where(land_mask == 0) if land_mask is not None else var_data
            total_ocean_cells = ocean_data.notnull().sum().item()
            nan_ocean_cells = ocean_data.isnull().sum().item()

            if total_ocean_cells > 0:
                vacancy_rate = (nan_ocean_cells / total_ocean_cells) * 100
                logger.info(f"Variable '{var_name}' ocean missing-value rate: {vacancy_rate:.2f}% ({nan_ocean_cells}/{total_ocean_cells} cells)")
            else:
                logger.info(f"Variable '{var_name}' has no valid ocean data.")


            has_nan = nan_ocean_cells > 0

            if has_nan:
                logger.debug(f"Interpolating variable '{var_name}' over ocean cells...")
                try:

                    data_to_interpolate = var_data.where(land_mask == 0) if land_mask is not None else var_data


                    logger.debug(f"  Stage 1: Applying ffill/bfill for variable '{var_name}'...")
                    filled_data = data_to_interpolate.ffill(dim=lon_dim).bfill(dim=lon_dim)
                    filled_data = filled_data.ffill(dim=lat_dim).bfill(dim=lat_dim)

                    logger.debug(f"  Stage 2: Applying '{method}' interpolation for variable '{var_name}'...")
                    interpolated_data = filled_data.interpolate_na(
                        dim=interp_dims,
                        method=method,
                        fill_value="extrapolate"
                    )

                    if interpolated_data.isnull().any():
                        logger.warning(f"  '{var_name}' still has NaNs after interpolation. Applying final bfill/ffill.")
                        interpolated_data = interpolated_data.bfill(dim=lon_dim).ffill(dim=lon_dim)
                        interpolated_data = interpolated_data.bfill(dim=lat_dim).ffill(dim=lat_dim)


                    interpolated_ds[var_name] = interpolated_data.where(land_mask == 0) if land_mask is not None else interpolated_data
                    logger.debug(f"Variable '{var_name}' interpolation completed.")


                    final_ocean_data = interpolated_ds[var_name].where(land_mask == 0) if land_mask is not None else interpolated_ds[var_name]
                    final_nan_ocean_cells = final_ocean_data.isnull().sum().item()
                    final_total_ocean_cells = final_ocean_data.notnull().sum().item() + final_nan_ocean_cells

                    if final_total_ocean_cells > 0:
                        coverage_rate = (1 - final_nan_ocean_cells / final_total_ocean_cells) * 100
                        logger.info(f"Variable '{var_name}' ocean coverage after interpolation: {coverage_rate:.2f}%")

                except Exception as e:
                    logger.error(f"Variable '{var_name}' interpolation failed: {e}; applying forward/backward filling.")
                    try:
                        interpolated_ds[var_name] = var_data.ffill(dim=lon_dim).bfill(dim=lon_dim).ffill(dim=lat_dim).bfill(dim=lat_dim)
                    except Exception as fallback_e:
                        logger.error(f"Fallback filling failed: {fallback_e}; skipping this variable.")
                        continue
            else:
                logger.debug(f"Variable '{var_name}' has no missing ocean values; skipping interpolation.")
        except Exception as e:
            logger.error(f"Processing variable '{var_name}' raised an unexpected error: {e}; skipping this variable.")
            continue

    logger.info("Dataset interpolation completed.")
    return interpolated_ds
def iterative_imputer_interpolation(
    dataset: xr.Dataset,
    variables: list[str] | None = None,
    land_mask: xr.DataArray | None = None
) -> xr.Dataset:
    """Fill missing grid values with multivariate iterative imputation.

    variables selects data variables; None selects all. An aligned land_mask
    excludes land cells. Returns the imputed dataset."""
    logger.info("Starting multivariate iterative imputation...")

    if variables is None:
        target_variables = list(dataset.data_vars)
    else:
        target_variables = [v for v in variables if v in dataset.data_vars]

    if not target_variables:
        logger.warning("No interpolation variables found; returning the input dataset.")
        return dataset

    logger.debug(f"Variables selected for iterative imputation: {target_variables}")


    data_stack = np.stack([dataset[v].values for v in target_variables], axis=0)

    original_shape = data_stack.shape
    n_features, height, width = original_shape


    data_reshaped = data_stack.reshape(n_features, -1).T


    land_indices = None
    if land_mask is not None:
        land_mask_flat = land_mask.values.flatten()
        land_indices = np.where(land_mask_flat)[0]

        data_reshaped[land_indices, :] = np.nan


    ocean_mask_flat = ~land_mask_flat if land_mask is not None else np.ones(data_reshaped.shape[0], dtype=bool)
    if not np.isnan(data_reshaped[ocean_mask_flat, :]).any():
        logger.info("Ocean cells contain no missing values; iterative imputation is complete.")
        return dataset


    imputer = IterativeImputer(
        max_iter=10,
        random_state=0,
        add_indicator=False
    )

    logger.debug("Running iterative imputation...")
    imputed_data_reshaped = imputer.fit_transform(data_reshaped)
    logger.debug("Iterative imputation completed.")


    if land_indices is not None:
        imputed_data_reshaped[land_indices, :] = np.nan


    imputed_data_stack = imputed_data_reshaped.T.reshape(original_shape)


    imputed_ds = dataset.copy()
    for i, var_name in enumerate(target_variables):
        imputed_ds[var_name] = (dataset[var_name].dims, imputed_data_stack[i])
        imputed_ds[var_name].attrs = dataset[var_name].attrs
        imputed_ds[var_name].attrs['interpolation_method'] = 'iterative_imputer_ocean_only'

    logger.info("Multivariate iterative imputation completed.")


    if land_mask is not None:

        total_ocean_cells = (land_mask == 0).sum().item()
        if total_ocean_cells > 0:

            total_nan_in_ocean = imputed_ds[target_variables].where(land_mask == 0).isnull().sum().to_array().sum().item()

            total_data_points_in_ocean = len(target_variables) * total_ocean_cells

            overall_coverage = (1 - total_nan_in_ocean / total_data_points_in_ocean) * 100
            logger.info(f"Overall ocean coverage after iterative imputation: {overall_coverage:.2f}%")
    else:

        total_nan = imputed_ds[target_variables].isnull().sum().to_array().sum().item()
        total_points = imputed_ds[target_variables].count().to_array().sum().item() + total_nan
        if total_points > 0:
            overall_coverage = (1 - total_nan / total_points) * 100
            logger.info(f"Overall data coverage after iterative imputation: {overall_coverage:.2f}%")

    return imputed_ds
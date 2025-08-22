import numpy as np
from scipy import ndimage

class RasterUtils:
    """Utility class for raster operations, specifically for applying height to depth transformations on raster files."""

    @staticmethod
    def is_empty(dem_data, nodata):
        """
        Function to check if the cleaned DEM data is empty.
        This function removes NaN values, -9999, 0, and infinite values from the DEM data."""
        # Remove NaN values, -9999, 0, and infinite values
        cleaned_data = dem_data[~(np.isnan(dem_data)) & ~(dem_data == nodata) & ~(dem_data == 0) & ~(np.isinf(dem_data))]
        return cleaned_data.size == 0

    @staticmethod
    def is_low_variance(dem_data, nodata, n=2):
        """
        Function to check if the number of unique values in the cleaned data array are less than or equal to 2.

        Returns:
        bool: True if the unique values array length is <= 2, False otherwise.
        """
        cleaned_data = dem_data[~(np.isnan(dem_data)) & ~(dem_data == nodata) & ~(dem_data == 0) & ~(np.isinf(dem_data))]
        return len(np.unique(cleaned_data)) <= n

    @staticmethod
    def replace_nodata_with_nan(dem_data, nodata=-9999):
        """
        Replace no-data values in the DEM data with NaN.
        """
        # Replace no-data values with NaN
        dem_data = np.where(dem_data == nodata, np.nan, dem_data)
        return dem_data
    
        
    @staticmethod
    def replace_nan_with_num(dem_data, fill_val=-9999):
        """
        Replace NaN values in the DEM data with a specified fill value.
        """
        dem_data = np.where(np.isnan(dem_data), fill_val, dem_data)
        return dem_data


    @staticmethod
    def fill_nodata_with_mean_window(dem_data, nodata=-9999, max_iterations=10, initial_window_size=3):
        """
        Fill no-data values in the DEM data with the mean of the surrounding values.

        Returns:
        numpy.ndarray: DEM data with no-data gaps filled.
        """
        # Create a mask for no-data values
        mask = np.where(dem_data == nodata, True, False)
        window_size = initial_window_size
        for iteration in range(max_iterations):
            if not np.any(mask):
                break  # Exit if no gaps remain

            # Fill no-data values with the mean of surrounding values
            filled_data = ndimage.generic_filter(
                dem_data, np.nanmean, size=window_size, mode='nearest', output=np.float32)
            dem_data[mask] = filled_data[mask]
            # Update the mask for remaining no-data values
            mask = np.where(dem_data == nodata, True, False)
            # Optionally increase the window size for subsequent iterations
            window_size += 2
        return dem_data
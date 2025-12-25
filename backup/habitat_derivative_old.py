import rasterio
import numpy as np
from skimage.measure import label, regionprops
from skimage.filters import threshold_otsu
from rasterio.transform import from_origin
from scipy.ndimage import gaussian_filter
from scipy import ndimage

def calculate_slope_aspect(dem):
    """ Calculate slope and aspect from a DEM. """
    x, y = np.gradient(dem)  # Gradient in the x and y direction
    slope = np.arctan(np.sqrt(x**2 + y**2)) * (180 / np.pi)  # Calculate slope in degrees
    aspect = np.arctan2(-y, x) * (180 / np.pi)  # Calculate aspect in degrees
    aspect = (aspect + 360) % 360  # Normalize to [0, 360)
    
    return slope, aspect

def calculate_roughness(dem, window_size):
    """ Calculate terrain roughness as the standard deviation. """
    return ndimage.generic_filter(dem, np.std, size=window_size)

def calculate_tpi(dem):
    """ Calculate Topographic Position Index (TPI). """
    mean_neighbors = ndimage.uniform_filter(dem, size=3)  # Mean elevation of the neighbors
    tpi = dem - mean_neighbors  # Calculate TPI
    return tpi

def calculate_tri(dem):
    """ Calculate Terrain Ruggedness Index (TRI). """
    diff_y, diff_x = np.gradient(dem)
    tri = np.sqrt(diff_x**2 + diff_y**2)  # Ruggedness calculated from the gradient
    return tri

def process_dem(input_dem, output_slope, output_aspect, output_roughness, output_tpi, output_tri):
    """ Read a DEM and compute slope, aspect, roughness, TPI, and TRI. Output each to TIFF files. """
    
    with rasterio.open(input_dem) as src:
        dem_data = src.read(1)  # Read the first band
        transform = src.transform  # Get the affine transform
        metadata = src.meta  # Copy metadata

    # Calculate slope and aspect
    slope, aspect = calculate_slope_aspect(dem_data)

    # Calculate roughness (use 3x3 window)
    roughness = calculate_roughness(dem_data, window_size=(6, 6))

    # Calculate TPI
    tpi = calculate_tpi(dem_data)

    # Calculate TRI
    tri = calculate_tri(dem_data)

    # Output each property to a TIFF file
    for data, output_file in zip([
                                # slope, 
                                # aspect, 
                                roughness, 
                                # tpi, 
                                # tri
                                  ],
                                  [
                                    # output_slope, 
                                    # output_aspect, 
                                    output_roughness, 
                                    # output_tpi, 
                                    # output_tri
                                    ]):
        metadata.update({'dtype': 'float32', 'count': 1})
        with rasterio.open(output_file, 'w', **metadata) as dst:
            dst.write(data.astype('float32'), 1)

if __name__ == "__main__":
    input_dem = 'path/to/your/input_dem.tif'  # Input DEM file
    output_slope = 'path/to/output_slope.tif'  # Output slope file
    output_aspect = 'path/to/output_aspect.tif'  # Output aspect file
    output_roughness = 'path/to/output_roughness.tif'  # Output roughness file
    output_tpi = 'path/to/output_tpi.tif'  # Output TPI file
    output_tri = 'path/to/output_tri.tif'  # Output TRI file

    process_dem(input_dem, output_slope, output_aspect, output_roughness, output_tpi, output_tri)

    print("Processed DEM and saved outputs.")
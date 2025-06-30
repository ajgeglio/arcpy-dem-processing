from email.mime import message
import arcpy
import rasterio
import os, glob
import numpy as np
from scipy import ndimage
from skimage.feature import local_binary_pattern
from arcpy.sa import *
from osgeo import gdal
from inpainter import Inpainter
from utils import WorkspaceCleaner, Utils, demUtils
time_convert = Utils().time_convert
from inpainter import *
import time
import warnings
from skimage.util import view_as_windows
from shannon import calculate_window_entropy, flow_direction_window, fast_entropy_numba

# Suppress joblib memmapping folder deletion warnings
warnings.filterwarnings(
    "ignore",
    message="Failed to delete temporary folder: ",
    module="joblib._memmapping_reducer"
)

# Suppress GDAL FutureWarning about UseExceptions
warnings.filterwarnings(
    "ignore",
    message="Neither gdal.UseExceptions\\(\\) nor gdal.DontUseExceptions\\(\\) has been explicitly called. In GDAL 4.0, exceptions will be enabled by default.",
    category=FutureWarning,
    module="osgeo.gdal"
)

class HabitatDerivatives:
    def __init__(
        self,
        input_dem=None,
        use_gdal=True,
        use_rasterio=False,
        chunk_size=None,
        output_folder=".",
        shannon_window=21,
        products=None,
        fill_iterations=1,
        fill_method=None,
        generate_boundary=True
    ):
        """
        Initialize the HabitatDerivatives class.

        Parameters:
        input_dem (str): Path to the input DEM file.
        use_gdal (bool): Flag to indicate whether to use GDAL for processing. Default is True.
        use_rasterio (bool): Flag to indicate whether to use Rasterio for processing. Default is False.
        chunk_size (int or None): Size of chunks for processing large DEMs. Default is None (no chunking).
        output_folder (str): Output folder for results.
        shannon_window (int): Window size for shannon index.
        products (list): List of products to generate.
        """
        self.input_dem = input_dem
        # Store original spatial reference, cell size, and snap raster for assertion
        dem_desc = arcpy.Describe(self.input_dem)
        self.original_spatial_ref = dem_desc.spatialReference
        self.original_cell_size = arcpy.management.GetRasterProperties(self.input_dem, "CELLSIZEX").getOutput(0)
        self.original_snap_raster = self.input_dem

        # Set arcpy.env variables to match input DEM
        arcpy.env.outputCoordinateSystem = self.original_spatial_ref
        arcpy.env.snapRaster = self.input_dem
        arcpy.env.cellSize = self.input_dem

        self.use_gdal = use_gdal
        self.use_rasterio = use_rasterio
        self.chunk_size = chunk_size
        self.output_folder = output_folder
        self.shannon_window = shannon_window
        self.products = products if products is not None else []
        self.fill_iterations = fill_iterations  # Number of iterations for filling gaps
        self.fill_method = fill_method # IDW or FocalStatistics or None
        self.generate_boundary = generate_boundary # for binary mask and boundary shapefile
        self.message_length = 0

        # Get input DEM file and name and create folders
        self.dem_name = Utils().sanitize_path_to_name(self.input_dem)
        print("DEM NAME:", self.dem_name)
        self.out_dem_folder = os.path.join(self.output_folder, self.dem_name)
        if not os.path.exists(self.out_dem_folder):
            os.makedirs(self.out_dem_folder)
        print("OUTPUT FOLDER:", self.out_dem_folder)

        # generate the cleaned data boundary and binary mask tif
        self.inpainter = Inpainter(input_dem, save_path=self.output_folder)

        if self.generate_boundary:
            self.binary_mask, dissolved_polygon = self.inpainter.get_data_boundary(min_area=50)

        if self.fill_method is not None:
            # Generate the fill raster
            filled_raster_path = self.inpainter.fill_internal_gaps_arcpy(
                input_mask=dissolved_polygon,
                method=self.fill_method,
                iterations=self.fill_iterations
            )
            # Clean up the workspace
            WorkspaceCleaner(self.inpainter).clean_up()
            # Set the filled raster as the input DEM for further processing
            self.input_dem = filled_raster_path

        else:
            WorkspaceCleaner(self.inpainter).clean_up()

    def calculate_lbp(self, dem_data, n_points, radius, method='default', nodata=None):
        """
        Generate a Local Binary Pattern (LBP) representation from a DEM using a sliding window approach.

        Parameters:
        dem_data(numpy.ndarray): Input DEM as a 2D NumPy array.
        radius (int): Radius of LBP neighborhood.
        n_points (int): Number of circularly symmetric neighbor points.
        method (str): Method to compute LBP ('default', 'uniform', etc.).

        Returns:
        numpy.ndarray: LBP-transformed DEM with the same shape.
        """
        # Create a mask for valid data (not nodata, not NaN)
        if nodata is not None:
            valid_mask = (~np.isnan(dem_data)) & (dem_data != nodata)
        else:
            valid_mask = ~np.isnan(dem_data)

        # Normalize the DEM data to the range 0-255 for valid data only
        valid_data = dem_data[valid_mask]
        if valid_data.size > 0:
            min_val = valid_data.min()
            max_val = valid_data.max()
            if max_val > min_val:
                normalized_data = np.zeros_like(dem_data, dtype=np.uint8)
                normalized_data[valid_mask] = ((dem_data[valid_mask] - min_val) / (max_val - min_val) * 255).astype(np.uint8)
            else:
                normalized_data = np.zeros_like(dem_data, dtype=np.uint8)
        else:
            normalized_data = np.zeros_like(dem_data, dtype=np.uint8)

        # Compute LBP for the entire DEM
        lbp = local_binary_pattern(normalized_data, P=n_points, R=radius, method=method)

        # Set LBP to 0 (or np.nan) where data is invalid
        lbp[~valid_mask] = 0  # or use np.nan if preferred and dtype allows

        return lbp

    def generate_hillshade_gdal(self, hillshade_output):
        """
        Calculate hillshade using GDAL.DEMProcessing with multiple options.
        
        :param dem_file: Path to the input DEM file.
        :param hillshade_output: Path to save the hillshade output file.
        """
        # Set GDAL DEMProcessing options
        options = gdal.DEMProcessingOptions(
            computeEdges=True,          # Compute edges for smoother output
            azimuth=315,                # Direction of light source (in degrees)
            altitude=45,                # Elevation of the light source (in degrees)
            scale=1.0,                  # Scale factor for vertical exaggeration
            zFactor=1.0                 # Vertical exaggeration
        )
        
        # Perform hillshade processing
        gdal.DEMProcessing(
            destName=hillshade_output,  # Output file path
            srcDS=self.input_dem,             # Input DEM file
            processing="hillshade",     # Specify hillshade calculation
            options=options             # Pass multiple options
        )
        print(f"Hillshade file saved at: {hillshade_output}")

    def calculate_hillshade(self, hillshade_output):
        """
        Calculate hillshade for a DEM file and save the output.

        Parameters:
        dem_file (str): Path to the input DEM file.
        hillshade_output (str): Path to save the hillshade output file.
        """
        if self.use_gdal:
            self.generate_hillshade_gdal(hillshade_output)
        else:
            raise NotImplementedError("Custom hillshade calculation is not implemented for non-GDAL methods.")

    def calculate_slope_gdal(self, dem_data, transform):
        """ Calculate slope from a DEM using GDAL."""
        # Create a temporary in-memory file for the DEM
        mem_drv = gdal.GetDriverByName('MEM')
        src_ds = mem_drv.Create('', dem_data.shape[1], dem_data.shape[0], 1, gdal.GDT_Float32)
        src_ds.SetGeoTransform(transform.to_gdal())  # Set geotransformation
        src_ds.GetRasterBand(1).WriteArray(dem_data)
        # Calculate slope
        slope_ds = gdal.DEMProcessing('', src_ds, 'slope', format='MEM', options=['-compute_edges'])# options = ['-zero_for_flat'] option can be added to calculate edges
        slope = slope_ds.GetRasterBand(1).ReadAsArray()
        return slope

    def generate_slope_gdal(self, output_slope):
        """
        Calculate slope from a DEM using GDAL and save to output file.
        """
        gdal.DEMProcessing(
            destName=output_slope,
            srcDS=self.input_dem,
            processing="slope",
            options=gdal.DEMProcessingOptions(computeEdges=True)
        )
        print(f"Slope file saved at: {output_slope}")

    def calculate_slope(self, dem_data, transform, output_slope):
        if self.use_gdal:
            if self.use_rasterio:
                return self.calculate_slope_gdal(dem_data, transform)
            else:
                self.generate_slope_gdal(output_slope)
        else:
            return self.calculate_slope_aspect_skimage(dem_data)[0]
        
    def calculate_aspect_gdal(self, dem_data, transform):
        """ Calculate slope and aspect from a DEM using GDAL. 
        This array contains the aspect values calculated from the Digital Elevation Model (DEM) using the GDAL library.
        Aspect values represent the compass direction that the slope faces. The values are typically in degrees, where:
        0° represents north,         180° represents south,        270° represents west."""
        # Create a temporary in-memory file for the DEM
        mem_drv = gdal.GetDriverByName('MEM')
        src_ds = mem_drv.Create('', dem_data.shape[1], dem_data.shape[0], 1, gdal.GDT_Float32)
        src_ds.SetGeoTransform(transform.to_gdal())  # Set geotransformation
        src_ds.GetRasterBand(1).WriteArray(dem_data)
        # Calculate aspect
        aspect_ds = gdal.DEMProcessing('', src_ds, 'aspect', format='MEM', options=['-compute_edges'])
        aspect = aspect_ds.GetRasterBand(1).ReadAsArray()
        return aspect
    
    def generate_aspect_gdal(self, output_aspect):
        gdal.DEMProcessing(
            destName=output_aspect,
            srcDS=self.input_dem,
            processing="aspect",
            options=gdal.DEMProcessingOptions(computeEdges=True)
        )
        print(f"Aspect file saved at: {output_aspect}")

    def calculate_aspect(self, dem_data, transform, output_aspect):
        if self.use_gdal:
            if self.use_rasterio:
                return self.calculate_aspect_gdal(dem_data, transform)
            else:
                self.generate_aspect_gdal(output_aspect)
        else:
            return self.calculate_slope_aspect_skimage(dem_data)[1]

    def calculate_roughness_gdal(self, dem_data, transform):
        """ Calculate terrain roughness using GDAL. """
        # Create a temporary in-memory file for the DEM
        mem_drv = gdal.GetDriverByName('MEM')
        src_ds = mem_drv.Create('', dem_data.shape[1], dem_data.shape[0], 1, gdal.GDT_Float32)
        src_ds.SetGeoTransform(transform.to_gdal())  # Set geotransformation
        # src_ds.GetRasterBand(1).SetNoDataValue(-9999)  # Set no-data value
        src_ds.GetRasterBand(1).WriteArray(dem_data)
        
        # Calculate roughness
        roughness_ds = gdal.DEMProcessing('', src_ds, 'Roughness', format='MEM', options = ['-compute_edges'])
        roughness = roughness_ds.GetRasterBand(1).ReadAsArray()
        
        return roughness
    
    def generate_roughness_gdal(self, output_roughness):
        """ Calculate terrain roughness using GDAL. """
        # Calculate roughness and save to output file
        gdal.DEMProcessing(
            destName=output_roughness,
            srcDS=self.input_dem,
            processing="roughness",
            options=gdal.DEMProcessingOptions(computeEdges=True)
        )
        print(f"Roughness file saved at: {output_roughness}")

    def calculate_roughness_skimage(self, dem_data):
        """ Calculate terrain roughness as the standard deviation. """
        return ndimage.generic_filter(dem_data, np.std, size=16)

    def calculate_roughness(self, dem_data, transform, output_roughness):
        if self.use_gdal:
            if self.use_rasterio:
                return self.calculate_roughness_gdal(dem_data, transform)
            else:
                self.generate_roughness_gdal(output_roughness)
        else:
            return self.calculate_roughness_skimage(dem_data)

    def generate_tpi_gdal(self, output_tpi):
        gdal.DEMProcessing(
            destName=output_tpi,
            srcDS=self.input_dem,
            processing="tpi",
            options=gdal.DEMProcessingOptions(computeEdges=True)
        )
        print(f"TPI file saved at: {output_tpi}")

    def calculate_tpi(self, dem_data, transform, output_tpi):
        if self.use_gdal:
            if self.use_rasterio:
                return self.calculate_tpi_gdal(dem_data, transform)
            else:
                self.generate_tpi_gdal(output_tpi)
        else:
            return self.calculate_tpi_skimage(dem_data)

    def generate_tri_gdal(self, output_tri):
        """
        Generate Terrain Ruggedness Index (TRI) from an input DEM using GDAL.

        :param output_tri: Path to save the output TRI file.
        """
        if output_tri:
            gdal.DEMProcessing(
                destName=output_tri,
                srcDS=self.input_dem, 
                processing='TRI', 
                options=gdal.DEMProcessingOptions(computeEdges=True)
            )
            print(f"TRI file saved at: {output_tri}")

    def calculate_tri(self, dem_data, transform, output_tri):
        if self.use_gdal:
            if self.use_rasterio:
                return self.calculate_tri_gdal(dem_data, transform)
            else:
                # If output_tri is None, return the array, else save to file
                return self.generate_tri_gdal(output_tri)
        else:
            return self.calculate_tri_skimage(dem_data)
    
    def calculate_flow_direction(self, dem_data):
        """
        Calculate the Flow Direction similar to ArcGIS (Spatial Analyst).
        https://pro.arcgis.com/en/pro-app/latest/tool-reference/spatial-analyst/how-flow-direction-works.htm
        """

        # Use ndimage.generic_filter to apply flow_direction_window over the DEM
        flow_direction = ndimage.generic_filter(
            dem_data,
            function=flow_direction_window,
            size=3,
            mode='nearest'
        ).astype(np.int32)

        return flow_direction
    
    def calculate_shannon_index_2d(self, dem_data):
        """
        Fast calculation of the Shannon diversity index for each window in a 2D grid.
        Uses skimage view_as_windows and numba for speed.
        """
        window_size = self.shannon_window
        flow_direction = self.calculate_flow_direction(dem_data)
        # Pad to handle borders
        pad = window_size // 2
        padded = np.pad(flow_direction, pad, mode='edge')
        # Create sliding windows
        windows = view_as_windows(padded, (window_size, window_size))
        # Use numba-accelerated entropy calculation
        shannon_indices = fast_entropy_numba(windows)
        return shannon_indices

    def return_dem_data(self, dem_data):
        return dem_data

    def process_dem(self): 

        if not self.products:
            self.products = ["slope", "aspect", "roughness", "tpi", "tri", "hillshade"]
        print()
        print("Generating products", self.products)
        input_dem = self.input_dem
        # define window for shannon index
        if isinstance(self.shannon_window, tuple):
            window = self.shannon_window[0]
        else:
            window = self.shannon_window
        # creating output file paths based on the product list
        output_files = {}
        for key in self.products:
            if key == "shannon_index":
                output_files[key] = os.path.join(
                    self.out_dem_folder,
                    f"{self.dem_name}_{key}_{self.shannon_window}.tif"
                )
            else:
                output_files[key] = os.path.join(
                    self.out_dem_folder,
                    f"{self.dem_name}_{key}.tif"
                )
        output_slope=output_files.get("slope")
        output_aspect=output_files.get("aspect")
        output_roughness=output_files.get("roughness")
        output_tpi=output_files.get("tpi")
        output_tri=output_files.get("tri")
        output_hillshade=output_files.get("hillshade")
        output_shannon_index=output_files.get("shannon_index")
        output_lbp_3_1=output_files.get("lbp-3-1")
        output_lbp_15_2=output_files.get("lbp-15-2")
        output_lbp_21_3=output_files.get("lbp-21-3")
        output_dem=output_files.get("dem")

        """ Read a DEM and compute slope, aspect, roughness, TPI, and TRI. Output each to TIFF files based on user input. """
        def generate_products():
            if output_slope:
                yield self.calculate_slope(dem_data, transform, output_slope), output_slope   # Slope
            if output_aspect:
                yield self.calculate_aspect(dem_data, transform, output_aspect), output_aspect  # Aspect
            if output_roughness:
                yield self.calculate_roughness(dem_data, transform, output_roughness), output_roughness  # Roughness
            if output_tpi:
                yield self.calculate_tpi(dem_data, transform, output_tpi), output_tpi  # TPI
            if output_tri:
                yield self.calculate_tri(dem_data, transform, output_tri), output_tri  # TRI
            if output_hillshade:
                yield self.calculate_hillshade(output_hillshade), output_hillshade # Hillshade
            if output_shannon_index:
                yield self.calculate_shannon_index_2d(dem_data), output_shannon_index  # Shannon Index
            if output_lbp_3_1:
                yield self.calculate_lbp(dem_data, 3, 1), output_lbp_3_1  # LBP
            if output_lbp_15_2:
                yield self.calculate_lbp(dem_data, 15, 2), output_lbp_15_2  # LBP
            if output_lbp_21_3:
                yield self.calculate_lbp(dem_data, 21, 3), output_lbp_21_3  # LBP
            if output_dem:
                yield self.return_dem_data(dem_data), output_dem  # used to chunk and merge data to get the same shape output as other products
        
        if not self.chunk_size:
            with rasterio.open(input_dem) as src:
                dem_data = src.read(1)  # Read the DEM data
                transform = src.transform  # Get the affine transform
                crs = src.crs  # Get the coordinate reference system
                metadata = src.meta  # Get metadata
                nodata = metadata['nodata']  # Get no-data value
                dem_data = self.replace_nodata_with_nan(dem_data, nodata)  # replace any no-data values with NaN
                # Check if the DEM data is empty after filling no-data values
                if self.is_empty(dem_data, nodata):
                    message = "The DEM data is empty after filling no-data values."
                    self.message_length = Utils.print_progress(message, self.message_length)
                    return None

            # Set arcpy.env variables before writing outputs
            arcpy.env.outputCoordinateSystem = self.original_spatial_ref
            arcpy.env.snapRaster = self.original_snap_raster
            arcpy.env.cellSize = self.original_cell_size

            # Process and write each product one at a time
            for data, output_file in generate_products():
                if data is not None:
                    with rasterio.open(
                        output_file,
                        'w',
                        driver='GTiff',
                        height=dem_data.shape[0],
                        width=dem_data.shape[1],
                        count=1,
                        dtype=data.dtype,
                        crs=crs,
                        transform=transform
                    ) as dst:
                        dst.write(data, 1)
                        dst.update_tags(**src.tags())
                # self.convert_tiff_to_gdal_raster(output_file, compress=False)

                # Assert output spatial reference matches input
                out_desc = arcpy.Describe(output_file)
                assert out_desc.spatialReference.name == self.original_spatial_ref.name, \
                    f"Spatial reference changed! Expected: {self.original_spatial_ref.name}, Got: {out_desc.spatialReference.name}"
                # Trim the shannon index to avoid artifacts
                # Only trim the shannon index product after conversion
                if "shannon" in str(output_file).lower():
                    shannon_dem = os.path.join(self.output_folder, self.dem_name, self.dem_name + "_shannon_index.tif")
                    self.inpainter.trim_raster(shannon_dem, self.binary_mask, overwrite=False)

        else:
            with rasterio.open(input_dem) as src:
                transform = src.transform
                crs = src.crs
                metadata = src.meta  # Get metadata
                nodata = metadata['nodata']  # Get no-data value
                tile_size = src.height // self.chunk_size

                # Calculate number of tiles in each direction and total tiles
                n_tiles_y = (src.height + tile_size - 1) // tile_size
                n_tiles_x = (src.width + tile_size - 1) // tile_size
                n_tiles = n_tiles_y * n_tiles_x
                tile_counter = 0

                for i in range(0, src.height, tile_size):
                    for j in range(0, src.width, tile_size):
                        tile_counter += 1
                        # Read a chunk of the DEM, deals with edge cases when the remaining pixels are smaller than the tile size
                        window = rasterio.windows.Window(j, i, min(tile_size, src.width - j), min(tile_size, src.height - i))
                        dem_data = src.read(1, window=window)
                        # dem_data = self.replace_nodata_with_nan(dem_data, nodata)  # Fill no-data values
                        # Check if the DEM data is low variance; skip if so
                        if demUtils.is_low_variance(dem_data, nodata, n=2):
                            message = f"Skipping tile {tile_counter}/{n_tiles} (low variance)"
                            self.message_length = Utils.print_progress(message, self.message_length)
                            continue
                        message = f"Processing tile {tile_counter}/{n_tiles}"
                        self.message_length = Utils.print_progress(message, self.message_length)
                        chunk_transform = rasterio.windows.transform(window, transform)
                        
                        # Process and write each product for the chunk
                        for data, output_file in generate_products():
                            # Ensure output folder exists for the product
                            output_dir = os.path.dirname(output_file)
                            product_name = os.path.basename(output_file).split(".")[0]
                            product_folder = os.path.join(output_dir, product_name)
                            os.makedirs(product_folder, exist_ok=True)

                            # Define the chunk output file path
                            chunk_output_file = os.path.join(product_folder, f"{product_name}_chunk_{i}_{j}.tif")

                            # Skip processing if the chunk already exists
                            if os.path.exists(chunk_output_file):
                                message = f"Skipping existing chunk at ({i}, {j}) for {output_file}."
                                self.message_length = Utils.print_progress(message, self.message_length)
                                continue

                            # Save the chunk using Rasterio
                            with rasterio.open(
                                chunk_output_file,
                                'w',
                                driver='GTiff',
                                height=dem_data.shape[0],
                                width=dem_data.shape[1],
                                count=1,
                                dtype=data.dtype,
                                crs=crs,
                                transform=chunk_transform
                            ) as dst:
                                dst.write(data, 1)
                                dst.update_tags(**src.tags())
                            # self.convert_tiff_to_gdal_raster(chunk_output_file, compress=False)

                            # Assert output spatial reference matches input
                            out_desc = arcpy.Describe(chunk_output_file)
                            assert out_desc.spatialReference.name == self.original_spatial_ref.name, \
                                f"Spatial reference changed! Expected: {self.original_spatial_ref.name}, Got: {out_desc.spatialReference.name}"

                # Merge the tiles and clean up
                for _, output_file in generate_products():
                    dem_chunks_path = os.path.join(os.path.dirname(output_file), os.path.basename(output_file).split(".")[0])
                    # merged_dem = demUtils.merge_dem(dem_chunks_path, remove=True)
                    merged_dem = demUtils.merge_dem_arcpy(dem_chunks_path, remove=True)
                    if "shannon" in str(merged_dem).lower() or "lbp" in str(merged_dem).lower():
                        self.inpainter.trim_raster(merged_dem, self.binary_mask, overwrite=True)
                    demUtils.compress_tiff_with_arcpy(merged_dem, format="TIFF", overwrite=True)

# example usage
if __name__ == "__main__":
    start_time = time.time()

    out_folder = "..\\habitat_derivatives"
    input_dem = "dem\\dem.tif"  # Path to the input DEM file
    products = ["shannon_index"] # Specify the products to generate, e.g., ["slope", "aspect", "roughness", "tpi", "tri", "hillshade", "shannon_index", "lbp-3-1"]
    # Create an instance of the HabitatDerivatives class with the specified parameters
    habitat_derivatives = HabitatDerivatives(
                                            input_dem=input_dem, 
                                            output_folder=out_folder,
                                            products=products,
                                            shannon_window=21,
                                            fill_iterations=1,
                                            fill_method=None,  # "IDW" or "FocalStatistics" or None
                                            chunk_size=32,  # Set to None for no chunking, or specify a chunk size
                                            )
    habitat_derivatives.process_dem()

    # Print time lapsed
    time_convert(time.time() - start_time)
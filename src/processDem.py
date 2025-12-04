import rasterio
import os
import arcpy
from arcpy.sa import *
from utils import Utils
from arcpyUtils import ArcpyUtils
from rasterUtils import RasterUtils
from gdalUtils import GdalUtils
from metafunctions import MetaFunctions
from inpainter import *
import time
import warnings
from rasterio.windows import Window
from derivatives import HabitatDerivatives

# function converts time to human readable format
time_convert = Utils().time_convert

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

class ProcessDem:
    """ Class to process DEM and generate habitat derivatives. """
    def __init__(
        self,
        input_dem=None,
        input_bs=None,
        binary_mask=None,
        use_gdal=True,
        divisions=None,
        output_folder=None,
        shannon_window=None,  # changed default to None
        products=None,
        fill_method=None,
        fill_iterations=1,
        generate_boundary=True
    ):
        """
        Initialize the HabitatDerivatives class.

        Parameters:
        input_dem (str): Path to the input DEM file.
        use_gdal (bool): Flag to indicate whether to use GDAL for processing. Default is True.
        use_rasterio (bool): Flag to indicate whether to use Rasterio for processing. Default is False.
        divisions (int or None): Size of chunks for processing large DEMs. Default is None (no chunking).
        output_folder (str): Output folder for results.
        shannon_window (int): Window size for shannon index.
        products (list): List of products to generate.
        """
        # Store original spatial reference, cell size, and snap raster for assertion
        self.input_dem = input_dem if input_dem else None
        self.binary_mask = binary_mask if binary_mask else None
        # Ensure raster statistics exist before transformation
        try:
            _ = arcpy.management.GetRasterProperties(self.input_dem, "MINIMUM").getOutput(0)
        except Exception:
            arcpy.management.CalculateStatistics(self.input_dem)

        self.input_bs = input_bs if input_bs else None

        if self.input_dem and self.input_bs:
            assert arcpy.Describe(self.input_dem).spatialReference.name == arcpy.Describe(self.input_bs).spatialReference.name, \
                "Spatial Ref not matching! You can use the function RasterUtils.transform_spatial_reference_arcpy(base_raster, transform_raster)"
            
        # converts to depth if it is in elevation
        self.input_dem = ArcpyUtils.apply_height_to_depth_transformation(self.input_dem, water_elevation=183.6)

        self.dem_name = Utils.sanitize_path_to_name(self.input_dem)
        dem_desc = arcpy.Describe(self.input_dem)
        self.original_spatial_ref = dem_desc.spatialReference
        self.original_cell_size = arcpy.management.GetRasterProperties(self.input_dem, "CELLSIZEX").getOutput(0)
        self.original_snap_raster = self.input_dem

        # Set arcpy.env variables to match input DEM
        arcpy.env.outputCoordinateSystem = self.original_spatial_ref
        arcpy.env.snapRaster = self.original_snap_raster
        arcpy.env.extent = dem_desc.extent
        arcpy.env.cellSize = self.original_cell_size
        print(f"ArcPy environment synchronized to: {self.input_dem}")
        
        self.use_gdal = use_gdal
        self.divisions = divisions
        self.output_folder = output_folder
        if self.output_folder is None:
            self.output_folder = os.path.dirname(self.input_dem)
        
        # Get input DEM file and name and create folders
        print("DEM NAME:", self.dem_name)
        self.habitat_derivatives_folder = os.path.join(self.output_folder, "habitat_derivatives")
        if not os.path.exists(self.habitat_derivatives_folder):
            os.makedirs(self.habitat_derivatives_folder)
        print("OUTPUT FOLDER:", self.habitat_derivatives_folder)
        arcpy.env.workspace = self.habitat_derivatives_folder
        
        # Set default shannon_window to [3, 9, 21] if not provided
        if shannon_window is None:
            self.shannon_window = [3, 9, 21]
        elif isinstance(shannon_window, int):
            self.shannon_window = [shannon_window]
        elif isinstance(shannon_window, (list, tuple)):
            self.shannon_window = list(shannon_window)
        else:
            raise ValueError("shannon_window must be int, list, or tuple")
        self.products = products if products is not None else []
        self.fill_iterations = fill_iterations  # Number of iterations for filling gaps
        self.fill_method = fill_method # IDW or FocalStatistics or None
        self.generate_boundary = generate_boundary # for binary mask and boundary shapefile
        self.message_length = 0

        # Initialize inpainter and binary mask according to the input parameters
        if self.input_dem and self.input_bs and self.binary_mask:
            print("Filling input rasters and using input binary mask to trim boundaries, assumed alignment")
            aligned_dem = self.input_dem
            aligned_bs = self.input_bs
            trimmed_dem_path, inpainter = MetaFunctions.fill_trim_with_intersection_mask(aligned_dem, aligned_bs, self.binary_mask, fill_method, fill_iterations)

        elif self.input_dem and self.input_bs and not self.binary_mask:
            print("Aligning dem and backscatter and then filling input raster, then generating boundaries and mask, and trimming rasters to binary mask")
            aligned_bs_path = ArcpyUtils.align_rasters(self.input_dem, self.input_bs)
            trimmed_dem_path, intersection_mask, inpainter = MetaFunctions.fill_trim_make_intersection_mask(self.input_dem, aligned_bs_path, fill_method, fill_iterations)
            self.binary_mask = intersection_mask

        elif self.input_dem and not self.input_bs and not self.binary_mask:
            if fill_method is not None:
                trimmed_dem_path, binary_mask, inpainter = MetaFunctions.fill_and_return_mask(self.input_dem, fill_method, fill_iterations, generate_boundary)
            if fill_method is None:
                print("Generating the binary mask from the inpur raster, no filling")
                trimmed_dem_path = self.input_dem
                self.binary_mask = ArcpyUtils.create_binary_mask(trimmed_dem_path, data_value=1, nodata_value=0)
                inpainter = Inpainter(trimmed_dem_path)

        elif self.input_dem and not self.input_bs and self.binary_mask:
            print("Using input raster and binary mask, without filling or trimming")
            inpainter = Inpainter(self.input_dem)
            trimmed_dem_path = self.input_dem

        self.input_dem = trimmed_dem_path
        self.inpainter = inpainter
        arcpy.env.extent = Raster(self.binary_mask).extent

    def process_dem(self): 
        """ Process the DEM to generate habitat derivatives. """
        if not self.input_dem:
            raise ValueError("Input DEM is not provided. Please provide a valid DEM path.")
        if not self.products:
            self.products = ["slope", "aspect", "roughness", "tpi", "tri", "hillshade"]
        print()
        print("Generating products", self.products)
        input_dem = self.input_dem
        print("........", input_dem)
        
        # define window for shannon index
        if isinstance(self.shannon_window, tuple):
            window = self.shannon_window[0]
        else:
            window = self.shannon_window
        
        # creating output file paths based on the product list
        output_files = {}
        for key in self.products:
            if key == "shannon_index":
                # For shannon_index, create output files for each window size
                for win in self.shannon_window:
                    output_files[f"{key}_{win}"] = os.path.join(
                        self.habitat_derivatives_folder,
                        f"{self.dem_name}_{key}_{win}.tif"
                    )
            else:
                output_files[key] = os.path.join(
                    self.habitat_derivatives_folder,
                    f"{self.dem_name}_{key}.tif"
                )
        output_slope=output_files.get("slope")
        output_aspect=output_files.get("aspect")
        output_roughness=output_files.get("roughness")
        output_tpi=output_files.get("tpi")
        output_tri=output_files.get("tri")
        output_hillshade=output_files.get("hillshade")
        output_lbp_3_1=output_files.get("lbp-3-1")
        output_lbp_15_2=output_files.get("lbp-15-2")
        output_lbp_21_3=output_files.get("lbp-21-3")
        output_dem=output_files.get("dem")

        """ Read a DEM and compute slope, aspect, roughness, TPI, and TRI. Output each to TIFF files based on user input. """
        def generate_products(input_dem, dem_data, transform, verbose):
            """ Generator function to yield each product's data and output file path. """
            habitat_derivatives = HabitatDerivatives(
                input_dem=input_dem,
                dem_data=dem_data,
                use_gdal=self.use_gdal,
                transform=transform,
                verbose=verbose
            )
            if output_slope:
                yield habitat_derivatives.calculate_slope(output_slope), output_slope   # Slope
            if output_aspect:
                yield habitat_derivatives.calculate_aspect(output_aspect), output_aspect  # Aspect
            if output_roughness:
                yield habitat_derivatives.calculate_roughness(output_roughness), output_roughness  # Roughness
            if output_tpi:
                yield habitat_derivatives.calculate_tpi(output_tpi), output_tpi  # TPI
            if output_tri:
                yield habitat_derivatives.calculate_tri(output_tri), output_tri  # TRI
            if output_hillshade:
                yield habitat_derivatives.calculate_hillshade(output_hillshade), output_hillshade # Hillshade
            # Loop through all shannon window sizes
            for win in self.shannon_window:
                key = f"shannon_index_{win}"
                if key in output_files:
                    yield habitat_derivatives.calculate_shannon_index_2d(win), output_files[key]
            if output_lbp_3_1:
                yield habitat_derivatives.calculate_lbp(3, 1), output_lbp_3_1  # LBP
            if output_lbp_15_2:
                yield habitat_derivatives.calculate_lbp(15, 2), output_lbp_15_2  # LBP
            if output_lbp_21_3:
                yield habitat_derivatives.calculate_lbp(21, 3), output_lbp_21_3  # LBP
            if output_dem:
                yield habitat_derivatives.return_dem_data(), output_dem  # used to chunk and merge data to get the same shape output as other products
        
        # Set arcpy.env variables before processing
        arcpy.env.outputCoordinateSystem = self.original_spatial_ref
        arcpy.env.snapRaster = self.original_snap_raster
        arcpy.env.cellSize = self.original_cell_size

        if not self.divisions:
            with rasterio.open(input_dem) as src:
                dem_data = src.read(1, masked=True)  # Read the DEM data with masked=True
                transform = src.transform  # Get the affine transform
                crs = src.crs  # Get the coordinate reference system
                metadata = src.meta  # Get metadata
                nodata = metadata['nodata']  # Get no-data value
                dem_data = RasterUtils.replace_nodata_with_nan(dem_data, nodata)  # replace any no-data values with NaN
                # Check if the DEM data is empty after filling no-data values
                if RasterUtils.is_empty(dem_data, nodata):
                    message = "The DEM data is empty after filling no-data values."
                    self.message_length = Utils.print_progress(message, self.message_length)
                    return None

            # Set arcpy.env variables before writing outputs (redundant, but ensures consistency)
            arcpy.env.outputCoordinateSystem = self.original_spatial_ref
            arcpy.env.snapRaster = self.original_snap_raster
            arcpy.env.cellSize = self.original_cell_size

            # Process and write each product one at a time
            for data, output_file in generate_products(input_dem, dem_data, transform, verbose=True):
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

                # Assert output spatial reference matches input
                out_desc = arcpy.Describe(output_file)
                assert out_desc.spatialReference.name == self.original_spatial_ref.name, \
                    f"Spatial reference changed! Expected: {self.original_spatial_ref.name}, Got: {out_desc.spatialReference.name}"
                # Trim the shannon index to avoid artifacts
                # Only trim the shannon index product after conversion
                if "shannon" in str(output_file).lower():
                    self.inpainter.trim_raster(output_file, self.binary_mask, overwrite=True)


        # If we are chunking, we need to process each chunk and then merge them
        elif self.divisions:
            # --- 1. Create a temporary folder for DEM chunks ---
            # Use abspath to ensure we are creating the folder exactly where we think we are
            dem_dir = os.path.dirname(os.path.abspath(self.input_dem))
            dem_chunk_temp_folder = os.path.join(dem_dir, "temp_dem_chunks_for_processing")
         
            if not os.path.exists(dem_chunk_temp_folder):
                try:
                    os.makedirs(dem_chunk_temp_folder)
                except OSError as e:
                    print(f"DEBUG: Failed to create temp folder: {e}")
                    raise

            # --- 2. Calculate Overlap ---
            # Calculate required overlap based on max window size to prevent edge effects
            # Formula: D = floor(W_max / 2). We add +1 for safety.
            max_filter_size = max(self.shannon_window) if self.shannon_window else 21
            overlap_px = (max_filter_size // 2) + 1
            print(f"Applying overlap of {overlap_px} pixels per tile side to prevent edge effects.")

            # --- 3. Process each DEM chunk ---
            with rasterio.open(input_dem) as src:
                # Get the original DEM's properties for later mosaicking of products
                original_dem_crs = src.crs

                # Calculate number of tiles in each direction and total tiles
                tile_size = max(1, src.height // self.divisions)
                n_tiles_y = (src.height + tile_size - 1) // tile_size
                n_tiles_x = (src.width + tile_size - 1) // tile_size
                total_tiles = n_tiles_y * n_tiles_x
                tile_counter = 0

                # Set arcpy.env variables before processing each chunk
                arcpy.env.outputCoordinateSystem = self.original_spatial_ref
                arcpy.env.snapRaster = self.original_snap_raster
                arcpy.env.cellSize = self.original_cell_size

                for i in range(0, src.height, tile_size):
                    for j in range(0, src.width, tile_size):
                        tile_counter += 1
                        # A. Define the "Target" Window (The final output size for this tile)
                        # This determines where the final pixels go in the mosaic
                        target_width = min(tile_size, src.width - j)
                        target_height = min(tile_size, src.height - i)
                        write_window = Window(j, i, target_width, target_height)
                        
                        # Calculate the transform for the FINAL output position
                        write_transform = src.window_transform(write_window)

                        # B. Define the "Read" Window (Target + Overlap)
                        # Clamp coordinates so we don't go outside the image
                        read_row_start = max(0, i - overlap_px)
                        read_col_start = max(0, j - overlap_px)
                        read_row_stop = min(src.height, i + target_height + overlap_px)
                        read_col_stop = min(src.width, j + target_width + overlap_px)

                        read_window = Window.from_slices(
                            (read_row_start, read_row_stop), 
                            (read_col_start, read_col_stop)
                        )
                        
                        # Calculate offsets: How many pixels did we actually add to the top/left?
                        # We need this to crop the result later.
                        pad_top = i - read_row_start
                        pad_left = j - read_col_start

                        # C. Read and Save the Buffered Chunk
                        # We perform calculations on the *Expanded* (Buffered) data
                        chunk_data_padded = src.read(1, window=read_window)
                        chunk_transform_padded = src.window_transform(read_window)
                        
                        # Define path for the temporary DEM chunk
                        chunk_dem_path = os.path.join(dem_chunk_temp_folder, f"{self.dem_name}_chunk_{i}_{j}.tif")
                        
                        # Save the PADDED DEM chunk to disk
                        with rasterio.open(
                            chunk_dem_path,
                            'w',
                            driver='GTiff',
                            height=chunk_data_padded.shape[0],
                            width=chunk_data_padded.shape[1],
                            count=1,
                            dtype=str(chunk_data_padded.dtype),
                            crs=original_dem_crs,
                            transform=chunk_transform_padded # Use the padded transform
                        ) as dst:
                            dst.write(chunk_data_padded, 1)

                        message = f"Processing tile {tile_counter}/{total_tiles} (Overlap: {overlap_px}px)"
                        self.message_length = Utils.print_progress(message, self.message_length)

                        # D. Generate Products on Padded Data
                        for data, output_file in generate_products(chunk_dem_path, chunk_data_padded, chunk_transform_padded, verbose=False):
                            # Ensure output folder exists for the product
                            output_dir = os.path.dirname(output_file)
                            product_name = os.path.basename(output_file).split(".")[0]
                            
                            # Define the chunk output file path
                            product_folder = os.path.join(output_dir, product_name)
                            os.makedirs(product_folder, exist_ok=True)

                            chunk_output_file = os.path.join(product_folder, f"{product_name}_chunk_{i}_chunk_{j}.tif")

                            # Skip processing if the chunk already exists
                            if os.path.exists(chunk_output_file):
                                message = f"Skipping existing chunk at ({i}_{j}) (row_col) for {output_file}."
                                self.message_length = Utils.print_progress(message, self.message_length)
                                continue
                            
                            if data is None:
                                # copy the output chunk to the folder
                                arcpy.Copy_management(output_file, chunk_output_file)
                                arcpy.Delete_management(output_file)

                            else:
                                # E. Crop the Result
                                # We calculated on the padded area, now we cut out the center "valid" area
                                # Slicing syntax: [start_row : end_row, start_col : end_col]
                                cropped_data = data[
                                    pad_top : pad_top + target_height, 
                                    pad_left : pad_left + target_width
                                ]

                                # F. Save the Cropped (Seamless) Chunk
                                with rasterio.open(
                                    chunk_output_file,
                                    'w',
                                    driver='GTiff',
                                    height=cropped_data.shape[0],
                                    width=cropped_data.shape[1],
                                    count=1,
                                    dtype=cropped_data.dtype,
                                    crs=original_dem_crs,
                                    transform=write_transform # Use the original TARGET transform
                                ) as dst:
                                    dst.write(cropped_data, 1)
                                    dst.update_tags(**src.tags())

                            # Assert output spatial reference matches input
                            out_desc = arcpy.Describe(chunk_output_file)
                            assert out_desc.spatialReference.name == self.original_spatial_ref.name, \
                                f"Spatial reference changed! Expected: {self.original_spatial_ref.name}, Got: {out_desc.spatialReference.name}"
                            
                # --- 4. Merge and Clean ---
                print() # Finalize progress bar

                for product_key, output_file in output_files.items(): 
                    # Determine the folder where chunks are stored
                    dem_chunks_path = os.path.join(os.path.dirname(output_file), os.path.basename(output_file).split(".")[0])
                    
                    if os.path.exists(dem_chunks_path): 
                        # STEP A: Merge, but DO NOT let the utility delete chunks yet (prevents the crash)
                        merged_dem = ArcpyUtils.merge_dem_arcpy(dem_chunks_path, remove_chunks=False)
                        
                        # STEP B: Post-process the merged result
                        trimmed_raster_path = self.inpainter.trim_raster(merged_dem, self.binary_mask, overwrite=True)
                        ArcpyUtils.compress_raster(trimmed_raster_path, format="TIFF", overwrite=True)
                        
                        # STEP C: Explicitly release references
                        del merged_dem 
                        
                        # STEP D: Robust cleanup
                        ArcpyUtils.cleanup_chunks(dem_chunks_path)
                        Utils.remove_additional_files(directory=os.path.dirname(dem_chunks_path))
                    else:
                        print(f"Warning: No chunks found for {product_key}. Skipping merge.")

            # Cleanup the temporary raw DEM chunks folder as well
            if 'dem_chunk_temp_folder' in locals():
                ArcpyUtils.cleanup_chunks(dem_chunk_temp_folder)

            # Reset Environment
            arcpy.env.outputCoordinateSystem = self.original_spatial_ref
            arcpy.env.snapRaster = self.original_snap_raster
            arcpy.env.cellSize = self.original_cell_size

        # Clean up additional files after processing
        Utils.remove_additional_files(directory=os.path.dirname(input_dem))

# example usage
if __name__ == "__main__":
    start_time = time.time()

    out_folder = "habitat_derivatives"
    input_dem = "dem\\dem.tif"  # Path to the input DEM file
    products = ["slope"] # Specify the products to generate, e.g., ["slope", "aspect", "roughness", "tpi", "tri", "hillshade", "shannon_index", "lbp-3-1"]
    # Create an instance of the HabitatDerivatives class with the specified parameters
    habitat_derivatives = ProcessDem(
                                    input_dem=input_dem, 
                                    output_folder=out_folder,
                                    products=products,
                                    shannon_window=21,
                                    fill_iterations=1,
                                    fill_method=None,  # "IDW" or "FocalStatistics" or None
                                    divisions=8,  # Set to None for no chunking, or specify a chunk size
                                    )
    habitat_derivatives.process_dem()

    # Print time lapsed
    time_convert(time.time() - start_time)
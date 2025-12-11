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
from landforms import Landforms
import numpy as np

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
        save_chunks=False
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
        arcpy.env.overwriteOutput = True
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
        self.save_chunks = save_chunks
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
                trimmed_dem_path, binary_mask, inpainter = MetaFunctions.fill_and_return_mask(self.input_dem, fill_method, fill_iterations)
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

        if not self.input_dem:
            raise ValueError("Input DEM is not provided. Please provide a valid DEM path.")
        if not self.products:
            self.products = ["slope", "aspect", "roughness", "tpi", "tri", "hillshade"]
        
        # --- FIX: Expand Bathymorphons to all sub-products ---
        if self.products:
            # Normalize input keys (handle 'geomorphons' alias if used)
            if "geomorphons" in self.products:
                self.products.remove("geomorphons")
                if "bathymorphons" not in self.products:
                    self.products.append("bathymorphons")

            if "bathymorphons" in self.products:
                self.products.remove("bathymorphons")
                # Add specific classes if not present
                for p in ["bathymorphons_raw", "bathymorphons_10c", "bathymorphons_6c", "bathymorphons_5c", "bathymorphons_4c"]:
                    if p not in self.products:
                        self.products.append(p)

        # Output file paths
        output_files = {}
        for key in self.products:
            if key == "shannon_index":
                for win in self.shannon_window:
                    output_files[f"{key}_{win}"] = os.path.join(self.habitat_derivatives_folder, f"{self.dem_name}_{key}_{win}.tif")
            # Handle Bathymorphons file naming
            elif "bathymorphons" in key:
                # e.g. bathymorphons_10c -> name_bathymorphons_10c.tif
                output_files[key] = os.path.join(self.habitat_derivatives_folder, f"{self.dem_name}_{key}.tif")
            else:
                output_files[key] = os.path.join(self.habitat_derivatives_folder, f"{self.dem_name}_{key}.tif")
        # define window for shannon index
        if isinstance(self.shannon_window, tuple):
            window = self.shannon_window[0]
        else:
            window = self.shannon_window
        
        self.output_files = output_files

    def process_dem(self): 
        """ Process the DEM to generate habitat derivatives. """
        output_files = self.output_files
        input_dem = self.input_dem
        print()
        print("Generating products", self.products)
        print("........", input_dem)
        
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
        # --- UPDATED GENERATOR: Accepts 'products_to_skip' ---
        def generate_products(input_dem, dem_data, transform, verbose, products_to_skip=None):
            """ Generator function to yield each product's data and output file path. """
            if products_to_skip is None:
                products_to_skip = []
                
            habitat_derivatives = HabitatDerivatives(
                input_dem=input_dem,
                dem_data=dem_data,
                use_gdal=self.use_gdal,
                transform=transform,
                verbose=verbose
            )
            
            # Note: We check if the KEY (e.g., 'slope') is in the skip list before calculating
            if output_slope and "slope" not in products_to_skip:
                yield habitat_derivatives.calculate_slope(output_slope), output_slope
            if output_aspect and "aspect" not in products_to_skip:
                yield habitat_derivatives.calculate_aspect(output_aspect), output_aspect
            if output_roughness and "roughness" not in products_to_skip:
                yield habitat_derivatives.calculate_roughness(output_roughness), output_roughness
            if output_tpi and "tpi" not in products_to_skip:
                yield habitat_derivatives.calculate_tpi(output_tpi), output_tpi
            if output_tri and "tri" not in products_to_skip:
                yield habitat_derivatives.calculate_tri(output_tri), output_tri
            if output_hillshade and "hillshade" not in products_to_skip:
                yield habitat_derivatives.calculate_hillshade(output_hillshade), output_hillshade
            
            for win in self.shannon_window:
                key = f"shannon_index_{win}"
                if key in output_files and key not in products_to_skip:
                    yield habitat_derivatives.calculate_shannon_index_2d(win), output_files[key]
            
            if output_lbp_3_1 and "lbp-3-1" not in products_to_skip:
                yield habitat_derivatives.calculate_lbp(3, 1), output_lbp_3_1
            if output_lbp_15_2 and "lbp-15-2" not in products_to_skip:
                yield habitat_derivatives.calculate_lbp(15, 2), output_lbp_15_2
            if output_lbp_21_3 and "lbp-21-3" not in products_to_skip:
                yield habitat_derivatives.calculate_lbp(21, 3), output_lbp_21_3
            if output_dem and "dem" not in products_to_skip:
                yield habitat_derivatives.return_dem_data(), output_dem
            
            # --- NEW: Landforms / Bathymorphons ---
            # Check which landforms are needed and not skipped
            needed_lf = [p for p in ["bathymorphons_raw", "bathymorphons_10c", "bathymorphons_6c", "bathymorphons_5c", "bathymorphons_4c"] 
                         if p in output_files and p not in products_to_skip]
            
            if needed_lf:
                # Calculate Raw + 10c (ArcPy execution)
                # Pass 'input_dem' (the chunk path) NOT 'chunk_dem_path'
                raw_arr, class10_arr = Landforms.calculate_landforms_chunk(
                    input_dem, 
                    angle_threshold=1, 
                    search_distance=10, 
                    skip_distance=5
                )
                
                if raw_arr is not None:
                    # Yield Raw
                    if "bathymorphons_raw" in needed_lf:
                        yield raw_arr, output_files["bathymorphons_raw"]
                    
                    # Yield 10c (from Tool Output)
                    if class10_arr is not None and "bathymorphons_10c" in needed_lf:
                        yield class10_arr, output_files["bathymorphons_10c"]
                        
                    # Yield Classifications (Derived from Raw)
                    if "bathymorphons_6c" in needed_lf:
                        yield Landforms.classify_chunk(raw_arr, "6c"), output_files["bathymorphons_6c"]
                    
                    if "bathymorphons_5c" in needed_lf:
                        yield Landforms.classify_chunk(raw_arr, "5c"), output_files["bathymorphons_5c"]
                        
                    if "bathymorphons_4c" in needed_lf:
                        yield Landforms.classify_chunk(raw_arr, "4c"), output_files["bathymorphons_4c"]

        # Reset Env
        arcpy.env.outputCoordinateSystem = self.original_spatial_ref
        arcpy.env.snapRaster = self.original_snap_raster
        arcpy.env.cellSize = self.original_cell_size
        
        # Set arcpy.env variables before processing
        arcpy.env.outputCoordinateSystem = self.original_spatial_ref
        arcpy.env.snapRaster = self.original_snap_raster
        arcpy.env.cellSize = self.original_cell_size

        if not self.divisions:
            with rasterio.open(input_dem) as src:
                dem_data = src.read(1, masked=True)
                transform = src.transform
                crs = src.crs
                metadata = src.meta
                src_nodata = metadata['nodata'] # Capture original nodata
                
                # Prepare data for processing (NaNs for floats)
                dem_data_filled = RasterUtils.replace_nodata_with_nan(dem_data, src_nodata)
                # --- ADD THIS SANITIZATION ---
                if np.issubdtype(dem_data_filled.dtype, np.floating):
                    with np.errstate(invalid='ignore'):
                        dem_data_filled[dem_data_filled < -15000.0] = np.nan
                # -----------------------------
                if RasterUtils.is_empty(dem_data_filled, src_nodata):
                    message = "The DEM data is empty after filling no-data values."
                    self.message_length = Utils.print_progress(message, self.message_length)
                    return None

                # Create a Boolean Mask of where the Data is MISSING
                # We will use this to clean up the output products
                if np.issubdtype(dem_data_filled.dtype, np.floating):
                    dem_missing_mask = np.isnan(dem_data_filled)
                else:
                    dem_missing_mask = (dem_data_filled == src_nodata)

            # Process and write each product one at a time
            for data, output_file in generate_products(input_dem, dem_data_filled, transform, verbose=True):
                if data is not None:
                    
                    # 1. APPLY MASK (Clean edges/background)
                    if np.issubdtype(data.dtype, np.floating):
                        data[dem_missing_mask] = np.nan
                        out_nodata = np.nan
                    else:
                        # For integer products (LBP, Hillshade), use 0 as NoData
                        data[dem_missing_mask] = 0
                        out_nodata = 0

                    # 2. WRITE WITH COMPRESSION
                    with rasterio.open(
                        output_file,
                        'w',
                        driver='GTiff',
                        compress='lzw',          # <--- Added Compression
                        height=dem_data_filled.shape[0],
                        width=dem_data_filled.shape[1],
                        count=1,
                        dtype=data.dtype,
                        crs=crs,
                        transform=transform,
                        nodata=out_nodata        # <--- Added Explicit NoData
                    ) as dst:
                        dst.write(data, 1)
                        dst.update_tags(**src.tags())

                # Assert output spatial reference matches input
                if os.path.exists(output_file):
                    out_desc = arcpy.Describe(output_file)
                    assert out_desc.spatialReference.name == self.original_spatial_ref.name, \
                        f"Spatial reference changed! Expected: {self.original_spatial_ref.name}, Got: {out_desc.spatialReference.name}"
                    
                    # Trim the shannon index to avoid specific window artifacts (Halo effect)
                    if "shannon" in str(output_file).lower():
                        self.inpainter.trim_raster(output_file, self.binary_mask, overwrite=True)


        # If we are chunking, we need to process each chunk and then merge them
        if self.divisions:
            dem_dir = os.path.dirname(os.path.abspath(self.input_dem))
            dem_chunk_temp_folder = os.path.join(dem_dir, "temp_dem_chunks_for_processing")
            if not os.path.exists(dem_chunk_temp_folder):
                try: os.makedirs(dem_chunk_temp_folder)
                except OSError: pass

            # Calculate required overlap based on max window size to prevent edge effects
            # Formula: D = floor(W_max / 2). We add +1 for safety
            # 1. Determine Max Window for Overlap
            max_shannon = max(self.shannon_window) if self.shannon_window else 0
            
            # Check if landforms are requested to adjust overlap
            max_geomorph = 20 if any("bathymorphons" in p for p in self.products) else 0 
            
            max_filter_size = max(max_shannon, max_geomorph)
            overlap_px = int(max_filter_size // 2) + 5 # Add padding
            
            print(f"Applying overlap of {overlap_px} pixels per tile side.")

            with rasterio.open(input_dem) as src:
                original_dem_crs = src.crs
                tile_size = max(1, src.height // self.divisions)
                n_tiles_y = (src.height + tile_size - 1) // tile_size
                n_tiles_x = (src.width + tile_size - 1) // tile_size
                total_tiles = n_tiles_y * n_tiles_x
                tile_counter = 0

                arcpy.env.outputCoordinateSystem = self.original_spatial_ref
                arcpy.env.snapRaster = self.original_snap_raster
                arcpy.env.cellSize = self.original_cell_size
                arcpy.env.overwriteOutput = True

                for i in range(0, src.height, tile_size):
                    for j in range(0, src.width, tile_size):
                        tile_counter += 1
                        
                        # --- OPTIMIZATION 1: Identify Skippable Products ---
                        products_to_skip = []
                        all_products_exist = True
                        
                        for prod_key, prod_master_path in output_files.items():
                            prod_dir = os.path.dirname(prod_master_path)
                            prod_name = os.path.basename(prod_master_path).split(".")[0]
                            chunk_folder = os.path.join(prod_dir, prod_name)
                            chunk_path = os.path.join(chunk_folder, f"{prod_name}_chunk_{i}_{j}.tif")
                            
                            if os.path.exists(chunk_path):
                                products_to_skip.append(prod_key)
                            else:
                                all_products_exist = False
                        
                        if all_products_exist:
                            message = f"Skipping Tile {tile_counter}/{total_tiles} (All products exist)"
                            self.message_length = Utils.print_progress(message, self.message_length)
                            continue
                        # -----------------------------------------------------

                        # A. Define Write Window
                        target_width = min(tile_size, src.width - j)
                        target_height = min(tile_size, src.height - i)
                        write_window = Window(j, i, target_width, target_height)
                        write_transform = src.window_transform(write_window)

                        # B. Define Read Window
                        read_row_start = max(0, i - overlap_px)
                        read_col_start = max(0, j - overlap_px)
                        read_row_stop = min(src.height, i + target_height + overlap_px)
                        read_col_stop = min(src.width, j + target_width + overlap_px)
                        read_window = Window.from_slices((read_row_start, read_row_stop), (read_col_start, read_col_stop))
                        
                        pad_top = i - read_row_start
                        pad_left = j - read_col_start
                        
                        chunk_dem_path = os.path.join(dem_chunk_temp_folder, f"{self.dem_name}_chunk_{i}_{j}.tif")
                        
                        # Optimization 2: Reuse temp chunk if available
                        chunk_data_padded = None
                        chunk_transform_padded = None
                        
                        if os.path.exists(chunk_dem_path):
                            try:
                                with rasterio.open(chunk_dem_path) as tmp_src:
                                    chunk_data_padded = tmp_src.read(1)
                                    chunk_transform_padded = tmp_src.transform
                            except Exception:
                                chunk_data_padded = None

                        if chunk_data_padded is None:
                            chunk_data_padded = src.read(1, window=read_window)
                            chunk_transform_padded = src.window_transform(read_window)
                            
                            if np.issubdtype(chunk_data_padded.dtype, np.floating):
                                with np.errstate(invalid='ignore'):
                                    mask = chunk_data_padded < -15000.0
                                    if np.any(mask):
                                        chunk_data_padded[mask] = np.nan

                            with rasterio.open(
                                chunk_dem_path, 'w', driver='GTiff', compress='lzw',
                                height=chunk_data_padded.shape[0], width=chunk_data_padded.shape[1],
                                count=1, dtype=str(chunk_data_padded.dtype),
                                crs=original_dem_crs, transform=chunk_transform_padded
                            ) as dst:
                                dst.write(chunk_data_padded, 1)

                        message = f"Processing tile {tile_counter}/{total_tiles} (Skipping: {len(products_to_skip)})"
                        self.message_length = Utils.print_progress(message, self.message_length)

                        # D. Generate Products (Pass Skip List)
                        for data, output_file in generate_products(chunk_dem_path, chunk_data_padded, chunk_transform_padded, verbose=False, products_to_skip=products_to_skip):
                            output_dir = os.path.dirname(output_file)
                            product_name = os.path.basename(output_file).split(".")[0]
                            product_folder = os.path.join(output_dir, product_name)
                            os.makedirs(product_folder, exist_ok=True)
                            chunk_output_file = os.path.join(product_folder, f"{product_name}_chunk_{i}_chunk_{j}.tif")

                            if data is None:
                                if arcpy.Exists(output_file):
                                    try:
                                        arcpy.management.Copy(output_file, chunk_output_file)
                                        arcpy.management.Delete(output_file)
                                    except Exception: pass
                            else:
                                cropped_data = data[pad_top : pad_top + target_height, pad_left : pad_left + target_width]
                                cropped_dem = chunk_data_padded[pad_top : pad_top + target_height, pad_left : pad_left + target_width]
                                
                                if np.issubdtype(cropped_dem.dtype, np.floating):
                                    mask = np.isnan(cropped_dem)
                                else:
                                    mask = (cropped_dem == src.nodata) if src.nodata is not None else (cropped_dem == 0)

                                if np.issubdtype(cropped_data.dtype, np.floating):
                                    cropped_data[mask] = np.nan
                                    out_nodata = np.nan
                                else:
                                    cropped_data[mask] = 0 
                                    out_nodata = 0

                                with rasterio.open(
                                    chunk_output_file, 'w', driver='GTiff', compress='lzw',
                                    height=cropped_data.shape[0], width=cropped_data.shape[1],
                                    count=1, dtype=cropped_data.dtype,
                                    crs=original_dem_crs, transform=write_transform, nodata=out_nodata
                                ) as dst:
                                    dst.write(cropped_data, 1)
                                    dst.update_tags(**src.tags())

                            if os.path.exists(chunk_output_file):
                                try:
                                    out_desc = arcpy.Describe(chunk_output_file)
                                except Exception: pass
                            else:
                                print(f"Warning: Output chunk was not created for tile {i},{j}.")
                            
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
                if not self.save_chunks:
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
import rasterio
import os
import arcpy
from arcpy.sa import *
from utils import Utils
from arcpyUtils import ArcpyUtils
from rasterUtils import RasterUtils
from gdalUtils import GdalUtils
from inpainter import Inpainter
from metafunctions import MetaFunctions
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
        min_area=50,
        water_elevation=183.6,
        keep_chunks=False,
        bypass_mosaic=False
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
        self.min_area = min_area
        # Ensure raster statistics exist before transformation
        try:
            # Try to get a property that requires stats
            _ = arcpy.management.GetRasterProperties(self.input_dem, "MINIMUM").getOutput(0)
        except Exception:
            print("Statistics missing or unreadable. Calculating...")
            # ignore_values="" handles NoData correctly
            # skip_existing=True ensures we don't recalculate if they actually exist but were just locked
            arcpy.management.CalculateStatistics(self.input_dem, skip_existing=True)

        self.input_bs = input_bs if input_bs else None

        if self.input_dem and self.input_bs:
            assert arcpy.Describe(self.input_dem).spatialReference.name == arcpy.Describe(self.input_bs).spatialReference.name, \
                "Spatial Ref not matching! You can use the function RasterUtils.transform_spatial_reference_arcpy(base_raster, transform_raster)"
            
        # converts to depth if it is in elevation
        self.input_dem = ArcpyUtils.apply_height_to_depth_transformation(self.input_dem, water_elevation)

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
        self.bypass_mosaic = bypass_mosaic
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
        self.keep_chunks = keep_chunks
        self.message_length = 0

        # Initialize inpainter and binary mask according to the input parameters
        # but ensure they exist in scope.
        fill_m = self.fill_method
        fill_i = self.fill_iterations
        min_area = self.min_area
        # 1. Handle Backscatter Alignment early (Unified)
        if self.input_bs and self.input_dem and not self.binary_mask:
            # Only align if not already handled by a specific complex function later
            if not self.divisions: 
                print("Aligning backscatter to DEM...")
                self.input_bs = ArcpyUtils.align_rasters(self.input_dem, self.input_bs)

        # 2. Primary Logic Tree
        if self.input_dem and self.binary_mask:
            if self.divisions:
                if fill_m is None:
                    print("No Fill method: Using input binary mask to trim final products.")
                else:
                    print("Filling deferred to tiling. WARNING: If input binary mask is not-filled, it will re-punch holes/voids during trimming.")
            else:
                # Global Fill + Trim with provided mask
                print("Filling and trimming using provided intersection mask...")
                self.input_dem = MetaFunctions.fill_trim_with_intersection_mask(
                    self.input_dem, self.input_bs, self.binary_mask, fill_m, fill_i, min_area
                )

        elif self.input_dem and not self.binary_mask:
            if self.divisions:
                if fill_m is None:
                    print("Tiling enabled: No filling. Generating mask from raw DEM.")
                    self.binary_mask = ArcpyUtils.create_binary_mask(self.input_dem, data_value=1, nodata_value=0)
                else:
                    print("Tiling enabled: Binary mask will be generated from final filled tiles.")
            else:
                # No tiling, no mask provided: Generate everything now
                if self.input_bs:
                    print("Generating intersection mask and filling globally...")
                    trimmed_path, mask = MetaFunctions.fill_trim_make_intersection_mask(
                        self.input_dem, self.input_bs, fill_m, fill_i, min_area
                    )
                    self.input_dem, self.binary_mask = trimmed_path, mask
                else:
                    if fill_m is not None:
                        print("Filling globally and generating mask...")
                        self.input_dem, self.binary_mask = MetaFunctions.fill_and_return_mask(
                            self.input_dem, fill_m, fill_i, min_area
                        )
                    else:
                        print("No filling: Generating mask from raw DEM.")
                        self.binary_mask = ArcpyUtils.create_binary_mask(self.input_dem, data_value=1, nodata_value=0)

        arcpy.env.extent = Raster(self.input_dem).extent

        if not self.input_dem:
            raise ValueError("Input DEM is not provided. Please provide a valid DEM path.")
        if not self.products:
            self.products = ["slope", "aspect", "roughness", "tpi", "tri", "hillshade"]
        
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
            # 2. Expand "lbp" -> "lbp-21-4"
            if "lbp" in self.products:
                self.products.remove("lbp")
                if "lbp-21-4" not in self.products:
                    self.products.append("lbp-21-4")

        # Output file paths
        output_files = {}
        for key in self.products:
            if key == "shannon":
                for win in self.shannon_window:
                    output_files[f"{key}{win}"] = os.path.join(self.habitat_derivatives_folder, f"{self.dem_name}_{key}{win}.tif")
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
        output_lbp_8_1=output_files.get("lbp-8-1")
        output_lbp_15_2=output_files.get("lbp-15-2")
        output_lbp_21_4=output_files.get("lbp-21-4")
        output_filled_dem=output_files.get("filled")

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
                key = f"shannon{win}"
                if key in output_files and key not in products_to_skip:
                    yield habitat_derivatives.calculate_shannon_index_2d(win), output_files[key]
            
            if output_lbp_8_1 and "lbp-8-1" not in products_to_skip:
                yield habitat_derivatives.calculate_lbp(8, 1), output_lbp_8_1
            if output_lbp_15_2 and "lbp-15-2" not in products_to_skip:
                yield habitat_derivatives.calculate_lbp(15, 2), output_lbp_15_2
            if output_lbp_21_4 and "lbp-21-4" not in products_to_skip:
                yield habitat_derivatives.calculate_lbp(21, 4), output_lbp_21_4
            if output_filled_dem and "filled" not in products_to_skip:
                yield habitat_derivatives.return_dem_data(), output_filled_dem
            
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
                        ArcpyUtils.trim_raster(output_file, self.binary_mask, overwrite=True)


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
                        chunks_to_skip = []

                        for prod_key, prod_master_path in output_files.items():
                            prod_dir = os.path.dirname(prod_master_path)
                            prod_name = os.path.basename(prod_master_path).split(".")[0]
                            chunk_folder = os.path.join(prod_dir, prod_name)
                            chunk_path = os.path.join(chunk_folder, f"{prod_name}_chunk_{i}_{j}.tif")
                            
                            if os.path.exists(chunk_path):
                                chunks_to_skip.append(prod_key)
                        
                        # --- BUG FIX START: Only continue if ALL products exist ---
                        if len(chunks_to_skip) == len(output_files):
                            message = f"Skipping Tile {tile_counter}/{total_tiles} (All products exist)"
                            self.message_length = Utils.print_progress(message, self.message_length)
                            continue
                        # --- BUG FIX END ---
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
                        
                        # C. Read and Save the Buffered Chunk
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
                            
                            # --- 1. SANITIZE BEFORE WRITING (Critical Fix) ---
                            # This prevents "Dirty" data (3.4e38) from ever touching the disk.
                            if np.issubdtype(chunk_data_padded.dtype, np.floating):
                                with np.errstate(invalid='ignore'):
                                    # Catch huge positive/negative artifacts
                                    mask = np.abs(chunk_data_padded) > 100000.0
                                    if np.any(mask):
                                        chunk_data_padded[mask] = src.nodata if src.nodata is not None else np.nan
                            # ------------------------------------------------

                            with rasterio.open(
                                chunk_dem_path, 'w', driver='GTiff', compress='lzw',
                                height=chunk_data_padded.shape[0], width=chunk_data_padded.shape[1],
                                count=1, dtype=str(chunk_data_padded.dtype),
                                crs=original_dem_crs, transform=chunk_transform_padded,
                                nodata=src.nodata 
                            ) as dst:
                                dst.write(chunk_data_padded, 1)

                        # --- Apply Tiled Filling ---
                        if self.divisions and self.fill_method is not None:
                            # Note: We check for NaNs OR the NoData value
                            should_fill = False
                            if np.isnan(chunk_data_padded).any():
                                should_fill = True
                            elif src.nodata is not None and (chunk_data_padded == src.nodata).any():
                                should_fill = True

                            if should_fill:
                                # --- BRANCH LOGIC ---
                                if self.fill_method == "IDW":
                                    chunk_dem_path = Inpainter.fill_chunk_idw(
                                        chunk_dem_path, 
                                        iterations=self.fill_iterations,
                                        power=2.0,
                                        search_radius=5.0 # Adjust search radius as needed
                                    )
                                elif self.fill_method == "FocalStatistics":
                                    chunk_dem_path = Inpainter.fill_chunk_focal_stats(
                                        chunk_dem_path, 
                                        iterations=self.fill_iterations,
                                        kernel_size=9
                                    )
                                # Reload data (Only once!)
                                with rasterio.open(chunk_dem_path) as src_filled:
                                    chunk_data_padded = src_filled.read(1)
                                    
                                # --- 2. POST-FILL SANITIZATION (Safety Net) ---
                                # If the Inpainter created new artifacts, we clean them 
                                # AND update the file so ArcPy doesn't choke.
                                if np.issubdtype(chunk_data_padded.dtype, np.floating):
                                    with np.errstate(invalid='ignore'):
                                        mask = np.abs(chunk_data_padded) > 100000.0
                                        if np.any(mask):
                                            message = f"Sanitized artifacts in tile {tile_counter}"
                                            self.message_length = Utils.print_progress(message, self.message_length)
                                            chunk_data_padded[mask] = np.nan
                                            
                                            # UPDATE THE FILE ON DISK
                                            with rasterio.open(chunk_dem_path, 'r+') as dst:
                                                dst.write(chunk_data_padded, 1)
                                    
                        message = f"Processing tile {tile_counter}/{total_tiles} (Skipping: {len(chunks_to_skip)})"
                        self.message_length = Utils.print_progress(message, self.message_length)
                        # D. Generate Products (Pass Skip List)
                        for data, output_file in generate_products(chunk_dem_path, chunk_data_padded, chunk_transform_padded, verbose=False, products_to_skip=chunks_to_skip):
                            output_dir = os.path.dirname(output_file)
                            product_name = os.path.basename(output_file).split(".")[0]
                            product_folder = os.path.join(output_dir, product_name)
                            os.makedirs(product_folder, exist_ok=True)
                            chunk_output_file = os.path.join(product_folder, f"{product_name}_chunk_{i}_{j}.tif")

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

                # ==========================================================
                # FIX 1: RESET EXTENT BEFORE MERGE
                # ==========================================================
                # This prevents the merge from being clipped to the last tile's corner
                arcpy.env.extent = self.original_snap_raster 
                arcpy.env.snapRaster = self.original_snap_raster
                # ==========================================================

                # ==========================================================
                # FIX 2: CREATE PRIORITY QUEUE
                # ==========================================================
                # Convert dictionary items to a list so we can sort them.
                merge_queue = list(output_files.items())
                
                # Sort logic: 
                # If product name is 'filled', give it index 0 (First).
                # All other products get index 1 (Second).
                merge_queue.sort(key=lambda x: 0 if x[0] == 'filled' else 1)

                for product_key, output_file in merge_queue: 
                    # Determine the folder where chunks are stored
                    dem_chunks_path = os.path.join(os.path.dirname(output_file), os.path.basename(output_file).split(".")[0])
                    
                    if os.path.exists(dem_chunks_path) and not self.bypass_mosaic: 
                        # STEP A: Merge
                        print(f"Merging chunks for {product_key}...")
                        merged_dem = ArcpyUtils.merge_dem_arcpy(dem_chunks_path, remove_chunks=False)
                        
                        # STEP B: Update Mask (If this is the filled DEM)
                        if product_key == 'filled':
                            if not self.binary_mask:
                                print("Generating authoritative binary mask from Filled DEM...")
                                # This updates self.binary_mask so subsequent loops (Slope, Aspect) use the correct one
                                self.binary_mask = ArcpyUtils.create_binary_mask(merged_dem, data_value=1, nodata_value=0)

                                if self.fill_method in ["FocalStatistics", "IDW"]:
                                    shrink_cells = 0
                                    
                                    if self.fill_method == "FocalStatistics":
                                        # Kernel was 9, so radius is 4
                                        shrink_cells = 4 * self.fill_iterations
                                        
                                    elif self.fill_method == "IDW":
                                        # Search radius was set to 5.0 in the tiling loop
                                        # If you changed the search_radius in the loop, update this number!
                                        idw_radius = 5 
                                        shrink_cells = idw_radius * self.fill_iterations
                                    
                                    if shrink_cells > 0:
                                        print(f"Removing {self.fill_method} halo (Shrinking by {shrink_cells} cells)...")
                                        # Apply Shrink (Erosion)
                                        shrunk_mask = Shrink(self.binary_mask, shrink_cells, [1]) # type: Raster
                                        # Overwrite the mask file
                                        shrunk_mask.save(self.binary_mask)
                                        # Clean up
                                        del shrunk_mask

                        # STEP C: Post-process (Trim)
                        # We trim ALL products (including 'filled' itself to clean edges) using the new mask
                        if self.binary_mask:
                            print(f"Trimming {product_key} to binary mask...")
                            trimmed_raster_path = ArcpyUtils.trim_raster(merged_dem, self.binary_mask, overwrite=True)
                        else:
                            trimmed_raster_path = merged_dem

                        # STEP D: Compress
                        ArcpyUtils.compress_raster(trimmed_raster_path, format="TIFF", overwrite=False)
                        
                        # STEP E: Cleanup
                        del merged_dem 
                        ArcpyUtils.cleanup_chunks(dem_chunks_path)
                        Utils.remove_additional_files(directory=os.path.dirname(dem_chunks_path))
                    else:
                        print(f"Warning: No chunks found for {product_key}. Skipping merge.")

            # Cleanup the temporary raw DEM chunks folder as well
            if 'dem_chunk_temp_folder' in locals():
                if not self.keep_chunks:
                    ArcpyUtils.cleanup_chunks(dem_chunk_temp_folder)

            # Reset Environment
            arcpy.env.outputCoordinateSystem = self.original_spatial_ref
            arcpy.env.snapRaster = self.original_snap_raster
            arcpy.env.cellSize = self.original_cell_size

        # Clean up additional files after processing
        Utils.remove_additional_files(directory=os.path.dirname(input_dem))


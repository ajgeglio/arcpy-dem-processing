from osgeo import gdal
import rasterio
from rasterio.enums import Resampling
import numpy as np
import os
import glob
import subprocess
from pathlib import Path
from utils import Utils
import time
import gc

class GdalUtils:
    """Utility class for handling GDAL operations, including compression and conversion of DEM files."""

    @staticmethod
    def raster_info(input_dem_path):
        '''
        print the hight, width, and pixel count of the raster path
        input: path to raster file
        output: print statement
        '''
        try:
            with rasterio.open(input_dem_path) as src:
                print(f"File: {input_dem_path}")
                print("-" * 30)
                print(f"Shape (Height, Width): {src.shape}")
                print(f"Rows (Height): {src.height}")
                print(f"Cols (Width):  {src.width}")
                
                total_pixels = src.height * src.width
                print(f"Total Pixels: {total_pixels:,}")
                
                # Calculate approximate memory size in RAM for float32 (4 bytes per pixel)
                # Result in Gigabytes (GB)
                size_gb = (total_pixels * 4) / (1024**3)
                print(f"Approximate RAM usage (Float32): {size_gb:.2f} GB")

        except Exception as e:
            print(f"Error reading file: {e}")

    @staticmethod
    def fix_raster_artifacts(raster_path, overwrite=True):
        """
        Universal cleaner for derivative products.
        Fixes:
        1. Infinity (inf) pixels
        2. Deep Negative NoData (-3.4e38)
        3. Massive Positive Artifacts (+3.4e38)
        4. Compresses output (LZW)
        """
        if not os.path.exists(raster_path):
            print(f"File not found: {raster_path}")
            return

        temp_path = raster_path.replace(".tif", "_temp_clean.tif")
        fname = os.path.basename(raster_path)
        
        # Thresholds
        LOWER_LIMIT = -15000.0  
        UPPER_LIMIT = 1e30      

        try:
            with rasterio.open(raster_path) as src:
                meta = src.meta.copy()
                
                # Check if data is floating point (Slope, Aspect, TRI, etc.)
                is_float = np.issubdtype(meta['dtype'], np.floating)
                
                # 1. Update Metadata for Compression and Nodata
                update_params = {
                    'compress': 'lzw',
                    'driver': 'GTiff',
                    'BIGTIFF': 'YES',
                    'tiled': True  # Tiled is better for large rasters
                }

                # Optimization: Add Predictor for better LZW compression
                # 2 = Horizontal differencing (Good for numbers)
                # 3 = Floating point predictor (Best for floats)
                if is_float:
                    update_params['predictor'] = 3
                    update_params['nodata'] = np.nan # Must be NaN for transparency in floats
                else:
                    update_params['predictor'] = 2
                    # Keep original nodata for integers, or default to 0
                    if meta['nodata'] is None:
                        update_params['nodata'] = 0

                meta.update(update_params)
                
                # Determine the fill value to use
                fill_value = meta['nodata']

                with rasterio.open(temp_path, 'w', **meta) as dst:
                    # block_windows allows processing 150GB files without RAM issues
                    for ji, window in src.block_windows(1):
                        # Read RAW data (masked=False is faster and easier to write back)
                        dem_data = src.read(1, window=window)
                        
                        has_changes = False
                        
                        # Only apply float cleaning logic to float rasters
                        if is_float:
                            # 1. Fix Infinity (inf)
                            if np.isinf(dem_data).any():
                                dem_data[np.isinf(dem_data)] = fill_value
                                has_changes = True
                            
                            # 2. Fix Extreme Values (Deep Negative / Massive Positive)
                            # Use errstate to silence warnings about comparing NaNs
                            with np.errstate(invalid='ignore'):
                                # Check Lower Limit (-3.4e38)
                                mask_low = (dem_data < LOWER_LIMIT)
                                if mask_low.any():
                                    dem_data[mask_low] = fill_value
                                    has_changes = True

                                # Check Upper Limit (+3.4e38)
                                mask_high = (dem_data > UPPER_LIMIT)
                                if mask_high.any():
                                    dem_data[mask_high] = fill_value
                                    has_changes = True
                                    
                                # Optional: Fix -9999 if it exists as a raw value
                                mask_9999 = (dem_data == -9999)
                                if mask_9999.any():
                                    dem_data[mask_9999] = fill_value
                                    has_changes = True

                        # Write the cleaned data
                        dst.write(dem_data, 1, window=window)

            # Swap files
            if overwrite:
                gc.collect() # Release file handles
                try:
                    os.remove(raster_path)
                    os.rename(temp_path, raster_path)
                    print(f"Fixed and Compressed: {fname}")
                except OSError as e:
                    print(f"Error overwriting {fname}: {e}. Output at {temp_path}")
            else:
                print(f"Cleaned file saved as: {temp_path}")

        except Exception as e:
            print(f"Error processing {fname}: {e}")
            if os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except:
                    pass

            if overwrite:
                gc.collect() # Release file locks
                try:
                    os.remove(raster_path)
                    os.rename(temp_path, raster_path)
                    print(f"Fixed: {fname}")
                except OSError as e:
                    print(f"Error overwriting {fname}: {e}")
            else:
                print(f"Cleaned file saved as: {temp_path}")

        except Exception as e:
            print(f"Error processing {fname}: {e}")
            if os.path.exists(temp_path):
                os.remove(temp_path)

    @staticmethod
    def compress_tiff_with_rasterio(dem_path):
        """
        Function that replaces nodata values with None, compresses using GDAL BigTIFF, 
        and compresses the DEM file using LZW compression.
        Parameters:
        dem_path (str): Path to the input DEM file.
        Returns:
        output_path (str): Path to the compressed DEM file.
        """
        # Open the input DEM file
        with rasterio.open(dem_path) as src:
            metadata = src.meta.copy()  # Copy metadata
            metadata.update({
                'dtype': 'float32',
                'compress': 'LZW',
                'nodata': None,  # Set no-data value to None (no explicit nodata)
                'driver': 'GTiff',
                'BIGTIFF': 'YES'
            })

            # Define output file path
            output_path = os.path.join(os.path.dirname(dem_path), os.path.basename(dem_path).split(".")[0] + "_compressed.tif")

            # Write the modified DEM to a new file in chunks
            with rasterio.open(output_path, 'w', **metadata) as dst:
                for ji, window in src.block_windows(1):  # Process by blocks
                    dem_data = src.read(1, window=window, masked=True)  # Ensure masked=True
                    dem_data = np.where(np.isnan(dem_data) | np.isinf(dem_data) | (dem_data <= -9999), None, dem_data)
                    dst.write(dem_data.astype('float32'), 1, window=window)
        print(f"Reduced and compressed DEM saved to {output_path}")
        return output_path

    @staticmethod
    def compress_tiff_with_gdal(input_dem_path, out_dem_folder, compress=True):
        """
        Convert a TIFF file to a GDAL raster TIFF using gdal_translate.
        
        :param input_dem_path: Path to the input TIFF file.
        :param output_tiff: Path to save the output raster TIFF file.
        """
        # Open the input TIFF file
        src_ds = gdal.Open(input_dem_path)
        dem_name = Utils.sanitize_path_to_name(input_dem_path)
        output_tiff = os.path.join(out_dem_folder, dem_name+"_gdal.tif")
        # Use gdal_translate to convert the file
        if compress:
            gdal.Translate(output_tiff, src_ds, format='GTiff', creationOptions=['COMPRESS=LZW', 'BIGTIFF=YES'], outputType=gdal.GDT_Float32)
        else:
            gdal.Translate(output_tiff, src_ds, format='GTiff', outputType=gdal.GDT_Float32)
        # Remove the original input TIFF file
        gdal.Dataset.__swig_destroy__(src_ds)  # Close the dataset
        del src_ds  # Delete the dataset reference
        os.remove(input_dem_path)
        print(f"Converted {input_dem_path} to {output_tiff} with compression={compress} as GDAL raster TIFF.")
        return output_tiff
    
    @staticmethod
    def apply_height_to_depth_transformation(input_raster_path, water_elevation=183.6):
        """
        Applies a height to depth transformation on a raster file using rasterio.
        Optimizes the output raster by building GDAL overviews (pyramids).
        """
        # Internal function to check min/max using robust NumPy (No Pylance error)
        def get_min_max_from_numpy(tif_file):
            with rasterio.open(tif_file) as src:
                data = src.read(1) # Read the entire raster band
                nodata = src.nodata
                
                if nodata is not None:
                    # Mask out nodata values using numpy
                    data_masked = np.ma.masked_equal(data, nodata)
                else:
                    data_masked = data
                    
                # Calculate min and max of the valid data
                min_val = data_masked.min()
                max_val = data_masked.max()
                
                if np.ma.is_masked(min_val):
                    return None, None # Entire raster is nodata

                return float(min_val), float(max_val)

        raster_dir = os.path.dirname(input_raster_path)
        raster_name = Utils.sanitize_path_to_name(input_raster_path)
        output_raster_path = os.path.join(raster_dir, f"{raster_name}_depth.tif")

        # Get raster statistics (min/max)
        min_val, max_val = get_min_max_from_numpy(input_raster_path)
        
        # Conditional checks for transformation
        if min_val < 0 and max_val < 0:
            print(f"Warning: Raster values look like depths. Min: {min_val:0.2f}, Max: {max_val:0.2f}. Did not transform from height to depth.")
            return input_raster_path
        elif min_val < 0 and max_val > 0:
            print(f"Warning: Raster DEM values range from MIN: {min_val:0.2f}, Max: {max_val:0.2f}. Did not transform from height to depth.")
            return input_raster_path
        else:
            # Use rasterio to apply the transformation in blocks
            with rasterio.open(input_raster_path) as src:
                profile = src.profile.copy()
                # Set the output data type to float32 for precision in depth
                profile.update({'compress': 'LZW', 'dtype': np.float32})
                
                with rasterio.open(output_raster_path, 'w', **profile) as dst:
                    for ji, window in src.block_windows(1):
                        arr = src.read(1, window=window)
                        # Use numpy's where to apply transformation only to valid data
                        transformed = np.where(
                            arr == src.nodata, 
                            src.nodata, 
                            (water_elevation - arr) * -1
                        )
                        dst.write(transformed, 1, window=window)
                        
            # --- File Lock Wait (Optional but recommended) ---
            time.sleep(2) 

            # --- GDAL/rasterio Optimization: Build Pyramids (Overviews) ---
            # This replaces the ArcPy BuildPyramids function and prevents the long 'verification' delays.
            print("Transform complete. Building GDAL Overviews (Pyramids) for optimization...")
            
            # Define the overview levels (e.g., powers of 2 up to 128)
            overviews = [2, 4, 8, 16, 32, 64, 128]
            
            # Open the output raster in 'r+' mode to add overviews and tags
            with rasterio.open(output_raster_path, 'r+') as dst:
                # Build the overviews using Nearest Neighbor resampling (fastest)
                dst.build_overviews(overviews, Resampling.nearest)
                
                # Update tags to mark the overviews as built and define the resampling method
                dst.update_tags(ns='rio_overview', **{'resampling': 'nearest'})
            
            print(f"Transformed raster saved to {output_raster_path}")
            return output_raster_path
    
    def convert_to_cog(input_path, output_path=None, compress="DEFLATE", tile_size=512):
        input_path = Path(input_path)
        if output_path is None:
            output_path = input_path.with_suffix(".cog.tif")
        else:
            output_path = Path(output_path)

        cmd = [
            "gdal_translate",
            str(input_path),
            str(output_path),
            "-of", "GTiff",
            "-co", "TILED=YES",
            "-co", f"COMPRESS={compress}",
            "-co", f"BLOCKXSIZE={tile_size}",
            "-co", f"BLOCKYSIZE={tile_size}",
            "-co", "COPY_SRC_OVERVIEWS=YES",
            "-co", "BIGTIFF=YES"
        ]

        print("Running command:")
        print(" ".join(cmd))
        subprocess.run(cmd, check=True)

        print(f"\nConversion complete: {output_path}")

    @staticmethod
    def merge_raster_gdal(raster_chunks_path, output_format="LZW", tile_size=512, remove=False):
        """
        Merges DEM files in the specified path using GDAL.
        Supports output as LZW-compressed TIFF or Deflate-compressed COG.
        Removes individual tiles after merge if requested.
        Replaces NaN and infinite values with nodata (-9999) and updates metadata.

        Parameters:
            dem_chunks_path (str): Path to folder containing DEM tiles.
            remove (bool): Whether to delete individual tiles after merge.
            output_format (str): "LZW" or "COG"
            tile_size (int): Tile size for COG output (default: 512)
        """

        dem_name = os.path.basename(raster_chunks_path)
        dem_files = glob.glob(os.path.join(raster_chunks_path, "*.tif"))

        if not dem_files:
            print(f"No DEM files found in the path: {raster_chunks_path}")
            return None
        
        # projections = set()
        # for tif in dem_files:
        #     ds = gdal.Open(tif)
        #     if ds:
        #         projections.add(ds.GetProjection())
        # if len(projections) > 1:
        #     print("Warning: DEM tiles have inconsistent projections.")


        # Build VRT
        vrt_options = gdal.BuildVRTOptions(resampleAlg='bilinear')
        vrt_dataset = gdal.BuildVRT('/vsimem/mosaic.vrt', dem_files, options=vrt_options)
        if vrt_dataset is None or vrt_dataset.RasterCount == 0:
            print("VRT creation failed or resulted in empty dataset.")
            return None

        # Define output path
        suffix = ".cog.tif" if output_format.upper() == "COG" else ".tif"
        merged_dem_path = os.path.join(os.path.dirname(raster_chunks_path), f"{dem_name}_merged{suffix}")

        # Set creation options
        if output_format.upper() == "COG":
            creation_options = [
                "TILED=YES",
                "COMPRESS=DEFLATE",
                f"BLOCKXSIZE={tile_size}",
                f"BLOCKYSIZE={tile_size}",
                "COPY_SRC_OVERVIEWS=YES",
                "BIGTIFF=YES"
            ]
        elif output_format.upper() == "LZW":
            creation_options = ["COMPRESS=LZW", "BIGTIFF=YES"]
        else:
            raise ValueError("output_format must be 'COG' or 'LZW'")

        print(f"DEM files to merge: {dem_files}")
        print(f"Output merged DEM path: {merged_dem_path}")

        # Translate VRT to final output
        try:
            result = gdal.Translate(
                merged_dem_path,
                vrt_dataset,
                format='GTiff',
                creationOptions=creation_options,
                noData=-9999
            )
        except Exception as e:
            print(f"gdal.Translate threw an exception: {e}")
            return None

        # Clean up VRT
        vrt_dataset = None

        if result is None:
            print(f"gdal.Translate failed to create output: {merged_dem_path}")
            return None

        # Confirm output file exists
        if not os.path.isfile(merged_dem_path):
            print(f"Output file was not created: {merged_dem_path}")
            return None

        if remove:
            for file in dem_files:
                os.remove(file)
            print("Individual DEM tiles have been removed.")

        print(f"Merged DEM saved at: {merged_dem_path}")
        return merged_dem_path
    
    def build_cog_overviews(raster_path, levels=[2, 4, 8, 16, 32], resampling='nearest'):
        """
        Builds external overviews (.ovr) for a Cloud Optimized GeoTIFF (COG).

        Parameters:
            raster_path (str): Path to the input COG raster file.
            levels (list): List of overview levels to generate.
            resampling (str): Resampling method (e.g., 'average', 'nearest', 'mode').

        Returns:
            None
        """
        if not os.path.isfile(raster_path):
            raise FileNotFoundError(f"Raster not found: {raster_path}")

        # Open in read-only mode to force external .ovr creation
        dataset = gdal.Open(raster_path, gdal.GA_ReadOnly)
        if not dataset:
            raise RuntimeError(f"Failed to open raster: {raster_path}")

        # Build external overviews
        result = dataset.BuildOverviews(resampling, levels)
        if result != 0:
            raise RuntimeError(f"Overview generation failed for {raster_path}")

        print(f"External overviews (.ovr) created for {raster_path} using {resampling} resampling.")

        dataset = None

    @staticmethod
    def validate_rasters(raster_list):
        results = {}
        for tif in raster_list:
            try:
                ds = gdal.Open(tif)
                if ds is None:
                    results[tif] = "Unreadable"
                else:
                    # Optional: check for raster bands
                    if ds.RasterCount == 0:
                        results[tif] = "No raster bands"
                    else:
                        results[tif] = "Valid"
            except Exception as e:
                results[tif] = f"Error: {str(e)}"
        return results
    
    def read_raster(raster_path):
        """
        Reads a raster file and returns its data as a numpy array.
        
        :param raster_path: Path to the raster file.
        :return: Numpy array of raster data.
        """
        with rasterio.open(raster_path) as src:
            return src.read(1, masked=True)  # Read the first band with masking
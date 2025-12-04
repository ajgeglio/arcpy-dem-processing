import arcpy
from arcpy.sa import Raster, SetNull, Con, IsNull, Minus, Times # Make sure 'Minus' and 'Raster' are imported
import os
from utils import Utils  # Assuming Utils is defined in src/utils.py
import shutil
import time
import shutil
import gc

class ArcpyUtils:

    @staticmethod
    def apply_height_to_depth_transformation(input_raster_path, water_elevation=183.6):
        """
        Applies a height to depth transformation on a raster file.
        Uses arcpy to get raster statistics (min/max) to avoid loading large rasters into memory.
        If all values are positive, uses rasterio to convert to depth.
        Returns the path to the transformed raster, or the original path if no transformation is needed.
        """
        # Check if the raster exists and is valid
        if not arcpy.Exists(input_raster_path):
            raise FileNotFoundError(f"Input raster not found at: {input_raster_path}")
        raster_dir = os.path.dirname(input_raster_path)
        raster_name = Utils.sanitize_path_to_name(input_raster_path)
        output_raster_path = os.path.join(raster_dir, f"{raster_name}_depth.tif")

        # Use arcpy to get raster statistics (min/max)
        arcpy.env.overwriteOutput = True
        # --- FIX: Ensure Statistics Exist ---
        try:
            # Attempt to read statistics first
            min_val = float(arcpy.management.GetRasterProperties(input_raster_path, "MINIMUM").getOutput(0))
        except arcpy.ExecuteError as e:
            # If statistics are missing (ERROR 001100), generate them
            if "001100" in str(e):
                print(f"Statistics missing for {input_raster_path}. Calculating statistics...")
                
                # Force calculation of statistics
                arcpy.management.CalculateStatistics(input_raster_path)
                
                # Re-read the statistics after generation
                min_val = float(arcpy.management.GetRasterProperties(input_raster_path, "MINIMUM").getOutput(0))
            else:
                # Re-raise other unexpected ArcPy errors
                raise
        # Now, read the maximum value, which should work if min_val succeeded
        max_val = float(arcpy.management.GetRasterProperties(input_raster_path, "MAXIMUM").getOutput(0))

        if min_val < 0 and max_val < 0:
            print(f"Warning: Raster values look like depths. Min: {min_val:0.2f}, Max: {max_val:0.2f}. Did not transform from height to depth.")
            return input_raster_path
        elif min_val < 0 and max_val > 0:
            print(f"Warning: Raster DEM values range from MIN: {min_val:0.2f}, Max: {max_val:0.2f}. Did not transform from height to depth.")
            return input_raster_path
        else:
            print("Converting raster using ArcPy Spatial Analyst...")

            #Set Compression and Pyramid Environment ---
            original_compression = arcpy.env.compression
            original_pyramid = arcpy.env.pyramid
            
            # Set desired compression for the output TIF
            arcpy.env.compression = "LZW" 
            
            # Prevent automatic pyramid building during .save() to allow custom, fast building later
            arcpy.env.pyramid = "NONE" 
            # --- END OF FIX ---

            # 1. Read the input raster
            in_raster = Raster(input_raster_path)
            
            # 2. Perform the calculation (Depth = (Water_Elevation - Height) * -1)
            transformed_raster = Minus(water_elevation, in_raster) * -1
            
            # 3. Save the output. This will now use LZW compression.
            transformed_raster.save(output_raster_path)
            
            # 4. Clean up environment settings immediately after save
            arcpy.env.compression = original_compression
            arcpy.env.pyramid = original_pyramid

            print("Transform complete. Calculating statistics and building pyramids for ArcPy optimization...")
            
            # 1. Calculate Statistics
            arcpy.management.CalculateStatistics(output_raster_path)
            
            # 2. Build Pyramids (using the optimized NEAREST technique)
            ArcpyUtils.build_pyramids(output_raster_path) 
            
            print(f"Transformed raster saved to {output_raster_path}")
            return output_raster_path

    @staticmethod
    def merge_dem_arcpy(dem_chunks_folder, remove_chunks=True):
        """
        Mosaics multiple DEM chunks from a folder into a single raster using arcpy.management.MosaicToNewRaster.

        Args:
            dem_chunks_folder (str): Path to the folder containing the DEM chunks.
            output_merged_dem_path (str): The full path including filename for the merged DEM output.
            spatial_ref: The spatial reference object for the output raster.
            cell_size: The cell size for the output raster.
            remove_chunks (bool): If True, removes the original chunk files folder after merging.

        Returns:
            str: Path to the merged DEM raster.
        """

        output_merged_dem_path = os.path.join(os.path.dirname(dem_chunks_folder), os.path.basename(dem_chunks_folder)+".tif")
        raster_files = []
        if not os.path.isdir(dem_chunks_folder):
            raise ValueError(f"Input chunk folder does not exist: {dem_chunks_folder}")

        for root, dirs, files in os.walk(dem_chunks_folder):
            for file in files:
                # Check for common raster extensions
                if file.lower().endswith((".tif", ".tiff", ".img", ".jp2", ".png")):
                    raster_files.append(os.path.join(root, file))

        if not raster_files:
            raise ValueError(f"No raster files found in the specified chunk directory: {dem_chunks_folder}")
        
        # using the first file to get properties
        raster = arcpy.Raster(raster_files[0])
        number_of_bands = raster.bandCount
        spatal_ref = raster.spatialReference
        cell_size = raster.meanCellWidth
        print("number of bands", number_of_bands)
        print("spatial reference", spatal_ref.name)
        print("cell size", cell_size)
        pixel_type_lookup = {
            "U1": "1_BIT", "U8": "8_BIT_UNSIGNED", "S16": "16_BIT_SIGNED",
            "U16": "16_BIT_UNSIGNED", "F32": "32_BIT_FLOAT", "F64": "64_BIT"
        }
        pixel_type = pixel_type_lookup.get(raster.pixelType, "32_BIT_FLOAT")  # Default fallback
        print("pixel type", pixel_type)
        # Ensure the output directory exists for the merged DEM
        output_dir = os.path.dirname(output_merged_dem_path)
        output_name = os.path.basename(output_merged_dem_path)

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        print(f"Merging {len(raster_files)} DEM chunks from '{dem_chunks_folder}' to '{output_merged_dem_path}'...")

        try:
            arcpy.management.MosaicToNewRaster(
                input_rasters=raster_files,
                output_location=output_dir,
                raster_dataset_name_with_extension=output_name,
                coordinate_system_for_the_raster=spatal_ref,
                cellsize=cell_size,
                pixel_type=pixel_type,
                number_of_bands=number_of_bands,
                mosaic_method="LAST",  # Use BLEND to avoid seams
                mosaic_colormap_mode="REJECT"
            )
            print(f"Successfully merged raster chunks to: {output_merged_dem_path}")
        except arcpy.ExecuteError:
            messages = arcpy.GetMessages(2)
            print(f"ArcPy Execute Error during mosaic of DEM chunks: {messages}")
            raise
        except Exception as e:
            print(f"An unexpected error occurred during mosaic of DEM chunks: {e}")
            raise
        finally:

            if remove_chunks:
                print(f"Removing temporary DEM chunks folder: {dem_chunks_folder}")
                try:
                    shutil.rmtree(dem_chunks_folder)
                    print("Temporary DEM chunks folder removed.")
                except Exception as e:
                    print(f"Error removing temporary DEM chunks folder '{dem_chunks_folder}': {e}")

        return output_merged_dem_path
    
    @staticmethod
    def compress_raster(input_raster_path, format="TIFF", overwrite=True):
        """
        Compress a large TIFF file using arcpy, reducing file size without changing spatial reference.
        Supports COG with compression if format="COG".
        """
        out_raster_folder = os.path.dirname(input_raster_path)
        arcpy.env.workspace = out_raster_folder
        arcpy.env.compression = "LZW" if format == "TIFF" else "NONE"  # Set compression type

        raster = arcpy.Raster(input_raster_path)
        input_ext = os.path.splitext(input_raster_path)[1][1:].lower()  # e.g., 'tif'
        pixel_type_lookup = {
            "U1": "1_BIT", "U8": "8_BIT_UNSIGNED", "S16": "16_BIT_SIGNED",
            "U16": "16_BIT_UNSIGNED", "F32": "32_BIT_FLOAT", "F64": "64_BIT"
        }
        pixel_type = pixel_type_lookup.get(raster.pixelType, "32_BIT_FLOAT")
        print("pixel type", pixel_type)

        raster_name = Utils.sanitize_path_to_name(input_raster_path)
        # Determine output extension and name logic
        if format.lower() == "tiff":
            ext = "tif"
        else:
            ext = format.lower()

        # Always use the sanitized raster_name for output
        # If input is .tif and output is .tif, add _compress
        if input_ext == "tif" and ext == "tif":
            output_raster = os.path.join(os.path.dirname(input_raster_path), raster_name + "_compress.tif")
        else:
            output_raster = os.path.join(os.path.dirname(input_raster_path), raster_name + "." + ext)

        print(f"Compressing {input_raster_path} to {output_raster} with format={format} using arcpy.")
        arcpy.env.overwriteOutput = True

        arcpy.management.CopyRaster(
            in_raster=input_raster_path,
            out_rasterdataset=output_raster,
            background_value="",
            nodata_value="",
            onebit_to_eightbit="NONE",
            colormap_to_RGB="NONE",
            pixel_type=pixel_type,
            scale_pixel_value="NONE",
            RGB_to_Colormap="NONE",
            format=format,
            transform="NONE"
        )
        if arcpy.Exists(output_raster):
            print(f"Raster compression successful: {output_raster}")
            if overwrite:
                try:
                    import gc
                    gc.collect()
                    if arcpy.Exists(input_raster_path):
                        arcpy.Delete_management(input_raster_path)
                    os.rename(output_raster, input_raster_path)
                    print(f"Original raster overwritten: {input_raster_path}")
                    return input_raster_path
                except Exception as e:
                    print(f"Failed to overwrite original raster: {e}")
                    return output_raster
            else:
                return output_raster
        else:
            print(f"Raster compression failed: {output_raster} does not exist after operation.")
            return None
    
    @staticmethod
    def create_valid_data_mask(input_raster, output_mask_path=None):
        """
        Generate a binary mask for the input raster: 1 for valid data, NoData.
        If output_mask_path is None, saves to temp folder.
        Returns the path to the binary mask raster.
        """
        if output_mask_path is None:
            name = Utils.sanitize_path_to_name(input_raster)
            output_mask_path = os.path.join(os.path.dirname(input_raster), f"{name}_valid_mask.tif")
        # Remove if exists
        if os.path.exists(output_mask_path):
            arcpy.Delete_management(output_mask_path)
        mask_raster = arcpy.sa.Con(arcpy.sa.IsNull(arcpy.Raster(input_raster)), 0, 1)
        # Convert binary mask to 1, NoData
        nodata_mask = SetNull(mask_raster == 0, mask_raster)
        nodata_mask.save(output_mask_path)
        return output_mask_path
    
    def create_binary_mask(input_raster, data_value=1, nodata_value=0):
        """
        Creates a binary mask raster where valid data is assigned 'data_value' (e.g., 1) 
        and NoData areas are assigned 'nodata_value' (e.g., 0).
        
        Args:
            input_raster_path (str): Path to the input DEM or raster.
            output_mask_path (str): Path to save the output binary mask raster.
            data_value (int): The value to assign to valid data areas.
            nodata_value (int): The value to assign to the raster's original NoData areas.
        """
        
        arcpy.env.overwriteOutput = True
        
        # 1. Read the input raster
        in_raster = Raster(input_raster)

        # 2. Check for Null (NoData) values using IsNull
        # This creates a temporary boolean raster: 1 where NoData, 0 where data exists.
        is_null_raster = IsNull(in_raster)
        
        # 3. Use Con (Conditional) to assign the final values
        # The logic is: 
        #   IF is_null_raster == 1 (i.e., IF the original pixel was NoData), assign nodata_value (0).
        #   ELSE (i.e., IF the original pixel was Data), assign data_value (1).
        binary_mask = Con(is_null_raster, nodata_value, data_value)
        
        # 4. Save the final output raster
        # Note: Use a low-memory integer type (e.g., "8_BIT_UNSIGNED") for binary masks
        name = Utils.sanitize_path_to_name(input_raster)
        output_mask_path = os.path.join(os.path.dirname(input_raster), "boundary_files", f"{name}_binary_mask.tif") 
        os.makedirs(os.path.dirname(output_mask_path), exist_ok=True)  # Ensure the directory exists
        binary_mask.save(output_mask_path)
        
        print(f"✅ Binary mask created and saved to: {output_mask_path}")
        return output_mask_path
    
    def mask_intersector(masks):
        """

        """
        intersection_mask = None
        output_mask_dir = os.path.dirname(os.path.dirname(masks[0]))
        output_mask_name = "valid_data_intersection_mask.tif"
        output_mask_path = os.path.join(output_mask_dir, output_mask_name)

        for i, mask_path in enumerate(masks):
            if not arcpy.Exists(mask_path):
                arcpy.AddWarning(f"Raster not found: {mask_path}. Skipping this raster for intersection mask.")
                continue

            # Create a binary mask for the current raster (1 for valid, 0 for NoData)
            current_binary_mask = Raster(mask_path)

            if intersection_mask is None:
                # For the first valid raster, initialize the intersection mask
                intersection_mask = current_binary_mask
            else:
                # For subsequent rasters, multiply with the cumulative intersection mask
                intersection_mask = intersection_mask * current_binary_mask

            # Release the current_raster object early to manage memory
            del current_binary_mask

            arcpy.ClearWorkspaceCache_management()

        # After the loop, intersection_mask will be a raster where:
        # - Any pixel that was NoData in *any* of the input rasters will be 0.
        # - Any pixel that was valid data in *all* input rasters will be 1.

        if intersection_mask is None:
            arcpy.AddWarning("No valid rasters found to create an intersection mask.")
            return None

        # Convert the binary mask (0s and 1s) to 1s and NoData
        # Set pixels where intersection_mask is 0 (meaning NoData in at least one input) to NoData.
        # Otherwise, keep the value (which will be 1).
        valid_intersection_mask_final = SetNull(intersection_mask == 0, intersection_mask)

        # It's good practice to ensure the final output is also cleared from memory if not immediately used
        del intersection_mask
        arcpy.ClearWorkspaceCache_management()
        valid_intersection_mask_final.save(output_mask_path)
        print(f"Valid data intersection mask saved to: {output_mask_path}")
        return output_mask_path
    
    @staticmethod
    def transform_spatial_reference_arcpy(base_raster, transform_raster, save_path=None):
        """
        Transforms the backscatter to the spatial reference used by the input_dem.
        Returns the path to the transformed backscatter raster.
        """
        # Get spatial reference from input_dem
        base_desc = arcpy.Describe(base_raster)
        base_sr = base_desc.spatialReference
        base_cell_size = arcpy.management.GetRasterProperties(base_raster, "CELLSIZEX").getOutput(0)

        tranform_raster_name = Utils.sanitize_path_to_name(transform_raster)
        transform_raster_path = os.path.dirname(transform_raster)

        if save_path is None:
            # Define output path
            transformed_path = os.path.join(transform_raster_path, tranform_raster_name+"_transformed.tif")
        else:
            transformed_path = os.path.join(save_path, tranform_raster_name+"_transformed.tif")

        # Project the input_bs to match the DEM's spatial reference
        arcpy.management.ProjectRaster(
            in_raster=base_raster,
            out_raster=transformed_path,
            out_coor_system=base_sr,
            resampling_type="NEAREST",
            cell_size=base_cell_size
        )
        return transformed_path
    
    @staticmethod
    def align_rasters(base_raster, transform_raster, replace=True, save_path=None):
        """
        Aligns the pixels of one raster (base_raster) to another (reference_raster)
        without changing the spatial resolution of the base_raster.

        Args:
            transform_raster (str): Path to the raster whose pixels need to be aligned.
                               This raster's spatial resolution will be preserved.
            base_raster (str): Path to the raster whose pixel alignment and
                                    spatial reference will be used for snapping.
            save_path (str, optional): Directory to save the aligned raster.
                                       If None, it will be saved in the same
                                       directory as base_raster.

        Returns:
            str: Path to the aligned raster.
        """
        arcpy.env.overwriteOutput = True # Allow overwriting existing files

        # Get spatial reference from the reference_raster
        reference_desc = arcpy.Describe(base_raster)
        reference_sr = reference_desc.spatialReference
        base_cell_size_x = reference_desc.meanCellWidth
        base_cell_size_y = reference_desc.meanCellHeight

        transform_raster_name = Utils.sanitize_path_to_name(transform_raster)
        transform_raster_dir = os.path.dirname(transform_raster)

        if save_path is None:
            # Define output path in the same directory as base_raster
            aligned_raster_path = os.path.join(transform_raster_dir, transform_raster_name + "_aligned.tif")
        else:
            # Define output path in the specified save_path
            aligned_raster_path = os.path.join(save_path, transform_raster_name + "_aligned.tif")

        # Set the snap raster environment. This is crucial for alignment.
        # The output raster's extent and cell origin will snap to this raster.
        arcpy.env.snapRaster = base_raster

        arcpy.management.ProjectRaster(
            in_raster=transform_raster,
            out_raster=aligned_raster_path,
            out_coor_system=reference_sr,
            resampling_type="NEAREST",
            cell_size=f"{base_cell_size_x} {base_cell_size_y}" # Preserve original cell size
            )


        # Check if the alignment was successful
        if not arcpy.Exists(aligned_raster_path):
            raise Exception(f"Failed to align raster: {transform_raster} to reference raster: {base_raster}")
        
        if replace:
            # Release file handles before deleting or renaming
            arcpy.ClearWorkspaceCache_management()

            # Replace the original transform_raster with the aligned raster
            try:
                arcpy.management.Delete(transform_raster)
            except Exception:
                print(f"cannot remove {transform_raster} is in use")


            arcpy.management.Rename(aligned_raster_path, transform_raster)
            aligned_raster_path = transform_raster     
        else:
            # If not replacing, just return the aligned raster path
            print(f"Aligned raster saved at: {aligned_raster_path}")
        return aligned_raster_path
    

    @staticmethod
    def con_fill(base_raster, transform_raster):
        """
        Fills NoData values in transform_raster where base_raster has valid data.
        A new file is only created if actual missing data is found and filled.

        Args:
            base_raster (str): Path to the base raster (reference for valid areas).
            transform_raster (str): Path to the raster to be filled/corrected.

        Returns:
            str: Path to the modified (or original if no changes) raster.
        """
        arcpy.env.overwriteOutput = True # Ensure overwrite is on

        # Input Raster objects
        input_dem_obj = Raster(base_raster)
        geomorphon_raster_obj = Raster(transform_raster)

        # 1. Check for missing data in transform_raster relative to base_raster
        # We need to count pixels where transform_raster is NoData AND base_raster is valid.
        
        # Create a boolean raster where 1 indicates a pixel needs filling, 0 otherwise
        needs_filling_raster = IsNull(geomorphon_raster_obj) & ~IsNull(input_dem_obj)
        
        # Sum the values in the needs_filling_raster to count how many pixels need filling
        # The output will be a single-cell raster containing the count.
        try:
            # Use Zonal Statistics to get the sum of the 'needs_filling_raster' over its entire extent.
            # Convert needs_filling_raster to an integer raster first if it's not already,
            # as ZonalStatistics requires integer zones. Multiplying by 1 ensures this.
            count_raster = arcpy.sa.ZonalStatisticsAsTable(
                in_zone_data=needs_filling_raster * 1, # Convert boolean to integer (0 or 1)
                zone_field="Value", # The field containing the values (0 or 1)
                in_value_raster=needs_filling_raster * 1, # The raster to calculate statistics from
                out_table="in_memory/temp_count_table", # Use in-memory table for efficiency
                ignore_nodata="DATA", # Only consider data cells
                statistics_type="SUM" # We want the sum of 1s (missing pixels)
            )

            # Read the sum from the in-memory table
            count = 0
            with arcpy.da.SearchCursor(count_raster, ["SUM"]) as cursor:
                for row in cursor:
                    count = row[0]
            
            # Clean up the in-memory table
            arcpy.management.Delete("in_memory/temp_count_table")

        except Exception as e:
            print(f"Error checking for missing data: {e}")
            print("Proceeding with fill operation to be safe.")
            count = 1 # Force fill if check fails

        if count == 0:
            print(f"No missing data in '{transform_raster}' relative to '{base_raster}'. No new file created.")
            # Release raster objects
            del input_dem_obj
            del geomorphon_raster_obj
            del needs_filling_raster
            arcpy.ClearWorkspaceCache_management()
            return transform_raster
        else:
            print(f"Found {int(count)} pixels in '{transform_raster}' that need filling relative to '{base_raster}'.")
            # Define the output path for the filled raster
            raster_name = os.path.basename(transform_raster).split('.')[0]
            con_filled_raster_path = os.path.join(os.path.dirname(transform_raster), f"{raster_name}_con_filled.tif")

            # The core Con logic:
            # Condition: Is the pixel in geomorphon_raster_obj NoData AND is the corresponding pixel in input_dem_obj valid?
            filled_raster = Con(
                IsNull(geomorphon_raster_obj) & ~IsNull(input_dem_obj),
                1, # If the condition is TRUE, fill with the value 1 (as per your original code)
                geomorphon_raster_obj # If the condition is FALSE, keep the original geomorphon value
            )

            # Save the filled raster
            filled_raster.save(con_filled_raster_path)

            # Release the raster objects to prevent file locks
            del input_dem_obj
            del geomorphon_raster_obj
            del filled_raster
            del needs_filling_raster # Also delete this temporary raster object
            arcpy.ClearWorkspaceCache_management() # Clear cache to release any remaining locks

            # Replace the original transform_raster with the con-filled one
            if arcpy.Exists(con_filled_raster_path):
                print(f"Con-filled raster saved at: {con_filled_raster_path}")
                try:
                    # Use CopyRaster for robust overwriting, which handles file locks better
                    arcpy.management.CopyRaster(con_filled_raster_path, transform_raster,
                                                "", "", "0", "NONE", "NONE", "NONE", "NONE", "NONE", "GRID")
                    print(f"Successfully replaced original raster with con-filled raster at: {transform_raster}")
                    # Clean up the temporary filled raster file after successful copy
                    arcpy.management.Delete(con_filled_raster_path)
                    return transform_raster
                except Exception as e:
                    print(f"Error replacing original raster with con-filled raster: {e}")
                    print(f"Original raster: {transform_raster} might still be in use. Final output is at {con_filled_raster_path}")
                    return con_filled_raster_path
            else:
                print(f"ERROR: Con-filled raster was not created at: {con_filled_raster_path}")
                return transform_raster
            
    @staticmethod
    def describe_raster_env(tif_path):
        desc = arcpy.Describe(tif_path)

        info = {
            "spatial_reference": desc.spatialReference.name,
            "cell_size_x": desc.meanCellWidth,
            "cell_size_y": desc.meanCellHeight,
            "extent": {
                "XMin": desc.extent.XMin,
                "YMin": desc.extent.YMin,
                "XMax": desc.extent.XMax,
                "YMax": desc.extent.YMax
            },
            "pixel_type": desc.pixelType,
            "band_count": desc.bandCount,
            "format": desc.format,
            "compression_type": getattr(desc, "compressionType", "Unknown")
        }

        return info

    def build_pyramids(raster_file, pyramid_level="", skip_first="NONE", resample_technique="NEAREST"):
        """
        Builds pyramids for a raster file using arcpy.management.BuildPyramids.
        pyramid_level: Specify the level of pyramids to build, or leave empty for default.
        skip_first: Specify whether to skip the first level of pyramids.
        resample_technique: Resampling technique to use, e.g., "NEAREST", "BILINEAR", "CUBIC".
        """
        arcpy.management.BuildPyramids(
            raster_file,
            pyramid_level=pyramid_level,
            SKIP_FIRST=skip_first,
            resample_technique=resample_technique
        )
        print(f"Pyramids built for {raster_file} with level={pyramid_level}, skip_first={skip_first}, resample_technique={resample_technique}.")

    @staticmethod
    # Helper function to force delete stubborn folders
    def cleanup_chunks(folder_path):
        if not os.path.exists(folder_path):
            return
        
        # 1. Release ArcPy locks
        arcpy.ClearWorkspaceCache_management()
        
        # 2. Force Python garbage collection to release file handles
        gc.collect()
        
        # 3. Try to delete with retries
        for attempt in range(5):
            try:
                shutil.rmtree(folder_path)
                break # Success
            except PermissionError:
                # Wait a moment for the OS to release the file handle
                time.sleep(1.0) 
            except Exception as e:
                print(f"Warning: Could not delete {folder_path}: {e}")
                break
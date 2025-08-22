import arcpy
import os
from arcpy.sa import Con, IsNull, Raster, SetNull
from arcpy import Extent # Import Extent object for manipulation
from metafunctions import MetaFunctions
from utils import Utils # Assuming Utils contains necessary utility functions

class RasterMasking:
    @staticmethod
    def _intersect_extents(ext1, ext2):
        """
        Returns the intersection of two arcpy.Extent objects, or None if they do not overlap.
        """
        xmin = max(ext1.XMin, ext2.XMin)
        xmax = min(ext1.XMax, ext2.XMax)
        ymin = max(ext1.YMin, ext2.YMin)
        ymax = min(ext1.YMax, ext2.YMax)
        if xmin >= xmax or ymin >= ymax:
            return None
        return Extent(xmin, ymin, xmax, ymax)

    @staticmethod
    def return_valid_data_mask_intersection(input_rasters, replace=True):
        """
        Generates a binary mask representing the intersection of valid data areas
        from a list of input rasters, limited to the common spatial extent.
        Returns a raster where 1 indicates valid data in all input rasters,
        and NoData otherwise.
        """
        if not input_rasters:
            arcpy.AddWarning("Input raster list is empty. Cannot create intersection mask.")
            return None

        arcpy.env.overwriteOutput = True

        valid_rasters = []
        common_extent = None
        first_valid_raster_obj = None
        intersection_mask = None

        print("Starting raster processing loop...")
        for raster_path in input_rasters:
            if not arcpy.Exists(raster_path):
                arcpy.AddWarning(f"Raster not found: {raster_path}. Skipping for intersection mask.")
                continue

            desc = arcpy.Describe(raster_path)
            current_raster_obj = Raster(raster_path)
            current_extent = desc.extent
            base_cell_size_x = desc.meanCellWidth
            base_cell_size_y = desc.meanCellHeight
            cell_size = f"{base_cell_size_x} {base_cell_size_y}" # Change cell size to the raster being processed
            # arcpy.env.cellSize = cell_size

            if first_valid_raster_obj is None:
                common_extent = current_extent
                first_valid_raster_obj = current_raster_obj
                first_valid_raster_coordinate_system = desc.spatialReference
                # Set snapRaster and output coordinate system to first valid raster
                arcpy.env.snapRaster = raster_path
                arcpy.env.outputCoordinateSystem = first_valid_raster_coordinate_system

            else:
                common_extent = RasterMasking._intersect_extents(common_extent, current_extent)
                if common_extent is None:
                    arcpy.AddWarning("Input rasters do not have any overlapping extent. Cannot create intersection mask.")
                    del current_raster_obj
                    return None
                # align the current raster to the first valid raster's coordinate system
                current_aligned_raster_path = os.path.join(
                    os.path.dirname(raster_path), Utils.sanitize_path_to_name(raster_path) + "_aligned.tif")
                backup_raster_path = os.path.join(
                    os.path.dirname(raster_path), Utils.sanitize_path_to_name(raster_path) + "_original.tif")
                arcpy.management.ProjectRaster(
                    in_raster=raster_path,
                    out_raster=current_aligned_raster_path,
                    out_coor_system=first_valid_raster_coordinate_system,
                    resampling_type="NEAREST",
                    cell_size=cell_size
                    )
                        # Check if the alignment was successful
                if not arcpy.Exists(current_aligned_raster_path):
                    raise Exception(f"Failed to align raster: {raster_path} to reference raster")
                if replace:
                    # only backup the original raster once
                    if arcpy.Exists(backup_raster_path):
                        arcpy.management.Delete(raster_path)
                    else:
                        arcpy.management.Rename(raster_path, backup_raster_path)
                    arcpy.management.Rename(current_aligned_raster_path, raster_path)
                else:
                    raster_path = current_aligned_raster_path
                del current_raster_obj
            valid_rasters.append(raster_path)

        if not valid_rasters:

            print("No valid rasters found after processing input list.")
            arcpy.AddWarning("No valid rasters found to create an intersection mask.")
            return None

        arcpy.env.extent = common_extent
        intersection_mask = Con(IsNull(Raster(valid_rasters[0])), 0, 1)

        for raster_path in valid_rasters[1:]:
            try:
                # Create binary mask for current raster
                current_raster_obj = Raster(raster_path)
                current_binary_mask = Con(IsNull(current_raster_obj), 0, 1)
                message = f"Processing raster: {raster_path}"
                message_length = Utils.print_progress(message)
                intersection_mask = intersection_mask * current_binary_mask

                del current_raster_obj
                del current_binary_mask

            except arcpy.ExecuteError as e:
                arcpy.AddWarning(f"Error describing or processing raster {raster_path}: {e}. Skipping.")
                continue
            except Exception as e:
                arcpy.AddWarning(f"Unexpected error with raster {raster_path}: {e}. Skipping.")
                continue

        # Save the final intersection mask after processing all rasters
        output_mask_dir = os.path.dirname(valid_rasters[0])
        output_mask_name = "all_raster_intersection_mask.tif"
        output_mask_path = os.path.join(output_mask_dir, output_mask_name)
        intersection_mask_final = SetNull(intersection_mask == 0, intersection_mask)
        message = f"Saving intersection mask to: {output_mask_path}"
        message_length = Utils.print_progress(message, previous_length=message_length)
        intersection_mask_final.save(output_mask_path)
        binary_mask_filled_path = MetaFunctions.fill_mask(output_mask_path)
        binary_mask_filled_object = Raster(binary_mask_filled_path)
        intersection_mask_final = SetNull(binary_mask_filled_object == 0, binary_mask_filled_object)
        intersection_mask_final.save(output_mask_path)
        arcpy.Delete_management(binary_mask_filled_path)
        # Clean up environment settings and objects
        arcpy.env.extent = "DEFAULT" # Reset extent
        arcpy.env.snapRaster = ""   # Clear snapRaster
        arcpy.ClearWorkspaceCache_management()
        if 'intersection_mask' in locals() and intersection_mask is not None:
            del intersection_mask
        if 'intersection_mask_final' in locals() and intersection_mask_final is not None:
            del intersection_mask_final

        return output_mask_path, valid_rasters


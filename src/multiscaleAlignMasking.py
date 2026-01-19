import arcpy
import os
from arcpy.sa import Con, IsNull, Raster, SetNull
from arcpy import Extent
# Assuming MetaFunctions and Utils are in your path
# from metafunctions import MetaFunctions
from utils import Utils

class MultiscaleAlignMasking:
    @staticmethod
    def _intersect_extents(ext1, ext2):
        """Returns the intersection of two arcpy.Extent objects."""
        xmin = max(ext1.XMin, ext2.XMin)
        xmax = min(ext1.XMax, ext2.XMax)
        ymin = max(ext1.YMin, ext2.YMin)
        ymax = min(ext1.YMax, ext2.YMax)
        if xmin >= xmax or ymin >= ymax:
            return None
        return Extent(xmin, ymin, xmax, ymax)

    @staticmethod
    def return_valid_data_mask_intersection(input_rasters, replace=True):
        if not input_rasters:
            arcpy.AddWarning("Input raster list is empty.")
            return None

        arcpy.env.overwriteOutput = True
        valid_rasters = []
        common_extent = None
        master_raster_path = None
        master_sr = None
        
        print(f"Aligning {len(input_rasters)} rasters to master grid...")

        for i, raster_path in enumerate(input_rasters):
            if not arcpy.Exists(raster_path):
                continue

            desc = arcpy.Describe(raster_path)
            
            # 1. Establish the Master (First Valid Raster)
            if master_raster_path is None:
                master_raster_path = raster_path
                master_sr = desc.spatialReference
                master_cell_x = desc.meanCellWidth
                master_cell_y = desc.meanCellHeight
                common_extent = desc.extent
                valid_rasters.append(raster_path)
                continue

            # 2. Alignment Logic for subsequent rasters
            aligned_path = os.path.join(os.path.dirname(raster_path), 
                                        os.path.splitext(os.path.basename(raster_path))[0] + "_aligned.tif")
            backup_path = os.path.join(os.path.dirname(raster_path), 
                                       os.path.splitext(os.path.basename(raster_path))[0] + "_original.tif")

            # Check if alignment is already done (Resume capability)
            if not arcpy.Exists(aligned_path) and not (replace and arcpy.Exists(backup_path)):
                # Calculate "Perfect Sub-pixel" cell size
                # Forces current cell size to be a clean divisor of the master (e.g., 0.5 inside 1.0)
                curr_cell_x = desc.meanCellWidth
                curr_cell_y = desc.meanCellHeight
                
                # Logic: Find how many small pixels fit in one big pixel, round it, then divide master by that int.
                div_x = max(1, round(master_cell_x / curr_cell_x))
                div_y = max(1, round(master_cell_y / curr_cell_y))
                target_cell_size = f"{master_cell_x / div_x} {master_cell_y / div_y}"

                with arcpy.EnvManager(snapRaster=master_raster_path, outputCoordinateSystem=master_sr):
                    arcpy.management.ProjectRaster(
                        in_raster=raster_path,
                        out_raster=aligned_path,
                        out_coor_system=master_sr,
                        resampling_type="NEAREST",
                        cell_size=target_cell_size
                    )

            # 3. Handle File Replacement/Cleanup
            final_proc_path = raster_path
            if arcpy.Exists(aligned_path):
                if replace:
                    arcpy.ClearWorkspaceCache_management()
                    if not arcpy.Exists(backup_path):
                        arcpy.management.Rename(raster_path, backup_path)
                    else:
                        arcpy.management.Delete(raster_path)
                    arcpy.management.Rename(aligned_path, raster_path)
                    final_proc_path = raster_path
                else:
                    final_proc_path = aligned_path

            # Update Common Extent
            new_desc = arcpy.Describe(final_proc_path)
            common_extent = MultiscaleAlignMasking._intersect_extents(common_extent, new_desc.extent)
            valid_rasters.append(final_proc_path)

        # 4. Generate Intersection Mask with Memory Management
        print("Generating Intersection Mask (Multi-pass logic)...")
        arcpy.env.extent = common_extent
        arcpy.env.snapRaster = master_raster_path
        arcpy.env.cellSize = "MINOF" # Ensures final mask is at the highest resolution (e.g. 0.5m)

        intersection_mask = Con(IsNull(Raster(valid_rasters[0])), 0, 1)
        
        for i, raster_path in enumerate(valid_rasters[1:], 1):
            curr_mask = Con(IsNull(Raster(raster_path)), 0, 1)
            intersection_mask = intersection_mask * curr_mask
            
            # Collapse expresson tree every 10 rasters to prevent memory crash
            if i % 10 == 0:
                temp_mask = os.path.join(arcpy.env.scratchFolder, f"tmp_mask_{i}.tif")
                intersection_mask.save(temp_mask)
                intersection_mask = Raster(temp_mask)

        # 5. Finalize and Save
        output_mask_path = os.path.join(os.path.dirname(master_raster_path), "all_raster_intersection_mask.tif")
        final_mask = SetNull(intersection_mask == 0, 1)
        final_mask.save(output_mask_path)

        # Optional Cleanup of intermediate envs
        arcpy.env.extent = "DEFAULT"
        arcpy.env.snapRaster = None
        arcpy.ClearWorkspaceCache_management()
        
        print(f"Success. Mask saved to: {output_mask_path}")
        return output_mask_path, valid_rasters

    @staticmethod
    def align_rasters_return_mask(input_dems, input_bss, BS_MARKER="BS"):
        # --- 1. Basic Validation ---
        print(f"FOUND DEMS ({len(input_dems)}):")
        print("\n".join([f"  - {os.path.basename(f)}" for f in input_dems]))

        if input_bss:
            print(f"FOUND BACKSCATTER ({len(input_bss)}):")
            print("\n".join([f"  - {os.path.basename(f)}" for f in input_bss]))
        else:
            print("NO BACKSCATTER FILES FOUND. Proceeding with DEMs only.")

        # --- 2. Combine for Processing ---
        # Combine lists. DEMs first ensures the highest res DEM is the primary snap raster
        input_rasters_list = input_dems + input_bss

        # --- 3. Create Intersection Mask and Align ---
        # This aligns everything to the first raster in the list (the highest priority DEM)
        intersection_mask, aligned_rasters_list = MultiscaleAlignMasking.return_valid_data_mask_intersection(input_rasters_list)
        # --- 4. Separate Outputs back into DEM and BS ---
        # Using case-insensitive check for robustness
        aligned_bss = [f for f in aligned_rasters_list if f"{BS_MARKER}".lower() in os.path.basename(f).lower()]
        aligned_dems = [f for f in aligned_rasters_list if f not in aligned_bss]

        # --- 5. Final Validation ---
        print("-" * 30)
        print("PROCESSING COMPLETE")
        print(f"Aligned DEMs: {len(aligned_dems)}")
        print(f"Aligned BS:   {len(aligned_bss)}")

        # Verify we didn't lose any DEMs during the intersection process
        if len(aligned_dems) != len(input_dems):
            print(f"WARNING: Input DEM count ({len(input_dems)}) matches Aligned DEM count ({len(aligned_dems)})? NO")
        else:
            print("Validation: Input and Aligned DEM counts match.")

        # --- 6. Cleanup ---
        print("-" * 30)
        print("STARTING CLEANUP")

        # Collect all files involved (Inputs, Aligned Outputs, and the Mask)
        all_involved_files = input_rasters_list + aligned_rasters_list
        if intersection_mask:
            all_involved_files.append(intersection_mask)

        # Extract unique directories using a set comprehension
        # This prevents trying to clean the same folder multiple times
        cleanup_dirs = {os.path.dirname(f) for f in all_involved_files if f and os.path.exists(os.path.dirname(f))}

        # Iterate and clean
        for directory in cleanup_dirs:
            # Optional: Print where we are cleaning
            Utils.remove_additional_files(directory=directory)

        print("Cleanup Complete.")
        return aligned_dems, aligned_bss, intersection_mask
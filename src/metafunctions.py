import os
from utils import Utils, WorkspaceCleaner
from arcpyUtils import ArcpyUtils
from inpainter import Inpainter
import arcpy

class MetaFunctions():
    @staticmethod
    def fill_trim_with_intersection_mask(aligned_input_dem, aligned_input_bs, intersection_mask, fill_method="IDW", fill_iterations=1, min_area=50):
        """
        Generate a binary intersection mask by multiplying the valid data masks of two rasters.
        Returns the path to the intersection mask raster.
        """
        inpainter = Inpainter(input_raster=aligned_input_dem)
        if aligned_input_bs:
            inpainterbs = Inpainter(input_raster=aligned_input_bs)
        if arcpy.CheckProduct("ArcInfo") == "Available":
            mask, dissolved_polygonf = inpainter.get_data_boundary(min_area=min_area)
            if aligned_input_bs:
                maskbs, dissolved_polygonbs = inpainterbs.get_data_boundary(min_area=min_area)
        else:
            mask, dissolved_polygonf = inpainter.get_data_boundary_majorityfilter(min_area=min_area)
            if aligned_input_bs:
                maskbs, dissolved_polygonbs = inpainterbs.get_data_boundary_majorityfilter(min_area=min_area)
        if fill_method is not None:
            # Generate the fill raster
            filled_dem_path = inpainter.fill_internal_gaps_arcpy(
                method=fill_method,
                iterations=fill_iterations,
                dissolved_polygon=dissolved_polygonf,
                overwrite=True
            )
            # Clean up the workspace
            WorkspaceCleaner(inpainter).clean_up()
            # Generate the fill for the other raster
            if aligned_input_bs:
                filled_bs_path = inpainterbs.fill_internal_gaps_arcpy(
                    method=fill_method,
                    iterations=fill_iterations,
                    dissolved_polygon=dissolved_polygonbs,
                    overwrite=True
                )
                WorkspaceCleaner(inpainterbs).clean_up()
        else:
            print("Fill method NONE, returning aligned rasters and trimming to binary mask boundary provided")
            filled_dem_path, filled_bs_path = aligned_input_dem, aligned_input_bs
            WorkspaceCleaner(inpainter).clean_up()
            if aligned_input_bs:
                WorkspaceCleaner(inpainterbs).clean_up()
        # Trim the rasters to the intersection mask
        trimmed_dem_path = ArcpyUtils.trim_raster(filled_dem_path, intersection_mask, overwrite=True)
        if aligned_input_bs:
            trimmed_bs_path = ArcpyUtils.trim_raster(filled_bs_path, intersection_mask, overwrite=True)
        return trimmed_dem_path

    @staticmethod
    def fill_trim_make_intersection_mask(input_dem, input_bs, fill_method, fill_iterations, min_area=50):
        """
        Generate a binary intersection mask by multiplying the valid data masks of two rasters.
        Returns the path to the intersection mask raster.
        """
        dem_name = Utils.sanitize_path_to_name(input_dem)
        directory = os.path.join(os.path.dirname(input_dem), "boundary_files")
        inpainter = Inpainter(input_raster=input_dem)
        if input_bs:
            inpainterbs = Inpainter(input_raster=input_bs)
        if arcpy.CheckProduct("ArcInfo") == "Available":
            mask, dissolved_polygonf = inpainter.get_data_boundary(min_area=min_area)
            if input_bs:
                maskbs, dissolved_polygonbs = inpainterbs.get_data_boundary(min_area=min_area)
        else:
            mask, dissolved_polygonf = inpainter.get_data_boundary_majorityfilter(min_area=min_area)
            if input_bs:
                maskbs, dissolved_polygonbs = inpainterbs.get_data_boundary_majorityfilter(min_area=min_area)
        if fill_method is not None:
            # Generate the fill raster
            filled_dem_path = inpainter.fill_internal_gaps_arcpy(
                method=fill_method,
                iterations=fill_iterations,
                dissolved_polygon=dissolved_polygonf,
                overwrite=True
            )
            # Clean up the workspace
            WorkspaceCleaner(inpainter).clean_up()
            input_dem = filled_dem_path
            if input_bs:
                # Generate the fill for the backscatter
                filled_bs_path = inpainterbs.fill_internal_gaps_arcpy(
                    method=fill_method,
                    iterations=fill_iterations,
                    dissolved_polygon=dissolved_polygonbs,
                    overwrite=True
                )
                WorkspaceCleaner(inpainterbs).clean_up()
                input_bs = filled_bs_path
        else:
            WorkspaceCleaner(inpainter).clean_up()
            if input_bs:
                WorkspaceCleaner(inpainterbs).clean_up()
        # create the intersection mask by multiplying the two masks
        output_mask_path = os.path.join(directory, f"{dem_name}_intersection_mask.tif")
        # Remove if exists
        if os.path.exists(output_mask_path):
            arcpy.Delete_management(output_mask_path)
        intersection_mask = arcpy.Raster(mask) * arcpy.Raster(maskbs)
        intersection_mask.save(output_mask_path)

        # Trim the rasters to the intersection mask
        trimmed_dem_path = ArcpyUtils.trim_raster(input_dem, output_mask_path, overwrite=True)
        if input_bs:
            trimmed_bs_path = ArcpyUtils.trim_raster(input_bs, output_mask_path, overwrite=True)
        return trimmed_dem_path, intersection_mask
    
    @staticmethod
    def fill_and_return_mask(input_dem, fill_method="IDW", fill_iterations=1, min_area=50):
        # generate the cleaned data boundary and binary mask tif
        inpainter = Inpainter(input_dem)

        if arcpy.CheckProduct("ArcInfo") == "Available":
            mask, dissolved_polygon = inpainter.get_data_boundary(min_area=min_area)
        else:
            mask, dissolved_polygon = inpainter.get_data_boundary_majorityfilter(min_area=min_area)

        if fill_method is not None:
            # Generate the fill raster
            filled_raster_path = inpainter.fill_internal_gaps_arcpy(
                method=fill_method,
                iterations=fill_iterations,
                dissolved_polygon=dissolved_polygon
            )
            # Clean up the workspace
            WorkspaceCleaner(inpainter).clean_up()

        else:
            WorkspaceCleaner(inpainter).clean_up()
            filled_raster_path = input_dem

        return filled_raster_path, mask
        
    @staticmethod
    def fill_mask_with_polygon_management(input_mask, min_area=50):
        arcpy.env.overwriteOutput = True
        # generate the cleaned data boundary and binary mask tif
        inpainter = Inpainter(input_mask)
        if arcpy.CheckProduct("ArcInfo") == "Available":
            mask, _ = inpainter.get_data_boundary(min_area=min_area)
        else:
            mask, _ = inpainter.get_data_boundary_majorityfilter(min_area=min_area)
        WorkspaceCleaner(inpainter).clean_up()
        return mask

    @staticmethod
    def return_mask_MajorityFilter(input_dem):
        # Create Inpainter just for mask generation
        inpainter = Inpainter(input_dem)
        # Generate Boundary/Mask (no Fill). We use the raw DEM to define the boundary
        mask, _ = inpainter.get_data_boundary_majorityfilter()
        return mask
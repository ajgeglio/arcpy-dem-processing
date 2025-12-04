import os
from utils import Utils, WorkspaceCleaner
from inpainter import Inpainter
import arcpy

class MetaFunctions():
    @staticmethod
    def fill_trim_with_intersection_mask(aligned_input_dem, aligned_input_bs, intersection_mask, fill_method="IDW", fill_iterations=1):
        """
        Generate a binary intersection mask by multiplying the valid data masks of two rasters.
        Returns the path to the intersection mask raster.
        """
        inpainter = Inpainter(input_raster=aligned_input_dem)
        mask, dissolved_polygon = inpainter.get_data_boundary(min_area=50)
        inpainterbs = Inpainter(input_raster=aligned_input_bs)
        maskbs, dissolved_polygonbs = inpainterbs.get_data_boundary(min_area=50)
        if fill_method is not None:
            # Generate the fill raster
            filled_dem_path = inpainter.fill_internal_gaps_arcpy(
                input_mask=dissolved_polygon,
                method=fill_method,
                iterations=fill_iterations
            )
            # Clean up the workspace
            WorkspaceCleaner(inpainter).clean_up()
            # Generate the fill for the other raster
            filled_bs_path = inpainterbs.fill_internal_gaps_arcpy(
                input_mask=dissolved_polygonbs,
                method=fill_method,
                iterations=fill_iterations
            )
            WorkspaceCleaner(inpainterbs).clean_up()
        else:
            print("Fill method NONE, returning aligned rasters and trimming to binary mask boundary provided")
            filled_dem_path, filled_bs_path = aligned_input_dem, aligned_input_bs
            WorkspaceCleaner(inpainter).clean_up()
            WorkspaceCleaner(inpainterbs).clean_up()
        # Trim the rasters to the intersection mask
        trimmed_dem_path = inpainter.trim_raster(filled_dem_path, intersection_mask, overwrite=True)
        trimmed_bs_path = inpainterbs.trim_raster(filled_bs_path, intersection_mask, overwrite=True)
        return trimmed_dem_path, inpainter

    @staticmethod
    def fill_trim_make_intersection_mask(input_dem, input_bs, fill_method, fill_iterations):
        """
        Generate a binary intersection mask by multiplying the valid data masks of two rasters.
        Returns the path to the intersection mask raster.
        """
        dem_name = Utils.sanitize_path_to_name(input_dem)
        directory = os.path.join(os.path.dirname(input_dem), "boundary_files")
        inpainter = Inpainter(input_raster=input_dem)
        mask, dissolved_polygon = inpainter.get_data_boundary(min_area=50)
        inpainterbs = Inpainter(input_raster=input_bs)
        maskbs, dissolved_polygonbs = inpainterbs.get_data_boundary(min_area=50)
        if fill_method is not None:
            # Generate the fill raster
            filled_dem_path = inpainter.fill_internal_gaps_arcpy(
                input_mask=dissolved_polygon,
                method=fill_method,
                iterations=fill_iterations
            )
            # Clean up the workspace
            WorkspaceCleaner(inpainter).clean_up()
            input_dem = filled_dem_path
            # Generate the fill for the other raster
            filled_bs_path = inpainterbs.fill_internal_gaps_arcpy(
                input_mask=dissolved_polygonbs,
                method=fill_method,
                iterations=fill_iterations
            )
            WorkspaceCleaner(inpainterbs).clean_up()
            input_bs = filled_bs_path
        else:
            WorkspaceCleaner(inpainter).clean_up()
            WorkspaceCleaner(inpainterbs).clean_up()
        # create the intersection mask by multiplying the two masks
        output_mask_path = os.path.join(directory, f"{dem_name}_intersection_mask.tif")
        # Remove if exists
        if os.path.exists(output_mask_path):
            arcpy.Delete_management(output_mask_path)
        intersection_mask = arcpy.Raster(mask) * arcpy.Raster(maskbs)
        intersection_mask.save(output_mask_path)

        # Trim the rasters to the intersection mask
        trimmed_dem_path = inpainter.trim_raster(input_dem, output_mask_path, overwrite=True)
        trimmed_bs_path = inpainterbs.trim_raster(input_bs, output_mask_path, overwrite=True)
        return trimmed_dem_path, intersection_mask, inpainter
    
    @staticmethod
    def fill_and_return_mask(input_dem, fill_method="IDW", fill_iterations=1, generate_boundary=True):
        # generate the cleaned data boundary and binary mask tif
        inpainter = Inpainter(input_dem)

        if generate_boundary:
            binary_mask, dissolved_polygon = inpainter.get_data_boundary(min_area=50)

        if fill_method is not None:
            # Generate the fill raster
            filled_raster_path = inpainter.fill_internal_gaps_arcpy(
                input_mask=dissolved_polygon,
                method=fill_method,
                iterations=fill_iterations
            )
            # Clean up the workspace
            WorkspaceCleaner(inpainter).clean_up()
            # Set the filled raster as the input DEM for further processing
            output_dem = filled_raster_path

        else:
            WorkspaceCleaner(inpainter).clean_up()
            output_dem = input_dem

        return output_dem, inpainter, binary_mask
        
    @staticmethod
    def fill_mask(input_mask):
        arcpy.env.overwriteOutput = True
        # generate the cleaned data boundary and binary mask tif
        inpainter = Inpainter(input_mask)
        binary_mask_path, dissolved_polygon = inpainter.get_data_boundary(min_area=50)
        WorkspaceCleaner(inpainter).clean_up()
        return binary_mask_path

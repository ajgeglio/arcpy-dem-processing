import arcpy
from arcpy import env
from arcpy.sa import Raster
import os
from rasterMasking import RasterMasking

class tester:
    try:        
        # Set up a scratch workspace
        arcpy.env.workspace = r"C:\Users\ageglio\AppData\Local\Temp\2"
        arcpy.env.overwriteOutput = True  # Allow overwriting existing files
        arcpy.env.scratchWorkspace = arcpy.env.workspace  # Set scratch workspace
        print(f"ArcPy workspace set to: {arcpy.env.workspace}")

        # Create dummy rasters with different extents and some NoData
        # Raster 1: Larger extent, some NoData
        arcpy.management.CreateRasterDataset(
            arcpy.env.workspace, "raster1.tif", 10, "32_BIT_FLOAT",
            arcpy.SpatialReference(4326), 1
        )
        r1_path = os.path.join(arcpy.env.workspace, "raster1.tif")
        # Initialize raster with constant value so UpdateCursor works
        arcpy.ia.CreateConstantRaster(0, "FLOAT", 1, 10, 10).save(r1_path)
        with arcpy.da.UpdateCursor(r1_path, ["VALUE"]) as cursor:
            for i, row in enumerate(cursor):
                if i < 5 or i > 15: # Create NoData at ends
                    row[0] = -9999
                else:
                    row[0] = i
                cursor.updateRow(row)
        arcpy.management.SetRasterProperties(r1_path, "NODATA", "VALUE", "-9999")
        arcpy.management.DefineProjection(r1_path, arcpy.SpatialReference(4326))
        print(f"Raster 1 created: {r1_path}")

        # Raster 2: Smaller extent, slightly offset, some NoData
        arcpy.management.CreateRasterDataset(
            arcpy.env.workspace, "raster2.tif", 10, "32_BIT_FLOAT",
            arcpy.SpatialReference(4326), 1
        )
        r2_path = os.path.join(arcpy.env.workspace, "raster2.tif")
        arcpy.ia.CreateConstantRaster(0, "FLOAT", 1, 10, 10).save(r2_path)
        arcpy.env.extent = arcpy.Extent(-10, -10, 10, 10) # Set a temporary extent for creation
        with arcpy.da.UpdateCursor(r2_path, ["VALUE"]) as cursor:
            for i, row in enumerate(cursor):
                if i % 3 == 0: # Create some NoData
                    row[0] = -9999
                else:
                    row[0] = i * 2
                cursor.updateRow(row)
        arcpy.management.SetRasterProperties(r2_path, "NODATA", "VALUE", "-9999")
        arcpy.management.DefineProjection(r2_path, arcpy.SpatialReference(4326))
        arcpy.env.extent = "DEFAULT" # Reset extent
        print(f"Raster 2 created: {r2_path}")


        # Raster 3: Similar extent to Raster 2, but different values
        arcpy.management.CreateRasterDataset(
            arcpy.env.workspace, "raster3.tif", 10, "32_BIT_FLOAT",
            arcpy.SpatialReference(4326), 1
        )
        r3_path = os.path.join(arcpy.env.workspace, "raster3.tif")
        arcpy.ia.CreateConstantRaster(0, "FLOAT", 1, 10, 10).save(r3_path)
        arcpy.env.extent = arcpy.Extent(-5, -5, 15, 15) # Set a temporary extent for creation
        with arcpy.da.UpdateCursor(r3_path, ["VALUE"]) as cursor:
            for i, row in enumerate(cursor):
                if i % 4 == 0: # Create some NoData
                    row[0] = -9999
                else:
                    row[0] = i + 100
                cursor.updateRow(row)
        arcpy.management.SetRasterProperties(r3_path, "NODATA", "VALUE", "-9999")
        arcpy.management.DefineProjection(r3_path, arcpy.SpatialReference(4326))
        arcpy.env.extent = "DEFAULT" # Reset extent
        print(f"Raster 3 created: {r3_path}")

        input_rasters_list = [r1_path, r2_path, r3_path]

        # Run the function
        print("\n--- Generating Intersection Mask ---")
        output_mask_path = RasterMasking.return_valid_data_mask_intersection(input_rasters_list)

        if output_mask_path:
            print(f"Intersection mask created at: {output_mask_path}")
            # Verify the properties of the output mask
            mask_desc = arcpy.Describe(output_mask_path)
            print(f"Output mask extent: {mask_desc.extent}")
            print(f"Output mask cell size: {mask_desc.meanCellWidth}")

            # Compare with original rasters' extents
            r1_desc = arcpy.Describe(r1_path)
            r2_desc = arcpy.Describe(r2_path)
            r3_desc = arcpy.Describe(r3_path)
            
            print(f"Raster 1 extent: {r1_desc.extent}")
            print(f"Raster 2 extent: {r2_desc.extent}")
            print(f"Raster 3 extent: {r3_desc.extent}")
            
            # The mask's extent should be the intersection of these.
            # You can visually inspect by loading the mask and original rasters in ArcMap/Pro.

    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        # Clean up dummy rasters
        if 'r1_path' in locals() and arcpy.Exists(r1_path):
            arcpy.management.Delete(r1_path)
        if 'r2_path' in locals() and arcpy.Exists(r2_path):
            arcpy.management.Delete(r2_path)
        if 'r3_path' in locals() and arcpy.Exists(r3_path):
            arcpy.management.Delete(r3_path)
        print("\nCleaned up dummy rasters.")

if __name__ == "__main__":
    tester()
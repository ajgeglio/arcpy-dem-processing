import arcpy
from arcpy import env
from arcpy.sa import * # Import Spatial Analyst tools like Con, SetNull
import os
import sys

# Check out extension for Spatial Analyst tools
arcpy.CheckOutExtension("Spatial")

# Add your source paths
sys.path.append("src")
from utils import Utils
from multiscaleAlignMasking import MultiscaleAlignMasking
from processDem import ProcessDem

class tester:
    try:        
        # --- FIX 1: Create the directory using OS ---
        temp_dir = os.path.join(os.getcwd(), "Temp")
        if not os.path.exists(temp_dir):
            os.makedirs(temp_dir)
            print(f"Created dir: {temp_dir}")
            
        arcpy.env.workspace = temp_dir
        arcpy.env.overwriteOutput = True 
        arcpy.env.scratchWorkspace = temp_dir
        print(f"ArcPy workspace set to: {arcpy.env.workspace}")

        # 1. Define the Coordinate System we want (WGS84)
        sr_wgs84 = arcpy.SpatialReference(4326)

        # 2. Set a small cell size so we get plenty of pixels (not just 1 or 2)
        cell_size = 1 

        # === Raster 1: Base Square (0 to 100) ===
        print("Generating Raster 1 (Base)...")
        # Define extent in environment so CreateConstantRaster picks it up
        arcpy.env.extent = arcpy.Extent(0, 0, 100, 100)
        
        r1_base = CreateConstantRaster(10, "FLOAT", cell_size)
        r1_path = os.path.join(temp_dir, "raster1.tif")
        r1_base.save(r1_path)
        arcpy.management.DefineProjection(r1_path, sr_wgs84)

        # === Raster 2: Shifted Right (20 to 120) ===
        # Intersection with R1 will be X: 20-100 (80 pixels wide)
        print("Generating Raster 2 (Offset X)...")
        arcpy.env.extent = arcpy.Extent(20, 0, 120, 100)
        
        r2_base = CreateConstantRaster(5, "FLOAT", cell_size)
        r2_path = os.path.join(temp_dir, "raster2.tif")
        r2_base.save(r2_path)
        arcpy.management.DefineProjection(r2_path, sr_wgs84)

        # === Raster 3: Shifted Up (0 to 100, but Y is 20 to 120) ===
        # Intersection with R1+R2 will be X: 20-100, Y: 20-100
        print("Generating Raster 3 (Offset Y)...")
        arcpy.env.extent = arcpy.Extent(0, 20, 100, 120)
        
        r3_base = CreateConstantRaster(20, "FLOAT", cell_size)
        r3_path = os.path.join(temp_dir, "raster3.tif")
        r3_base.save(r3_path)
        arcpy.management.DefineProjection(r3_path, sr_wgs84)

        # Reset extent to default so future tools aren't limited
        arcpy.env.extent = "DEFAULT"

        input_rasters_list = [r1_path, r2_path, r3_path]

        # --- Run the function ---
        print("\n--- Generating Intersection Mask ---")
        
        # NOTE: Ensure this class is actually imported or defined above
        intersection_mask, valid_rasters = MultiscaleAlignMasking.return_valid_data_mask_intersection(input_rasters_list)
        # Generate terrain products of all of the dems and align the backscatter files
        for input_dem in valid_rasters:
            # Create an instance of the ProcessDEM class with the specified parameters
            generateHabitatDerivates = ProcessDem(
                                            input_dem=input_dem,
                                            input_bs=None,
                                            binary_mask=intersection_mask,
                                            divisions=None,  # Divide the height by this to run tile processing
                                            shannon_window=[3, 9, 21],
                                            fill_method="IDW",
                                            fill_iterations=1,
                                            water_elevation=183.6,
                                            keep_chunks=False,
                                            bypass_mosaic=False,
                                            )
            # Process the DEM and generate the habitat derivatives
            generateHabitatDerivates.process_dem()

    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"An error occurred: {e}")
        
    finally:
        # cleanup logic...
        print("\nFinished.")

if __name__ == "__main__":
    tester()
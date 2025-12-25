import arcpy
from arcpy.sa import *
import os
import tempfile
import time

class inpainter:
    def __init__(self, input_dem_path, save_path="habitat_derivatives"):
        """Initialize the inpainter with the input DEM path and optional save path."""
        self.input_dem_path = input_dem_path
        self.temp_workspace = tempfile.mkdtemp()  # Use a temporary folder for intermediate outputs
        self.tif_name = self.sanitize_name(os.path.splitext(os.path.basename(self.input_dem_path))[0])
        self.save_path = os.path.join(save_path, self.tif_name)
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

    def sanitize_name(self, name):
        """Replace invalid characters in the raster name."""
        return name.replace(".", "_").replace(" ", "_")

    def fill_internal_gaps_arcpy(self, max_iterations=1, smoothing_radius=1):
        """Fill internal gaps in the DEM using smoothing and filling."""
        arcpy.env.overwriteOutput = True
        arcpy.env.workspace = self.temp_workspace
        arcpy.env.scratchWorkspace = self.temp_workspace

        input_tif = self.input_dem_path
        converted_input = os.path.join(self.temp_workspace, f"{self.tif_name}_converted.tif")

        try:
            # Step 1: Verify the input raster
            print("Verifying input raster...")
            if not arcpy.Exists(input_tif):
                raise FileNotFoundError(f"Input raster not found: {input_tif}")
            desc = arcpy.Describe(input_tif)
            print(f"Input raster properties: Format={desc.format}, SpatialReference={desc.spatialReference.name}")

            # Step 2: Convert the input raster to a supported format
            print("Converting input raster to supported format...")
            arcpy.CopyRaster_management(input_tif, converted_input, format="TIFF")
            print(f"Converted raster saved to: {converted_input}")

            # Step 4: Iteratively smooth and fill NoData values
            print("Filling NoData values iteratively with smoothing...")
            filled_raster = Raster(converted_input)
            for iteration in range(max_iterations):
                print(f"Pass {iteration + 1} of smoothing and filling...")
                # Apply smoothing using FocalStatistics
                smoothed_raster = FocalStatistics(filled_raster, NbrCircle(smoothing_radius, "CELL"), "MEAN", "DATA")
                # Fill gaps after smoothing
                filled_raster = Fill(smoothed_raster)
                smoothed_raster = None  # Release smoothed_raster object
            filled_raster_path = os.path.join(self.save_path, f"{self.tif_name}_filled_raster.tif")
            filled_raster.save(filled_raster_path)
            print(f"Filled raster saved to: {filled_raster_path}")

        except Exception as e:
            print(f"An error occurred: {e}")

        finally:
            # Clean up temporary files
            print("Cleaning up temporary files...")
            arcpy.Delete_management("in_memory")  # Clear in-memory workspace
            filled_raster = None
            arcpy.env.workspace = None
            arcpy.env.scratchWorkspace = None

            for file in os.listdir(self.temp_workspace):
                file_path = os.path.join(self.temp_workspace, file)
                for _ in range(5):  # Retry up to 5 times
                    try:
                        os.remove(file_path)
                        break
                    except PermissionError:
                        print(f"Retrying deletion of {file_path}...")
                        time.sleep(1)
                else:
                    print(f"Could not delete {file_path}")

            try:
                os.rmdir(self.temp_workspace)
                print("Temporary files cleaned up.")
            except Exception as cleanup_error:
                print(f"Could not remove temporary directory: {cleanup_error}")
        
        return filled_raster_path

    def create_binary_mask(self, input_raster, boundary_shapefile):
        """Create a binary mask using the input raster extents and a boundary shapefile."""
        print("Creating binary mask...")
        boundary_raster = os.path.join(self.save_path, f"{self.tif_name}_boundary.tif")
        binary_mask = os.path.join(self.save_path, f"{self.tif_name}_binary_mask.tif")

        # Set the environment to match the input raster
        print("Setting environment to match input raster...")
        arcpy.env.extent = input_raster
        arcpy.env.snapRaster = input_raster
        arcpy.env.cellSize = input_raster

        # Rasterize the boundary shapefile
        print("Rasterizing the boundary shapefile...")
        arcpy.PolygonToRaster_conversion(boundary_shapefile, "FID", boundary_raster, "CELL_CENTER", "", input_raster)
        print(f"Boundary raster created: {boundary_raster}")

        # Generate the binary mask (1 for boundary, 0 for everything else)
        print("Generating binary mask...")
        binary_mask_raster = Con(IsNull(Raster(boundary_raster)), 0, 1)
        binary_mask_raster.save(binary_mask)
        print(f"Binary mask saved to: {binary_mask}")
        return binary_mask

    def trim_raster(self, raster, binary_mask):
        """Trim the raster using the binary mask and replace border zeros with NoData."""
        print("Trimming the raster using the binary mask...")
        name = os.path.splitext(os.path.basename(raster))[0]
        trimmed_raster_path = os.path.join(self.save_path, f"{name}_trimmed.tif")
        mask = Raster(binary_mask)
        
        # Convert binary mask to 1, NoData
        print("Converting binary mask to 1, NoData...")
        mask = Con(mask == 1, 1, None)

        # Apply the mask to the raster
        trimmed_raster = Raster(raster) * mask
        
        # Save the trimmed raster
        trimmed_raster.save(trimmed_raster_path)
        print(f"Trimmed raster saved to: {trimmed_raster_path}")
        return trimmed_raster_path

    def get_filled_data_boundary(self, min_area=1000, shrink_pixels=3, max_iterations=2, smoothing_radius=3):
        """Generate a shapefile of the data boundary, create a binary mask, and trim the filled raster."""
        filled_raster = self.fill_internal_gaps_arcpy(max_iterations=max_iterations, smoothing_radius=smoothing_radius)
        temp_integer_raster = None
        temp_polygon = None
        dissolved_polygon = None
        cleaned_polygon = None
        trimmed_raster = None

        try:
            print("Generating data boundary shapefile...")

            # Ensure the shapefile directory exists
            shapefile_dir = os.path.join(self.save_path, "shapefile")
            if not os.path.exists(shapefile_dir):
                os.makedirs(shapefile_dir)

            # Step 1: Convert the raster to integer type
            temp_integer_raster = os.path.join(self.save_path, f"{self.tif_name}_integer.tif")
            print("Converting raster to integer type...")
            integer_raster = Int(Raster(filled_raster))
            integer_raster.save(temp_integer_raster)
            print(f"Integer raster saved to: {temp_integer_raster}")

            # Step 2: Convert the integer raster to polygons
            temp_polygon = os.path.join(self.save_path, "temp_polygon.shp")
            print("Converting raster to polygons...")
            arcpy.RasterToPolygon_conversion(temp_integer_raster, temp_polygon, "NO_SIMPLIFY", "VALUE")
            print(f"Temporary polygon shapefile created: {temp_polygon}")

            # Step 3: Dissolve polygons to create a single boundary
            dissolved_polygon = os.path.join(self.save_path, "dissolved_polygon.shp")
            print("Dissolving polygons to create a single boundary...")
            arcpy.Dissolve_management(temp_polygon, dissolved_polygon)
            print(f"Dissolved polygon shapefile created: {dissolved_polygon}")

            # Step 4: Eliminate small polygons
            cleaned_polygon = os.path.join(self.save_path, "cleaned_polygon.shp")
            print(f"Removing small polygons smaller than {min_area} square units...")
            arcpy.EliminatePolygonPart_management(dissolved_polygon, cleaned_polygon, "AREA", min_area)
            print(f"Cleaned polygon shapefile created: {cleaned_polygon}")

            # Step 5: Shrink the boundary by n pixels
            output_shapefile = os.path.join(shapefile_dir, f"{self.tif_name}_boundary.shp")
            binary_mask = os.path.join(self.save_path, f"{self.tif_name}_binary_mask.tif")
            if shrink_pixels > 0:
                print(f"Shrinking the boundary by {shrink_pixels} pixels...")
                shrink_distance = -shrink_pixels  # Negative distance for shrinking
                arcpy.Buffer_analysis(cleaned_polygon, output_shapefile, shrink_distance, "FULL", "ROUND", "ALL")
                print(f"Shrunken boundary shapefile saved to: {output_shapefile}")
            else:
                # If no shrinking is needed, save the cleaned polygon as the final output
                arcpy.CopyFeatures_management(cleaned_polygon, output_shapefile)
                print(f"Boundary shapefile saved to: {output_shapefile}")

            # Create the binary mask
            binary_mask = self.create_binary_mask(filled_raster, output_shapefile)

            # Trim the filled raster using the binary mask
            trimmed_raster = self.trim_raster(filled_raster, binary_mask)

        except Exception as e:
            print(f"An error occurred while generating the data boundary: {e}")
            trimmed_raster = None  # Ensure trimmed_raster is defined even if an error occurs

        finally:
            # Clean up temporary files
            if temp_integer_raster and arcpy.Exists(temp_integer_raster):
                arcpy.Delete_management(temp_integer_raster)
                print("Temporary integer raster deleted.")
            if temp_polygon and arcpy.Exists(temp_polygon):
                arcpy.Delete_management(temp_polygon)
                print("Temporary polygon shapefile deleted.")
            if dissolved_polygon and arcpy.Exists(dissolved_polygon):
                arcpy.Delete_management(dissolved_polygon)
                print("Temporary dissolved polygon shapefile deleted.")
            if cleaned_polygon and arcpy.Exists(cleaned_polygon):
                arcpy.Delete_management(cleaned_polygon)
                print("Temporary cleaned polygon shapefile deleted.")

        return output_shapefile, binary_mask, trimmed_raster

if __name__ == "__main__":
    input_dem_path = r"C:\Users\ageglio\Documents\NLM_DataRelease\NLM_DataRelease\IngallsPoint_2021\0.5m\IP_BY_0.5m.tif"
    inpainter_instance = inpainter(input_dem_path)
    # boundary_shapefile, binary_mask, trimmed_raster = inpainter_instance.get_filled_data_boundary(
    #     shrink_pixels=3, min_area=1000, max_iterations=2, smoothing_radius=3
    # )
    # print(f"Boundary shapefile: {boundary_shapefile}")
    # print(f"Binary mask: {binary_mask}")
    # print(f"Trimmed raster: {trimmed_raster}")
    in_raster = r"habitat_derivatives\IP_BY_0_5m\IP_BY_0_5m_shannon_index_gdal.tif"
    trimmer = r"habitat_derivatives\IP_BY_0_5m\IP_BY_0_5m_binary_mask.tif"
    inpainter_instance.trim_raster(in_raster, trimmer)
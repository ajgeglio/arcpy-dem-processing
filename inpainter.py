import traceback
import os, tempfile
from utils import Utils
import arcpy
from arcpy.sa import *
import numpy as np
from scipy.ndimage import convolve

class Inpainter:
    def __init__(self, input_dem_path, save_path="."):
        """Initialize the inpainter with the input DEM path and optional save path."""
        self.input_dem_path = input_dem_path
        self.dem_name = Utils().sanitize_path_to_name(self.input_dem_path)
        self.temp_workspace = tempfile.mkdtemp()  # Use a temporary folder for intermediate outputs
        self.temp_files = []  # Track temporary files
        # create the save path if it doesn't exist
        self.save_path = os.path.join(save_path, self.dem_name)
        os.makedirs(self.save_path, exist_ok=True)
        self.local_temp = os.path.join(self.save_path, "temp")
        os.makedirs(self.local_temp, exist_ok=True)
        self.message_length = 0
        self.last_fill_iterations = None

        # --- 0. License Check ---
        if arcpy.CheckExtension("Spatial") == "Available":
            arcpy.CheckOutExtension("Spatial")
            arcpy.AddMessage("Spatial Analyst extension checked out.")
        else:
            arcpy.AddError("Spatial Analyst extension is not available. Cannot proceed.")
            raise arcpy.ExecuteError("Spatial Analyst license is unavailable.")

        # --- 1. Validate Inputs ---
        if not arcpy.Exists(self.input_dem_path):
            arcpy.AddError(f"Input DEM raster '{self.input_dem_path}' does not exist.")
            raise FileNotFoundError(f"Input DEM raster '{self.input_dem_path}' does not exist.")

        # Store original spatial reference, cell size, and snap raster for assertion
        dem_desc = arcpy.Describe(self.input_dem_path)
        self.original_spatial_ref = dem_desc.spatialReference
        self.original_cell_size = arcpy.management.GetRasterProperties(self.input_dem_path, "CELLSIZEX").getOutput(0)
        self.original_snap_raster = self.input_dem_path

    def impute_dem_gaps_idw(self, input_dem_path, input_mask, idw_power=2.0, search_radius_fixed_units=5.0, iterations=1):
        """
        Fills gaps in a Digital Elevation Model (DEM) using Inverse Distance Weighted (IDW)
        interpolation based on valid DEM cell values converted to points. Repeats the process
        for a specified number of iterations to fill additional gaps.

        Returns:
            Interpolated raster object or False if an error occurs.
        """
        # Store original environment settings to restore them later
        original_settings = {
            "overwriteOutput": arcpy.env.overwriteOutput,
            "outputCoordinateSystem": arcpy.env.outputCoordinateSystem,
            "cellSize": arcpy.env.cellSize,
            "snapRaster": arcpy.env.snapRaster,
            "mask": arcpy.env.mask
        }
        if not arcpy.Exists(input_mask):
            arcpy.AddError(f"Input mask polygon '{input_mask}' does not exist.")
            quit()
        
        temp_points_fc = None # Initialize for the finally block
        current_raster = input_dem_path

        try:
            arcpy.AddMessage(f"Starting DEM gap filling using IDW for {iterations} iteration(s)...")
           
            # --- 2. Set Environment Settings ---
            dem_desc = arcpy.Describe(input_dem_path)
            dem_spatial_ref = dem_desc.spatialReference
            
            # Get cell size (use CELLSIZEX, assuming square cells, common for DEMs)
            cell_size_x_result = arcpy.management.GetRasterProperties(input_dem_path, "CELLSIZEX")
            dem_cell_size = float(cell_size_x_result.getOutput(0))
            
            arcpy.env.outputCoordinateSystem = dem_spatial_ref
            arcpy.env.cellSize = dem_cell_size
            arcpy.env.snapRaster = input_dem_path
            arcpy.env.mask = input_mask

            for i in range(iterations):
                arcpy.AddMessage(f"IDW iteration {i+1} of {iterations}...")

                # Convert DEM to points (valid cells only)
                temp_points_fc = arcpy.CreateUniqueName(f"dem_idw_pts_{i}", "in_memory")
                # Only use valid (non-NoData) cells for points
                arcpy.conversion.RasterToPoint(
                    in_raster=arcpy.sa.SetNull(current_raster, current_raster, "VALUE IS NULL"),
                    out_point_features=temp_points_fc,
                    raster_field="VALUE"
                )

                point_count_result = arcpy.management.GetCount(temp_points_fc)
                point_count = int(point_count_result.getOutput(0))
                if point_count == 0:
                    arcpy.AddError("No valid data points were generated from the DEM (this can occur if the DEM "
                                "has no data within the mask or is entirely NoData). IDW requires input points. Cannot proceed.")
                    return False 
                arcpy.AddMessage(f"{point_count} points generated for IDW processing.")

                # Run IDW interpolation
                idw_search_radius_obj = arcpy.sa.RadiusFixed(search_radius_fixed_units)
                idw_interpolated_raster = arcpy.sa.Idw(
                    in_point_features=temp_points_fc,
                    z_field="grid_code",
                    power=idw_power,
                    search_radius=idw_search_radius_obj
                )

                # Combine original raster and IDW result: fill only NoData cells
                current_raster = arcpy.sa.Con(
                    arcpy.sa.IsNull(current_raster),
                    idw_interpolated_raster,
                    current_raster
                )

                # Clean up temp points
                if temp_points_fc and arcpy.Exists(temp_points_fc):
                    try:
                        arcpy.management.Delete(temp_points_fc)
                    except Exception:
                        pass

            arcpy.AddMessage("IDW interpolation complete.")
            return current_raster

        except arcpy.ExecuteError:
            arcpy.AddError(f"ArcPy ExecuteError: {arcpy.GetMessages(2)}")
            arcpy.AddMessage(traceback.format_exc())
            return False
        except Exception as e:
            arcpy.AddError(f"An unexpected Python error occurred: {str(e)}")
            arcpy.AddError(traceback.format_exc())
            return False
        finally:
            # Reset environment settings to their original state
            arcpy.env.overwriteOutput = original_settings["overwriteOutput"]
            arcpy.env.outputCoordinateSystem = original_settings["outputCoordinateSystem"]
            arcpy.env.cellSize = original_settings["cellSize"]
            arcpy.env.snapRaster = original_settings["snapRaster"]
            arcpy.env.mask = original_settings["mask"]
            arcpy.CheckInExtension("Spatial")
            arcpy.AddMessage("Environment settings reset and Spatial Analyst extension checked in.")
            arcpy.AddMessage("Process finished.")

    def impute_gaps_focal_statistics(self, input_tif, iterations, snap_raster=None):
        desc = arcpy.Describe(input_tif)
        converted_input = os.path.join(self.temp_workspace, f"{self.dem_name}_converted.tif")
        print(f"Input raster properties: Format={desc.format}, SpatialReference={desc.spatialReference.name}")

        # Store original environment settings to restore them later
        original_settings = {
            "snapRaster": arcpy.env.snapRaster,
            "cellSize": arcpy.env.cellSize,
            "outputCoordinateSystem": arcpy.env.outputCoordinateSystem
        }

        try:
            # Set environment to match input raster
            arcpy.env.snapRaster = snap_raster if snap_raster else input_tif
            arcpy.env.cellSize = input_tif
            arcpy.env.outputCoordinateSystem = desc.spatialReference

            message = "Converting input raster to supported format..."
            self.message_length = Utils.print_progress(message, self.message_length)
            arcpy.CopyRaster_management(input_tif, converted_input, format="TIFF")
            print(f"Converted raster saved to: {converted_input}")
            message = "Attempting filling NoData values..."
            self.message_length = Utils.print_progress(message, self.message_length)
            fc_filled_raster = Raster(converted_input)

            for iteration in range(iterations):
                message = f"Pass {iteration + 1} of filling with neighborhood focal statistics..."
                self.message_length = Utils.print_progress(message, self.message_length)
                neighborhood = NbrRectangle(3, 3, "CELL")
                fc_filled_raster = FocalStatistics(fc_filled_raster, neighborhood, "MEAN", "DATA")
            return fc_filled_raster

        finally:
            # Restore environment settings
            arcpy.env.snapRaster = original_settings["snapRaster"]
            arcpy.env.cellSize = original_settings["cellSize"]
            arcpy.env.outputCoordinateSystem = original_settings["outputCoordinateSystem"]

    def fill_internal_gaps_arcpy(self, input_mask, method="IDW", iterations=1, idw_power=2.0, search_radius=5.0):
        """Fill internal gaps in the DEM using smoothing and filling."""
        self.last_fill_iterations = iterations
        # Set environment to match original DEM
        dem_desc = arcpy.Describe(self.input_dem_path)
        arcpy.env.outputCoordinateSystem = dem_desc.spatialReference
        arcpy.env.snapRaster = self.input_dem_path
        arcpy.env.cellSize = self.input_dem_path
        arcpy.env.overwriteOutput = True
        arcpy.env.workspace = self.temp_workspace
        arcpy.env.scratchWorkspace = self.temp_workspace
        input_tif = self.input_dem_path
        filled_raster_path = os.path.join(self.save_path, f"{self.dem_name}_filled.tif")
        
        if not arcpy.Exists(input_mask):
            arcpy.AddError(f"Input mask polygon '{input_mask}' does not exist.")
            raise FileNotFoundError(f"Input mask polygon '{input_mask}' does not exist.")

        # Step 1: Verify the input raster
        self.message_length = Utils.print_progress("Verifying input raster...", self.message_length)
        if not arcpy.Exists(input_tif):
            raise FileNotFoundError(f"Input raster not found: {input_tif}")
        try:
            if method == "FocalStatistics":
                filled_raster = self.impute_gaps_focal_statistics(input_tif, iterations=iterations)
            elif method == "IDW":
                filled_raster = self.impute_dem_gaps_idw(input_tif, input_mask, idw_power=idw_power, search_radius_fixed_units=search_radius, iterations=iterations)
            
            # --- 5. Save the output raster ---
            filled_raster.save(filled_raster_path)
            arcpy.AddMessage(f"Saved filled DEM to: '{filled_raster_path}'...")
            arcpy.AddMessage(f"Filled DEM '{os.path.basename(filled_raster_path)}' saved successfully.")
            arcpy.AddMessage("DEM gap filling process completed successfully.")

            # Assert spatial reference, cell size, and snap raster are unchanged
            out_desc = arcpy.Describe(filled_raster_path)
            assert out_desc.spatialReference.name == self.original_spatial_ref.name, \
                f"Spatial reference changed! Expected: {self.original_spatial_ref.name}, Got: {out_desc.spatialReference.name}"
            # Optionally, check cell size (as string for float comparison)
            cell_size_x = arcpy.management.GetRasterProperties(filled_raster_path, "CELLSIZEX").getOutput(0)
            assert str(cell_size_x) == str(self.original_cell_size), \
                f"Cell size changed! Expected: {self.original_cell_size}, Got: {cell_size_x}"
            # Snap raster is an environment setting, not a property of the raster, so just ensure it's set
            assert arcpy.env.snapRaster == self.original_snap_raster, \
                f"Snap raster changed! Expected: {self.original_snap_raster}, Got: {arcpy.env.snapRaster}"

        except Exception as e:
            print(f"An error occurred: {e}")

        return filled_raster_path

    def create_binary_mask(self, input_dem_path, boundary_shapefile):
        """Create a binary mask using the input raster extents and a boundary shapefile."""
        message = "Creating binary mask..."
        self.message_length = Utils.print_progress(message, self.message_length)
        boundary_raster = os.path.join(self.local_temp, "boundary.tif")
        binary_mask = os.path.join(self.save_path, f"{self.dem_name}_binary_mask.tif")
        if os.path.exists(binary_mask):
            arcpy.Delete_management(binary_mask)
        try:
            # Set the environment to match the input raster
            message = "Setting environment to match input raster..."
            self.message_length = Utils.print_progress(message, self.message_length)
            arcpy.env.overwriteOutput = True
            arcpy.env.extent = input_dem_path
            arcpy.env.snapRaster = input_dem_path
            arcpy.env.cellSize = input_dem_path

            # Rasterize the boundary shapefile
            message = "Rasterizing the boundary shapefile..."
            self.message_length = Utils.print_progress(message, self.message_length)
            arcpy.PolygonToRaster_conversion(boundary_shapefile, "FID", boundary_raster, "CELL_CENTER", "", input_dem_path)
            self.temp_files.extend([boundary_raster])

            # Generate the binary mask (1 for boundary, 0 for everything else)
            message = "Generating binary mask..."
            self.message_length = Utils.print_progress(message, self.message_length)
            binary_mask_raster = Con(IsNull(Raster(boundary_raster)), 0, 1)

            # Save the binary mask
            binary_mask_raster.save(binary_mask)

        except Exception as e:
            print(f"An error occurred while creating the binary mask: {e}")

        finally:
            # Reset the environment settings to avoid referencing deleted files
            arcpy.env.snapRaster = None
            arcpy.env.extent = None
            arcpy.env.cellSize = None

        return binary_mask

    def trim_raster(self, input_raster_path, binary_mask, overwrite=False):
        """Trim the raster using the binary mask, replacing zeros with NoData.
        If overwrite=True, attempts to overwrite the original raster; otherwise, saves as a new file.
        Note: Overwriting a TIFF in-place with ArcPy Raster.save() is not supported. Always write to a new file, then replace if needed.
        """
        if not arcpy.Exists(input_raster_path):
            raise FileNotFoundError(f"Input raster not found: {input_raster_path}")
        else:
            message = "Trimming the raster using the binary mask..."
            self.message_length = Utils.print_progress(message, self.message_length)
            name = Utils().sanitize_path_to_name(input_raster_path)
            # Always write to a temporary file first
            trimmed_raster_path = os.path.join(self.save_path, f"{name}_trimmed.tif")
            mask = Raster(binary_mask)
            # Convert binary mask to 1, NoData
            nodata_mask = SetNull(mask == 0, mask)
            message = "Converting binary mask to 1, NoData..."
            self.message_length = Utils.print_progress(message, self.message_length)

            # Apply the mask to the raster
            trimmed_raster = Raster(input_raster_path) * nodata_mask

            # Save the trimmed raster
            trimmed_raster.save(trimmed_raster_path)
            print(f"Trimmed raster saved to: {trimmed_raster_path}")

            # If overwrite=True, replace the original file after saving
            if overwrite:
                try:
                    # Release ArcPy raster references
                    del trimmed_raster
                    import gc # Imports the garbage collection module, which provides access to the Python garbage collector.
                    gc.collect() # Explicitly runs garbage collection to free up unreferenced memory objects.
                    # Use ArcPy to delete the original raster (handles locks better than os.remove)
                    if arcpy.Exists(input_raster_path):
                        arcpy.Delete_management(input_raster_path)
                    # Now rename the trimmed raster to the original path
                    os.rename(trimmed_raster_path, input_raster_path)
                    print(f"Original raster overwritten: {input_raster_path}")
                    return input_raster_path
                except Exception as e:
                    print(f"Failed to overwrite original raster: {e}")
                    return trimmed_raster_path
            else:
                return trimmed_raster_path

    def get_data_boundary(self, min_area=50, shrink_boundary_pixels=0):
        """
        Generate a shapefile of the data boundary, create a binary mask, and trim the filled raster.

        min_area is only set by the caller (e.g., HabitatDerivatives), not overwritten here.
        """

        input_tif = self.input_dem_path
        temp_integer_raster = None
        temp_polygon = None
        dissolved_polygon = None
        cleaned_polygon = None
        binary_mask = None
        
        try:
            message = "Setting environment to match input raster..."
            self.message_length = Utils.print_progress(message, self.message_length)
            arcpy.env.overwriteOutput = True
            arcpy.env.extent = input_tif
            arcpy.env.snapRaster = input_tif
            arcpy.env.cellSize = input_tif

            message = "Generating data boundary shapefile..."
            self.message_length = Utils.print_progress(message, self.message_length)
            # Ensure the shapefile directory exists
            shapefile_dir = os.path.join(self.save_path, "boundary_shapefile")
            os.makedirs(shapefile_dir, exist_ok=True)

            # Step 1: Convert the raster to integer type
            temp_integer_raster = os.path.join(self.local_temp, "integer.tif")
            message = "Converting raster to integer type..."
            self.message_length = Utils.print_progress(message, self.message_length)
            integer_raster = Int(Raster(input_tif))
            integer_raster.save(temp_integer_raster)

            # Step 2: Convert the integer raster to polygons
            temp_polygon = os.path.join(self.local_temp, "temp_polygon.shp")
            message = "Converting raster to polygons..."
            self.message_length = Utils.print_progress(message, self.message_length)
            arcpy.RasterToPolygon_conversion(temp_integer_raster, temp_polygon, "NO_SIMPLIFY", "VALUE")

            # Step 3: Dissolve polygons to create a single boundary
            dissolved_polygon = os.path.join(self.local_temp, "dissolved_polygon.shp")
            # Remove all files with the dissolved_polygon base name (shapefile and its sidecar files)
            for ext in [".shp", ".shx", ".dbf", ".prj", ".cpg", ".sbn", ".sbx", ".fbn", ".fbx", ".ain", ".aih", ".ixs", ".mxs", ".atx", ".shp.xml", ".qix"]:
                f = dissolved_polygon.replace(".shp", ext)
                if os.path.exists(f):
                    os.remove(f)
            message = "Dissolving polygons..."
            self.message_length = Utils.print_progress(message, self.message_length)
            arcpy.Dissolve_management(temp_polygon, dissolved_polygon)

            # Step 4: Eliminate small polygons (Requires Advanced License)

            cleaned_polygon = os.path.join(self.local_temp, "cleaned_polygon.shp")
            if os.path.exists(cleaned_polygon):
                os.remove(cleaned_polygon)  # Delete existing file
            print(f"Minimum area for elimination: {min_area} square units")
            if arcpy.CheckProduct("ArcInfo") == "Available":
                # Do not overwrite min_area here
                arcpy.EliminatePolygonPart_management(
                    in_features=dissolved_polygon,
                    out_feature_class=cleaned_polygon,
                    part_area=min_area,
                    part_option="ANY"
                )
            else:
                arcpy.AddWarning("Advanced license not available. Cannot use EliminatePolygonPart_management.")
                # If the advanced license is not available, we cannot use EliminatePolygonPart_management
                print("Unable to use EliminatePolygonPart_management with min_area.")
                # Use a different method to filter polygons based on area (not yet implemented)
                # For now, just copy the dissolved polygon to cleaned_polygon
                cleaned_polygon = dissolved_polygon

            dissolved_polygon_path = os.path.join(shapefile_dir, f"{self.dem_name}_boundary.shp")
            if os.path.exists(dissolved_polygon_path):
                message = "Deleting existing shapefile"
                self.message_length = Utils.print_progress(message, self.message_length)
                arcpy.Delete_management(dissolved_polygon_path)  # Ensure no file conflicts

            if shrink_boundary_pixels > 0:
                # Step 5: Shrink the boundary by n pixels if necessary
                message = f"Shrinking the boundary by {shrink_boundary_pixels} pixels..."
                self.message_length = Utils.print_progress(message, self.message_length)
                shrink_distance = -shrink_boundary_pixels  # Negative distance for shrinking
                arcpy.Buffer_analysis(cleaned_polygon, dissolved_polygon_path, shrink_distance, "FULL", "ROUND", "ALL")
            else:
                arcpy.CopyFeatures_management(cleaned_polygon, dissolved_polygon_path)

            # Create the binary mask
            binary_mask_path = self.create_binary_mask(input_tif, dissolved_polygon_path)

            # Track temporary files for cleanup
            self.temp_files.extend([temp_integer_raster, temp_polygon, dissolved_polygon, cleaned_polygon])

        except Exception as e:
            print(f"An error occurred while generating the data boundary: {e}")
            raise  # Re-raise the exception for the caller to handle

        finally:
            # Reset the environment settings to avoid referencing deleted files
            arcpy.env.snapRaster = None
            arcpy.env.extent = None
            arcpy.env.cellSize = None

        return binary_mask_path, dissolved_polygon_path

    def remove_isolated_pixels(self, output_path=None, nodata_value=None):
        """
        Not currently used, but provided for the potential future need to clean up isolated pixels in the DEM.

        Removes isolated pixels from the DEM that are surrounded by NoData or have 3 or more sides exposed,
        or neighbor a pixel that has 3 or more sides exposed.

        Args:
            dem_raster_path (str): Path to the input DEM raster.
            output_path (str): Path to save the cleaned DEM raster. If None, will overwrite input.
            nodata_value (float or int, optional): Value representing NoData in the DEM. If None, will infer from raster.

        Returns:
            str: Path to the cleaned DEM raster.
        """
        # Read DEM as numpy array
        with arcpy.da.SearchCursor(self.input_dem_path, ["Value"]) as cursor:
            pass  # Just to ensure arcpy is imported for context, but we use rasterio for numpy array

        import rasterio
        with rasterio.open(self.input_dem_path) as src:
            dem = src.read(1)
            meta = src.meta.copy()
            if nodata_value is None:
                nodata_value = src.nodata if src.nodata is not None else -9999

        # Create a mask of valid pixels (not nodata)
        valid_mask = (dem != nodata_value) & (~np.isnan(dem))

        # Define a 3x3 kernel to count valid neighbors
        kernel = np.array([[1,1,1],
                           [1,0,1],
                           [1,1,1]])

        # Count valid neighbors for each pixel
        neighbor_count = convolve(valid_mask.astype(int), kernel, mode='constant', cval=0)

        # Find pixels with 3 or fewer valid neighbors (i.e., 3+ sides exposed)
        exposed_mask = (valid_mask) & (neighbor_count <= 3)

        # Find pixels that neighbor an exposed pixel
        neighbor_exposed = convolve(exposed_mask.astype(int), kernel, mode='constant', cval=0) > 0

        # Remove (set to nodata) pixels that are exposed or neighbor an exposed pixel
        to_remove = exposed_mask | ((valid_mask) & neighbor_exposed)
        cleaned_dem = dem.copy()
        cleaned_dem[to_remove] = nodata_value

        # Write the cleaned DEM to output
        if output_path is None:
            output_path = self.input_dem_path  # Overwrite input
        with rasterio.open(output_path, 'w', **meta) as dst:
            dst.write(cleaned_dem, 1)

        return output_path

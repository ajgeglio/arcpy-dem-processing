import traceback
import os, tempfile
from utils import Utils
import arcpy
from arcpy.sa import *

class Inpainter:
    def __init__(self, input_raster, save_path=None):
        """Initialize the inpainter with the input raster path and optional save path."""
        self.temp_workspace = tempfile.mkdtemp()  # Use a temporary folder for intermediate outputs
        self.temp_files = []  # Track temporary files

        # --- 0. License Check ---
        if arcpy.CheckExtension("Spatial") == "Available":
            arcpy.CheckOutExtension("Spatial")
            arcpy.AddMessage("Spatial Analyst extension checked out.")
        else:
            arcpy.AddError("Spatial Analyst extension is not available. Cannot proceed.")
            raise arcpy.ExecuteError("Spatial Analyst license is unavailable.")

        # Store original spatial reference, cell size, and snap raster for assertion
        self.input_raster = input_raster
        self.dem_name = Utils.sanitize_path_to_name(self.input_raster)
        dem_desc = arcpy.Describe(self.input_raster)
        self.original_spatial_ref = dem_desc.spatialReference
        self.original_cell_size = arcpy.management.GetRasterProperties(self.input_raster, "CELLSIZEX").getOutput(0)
        self.original_snap_raster = self.input_raster

        # Set arcpy.env variables to match input raster
        arcpy.env.outputCoordinateSystem = self.original_spatial_ref
        arcpy.env.snapRaster = self.original_snap_raster
        arcpy.env.cellSize = self.original_cell_size

        # --- 1. Validate Inputs ---
        if not arcpy.Exists(self.input_raster):
            arcpy.AddError(f"Input raster '{self.input_raster}' does not exist.")
            raise FileNotFoundError(f"Input raster '{self.input_raster}' does not exist.")
        
        # create the save path if it doesn't exist
        self.filled_path = os.path.join(os.path.dirname(self.input_raster), "filled")
        if save_path is None:
            self.save_path = os.path.join(os.path.dirname(self.input_raster), "boundary_files")
        else:
            self.save_path = save_path
        self.local_temp = os.path.join(self.save_path, self.dem_name+"_local_temp")
        
        os.makedirs(self.local_temp, exist_ok=True)
        os.makedirs(self.save_path, exist_ok=True)
        os.makedirs(self.filled_path, exist_ok=True)

        self.message_length = 0
        self.last_fill_iterations = None
        arcpy.env.workspace = self.local_temp
        arcpy.env.scratchWorkspace = self.temp_workspace


    def impute_dem_gaps_idw(self, input_raster, input_mask, idw_power=2.0, search_radius_fixed_units=5.0, iterations=1):
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
            "mask": arcpy.env.mask,
            "extent": arcpy.env.extent,
            "workspace": arcpy.env.workspace,
            "scratchWorkspace": arcpy.env.scratchWorkspace
        }
        if not arcpy.Exists(input_mask):
            arcpy.AddError(f"Input mask polygon '{input_mask}' does not exist.")
            quit()
        
        temp_points_fc = None # Initialize for the finally block
        current_raster = input_raster

        try:
            arcpy.AddMessage(f"Starting DEM gap filling using IDW for {iterations} iteration(s)...")
           
            # --- 2. Set Environment Settings ---
            dem_desc = arcpy.Describe(input_raster)
            dem_spatial_ref = dem_desc.spatialReference
            
            # Get cell size (use CELLSIZEX, assuming square cells, common for DEMs)
            cell_size_x_result = arcpy.management.GetRasterProperties(input_raster, "CELLSIZEX")
            dem_cell_size = float(cell_size_x_result.getOutput(0))
            
            arcpy.env.outputCoordinateSystem = dem_spatial_ref
            arcpy.env.cellSize = dem_cell_size
            arcpy.env.snapRaster = input_raster
            arcpy.env.mask = input_mask
            arcpy.env.extent = dem_desc.extent
            arcpy.env.workspace = self.local_temp
            arcpy.env.scratchWorkspace = self.temp_workspace
            arcpy.env.overwriteOutput = True

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
            arcpy.env.extent = original_settings["extent"]
            arcpy.env.workspace = original_settings["workspace"]
            arcpy.env.scratchWorkspace = original_settings["scratchWorkspace"]
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
            "outputCoordinateSystem": arcpy.env.outputCoordinateSystem,
            "extent": arcpy.env.extent,
            "workspace": arcpy.env.workspace,
            "scratchWorkspace": arcpy.env.scratchWorkspace,
            "overwriteOutput": arcpy.env.overwriteOutput
        }

        try:
            # Set environment to match input raster
            arcpy.env.snapRaster = snap_raster if snap_raster else input_tif
            arcpy.env.cellSize = input_tif
            arcpy.env.outputCoordinateSystem = desc.spatialReference
            arcpy.env.extent = desc.extent
            arcpy.env.workspace = self.local_temp
            arcpy.env.scratchWorkspace = self.temp_workspace
            arcpy.env.overwriteOutput = True

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
            arcpy.env.extent = original_settings["extent"]
            arcpy.env.workspace = original_settings["workspace"]
            arcpy.env.scratchWorkspace = original_settings["scratchWorkspace"]
            arcpy.env.overwriteOutput = original_settings["overwriteOutput"]

    def fill_internal_gaps_arcpy(self, input_mask, method="IDW", iterations=1, idw_power=2.0, search_radius=5.0):
        """Fill internal gaps in the DEM using smoothing and filling."""
        self.last_fill_iterations = iterations
        # Store original environment settings to restore them later
        original_settings = {
            "outputCoordinateSystem": arcpy.env.outputCoordinateSystem,
            "snapRaster": arcpy.env.snapRaster,
            "cellSize": arcpy.env.cellSize,
            "overwriteOutput": arcpy.env.overwriteOutput,
            "workspace": arcpy.env.workspace,
            "scratchWorkspace": arcpy.env.scratchWorkspace,
            "extent": arcpy.env.extent,
            "mask": arcpy.env.mask
        }
        try:
            # Set environment to match original DEM
            dem_desc = arcpy.Describe(self.input_raster)
            arcpy.env.outputCoordinateSystem = dem_desc.spatialReference
            arcpy.env.snapRaster = self.input_raster
            arcpy.env.cellSize = self.input_raster
            arcpy.env.overwriteOutput = True
            arcpy.env.workspace = self.local_temp
            arcpy.env.scratchWorkspace = self.temp_workspace
            arcpy.env.extent = arcpy.Describe(self.input_raster).extent
            arcpy.env.mask = input_mask if method == "IDW" else None
            input_tif = self.input_raster
            filled_raster_path = os.path.join(self.filled_path, f"{self.dem_name}_filled.tif")
            
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
                if str(arcpy.env.snapRaster) != str(self.original_snap_raster):
                    print(f"DEBUG: Snap raster mismatch!")
                    print(f"Expected: {self.original_snap_raster}")
                    print(f"Got: {arcpy.env.snapRaster}")
                    # Only raise an error if the difference is more than just case
                    if str(arcpy.env.snapRaster).lower() == str(self.original_snap_raster).lower():
                        print("WARNING: Snap raster paths differ only by case. This is usually not an issue on Windows.")
                        # Do not raise an error, continue
                    else:
                        raise AssertionError(
                            f"Snap raster changed! Expected: {self.original_snap_raster}, Got: {arcpy.env.snapRaster}"
                        )
                return filled_raster_path

            except Exception as e:
                print(f"An error occurred: {e}")

        finally:
            # Restore environment settings
            arcpy.env.outputCoordinateSystem = original_settings["outputCoordinateSystem"]
            arcpy.env.snapRaster = original_settings["snapRaster"]
            arcpy.env.cellSize = original_settings["cellSize"]
            arcpy.env.overwriteOutput = original_settings["overwriteOutput"]
            arcpy.env.workspace = original_settings["workspace"]
            arcpy.env.scratchWorkspace = original_settings["scratchWorkspace"]
            arcpy.env.extent = original_settings["extent"]
            arcpy.env.mask = original_settings["mask"]

    def create_binary_mask(self, input_raster, boundary_shapefile):
        """Create a binary mask using the input raster extents and a boundary shapefile."""
        message = "Creating binary mask..."
        self.message_length = Utils.print_progress(message, self.message_length)
        boundary_raster = os.path.join(self.local_temp, "boundary.tif")
        binary_mask = os.path.join(self.save_path, f"{self.dem_name}_binary_mask.tif")
        if os.path.exists(binary_mask):
            arcpy.Delete_management(binary_mask)
        # Store original environment settings to restore them later
        # --- CHANGED: Use input_raster properties for extent, snapRaster, cellSize ---
        desc = arcpy.Describe(input_raster)
        cell_size = arcpy.management.GetRasterProperties(input_raster, "CELLSIZEX").getOutput(0)
        original_settings = {
            "overwriteOutput": arcpy.env.overwriteOutput,
            "extent": desc.extent,
            "snapRaster": input_raster,
            "cellSize": cell_size,
            "workspace": arcpy.env.workspace,
            "scratchWorkspace": arcpy.env.scratchWorkspace
        }
        try:
            # Set the environment to the intersection extent of DEM and backscatter
            message = "Setting environment to input DEM..."
            self.message_length = Utils.print_progress(message, self.message_length)
            arcpy.env.overwriteOutput = True
            arcpy.env.extent = desc.extent
            arcpy.env.snapRaster = input_raster
            arcpy.env.cellSize = cell_size
            arcpy.env.workspace = self.local_temp
            arcpy.env.scratchWorkspace = self.temp_workspace

            # Rasterize the boundary shapefile
            message = "Rasterizing the boundary shapefile..."
            self.message_length = Utils.print_progress(message, self.message_length)
            arcpy.PolygonToRaster_conversion(boundary_shapefile, "FID", boundary_raster, "CELL_CENTER", "", input_raster)
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
            arcpy.env.overwriteOutput = original_settings["overwriteOutput"]
            arcpy.env.snapRaster = original_settings["snapRaster"]
            arcpy.env.extent = original_settings["extent"]
            arcpy.env.cellSize = original_settings["cellSize"]
            arcpy.env.workspace = original_settings["workspace"]
            arcpy.env.scratchWorkspace = original_settings["scratchWorkspace"]

        return binary_mask

    def trim_raster(self, input_raster_path, binary_mask, overwrite=False):
        """Trim the raster using the binary mask, replacing zeros with NoData.
        If overwrite=True, attempts to overwrite the original raster; otherwise, saves as a new file.
        Note: Overwriting a TIFF in-place with ArcPy Raster.save() is not supported. Always write to a new file, then replace if needed.
        """
        # Store original environment settings to restore them later
        original_settings = {
            "snapRaster": arcpy.env.snapRaster,
            "cellSize": arcpy.env.cellSize,
            "outputCoordinateSystem": arcpy.env.outputCoordinateSystem,
            "extent": arcpy.env.extent,
            "workspace": arcpy.env.workspace,
            "scratchWorkspace": arcpy.env.scratchWorkspace,
            "overwriteOutput": arcpy.env.overwriteOutput
        }
        try:
            if not arcpy.Exists(input_raster_path):
                raise FileNotFoundError(f"Input raster not found: {input_raster_path}")
            else:
                message = "Trimming the raster using the binary mask..."
                self.message_length = Utils.print_progress(message, self.message_length)
                name = Utils.sanitize_path_to_name(input_raster_path)
                # Always write to a temporary file first
                trimmed_raster_path = os.path.join(self.save_path, f"{name}_trimmed.tif")
                mask = Raster(binary_mask)
                # Convert binary mask to 1, NoData
                valid_mask = SetNull(mask == 0, mask)
                message = "Converting binary mask to 1, NoData..."
                self.message_length = Utils.print_progress(message, self.message_length)

                # Set environment to match input raster
                arcpy.env.snapRaster = input_raster_path
                arcpy.env.cellSize = input_raster_path
                arcpy.env.outputCoordinateSystem = arcpy.Describe(input_raster_path).spatialReference
                arcpy.env.extent = arcpy.Describe(input_raster_path).extent
                arcpy.env.workspace = self.local_temp
                arcpy.env.scratchWorkspace = self.temp_workspace
                arcpy.env.overwriteOutput = True

                # Apply the mask to the raster
                trimmed_raster = Raster(input_raster_path) * valid_mask

                # Save the trimmed raster
                trimmed_raster.save(trimmed_raster_path)
                print(f"Trimmed raster saved to: {trimmed_raster_path}")

                # If overwrite=True, replace the original file after saving
                if overwrite:
                    try:
                        # Release ArcPy raster references
                        # This is important to avoid file locks when trying to delete or rename files
                        message = "Attempting to overwrite the original raster..."
                        self.message_length = Utils.print_progress(message, self.message_length)    
                        del trimmed_raster
                        # Delete associated files (.tif.aux.xml, .tfw, etc.)
                        base, ext = os.path.splitext(trimmed_raster_path)
                        for suffix in [".tif.aux.xml", ".tfw", ".tif.ovr"]:
                            aux_file = base + suffix
                            if os.path.exists(aux_file):
                                os.remove(aux_file)
                        import gc # Imports the garbage collection module, which provides access to the Python garbage collector.
                        gc.collect() # Explicitly runs garbage collection to free up unreferenced memory objects.
                        # Use ArcPy to delete the original raster (handles locks better than os.remove)
                        if arcpy.Exists(input_raster_path):
                            arcpy.Delete_management(input_raster_path)
                        # Now rename the trimmed raster to the original path
                        arcpy.management.CopyRaster(in_raster=trimmed_raster_path, out_rasterdataset= input_raster_path, format="TIFF")
                        arcpy.Delete_management(trimmed_raster_path)  # Remove the temporary trimmed file
                        # Add the original raster to temp files for cleanup
                        print(f"Original raster overwritten: {input_raster_path}")
                        return input_raster_path
                    except Exception as e:
                        print(f"Failed to overwrite original raster: {e}")
                        return trimmed_raster_path
                else:
                    return trimmed_raster_path
        finally:
            # Restore environment settings
            arcpy.env.snapRaster = original_settings["snapRaster"]
            arcpy.env.cellSize = original_settings["cellSize"]
            arcpy.env.outputCoordinateSystem = original_settings["outputCoordinateSystem"]
            arcpy.env.extent = original_settings["extent"]
            arcpy.env.workspace = original_settings["workspace"]
            arcpy.env.scratchWorkspace = original_settings["scratchWorkspace"]
            arcpy.env.overwriteOutput = original_settings["overwriteOutput"]

    def get_data_boundary(self, min_area=50, shrink_boundary_pixels=0):
        """
        Generate a shapefile of the data boundary, create a binary mask, and trim the filled raster.

        min_area is only set by the caller (e.g., HabitatDerivatives), not overwritten here.
        """

        input_raster = self.input_raster
        temp_integer_raster = None
        temp_polygon = None
        dissolved_polygon = None
        cleaned_polygon = None
        
        # Store original environment settings to restore them later
        original_settings = {
            "overwriteOutput": arcpy.env.overwriteOutput,
            "extent": arcpy.env.extent,
            "snapRaster": arcpy.env.snapRaster,
            "cellSize": arcpy.env.cellSize,
            "workspace": arcpy.env.workspace,
            "scratchWorkspace": arcpy.env.scratchWorkspace
        }
        try:
            message = "Setting environment to DEM and properties..."
            self.message_length = Utils.print_progress(message, self.message_length)
            desc = arcpy.Describe(input_raster)  # Set once and reuse
            cell_size = arcpy.management.GetRasterProperties(input_raster, "CELLSIZEX").getOutput(0)
            arcpy.env.overwriteOutput = True
            arcpy.env.extent = desc.extent
            arcpy.env.snapRaster = input_raster
            arcpy.env.cellSize = cell_size
            arcpy.env.workspace = self.local_temp
            arcpy.env.scratchWorkspace = self.temp_workspace

            message = "Generating data boundary shapefile..."
            self.message_length = Utils.print_progress(message, self.message_length)
            # Ensure the shapefile directory exists
            shapefile_dir = os.path.join(self.save_path, "boundary_shapefile")
            os.makedirs(shapefile_dir, exist_ok=True)

            # Step 1: Convert the raster to integer type
            temp_integer_raster = os.path.join(self.local_temp, "integer.tif")
            message = "Converting raster to integer type..."
            self.message_length = Utils.print_progress(message, self.message_length)
            integer_raster = Int(Raster(input_raster))
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
            message = "min_area for polygon management set to: " + str(min_area)
            self.message_length = Utils.print_progress(message, self.message_length)
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

            # Create the binary mask (will use input_tif extent)
            binary_mask_path = self.create_binary_mask(input_raster, dissolved_polygon_path)

            # Track temporary files for cleanup
            self.temp_files.extend([temp_integer_raster, temp_polygon, dissolved_polygon, cleaned_polygon])

        except Exception as e:
            print(f"An error occurred while generating the data boundary: {e}")
            raise  # Re-raise the exception for the caller to handle

        finally:
            # Reset the environment settings to avoid referencing deleted files
            arcpy.env.overwriteOutput = original_settings["overwriteOutput"]
            arcpy.env.snapRaster = original_settings["snapRaster"]
            arcpy.env.extent = original_settings["extent"]
            arcpy.env.cellSize = original_settings["cellSize"]
            arcpy.env.workspace = original_settings["workspace"]
            arcpy.env.scratchWorkspace = original_settings["scratchWorkspace"]

        return binary_mask_path, dissolved_polygon_path



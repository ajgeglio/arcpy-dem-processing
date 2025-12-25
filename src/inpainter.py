import traceback
import os, tempfile
from utils import Utils
import arcpy
from arcpy.sa import *
import uuid
from arcpy.sa import FocalStatistics, Con, IsNull, Raster, NbrRectangle, Abs, SetNull, Idw, RadiusFixed
import shutil

class Inpainter:
    def __init__(self, input_raster, save_path=None):
        self.temp_files = []

        if arcpy.CheckExtension("Spatial") == "Available":
            arcpy.CheckOutExtension("Spatial")
        else:
            raise arcpy.ExecuteError("Spatial Analyst license is unavailable.")

        self.input_raster = input_raster
        self.dem_name = Utils.sanitize_path_to_name(self.input_raster)
        
        if not arcpy.Exists(self.input_raster):
            raise FileNotFoundError(f"Input raster '{self.input_raster}' does not exist.")
        
        # Path Setup
        self.filled_path = os.path.join(os.path.dirname(self.input_raster))
        
        if save_path is None:
            self.save_path = os.path.join(os.path.dirname(self.input_raster), "boundary_files")
        else:
            self.save_path = save_path
            
        # Unique Temp Folder
        unique_id = uuid.uuid4().hex[:6]
        self.local_temp = os.path.join(self.save_path, f"{self.dem_name}_temp_{unique_id}")
        self.temp_workspace = self.local_temp
        
        os.makedirs(self.local_temp, exist_ok=True)
        os.makedirs(self.save_path, exist_ok=True)
        
        # Temp FileGDB
        self.temp_gdb = os.path.join(self.local_temp, "intermediate_calc.gdb")
        if not arcpy.Exists(self.temp_gdb):
            try:
                arcpy.management.CreateFileGDB(self.local_temp, "intermediate_calc.gdb")
            except Exception as e:
                print(f"Warning: Could not create temp GDB: {e}")

        arcpy.env.workspace = self.temp_gdb 
        arcpy.env.scratchWorkspace = self.local_temp 
        
        self.message_length = 0
        self.last_fill_iterations = None

        # Set workspace to the NEW unique folder
        arcpy.env.workspace = self.temp_gdb 
        arcpy.env.scratchWorkspace = self.local_temp

    def impute_gaps_idw(self, idw_power=2.0, search_radius_fixed_units=5.0, iterations=1):
        original_settings = {
            "overwriteOutput": arcpy.env.overwriteOutput,
            "extent": arcpy.env.extent,
            "mask": arcpy.env.mask,
            "cellSize": arcpy.env.cellSize  # Backup cell size too
        }
        current_raster = self.input_raster
        temp_points_fc = None

        try:
            arcpy.AddMessage(f"Starting DEM gap filling using IDW for {iterations} iteration(s)...")
            dem_desc = arcpy.Describe(self.input_raster)
            
            # Ensure environment matches input EXACTLY
            arcpy.env.extent = dem_desc.extent
            arcpy.env.overwriteOutput = True
            arcpy.env.cellSize = dem_desc.meanCellHeight # Force cell size in env
            
            # Create Radius Object outside the function call for safety
            # Cast to float to ensure no type confusion
            radius_obj = arcpy.sa.RadiusFixed(float(search_radius_fixed_units))

            for i in range(iterations):
                arcpy.AddMessage(f"IDW iteration {i+1} of {iterations}...")
                
                temp_point_name = f"dem_idw_pts_{i}"
                if arcpy.Exists(self.temp_gdb):
                    temp_points_fc = os.path.join(self.temp_gdb, temp_point_name)
                else:
                    temp_points_fc = os.path.join(self.local_temp, f"{temp_point_name}.shp")

                if arcpy.Exists(temp_points_fc):
                    arcpy.management.Delete(temp_points_fc)

                # OPTIMIZATION: Removed redundant SetNull. 
                # RasterToPoint automatically ignores NoData values.
                arcpy.conversion.RasterToPoint(
                    in_raster=current_raster,
                    out_point_features=temp_points_fc,
                    raster_field="VALUE"
                )

                if int(arcpy.management.GetCount(temp_points_fc).getOutput(0)) == 0:
                    arcpy.AddError("No valid data points generated. Cannot run IDW.")
                    return None
                
                # FIX: Use Keyword Arguments to ensure correct mapping
                idw_interpolated_raster = arcpy.sa.Idw(
                    in_point_features=temp_points_fc,
                    z_field="grid_code",
                    cell_size=self.input_raster, # Explicitly pass cell size
                    power=float(idw_power),
                    search_radius=radius_obj
                )

                # Fill gaps: If current is Null, use IDW. Else use current.
                current_raster = arcpy.sa.Con(
                    arcpy.sa.IsNull(current_raster),
                    idw_interpolated_raster,
                    current_raster
                )

                if temp_points_fc and arcpy.Exists(temp_points_fc):
                    try: arcpy.management.Delete(temp_points_fc)
                    except: pass

            return current_raster

        except Exception as e:
            # Print full traceback to help debug underlying IDW crashes
            import traceback
            traceback.print_exc()
            arcpy.AddError(f"IDW Error: {str(e)}")
            return None
        finally:
            # Restore all settings
            arcpy.env.extent = original_settings["extent"]
            arcpy.env.mask = original_settings["mask"]
            arcpy.env.overwriteOutput = original_settings["overwriteOutput"]
            arcpy.env.cellSize = original_settings["cellSize"]

    def impute_gaps_focal_statistics(self, iterations, snap_raster=None, kernel_size=9):
        """
        Fills gaps using FocalStatistics.
        kernel_size=9 is roughly equivalent to 4 iterations of a 3x3 kernel but faster.
        """
        desc = arcpy.Describe(self.input_raster)
        converted_input = os.path.join(self.temp_gdb, "focal_convert_temp")
        
        try:
            arcpy.env.snapRaster = snap_raster if snap_raster else self.input_raster
            arcpy.env.cellSize = self.input_raster
            arcpy.env.outputCoordinateSystem = desc.spatialReference
            arcpy.env.extent = desc.extent
            arcpy.env.overwriteOutput = True
            
            # Copy to GDB raster
            arcpy.management.CopyRaster(self.input_raster, converted_input)
            
            fc_filled_raster = Raster(converted_input)

            for iteration in range(iterations):
                message = f"Pass {iteration + 1} of filling with Focal Stats (Size {kernel_size})..."
                self.message_length = Utils.print_progress(message, self.message_length)
                
                # FIX: Use the larger kernel size here
                neighborhood = NbrRectangle(kernel_size, kernel_size, "CELL")
                
                focal_mean = FocalStatistics(fc_filled_raster, neighborhood, "MEAN", "DATA")
                
                # Include the sanitization fix we discussed earlier
                clean_focal = SetNull(Abs(focal_mean) > 100000, focal_mean)
                
                fc_filled_raster = Con(IsNull(fc_filled_raster), clean_focal, fc_filled_raster)
                
            return fc_filled_raster

        except Exception as e:
            print(f"Error in Focal Statistics: {e}")
            raise

    def fill_internal_gaps_arcpy(self, method="IDW", iterations=1, idw_power=2.0, search_radius=5.0, dissolved_polygon=None, overwrite=True):
        self.last_fill_iterations = iterations
        
        if not os.path.exists(self.filled_path):
            os.makedirs(self.filled_path, exist_ok=True)

        original_settings = {
            "mask": arcpy.env.mask,
            "workspace": arcpy.env.workspace,
            "overwrite": arcpy.env.overwriteOutput
        }

        try:
            # --- Environment Setup ---
            dem_desc = arcpy.Describe(self.input_raster)
            arcpy.env.outputCoordinateSystem = dem_desc.spatialReference
            arcpy.env.snapRaster = self.input_raster
            arcpy.env.cellSize = self.input_raster
            arcpy.env.overwriteOutput = True
            arcpy.env.workspace = self.temp_gdb
            arcpy.env.extent = dem_desc.extent
            
            # IDW needs a mask to constrain interpolation; FocalStats needs None to "grow" into gaps
            arcpy.env.mask = dissolved_polygon if method == "IDW" else None

            # --- Path Logic & Backup ---
            if overwrite:
                backup_path = os.path.join(self.filled_path, f"{self.dem_name}_original.tif")
                if not arcpy.Exists(backup_path):
                    arcpy.AddMessage(f"Creating backup of original at: {backup_path}")
                    arcpy.Copy_management(self.input_raster, backup_path)
                
                final_target_path = self.input_raster
                # Save to GDB first to avoid file lock issues during processing
                temp_output = os.path.join(self.temp_gdb, "filled_raster_temp")
            else:
                final_target_path = os.path.join(self.filled_path, f"{self.dem_name}_filled.tif")
                temp_output = final_target_path

            # --- Processing ---
            filled_raster = None
            if method == "FocalStatistics":
                filled_raster = self.impute_gaps_focal_statistics(iterations=iterations)
                
                # Clip expansion back to the study area if a polygon is provided
                if dissolved_polygon and arcpy.Exists(dissolved_polygon):
                    arcpy.AddMessage("Clipping Focal Statistics expansion to boundary...")
                    filled_raster = arcpy.sa.ExtractByMask(filled_raster, dissolved_polygon)
                    
            elif method == "IDW":
                filled_raster = self.impute_gaps_idw(idw_power, search_radius, iterations)
            
            if filled_raster is None:
                raise RuntimeError(f"Gap filling failed: {method}")

            # --- Save and Replace Logic ---
            arcpy.AddMessage(f"Saving temporary result to workspace...")
            filled_raster.save(temp_output)
            
            # If overwriting, we must delete the active pointer to the raster object 
            # to release the file lock on 'self.input_raster'
            if overwrite:
                del filled_raster 
                arcpy.AddMessage(f"Overwriting original DEM: {final_target_path}")
                arcpy.CopyRaster_management(temp_output, final_target_path)
                # Cleanup temp GDB item
                arcpy.Delete_management(temp_output)
            
            arcpy.AddMessage("Gap fill complete.")
            return final_target_path

        except Exception as e:
            arcpy.AddError(f"Fill Error: {e}")
            raise 
        finally:
            # Restore original environment
            arcpy.env.mask = original_settings["mask"]
            arcpy.env.workspace = original_settings["workspace"]
            arcpy.env.overwriteOutput = original_settings["overwrite"]

    @staticmethod
    def fill_chunk_focal_stats(chunk_path, iterations=1, kernel_size=9): # <--- Added kernel_size=9 default
        """Static method for tiled filling with adjustable kernel."""
        import arcpy
        from arcpy.sa import FocalStatistics, Con, IsNull, Raster, NbrRectangle, Abs, SetNull
        
        try:
            filled_path = chunk_path.replace(".tif", "_filled.tif")
            arcpy.env.overwriteOutput = True
            arcpy.env.snapRaster = chunk_path
            arcpy.env.cellSize = chunk_path
            arcpy.env.extent = chunk_path
            
            current_raster = Raster(chunk_path)
            
            for i in range(iterations):
                # FIX: Use the kernel_size argument
                neighborhood = NbrRectangle(kernel_size, kernel_size, "CELL")
                focal_mean = FocalStatistics(current_raster, neighborhood, "MEAN", "DATA")
                
                # Sanitize artifacts
                clean_focal = SetNull(Abs(focal_mean) > 100000, focal_mean)
                
                current_raster = Con(IsNull(current_raster), clean_focal, current_raster)
            
            # Final Clean
            final_clean = SetNull(Abs(current_raster) > 100000, current_raster)
            
            final_clean.save(filled_path)
            del current_raster, final_clean, focal_mean, clean_focal
            
            if arcpy.Exists(filled_path):
                arcpy.ClearWorkspaceCache_management()
                try:
                    arcpy.management.Delete(chunk_path)
                    arcpy.management.Rename(filled_path, chunk_path)
                    return chunk_path
                except:
                    return filled_path
            return chunk_path
        except Exception as e:
            print(f"Error filling chunk: {e}")
            return chunk_path
        
    @staticmethod
    def fill_chunk_idw(chunk_path, iterations=1, power=2.0, search_radius=5.0):
        """
        Static method for tiled filling using IDW. 
        Uses a temporary File Geodatabase to avoid Shapefile 2GB limits.
        """
        
        # Setup paths
        chunk_dir = os.path.dirname(chunk_path)
        unique_id = uuid.uuid4().hex[:6]
        
        # FIX: Create a temp GDB for this chunk to avoid Shapefile size limits
        temp_gdb_name = f"temp_idw_{unique_id}.gdb"
        temp_gdb_path = os.path.join(chunk_dir, temp_gdb_name)
        
        # Temp Feature Class inside the GDB
        temp_points = os.path.join(temp_gdb_path, "temp_pts")
        filled_path = chunk_path.replace(".tif", "_filled.tif")

        try:
            # Create the temp GDB
            if not arcpy.Exists(temp_gdb_path):
                arcpy.management.CreateFileGDB(chunk_dir, temp_gdb_name)
                
            arcpy.env.overwriteOutput = True
            arcpy.env.snapRaster = chunk_path
            arcpy.env.cellSize = chunk_path
            arcpy.env.extent = chunk_path
            
            current_raster = Raster(chunk_path)
            radius_obj = RadiusFixed(float(search_radius))

            for i in range(iterations):
                # 1. Convert Valid Data to Points (Stores in GDB now)
                arcpy.conversion.RasterToPoint(
                    in_raster=current_raster,
                    out_point_features=temp_points,
                    raster_field="VALUE"
                )
                
                if int(arcpy.management.GetCount(temp_points).getOutput(0)) == 0:
                    break

                # 2. Run IDW
                idw_raster = Idw(
                    in_point_features=temp_points,
                    z_field="grid_code",
                    cell_size=chunk_path,
                    power=float(power),
                    search_radius=radius_obj
                )
                
                # 3. Sanitize and Fill
                clean_idw = SetNull(Abs(idw_raster) > 100000, idw_raster)
                current_raster = Con(IsNull(current_raster), clean_idw, current_raster)
                
                # Cleanup points inside GDB for next iteration
                if arcpy.Exists(temp_points):
                    arcpy.management.Delete(temp_points)

            # Final Save
            final_clean = SetNull(Abs(current_raster) > 100000, current_raster)
            final_clean.save(filled_path)
            
            # Explicit Memory Cleanup
            del current_raster, final_clean, clean_idw, idw_raster, radius_obj
            
            if arcpy.Exists(filled_path):
                arcpy.ClearWorkspaceCache_management()
                try:
                    arcpy.management.Delete(chunk_path)
                    arcpy.management.Rename(filled_path, chunk_path)
                    return chunk_path
                except:
                    return filled_path
            return chunk_path

        except Exception as e:
            print(f"Error filling chunk with IDW: {e}")
            return chunk_path
        finally:
            # CLEANUP: Delete the entire temp GDB folder
            if os.path.exists(temp_gdb_path):
                try:
                    shutil.rmtree(temp_gdb_path)
                except:
                    pass
    
    def create_binary_mask_from_boundary(self, boundary_shapefile):
        """
        Create binary mask. 
        Uses CreateConstantRaster + ExtractByMask (Spatial Analyst) which is often 
        more stable on massive datasets than FeatureToRaster (Conversion).
        """
        message = "Creating binary mask..."
        self.message_length = Utils.print_progress(message, self.message_length)
        
        binary_mask = os.path.join(self.save_path, f"{self.dem_name}_binary_mask.tif")
        
        if os.path.exists(binary_mask):
            arcpy.Delete_management(binary_mask)

        # Sanity Check
        if int(arcpy.management.GetCount(boundary_shapefile).getOutput(0)) == 0:
            raise RuntimeError("Boundary shapefile is empty. Cannot create mask.")

        try:
            # Prepare Environment
            desc = arcpy.Describe(self.input_raster)
            arcpy.env.overwriteOutput = True
            arcpy.env.snapRaster = self.input_raster
            arcpy.env.cellSize = self.input_raster
            arcpy.env.extent = desc.extent
            arcpy.env.workspace = self.temp_gdb
            
            # CRITICAL: LZW Compression
            arcpy.env.compression = "LZW"

            # 1. Generating Constant Raster
            message = "Generating Constant Raster..."
            self.message_length = Utils.print_progress(message, self.message_length)
            
            # Create a virtual raster of 1s with exact dimensions of input
            # arguments: constant_value, data_type, cell_size, extent
            const_raster = arcpy.sa.CreateConstantRaster(
                1, 
                "INTEGER", 
                self.input_raster, 
                self.input_raster
            )

            message = "Extracting by Mask (Rasterizing)..."
            self.message_length = Utils.print_progress(message, self.message_length)
            
            # Use the boundary polygon to mask the constant raster
            # Result: 1 inside polygon, NoData outside
            masked_raster = arcpy.sa.ExtractByMask(const_raster, boundary_shapefile)

            message = "Saving final binary mask (0/1)..."
            self.message_length = Utils.print_progress(message, self.message_length)
            
            # Convert NoData to 0 for the final binary mask
            # Con(IsNull(raster), 0, 1) -> If NoData, set 0. Else set 1.
            # Note: masked_raster is 1 or NoData.
            binary_mask_raster = Con(IsNull(masked_raster), 0, 1)
            binary_mask_raster.save(binary_mask)
            
        except Exception as e:
            print(f"Error creating binary mask: {e}")
            raise
        finally:
            arcpy.ClearEnvironment("compression")

        return binary_mask

    def get_data_boundary(self, min_area=50):
        """
        Generate boundary vectors and mask.
        Uses GDB Feature Classes for intermediate vectors to avoid Shapefile 2GB limit.
        """
        input_raster = self.input_raster
        
        # Intermediate paths in GDB (Safe for large data)
        temp_integer_raster = os.path.join(self.temp_gdb, "integer_temp")
        temp_polygon = os.path.join(self.temp_gdb, "temp_polygon_fc")
        dissolved_polygon_fc = os.path.join(self.temp_gdb, "dissolved_polygon_fc")
        cleaned_polygon_fc = os.path.join(self.temp_gdb, "cleaned_polygon_fc")
        
        # Final Output Shapefile (Simplified, so likely safe size)
        shapefile_dir = os.path.join(self.save_path, "boundary_shapefile")
        os.makedirs(shapefile_dir, exist_ok=True)
        final_shapefile_path = os.path.join(shapefile_dir, f"{self.dem_name}_boundary.shp")

        original_settings = {
            "overwriteOutput": arcpy.env.overwriteOutput,
            "extent": arcpy.env.extent,
            "snapRaster": arcpy.env.snapRaster,
            "cellSize": arcpy.env.cellSize,
            "workspace": arcpy.env.workspace,
            "scratchWorkspace": arcpy.env.scratchWorkspace
        }
        
        try:
            message = "Setting environment..."
            self.message_length = Utils.print_progress(message, self.message_length)
            
            arcpy.env.overwriteOutput = True
            arcpy.env.extent = input_raster
            arcpy.env.snapRaster = input_raster
            arcpy.env.cellSize = input_raster
            arcpy.env.workspace = self.temp_gdb
            arcpy.env.scratchWorkspace = self.temp_gdb

            message = "Converting raster to integer type..."
            self.message_length = Utils.print_progress(message, self.message_length)
            
            # Use SetNull to preserve NoData properly
            data_mask = arcpy.sa.SetNull(arcpy.sa.IsNull(self.input_raster), 1)
            # Save to GDB Raster
            temp_integer_raster_obj = arcpy.sa.Int(data_mask)
            temp_integer_raster_obj.save(temp_integer_raster)

            message = "Converting raster to polygons (GDB)..."
            self.message_length = Utils.print_progress(message, self.message_length)
            # Conversion to FileGDB Feature Class supports huge outputs
            arcpy.RasterToPolygon_conversion(temp_integer_raster, temp_polygon, "NO_SIMPLIFY", "VALUE")

            message = "Dissolving polygons..."
            self.message_length = Utils.print_progress(message, self.message_length)
            arcpy.Dissolve_management(temp_polygon, dissolved_polygon_fc)

            message = "Filtering small polygons..."
            self.message_length = Utils.print_progress(message, self.message_length)
            
            if arcpy.CheckProduct("ArcInfo") == "Available":
                print("Using ArcInfo EliminatePolygonPart_management")
                arcpy.EliminatePolygonPart_management(
                    in_features=dissolved_polygon_fc,
                    out_feature_class=cleaned_polygon_fc,
                    part_area=min_area,
                    part_option="ANY"
                )
            else:
                arcpy.CopyFeatures_management(dissolved_polygon_fc, cleaned_polygon_fc)

            if os.path.exists(final_shapefile_path):
                arcpy.Delete_management(final_shapefile_path)

            message = "Saving final boundary shapefile..."
            self.message_length = Utils.print_progress(message, self.message_length)
            arcpy.CopyFeatures_management(cleaned_polygon_fc, final_shapefile_path)

            # Do this before rasterization to prevent errors from Dissolve artifacts
            message = "Repairing boundary geometry..."
            self.message_length = Utils.print_progress(message, self.message_length)
            arcpy.management.RepairGeometry(final_shapefile_path)

            # Create the binary mask from the simplified shapefile
            binary_mask_path = self.create_binary_mask_from_boundary(final_shapefile_path)

        except Exception as e:
            print(f"Error generating boundary: {e}")
            raise 

        finally:
            # Cleanup temp GDB items to save space
            for item in [temp_integer_raster, temp_polygon, dissolved_polygon_fc, cleaned_polygon_fc]:
                if arcpy.Exists(item):
                    try: arcpy.Delete_management(item)
                    except: pass
            
            # Restore Env
            for k, v in original_settings.items():
                setattr(arcpy.env, k, v)

        return binary_mask_path, final_shapefile_path
    
    def get_data_boundary_MajorityFilter(self):
        """
        Generate binary mask and boundary vector.
        
        NEW APPROACH (Raster-First):
        1. Generate Binary Mask directly from Raster (Fast).
        2. Clean Mask using MajorityFilter (Removes noise).
        3. Convert Clean Mask to Polygon (Fast).
        
        This avoids the "Raster->Vector->Raster" roundtrip that crashes on large files.
        """
        input_raster = self.input_raster
        
        # Paths
        binary_mask_path = os.path.join(self.save_path, f"{self.dem_name}_binary_mask.tif")

        
        shapefile_dir = os.path.join(self.save_path, "boundary_shapefile")
        os.makedirs(shapefile_dir, exist_ok=True)
        final_shapefile_path = os.path.join(shapefile_dir, f"{self.dem_name}_boundary.shp")

        # GDB Paths
        raw_mask_raster = os.path.join(self.temp_gdb, "raw_mask")
        clean_mask_raster = os.path.join(self.temp_gdb, "clean_mask")
        temp_polygon = os.path.join(self.temp_gdb, "temp_boundary_poly")

        original_settings = {
            "overwriteOutput": arcpy.env.overwriteOutput,
            "extent": arcpy.env.extent,
            "snapRaster": arcpy.env.snapRaster,
            "cellSize": arcpy.env.cellSize,
            "compression": arcpy.env.compression
        }
        
        try:
            if not arcpy.Exists(binary_mask_path):
                message = "Generating Raster Mask..."
                self.message_length = Utils.print_progress(message, self.message_length)
                
                arcpy.env.overwriteOutput = True
                arcpy.env.extent = input_raster
                arcpy.env.snapRaster = input_raster
                arcpy.env.cellSize = input_raster
                arcpy.env.compression = "LZW"

                # 1. Create Raw Mask (1=Data, 0=NoData)
                # Use Con(IsNull) logic which is extremely fast and stable
                # If IsNull(raster) is True -> 0 (Background)
                # Else -> 1 (Data)
                raw_mask_obj = Con(IsNull(Raster(input_raster)), 0, 1)
                
                # Save raw mask temporarily
                raw_mask_obj.save(raw_mask_raster)

                # 2. Clean the Mask (Remove "Salt and Pepper" noise)
                # MajorityFilter helps remove single stray pixels (islands)
                # Repeated application mimics removing small polygons
                message = "Cleaning Mask (Removing noise)..."
                self.message_length = Utils.print_progress(message, self.message_length)
                
                # Apply MajorityFilter (4 neighbors, Majority)
                # This replaces isolated 1s with 0s and isolated 0s with 1s
                cleaned_obj = MajorityFilter(raw_mask_obj, "EIGHT", "MAJORITY")
                
                # Optional: Run BoundaryClean to smooth edges (blocky -> smooth)
                # cleaned_obj = BoundaryClean(cleaned_obj, "NO_SORT", "TWO_WAY")
                
                # Save final TIFF mask
                message = "Saving Binary Mask TIFF..."
                self.message_length = Utils.print_progress(message, self.message_length)
                cleaned_obj.save(binary_mask_path)

            else:
                cleaned_obj = Raster(binary_mask_path)
            
            if not arcpy.Exists(final_shapefile_path):
                # --- FIX: Filter in Raster Domain (Avoids SQL) ---
                message = "Isolating data areas..."
                self.message_length = Utils.print_progress(message, self.message_length)
                
                # Set 0s to NoData. Result contains ONLY 1s.
                # RasterToPolygon will ignores NoData, so we get only the boundary we want.
                only_ones_raster = SetNull(cleaned_obj == 0, cleaned_obj)
                
                # 3. Generate Vector Boundary
                message = "Converting to Boundary Shapefile..."
                self.message_length = Utils.print_progress(message, self.message_length)
                
                # Convert
                arcpy.conversion.RasterToPolygon(only_ones_raster, temp_polygon, "SIMPLIFY", "Value")

                # Save final shapefile
                if os.path.exists(final_shapefile_path):
                    arcpy.Delete_management(final_shapefile_path)
                    
                arcpy.management.CopyFeatures(temp_polygon, final_shapefile_path)
                
                message = "Boundary generation complete."
                self.message_length = Utils.print_progress(message, self.message_length)

        except Exception as e:
            print(f"Error generating boundary: {e}")
            raise 

        finally:
            # Cleanup GDB items
            for item in [raw_mask_raster, clean_mask_raster, temp_polygon]:
                if arcpy.Exists(item):
                    try: arcpy.Delete_management(item)
                    except: pass
            
            # Restore Env
            for k, v in original_settings.items():
                setattr(arcpy.env, k, v)

        return binary_mask_path, final_shapefile_path
import traceback
import os, tempfile
from utils import Utils
import arcpy
from arcpy.sa import *
import uuid
import rasterio
from rasterio import features
import json

class Inpainter:
    def __init__(self, input_raster, save_path=None):

        if arcpy.CheckExtension("Spatial") == "Available":
            arcpy.CheckOutExtension("Spatial")
        else:
            arcpy.AddError("Spatial Analyst extension is not available.")
            raise arcpy.ExecuteError("Spatial Analyst license is unavailable.")
        
        self.input_raster = input_raster
        self.dem_name = Utils.sanitize_path_to_name(self.input_raster)
        dem_desc = arcpy.Describe(self.input_raster)
        self.original_spatial_ref = dem_desc.spatialReference
        self.original_cell_size = arcpy.management.GetRasterProperties(self.input_raster, "CELLSIZEX").getOutput(0)
        self.original_snap_raster = self.input_raster

        # Set env variables
        arcpy.env.outputCoordinateSystem = self.original_spatial_ref
        arcpy.env.snapRaster = self.original_snap_raster
        arcpy.env.cellSize = self.original_cell_size

        if not arcpy.Exists(self.input_raster):
            raise FileNotFoundError(f"Input raster '{self.input_raster}' does not exist.")
        
        # Path Setup
        self.filled_path = os.path.join(os.path.dirname(self.input_raster), "filled")
        
        if save_path is None:
            self.save_path = os.path.join(os.path.dirname(self.input_raster), "boundary_files")
        else:
            self.save_path = save_path
            
        unique_id = uuid.uuid4().hex[:6]
        self.local_temp = os.path.join(self.save_path, f"{self.dem_name}_temp_{unique_id}")
        os.makedirs(self.local_temp, exist_ok=True)
        os.makedirs(self.save_path, exist_ok=True)
        
        # Temp FileGDB (Critical for large data)
        self.temp_gdb = os.path.join(self.local_temp, "intermediate_calc.gdb")
        if not arcpy.Exists(self.temp_gdb):
            try:
                arcpy.management.CreateFileGDB(self.local_temp, "intermediate_calc.gdb")
            except Exception as e:
                print(f"Warning: Could not create temp GDB: {e}")

        self.message_length = 0
        self.last_fill_iterations = None

        # Set workspace to the NEW unique folder
        arcpy.env.workspace = self.temp_gdb 
        arcpy.env.scratchWorkspace = self.local_temp

    def impute_dem_gaps_idw(self, idw_power=2.0, search_radius_fixed_units=5.0, iterations=1):
        original_settings = {
            "overwriteOutput": arcpy.env.overwriteOutput,
            "extent": arcpy.env.extent,
            "mask": arcpy.env.mask
        }
        current_raster = self.input_raster
        temp_points_fc = None

        try:
            arcpy.AddMessage(f"Starting DEM gap filling using IDW for {iterations} iteration(s)...")
            dem_desc = arcpy.Describe(self.input_raster)
            arcpy.env.extent = dem_desc.extent
            arcpy.env.overwriteOutput = True

            for i in range(iterations):
                arcpy.AddMessage(f"IDW iteration {i+1} of {iterations}...")
                
                temp_point_name = f"dem_idw_pts_{i}"
                if arcpy.Exists(self.temp_gdb):
                    temp_points_fc = os.path.join(self.temp_gdb, temp_point_name)
                else:
                    temp_points_fc = os.path.join(self.local_temp, f"{temp_point_name}.shp")

                if arcpy.Exists(temp_points_fc):
                    arcpy.management.Delete(temp_points_fc)

                arcpy.conversion.RasterToPoint(
                    in_raster=arcpy.sa.SetNull(current_raster, current_raster, "VALUE IS NULL"),
                    out_point_features=temp_points_fc,
                    raster_field="VALUE"
                )

                if int(arcpy.management.GetCount(temp_points_fc).getOutput(0)) == 0:
                    arcpy.AddError("No valid data points generated. Cannot run IDW.")
                    return None
                
                idw_interpolated_raster = arcpy.sa.Idw(
                    temp_points_fc, "grid_code", idw_power,
                    arcpy.sa.RadiusFixed(search_radius_fixed_units)
                )

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
            arcpy.AddError(f"IDW Error: {str(e)}")
            return None
        finally:
            arcpy.env.extent = original_settings["extent"]
            arcpy.env.mask = original_settings["mask"]
            arcpy.env.overwriteOutput = original_settings["overwriteOutput"]

    def impute_gaps_focal_statistics(self, iterations, snap_raster=None):
        desc = arcpy.Describe(self.input_raster)
        # Use GDB for temp raster to handle size
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
                message = f"Pass {iteration + 1} of filling with Focal Stats..."
                self.message_length = Utils.print_progress(message, self.message_length)
                neighborhood = NbrRectangle(3, 3, "CELL")
                focal_mean = FocalStatistics(fc_filled_raster, neighborhood, "MEAN", "DATA")
                fc_filled_raster = Con(IsNull(fc_filled_raster), focal_mean, fc_filled_raster)
                
            return fc_filled_raster

        except Exception as e:
            print(f"Error in Focal Statistics: {e}")
            raise

    def fill_internal_gaps_arcpy(self, method="IDW", iterations=1, idw_power=2.0, search_radius=5.0, dissolved_polygon=None):
        self.last_fill_iterations = iterations
        
        if not os.path.exists(self.filled_path):
            os.makedirs(self.filled_path, exist_ok=True)

        original_settings = {
            "mask": arcpy.env.mask,
            "workspace": arcpy.env.workspace
        }

        try:
            dem_desc = arcpy.Describe(self.input_raster)
            arcpy.env.outputCoordinateSystem = dem_desc.spatialReference
            arcpy.env.snapRaster = self.input_raster
            arcpy.env.cellSize = self.input_raster
            arcpy.env.overwriteOutput = True
            arcpy.env.workspace = self.temp_gdb # Use GDB
            arcpy.env.extent = dem_desc.extent
            
            # Mask is set for IDW, but None for FocalStats to allow expansion
            arcpy.env.mask = dissolved_polygon if method == "IDW" else None
            
            filled_raster_path = os.path.join(self.filled_path, f"{self.dem_name}_filled.tif")
            
            filled_raster = None
            if method == "FocalStatistics":
                filled_raster = self.impute_gaps_focal_statistics(iterations=iterations)
                # Clip expansion if mask available
                if dissolved_polygon and arcpy.Exists(dissolved_polygon):
                    arcpy.AddMessage("Clipping Focal Statistics expansion...")
                    filled_raster = arcpy.sa.ExtractByMask(filled_raster, dissolved_polygon)
            elif method == "IDW":
                filled_raster = self.impute_dem_gaps_idw(idw_power, search_radius, iterations)
            
            if filled_raster is None:
                raise RuntimeError(f"Gap filling failed: {method}")
                
            filled_raster.save(filled_raster_path)
            arcpy.AddMessage(f"Saved filled DEM to: '{filled_raster_path}'")
            return filled_raster_path

        except Exception as e:
            print(f"Fill Error: {e}")
            raise 
        finally:
            arcpy.env.mask = original_settings["mask"]
            arcpy.env.workspace = original_settings["workspace"]

    def create_binary_mask(self, boundary_shapefile):
        """
        Create binary mask using Rasterio Tiled Rasterization.
        Fixes the 'No valid geometry' error by converting Esri JSON to GeoJSON.
        """
        import rasterio
        from rasterio import features
        import json
        
        message = "Creating binary mask (Tiled/Rasterio)..."
        self.message_length = Utils.print_progress(message, self.message_length)
        
        binary_mask = os.path.join(self.save_path, f"{self.dem_name}_binary_mask.tif")
        
        if os.path.exists(binary_mask):
            try: arcpy.Delete_management(binary_mask)
            except: os.remove(binary_mask)

        # 1. Read Boundary Geometries and Convert to GeoJSON
        polygons = []
        try:
            with arcpy.da.SearchCursor(boundary_shapefile, ["SHAPE@"]) as cursor:
                for row in cursor:
                    # Get Esri JSON
                    esri_json = json.loads(row[0].JSON)
                    
                    # --- FIX: Convert Esri JSON to GeoJSON for Rasterio ---
                    # Rasterio expects {"type": "Polygon", "coordinates": ...}
                    # Esri provides {"rings": ...}
                    
                    if 'rings' in esri_json:
                        geo_json = {
                            'type': 'Polygon',
                            'coordinates': esri_json['rings']
                        }
                        polygons.append(geo_json)
                    elif 'paths' in esri_json:
                        # Fallback for lines (unlikely for mask, but safe)
                        geo_json = {
                            'type': 'LineString',
                            'coordinates': esri_json['paths']
                        }
                        polygons.append(geo_json)
                    # ------------------------------------------------------

        except Exception as e:
            raise RuntimeError(f"Failed to read boundary shapefile: {e}")

        if not polygons:
            raise RuntimeError("Boundary shapefile is empty or could not be converted to GeoJSON.")

        try:
            # 2. Open Input Raster to get Grid Definition
            with rasterio.open(self.input_raster) as src:
                meta = src.meta.copy()
                meta.update({
                    'driver': 'GTiff',
                    'dtype': 'uint8',
                    'count': 1,
                    'compress': 'lzw',
                    'nodata': 0,
                    'BIGTIFF': 'YES',
                    'tiled': True
                })

                # 3. Create Output and Rasterize Window-by-Window
                with rasterio.open(binary_mask, 'w', **meta) as dst:
                    
                    total_blocks = len(list(dst.block_windows(1)))
                    processed = 0
                    
                    for ji, window in dst.block_windows(1):
                        processed += 1
                        if processed % 500 == 0:
                            msg = f"Rasterizing mask block {processed}/{total_blocks}..."
                            self.message_length = Utils.print_progress(msg, self.message_length)

                        window_transform = dst.window_transform(window)
                        
                        # Rasterize using the converted GeoJSON polygons
                        mask_arr = features.rasterize(
                            shapes=polygons,
                            out_shape=(window.height, window.width),
                            transform=window_transform,
                            fill=0,
                            default_value=1,
                            dtype='uint8'
                        )
                        
                        dst.write(mask_arr, 1, window=window)

            print() 
            return binary_mask

        except Exception as e:
            print(f"\nError creating binary mask: {e}")
            if os.path.exists(binary_mask):
                try: os.remove(binary_mask)
                except: pass
            raise
    
    def create_binary_mask_arcpy(self, boundary_shapefile):
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

            # 1. Repair Geometry
            # Complex polygons from Dissolve can cause rasterization crashes
            message = "Repairing boundary geometry..."
            self.message_length = Utils.print_progress(message, self.message_length)
            arcpy.management.RepairGeometry(boundary_shapefile)

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
            binary_mask_path = self.create_binary_mask(final_shapefile_path)

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

    @staticmethod
    def fill_chunk_focal_stats(chunk_path, iterations=1):
        """Static method for tiled filling."""
        import arcpy
        from arcpy.sa import FocalStatistics, Con, IsNull, Raster, NbrRectangle
        
        try:
            filled_path = chunk_path.replace(".tif", "_filled.tif")
            arcpy.env.overwriteOutput = True
            arcpy.env.snapRaster = chunk_path
            arcpy.env.cellSize = chunk_path
            arcpy.env.extent = chunk_path
            
            current_raster = Raster(chunk_path)
            for i in range(iterations):
                neighborhood = NbrRectangle(3, 3, "CELL")
                focal_mean = FocalStatistics(current_raster, neighborhood, "MEAN", "DATA")
                current_raster = Con(IsNull(current_raster), focal_mean, current_raster)
            
            current_raster.save(filled_path)
            del current_raster
            
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
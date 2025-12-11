import os
import json
import numpy as np
import arcpy
from arcpy.sa import *
from osgeo import gdal
from utils import Utils

class Landforms:
    """
    Refactored Class for generating Bathymorphons/Geomorphons.
    Includes static methods for chunk-based processing.
    """
    
    # Static lookup tables for classification
    COMBINED_LOOKUP = {
        "10c": {
            (0, 0): "FL", (0, 1): "FL", (0, 2): "FL", (0, 3): "FS", (0, 4): "FS", (0, 5): "VL", (0, 6): "VL", (0, 7): "VL", (0, 8): "PT",
            (1, 0): "FL", (1, 1): "FL", (1, 2): "FS", (1, 3): "FS", (1, 4): "FS", (1, 5): "VL", (1, 6): "VL", (1, 7): "VL",
            (2, 0): "FL", (2, 1): "SH", (2, 2): "SL", (2, 3): "SL", (2, 4): "CN", (2, 5): "CN", (2, 6): "VL",
            (3, 0): "SH", (3, 1): "SH", (3, 2): "SL", (3, 3): "SL", (3, 4): "SL", (3, 5): "CN",
            (4, 0): "SH", (4, 1): "SH", (4, 2): "CV", (4, 3): "SL", (4, 4): "SL",
            (5, 0): "RI", (5, 1): "RI", (5, 2): "CV", (5, 3): "CV",
            (6, 0): "RI", (6, 1): "RI", (6, 2): "RI",
            (7, 0): "RI", (7, 1): "RI",
            (8, 0): "PK"
        },
        "6c": {
            (0, 0): "FL", (0, 1): "FL", (0, 2): "FL", (0, 3): "FS", (0, 4): "FS", (0, 5): "VL", (0, 6): "VL", (0, 7): "VL", (0, 8): "VL",
            (1, 0): "FL", (1, 1): "FL", (1, 2): "FS", (1, 3): "FS", (1, 4): "FS", (1, 5): "VL", (1, 6): "VL", (1, 7): "VL",
            (2, 0): "FL", (2, 1): "SH", (2, 2): "SL", (2, 3): "SL", (2, 4): "SL", (2, 5): "VL", (2, 6): "VL",
            (3, 0): "SH", (3, 1): "SH", (3, 2): "SL", (3, 3): "SL", (3, 4): "SL", (3, 5): "SL",
            (4, 0): "SH", (4, 1): "SH", (4, 2): "SL", (4, 3): "SL", (4, 4): "SL",
            (5, 0): "RI", (5, 1): "RI", (5, 2): "RI", (5, 3): "SL",
            (6, 0): "RI", (6, 1): "RI", (6, 2): "RI",
            (7, 0): "RI", (7, 1): "RI",
            (8, 0): "RI"
        },
        "5c": {
            (0, 0): "FL", (0, 1): "FL", (0, 2): "FL", (0, 3): "SL", (0, 4): "VL", (0, 5): "VL", (0, 6): "VL", (0, 7): "VL", (0, 8): "VL",
            (1, 0): "FL", (1, 1): "FL", (1, 2): "SL", (1, 3): "SL", (1, 4): "VL", (1, 5): "VL", (1, 6): "VL", (1, 7): "VL",
            (2, 0): "FL", (2, 1): "SL", (2, 2): "SL", (2, 3): "SL", (2, 4): "SL", (2, 5): "VL", (2, 6): "VL",
            (3, 0): "SL", (3, 1): "SL", (3, 2): "SL", (3, 3): "SL", (3, 4): "SL", (3, 5): "SL",
            (4, 0): "RI", (4, 1): "RI", (4, 2): "SL", (4, 3): "SL", (4, 4): "SL",
            (5, 0): "RI", (5, 1): "RI", (5, 2): "RI", (5, 3): "SL",
            (6, 0): "RI", (6, 1): "RI", (6, 2): "RI",
            (7, 0): "RI", (7, 1): "RI",
            (8, 0): "PK"
        },
        "4c": {
            (0, 0): "FL", (0, 1): "FL", (0, 2): "FL", (0, 3): "SL", (0, 4): "VL", (0, 5): "VL", (0, 6): "VL", (0, 7): "VL", (0, 8): "VL",
            (1, 0): "FL", (1, 1): "FL", (1, 2): "SL", (1, 3): "SL", (1, 4): "VL", (1, 5): "VL", (1, 6): "VL", (1, 7): "VL",
            (2, 0): "FL", (2, 1): "SL", (2, 2): "SL", (2, 3): "SL", (2, 4): "SL", (2, 5): "VL", (2, 6): "VL",
            (3, 0): "SL", (3, 1): "SL", (3, 2): "SL", (3, 3): "SL", (3, 4): "SL", (3, 5): "SL",
            (4, 0): "RI", (4, 1): "RI", (4, 2): "SL", (4, 3): "SL", (4, 4): "SL",
            (5, 0): "RI", (5, 1): "RI", (5, 2): "RI", (5, 3): "SL",
            (6, 0): "RI", (6, 1): "RI", (6, 2): "RI",
            (7, 0): "RI", (7, 1): "RI",
            (8, 0): "RI"
        }
    }
    
    ABRV_TO_INT = {
        "FL": 1, "PK": 2, "RI": 3, "SH": 4, "CV": 5, "SL": 6, "CN": 7, "FS": 8, "VL": 9, "PT": 10, "Unknown": 0
    }

    def __init__(self):
        pass

    @staticmethod
    def calculate_landforms_chunk(chunk_path, angle_threshold=1, distance_units="METERS", search_distance=10, skip_distance=5, z_unit="METER"):
        """
        Runs the ArcPy GeomorphonLandforms tool on a specific chunk file.
        Returns the result as a NumPy array (int32).
        Uses positional arguments to avoid keyword errors.
        """
        import uuid
        
        # Define paths
        temp_10c_name = f"geo10c_{uuid.uuid4().hex[:8]}.tif"
        temp_raw_name = f"georaw_{uuid.uuid4().hex[:8]}.tif"
        
        temp_10c_path = os.path.join(os.path.dirname(chunk_path), temp_10c_name)
        temp_raw_path = os.path.join(os.path.dirname(chunk_path), temp_raw_name)

        # Capture Global Environment (to restore later)
        global_extent = arcpy.env.extent
        global_snap = arcpy.env.snapRaster

        try:
            if arcpy.CheckExtension("Spatial") == "Available":
                arcpy.CheckOutExtension("Spatial")

            # This ensures output grid matches input grid 1:1, ignoring global masks/extents
            arcpy.env.extent = chunk_path
            arcpy.env.snapRaster = chunk_path       

            # Order: in_raster, search_distance, skip_distance, flatness_threshold, flatness_angle
            out_raster_10c  = GeomorphonLandforms(
                chunk_path,      # in_raster
                temp_raw_path,
                angle_threshold, distance_units,
                search_distance, skip_distance, z_unit
            )
            
            # Save the main return (10-class) to disk
            out_raster_10c.save(temp_10c_path)
            
            # --- Helper to read with GDAL ---
            def read_gdal(path):
                if not os.path.exists(path): return None
                ds = gdal.Open(path)
                if ds is None: return None
                band = ds.GetRasterBand(1)
                arr = band.ReadAsArray()
                nd = band.GetNoDataValue()
                if nd is not None:
                    arr[arr == nd] = 0
                ds = None
                return arr.astype(np.int32)  

            # Read both results
            raw_array = read_gdal(temp_raw_path)
            class10_array = read_gdal(temp_10c_path)
            
            return raw_array, class10_array

        except Exception as e:
            print(f"Error generating landforms for chunk {chunk_path}: {e}")
            return None, None
            
        finally:
            # Restore Environment
            arcpy.env.extent = global_extent
            arcpy.env.snapRaster = global_snap
            
            # Cleanup
            for p in [temp_10c_path, temp_raw_path]:
                if os.path.exists(p):
                    try: arcpy.management.Delete(p)
                    except: 
                        try: os.remove(p)
                        except: pass

    @staticmethod
    def classify_single_pixel(number, lookup_table):
        """Helper for vectorization."""
        if number is None or number <= 0: 
            return 0 
        
        ternary_patterns = []
        n = int(number)
        while n > 0:
            ternary_patterns.append(n % 3)
            n //= 3
        
        while len(ternary_patterns) < 8:
            ternary_patterns.append(0) 
        
        minuses = ternary_patterns.count(0)
        pluses = ternary_patterns.count(2)
        
        land_abrv = lookup_table.get((minuses, pluses), "Unknown")
        return Landforms.ABRV_TO_INT.get(land_abrv, 0)

    @staticmethod
    def classify_chunk(raw_array, classes="10c"):
        """
        Takes a raw geomorphon numpy array and returns a classified numpy array.
        """
        lookup = Landforms.COMBINED_LOOKUP.get(classes, Landforms.COMBINED_LOOKUP[classes])
        
        def _classify(x):
            return Landforms.classify_single_pixel(x, lookup)
        
        vectorized_func = np.vectorize(_classify, otypes=[np.int32])
        classified_array = vectorized_func(raw_array)
        
        return classified_array
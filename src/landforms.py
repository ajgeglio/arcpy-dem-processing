import os
import numpy as np
import arcpy
from arcpy.sa import *
from osgeo import gdal
import gc

class Landforms:
    """
    Refactored Class for generating Bathymorphons/Geomorphons.
    Includes static methods for chunk-based processing and LUT optimization.
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

    _LUT_CACHE = {}

    def __init__(self):
        pass

    @staticmethod
    def calculate_landforms_chunk(chunk_path, angle_threshold=1, distance_units="METERS", search_distance=10, skip_distance=5, z_unit="METER"):
        """
        Runs the ArcPy GeomorphonLandforms tool on a specific chunk file.
        Returns the result as a NumPy array (int32).
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

            # Ensure output grid matches input grid 1:1
            arcpy.env.extent = chunk_path
            arcpy.env.snapRaster = chunk_path       

            # Tool Execution (5-Arg Signature)
            # We omit z_unit and distance_units as positional args to prevent errors
            out_raster_10c  = GeomorphonLandforms(
                chunk_path,      # 1. in_raster
                temp_raw_path,   # 2. out_geomorphon_raster (RAW)
                angle_threshold, # 3. flatness_threshold
                distance_units,
                search_distance, # 4. search_distance
                skip_distance,    # 5. skip_distance
                z_unit
            )
            
            # Save the main return (10-class) to disk
            out_raster_10c.save(temp_10c_path)
            
            # Release File Locks
            del out_raster_10c
            arcpy.ClearWorkspaceCache_management()
            gc.collect()

            # --- Helper to read with GDAL ---
            def read_gdal(path):
                if not os.path.exists(path): return None
                try:
                    ds = gdal.Open(path)
                    if ds is None: return None
                    band = ds.GetRasterBand(1)
                    arr = band.ReadAsArray()
                    nd = band.GetNoDataValue()
                    if nd is not None:
                        arr[arr == nd] = 0
                    ds = None 
                    return arr.astype(np.int32)
                except Exception as e:
                    print(f"GDAL Read Error on {path}: {e}")
                    return None

            # Read both results
            raw_array = read_gdal(temp_raw_path)
            class10_array = read_gdal(temp_10c_path)
            
            # Sync Backgrounds: If 10c is background(0), force Raw to 0
            # If 10-Class says it's background (0), mark Raw as -1.
            if raw_array is not None and class10_array is not None:
                if raw_array.shape == class10_array.shape:
                    background_mask = (class10_array == 0)
                    raw_array[background_mask] = -1

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
    def _build_lut(classes="10c"):
        """
        Builds a fast NumPy lookup array for mapping Raw Integers -> Class Integers.
        """
        if classes in Landforms._LUT_CACHE:
            return Landforms._LUT_CACHE[classes]

        # Safety Fallback: Use 10c if requested class not found
        lookup_dict = Landforms.COMBINED_LOOKUP.get(classes, Landforms.COMBINED_LOOKUP["10c"])
        
        # Max value for 8 ternary digits is 6561 (3^8)
        max_val = 6562 
        lut_array = np.zeros(max_val, dtype=np.int32)
        
        for i in range(max_val):
            # Calculate ternary counts manually
            n = i
            zeros = 0
            twos = 0
            
            digits_count = 0
            while n > 0 and digits_count < 8:
                rem = n % 3
                if rem == 0: zeros += 1
                elif rem == 2: twos += 1
                n //= 3
                digits_count += 1
            
            # Handle padding (leading zeros)
            zeros += (8 - digits_count)
            
            # Lookup
            land_abrv = lookup_dict.get((zeros, twos), "Unknown")
            lut_array[i] = Landforms.ABRV_TO_INT.get(land_abrv, 0)
            
        Landforms._LUT_CACHE[classes] = lut_array
        return lut_array

    @staticmethod
    def classify_single_pixel(number, lookup_table):
        """
        Helper for vectorization.
        Matches original proven logic for ternary pattern decoding.
        Not used but keeping for reference
        """
        # Original logic: Handle negatives/NaNs/None as NoData
        if number is None or number <= 0 or np.isnan(number):
            return 0 
        
        ternary_patterns = []
        n = int(number)
        
        # Standard ternary conversion
        while n > 0:
            ternary_patterns.append(n % 3)
            n //= 3
        
        # --- MATCHING ORIGINAL LOGIC ---
        # 1. Reverse to correct order
        ternary_patterns.reverse()
        
        # 2. Pad with zeros at the BEGINNING (MSB)
        while len(ternary_patterns) < 8:
            ternary_patterns.insert(0, 0)
        # -------------------------------
        
        minuses = ternary_patterns.count(0)
        pluses = ternary_patterns.count(2)
        
        # Lookup using tuple key (minuses, pluses)
        # Note: Your lookup table keys are tuples: (minuses, pluses)
        # Ensure your COMBINED_LOOKUP keys match this format.
        
        land_abrv = lookup_table.get((minuses, pluses), "Unknown")
        return Landforms.ABRV_TO_INT.get(land_abrv, 0)
                                     
    @staticmethod
    def classify_chunk(raw_array, classes="10c"):
        """
        Classifies a chunk using fast NumPy array indexing (LUT).
        """
        if raw_array is None: return None
        
        # 1. Get/Build LUT
        lut = Landforms._build_lut(classes)
        
        # 2. Prepare Output
        # Initialize with 0 (NoData)
        classified_array = np.zeros_like(raw_array, dtype=np.int32)
        
        # 3. Identify Valid Data
        # Valid data is >= 0 (Peaks are 0) AND within LUT bounds
        # Background is -1 (from calculate_landforms_chunk)
        valid_mask = (raw_array >= 0) & (raw_array < len(lut))
        
        # 4. Apply LUT only to valid pixels
        classified_array[valid_mask] = lut[raw_array[valid_mask]]
        
        return classified_array
"""
shannon.py

A utility module for calculating Flow Direction and Shannon Entropy (Diversity Index)
on DEM data. This module contains a class `ShannonDerivatives` which serves as a 
repository for various implementation strategies (NumPy, SciPy, Numba, ArcPy, scikit-image).

Dependencies:
    - numpy
    - scipy
    - numba (for _numba methods)
    - arcpy & osgeo.gdal (for _arcpy methods)
    - skimage (for _skimage methods)
"""

import os
import numpy as np
from scipy import ndimage
from scipy.stats import entropy
from numba import njit

# Try imports for optional dependencies to prevent immediate import errors
try:
    import arcpy
    from arcpy.sa import *
    from osgeo import gdal
except ImportError:
    arcpy = None
    gdal = None

try:
    from skimage.filters.rank import entropy as skimage_entropy
    from skimage.morphology import footprint_rectangle as square
except ImportError:
    skimage_entropy = None
    square = None


# ==============================================================================
# MODULE LEVEL HELPERS & JIT FUNCTIONS
# ==============================================================================

def _calculate_window_entropy_scipy(window):
    """
    Callback for scipy.ndimage.generic_filter.
    Calculates Shannon Entropy (base 2) ignoring 0 (NoData).
    """
    valid_pixels = window[window != 0]
    if valid_pixels.size == 0:
        return 0.0
    _, counts = np.unique(valid_pixels, return_counts=True)
    return entropy(counts, base=2)

def _flow_direction_window_numpy(window):
    """
    Callback for scipy.ndimage.generic_filter (NumPy approach).
    """
    # D8 Direction encoding
    direction_encoding = np.array([[32, 64, 128],
                                   [16, 0, 1],
                                   [8, 4, 2]])
    if np.isnan(window).all():
        return 0
    diff = window[1, 1] - window
    max_diff = np.max(diff)
    return direction_encoding[diff == max_diff].sum() if max_diff > 0 else 0

@njit(fastmath=False, boundscheck=False)
def _flow_direction_numba_jit(dem_data):
    """
    Low-level Numba implementation for D8 Flow Direction.
    """
    rows, cols = dem_data.shape
    out = np.zeros((rows, cols), dtype=np.int32)
    
    # Offsets implicit in logic: 
    # East=1, SE=2, S=4, SW=8, W=16, NW=32, N=64, NE=128
    
    for i in range(1, rows - 1):
        for j in range(1, cols - 1):
            center = dem_data[i, j]
            if np.isnan(center):
                continue
                
            max_diff = -1.0
            flow_code = 0
            
            # Check 8 neighbors (Unrolled for speed)
            # 1: Right (0, 1)
            val = dem_data[i, j+1]
            if not np.isnan(val):
                diff = center - val
                if diff > max_diff:
                    max_diff = diff
                    flow_code = 1
                elif diff == max_diff and diff > 0:
                    flow_code += 1
            
            # 2: Bottom Right (1, 1)
            val = dem_data[i+1, j+1]
            if not np.isnan(val):
                diff = center - val
                if diff > max_diff:
                    max_diff = diff
                    flow_code = 2
                elif diff == max_diff and diff > 0:
                    flow_code += 2
            
            # 4: Bottom (1, 0)
            val = dem_data[i+1, j]
            if not np.isnan(val):
                diff = center - val
                if diff > max_diff:
                    max_diff = diff
                    flow_code = 4
                elif diff == max_diff and diff > 0:
                    flow_code += 4
            
            # 8: Bottom Left (1, -1)
            val = dem_data[i+1, j-1]
            if not np.isnan(val):
                diff = center - val
                if diff > max_diff:
                    max_diff = diff
                    flow_code = 8
                elif diff == max_diff and diff > 0:
                    flow_code += 8

            # 16: Left (0, -1)
            val = dem_data[i, j-1]
            if not np.isnan(val):
                diff = center - val
                if diff > max_diff:
                    max_diff = diff
                    flow_code = 16
                elif diff == max_diff and diff > 0:
                    flow_code += 16

            # 32: Top Left (-1, -1)
            val = dem_data[i-1, j-1]
            if not np.isnan(val):
                diff = center - val
                if diff > max_diff:
                    max_diff = diff
                    flow_code = 32
                elif diff == max_diff and diff > 0:
                    flow_code += 32
            
            # 64: Top (-1, 0)
            val = dem_data[i-1, j]
            if not np.isnan(val):
                diff = center - val
                if diff > max_diff:
                    max_diff = diff
                    flow_code = 64
                elif diff == max_diff and diff > 0:
                    flow_code += 64

            # 128: Top Right (-1, 1)
            val = dem_data[i-1, j+1]
            if not np.isnan(val):
                diff = center - val
                if diff > max_diff:
                    max_diff = diff
                    flow_code = 128
                elif diff == max_diff and diff > 0:
                    flow_code += 128

            if max_diff > 0:
                out[i, j] = flow_code
                
    return out

@njit(fastmath=False, boundscheck=False)
def _entropy_numba_jit(padded_data, window_size):
    """
    Low-level Numba implementation for Windowed Entropy.
    Uses pre-allocation and sorting to avoid memory fragmentation.
    """
    rows, cols = padded_data.shape
    out_rows = rows - window_size + 1
    out_cols = cols - window_size + 1
    
    result = np.zeros((out_rows, out_cols), dtype=np.float32)
    
    # Pre-allocate buffer outside loops
    window_area = window_size * window_size
    temp_window = np.empty(window_area, dtype=np.int32)
    
    for i in range(out_rows):
        for j in range(out_cols):
            valid_count = 0
            
            # Extract window
            for wi in range(window_size):
                for wj in range(window_size):
                    val = padded_data[i + wi, j + wj]
                    if val != 0: 
                        temp_window[valid_count] = val
                        valid_count += 1
            
            if valid_count == 0:
                result[i, j] = 0.0
                continue
                
            # Sort valid portion
            active_slice = temp_window[:valid_count]
            active_slice.sort()
            
            # Calculate entropy on sorted array
            entropy_val = 0.0
            current_val = active_slice[0]
            current_count = 1
            
            for k in range(1, valid_count):
                if active_slice[k] == current_val:
                    current_count += 1
                else:
                    p = current_count / valid_count
                    entropy_val -= p * np.log2(p)
                    current_val = active_slice[k]
                    current_count = 1
            
            # Tail
            p = current_count / valid_count
            entropy_val -= p * np.log2(p)
            
            result[i, j] = entropy_val

    return result


# ==============================================================================
# MAIN CLASS
# ==============================================================================

class ShannonDerivatives:
    """
    A collection of methods to calculate Flow Direction and Shannon Entropy.
    Includes variations for NumPy, Numba, SciPy, and ArcPy/GDAL.
    """
    def __init__(self, dem_data=None, input_dem_path=None):
        """
        Args:
            dem_data (np.ndarray): 2D numpy array of elevation/depth.
            input_dem_path (str): Filepath to the DEM (Required for ArcPy methods).
        """
        self.dem_data = dem_data
        self.input_dem_path = input_dem_path

    # ==========================================================================
    # FLOW DIRECTION METHODS
    # ==========================================================================

    def calculate_flow_direction_numpy(self):
        """
        Calculate Flow Direction using scipy.ndimage.generic_filter.
        Dependencies: SciPy, NumPy.
        Pros: Simple, pure Python/C backend.
        Cons: Slowest method (Python callback overhead).
        """
        if self.dem_data is None: raise ValueError("dem_data is required.")
        return ndimage.generic_filter(
            self.dem_data,
            function=_flow_direction_window_numpy,
            size=3,
            mode='nearest'
        ).astype(np.int32)

    def calculate_flow_direction_numba(self):
        """
        Calculate Flow Direction using Numba JIT.
        Dependencies: Numba.
        Pros: Extremely fast CPU execution.
        Cons: Fragile with memory layouts or masked arrays.
        """
        if self.dem_data is None: raise ValueError("dem_data is required.")
        
        # Sanitize data for Numba
        data = self.dem_data.copy()
        if np.ma.is_masked(data):
            data = data.filled(np.nan)
        if not data.flags['C_CONTIGUOUS']:
            data = np.ascontiguousarray(data, dtype=np.float32)
            
        return _flow_direction_numba_jit(data)

    def calculate_flow_direction_arcpy(self):
        """
        Calculate Flow Direction using ArcPy Spatial Analyst.
        Dependencies: ArcPy, GDAL.
        Pros: Most robust for large data, handles NoData natively via C++.
        Cons: Disk I/O overhead.
        
        Note: Bypasses 'Pixel Block' memory limits by saving to disk and reading back via GDAL.
        """
        if not arcpy: raise ImportError("ArcPy is not installed.")
        if not self.input_dem_path: raise ValueError("input_dem_path is required for ArcPy method.")

        # Capture Global Environment
        global_extent = arcpy.env.extent
        global_snap = arcpy.env.snapRaster
        global_cell = arcpy.env.cellSize
        temp_flow_path = None

        try:
            # Load Raster via ArcPy
            input_raster = arcpy.Raster(self.input_dem_path)
            
            # CRITICAL: Force environment to local chunk extent
            arcpy.env.extent = input_raster.extent
            arcpy.env.snapRaster = input_raster
            arcpy.env.cellSize = input_raster

            # Sanitize: Handle deep negative values
            clean_raster = arcpy.sa.SetNull(input_raster < -15000.0, input_raster)
            
            # Execute Tool
            flow_dir_raster = arcpy.sa.FlowDirection(clean_raster, force_flow="NORMAL")
            
            # Save to temp file
            temp_flow_path = self.input_dem_path.replace(".tif", "_flow_temp.tif")
            if arcpy.Exists(temp_flow_path):
                arcpy.management.Delete(temp_flow_path)
            flow_dir_raster.save(temp_flow_path)
            
            # Read back via GDAL
            ds = gdal.Open(temp_flow_path)
            if ds is None: raise RuntimeError("GDAL failed to open temp flow raster.")
            flow_dir_arr = ds.GetRasterBand(1).ReadAsArray()
            ds = None 

            return flow_dir_arr.astype(np.int32)

        except Exception as e:
            print(f"ArcPy Flow Direction Error: {e}")
            raise
        finally:
            # Restore Environment
            arcpy.env.extent = global_extent
            arcpy.env.snapRaster = global_snap
            arcpy.env.cellSize = global_cell
            
            # Cleanup
            if temp_flow_path and arcpy.Exists(temp_flow_path):
                try: arcpy.management.Delete(temp_flow_path)
                except: pass
                if os.path.exists(temp_flow_path):
                    try: os.remove(temp_flow_path)
                    except: pass

    # ==========================================================================
    # SHANNON ENTROPY METHODS
    # ==========================================================================

    def calculate_shannon_python_loop(self, flow_direction, window_size):
        """
        Calculate Shannon Index using a pure Python List Comprehension.
        Dependencies: SciPy (for entropy).
        Pros: No complex dependencies, easiest to debug.
        Cons: Extremely slow for large rasters. Use only for small tests.
        """
        rows, cols = flow_direction.shape
        shannon_indices = np.zeros((rows - window_size + 1, cols - window_size + 1))

        # Serial calculation
        results = [
            _calculate_window_entropy_scipy(
                flow_direction[i:i + window_size, j:j + window_size]
            )
            for i in range(rows - window_size + 1)
            for j in range(cols - window_size + 1)
        ]

        # Reshape results back to grid
        for idx, value in enumerate(results):
            i = idx // (cols - window_size + 1)
            j = idx % (cols - window_size + 1)
            shannon_indices[i, j] = value

        return shannon_indices.astype(np.float32)

    def calculate_shannon_scipy(self, flow_direction, window_size):
        """
        Calculate Shannon Index using SciPy Generic Filter.
        Dependencies: SciPy.
        Pros: Robust, handles padding/windowing automatically.
        Cons: Slower than Numba/Skimage due to Python callback.
        """
        return ndimage.generic_filter(
            flow_direction,
            _calculate_window_entropy_scipy,
            size=window_size,
            mode='constant',
            cval=0.0
        ).astype(np.float32)

    def calculate_shannon_numba(self, flow_direction, window_size):
        """
        Calculate Shannon Index using Numba JIT.
        Dependencies: Numba.
        Pros: Very fast.
        Cons: Requires manual padding of input before calling.
        """
        # Manual Padding required for Numba sliding window
        pad = window_size // 2
        padded = np.pad(flow_direction, pad, mode='constant', constant_values=0)
        
        return _entropy_numba_jit(padded, window_size).astype(np.float32)

    def calculate_shannon_skimage(self, flow_direction, window_size):
        """
        Calculate Shannon Index using scikit-image Rank Filter.
        Dependencies: scikit-image.
        Pros: Fastest and most stable (Pure C implementation).
        Cons: Input must be cast to uint8 or uint16.
        """
        if skimage_entropy is None: raise ImportError("scikit-image is not installed.")
        
        # Define Footprint
        footprint = square(window_size)
        
        # Ensure input fits in uint8/uint16 (Flow dir is usually 0-128, so uint8 is safe)
        flow_uint = flow_direction.astype(np.uint8)
        
        return skimage_entropy(flow_uint, footprint=footprint).astype(np.float32)
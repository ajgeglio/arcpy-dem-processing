import arcpy
import numpy as np
from skimage.feature import local_binary_pattern
from scipy import ndimage
from osgeo import gdal
from arcpy.sa import *
from shannon import ShannonDerivatives


class HabitatDerivatives:
    def __init__(self, input_dem, dem_data, use_gdal=True, transform=None, verbose=False):
        self.input_dem = input_dem
        self.dem_data = dem_data
        self.use_gdal = use_gdal
        self.transform = transform  # Optional, used for GDAL processing
        self.verbose = verbose

    def calculate_lbp(self, n_points, radius, method='uniform', nodata=None):
        """
        Generate LBP using raw float data.
        Returns float32 to ensure compatibility with ArcPy mosaicking.
        """
        dem_data = self.dem_data.copy()
        
        # 1. Handle NaNs/NoData
        if np.ma.is_masked(dem_data):
            dem_data = dem_data.filled(np.nan)
        
        valid_mask = ~np.isnan(dem_data)
        
        if valid_mask.any():
            safe_fill = np.nanmin(dem_data)
            dem_data[~valid_mask] = safe_fill
        else:
            # Return float32 array of NaNs
            return np.full_like(dem_data, np.nan, dtype=np.float32)

        # 2. Compute LBP
        lbp = local_binary_pattern(dem_data, P=n_points, R=radius, method=method)

        # 3. Restore NaNs
        # lbp from skimage is float64. We must cast to float32.
        lbp = lbp.astype(np.float32)
        lbp[~valid_mask] = np.nan
        
        if nodata is not None:
             lbp[dem_data == nodata] = np.nan

        return lbp

    def generate_hillshade_gdal(self, hillshade_output):
        """
        Calculate hillshade using GDAL.DEMProcessing with multiple options.
        
        :param dem_file: Path to the input DEM file.
        :param hillshade_output: Path to save the hillshade output file.
        """
        options = gdal.DEMProcessingOptions(
            computeEdges=True,          # Compute edges for smoother output
            azimuth=315,                # Direction of light source (in degrees)
            altitude=45,                # Elevation of the light source (in degrees)
            scale=1.0,                  # Scale factor for vertical exaggeration
            zFactor=1.0                 # Vertical exaggeration
        )
        
        # Perform hillshade processing
        gdal.DEMProcessing(
            destName=hillshade_output,  # Output file path
            srcDS=self.input_dem,             # Input DEM file
            processing="hillshade",     # Specify hillshade calculation
            options=options             # Pass multiple options
        )
        if self.verbose:
            print(f"Hillshade file saved at: {hillshade_output}")

    def calculate_hillshade(self, hillshade_output):
        """
        Calculate hillshade for a DEM file and save the output.

        Parameters:
        dem_file (str): Path to the input DEM file.
        hillshade_output (str): Path to save the hillshade output file.
        """
        if self.use_gdal:
            # Use GDAL to generate hillshade
            self.generate_hillshade_gdal(hillshade_output)
        else:
            raise NotImplementedError("Custom hillshade calculation is not implemented for non-GDAL methods.")

    def calculate_slope_gdal(self):
        """ Calculate slope from a DEM using GDAL."""
        dem_data = self.dem_data
        # Create a temporary in-memory file for the DEM
        mem_drv = gdal.GetDriverByName('MEM')
        src_ds = mem_drv.Create('', dem_data.shape[1], dem_data.shape[0], 1, gdal.GDT_Float32)
        src_ds.SetGeoTransform(self.transform.to_gdal())  # Set geotransformation
        src_ds.GetRasterBand(1).WriteArray(dem_data)
        # Calculate slope
        slope_ds = gdal.DEMProcessing('', src_ds, 'slope', format='MEM', options=['-compute_edges'])# options = ['-zero_for_flat'] option can be added to calculate edges
        slope = slope_ds.GetRasterBand(1).ReadAsArray()
        return slope

    def generate_slope_gdal(self, output_slope):
        """
        Calculate slope from a DEM using GDAL and save to output file.
        """
        gdal.DEMProcessing(
            destName=output_slope,
            srcDS=self.input_dem,
            processing="slope",
            options=gdal.DEMProcessingOptions(computeEdges=True)
        )
        if self.verbose:
            print(f"Slope file saved at: {output_slope}")

    def calculate_slope(self, output_slope):
        if self.use_gdal:
            try:
                self.generate_slope_gdal(output_slope)   
            except Exception as e:
                return self.calculate_slope_gdal()
        else:
            raise NotImplementedError("Custom slope calculation is not implemented for non-GDAL methods.")
        
    def calculate_aspect_gdal(self):
        """ Calculate slope and aspect from a DEM using GDAL. 
        This array contains the aspect values calculated from the Digital Elevation Model (DEM) using the GDAL library.
        Aspect values represent the compass direction that the slope faces. The values are typically in degrees, where:
        0° represents north,         180° represents south,        270° represents west."""
        dem_data = self.dem_data
        # Create a temporary in-memory file for the DEM
        mem_drv = gdal.GetDriverByName('MEM')
        src_ds = mem_drv.Create('', dem_data.shape[1], dem_data.shape[0], 1, gdal.GDT_Float32)
        src_ds.SetGeoTransform(self.transform.to_gdal())  # Set geotransformation
        src_ds.GetRasterBand(1).WriteArray(dem_data)
        # Calculate aspect
        aspect_ds = gdal.DEMProcessing('', src_ds, 'aspect', format='MEM', options=['-compute_edges'])
        aspect = aspect_ds.GetRasterBand(1).ReadAsArray()
        return aspect
    
    def generate_aspect_gdal(self, output_aspect):
        gdal.DEMProcessing(
            destName=output_aspect,
            srcDS=self.input_dem,
            processing="aspect",
            options=gdal.DEMProcessingOptions(computeEdges=True)
        )
        if self.verbose:
            print(f"Aspect file saved at: {output_aspect}")

    def calculate_aspect(self, output_aspect):
        if self.use_gdal:
            try:
                self.generate_aspect_gdal(output_aspect)
            except Exception as e:
                return self.calculate_aspect_gdal()
        else:
            raise NotImplementedError("Custom aspect calculation is not implemented for non-GDAL methods.")

    def calculate_roughness_gdal(self):
        """ Calculate terrain roughness using GDAL. """
        dem_data = self.dem_data
        # Create a temporary in-memory file for the DEM
        mem_drv = gdal.GetDriverByName('MEM')
        src_ds = mem_drv.Create('', dem_data.shape[1], dem_data.shape[0], 1, gdal.GDT_Float32)
        src_ds.SetGeoTransform(self.transform.to_gdal())  # Set geotransformation
        # src_ds.GetRasterBand(1).SetNoDataValue(-9999)  # Set no-data value
        src_ds.GetRasterBand(1).WriteArray(dem_data)
        # Calculate roughness
        roughness_ds = gdal.DEMProcessing('', src_ds, 'Roughness', format='MEM', options = ['-compute_edges'])
        roughness = roughness_ds.GetRasterBand(1).ReadAsArray()
        return roughness
    
    def generate_roughness_gdal(self, output_roughness):
        """ Calculate terrain roughness using GDAL. """
        # Calculate roughness and save to output file
        gdal.DEMProcessing(
            destName=output_roughness,
            srcDS=self.input_dem,
            processing="roughness",
            options=gdal.DEMProcessingOptions(computeEdges=True)
        )
        if self.verbose:
            print(f"Roughness file saved at: {output_roughness}")

    def calculate_roughness_skimage(self):
        """ Calculate terrain roughness as the standard deviation. """
        return ndimage.generic_filter(self.dem_data, np.std, size=16)

    def calculate_roughness(self, output_roughness):
        if self.use_gdal:
            try:
                self.generate_roughness_gdal(output_roughness)
            except Exception as e:
                return self.calculate_roughness_gdal()
        else:
            return self.calculate_roughness_skimage()

    def generate_tpi_gdal(self, output_tpi):
        gdal.DEMProcessing(
            destName=output_tpi,
            srcDS=self.input_dem,
            processing="tpi",
            options=gdal.DEMProcessingOptions(computeEdges=True)
        )
        if self.verbose:
            print(f"TPI file saved at: {output_tpi}")

    def calculate_tpi(self, output_tpi):
        if self.use_gdal:
            self.generate_tpi_gdal(output_tpi)
        else:
            raise NotImplementedError("Custom tpi calculation is not implemented for non-GDAL methods.")

    def generate_tri_gdal(self, output_tri):
        """
        Generate Terrain Ruggedness Index (TRI) from an input DEM using GDAL.

        :param output_tri: Path to save the output TRI file.
        """
        if output_tri:
            gdal.DEMProcessing(
                destName=output_tri,
                srcDS=self.input_dem, 
                processing='TRI', 
                options=gdal.DEMProcessingOptions(computeEdges=True)
            )
            if self.verbose:
                print(f"TRI file saved at: {output_tri}")

    def calculate_tri_numpy(self, output_tri):
        """
        Calculate Terrain Ruggedness Index (TRI) using NumPy.
        Replaces GDAL implementation to prevent 'inf' pixels caused by 
        squaring massive NoData values (Overflow).
        
        Formula: sqrt(sum((neighbor - center)^2))
        """
        # 1. Sanitize Data (The Fix for INF)
        data = self.dem_data.copy()
        
        # Handle Masked Arrays
        if np.ma.is_masked(data):
            data = data.filled(np.nan)
        
        # Replace large negative NoData values (e.g., -3.4e38) with NaN
        # Squaring -3.4e38 causes float32 overflow -> Infinity
        data[data < -15000.0] = np.nan

        # 2. Pad array to handle edges (pad with NaN)
        padded = np.pad(data, 1, mode='constant', constant_values=np.nan)
        
        # 3. Vectorized Neighbor Differences
        center = padded[1:-1, 1:-1]
        sq_diff_sum = np.zeros_like(center)
        
        rows, cols = padded.shape
        
        # Iterate over 3x3 window offsets
        for dy in [-1, 0, 1]:
            for dx in [-1, 0, 1]:
                if dy == 0 and dx == 0:
                    continue
                
                # Create slice for neighbor
                # Logic: padded[1+dy : -1+dy, 1+dx : -1+dx]
                # Adjusting for end-of-array slicing conventions
                y_start, y_end = 1 + dy, (rows - 1 + dy)
                x_start, x_end = 1 + dx, (cols - 1 + dx)
                
                # Handle slice boundaries
                neighbor = padded[
                    y_start : y_end if y_end < rows else None, 
                    x_start : x_end if x_end < cols else None
                ]
                
                # Calculate squared difference
                diff = neighbor - center
                sq_diff = diff * diff
                
                # Treat NaN diffs as 0 so they don't corrupt the sum
                # (Standard TRI behavior: ignore missing neighbors)
                mask = np.isnan(sq_diff)
                sq_diff[mask] = 0
                
                sq_diff_sum += sq_diff

        # 4. Calculate Sqrt
        tri = np.sqrt(sq_diff_sum)
        
        # 5. Restore NoData where center was NoData
        tri[np.isnan(center)] = np.nan
        
        # 6. Final Safety Check for Inf
        tri[np.isinf(tri)] = np.nan
        
        return tri.astype(np.float32)
    
    def calculate_tri(self, output_tri):
        if self.use_gdal:
            self.generate_tri_gdal(output_tri)
        else:
            self.calculate_tri_numpy(output_tri)
            raise NotImplementedError("Custom tri calculation is not implemented for non-GDAL methods.")

    def return_dem_data(self):
        return self.dem_data

    # ==========================================================================
    # SHANNON INDEX (Delegated to ShannonDerivatives)
    # ==========================================================================
    
    def calculate_shannon_index_2d(self, window_size):
        """
        Calculate the Shannon diversity index for each window in a 2D grid.
        
        This method acts as a wrapper/orchestrator. It initializes the 
        ShannonDerivatives utility with the current data, calculates Flow Direction
        using the robust ArcPy strategy, and then calculates Entropy using the 
        fast scikit-image strategy.
        """
        # 1. Initialize the utility class with current data and path
        shannon_tool = ShannonDerivatives(
            dem_data=self.dem_data, 
            input_dem_path=self.input_dem
        )
        
        # 2. Step A: Calculate Flow Direction
        # We use the ArcPy method because it handles the 152GB memory issue 
        # via the SetNull/Disk-I/O strategy we implemented.
        try:
            flow_direction = shannon_tool.calculate_flow_direction_arcpy()
        except ImportError:
            # Fallback if ArcPy isn't available (e.g. testing elsewhere)
            print("Warning: ArcPy not found, falling back to Numba for Flow Direction.")
            flow_direction = shannon_tool.calculate_flow_direction_numba()

        # 3. Step B: Calculate Entropy
        # We use skimage because it is the C-optimized, stable version.
        try:
            shannon_index = shannon_tool.calculate_shannon_skimage(
                flow_direction, 
                window_size
            )
        except ImportError:
             # Fallback to SciPy if skimage is missing
            print("Warning: scikit-image not found, falling back to SciPy for Entropy.")
            shannon_index = shannon_tool.calculate_shannon_scipy(
                flow_direction, 
                window_size
            )
            
        return shannon_index
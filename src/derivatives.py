import numpy as np
from skimage.feature import local_binary_pattern
from scipy import ndimage
from osgeo import gdal
from arcpy.sa import *
from skimage.util import view_as_windows
from shannon import flow_direction_window, fast_entropy_numba

class HabitatDerivatives:
    def __init__(self, input_dem, dem_data, use_gdal, transform=None, verbose=False):
        self.input_dem = input_dem
        self.dem_data = dem_data
        self.use_gdal = use_gdal
        self.transform = transform  # Optional, used for GDAL processing
        self.verbose = verbose

    def calculate_lbp(self, n_points, radius, method='default', nodata=None):
        """
        Generate a Local Binary Pattern (LBP) representation from a DEM using a sliding window approach.

        Parameters:
        dem_data(numpy.ndarray): Input DEM as a 2D NumPy array.
        radius (int): Radius of LBP neighborhood.
        n_points (int): Number of circularly symmetric neighbor points.
        method (str): Method to compute LBP ('default', 'uniform', etc.).

        Returns:
        numpy.ndarray: LBP-transformed DEM with the same shape.
        """
        dem_data = self.dem_data
        # Create a mask for valid data (not nodata, not NaN)
        if nodata is not None:
            valid_mask = (~np.isnan(dem_data)) & (dem_data != nodata)
        else:
            valid_mask = ~np.isnan(dem_data)

        # Normalize the DEM data to the range 0-255 for valid data only
        valid_data = dem_data[valid_mask]
        if valid_data.size > 0:
            min_val = valid_data.min()
            max_val = valid_data.max()
            if max_val > min_val:
                normalized_data = np.zeros_like(dem_data, dtype=np.uint8)
                normalized_data[valid_mask] = ((dem_data[valid_mask] - min_val) / (max_val - min_val) * 255).astype(np.uint8)
            else:
                normalized_data = np.zeros_like(dem_data, dtype=np.uint8)
        else:
            normalized_data = np.zeros_like(dem_data, dtype=np.uint8)

        # Compute LBP for the entire DEM
        lbp = local_binary_pattern(normalized_data, P=n_points, R=radius, method=method)

        # Set LBP to 0 (or np.nan) where data is invalid
        lbp[~valid_mask] = 0  # or use np.nan if preferred and dtype allows

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

    def calculate_tri(self, output_tri):
        if self.use_gdal:
            self.generate_tri_gdal(output_tri)
        else:
            raise NotImplementedError("Custom tri calculation is not implemented for non-GDAL methods.")

    def calculate_flow_direction(self):
        """
        Calculate the Flow Direction similar to ArcGIS (Spatial Analyst).
        https://pro.arcgis.com/en/pro-app/latest/tool-reference/spatial-analyst/how-flow-direction-works.htm
        """

        # Use ndimage.generic_filter to apply flow_direction_window over the DEM
        flow_direction = ndimage.generic_filter(
            self.dem_data,
            function=flow_direction_window,
            size=3,
            mode='nearest'
        ).astype(np.int32)

        return flow_direction

    def calculate_shannon_index_2d(self, window_size):
        """
        Fast calculation of the Shannon diversity index for each window in a 2D grid.
        Uses skimage view_as_windows and numba for speed.
        """
        flow_direction = self.calculate_flow_direction()
        # Pad to handle borders
        pad = window_size // 2
        padded = np.pad(flow_direction, pad, mode='edge')
        # Create sliding windows
        windows = view_as_windows(padded, (window_size, window_size))
        # Use numba-accelerated entropy calculation
        shannon_indices = fast_entropy_numba(windows)
        return shannon_indices

    def return_dem_data(self):
        return self.dem_data

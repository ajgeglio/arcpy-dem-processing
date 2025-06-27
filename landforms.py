import rasterio
import os, json
import numpy as np
import matplotlib.pyplot as plt
import arcpy
from arcpy.sa import *
from utils import Utils
from derivatives import HabitatDerivatives
from landforms import GeomorphonLandforms

class Landforms:
    def __init__(self, input_dem_path, geomorphons_directory="..\\geomorphons", local_workspace="..\\local_workspace"):
        self.geomorphons_directory = geomorphons_directory
        self.local_workspace = local_workspace
        self.input_dem = input_dem_path
        
        # Extract DEM name and define output raster paths
        self.input_dem_name = Utils().sanitize_path_to_name(input_dem_path)
        self.geomorphons_folder = os.path.join(self.geomorphons_directory, self.input_dem_name)
        self.bathymorphons_raster = f"{self.input_dem_name}_bathymorphons.tif"

        
        # Ensure output and workspace directories exist
        os.makedirs(self.geomorphons_folder, exist_ok=True)
        os.makedirs(self.local_workspace, exist_ok=True)
        arcpy.env.overwriteOutput = True
        arcpy.env.workspace = self.local_workspace
        arcpy.env.scratchWorkspace = self.local_workspace
        # --- 0. License Check ---
        if arcpy.CheckExtension("Spatial") == "Available":
            arcpy.CheckOutExtension("Spatial")
            arcpy.AddMessage("Spatial Analyst extension checked out.")
        else:
            arcpy.AddError("Spatial Analyst extension is not available. Cannot proceed.")
            # Raising an exception here is good practice if this function is part of a larger script
            # that might depend on its successful execution.
            raise arcpy.ExecuteError("Spatial Analyst license is unavailable.")

    def calculate_geomorphon_landforms(self, angle_threshold=1, distance_units="METERS", search_distance=10, skip_distance=5, z_unit="METER"):
        """
        Calculates geomorphons over a specified search distance, skipping cells within a defined distance of the target cell.
        The calculated geomorphons are classified into landforms and saved as a raster.

        Parameters:
        - input_dem (str): Path to the input DEM file.
        - angle_threshold (int): Angle threshold for flat terrain classification.
        - distance_units (str): Units for distance (e.g., "METERS").
        - search_distance (int): Search distance for geomorphon calculation.
        - skip_distance (int): Skip distance for geomorphon calculation.
        - z_unit (str): Units for elevation (e.g., "METER").

        Returns:
        - str: Path to the saved geomorphons raster file.
        """
        # Set environment settings
        arcpy.env.workspace = self.local_workspace

        # Check out the ArcGIS Spatial Analyst extension license
        arcpy.CheckOutExtension("Spatial")

        out_raster = os.path.join(self.geomorphons_folder, f"{self.input_dem_name}_landforms_orig.tif")
        
        # Execute the GeomorphonLandforms tool
        out_geomorphon_landforms = GeomorphonLandforms(
            self.input_dem, self.bathymorphons_raster, angle_threshold, distance_units,
            search_distance, skip_distance, z_unit
        )
        # Save the output raster
        out_geomorphon_landforms.save(out_raster)

    def analyze_raster_data(raster):
        # Open the raster file
        with rasterio.open(raster) as src:
            # Read the raster data as a numpy array
            raster_data = src.read(1)  # Read the first band
            # Get raster metadata
            metadata = src.meta
            # Read the metadata tags
            tags = src.tags()
            print("Tags in the DEM file:")
            print(json.dumps(tags, indent=4))  # Pretty print the tags

        # Mask the nodata value
        masked_data = raster_data[raster_data != metadata['nodata']]
        # Analyze the raster values
        min_value = np.min(masked_data)
        max_value = np.max(masked_data)
        unique_values = len(np.unique(masked_data))

        print(f"Raster Metadata: {metadata}")
        print(f"Minimum Value: {min_value}")
        print(f"Maximum Value: {max_value}")
        print(f"Number of unique values ignoring nodata: {unique_values}")
        # Create the histogram
        plt.hist(masked_data, bins=np.arange(1, 12) - 0.5, color='blue', edgecolor='black')
        plt.xticks(range(1, 11))  # Ensure x-axis is 1-10 by 1
        plt.title("Histogram of Raster Data")
        plt.xlabel("Value")
        plt.ylabel("Frequency")
        plt.show()
    
    def classify_landform_from_bathymorphon(self, number, classes="10c"):
        """
        Function to calculate potential local ternary patterns for a given integer number.
        Then pass the ternary pattern to the lookup table to classify the landform.
        The lookup table is based on the number of minuses and pluses in the ternary pattern.

        Parameters:
        number (int): The input integer number.

        Returns:
        list: A list of potential local ternary patterns with a length of 8.
        """
        combined_lookup_table = {
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
                (4, 0): "SH", (4, 1): "SH", (4, 2): "SL", (4, 3): "SL", (4, 4): "SL",
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
        landform_abrv_to_int_dict = {
            "FL": 1, "PK": 2, "RI": 3, "SH": 4, "CV": 5, "SL": 6, "CN": 7, "FS": 8, "VL": 9, "PT": 10
        }
        if number < 0:
            raise ValueError("The input number must be a non-negative integer.")
        
        ternary_patterns = []
        while number > 0:
            ternary_patterns.append(number % 3)
            number //= 3
        
        # Reverse the list to get the ternary pattern in the correct order
        ternary_patterns.reverse()
        
        # Ensure the result has exactly 8 numbers by padding with zeros at the beginning
        while len(ternary_patterns) < 8:
            ternary_patterns.insert(0, 0)
        # count the minuses and plusses based on the local ternary patterns
        minuses, pluses = ternary_patterns.count(0), ternary_patterns.count(2)
        lookup_tuple = (minuses, pluses)

        land_abrv = combined_lookup_table[classes].get(lookup_tuple, "Unknown")
        land_int = landform_abrv_to_int_dict.get(land_abrv, "Unknown")
        return land_int

    def classify_bathymorphons(self, classes="10c"):
        """
        Process a raster file, classify its values using a given function, and save the modified raster.

        Parameters:
        input_raster_path (str): Path to the input raster file.
        value_to_label (dict): Dictionary mapping classified values to labels.
        classes (str): Classification scheme to use (default is "10c").

        Returns:
        str: Path to the saved output raster file.
        """
        value_to_label = {1:"Flat", 2:"Peak", 3:"Ridge", 4:"Shoulder", 5:"Spur (Convex slope)", 6:"Slope", 7:"Hollow (Concave slope)", 8:"Footslope", 9:"Valley", 10:"Pit"}
        bathymorphon_raster_path = os.path.join(self.local_workspace, self.bathymorphons_raster)
        with rasterio.open(bathymorphon_raster_path) as src:
            # Read the raster data as a numpy array
            raster_data = src.read(1)  # Read the first band
            # Get raster metadata
            metadata = src.meta

        raster_name = self.input_dem_name
        # Mask the nodata value
        masked_data = raster_data[raster_data != metadata['nodata']]

        # Apply the classification function to the masked data
        vectorized_classify = np.vectorize(lambda x: self.classify_landform_from_bathymorphon(x, classes=classes))
        masked_data = vectorized_classify(masked_data)

        # Update the raster data with the classified values
        raster_data[raster_data != metadata['nodata']] = masked_data

        # Define the output file path
        output_file = os.path.join(self.geomorphons_folder, f"{raster_name}_{classes}.tif")

        # Write the modified raster data to the output file
        with rasterio.open(output_file, 'w', **metadata) as dst:
            # Write the modified raster data to the output file
            dst.write(raster_data, 1)
            # Update metadata with value_to_label as a JSON string
            dst.update_tags(**{"value_to_label": json.dumps(value_to_label)})
        converted_tiff = HabitatDerivatives(input_dem=self.input_dem , output_folder=self.geomorphons_directory).convert_tiff_to_gdal_raster(output_file, compress=False)
        return converted_tiff
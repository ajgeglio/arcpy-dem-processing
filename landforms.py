import rasterio
import os, json
import numpy as np
import matplotlib.pyplot as plt
import arcpy
import shutil
import arcpy.management
from arcpy.sa import *
from utils import Utils
from landforms import GeomorphonLandforms

class Landforms:
    def __init__(self, input_raster_path):
        self.input_dem = input_raster_path
        self.input_dem_name = Utils.sanitize_path_to_name(input_raster_path)

        # Determine the base directory for outputs (one level up from input_raster_path)
        base_output_dir = os.path.dirname(os.path.dirname(input_raster_path))
        self.geomorphons_directory = os.path.join(base_output_dir, "geomorphons")
        self.local_workspace = self.geomorphons_directory # Use the same for workspace
        
        # --- 0. License Check ---
        if arcpy.CheckExtension("Spatial") == "Available":
            arcpy.CheckOutExtension("Spatial")
            arcpy.AddMessage("Spatial Analyst extension checked out.")
        else:
            arcpy.AddError("Spatial Analyst extension is not available. Cannot proceed.")
            raise arcpy.ExecuteError("Spatial Analyst license is unavailable.")

        # Ensure output and workspace directories exist
        os.makedirs(self.geomorphons_directory, exist_ok=True)
        os.makedirs(self.local_workspace, exist_ok=True)

        # Set arcpy environment settings
        arcpy.env.workspace = self.local_workspace
        arcpy.env.overwriteOutput = True
        arcpy.env.scratchWorkspace = self.local_workspace

        # Store original raster env settings
        dem_desc = arcpy.Describe(self.input_dem)
        self.original_spatial_ref = dem_desc.spatialReference
        self.original_cell_size = float(arcpy.management.GetRasterProperties(self.input_dem, "CELLSIZEX").getOutput(0))
        self.original_extent = self.input_dem
        self.original_snap_raster = self.input_dem

        # Set arcpy.env variables to match input DEM
        # These will be explicitly reset in methods that need specific alignments
        arcpy.env.outputCoordinateSystem = self.original_spatial_ref
        arcpy.env.snapRaster = self.original_snap_raster
        arcpy.env.cellSize = self.original_cell_size
        arcpy.env.extent = self.original_extent

        # Extract DEM name and define output raster paths
        self.bathymorphons_raster = f"{self.input_dem_name}_bathymorphons.tif"
        self.bathymorphon_raster_path = os.path.join(self.local_workspace, self.bathymorphons_raster)

    def __del__(self):
        if arcpy.CheckExtension("Spatial") == "Available":
            try:
                arcpy.CheckInExtension("Spatial")
                arcpy.AddMessage("Spatial Analyst extension checked in.")
            except Exception as e:
                arcpy.AddWarning(f"Could not check in Spatial Analyst extension: {e}")

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
        # Set environment settings to match original raster before processing
        arcpy.env.outputCoordinateSystem = self.original_spatial_ref
        arcpy.env.snapRaster = self.original_snap_raster # Snap to its own grid initially
        arcpy.env.cellSize = self.original_cell_size
        arcpy.env.extent = self.original_extent
        arcpy.env.workspace = self.local_workspace
        arcpy.env.scratchWorkspace = self.local_workspace

        arcpy.CheckOutExtension("Spatial")

        raw_geomorphon_output_path  = os.path.join(self.geomorphons_directory, f"{self.input_dem_name}_landforms_orig.tif")
        
        # Execute the GeomorphonLandforms tool
        out_geomorphon_landforms = GeomorphonLandforms(
            self.input_dem, self.bathymorphon_raster_path, angle_threshold, distance_units,
            search_distance, skip_distance, z_unit
        )

        # Check if the output raster is valid
        if out_geomorphon_landforms is None:
            arcpy.AddError("GeomorphonLandforms tool did not return a valid output.")
            return None

        out_geomorphon_landforms.save(raw_geomorphon_output_path)
        final_output_raster_path = raw_geomorphon_output_path

        if Raster(raw_geomorphon_output_path) is None:
            arcpy.AddError(f"Failed to save the geomorphons raster: {raw_geomorphon_output_path}")
            raise Exception("Geomorphon fill failed.")
       
        # At this point, final_output_raster_path holds the geomorphons raster,
        # snapped to the master grid (if applicable) and ready for final clipping.
        return final_output_raster_path

    @staticmethod
    def analyze_geomorphon_data(raster):
        """
        Analyze a classified geomorphon raster and plot a histogram of terrain classes.

        Parameters:
        raster (str): Path to the classified raster file.
        """
        with rasterio.open(raster) as src:
            raster_data = src.read(1, masked=True)
            metadata = src.meta
            tags = src.tags()

        value_to_label = json.loads(tags["value_to_label"])
        label_map = {int(k): v for k, v in value_to_label.items()}

        colors = plt.cm.get_cmap("tab10", 10).colors

        # Mask out nodata values
        valid_mask = raster_data != metadata.get('nodata', None)
        masked_data = raster_data[valid_mask]

        fig, ax = plt.subplots()
        for i, class_id in enumerate(sorted(label_map.keys())):
            class_data = masked_data[masked_data == class_id]
            ax.hist(class_data,
                    bins=np.arange(class_id - 0.5, class_id + 1.5, 1),
                    color=colors[i % len(colors)],
                    edgecolor='black',
                    label=label_map[class_id])

        ax.legend(title="Terrain Classes")
        ax.set_xlabel("DEM Class ID")
        ax.set_ylabel("Frequency")
        ax.set_title("Histogram of Terrain Classes")
        plt.xticks(range(1, 11))
        plt.tight_layout()
        plt.show()

    
    def classify_landform_from_bathymorphon(self, number, classes="10c"):
        """
        https://www.hydroffice.org/manuals/bress/stable/user_manual_landforms_tab.html#bathymorphons
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
        landform_abrv_to_int_dict = {
            "FL": 1, "PK": 2, "RI": 3, "SH": 4, "CV": 5, "SL": 6, "CN": 7, "FS": 8, "VL": 9, "PT": 10
        }
        if number < 0 or number is None or np.isnan(number):
            # Return the input value (nodata) or np.nan for nodata cells
            return number
        
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
        # Set environment settings to match original raster before processing
        arcpy.env.outputCoordinateSystem = self.original_spatial_ref
        arcpy.env.snapRaster = self.original_snap_raster
        arcpy.env.cellSize = self.original_cell_size
        value_to_label = {1:"Flat", 2:"Peak", 3:"Ridge", 4:"Shoulder", 5:"Spur (Convex slope)", 6:"Slope", 7:"Hollow (Concave slope)", 8:"Footslope", 9:"Valley", 10:"Pit"}
        
        with rasterio.open(self.bathymorphon_raster_path) as src:
            # Read the raster data as a numpy array
            raster_data = src.read(1)  # Read the first band with masked=True
            # Get raster metadata
            metadata = src.meta

        raster_name = self.input_dem_name
        # Mask the nodata value
        masked_data = raster_data[raster_data != metadata["nodata"]]

        # Apply the classification function to the masked data
        vectorized_classify = np.vectorize(lambda x: self.classify_landform_from_bathymorphon(x, classes=classes))
        masked_data = vectorized_classify(masked_data)

        # Update the raster data with the classified values
        raster_data[raster_data != metadata['nodata']] = masked_data

        # Define the output file path
        output_file = os.path.join(self.geomorphons_directory, f"{raster_name}_{classes}.tif")

        # Write the modified raster data to the output file
        with rasterio.open(output_file, 'w', **metadata) as dst:
            # Write the modified raster data to the output file
            dst.write(raster_data, 1)
            # Update metadata with value_to_label as a JSON string
            dst.update_tags(**{"value_to_label": json.dumps(value_to_label)})

        # Reset arcpy.env variables after processing
        arcpy.env.outputCoordinateSystem = self.original_spatial_ref
        arcpy.env.snapRaster = self.original_snap_raster
        arcpy.env.cellSize = self.original_cell_size

        # Return the path to the saved output raster file
        return output_file
       
    def generate_landforms(self):
        
        landforms = Landforms(self.input_dem)
        output_raster_path = landforms.calculate_geomorphon_landforms()
        if output_raster_path is None:
            print("Failed to calculate geomorphon landforms.")
            return
        raster_directory = os.path.dirname(output_raster_path)
        # # calculate the 10class solution
        output_file10c = landforms.classify_bathymorphons(classes="10c")
        print(f"Modified raster data saved to {output_file10c}")

        # # calculate the 6class solution
        output_file6c = landforms.classify_bathymorphons(classes="6c")
        print(f"Modified raster data saved to {output_file6c}")

        # # calculate the 5class solution
        output_file5c = landforms.classify_bathymorphons(classes="5c")
        print(f"Modified raster data saved to {output_file5c}")

        # # calculate the 4class solution
        output_file4c = landforms.classify_bathymorphons(classes="4c")
        print(f"Modified raster data saved to {output_file4c}")
        
        return raster_directory
    
# example usage
if __name__ == "__main__":
    path = "path/to/dem_raster.tif"
    landforms = Landforms(input_raster_path=path)
    landforms.calculate_geomorphon_landforms()
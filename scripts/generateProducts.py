import arcpy
import argparse
import sys
import os
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "src"))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)
from utils import Utils
from processDem import ProcessDem
from extents import GetExtents
from rasterMasking import RasterMasking
from landforms import Landforms

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Process DEM to generate habitat derivatives.")
    parser.add_argument("--input_dem", type=str, required=True, help="Path to the input DEM file.")
    parser.add_argument("--input_bs", type=str, required=False, help="Path to the input backscatter file.")
    parser.add_argument("--divisions", type=int, default = None, help="Divide the height by this to run tile processing of DEM file, automatic overalap computed.")
    parser.add_argument("--shannon_window", type=int, default = [3, 9, 21], help="Window size for shannon index.")
    parser.add_argument("--fill_iterations", type=int, default=1, help="Number of iterations for filling voids in the DEM.")
    parser.add_argument("--fill_method", type=str, default=None, choices=["IDW", "FocalStatistics", None], help="Method to fill voids in the DEM, or skip with None.")
    parser.add_argument("--out_folder", type=str, default=None, help="Output folder path for habitat derivatives.")
    parser.add_argument("--products", type=str, nargs="+", default=["slope", "aspect", "roughness", "tpi", "tri", "hillshade", "shannon_index"],
                        choices=["slope", "aspect", "roughness", "tpi", "tri", "hillshade", "shannon_index", "lbp-3-1", "lbp-15-2", "lbp-21-3", "dem"],
                        metavar="PRODUCTS",
                        help="List of products to generate (e.g., slope, aspect, roughness, hillshade, shannon_index, lbp-3-1, lbp-15-2, lbp-21-3, dem)")
    parser.add_argument("--generate_geomorphons", action="store_true", help="Generate geomorphons from the DEM. This will create landforms using ArcGIS Pro's GeomorphonLandforms tool.")

    # Parse arguments
    args = parser.parse_args()
    input_dem = args.input_dem
    input_bs = args.input_bs if args.input_bs else None  # Optional backscatter input
    # Set default output folder if not provided
    out_folder = args.out_folder
    products = args.products
    divisions = args.divisions   
    shannon_window = args.shannon_window
    fill_method = args.fill_method # IDW, FocalStatistics, None
    fill_iterations = args.fill_iterations
    # Validate input arguments

    print("ORIGINAL PATH:", input_dem)
    print("PRODUCTS:", products)
    print()
    print(GetExtents.return_min_max_tif_df([input_dem]))
    print()
    # Create an instance of the HabitatDerivatives class with the specified parameters
    generateDerivatives = ProcessDem(
                                                input_dem=input_dem, 
                                                input_bs=input_bs,
                                                output_folder=out_folder,
                                                products=products,
                                                shannon_window=shannon_window,
                                                fill_iterations=fill_iterations,
                                                fill_method=fill_method,
                                                divisions=divisions,
                            )
    generateDerivatives.process_dem()

    if args.generate_geomorphons:
        filled_dem = os.path.join(os.path.dirname(input_dem), "filled", Utils.sanitize_path_to_name(input_dem) + "_filled.tif")
        # create original landforms from ArcGIS Pro
        landforms = Landforms(filled_dem)
        landforms.calculate_geomorphon_landforms()

        # calculate the 10class solution
        output_file10c = landforms.classify_bathymorphons(classes="10c")
        print(f"Modified raster data saved to {output_file10c}")

        # calculate the 6class solution
        output_file6c = landforms.classify_bathymorphons(classes="6c")
        print(f"Modified raster data saved to {output_file6c}")

        # calculate the 5class solution
        output_file5c = landforms.classify_bathymorphons(classes="5c")
        print(f"Modified raster data saved to {output_file5c}")

        # calculate the 4class solution
        output_file4c = landforms.classify_bathymorphons(classes="4c")
        print(f"Modified raster data saved to {output_file4c}")

    print("Processing completed successfully.")


if __name__ == "__main__":
    main()
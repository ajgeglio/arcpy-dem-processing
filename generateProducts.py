import os
import sys
import argparse

# Ensure src is on sys.path regardless of working directory
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "src"))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from derivatives import HabitatDerivatives
from extents import GetCoordinates

# Set up argument parser
parser = argparse.ArgumentParser(description="Process DEM to generate habitat derivatives.")
parser.add_argument("--input_dem", type=str, required=True, help="Path to the input DEM file.")
parser.add_argument("--chunk_size", type=int, default = None, help="Tile size to proocess DEM file.")
parser.add_argument("--shannon_window", type=int, default = 21, help="Window size for shannon index.")
parser.add_argument("--out_folder", type=str, default="habitat_derivatives", help="Output folder path for habitat derivatives.")
parser.add_argument("--products", type=str, nargs="+", default=["slope", "aspect", "roughness", "tpi", "tri", "hillshade"],
                    help="List of products to generate (e.g., slope, aspect, roughness, hillshade, shannon_index, lbp-3-1, lbp-15-2, lbp-21-3, dem)")

# Parse arguments
args = parser.parse_args()
input_dem = args.input_dem
out_folder = args.out_folder
products = args.products
chunk_size = args.chunk_size   
shannon_window = args.shannon_window

print("ORIGINAL PATH:", input_dem)
print("PRODUCTS:", products)
print()
print(GetCoordinates().return_min_max_tif_df([input_dem]))
print()
# Create an instance of the HabitatDerivatives class with the specified parameters
habitat_derivatives = HabitatDerivatives(
                                            input_dem=input_dem, 
                                            output_folder=out_folder,
                                            products=products,
                                            shannon_window=args.shannon_window,
                                            fill_iterations=1,
                                            fill_method="IDW",
                                            chunk_size=args.chunk_size,
                        )
habitat_derivatives.process_dem()

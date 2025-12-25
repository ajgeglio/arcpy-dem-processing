import HabitatDerivatives
import os
import argparse

# Set up argument parser
parser = argparse.ArgumentParser(description="Process DEM to generate habitat derivatives.")
parser.add_argument("--input_dem", type=str, required=True, help="Path to the input DEM file.")
parser.add_argument("--chunk_size", type=int, default = None, help="Tile size to proocess DEM file.")
parser.add_argument("--shannon_window", type=int, default = 9, help="Window size for shannon index.")
parser.add_argument("--out_folder", type=str, default="habitat_derivatives", help="Output folder path for habitat derivatives.")
parser.add_argument("--products", type=str, nargs="+", default=["slope", "aspect", "roughness", "tpi", "tri", "hillshade"],
                    help="List of products to generate (e.g., slope, aspect, roughness, hillshade, shannon_index, lbp-3-1, lbp-15-2, lbp-21-3, dem)")

# Parse arguments
args = parser.parse_args()
input_dem = args.input_dem
out_folder = args.out_folder
products = args.products
shannon_window = args.shannon_window

# Ensure output folder exists
if not os.path.exists(out_folder):
    os.makedirs(out_folder)

# Get input DEM file and name
dem_name = os.path.splitext(os.path.basename(input_dem))[0]
dem_name = dem_name.replace(".", "_").replace(" ", "_")
print("Creating habitat derivatives for", dem_name)
out_dem_folder = os.path.join(out_folder, dem_name)
if not os.path.exists(out_dem_folder):
    os.mkdir(out_dem_folder)

# Generate output file paths
product_dict = {key: os.path.join(out_dem_folder, dem_name + "_" + key + ".tif") for key in products}
print("generating", product_dict)
# Process DEM
# Dynamically pass output file paths based on the keys in the output_files dictionary
HabitatDerivatives(chunk_size=args.chunk_size, use_gdal=True, use_rasterio=False).process_dem(
    input_dem,
    shannon_window=shannon_window,
    output_slope=product_dict.get("slope"),
    output_aspect=product_dict.get("aspect"),
    output_roughness=product_dict.get("roughness"),
    output_tpi=product_dict.get("tpi"),
    output_tri=product_dict.get("tri"),
    output_hillshade=product_dict.get("hillshade"),
    output_shannon_index=product_dict.get("shannon_index"),
    output_lbp_3_1=product_dict.get("lbp-3-1"),
    output_lbp_15_2=product_dict.get("lbp-15-2"),
    output_lbp_21_3=product_dict.get("lbp-21-3"),
    output_dem=product_dict.get("dem")
)
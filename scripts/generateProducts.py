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
    parser.add_argument("--input_bs", type=str, required=False, help="Path to the input backscatter file, must contain 'BS' in file name.")
    parser.add_argument("--input_binary_mask", type=str, required=False, help="Path to the input binary mask file used for boundary control.")
    parser.add_argument("--divisions", type=int, default = None, help="Divide the height by this to run tile processing of DEM file, automatic overalap computed.")
    parser.add_argument("--shannon_window", type=int, default = [3, 9, 21], help="Window size for shannon index.")
    parser.add_argument("--fill_method", type=str, default="IDW", choices=["IDW", "FocalStatistics", "NoFill"], help="Method to fill voids in the DEM, or skip with None.")
    parser.add_argument("--fill_iterations", type=int, default=1, help="Number of iterations for filling voids in the DEM.")
    parser.add_argument("--products", type=str, nargs="+", default=["slope", "aspect", "roughness", "tpi", "tri", "hillshade", "shannon_index"],
                        choices=["slope", "aspect", "roughness", "tpi", "tri", "hillshade", "shannon_index", "dem", "lbp-3-1", "lbp-15-2", "lbp-21-3"],
                        metavar="PRODUCTS",
                        help="List of products to generate (e.g., slope, aspect, roughness, hillshade, shannon_index, ...)")
    parser.add_argument("--generate_geomorphons", action="store_true", help="Generate geomorphons from the DEM. This will create landforms using ArcGIS Pro's GeomorphonLandforms tool.")

    # Parse arguments
    args = parser.parse_args()
    # Define markers used in filenames
    BATHY_MARKER = "BY"  # Bathymetry
    BS_MARKER = "BS"   # Backscatter
    INPUT_DEM = args.input_dem
    INPUT_BS = args.input_bs if args.input_bs else None  # Optional backscatter input
    BINARY_MASK = args.input_binary_mask if args.input_binary_mask else None  # Optional binary mask input
    RASTER_DIR = os.path.dirname(os.path.abspath(INPUT_DEM))  # Directory of the input DEM
    PRODUCTS = args.products
    # Basic validation
    if not INPUT_DEM:
        raise FileNotFoundError(f"No DEMs found in {RASTER_DIR}")

    if INPUT_BS:
        print(f"FOUND BACKSCATTER ({INPUT_BS}):")
        # --- 2. Combine for Processing ---
        # Combine lists. DEMs first ensures the highest res DEM is the primary snap raster
        input_rasters_list = [INPUT_DEM, INPUT_BS]
    else:
        print("NO BACKSCATTER FILES FOUND. Proceeding with DEMs only.")
        input_rasters_list = [INPUT_DEM]

    print(f"PRODUCTS: {PRODUCTS}")

    # --- 6. Cleanup ---
    print("-" * 30)
    print("STARTING CLEANUP")

    # Collect all files involved (Inputs, Aligned Outputs, and the Mask)
    all_involved_files = input_rasters_list
    if BINARY_MASK:
        all_involved_files.append(BINARY_MASK)

    # Extract unique directories using a set comprehension
    # This prevents trying to clean the same folder multiple times
    cleanup_dirs = {os.path.dirname(f) for f in all_involved_files if f and os.path.exists(os.path.dirname(f))}

    # Iterate and clean
    for directory in cleanup_dirs:
        # Optional: Print where we are cleaning
        Utils.remove_additional_files(directory=directory)

    print("Cleanup Complete.")

    # Set default output folder if not provided
    products = args.products
    divisions = args.divisions   
    shannon_window = args.shannon_window
    fill_method = args.fill_method # IDW, FocalStatistics, None
    if fill_method == "NoFill":
        fill_method = None
    fill_iterations = args.fill_iterations
    # Validate input arguments

    print("ORIGINAL PATH:", INPUT_DEM)
    print("PRODUCTS:", products)
    print()
    GetExtents.return_min_max_tif_df([INPUT_DEM])
    print(GetExtents.return_min_max_tif_df([INPUT_DEM]))
    print()
    # Create an instance of the HabitatDerivatives class with the specified parameters
    generateDerivatives = ProcessDem(
                            input_dem=INPUT_DEM, 
                            input_bs=INPUT_BS,
                            binary_mask=BINARY_MASK,
                            output_folder=RASTER_DIR,
                            products=products,
                            shannon_window=shannon_window,
                            fill_method=fill_method,
                            fill_iterations=fill_iterations,
                            divisions=divisions,
                            )
    generateDerivatives.process_dem()

    if args.generate_geomorphons:
        # create original landforms from ArcGIS Pro
        if fill_method == "NoFill":
            base_dem = args.input_dem
        else:
            base_dem = os.path.join(os.path.dirname(args.input_dem), "filled", f"*{BATHY_MARKER}*.tif")
        print("creating landforms for:", os.path.basename(base_dem))
        landofrms_directory = Landforms(base_dem).generate_landforms()
        # Clean up additional files after processing
        Utils.remove_additional_files(directory=landofrms_directory)

    print("Processing completed successfully.")


if __name__ == "__main__":
    main()
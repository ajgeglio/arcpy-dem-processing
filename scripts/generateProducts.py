import arcpy
import argparse
import sys
import os

# Setup paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "src"))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from utils import Utils
from processDem import ProcessDem
from extents import GetExtents
from rasterMasking import RasterMasking

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Process DEM to generate habitat derivatives.")
    parser.add_argument("--input_dem", type=str, required=True, help="Path to the input DEM file.")
    parser.add_argument("--input_bs", type=str, required=False, help="Path to the input backscatter file, must contain 'BS' in file name.")
    parser.add_argument("--input_binary_mask", type=str, required=False, help="Path to the input binary mask file used for boundary control.")
    parser.add_argument("--divisions", type=int, default=None, help="Divide the height by this to run tile processing of DEM file, automatic overlap computed.")
    parser.add_argument("--shannon_window", type=int, nargs="+", default=[3, 9, 21], help="Window sizes for shannon index (e.g. 3 9 21).")
    parser.add_argument("--fill_method", type=str, default="IDW", choices=["IDW", "FocalStatistics", "NoFill"], help="Method to fill voids in the DEM, or skip filling with NoFill.")
    parser.add_argument("--fill_iterations", type=int, default=1, help="Number of iterations for filling voids in the DEM.")
    parser.add_argument("--save_chunks", action="store_true", help="Save the DEM chunks after mosaic. Best to do this during testing of large products.")
    
    # Updated choices to include landform options
    parser.add_argument("--products", type=str, nargs="+", 
                        default=["slope", "aspect", "roughness", "tpi", "tri", "hillshade", "shannon_index"],
                        choices=[
                            "slope", "aspect", "roughness", "tpi", "tri", "hillshade", "shannon_index", "dem", 
                            "lbp-3-1", "lbp-15-2", "lbp-21-3", 
                            "bathymorphons", 
                            "None"
                        ],
                        metavar="PRODUCTS",
                        help="List of products to generate.")

    # Parse arguments
    args = parser.parse_args()
    
    INPUT_DEM = args.input_dem
    INPUT_BS = args.input_bs if args.input_bs else None
    BINARY_MASK = args.input_binary_mask if args.input_binary_mask else None
    RASTER_DIR = os.path.dirname(os.path.abspath(INPUT_DEM))
    PRODUCTS = args.products

    # Basic validation
    if not INPUT_DEM:
        raise FileNotFoundError(f"No DEMs found in {RASTER_DIR}")

    if INPUT_BS:
        print(f"FOUND BACKSCATTER ({INPUT_BS}):")
        input_rasters_list = [INPUT_DEM, INPUT_BS]
    else:
        print("NO BACKSCATTER FILES FOUND. Proceeding with DEMs only.")
        input_rasters_list = [INPUT_DEM]

    print(f"PRODUCTS: {PRODUCTS}")

    # --- Cleanup Logic ---
    print("-" * 30)
    print("STARTING PRE-CLEANUP")
    
    all_involved_files = input_rasters_list.copy()
    if BINARY_MASK:
        all_involved_files.append(BINARY_MASK)

    cleanup_dirs = {os.path.dirname(f) for f in all_involved_files if f and os.path.exists(os.path.dirname(f))}

    for directory in cleanup_dirs:
        Utils.remove_additional_files(directory=directory)

    print("Cleanup Complete.")

    # Setup Variables
    DIVISIONS = args.divisions   
    SHANNON_WIN = args.shannon_window
    SAVE_CHUNKS = args.save_chunks
    FILL_METHOD = args.fill_method
    if FILL_METHOD == "NoFill":
        FILL_METHOD = None
    FILL_ITERATIONS = args.fill_iterations

    print("ORIGINAL PATH:", INPUT_DEM)
    print()
    
    # Optional Extent Check
    try:
        print(GetExtents.return_min_max_tif_df([INPUT_DEM]))
    except Exception as e:
        print(f"Could not get extents (non-fatal): {e}")
    print()

    # --- Process Execution ---
    if PRODUCTS == ["None"] or len(PRODUCTS) == 0:
        print("No products specified.")
    else:
        generateDerivatives = ProcessDem(
                                input_dem=INPUT_DEM, 
                                input_bs=INPUT_BS,
                                binary_mask=BINARY_MASK,
                                output_folder=RASTER_DIR,
                                products=PRODUCTS,
                                shannon_window=SHANNON_WIN,
                                fill_method=FILL_METHOD,
                                fill_iterations=FILL_ITERATIONS,
                                divisions=DIVISIONS,
                                save_chunks=SAVE_CHUNKS
                                )
        generateDerivatives.process_dem()

    print("Processing completed successfully.")

if __name__ == "__main__":
    main()
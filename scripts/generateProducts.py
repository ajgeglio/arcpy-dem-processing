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
    parser.add_argument("--keep_chunks", action="store_true", help="Save the DEM chunks. Best to do this during testing of large products.")
    parser.add_argument("--bypass_depth", action="store_true", help="Bypass creation of the depth raster, the default behavior is to convert to depth if elevation DEM is given.")
    parser.add_argument("--water_elevation", type=int, default=183.6, help="Water elevation (m) for the depth raster, default is the a high elevation marker for the Great Lakes")
    parser.add_argument("--bypass_mosaic", action="store_true", help="Bypass creation of the depth raster, the default behavior is to convert to depth if elevation DEM is given.")

    # Updated choices to include landform options
    parser.add_argument("--products", type=str, nargs="+", 
                        default=["slope", "aspect", "roughness", "tpi", "tri", "hillshade"],
                        choices=[
                            "slope", "aspect", "roughness", "tpi", "tri", "hillshade", # standard gdal products
                            "shannon_index",  # window radii based on input
                            "lbp-8-1", "lbp-15-2", "lbp-21-4", "lbp", # defaults to n=21 and r=4
                            "bathymorphons", # defaults to create all
                            "dem", # just return dem
                            "None" # don't process dem
                        ],
                        metavar="PRODUCTS",
                        help="List of products to generate.")

    # Parse arguments
    args = parser.parse_args()
    
    # Setup Variables
    INPUT_DEM = args.input_dem
    INPUT_BS = args.input_bs if args.input_bs else None
    BINARY_MASK = args.input_binary_mask if args.input_binary_mask else None
    RASTER_DIR = os.path.dirname(os.path.abspath(INPUT_DEM))
    PRODUCTS = args.products
    BYPASSDEPTH = args.bypass_depth
    BYPASSMOSAIC = args.bypass_mosaic
    DIVISIONS = args.divisions   
    SHANNON_WIN = args.shannon_window
    KEEP_CHUNKS = args.keep_chunks
    WATERELEVATION = args.water_elevation

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
                                keep_chunks=KEEP_CHUNKS,
                                bypass_depth=BYPASSDEPTH,
                                bypass_mosaic=BYPASSMOSAIC,
                                water_elevation=WATERELEVATION
                                )
        generateDerivatives.process_dem()

    print("Processing completed successfully.")

if __name__ == "__main__":
    main()
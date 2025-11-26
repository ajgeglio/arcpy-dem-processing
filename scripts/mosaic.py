import sys
import os
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "src"))
sys.path.append(SRC_DIR)
from utils import Utils, demUtils
from derivatives import HabitatDerivatives
from extents import GetCoordinates


def main(path):
    
    demUtils.merge_dem_arcpy(dem_chunks_folder = path)

if __name__ == "__main__":
    path = r"H:\sfm_muskegon_20230724\75m\20230724_75m_dem_tiles_test"
    main(path)

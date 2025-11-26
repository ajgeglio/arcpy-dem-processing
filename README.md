# USGS Digital Elevation Model (DEM) Processing

This repository contains scripts and tools for processing, analyzing, and visualizing data related to the USGS Northern Lake Michigan Reefs project. The project focuses on the identification, mapping, and assessment of reef habitats in the northern part of Lake Michigan using various geospatial and analytical techniques.

## Table of Contents

- [Project Overview](#project-overview)
- [Repository Structure](#repository-structure)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Data Sources](#data-sources)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

## Project Overview

The goal of this project is to process and analyze geospatial datasets (e.g., bathymetry, substrate, biological surveys) to support the study and management of reef habitats in Northern Lake Michigan. The repository includes scripts for:

- Data cleaning and preprocessing
- Spatial analysis and modeling
- Visualization and mapping
- Exporting results for further use

## Raster Processing & Product Generation Workflow

We generate a suite of spatial derivative products from the original Digital Elevation Model (DEM) and Backscatter intensity data. These products serve as the statistical basis for substrate classification in multibeam survey analysis. To ensure statistical validity and seamless output, the following processing chain is applied:

<p align="left">
  <img src="thumb1.png" width="24%" alt="Hillshade"/>
  <img src="thumb2.png" width="24.6%" alt="Shannon Entropy"/>
</p>
<p align="left">
  <em><strong>Figure 1.</strong> Derived Products of Bay Harbor, Lake Michigan multibeam survey. Hillshade shown on left, and the Shannon entropy texture analysis on the right.</em>
</p>

### 1. Pre-Processing and Alignment
*   **Grid Standardization:** All input rasters (including multi-resolution backscatter) are projected and aligned to the 1-meter DEM, which serves as the "Master Snap Raster." This ensures pixel-for-pixel correspondence across all data layers.
*   **Intersection Masking:** A binary intersection mask is generated to identify the common spatial extent where valid data exists for *all* input layers (DEM + Backscatter). Areas missing data in any single layer are excluded from the final analysis.
*   **Gap Filling (Inpainting):** Small internal gaps and NoData artifacts within the common extent are inpainted using an iterative Inverse Distance Weighted (IDW) or Focal Statistics approach. This ensures continuous surfaces required for neighborhood-based calculations (like texture metrics) without altering the original data significantly.
*   **Data Trimming:** All inputs are trimmed to the clean intersection mask, removing noisy edges and ensuring a unified dataset boundary.

### 2. Derivative Calculation & Tiled Processing
To handle high-resolution datasets efficiently, rasters are segmented into manageable processing tiles (divisions).

*   **Edge Effect Mitigation:** A dynamic overlap buffer calculated as `⌊W_max / 2⌋ + 1` is applied to every tile. This allows neighborhood operations (e.g., Shannon Entropy, Slope) to "see" into the adjacent tile, effectively eliminating seaming artifacts and edge discontinuities in the final output.
*   **Product Generation:** The following derivatives are calculated for every pixel:
    *   **Morphometrics:** Slope, Aspect, Roughness, TPI (Topographic Position Index), TRI (Terrain Ruggedness Index), Hillshade.
    *   **Texture Analysis:** Shannon Entropy (calculated at multiple window sizes, e.g., 3x3, 9x9, 21x21) and Local Binary Patterns (LBP).
    *   **Bathymorphons:** 10-class, 6-class, 5-class, 4-class, and raw bathymorphons are generated.

### 3. Post-Processing and Quality Assurance
*   **Seamless Mosaicking:** Processed tiles—minus their overlap buffers—are mathematically merged back into a single continuous raster.
*   **Artifact Removal:** A final trimming pass is applied using the original binary mask. This removes "halo" artifacts that often appear on the outer edges of texture-derived products (like Shannon Index) due to windowing functions.
*   **Metadata Validation:** The final products undergo strict validation to ensure they retain the exact spatial reference, cell size, and extent of the original source DEM.

## Repository Structure

```
USGS_Northern_LM_Reefs_processing/
│
├── data/                # Raw and processed data files (not included in repo)
├── scripts/             # Python and/or R scripts for data processing
├── notebooks/           # Jupyter or RMarkdown notebooks for analysis and visualization
├── outputs/             # Generated outputs (figures, tables, shapefiles, etc.)
├── requirements.txt     # Python dependencies
├── environment.yml      # Conda environment (if applicable)
├── README.md            # This file
└── LICENSE              # License information
```

## Requirements

- Advanced ArcGIS Pro 3.4 Licence
- Python 3.11+ (or R, if using R scripts)
- Recommended: [Anaconda](https://www.anaconda.com/products/distribution) for environment management

### Python Dependencies

Install dependencies with:

if using conda:

```bash
conda env create -f environment.yml
conda activate usgs-reefs
```

Typical dependencies include:
- arcpy3.4
- numpy
- pandas
- geopandas
- rasterio
- matplotlib
- seaborn
- scikit-learn
- jupyter

(See `requirements.txt` or `environment.yml` for the full list.)

## Installation

1. Clone the repository:

   ```bash
   git clone https://code.usgs.gov/great-lakes-science-center/computer-vision/usgs_northern_lm_reefs_processing.git
   cd USGS_Northern_LM_Reefs_processing
   ```

2. Install dependencies as described above.

3. Place raw data files in the `data/` directory as described in [Data Sources](#data-sources).

## Usage

1. Review and configure any paths or parameters in the scripts under `scripts/` or notebooks under `notebooks/`.
2. Run the desired script or notebook. For example:

   ```bash
   python scripts/generateProducts.py --args
   ```

   or open a notebook:

   ```bash
   jupyter notebook notebooks/USGS_Northern_LM_Reefs_processing.ipynb
   ```

3. Outputs will be saved to the `geomorphons/` and `habitat_derivatives` directories.

## Data Sources

Data required for processing is not included in this repository due to size and licensing restrictions. Typical data sources include:

- USGS bathymetric survey digital-elevation-model files in '.tif' format

Please contact the project lead or refer to the project documentation for instructions on obtaining and organizing these datasets.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request. For major changes, please discuss them first.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Contact

For questions or collaboration, contact:

- Anthony Geglio (ageglio@mtu.edu)
- Peter Esselman Advanced Techonology Lab
- USGS Great Lakes Science Center
- Ann Arbor, MI
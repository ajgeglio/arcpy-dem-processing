# Arcpy3.4 Raster DEM Processing

This repository contains scripts and tools for processing, analyzing, and visualizing data related to digital elevation models. The project focuses on the identification, mapping, and assessment of reef habitats in the northern part of Lake Michigan using various geospatial and analytical techniques.

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

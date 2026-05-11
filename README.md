# Overview

To address measuring the mortality rate of grape vines, vinDAR will produce a reproducible, scalable workflow for detecting missing grapevines in French vineyards using HD LiDAR point clouds, parcel boundaries, and PDO/AOC spacing regulations. The pipeline transforms raw LiDAR tiles into parcel‑level indicators of vineyard structure, spacing, and vine mortality.

The workflow includes:
- parcel–AOC integration
- LiDAR tile discovery
- in‑memory COPC downloading
- PDAL‑based tile merging
- DBSCAN vine‑centroid detection
- 2D FFT row‑orientation estimation
- spacing‑grid evaluation
- missing‑vine scoring
- parallel parcel processing

The project was developed as part of the Master of Environmental Data Science (MEDS) program at UCSB.

# Analysis Description

This project develops a modeling and data‑integration framework to estimate long‑term vine mortality. The core contribution of this project is to provide a reproducible, LIDAR-based workflow for detecting missing grapevines at the parcel level. This workflow transforms raw HD LIDAR point clouds into meaningful indicators of vineyard structure, geometry and vine mortality. The analysis combines spatial preprocessing, height filtering, machine learning clustering, frequency detection and gap analyusis to produce parcel-level estimates of missing vines.

The second possible analysis may allow for the evaluations of vineyard vulnerability under climate stress, and enable the linkage of these risks to PDO/AOC regulations, grape varieties, and wine‑region characteristics. This workflow combines LiDAR data, ecological modeling models, sensitivity analysis and regulatory data extraction.

Our team is in the process in creating a reproducible workflow that detects missing vines using LiDAR point clouds, parcel boundaries, and regulatory spacing rules. The workflow integrates:

# Pipeline Summary
The vinDAR workflow consists of the following steps:

1. Load AOC boundaries, spacing rules, and parcels
2. Assign each parcel to its highest‑designation AOC
3. Load LiDAR tile metadata and identify tiles intersecting each parcel
4. Download LiDAR tiles in memory (rate‑limited)
5. Merge tiles using PDAL
6. Clip LiDAR to parcel geometry and filter vegetation classes
7. Detect vine centroids using DBSCAN
8. Estimate row orientation using a 2D FFT
9. Generate candidate spacing combinations
10. Build expected vine grids and compute RMSE
11. Select best spacing and compute missing‑vine percentage
12. Process parcels in parallel and write results to CSV

The result is a scalable methodology for quantifying vine mortality across French wine regions, beginning with the Rhône Valley.

This is a capstone project for the [Master of Environmental Data Science](https://bren.ucsb.edu/masters-programs/master-environmental-data-science) at the Bren School of Environmental Science and Management, University of California, Santa Barbara.

# Data Sources

| **Data** | **Source** | **Use** |
|------------------------|------------------------|------------------------|
| LiDAR HD Point Clouds (.laz) | IGN (Institut National de l’Information Géographique et Forestière) | Vine canopy detection, clustering, gap analysis |
| Parcel Boundaries (.shp/.gpkg) | [Registre Parcellaire Graphique](data.gouv.fr) | Parcel clipping, spatial alignment |
| Wine Region / PDO Identifiers (.csv) | Public agriculture datasets | Linking parcels to regulatory regions |
| eAmbrosia Regulatory Data | EU Commission | Vine spacing rules, irrigation allowances |
| CVI Vine‑Age Data | INRAE (restricted) | Validation of missing‑vine detection |
| Environmental Data (temperature, drought, slope) | Various public datasets | Sensitivity analysis inputs |


# Repositories in This Organization

```         
vinDAR/
├── README.md
├── LICENSE
├── environment.yml
├── pyproject.toml
├── main.py
├── notebooks/
│   ├── exploratory_analysis.ipynb
│   └── prototype_clustering.ipynb
├── src/
│   └── vineyard_analysis/
│       ├── config.py
│       ├── io/
│       │   ├── aoc.py
│       │   ├── parcels.py
│       │   └── zones.py
│       ├── lidar/
│       │   ├── lidar_file_urls.py
│       │   └── download_all.py
│       └── analysis/
│           ├── clustering.py
│           ├── row_analysis.py
│           └── process_parcel.py
└── test/
    └── test_parcel_processing.py

```

Key Directories
- src/vineyard_analysis/ — core Python package
- io/ — loading parcels, AOC boundaries, spacing rules, and tile metadata
- lidar/ — LiDAR tile discovery, downloading, and PDAL merging
- analysis/ — clustering, FFT row detection, spacing evaluation, parcel processing
- notebooks/ — exploratory analysis and prototyping
- main.py — orchestrates the full pipeline

# Running the Pipeline

1) Install dependencies

conda env create -f environment.yml
conda activate vindar

2) Configure analysis

Edit src/vineyard_analysis/config.py:

- WINE_REGION
- ADMINISTRATIVE_REGION
- PDO_NAME
- DEPT_PREFIX

3) Run the workflow

python main.py

4) Output

- A CSV file containing:
- parcel ID
- best row spacing
- best plant spacing
- RMSE
- expected vine count
- detected vine count
- missing‑vine percentage


# Authors and Team

Team Members: - [Joshua Ferrer‑Lozano](https://github.com/Awoo56709) - [Stephan Kadonoff](https://github.com/SRKads1998) - [Jay Kim](https://github.com/jwonyk) - [William Mullins](https://github.com/willrmull)

# Client:

Jean Sauveur-Ay

Faculty Advisor: Andrew Plantinga

# Acknowledgements

We thank the Bren School faculty and staff for guidance, and acknowledge the organizations that provided regulatory, spatial, and grape‑quality data used in this project. Additional thanks to collaborators who supported data acquisition, domain interpretation, and methodological review.

# Key References & Data Sources EU & French PDO/AOC regulatory documents

-   Grape‑quality datasets (variety‑level and region‑level attributes)

-   Climate and environmental covariates used in stress modeling

-   Sensitivity analysis tools and ecological modeling literature

# License

This project is released under the MIT License unless otherwise noted in individual repositories.

# Disclaimer

This organization is structured to support transparency, reproducibility, and long‑term usability for both the client and the broader community.

# Analysis Description

This project develops a modeling and data‑integration framework to estimate long‑term vine mortality. The core contribution of this project is to provide a reproducible, LIDAR-based workflow for detecting missing grapevines at the parcel level. This workflow transforms raw HD LIDAR point clouds into meaningful indicators of vineyard structure, geometry and vine mortality. The analysis combines spatial preprocessing, height filtering, machine learning clustering, frequency detection and gap analyusis to produce parcel-level estimates of missing vines.

The second possible analysis may allow for the evaluations of vineyard vulnerability under climate stress, and enable the linkage of these risks to PDO/AOC regulations, grape varieties, and wine‑region characteristics. This workflow combines LiDAR data, ecological modeling models, sensitivity analysis and regulatory data extraction.

Our team is in the process in creating a reproducible workflow that detects missing vines using LiDAR point clouds, parcel boundaries, and regulatory spacing rules. The workflow integrates:

-   vegetation height filtering

-   DBSCAN clustering for vine detection

-   Fourier‑based row orientation estimation

-   gap detection and mortality estimation

-   SQL‑based data integration

-   sensitivity analysis of environmental and geometric factors

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
├── .git/
├── .gitignore
├── .Rproj.user/
├── .Rhistory
├── LICENSE
├── README.md
├── vinDAR.Rproj
├── nano.30808.save
└── src/
    ├── data_import.py
    ├── exploratory_analysis.ipynb
    ├── filename.csv
    └── lidar_improved.png
```

src: 
- data_import.py - import file 
- exploratory_analysis.ipynb - main analysis 
- filename.csv - data 
- lidar_improved.png - image render

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

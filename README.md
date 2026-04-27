# vinDAR

README


vinDAR: Assessing the Health of Vinyards Using HD LIDAR Data

Project Description
This project develops a modeling and data‑integration framework to estimate long‑term vine mortality. The core contribution of this project is to provide a reproducible, LIDAR-based workflow for detecting missing grapvines at the parcel level. This workflow transforms raw HD LIDAR point clouds into meaningful indicators of vineyard structure, geometry and vine mortality. The analysis combines spatial preprocessing, height filtering, machine learning clustering, frequency detection and gap analyusis to produce parcel-level estimates of missing vines.

The second possible analysis may allow for the evaluations of vineyard vulnerability under climate stress, and enebale the linkage of these risks to PDO/AOC regulations, grape varieties, and wine‑region characteristics. This workflow combines LiDAR data, ecological modeling models, sensitivity analysis and regulatory data extraction.

This is a capstone project for the
Master of Environmental Data Science at the
Bren School of Environmental Science and Management,
University of California, Santa Barbara.

Abstract
Climate change is reshaping vineyard viability across Europe, yet regional regulations, spacing rules, and varietal characteristics create uneven vulnerability across PDOs and wine types. vinDAR integrates climate‑stress mortality modeling, LiDAR‑based detection workflows, and PDO regulatory data to estimate long‑term vine loss and identify which regions, varieties, and wine styles are most susceptible to mortality.

Repositories in This Organization

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

src - 
  - data_import.py - import file
  - exploratory_analysis.ipynb - main analysis
  - filename.csv - data
  - lidar_improved.png - image render

Each repository includes its own README describing purpose, structure, and usage.

Team

Team Members:

Joshua Ferrer‑Lozano
Stephan Kadonoff
Jay Kim
William Mullins

Client:  
Jean Sauveur-Ay

Faculty Advisor:  
Andrew Plantinga

Acknowledgements
We thank the Bren School faculty and staff for guidance, and acknowledge the organizations that provided regulatory, spatial, and grape‑quality data used in this project. Additional thanks to collaborators who supported data acquisition, domain interpretation, and methodological review.

Key References & Data Sources
EU & French PDO/AOC regulatory documents

Grape‑quality datasets (variety‑level and region‑level attributes)

Climate and environmental covariates used in stress modeling

Sensitivity analysis tools and ecological modeling literature

License
This project is released under the MIT License unless otherwise noted in individual repositories.

How to Use This Organization
Each repository includes:

A clear README describing purpose and entry points

A walkthrough of directory structure

Standardized file naming conventions

Reproducible code and documented workflows

Appropriate licensing and contributor information

This organization is structured to support transparency, reproducibility, and long‑term usability for both the client and the broader community.
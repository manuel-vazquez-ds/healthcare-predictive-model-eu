# Healthcare Mortality Prediction in Europe (NUTS)  
### Machine Learning-based Predictive Modeling using Demographic and Spatial Data  

[![License: CC BY 4.0](https://licensebuttons.net/l/by/4.0/88x31.png)](https://creativecommons.org/licenses/by/4.0/)  
[![Python](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-Machine%20Learning-orange.svg)](https://scikit-learn.org/)
[![LightGBM](https://img.shields.io/badge/LightGBM-Boosting-green.svg)](https://lightgbm.readthedocs.io/)
[![XGBoost](https://img.shields.io/badge/XGBoost-Boosting-red.svg)](https://xgboost.readthedocs.io/)
[![GeoPandas](https://img.shields.io/badge/GeoPandas-Geospatial-yellow.svg)](https://geopandas.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange.svg)](https://jupyter.org/)

Predictive model for estimating mortality by ICD-10 causes across European regions using machine learning and demographic data.

## Translations

- [Leer en Castellano (Spanish)](README_es.md)
- [Llegir en Català (Catalan)](README_ca.md)

## Overview  

This repository contains an adaptation of a bachelor's thesis project into a machine learning system to predict mortality by ICD-10 causes across European regions (NUTS classification), using demographic and geographical data.  

It is an adaptation of a real-world predictive system originally developed for healthcare resource planning within the Catalan public health system.

The objective is to demonstrate that, in low-observation scenarios (limited time series), combining richer feature engineering with boosting models significantly outperforms traditional time-series approaches.

## Key Features

- High predictive performance (R² > 0.9 across all groups)
- Model selection and hyperparameter optimization (LightGBM / XGBoost / Linear Regression / Random Forest / Gradient Boosting)
- Automated ETL pipeline using Eurostat open data
- Predictive modeling of mortality using ICD-10 groups
- Application across European NUTS2 regions
- Dual execution: scripts and interactive GUI (ipywidgets)

## Results

The model was evaluated across multiple ICD-10 cause groups, achieving strong predictive performance:

- Consistent **R² values above 0.9**
- Low prediction errors across all evaluated regions
- Robust generalization despite limited temporal data

This validates the effectiveness of feature-rich boosting models over traditional time-series approaches in constrained datasets.

## Sample Results

The evaluation was executed over **188 NUTS2 (basic) regions**.  
The groups of causes of death evaluated are:
- **C**: Malignant neoplasms (C00-C97)
- **G-H**: Diseases of the nervous system and the sense organs (G00-H95)
- **I**: Diseases of the circulatory system (I00-I99)
- **J**: Diseases of the respiratory system (J00-J99)
- **K**: Diseases of the digestive system (K00-K93)
  
Below is a summary of the predictions and errors obtained:

| ICD-10 (CIM-10) | Deaths | Prediction | Error | MAE | RMSE | R<sup>2</sup> | SMAPE |
|---|---|---|---|---|---|---|---|
| **C** | 693.551 | 715.918,03 | 22.367,03 | 218,03 | 345,90 | 0,99 | 7,15 |
| **G-H** | 140.655 | 140.792,07 | 137,07 | 50,69 | 75,80 | 0,99 | 8,93 |
| **I** | 1.044.016 | 1.086.477,05 | 42.461,05 | 381,87 | 684,69 | 0,98 | 8,23 |
| **J** | 267.096 | 264.483,66 | -2.612,34 | 107,16 | 184,52 | 0,99 | 9,65 |
| **K** | 117.581 | 117.197,80 | -383,20 | 37,48 | 65,26 | 0,99 | 7,58 |

*(Data extraction and model execution date: 2026-04-15)*

## Real-World Context

This project is based on a bachelor's thesis focused on predicting healthcare resource utilization in the Catalan public health system.

The original model was designed to support decision-making in healthcare planning by forecasting hospital activity.  
This adaptation extends the methodology to a European-scale mortality prediction problem using open data.

## Tech Stack

> Focused on predictive modeling in constrained datasets using feature engineering and boosting techniques.

- **Programming Language**: Python (>= 3.12)
- **Data Processing**: Pandas, NumPy
- **Machine Learning**: Scikit-learn, LightGBM, XGBoost
- **Geospatial Analysis**: GeoPandas, Shapely
- **Data Visualization**: Matplotlib
- **Model Persistence**: Joblib
- **Data Retrieval**: Pooch (Eurostat integration)
- **File Handling**: OpenPyXL, XlsxWriter
- **Interactive Interface**: Jupyter Notebook, ipywidgets, ipyfilechooser
- **Environment Management**: uv / pip (via pyproject.toml)
- **Modeling Approach**: Feature engineering + boosting-based models

## Datasets

The project uses the following datasets automatically fetched from Eurostat during the ETL process using `pooch`:
- [Causes of death - deaths by NUTS 2 region of residence and occurrence, 3 year average](https://doi.org/10.2908/HLTH_CD_YRO)
- [Population on 1 January by age, sex and NUTS 2 region](https://doi.org/10.2908/DEMO_R_PJANGROUP)
- [Territorial units for statistics (NUTS)](https://ec.europa.eu/eurostat/web/gisco/geodata/statistical-units/territorial-units-statistics) (Year NUTS 2024 - in geometry polygons RG - scale 20M - reference system EPSG:4326 - GeoPackage format).

## Repository Structure

The workflow is divided into logical steps with their respective representations in Jupyter Notebooks and pure `.py` scripts:

* `mvm_eines.py`: Helper functions module containing reusable code across all notebooks/scripts.
* `etl.ipynb` / `etl.py`: Extracts, transforms, and loads the data from Eurostat.
* `modelitza.ipynb` / `modelitza.py`: The core modeling notebook. It cross-validates, identifies the best boosting model, and optimizes hyperparameters for each regional level and cause.
* `predict.ipynb` / `predict.py`: Generates the final predictions and inference over new periods using the previously trained models.
* `main_gui.ipynb`: A visual, `ipywidgets`-based interface that allows you to easily parameterize and run the entire pipeline interactively without dealing directly with the code configuration.
* `example_model_results.xlsx`: Report output example generated by the process natively.
* `01_VAZQUEZ_MARTI_MANUEL_MEMÒRIA.pdf`: The original bachelor's thesis document outlining the methodology in detail.

>(Note: Unlike the original thesis work, this adaptation does not include the Textual CLI UI nor SHAP-based explanatory analysis, focusing solidly on predictive execution adaptability.)

## Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd <repository-name>
   ```

2. **Install dependencies:**
   The repository uses `uv` for minimal execution overhead. If using `uv`:
   ```bash
   uv sync
   ```
   Or if you prefer standard `pip`:
   ```bash
   pip install -r requirements.txt
   ```

## Execution

You can run the project in two main ways:

### Interactive Execution (Recommended for visual exploration)
Open `main_gui.ipynb` in your Jupyter environment. Using the interface, select the scripts to execute (`etl`, `modelitza`, `predict`), configure language and region parameters, and check the non-blocking execution logs directly within the notebook interface.

### Script Execution
Execute the extracted python scripts in sequence. Make sure to define the required variables (like paths or targets inside the code or via modifications if needed).
```bash
python etl.py
python modelitza.py
python predict.py
```

### Notebook / Script Parameters and Outputs

#### `etl.ipynb` / `etl.py`
- **Parameters**: None required.
- **Generated files**:
  - `cd_df_sense_pivotar.xlsx`: Contains the ICD-10 (CIM-10) causes of death for the 5 selected groups from each NUTS territory from 2011 to 2023.
  - `cd_df_brut.xlsx`: Pivoted data of causes of death restricted to NUTS level.
  - `cd_df_net.xlsx`: Same as above but with cleaned data (without nulls and with corrected values).
  - `mv_nuts2_xy.xlsx`: Dataset of NUTS2 territories with their X and Y centroids and the ancestor NUTS to which each hierarchically belongs. Generated for easy use in visualizations and additional work in external programs.
  - `mv_nuts2_xy.csv`: Same as above but in CSV format, for use with scripts.
  - `dades_to_train_test.xlsx`: Dataset with the data prepared to be used in training and testing the model. Generated for easy use in visualizations and additional work.
  - `dades_to_train_test.csv`: Same as above but in CSV format, for use with scripts.
  - `pop_predictor.xlsx`: Dataset prepared to obtain prediction results from recent years, ignoring causes of death. Generated for easy use in visualizations and external programs.
  - `pop_predictor.csv`: Same as above but in CSV format, for script use.

#### `modelitza.ipynb` / `modelitza.py`
- **Parameters**: Can be executed without parameters or by adding:
  - `-b`, `--basic`: Trains models with default, non-optimized hyperparameters. Drastically reduces execution time.
- **Generated files**:
  - `analisi_models.xlsx`: Contains the weighted score of models trained with different hyperparameters, with their evaluation metrics, ordered from highest to lowest score. Determines which model and hyperparameters to use to analyze each cause of death.
  - `selected_models.pkl`: Contains the trained models with the selected hyperparameters for each cause of death, ready to be used in the prediction script. It is a compressed binary file generated with `joblib`.
  - `dadesModel.pkl`: Contains the data used to train and test the models. It is a compressed binary file generated with `joblib`.
  - `informe_analisi_modelitzacio_YYYYMMDD.xlsx`: Report with analysis results applied to training and test data. Although training is done at a global level, SMAPE metrics are shown for NUTS0, NUTS1, and NUTS2 levels. (YYYYMMDD is the date the report was generated).

#### `predict.ipynb` / `predict.py`
- **Parameters**: Can be executed without parameters or by adding:
  - `-r`, `--rolling`: To train the model with the entire dataset (training and testing), which can improve accuracy and does not penalize overfitting, as the data to predict are completely independent. *This parameter is mutually exclusive with `-t, --train`.*
  - `-t`, `--train`: To retrain the model with the same hyperparameters but with new data. Useful if new rows were added to the original dataset or another dataset is used. *This parameter is mutually exclusive with `-r, --rolling`.*
  - `-p`, `--predir PREDIR`: Where *PREDIR* is the dataset from which to predict how many causes of death of each type will be generated; defaults to `pop_predictor.csv`.
- **Generated files**:
  - `prediccio_yyyy_YYYY.xlsx`: Dataset with the predictions of causes of death for each evaluated NUTS2 territory. (yyyy_YYYY is the range of years for which predictions were made).
  - `prediccio_yyyy_YYYY.csv`: Same as above but in CSV format, for script use.

## Suggestions for Improvement and Use

- You can change or add causes of death groups; the functionality remains the same.
- Hyperparameter sets can be added and adjusted to desired values for greater accuracy. Similarly, other machine learning models can be added to compare with current ones.
- As demonstrated, it is also possible to apply the model to similar datasets; it just needs adaptation.
- At a deeper level, the model can be improved by adding more predictor variables that might influence the target variables.

## Author

**Manuel Vázquez Martí**  
Applied Data Science Graduate (UOC)  
Final grade: 9.02 / 10

- Background in healthcare systems (CatSalut)
- Focused on predictive modeling and data-driven decision-making

## Acknowledgements
- To Alba Benaque Vidal, Manager of Organizations at Servei Català de la Salut, promoter of the project.
- To Sílvia Prior Cayuela, Technical Referent and Tutor from Servei Català de la Salut for her help and guidance on this project.
- To Alfons Marquès i Cancio, Director of the thesis, for his patience and suggestions.

## License
This project and its theoretical foundation, based on the original Bachelor's Thesis, are distributed under the [Creative Commons Attribution 4.0 (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/) license.

_April 2026_


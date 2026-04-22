# Predicción de Mortalidad Sanitaria en Europa (NUTS)  
### Modelado Predictivo basado en Machine Learning utilizando Datos Demográficos y Espaciales  

Modelo predictivo para estimar la mortalidad por causas CIM-10 (ICD-10) en las regiones europeas utilizando machine learning y datos demográficos.  

## Traducciones

- [Read in English](README.md)
- [Llegir en Català](README_ca.md)

## Visión General  

Este repositorio contiene la adaptación de un trabajo de final de grado en un sistema de machine learning para predecir la mortalidad por causas CIM-10 (ICD-10) en las regiones europeas (clasificación NUTS), utilizando datos demográficos y geográficos.

Es una adaptación de un sistema predictivo del mundo real desarrollado originalmente para la planificación de recursos sanitarios dentro del sistema público de salud catalán.

El objetivo es demostrar que, en escenarios de baja observación (series temporales limitadas), combinar una ingeniería de características más rica con modelos de boosting supera significativamente a los enfoques tradicionales de series temporales.

## Características Principales

- Alto rendimiento predictivo (R² > 0.9 en todos los grupos)
- Selección de modelos y optimización de hiperparámetros (LightGBM / XGBoost / Regresión Lineal / Random Forest / Gradient Boosting)
- Pipeline ETL automatizado utilizando datos abiertos de Eurostat
- Modelado predictivo de la mortalidad utilizando grupos CIM-10
- Aplicación en las regiones europeas NUTS2
- Ejecución dual: scripts e interfaz gráfica interactiva (GUI) (ipywidgets)

## Resultados

El modelo fue evaluado en múltiples grupos de causas CIM-10, logrando un sólido rendimiento predictivo:

- Valores de **R² consistentes por encima de 0.9**
- Bajos errores de predicción en todas las regiones evaluadas
- Generalización robusta a pesar de los datos temporales limitados

Esto valida la eficacia de los modelos de boosting ricos en características sobre los enfoques tradicionales de series temporales en conjuntos de datos restringidos.

## Muestra de Resultados

La evaluación fue ejecutada sobre **188 regiones NUTS2 (básicas)**.  
Los grupos de causas de muerte evaluados son:
- **C**: Neoplasias malignas (C00-C97)
- **G-H**: Enfermedades del sistema nervioso y de los órganos de los sentidos (G00-H95)
- **I**: Enfermedades del sistema circulatorio (I00-I99)
- **J**: Enfermedades del sistema respiratorio (J00-J99)
- **K**: Enfermedades del sistema digestivo (K00-K93)

A continuación, se presenta un resumen de las predicciones y errores obtenidos:

| ICD-10 (CIM-10) | Defunciones | Predicción | Error | MAE | RMSE | R<sup>2</sup> | SMAPE |
|---|---|---|---|---|---|---|---|
| **C** | 693.551 | 715.918,03 | 22.367,03 | 218,03 | 345,90 | 0,99 | 7,15 |
| **G-H** | 140.655 | 140.792,07 | 137,07 | 50,69 | 75,80 | 0,99 | 8,93 |
| **I** | 1.044.016 | 1.086.477,05 | 42.461,05 | 381,87 | 684,69 | 0,98 | 8,23 |
| **J** | 267.096 | 264.483,66 | -2.612,34 | 107,16 | 184,52 | 0,99 | 9,65 |
| **K** | 117.581 | 117.197,80 | -383,20 | 37,48 | 65,26 | 0,99 | 7,58 |

*(Fecha de extracción de datos y ejecución del modelo: 15/04/2026)*

## Contexto del Mundo Real

Este proyecto se basa en un trabajo de final de grado centrado en la predicción del uso de recursos sanitarios en el sistema público de salud de Cataluña.

El modelo original fue diseñado para apoyar la toma de decisiones en la planificación sanitaria previendo la actividad hospitalaria.  
Esta adaptación amplía la metodología a un problema de predicción de mortalidad a escala europea utilizando datos abiertos.

## Stack Tecnológico

> Centrado en el modelado predictivo en conjuntos de datos restringidos utilizando ingeniería de características y técnicas de boosting.

- **Lenguaje de Programación**: Python (>= 3.12)
- **Procesamiento de Datos**: Pandas, NumPy
- **Machine Learning**: Scikit-learn, LightGBM, XGBoost
- **Análisis Geoespacial**: GeoPandas, Shapely
- **Visualización de Datos**: Matplotlib
- **Persistencia de Modelos**: Joblib
- **Recuperación de Datos**: Pooch (integración con Eurostat)
- **Manejo de Archivos**: OpenPyXL, XlsxWriter
- **Interfaz Interactiva**: Jupyter Notebook, ipywidgets, ipyfilechooser
- **Gestión del Entorno**: uv / pip (vía pyproject.toml)
- **Enfoque de Modelado**: Ingeniería de características + modelos basados en boosting

## Conjuntos de Datos

El proyecto utiliza los siguientes conjuntos de datos obtenidos automáticamente de Eurostat durante el proceso de ETL utilizando `pooch`:
- [Causes of death - deaths by NUTS 2 region of residence and occurrence, 3 year average](https://doi.org/10.2908/HLTH_CD_YRO)
- [Population on 1 January by age, sex and NUTS 2 region](https://doi.org/10.2908/DEMO_R_PJANGROUP)
- [Territorial units for statistics (NUTS)](https://ec.europa.eu/eurostat/web/gisco/geodata/statistical-units/territorial-units-statistics) (Año NUTS 2024 - en geometría polígonos RG - escala 20M - en el sistema de referencia EPSG:4326 - formato GeoPackage).

## Estructura del Repositorio

El flujo de trabajo se divide en pasos lógicos con sus respectivas representaciones en Jupyter Notebooks y scripts puros `.py`:

* `mvm_eines.py`: Módulo de funciones auxiliares que contiene código reutilizable en todos los notebooks/scripts.
* `etl.ipynb` / `etl.py`: Extrae, transforma y carga los datos de Eurostat.
* `modelitza.ipynb` / `modelitza.py`: El notebook principal de modelización. Realiza la validación cruzada, identifica el mejor modelo de boosting y optimiza los hiperparámetros para cada nivel regional y causa.
* `predict.ipynb` / `predict.py`: Genera las predicciones finales e inferencia sobre nuevos periodos usando los modelos previamente entrenados.
* `main_gui.ipynb`: Una interfaz visual basada en `ipywidgets` que permite parametrizar y ejecutar fácilmente todo el pipeline de forma interactiva sin interactuar directamente con la configuración del código.
* `example_model_results.xlsx`: Ejemplo de salida del informe generado por el proceso de forma nativa.
* `01_VAZQUEZ_MARTI_MANUEL_MEMÒRIA.pdf`: El documento original del trabajo de final de grado que detalla la metodología.

>(Nota: A diferencia del trabajo original de grado, esta adaptación no incluye la interfaz de línea de comandos de Textual ni el análisis explicativo basado en SHAP, centrándose sólidamente en la adaptabilidad de la ejecución predictiva.)

## Instalación y Configuración

1. **Clonar el repositorio:**
   ```bash
   git clone <repository-url>
   cd <repository-name>
   ```

2. **Instalar dependencias:**
   El repositorio utiliza `uv` para reducir la sobrecarga de ejecución. Si usas `uv`:
   ```bash
   uv sync
   ```
   O si prefieres el `pip` estándar:
   ```bash
   pip install -r requirements.txt
   ```

## Ejecución

Puedes ejecutar el proyecto de dos formas principales:

### Ejecución Interactiva (Recomendado para exploración visual)
Abre `main_gui.ipynb` en tu entorno Jupyter. Usando la interfaz, selecciona los scripts a ejecutar (`etl`, `modelitza`, `predict`), configura los parámetros de idioma y región, y revisa los registros de ejecución en la consola no bloqueante directamente dentro de la interfaz del notebook.

### Ejecución de Scripts
Ejecuta los scripts de python extraídos en secuencia. Asegúrate de definir las variables requeridas (como rutas o variables objetivo dentro del código si fuera necesario).
```bash
python etl.py
python modelitza.py
python predict.py
```

### Parámetros de Notebooks / Scripts y Archivos Generados

#### `etl.ipynb` / `etl.py`
- **Parámetros**: No necesita parámetros.
- **Archivos generados**:
  - `cd_df_sense_pivotar.xlsx`: Contiene las causas de muerte ICD-10 (CIM-10) de los 5 grupos elegidos de cada territorio NUTS desde 2011 a 2023.
  - `cd_df_brut.xlsx`: Datos ya pivotados de las causas de muerte y sólo a nivel NUTS.
  - `cd_df_net.xlsx`: El mismo que el anterior pero con los datos ya depurados (sin nulos y con los valores corregidos).
  - `mv_nuts2_xy.xlsx`: Dataset de territorios NUTS2 con sus centroides X e Y y los NUTS ancestros a los que pertenece cada uno jerárquicamente. Se genera para uso fácil en visualizaciones y trabajos adicionales en programas externos.
  - `mv_nuts2_xy.csv`: El mismo que el anterior pero en formato CSV, para uso con los scripts.
  - `dades_to_train_test.xlsx`: Dataset con los datos ya preparados para ser usados en el entrenamiento y testeo del modelo. Se genera para uso fácil en visualizaciones y trabajos adicionales en programas externos.
  - `dades_to_train_test.csv`: El mismo que el anterior pero en formato CSV, para uso con los scripts.
  - `pop_predictor.xlsx`: Dataset preparado para obtener resultados de predicción de los últimos años, desconociendo las causas de muerte. Se genera para uso fácil en visualizaciones y trabajos adicionales en programas externos.
  - `pop_predictor.csv`: El mismo que el anterior pero en formato CSV, para uso con los scripts.

#### `modelitza.ipynb` / `modelitza.py`
- **Parámetros**: Se puede ejecutar sin parámetros o añadiendo:
  - `-b`, `--basic`: Con el que entrena los modelos con hiperparámetros por defecto, no optimizados. Reduce drásticamente el tiempo de ejecución.
- **Archivos generados**:  
  - `analisi_models.xlsx`: Contiene la puntuación ponderada de los modelos entrenados con distintos hiperparámetros, con sus métricas de evaluación, ordenados de mayor a menor puntuación. Determina qué modelo y qué hiperparámetros usar para analizar cada causa de muerte.
  - `selected_models.pkl`: Contiene los modelos entrenados con los hiperparámetros seleccionados para cada causa de muerte, para poderse usar en el script de predicción. Es un archivo binario generado con `joblib` con compresión añadida.
  - `dadesModel.pkl`: Contiene los datos usados para entrenar y testear los modelos. Es un archivo binario generado con `joblib` con compresión añadida.
  - `informe_analisi_modelitzacio_YYYYMMDD.xlsx`: Informe con los resultados del análisis aplicados a los datos de entrenamiento y testeo. Aunque el entrenamiento se haya realizado a nivel global, se muestran métricas SMAPE para niveles NUTS0, NUTS1 y NUTS2. (YYYYMMDD es la fecha en la que se generó el informe).

#### `predict.ipynb` / `predict.py`
- **Parámetros**: Se puede ejecutar sin parámetros o añadiendo:
  - `-r`, `--rolling`: Para entrenar el modelo con todo el conjunto de datos (entrenamiento y testeo), puede mejorar la precisión y no penaliza el sobreajuste, ya que los datos a predecir son totalmente independientes. *Este parámetro es excluyente con `-t, --train`.*
  - `-t`, `--train`: Para entrenar el modelo de nuevo con los mismos hiperparámetros, pero con nuevos datos. Esto puede ser útil si se han añadido nuevas filas al dataset original o se utiliza otro dataset. *Este parámetro es excluyente con `-r, --rolling`.*
  - `-p`, `--predir PREDIR`: Donde *PREDIR* es el conjunto de datos del cual se quieren predecir cuántas causas de muerte de cada tipo se generarán, por defecto es `pop_predictor.csv`.
- **Archivos generados**:
  - `prediccio_yyyy_YYYY.xlsx`: Dataset con las predicciones de las causas de muerte para cada territorio NUTS2 evaluado. (yyyy_YYYY es el rango de años para el cual se han realizado las predicciones).
  - `prediccio_yyyy_YYYY.csv`: El mismo que el anterior pero en formato CSV, para uso con los scripts.

## Sugerencias de Mejora y Uso

- Se puede cambiar o añadir grupos de causa de muerte, la funcionalidad es la misma.
- Se pueden añadir conjuntos de hiperparámetros y ajustarlos a los deseados para una mayor precisión. De la misma manera se pueden añadir otros modelos de machine learning para comparar con los actuales.
- Como está demostrado, también es posible aplicar el modelo a conjuntos de datos similares, sólo hace falta adaptarlo.
- A nivel más profundo, se puede mejorar el modelo añadiendo más variables predictoras, que puedan influir en las variables objetivo.

## Autor

**Manuel Vázquez Martí**  
Graduado en Ciencia de Datos Aplicada (UOC)  
Nota final: 9.02 / 10

- Experiencia en sistemas sanitarios (CatSalut)
- Enfocado en el modelado predictivo y la toma de decisiones basada en datos

## Agradecimientos
- A Alba Benaque Vidal, gerente de Organizaciones del Servei Català de la Salut, promotora del proyecto.
- A Sílvia Prior Cayuela, referente técnico y tutora del Servei Català de la Salut por su ayuda y orientación en el proyecto.
- A Alfons Marquès i Cancio, director del trabajo, por su paciencia y sus sugerencias.

## Licencia
Este proyecto y su fundamentación teórica, basada en el Trabajo de Final de Grado original, se distribuyen bajo la licencia [Creative Commons Attribution 3.0 España (CC BY 3.0 ES)](https://creativecommons.org/licenses/by/3.0/es/).

_Abril 2026_

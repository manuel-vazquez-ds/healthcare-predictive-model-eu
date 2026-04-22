# Predicció de Mortalitat Sanitària a Europa (NUTS)  
### Modelatge Predictiu basat en Machine Learning utilitzant Dades Demogràfiques i Espacials  

Model predictiu per estimar la mortalitat per causes CIM-10 (ICD-10) a les regions europees utilitzant machine learning i dades demogràfiques.  

## Traduccions

- [Read in English](README.md)
- [Leer en Castellano](README_es.md)

## Visió General  

Aquest repositori conté l'adaptació d'un treball de final de grau en un sistema de machine learning per predir la mortalitat per causes CIM-10 (ICD-10) a les regions europees (classificació NUTS), utilitzant dades demogràfiques i geogràfiques.

És una adaptació d'un sistema predictiu del món real desenvolupat originalment per a la planificació de recursos sanitaris dins del sistema públic de salut català.

L'objectiu és demostrar que, en escenaris de baixa observació (sèries temporals limitades), combinar una enginyeria de característiques més rica amb models de boosting supera significativament els enfocaments tradicionals de sèries temporals.

## Característiques Principals

- Alt rendiment predictiu (R² > 0.9 en tots els grups)
- Selecció de models i optimització d'hiperparàmetres (LightGBM / XGBoost / Regressió Lineal / Random Forest / Gradient Boosting)
- Pipeline ETL automatitzat utilitzant dades obertes de l'Eurostat
- Modelatge predictiu de la mortalitat utilitzant grups CIM-10
- Aplicació a les regions europees NUTS2
- Execució dual: scripts i interfície gràfica interactiva (GUI) (ipywidgets)

## Resultats

El model va ser avaluat en múltiples grups de causes CIM-10, aconseguint un rendiment predictiu sòlid:

- Valors de **R² consistents per sobre de 0.9**
- Baixos errors de predicció en totes les regions avaluades
- Generalització robusta malgrat les dades temporals limitades

Això valida l'eficàcia dels models de boosting rics en característiques per sobre dels enfocaments tradicionals de sèries temporals en conjunts de dades restringits.

## Mostra de Resultats

L'avaluació va ser executada sobre **188 regions NUTS2 (bàsiques)**.  
Els grups de causes de mort avaluats són:
- **C**: Neoplàsies malignes (C00-C97)
- **G-H**: Malalties del sistema nerviós i dels òrgans dels sentits (G00-H95)
- **I**: Malalties del sistema circulatori (I00-I99)
- **J**: Malalties del sistema respiratori (J00-J99)
- **K**: Malalties del sistema digestiu (K00-K93)

A continuació, es presenta un resum de les prediccions i errors obtinguts:

| ICD-10 (CIM-10) | Defuncions | Predicció | Error | MAE | RMSE | R<sup>2</sup> | SMAPE |
|---|---|---|---|---|---|---|---|
| **C** | 693.551 | 715.918,03 | 22.367,03 | 218,03 | 345,90 | 0,99 | 7,15 |
| **G-H** | 140.655 | 140.792,07 | 137,07 | 50,69 | 75,80 | 0,99 | 8,93 |
| **I** | 1.044.016 | 1.086.477,05 | 42.461,05 | 381,87 | 684,69 | 0,98 | 8,23 |
| **J** | 267.096 | 264.483,66 | -2.612,34 | 107,16 | 184,52 | 0,99 | 9,65 |
| **K** | 117.581 | 117.197,80 | -383,20 | 37,48 | 65,26 | 0,99 | 7,58 |

*(Data d'extracció de dades i execució del model: 15/04/2026)*

## Context del Món Real

Aquest projecte es basa en un treball de final de grau centrat en la predicció de l'ús de recursos sanitaris en el sistema públic de salut de Catalunya.

El model original va ser dissenyat per donar suport a la presa de decisions en la planificació sanitària preveient l'activitat hospitalària.  
Aquesta adaptació amplia la metodologia a un problema de predicció de mortalitat a escala europea utilitzant dades obertes.

## Stack Tecnològic

> Centrat en el modelatge predictiu en conjunts de dades restringits utilitzant enginyeria de característiques i tècniques de boosting.

- **Llenguatge de Programació**: Python (>= 3.12)
- **Processament de Dades**: Pandas, NumPy
- **Machine Learning**: Scikit-learn, LightGBM, XGBoost
- **Anàlisi Geoespacial**: GeoPandas, Shapely
- **Visualització de Dades**: Matplotlib
- **Persistència de Models**: Joblib
- **Recuperació de Dades**: Pooch (integració amb l'Eurostat)
- **Maneig d'Arxius**: OpenPyXL, XlsxWriter
- **Interfície Interactiva**: Jupyter Notebook, ipywidgets, ipyfilechooser
- **Gestió de l'Entorn**: uv / pip (via pyproject.toml)
- **Enfocament de Modelatge**: Enginyeria de característiques + models basats en boosting

## Conjunts de Dades

El projecte utilitza els següents conjunts de dades obtinguts automàticament de l'Eurostat durant el procés d'ETL utilitzant `pooch`:
- [Causes of death - deaths by NUTS 2 region of residence and occurrence, 3 year average](https://doi.org/10.2908/HLTH_CD_YRO)
- [Population on 1 January by age, sex and NUTS 2 region](https://doi.org/10.2908/DEMO_R_PJANGROUP)
- [Territorial units for statistics (NUTS)](https://ec.europa.eu/eurostat/web/gisco/geodata/statistical-units/territorial-units-statistics) (Any NUTS 2024 - en geometria polígons RG - escala 20M - en el sistema de referència EPSG:4326 - format GeoPackage).

## Estructura del Repositori

El flux de treball es divideix en passos lògics amb les seves respectives representacions en Jupyter Notebooks i scripts purs `.py`:

* `mvm_eines.py`: Mòdul de funcions auxiliars que conté codi reutilitzable en tots els notebooks/scripts.
* `etl.ipynb` / `etl.py`: Extreu, transforma i carrega les dades de l'Eurostat.
* `modelitza.ipynb` / `modelitza.py`: El notebook principal de modelització. Realitza la validació creuada, identifica el millor model de boosting i optimitza els hiperparàmetres per a cada nivell regional i causa.
* `predict.ipynb` / `predict.py`: Genera les prediccions finals i la inferència sobre nous períodes utilitzant els models prèviament entrenats.
* `main_gui.ipynb`: Una interfície visual basada en `ipywidgets` que permet parametritzar i executar fàcilment tot el pipeline de forma interactiva sense interactuar directament amb la configuració del codi.
* `example_model_results.xlsx`: Exemple de sortida de l'informe generat pel procés de manera nativa.
* `01_VAZQUEZ_MARTI_MANUEL_MEMÒRIA.pdf`: El document original del treball de final de grau on es detalla la metodologia.

>(Nota: A diferència del treball original de grau, aquesta adaptació no inclou la interfície de línia de comandes de Textual ni l'anàlisi explicatiu basat en SHAP, centrant-se sòlidament en l'adaptabilitat de l'execució predictiva.)

## Instal·lació i Configuració

1. **Clonar el repositori:**
   ```bash
   git clone <repository-url>
   cd <repository-name>
   ```

2. **Instal·lar dependències:**
   El repositori utilitza `uv` per a reduir la sobrecàrrega d'execució. Si fas servir `uv`:
   ```bash
   uv sync
   ```
   O si prefereixes el `pip` estàndard:
   ```bash
   pip install -r requirements.txt
   ```

## Execució

Pots executar el projecte de dues formes principals:

### Execució Interactiva (Recomanat per a exploració visual)
Obre `main_gui.ipynb` en el teu entorn Jupyter. Utilitzant la interfície, selecciona els scripts a executar (`etl`, `modelitza`, `predict`), configura els paràmetres d'idioma i regió, i revisa els registres d'execució a la consola no bloquejant directament dins de la interfície del notebook.

### Execució de Scripts
Executa els scripts de python extrets en seqüència. Assegura't de definir les variables necessàries (com rutes o variables objectiu dins del codi si formés falta).
```bash
python etl.py
python modelitza.py
python predict.py
```

### Paràmetres dels Notebooks / Scripts i Arxius Generats

#### `etl.ipynb` / `etl.py`
- **Paràmetres**: No necessita paràmetres.
- **Arxius generats**:
  - `cd_df_sense_pivotar.xlsx`: Conté les causes de mort ICD-10 (CIM-10) dels 5 grups escollits de cada territori NUTS des del 2011 al 2023.
  - `cd_df_brut.xlsx`: Dades ja pivotades de les causes de mort i només a nivell NUTS.
  - `cd_df_net.xlsx`: El mateix que l'anterior però amb les dades ja depurades (sense nuls i amb els valors corregits).
  - `mv_nuts2_xy.xlsx`: Dataset de territoris NUTS2 amb els seus centroides X i Y i els NUTS ancestres als quals pertany cadascun jeràrquicament. Es genera per a un ús fàcil en visualitzacions i treballs addicionals en programes externs.
  - `mv_nuts2_xy.csv`: El mateix que l'anterior però en format CSV, per a l'ús amb els scripts.
  - `dades_to_train_test.xlsx`: Dataset amb les dades ja preparades per ser utilitzades en l'entrenament i testejat del model. Es genera per a un ús fàcil en visualitzacions i treballs addicionals en programes externs.
  - `dades_to_train_test.csv`: El mateix que l'anterior però en format CSV, per a l'ús amb els scripts.
  - `pop_predictor.xlsx`: Dataset preparat per obtenir resultats de predicció dels darrers anys, desconeixent les causes de mort. Es genera per a un ús fàcil en visualitzacions i treballs addicionals en programes externs.
  - `pop_predictor.csv`: El mateix que l'anterior però en format CSV, per a l'ús amb els scripts.

#### `modelitza.ipynb` / `modelitza.py`
- **Paràmetres**: Es pot executar sense paràmetres o afegint:
  - `-b`, `--basic`: Amb el qual entrena els models amb hiperparàmetres per defecte, no optimitzats. Redueix dràsticament el temps d'execució.
- **Arxius generats**:  
  - `analisi_models.xlsx`: Conté la puntuació ponderada dels models entrenats amb diferents hiperparàmetres, amb les seves mètriques d'avaluació, ordenats de major a menor puntuació. Determina quin model i quins hiperparàmetres utilitzar per analitzar cada causa de mort.
  - `selected_models.pkl`: Conté els models entrenats amb els hiperparàmetres seleccionats per a cada causa de mort, per poder-se utilitzar en l'script de predicció. És un arxiu binari generat amb `joblib` amb compressió afegida.
  - `dadesModel.pkl`: Conté les dades utilitzades per entrenar i testejar els models. És un arxiu binari generat amb `joblib` amb compressió afegida.
  - `informe_analisi_modelitzacio_YYYYMMDD.xlsx`: Informe amb els resultats de l'anàlisi aplicats a les dades d'entrenament i testejat. Tot i que l'entrenament s'hagi realitzat a nivell global, es mostren mètriques SMAPE per a nivells NUTS0, NUTS1 i NUTS2. (YYYYMMDD és la data en què es va generar l'informe).

#### `predict.ipynb` / `predict.py`
- **Paràmetres**: Es pot executar sense paràmetres o afegint:
  - `-r`, `--rolling`: Per entrenar el model amb tot el conjunt de dades (entrenament i testejat), pot millorar la precisió i no penalitza el sobreajustament, ja que les dades a predir són totalment independents. *Aquest paràmetre és excloent amb `-t, --train`.*
  - `-t`, `--train`: Per entrenar el model de nou amb els mateixos hiperparàmetres, però amb noves dades. Això pot ser útil si s'han afegit noves files al dataset original o s'utilitza un altre dataset. *Aquest paràmetre és excloent amb `-r, --rolling`.*
  - `-p`, `--predir PREDIR`: On *PREDIR* és el conjunt de dades del qual es volen predir quantes causes de mort de cada tipus es generaran, per defecte és `pop_predictor.csv`.
- **Arxius generats**:
  - `prediccio_yyyy_YYYY.xlsx`: Dataset amb les prediccions de les causes de mort per a cada territori NUTS2 avaluat. (yyyy_YYYY és el rang d'anys per al qual s'han realitzat les prediccions).
  - `prediccio_yyyy_YYYY.csv`: El mateix que l'anterior però en format CSV, per a l'ús amb els scripts.

## Suggeriments de Millora i Ús

- Es poden canviar o afegir grups de causa de mort, la funcionalitat és la mateixa.
- Es poden afegir conjunts d'hiperparàmetres i ajustar-los als desitjats per a una major precisió. De la mateixa manera, es poden afegir altres models de machine learning per comparar amb els actuals.
- Com està demostrat, també és possible aplicar el model a conjunts de dades similars, només cal adaptar-lo.
- A un nivell més profund, es pot millorar el model afegint més variables predictores, que puguin influir en les variables objectiu.

## Autor

**Manuel Vázquez Martí**  
Graduat en Ciència de Dades Aplicada (UOC)  
Nota final: 9.02 / 10

- Experiència en sistemes sanitaris (CatSalut)
- Enfocat en el modelatge predictiu i la presa de decisions basada en dades

## Agraïments
- A Alba Benaque Vidal, gerent d'Organitzacions del Servei Català de la Salut, promotora del projecte.
- A Sílvia Prior Cayuela, referent tècnic i tutora del Servei Català de la Salut per la seva ajuda i orientació en el projecte.
- A Alfons Marquès i Cancio, director del treball, per la seva paciència i els seus suggeriments.

## Llicència
Aquest projecte i la seva fonamentació teòrica, basada en el Treball de Final de Grau original, es distribueixen sota la llicència [Creative Commons Attribution 3.0 España (CC BY 3.0 ES)](https://creativecommons.org/licenses/by/3.0/es/).

_Abril 2026_

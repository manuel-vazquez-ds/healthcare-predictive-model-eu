# %% [markdown]
# **Version 01:**  
# Improved and changed file of causes of death  
# 
# **Version 02:**  
# Added an improved method to read the dataset.
# 
# **Version 03:**  
# Changed the dataset to process to **"Causes of death - deaths by NUTS 2 region of residence and occurrence, 3 year average"**
# 
# **Version 04:**  
# Created a python file(mvm_eines.py) to store reusable functions.
# 
# **Version 05:**  
# Start to work with Eurostat population  
# 
# **Version 06:**  
# Continue doing ETL in Eurostat population  
# 
# **Version 07:**  
# Still preparing Eurostat population but changing loading policies adding package **pooch**  
# 
# **Version 08:**  
# Ending NUTS2 ETL and finishing prepare model and prediction data  
# 
# **Version 09:**  
# <!-- en: --><b> All ".copy()" procedures are removed when copying dataframes to reduce memory load</b><br>
# <!-- es: --><i> Se eliminan todos los procedimientos ".copy()" al copiar dataframes para reducir la carga de memoria</i><br>
# <!-- ca: --><i> Es remouen tots els ".copy()" procediments al copiar dataframes per reduïr càrrega de memòria</i><br>
# 
# **Version 10 / Version etl.ipynb:**  
# <!-- en: --><b>Eliminating part of the test code and renaming to etl.ipynb.<br>All notebook display() outputs have also been converted to print() format.</b><br>
# <!-- es: --><i>Eliminando parte del código de prueba y renombrando a <b>etl.ipynb</b>.<br>También se han convertido todas las salidas por pantalla del formato display() del notebook a formato print().</i><br>
# <!-- ca: --><i>Eliminant part del codi de prova i renombrant a <b>etl.ipynb</b>.<br>També s'han convertit totes les sortides per pantalla del format display() del notebook a format print().</i><br>
# 
# 

# %% [markdown]
# ### Execution context / (es) Contexto de ejecución / (ca) Context d'execució
# ---
# <!--en: --><b>Results are intentionally kept visible during execution to monitor the data transformation's progress.</b><br>
# <!--es: --><i>Los resultados se mantienen visibles durante la ejecución para monitorear el progreso de la transformación de datos.</i><br>
# <!--ca: --><i>Els resultats es mantenen visibles durant l'execució per monitorar el progrés de la transformació de dades.</i><br>
# ---

# %%
print("\n\nExecution context / (es) Contexto de ejecución / (ca) Context d'execució")
print("-"*80)
print("Results are intentionally kept visible during execution to monitor the data transformation\'s progress.")
print("Se han mantenido los resultados visibles intencionadamente durante la ejecución para monitorear el progreso de la transformación de los datos.")
print("S\'han mantingut les sortides del notebook visibles expressament per monitorar el procés de transformació.")
print("-"*80+"\n")

# %%
import re

# en: I add my local module with some reusable functions made by me
# es: Añado mi módulo local con algunas funciones reutilizables hechas por mí
# ca: Afegixo el meu mòdul local amb algunes funcions reutilitzables fetes per mi

import mvm_eines as mv

# %% [markdown]
# <!-- en: --><b> Now, we are going to present the results</b><br>
# <!-- es: --><i> Ahora, vamos a presentar los resultados</i><br>
# <!-- ca: --><i> Ara, aAnem a presentar els resultats</i><br>

# %%
# en: Link to the causes of death, as it is stated by Eurostat
# es: Enlace a las causas de muerte, tal como las pone Eurostat
# ca: Adreça de les causes de mort, tal com la posa Eurostat

# https://ec.europa.eu/eurostat/api/dissemination/sdmx/2.1/data/hlth_cd_yro?format=TSV&compressed=true

# en: Load from my disc
# es: Carga desde mi disco
# ca: Càrrega des del meu disc

hlth_cd_yro_file = "./estat_hlth_cd_yro.tsv.gz"
hlth_cd_yro_url = "https://ec.europa.eu/eurostat/api/dissemination/sdmx/2.1/data/hlth_cd_yro?format=TSV&compressed=true"
df_cd=mv.load_dataset(hlth_cd_yro_file,hlth_cd_yro_url)


# %% [markdown]
# <!-- en: --><b> We can see that the first column contains a set of non-numeric variables, so we will have to convert them</b><br>
# <!-- es: --><i> Podemos ver que la primera columna contiene un conjunto de variables no numéricas, por lo que tendremos que convertirlas</i><br>
# <!-- ca: --><i> Podem veure que la primera columna conté un conjunt de variables no numériques, pel que les haurem de convertir</i><br>

# %%
df_cd= mv.clean_first_column(df_cd)
print(df_cd.head())

# %% [markdown]
# I'll start by removing all [age] different from 'TOTAL'

# %%
df_cd = df_cd[df_cd['age']=='TOTAL']


# %% [markdown]
# <!-- en: --><b> We retains only five ICD-10 codes</b><br>
# <!-- es: --><i> Retenemos solo cinco códigos CIM-10</i><br>
# <!-- ca: --><i> Retenem només cinc codis CIM-10</i><br>
# 
# [C] Malignant neoplasms (C00-C97)  
# [G-H] Diseases of the nervous system and the sense organs (G00-H95)  
# [I] Diseases of the circulatory system (I00-I99)   
# [J] Diseases of the respiratory system (J00-J99)  
# [K] Diseases of the digestive system (K00-K93)  
# 
# <!-- en: --><b>To filter sex</b><br>
# <!-- es: --><i> Para filtrar sexo</i><br>
# <!-- ca: --><i> Per filtrar sexe</i><br>
# 
# T Total
# 
# <!-- en: --><b>To filter deaths reported</b><br>
# <!-- es: --><i> Para filtrar muertes informadas</i><br>
# <!-- ca: --><i> Per filtrar morts informades</i><br>
# TOT_IN All deaths reported in the country

# %%

try:
    df_cd=df_cd[(df_cd['icd10'].isin(['C', 'G_H', 'I', 'J', 'K']) & (df_cd['sex']=='T')
                 & (df_cd['resid']=='TOT_IN'))].drop(['freq','unit','sex','age','resid'],axis=1)
except:
    print("Executed previously before")
finally:
    print(df_cd.head())
    print(f'New dataframe\'s dimensions: {df_cd.shape}')

# %% [markdown]
# <!-- en: --><b>Now, I'm going to convert year columns to numeric</b><br>
# <!-- es: --><i>Ahora, voy a convertir las columnas de año a numeric</i><br>
# <!-- ca: --><i>Ara, vaig a convertir les columnes d'any a numeric</i><br>

# %%

df_cd = mv.clean_year_cols(df_cd)
print(df_cd.head())


# %%
# en: We export the dataframe without pivoting
# es: Exportamos el dataframe sin pivotar
# ca: exportem el dataframe sense pivotar

df_cd.to_excel('cd_df_sense_pivotar.xlsx', index=False)

# %% [markdown]
# That's the moment where we only retains data in NUTS2 level.

# %%
df_cd = df_cd[df_cd['geo'].str.len()==4]
print(f'New dataframe\'s dimensions: {df_cd.shape}')

# %% [markdown]
# And now it's time to pivot the data, passing all existing years to rows and all ic10 codes to columns.

# %%
# en: Passing years' columns to rows
# es: Pasando las columnas de años a filas
# ca: Passant les columnes d'anys a files

df_cd_pivot = df_cd.melt(id_vars=['icd10','geo'],var_name='year',value_name='deaths')
# en: Now, we pass icd10 death causes to columns
# es: Ahora pasamos causas de muerte cim10 a columnas
# ca: Ara passem causas de mort cim10 a columnes

df_cd_pivot = df_cd_pivot.pivot(index=['year','geo'],columns='icd10',values='deaths').reset_index()
df_cd_pivot['year']=df_cd_pivot['year'].astype('int')
df_cd_pivot.columns.name=None
print(df_cd_pivot.head())
print(f'New dataframe\'s dimensions: {df_cd_pivot.shape}')
df_cd = df_cd_pivot

# %%
print(df_cd.head())
# en: We show rows with missing values
# es: Mostramos las filas con valores nulos
# ca: Mostrem les files amb valors nuls

rows_df_cd = df_cd.shape[0]
print(f'files en df: {rows_df_cd}')
rows_df_cd_na = df_cd[df_cd.isna().any(axis=1)].shape[0]
print(f'files amb na: {rows_df_cd_na}')
print (f'amb df_cd net, haurien de quedar: {rows_df_cd-rows_df_cd_na}')
df_cd.to_excel('df_cd_brut.xlsx', index=False)
print(df_cd[df_cd[['C','G_H','I','J','K']].isna().any(axis=1)].shape[0])
# drop na rows from df_cd
df_cd.dropna(how='any',inplace=True)
print(df_cd.shape[0])
df_cd.to_excel('df_cd_net.xlsx',index=False)


# %%
pop_euro_url ='https://ec.europa.eu/eurostat/api/dissemination/sdmx/2.1/data/demo_r_pjangroup?format=TSV&compressed=true'
pop_euro_file = './estat_demo_r_pjangroup.tsv.gz'
pop_euro = mv.load_dataset(pop_euro_file,pop_euro_url)
print(pop_euro.head())

# %% [markdown]
# Let's remove all columns from years before 2011

# %%
pop_euro.rename(columns=lambda x: re.sub(r'(\d{4})\s+$', r'\1', x), inplace=True)


# %%
# en: Now we select all columns to drop (all before 2011)
# es: Ahora seleccionamos todas las columnas a eliminar (todas antes de 2011)
# ca: Ara seleccionem totes les columnes a eliminar (totes abans de 2011)

pop_euro.drop(columns = [x for x in pop_euro.columns[1:] if int(x)<2011],inplace=True)
print(pop_euro.head())

# %% [markdown]
# Now, let's convert the first column to multiple variables.

# %%
pop_euro = mv.clean_first_column(pop_euro)
print(pop_euro.head())

# en: ... and we show the possible different values of each column
# es: ... y mostramos los posibles diferentes valores de cada columna
# ca: ... i mostrem els possibles diferents valors de cada columna

print(f'\n(en) Unique values per variable \\ (es) Valores únicos por variable \\ (ca) Valors únics per variable')
for x in range(5):
    print(f'Variable \'{pop_euro.columns[x]}\': {pop_euro[pop_euro.columns[x]].unique()}')

# %%
# primera prova
pop_euro = mv.clean_year_cols(pop_euro)
print(pop_euro)

# %% [markdown]
# Now, retain only NUTS2

# %%
pop_euro = pop_euro[pop_euro['geo'].str.len()==4]
print(f'New dataframe\'s dimensions: {pop_euro.shape}')
print(pop_euro.head())

# %% [markdown]
# <!-- en: --><b> Removing unnecessary columns</b><br>
# <!-- es: --><i> Eliminando columnas innecesarias</i><br>
# <!-- ca: --><i> Eliminant columnes innecesaries</i><br>

# %%
pop_euro.drop(columns=['freq','unit'],inplace=True)
print(pop_euro.head())

# %% [markdown]
# Removing unnecessary values

# %%
# en: I'll do it column by column
# es: Lo haré columna por columna
# ca: Ho faré columna per columna

pop_euro = pop_euro.loc[~pop_euro['sex'].isin(['T'])]
pop_euro = pop_euro.loc[~pop_euro['age'].isin(['TOTAL', 'UNK', 'Y_GE75', 'Y_GE80'])]

print(pop_euro.head())
print(f'New dataframe\'s dimensions: {pop_euro.shape}')

# %%
pop_euro[(pop_euro==0).any(axis=1)]

# %%
pop_euro.loc[pop_euro['geo'].str.contains('..XX',regex=True)]

# %%
pop_euro = pop_euro.loc[~pop_euro['geo'].str.contains('..XX',regex=True)]
print(pop_euro.head())
print(f'New dataframe\'s dimensions: {pop_euro.shape}')

# %%
pop_euro[(pop_euro==0).any(axis=1)]

# %%
pop_euro = pop_euro.loc[pop_euro['geo']!='NO0B']
print(pop_euro.head())
print(f'New dataframe\'s dimensions: {pop_euro.shape}')

# %% [markdown]
# Now, melting and pivoting

# %%
pop_euro_pivot = pop_euro.assign(age_sex=lambda x:x['age']+"_"+x['sex']).drop(columns=['sex','age'])
print(pop_euro_pivot.head())
# en: Passing years' columns to rows
# es: Pasando las columnas de años a filas
# ca: Passant les columnes d'anys a files

pop_euro_pivot = pop_euro_pivot.melt(id_vars=['age_sex','geo'],var_name='year',value_name='population')
pop_euro_pivot['year']=pop_euro_pivot['year'].astype('int')
pop_euro_pivot =pop_euro_pivot.pivot(index=['year', 'geo'],columns='age_sex',values='population')
pop_euro_pivot.reset_index(inplace=True)
pop_euro_pivot.columns.name=None
print(pop_euro_pivot.head())
print(f'New dataframe\'s dimensions: {pop_euro_pivot.shape}')

# %%
pop_euro_pivot[(pop_euro_pivot.isna().any(axis=1))]

# %% [markdown]
# <!--en: --><b>Finally, I remove all nuts2-year with any NA value per row</b><br>
# <!--es: --><i>Para finalizar, elimino todos los años NUTS2 con algún NA por fila</i><br>
# <!--ca: --><i>Per finalitzar, elimino tots els anys NUTS2 amb algun NA per fila</i><br>

# %%
pop_euro_pivot = pop_euro_pivot[~(pop_euro_pivot.isna().any(axis=1))]
pop_euro_pivot.reset_index()
print(pop_euro_pivot.head())
print(f'New dataframe\'s dimensions: {pop_euro_pivot.shape}')
pop_euro = pop_euro_pivot

# %% [markdown]
# **Anem amb les dades geogràfiques**  
# _Let's with geographical data_  
# _Vamos com los datos geográficos_

# %%
# New in version 7
places_file = "NUTS_RG_20M_2024_4326.gpkg"
places_url = "https://gisco-services.ec.europa.eu/distribution/v2/nuts/gpkg/NUTS_RG_20M_2024_4326.gpkg"
places=mv.load_geo(places_file,places_url)
print(places.head())

# %% [markdown]
# **Ara, converteixo les dades en format NUTS2 amb centroid i referència als nivells superiors**  
# _Now, I convert the data to NUTS2 format with centroid and reference to upper levels_  
# _Ahora, convierto los datos en formato NUTS2 con centroide y referencia a los niveles superiores_

# %%
# Nou en versió 7, reduïm tot el procés generat a la versió 6.
# New in version 7, we reduce the entire process generated in version 6.
# Nuevo en la versión 7, reducimos todo el proceso generado en la versión 6.

nuts2 = mv.get_NUTS2(places)
print(nuts2.head(10))

# %% [markdown]
# **Ara, ja tenim totes les dades netes i necessàries per poder muntar els fitxers necessaris per fer la modelització**  
# _Now, we have all the data cleaned and necessary to be able to mount the files needed to do the modeling_  
# _Ahora, ya tenemos todas las datos limpios y necesarios para montar los archivos necesarios para hacer el modelo_

# %%
df_model_data = df_cd.merge(pop_euro, on=['year', 'geo'], how='inner')
print(df_model_data.head())
print(f'New dataframe\'s dimensions: {df_model_data.shape}')

# %%
print(df_model_data.columns)

# %%
df_model_data = mv.nuts2_to_centroids(df_model_data, "geo")
print(df_model_data.head())

# %% [markdown]
# <!-- en: --><b>Note: These data, to train the model, do not have to contain the entire population, nor all causes of death of a given type.<br>What it contains are the consistent data.</b><br>
# <!-- es: --><i>Nota: Estos datos, para entrenar el modelo, no tienen que contener toda la población, ni todas las causas de muerte de un tipo determinado.<br>Lo que contiene son los datos consistentes.</i><br>
# <!-- ca: --><i>Nota: Aquestes dades, per entrenar al model, ni han de contenir tota la població, ni totes les causes de mort d'un tipus determinat. <br>El que conté son les dades consistents.<i><br>
# 

# %%
# ca: Anem a gravar les dades al disc
# en: Let's to save data on disc
# es: Guardemos los datos en disco

try:
    df_model_data.to_csv("dades_to_train_test.csv",index=False)
except:
    pass
try:
    df_model_data.to_excel("dades_to_train_test.xlsx",index=False)
except:
    pass

# %%
print(f'mirem si hi ha algun na: {df_model_data.isna().any().any()}')

# %%
# ca: Anem a conservar els anys que conté el conjunt de dades per fer-los servir més tard
# en: Let's to preserve years contained in dataset to use later.
# es: Vamos a conservar los años que contiene el dataset para usarlos más tarde.

anys_en_model = df_model_data.year.unique().tolist()
print(anys_en_model)

# %% [markdown]
# <!-- ca: --> <b>Ara que ja tenim les dades del model preparades, anem a escollir dades per fer inferència i calcular les causes de mort. </b><br>
# <!-- en: --> <i>Now that we have the data of the model prepared, we will select data to make inference and calculate the causes of death.</i><br>
# <!-- es: --> <i>Ya tenemos los datos del modelo preparados, vamos a escoger datos para hacer inferencia y calcular las causas de muerte. </i></small>

# %%
print(pop_euro.year.unique())
print(pop_euro.shape)
print(df_model_data.shape)

# %%
# ca: utilitzarem el rang d'anys que no apareix en el dataset d'entrenament
# en: we use the range of years that are not in the training dataset
# es: utilizaremos el rango de años que no están en el conjunto de entrenamiento

pop_predictor = pop_euro[~pop_euro.year.isin(anys_en_model)]
print(pop_predictor.shape)
print(f'Predictor dataframe dimensions: {pop_predictor.shape}')
print(f'Predictor dataframe years: {pop_predictor.year.unique()}')
print(pop_predictor.head())


# %% [markdown]
# <!-- ca: --> <b>Canviem l'identificador NUTS2 pels centroids</b><br>
# <!-- en: --> <i><small> Let's change the NUTS2 identifier to the centroids</small></i><br>
# <!-- es: --> <i><small> Cambiamos el identificador NUTS2 por los centroides</small></i><br>

# %%
pop_predictor = mv.nuts2_to_centroids(pop_predictor,'geo')
print(pop_predictor.head())

# %%
# ca: Anem a gravar les dades per fer inferència al disc
# en: Let's to save data to do inference to disk
# es: Guardemos los datos para hacer inferéncia en el disco

try:
    pop_predictor.to_csv("pop_predictor.csv",index=False)
except:
    pass
try:
    pop_predictor.to_excel("pop_predictor.xlsx",index=False)
except:
    pass

# %% [markdown]
# <!-- ca: -->
# - Qualsevol persona pot generar el seu dataset predictor, mantenint la mateixa estructura.  
# Recordar que els centroides es poden obtenir amb la funció <i>nuts2_to_centroids</i> del mòdul <i>mvm_eines.py</i>.  
# 
# <!-- en: -->
# - Any person can generate their predictor dataset, maintaining the same structure.  
# Remember that the centroids can be obtained with the function <i>nuts2_to_centroids</i> in the <i>mvm_eines.py</i> module.  
# 
# <!-- es: -->
# - Cualquier persona puede generar su dataset predictor, manteniendo la misma estructura.  
# Recordar que los centroides se pueden obtener con la función <i>nuts2_to_centroids</i> del módulo <i>mvm_eines.py</i>.
# 

# %% [markdown]
# <!-- en --> <b>pop_predictor DataFrame Column Descriptions</b><br>
# <!-- ca --> <i>Descripció de cada columna de pop_predictor DataFrame</i><br>
# <!-- es --> Descripción de cada columna de pop_predictor DataFrame<br><br>
# 
# | Column Name | Description | Data Type |
# |-------------|-------------|-----------|
# | year | Year of the population data | Integer |
# | Y10-14_F | Population count for females aged 10-14 years | Float |
# | Y10-14_M | Population count for males aged 10-14 years | Float |
# | Y15-19_F | Population count for females aged 15-19 years | Float |
# | Y15-19_M | Population count for males aged 15-19 years | Float |
# | Y20-24_F | Population count for females aged 20-24 years | Float |
# | Y20-24_M | Population count for males aged 20-24 years | Float |
# | Y25-29_F | Population count for females aged 25-29 years | Float |
# | Y25-29_M | Population count for males aged 25-29 years | Float |
# | Y30-34_F | Population count for females aged 30-34 years | Float |
# | Y30-34_M | Population count for males aged 30-34 years | Float |
# | Y35-39_F | Population count for females aged 35-39 years | Float |
# | Y35-39_M | Population count for males aged 35-39 years | Float |
# | Y40-44_F | Population count for females aged 40-44 years | Float |
# | Y40-44_M | Population count for males aged 40-44 years | Float |
# | Y45-49_F | Population count for females aged 45-49 years | Float |
# | Y45-49_M | Population count for males aged 45-49 years | Float |
# | Y5-9_F | Population count for females aged 5-9 years | Float |
# | Y5-9_M | Population count for males aged 5-9 years | Float |
# | Y50-54_F | Population count for females aged 50-54 years | Float |
# | Y50-54_M | Population count for males aged 50-54 years | Float |
# | Y55-59_F | Population count for females aged 55-59 years | Float |
# | Y55-59_M | Population count for males aged 55-59 years | Float |
# | Y60-64_F | Population count for females aged 60-64 years | Float |
# | Y60-64_M | Population count for males aged 60-64 years | Float |
# | Y65-69_F | Population count for females aged 65-69 years | Float |
# | Y65-69_M | Population count for males aged 65-69 years | Float |
# | Y70-74_F | Population count for females aged 70-74 years | Float |
# | Y70-74_M | Population count for males aged 70-74 years | Float |
# | Y75-79_F | Population count for females aged 75-79 years | Float |
# | Y75-79_M | Population count for males aged 75-79 years | Float |
# | Y80-84_F | Population count for females aged 80-84 years | Float |
# | Y80-84_M | Population count for males aged 80-84 years | Float |
# | Y_GE85_F | Population count for females aged 85 years and over | Float |
# | Y_GE85_M | Population count for males aged 85 years and over | Float |
# | Y_LT5_F | Population count for females under 5 years | Float |
# | Y_LT5_M | Population count for males under 5 years | Float |
# | centroid_x | X coordinate of the geographic centroid | Float |
# | centroid_y | Y coordinate of the geographic centroid | Float |



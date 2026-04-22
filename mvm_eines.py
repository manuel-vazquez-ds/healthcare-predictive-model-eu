import pandas as pd
import re
import geopandas as gpd
import pooch
import openpyxl

# en: Function to download directly in the Eurostat format
# es: Función para descargar directamente en el formato Eurostat
# ca: Funció per descarregar directament en el format Eurostat

def download_from_eurostat(url:str,sep:str='\t',encoding:str='utf-8')->pd.DataFrame:
    try:
        el_df = pd.read_csv(url,sep=sep,encoding=encoding,compression='gzip')
    except:
        raise RuntimeError(f"Error recovering dataset \'{url}\'")
    
    return el_df
# en: Using pooch, we're going to create a high-level function
#     to try to load the files, checking locally first and then at Eurostat.
# es: Utilizando pooch, vamos a crear una función de nivel superior para intentar cargar los archivos,
#     mirando primero localmente y luego en Eurostat.
# ca: Utilitzant pooch, anem a fer una funció de nivell superior per mirar de carregar els fitxers,
#     primer que miri en local i posteriorment a Eurostat.

def load_dataset(file:str,url:str,sep:str='\t',encoding:str='utf-8',)->pd.DataFrame:
    try:
        return pd.read_csv(pooch.retrieve(url=url,fname=file,path="."),sep=sep,encoding=encoding,compression='gzip')
    except:
        raise RuntimeError(f"Error recovering dataset \'{url}\'")

# en: Function to clean the first column, which has labels
# es: Función para limpiar la primera columna, que tiene etiquetas
# ca: Funció per netejar la primera columna, que té etiquetes

def clean_first_column(df:pd.DataFrame)->pd.DataFrame:
    # en: Make a set of columns from the first column, which has several labels.
    # es: Crea el conjunto de columnas a partir de la primera columna, que tiene varias etiquetas.
    # ca: Crea el conjunt de columnes a partir de la primera columna, que té vàries etiquetes.

    df0 = df.iloc[:,0].str.split(',', expand=True)
    labels_cols = df.columns[0].split(',')
    # en: Removing '\TIME*', from the last ocurrence (if exists).
    # es: Eliminando '\TIME*', de la última ocurrencia (si existe).
    # ca: Eliminant '\TIME*', de l'última ocurrència (si existeix).

    labels_cols[-1] = labels_cols[-1].partition('\\TIME')[0]
    df0.columns =labels_cols
    return pd.concat([ df0, df.iloc[:,1:]], axis=1)

# en: Function to clean the columns with year format, and convert the non numeric to N/A
# es: Función para limpiar las columnas con formato año, y convertir los no numéricos a N/A
# ca: Funció per netejar les columnes amb format any, i convertir els no numèrics a N/A

def clean_year_cols(df:pd.DataFrame)->pd.DataFrame:
    # en: Rename (clean) year columns inplace, removing trailing (ending) spaces.
    # es: Renombra (limpia) las columnas de año en su lugar, eliminando espacios finales.
    # ca: Renombra (neteja) les columnes de any en el seu lloc, eliminant espais finals.
    
    df.rename(columns=lambda x: re.sub(r'(\d{4})\s+$', r'\1', x), inplace=True)
    # en: Now, to numeric
    # es: Ahora, a numérico
    # ca: Ara, a numèric

    year_cols=[x for x in df.columns if re.match(r'\d{4}',x)]
    for x in year_cols:
        df[x] = pd.to_numeric(df[x],errors='coerce')
    return df

# en: A super function to read data with Pooch.
# es: Una super función de lectura abreviada con Pooch.
# ca: Una super funció de lectura abreviada amb Pooch.

def load_geo(file:str,url:str)->gpd.geodataframe.GeoDataFrame:
    try:
        return gpd.read_file(pooch.retrieve(url=url,fname=file,path="."))
    except:
        raise RuntimeError(f"Error recovering geodata \'{url}\'")
    
# en: Here we create the base dataframe to obtain NUTS2 with centroids.
# es: Aquí creamos el dataframe base para obtener NUTS2 con centroides.
# ca: Aquí creem el dataframe base per obtenir NUTS2 amb centroides.

def get_NUTS2(geodata:gpd.geodataframe.GeoDataFrame)->pd.DataFrame:
    nuts = geodata[geodata.LEVL_CODE<3][['NUTS_ID', 'LEVL_CODE', 'NUTS_NAME']]
    nuts[['centroid_x', 'centroid_y']] = geodata.geometry.to_crs("EPSG:25831").centroid.get_coordinates()
    nuts.reset_index(inplace=True,drop=True)
    # en: Prepare two temporary dataframes, one for each NUTS level.
    # es: Preparando dos dataframes temporales, uno para cada nivel de NUTS.
    # ca: Preparant dos dataframes temporals, un per a cada nivell de NUTS.

    nuts0, nuts1 = pd.DataFrame(), pd.DataFrame()
    nuts0[['NUTS0_ID', 'NUTS0_NAME']] = nuts[nuts.LEVL_CODE==0][['NUTS_ID', 'NUTS_NAME']].reset_index(drop=True)
    nuts1[['NUTS1_ID', 'NUTS1_NAME']] = nuts[nuts.LEVL_CODE==1][['NUTS_ID', 'NUTS_NAME']].reset_index(drop=True)

    # en: Make a cleared NUTS LEVEL 2.
    # es: Hacer un NUTS LEVEL 2 limpio.
    # ca: Fer un NUTS LEVEL 2 net.
    
    nuts2 = nuts[nuts.LEVL_CODE==2].drop(columns='LEVL_CODE').reset_index(drop=True)

    # en: Prepare NUTS2 to merge info from upper levels.
    # es: Preparar NUTS2 para fusionar información de niveles superiores.
    # ca: Preparar NUTS2 per fusionar informació de nivells superiors.
    
    nuts2['1'] = nuts2.NUTS_ID.str[:3]
    nuts2['0'] = nuts2.NUTS_ID.str[:2]
    nuts2 = nuts2.merge(nuts1, left_on='1',right_on='NUTS1_ID',validate="many_to_one").drop(columns='1').copy()
    nuts2 = nuts2.merge(nuts0, left_on='0',right_on='NUTS0_ID',validate="many_to_one").drop(columns='0').copy()

    # en: Round centroids for testing purposes.
    # es: Redondear centroides para fines de prueba.
    # ca: Arrodonir centroides per a propòsits de prova.
    # nuts[['centroid_x', 'centroid_y']] =nuts.assign(centroid_x=lambda x: x.centroid_x.round(2), centroid_y=lambda x: x.centroid_y.round(2))
    
    nuts2 = nuts2.assign(centroid_x=lambda x: x.centroid_x.round(2), centroid_y=lambda x: x.centroid_y.round(2))

    # en: Save to CSV for other purposes if necessary.
    # es: Guardar en CSV para otros fines si es necesario.
    # ca: Guardar en CSV per a altres propòsits si és necessari.
    
    try:
        nuts2.to_csv("mv_nuts2_xy.csv",index=False,)
    except:
        print("error nuts")
        pass
    try:
        nuts2.to_excel("mv_nuts2_xy.xlsx",index=False ,engine="openpyxl")
    except:
        pass
    return nuts2

def nuts2_to_centroids(df:pd.DataFrame, column:str="NUTS_ID") -> pd.DataFrame:
    df = df.copy()
    # en: Read from nuts2_xy.csv file.
    # es: Leer del archivo nuts2_xy.csv.
    # ca: Llegir del fitxer nuts2_xy.csv.
    
    nuts2cx = pd.read_csv("mv_nuts2_xy.csv")
    if column == "NUTS_ID":
        return df.merge(nuts2cx[['NUTS_ID','centroid_x','centroid_y']], on="NUTS_ID").drop(columns=['NUTS_ID']).copy()
    return df.merge(nuts2cx[['NUTS_ID','centroid_x','centroid_y']], left_on=column,right_on="NUTS_ID").drop(columns=[column, 'NUTS_ID']).copy()
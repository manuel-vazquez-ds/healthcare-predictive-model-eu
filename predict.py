# %% [markdown]
# ### Predictor adaptation to obtain causes of death by NUTS region
# ### Adaptación del predictor para obtener las causas de muerte por región NUTS
# ### Adaptació del predictor per obtenir les causes de morts per regió NUTS
# 
# ---

# %% [markdown]
# **Unique version / Versión única / Versió Única:**  
# 
# --ca: Sols hem fet els canvis mínims per adaptar el predictor al nou conjunt de dades.
# 
# 

# %%
import pandas as pd
# CARREGUEM TOTES LES LLIBRERIES DELS MODELS, perquè poden fer falta depenent dels models que vinguin
# dins del fitxer dels models seleccionats.
from sklearn.base import clone
import joblib
import os, pathlib
import argparse

# %%
#-ca: variables globals

rolling = False
train = False
predir = prediccio_final = None

#-ca: Perquè funcioni al notebook, mirar si error i fer servir cwd

try:
    ruta_script=pathlib.Path(__file__).resolve().parent
except:
    ruta_script=pathlib.Path.cwd()


# %%
def read_model():
    if not os.path.exists(f"selected_models.pkl"):
        raise FileNotFoundError("Models file (selected_models.pkl) isn\'t found. Run routine to train it."+
                                "\n(es) No se encuentra el archivo de modelos (selected_models.pkl). Ejecutar rutina para entrenarlo."+
                                "\n(ca) No es troba l\'arxiu de models (selected_models.pkl). Executar rutina per entrenar-lo.")
    
    model = joblib.load(f"selected_models.pkl")
    
    return model

# %%
def read_dadesXrolling():
    if not os.path.exists(f"dadesModel.pkl"):
        raise FileNotFoundError("Trained data file (dadesModel.pkl) isn\'t found. Run routine to train it."+
                                "\n(es) No se encuentra el archivo de datos entrenados (dadesModel.pkl). Ejecutar rutina para entrenarlo."+
                                "\n(ca) No es troba l\'arxiu de dades entrenades (dadesModel.pkl). Executar rutina per entrenar-lo.")
    
    #-ca: Versió modificada per tenir dades per fer anàlisi explicatiu en el meu treball de fi de grau

    features, targets, *_ = joblib.load(f"dadesModel.pkl")
    
    return features, targets

# %%
def read_preds(dades_a_pred):
    if not os.path.exists(dades_a_pred):
        raise FileNotFoundError("Data to predict file isn\'t found. Run routine to generate the projection."+
                                "\n(es) No se encuentra el archivo de datos a predir. Ejecutar rutina para generar la proyección."+
                                "\n(ca) No es troba l\'arxiu de dades a predir. Executar rutina per generar la projecció.")
    
    dades = pd.read_csv(dades_a_pred)
    
    return dades

# %%
def read_nuts2():
    if not os.path.exists(f"mv_nuts2_xy.csv"):
        raise FileNotFoundError("NUTS2 territory reference file isn\'t found. Run routine to get the NUTS file and generate the mapping."+
                                "\n(es) No se encuentra el archivo de la referencia de territorios NUTS2. Ejecutar rutina para obtener el fichero de NUTS y generar el mapeo."+
                                "\n(ca) No es troba l\'arxiu de la referència de territorios NUTS2. Executar rutina per obtenir el fitxer de NUTS i generar el mapeig.")
    
    nuts2 = pd.read_csv(f"mv_nuts2_xy.csv")
    
    return nuts2

# %%
def read_train_test_data():
    if not os.path.exists(f"dades_to_train_test.csv"):
        raise FileNotFoundError("Training file (dades_to_train_test.csv) isn\'t ready. Run routine to train it."+
                                "\n(es) No se encuentra preparado el archivo para hacer el entrenamiento (dades_to_train_test.csv). Ejecutar rutina para entrenarlo."+
                                "\n(ca) No es troba preparat l\'arxiu per fer l\'entrenament (dades_to_train_test.csv). Executar rutina per entrenar-lo.")

    #-ca: Codi basat en el predictor.
    # Es llegeix el fitxer sencer, ha de tenir el mateix efecte que fer un rolling
    # si no s'han preparat dades noves

    df = pd.read_csv(f"dades_to_train_test.csv")
    feature_cols = [col for col in df.columns if col not in ['year', 'C', 'G_H', 'I', 'J', 'K']]
    target_vars = ['C', 'G_H', 'I', 'J', 'K']
    
    #-ca: Ordeno per any

    df.sort_values(by='year', inplace=True)

    return df[feature_cols], df[target_vars]


# %%
#-ca: mirem de passar els paràmetres de manera global

def modela():
    columns=['Y10-14_F', 'Y10-14_M', 'Y15-19_F',
       'Y15-19_M', 'Y20-24_F', 'Y20-24_M', 'Y25-29_F', 'Y25-29_M', 'Y30-34_F',
       'Y30-34_M', 'Y35-39_F', 'Y35-39_M', 'Y40-44_F', 'Y40-44_M', 'Y45-49_F',
       'Y45-49_M', 'Y5-9_F', 'Y5-9_M', 'Y50-54_F', 'Y50-54_M', 'Y55-59_F',
       'Y55-59_M', 'Y60-64_F', 'Y60-64_M', 'Y65-69_F', 'Y65-69_M', 'Y70-74_F',
       'Y70-74_M', 'Y75-79_F', 'Y75-79_M', 'Y80-84_F', 'Y80-84_M', 'Y_GE85_F',
       'Y_GE85_M', 'Y_LT5_F', 'Y_LT5_M', 'centroid_x', 'centroid_y']
    
    dades = predir if predir.endswith(".csv") else predir + ".csv"

    #-ca: abans haure de comprovar si existeix el fitxer

    a_predir= read_preds(dades)

    #-ca: també llegirem el fitxer dels territoris NUTS2 de referència per matxembrar-lo després

    nuts2 = read_nuts2()

    missing_columns= [column for column in columns if column not in a_predir.columns]
    if missing_columns:
        raise Exception("This dataset doesn't have the necessary columns."+
        "\n(es) Este conjunto de datos no tiene las columnas necesarias."+
        "\n(ca) Aquest conjunt de dades no té les columnes necessàries.")
    
    columnes_nb=['year', 'centroid_x', 'centroid_y']
    new_base= a_predir[[column for column in columnes_nb if column in a_predir.columns]].copy()

    #-ca: llegim el model

    el_model = read_model()

    # si model.clone(), etc...

    # versió 01
    if rolling:
        features, targets = read_dadesXrolling()
    elif train:
        features, targets = read_train_test_data()

    #-ca: comprovem per cadascun dels tipus de mort

    for death in el_model.keys():
        if rolling or train:
            rmodel= clone(el_model[death]['model'])
            rmodel.fit(features, targets[death])
            el_model[death]['model'] = rmodel
        
        new_base[death] = el_model[death]['model'].predict(a_predir[columns]).round(0).astype(int)
    
    #-ca: matxembrem amb els territoris NUTS2

    # new_NUTS_base=new_base.merge(nuts2[["Nom", "Codi", "Codi sector sanitari", "Nom sector sanitari",
    #                                     "Codi regió sanitària","Nom regió sanitària","centroid_x","centroid_y"]], how="left", on= ["centroid_x","centroid_y"]).copy()
    new_NUTS_base=new_base.merge(nuts2, how="left", on= ["centroid_x","centroid_y"]).copy()


    if new_NUTS_base.isna().any().any():
        print(f"Error joining the set's values /"+
         " (es) Error uniendo los valores del conjunto /"+
         " (ca) Error unint els valors del conjunt")
        raise Exception("NUTS2 territories don't match, error joining the set."+
         "\n(es) Los territorios NUTS2 no concuerdan, error uniendo el conjunto."+
         "\n(ca) Territoris NUTS2 no concorden, error unint el conjunt.")
   
    #-ca: eliminem els centroides per deixar-lo mes net

    new_NUTS_base.drop(["centroid_x","centroid_y"], axis=1, inplace=True)

    return new_NUTS_base





# %% [markdown]
# Guardem els resultats

# %%
def guardar_fitxers():
    import os
    try:
        pred_any_min = prediccio_final["year"].min()
        pred_any_max = prediccio_final["year"].max()
        prediccio_final.to_csv(f"prediccio_"+f"{pred_any_min}"+
                                     "_"+f"{pred_any_max}"+".csv")
        prediccio_final.to_excel(f"prediccio_"+f"{pred_any_min}"+
                                     "_"+f"{pred_any_max}"+".xlsx", index=False)
    except Exception as e:
        #-ca: Retorna false si hi ha un error
        return False
    return True

# %%

def principal(**arguments):
    global rolling, train, predir, prediccio_final
        
    rolling= arguments["rolling"]
    train= arguments["train"]
    predir= arguments["predir"]

    # per tests
    # rolling= True
    # train= True
    prediccio_final = modela()

    if guardar_fitxers():
        print("Prediction generated / (es) Predicción generada / (ca) Predicció generada")
        return True
    else:
        print("Error generating the prediction / (es) Error generando la predicción / (ca) Error generant la predicció")
        return False

# %% [markdown]
# Part de la versió 01

# %%
def retorna_arguments():
    arguments = argparse.ArgumentParser(description="Programa predictor de tipus d'alta")
    un_dels_dos = arguments.add_mutually_exclusive_group()
    un_dels_dos.add_argument("-r","--rolling", action="store_true", help="Utilitzar el mateix model però fent rolling-training de les dades")
    un_dels_dos.add_argument("-t","--train", action="store_true", help="Utilitzar el mateix model entrenant-lo amb dades noves")

    arguments.add_argument("-p","--predir", type=str, default="pop_predictor.csv", help="Fitxer de dades a predir")

    # imprimeixo les opcions
    # arguments.print_help()

    # els passem com a known per si estem executant el notebook
    #   (passem la primera part de la tupla, la coneguda)
    return vars(arguments.parse_known_args()[0])

if __name__ == "__main__":
    arguments = retorna_arguments()
    principal(**arguments)



# %% [markdown]
# ### Model adaptation to calculate causes of death by NUTS regions
# ### Adaptación del modelo para calcular las causas de muerte por región NUTS
# ### Adaptació del model per calcular les causes de morts per regió NUTS
# 
# ---

# %% [markdown]
# **Version \ Versió: 05**  
# <!-- en: --><b> Setting the final name (modelitza) and removing test execution lines.</b></br>
# <!-- es: --><i> Poniendo el nombre definitivo (modelitza) y eliminando líneas de ejecución de prueba.</i></br>
# <!-- ca: --><i> Posant el nom definitiu (modelitza) i eliminant línies d'execució de prova.</i></br>
# 
# **Version \ Versió: 04**
# <!-- en: --><b> Doing pending translations.</b></br>
# <!-- es: --><i> Haciendo las traducciones pendientes.</i></br>
# <!-- ca: --><i> Fent les traduccions pendents.</i></br></br>
# 
# **Version \ Versió: 03**
# <!-- en: --><b> We continue translating texts and especially the output of the final report.  </b></br>
# <!-- es: --><i> Continuamos traduciendo textos y sobre todo la salida del informe final.  </i></br>
# <!-- ca: --><i> Continuem traduint textos i sobretot la sortida de l'informe final.  </i></br></br>
# 
# **Version \ Versió: 02**  
# <!-- en: --><b> Most of the variables were renamed so that they better adapt to the data worked on.  </b></br>
# <!-- es: --><i> Renombradas la mayoría de las variables para que se adaptaran más a los datos trabajados.  </i></br>
# <!-- ca: --><i> Renombrades la majoria de les variables perquè s'adaptèssin més a les dades treballades.  </i></br></br>
# 
# **Version \ Versió: 01**
# <!-- en: --><b> Original datasets changed for those processed with Eurostat data, for a first quick testing.  </b></br>
# <!-- es: --><i> Cambiados los datasets originales por los tratados con los datos de Eurostat, para un primer testing rápido.  </i></br>
# <!-- ca: --><i> Canviats els datasets originals pels tractats amb les dades d'Eurostat, per primer testing ràpid.  </i></br>

# %% [markdown]
# ---

# %%
# Import necessary libraries
from IPython.display import display
import argparse
import os
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, root_mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
import numpy as np
import joblib
from datetime import date
import time
import pathlib

# en: For it to work in the notebook, check for error and use cwd
# es: Para que funcione en el notebook, comprobar si hay error y usar cwd
# ca: Perquè funcioni al notebook, mirar si error i fer servir cwd


try:
    ruta_script=pathlib.Path(__file__).resolve().parent
except:
    ruta_script=pathlib.Path.cwd()



# %%
# en: I leave a basic variable for the default analysis type
# es: Dejo una variable básica para el tipo de análisis por defecto
# ca: Deixo una variable bàsica pel tipus d'anàlisi per defecte


basicAnalisys = False

# en: set to True for testing
# es: ponemos a True para test
# ca: posem a True per a test


# basicAnalisys = True

# en: add global variables resulting from converting parts to modules
# es: añadimos variables globales surgidas de convertir a módulos las partes
# ca: afegim variables globals sorgides de convertir a mòduls les parts


df = train_df= test_df = feature_cols = target_vars = None
dfAval = models = metrics_table = models_and_params = analysisTime = None
dfBloc_models = dfBloc_full = dfBloc_NUTS2 = None
dfBloc_NUTS1 = dfBloc_NUTS0 = dfBloc_total = None


# %% [markdown]
# <!-- en: --><b> Prepare smape function to calculate the SMAPE (Symmetric Mean Absolute Percentage Error) for each target.  </b></br>
# <!-- es: --><i> Preparo función smape para calcular el SMAPE (Symmetric Mean Absolute Percentage Error) para cada target.  </i></br>
# <!-- ca: --><i> Preparo funció smape per calcular el SMAPE (Symmetric Mean Absolute Percentage Error) per a cada target.  </i></br></br>
# 
# $$
# SMAPE = 100 \times \frac{1}{n}\sum_{i=1}^{n}
# \frac{|y_i - \hat{y}_i|}{(|y_i| + |\hat{y}_i|)/2}
# $$

# %%
def smapeMVM(y_true, y_pred):
    return 100 * np.mean(np.abs(y_true - y_pred) / ((np.abs(y_true) + np.abs(y_pred)) / 2))

# %%
def phase_one():
    # en: Read the base data
    # es: Leemos los datos base
    # ca: Llegim les dades base


    global df, feature_cols, target_vars, train_df, test_df
    df = pd.read_csv('dades_to_train_test.csv')
    
    # en: Define features and targets
    # es: Definimos features y targets
    # ca: Definim features i targets


    feature_cols = [col for col in df.columns if col not in ['year','C','G_H', 'I', 'J', 'K']]
    target_vars = ['C', 'G_H', 'I', 'J', 'K']

    # en: Sort by year
    # es: Ordeno por año
    # ca: Ordeno per any


    df.sort_values(by='year', inplace=True)
    
    # en: Obtain the training and test sets and the maximum year
    # es: Obtengo los conjuntos de entrenamiento y test y el año máximo
    # ca: Obtinc els conjunts d'entrenament i test i l'any màxim


    max_df_year = df['year'].max()
    train_df = df[df['year'] < max_df_year]
    test_df = df[df['year'] == max_df_year]


# %% [markdown]
# Aquí entrenem el model amb totes les possiblitats.

# %%
def phase_two():
    # en: We prepare a set where all results are saved to be able to make a dataframe.
    # es: Prepararemos un conjunto donde se guarden todos los resultados para poder hacer un dataframe.
    # ca: Prepararem un conjunt on es guardin tots els resultats per poder fer un dataframe.


    lsencer=[]

    # en: We move X_train and X_test here because they do not change
    # es: Movemos X_train y X_test aquí porque no cambian
    # ca: Movem X_train i X_test aquí perquè no canvien


    X_train = train_df[feature_cols]
    X_test = test_df[feature_cols]

    # en: New table to evaluate the results, start by passing X_test with centroids
    # es: Nueva tabla para evaluar los resultados, empezamos pasando X_test con los centroides
    # ca: Nova taula per avaluar els resultats, comencem passant X_test amb els centroides

    
    global dfAval, models, models_and_params, analysisTime
    dfAval = X_test[["centroid_x", "centroid_y"]].copy()

    completeModels = {
        'Linear Regression':{
            'estimador':LinearRegression(),
            'param_grid':{'fit_intercept': [True, False]}
            },
        'Random Forest':{
            'estimador':RandomForestRegressor(random_state=2025, n_jobs=-1),
            'param_grid':{'n_estimators': [200, 400],
                        'max_depth': [None, 10, 20, 30],
                        'min_samples_split': [2, 5]}
            },
        'Gradient Boosting':{
            'estimador':GradientBoostingRegressor(random_state=2025),
            'param_grid':{'n_estimators': [100, 200, 300],
                        'learning_rate': [0.02, 0.1],
                        'max_depth': [3, 5, 7]}
            },
        'XGBoost':{
            'estimador':XGBRegressor(random_state=2025, n_jobs=-1),
            'param_grid':{'n_estimators': [100, 200, 300],
                        'learning_rate': [0.02, 0.1],
                        'max_depth': [3, 5, 7],
                        'subsample': [0.8, 1.0]}
                        
        } ,
        'LightGBM':{
            'estimador':LGBMRegressor(random_state=2025, n_jobs=-1, force_col_wise=True, verbose=-1),
            'param_grid':{'n_estimators': [200, 300, 400],
                        'learning_rate': [0.02, 0.1],
                        'num_leaves': [31, 42, 63]}
        }
    }
    # en: create a new smaller models for testing
    # es: hago un nuevo models más pequeño para hacer pruebas
    # ca: faig un nou models més petit per fer proves


    basicModels = {
        'Linear Regression':{
            'estimador':LinearRegression(n_jobs=-1),
            'param_grid':{'fit_intercept': [True, False]}
            },
        'Random Forest':{
            'estimador':RandomForestRegressor(random_state=2025, n_jobs=-1),
            'param_grid':{'n_estimators': [200],
                        'max_depth': [None, 10],
                        'min_samples_split': [5]}
            },
        'Gradient Boosting':{
            'estimador':GradientBoostingRegressor(random_state=2025),
            'param_grid':{'n_estimators': [100],
                        'learning_rate': [0.1],
                        'max_depth': [3]}
            },
        'XGBoost':{
            'estimador':XGBRegressor(random_state=2025, n_jobs=-1),
            'param_grid':{'n_estimators': [100],
                        'learning_rate': [0.1],
                        'max_depth': [ 5],
                        'subsample': [1.0]}
                        
        } ,
        'LightGBM':{
            'estimador':LGBMRegressor(random_state=2025,n_jobs=-1,force_col_wise=True, verbose=-1),
            'param_grid':{'n_estimators': [200],
                        'learning_rate': [0.1],
                        'num_leaves': [31]}
        }
    }

    # Nou versió 07
    models = basicModels if basicAnalisys else completeModels

    # per guardar els models i els parametres
    models_and_params = {}

    # Comencem a controlar temps de l'anaàlisi (versió 07)
    print("Analysis's start... \\ Inicio del análisis... \\ Inici de l\'anàlisi...")
    analysisTime=time.time()

    for camps in target_vars:
        y_train = train_df[camps]
        y_test = test_df[camps]

        # en: Add y_test to dfAval
        # es: Añadimos y_test a dfAval
        # ca: Afegim y_test a dfAval


        dfAval[camps] = y_test


        # Store results
        results = {}
        models_and_params[camps] = {}

        for name, model in models.items():

            estimador = model['estimador']
            param_grid = model['param_grid']

            # en: we find the optimal hyperparameters (verbose can be =1, leaving it)
            # es: encontramos los hiperparámetros óptimos (el verbose puede ser =1, lo dejo)
            # ca: trobem els hiperparàmetres òptims (el verbose pot ser =1, i deixo)


            # en: remove n_jobs from here
            # es: quito de aquí el n_jobs
            # ca: trec d'aqui el n_jobs


            gs=GridSearchCV(estimator=estimador, param_grid=param_grid, cv=5, scoring='neg_mean_squared_error', verbose=0)

            gs.fit(X_train, y_train)
            best_model=gs.best_estimator_
            preds=best_model.predict(X_test)

            models_and_params[camps][name]={}
            models_and_params[camps][name]["params"]= gs.best_params_
            models_and_params[camps][name]["best_model"]=best_model

            dfAval[camps+"_"+name] = preds
            
            mae = mean_absolute_error(y_test, preds)
            rmse = np.sqrt(mean_squared_error(y_test, preds))   
            r2 = r2_score(y_test, preds)
            smape = smapeMVM(y_test, preds)

            results[name] = {'MAE': mae, 'RMSE': rmse, 'R2':r2, 'SMAPE':smape}

        msencer=pd.DataFrame(results).unstack(level=1).reset_index().set_axis(["Model","Metric","Coeficient"],axis=1)
        msencer["ICD-10"]=camps
        lsencer.append(msencer)

    analysisTime = time.time() - analysisTime
    anMinutes = int(analysisTime // 60)
    anSeconds = analysisTime % 60
    analysisTime = f"{anMinutes} minutes and {anSeconds:.2f} seconds"
    print(f"...analysis\'s end \\ ...fin del análisis \\ ...fi de l\'anàlisi, total time: {analysisTime}")

# %% [markdown]
# Un cop generats els models, fem la valoració per cada diagnòstic.

# %%
def ponderacio(dades):
    en_models = ['Linear Regression', 'Random Forest', 'Gradient Boosting', 'XGBoost', 'LightGBM']
    en_diagnostics = ['C', 'G_H', 'I', 'J', 'K']
    en_metriques = ["RMSE", "MAE", "R2", "SMAPE"]
    fun_metriques = {"RMSE": lambda diagnostic, model:mean_absolute_error(diagnostic, model),
                      "MAE": lambda diagnostic, model:root_mean_squared_error(diagnostic, model),
                      "R2": lambda diagnostic, model: r2_score(diagnostic, model),
                      "SMAPE": lambda diagnostic, model: smapeMVM(diagnostic, model)}
    normaliza=MinMaxScaler()
    causes_table=[]
    for diagnostic in en_diagnostics:
        taula_scores={"Model":en_models}
        for metrica in en_metriques:
            llista_metr=[]
            for model in en_models:
                llista_metr.append(fun_metriques[metrica](dades[diagnostic],dades[diagnostic+"_"+model]))
            taula_scores[metrica]=llista_metr
        dfCause=pd.DataFrame(taula_scores)
        # en: normalize, on the one hand the KeyError metrics
        # es: normalizamos, por un lado las métricas de KeyError
        # ca: normalitzem, per una banda les mètriques d'KeyError


        merrors = ["RMSE", "MAE", "SMAPE"]
        for x in merrors:
            # en: we do them subtracting from 1 to scale with r2
            # es: las hacemos restando sobre 1 para escalar con r2
            # ca: les fem restant sobre 1 per escalar amb r2


            dfCause[[per_merror+"_norm" for per_merror in merrors]] = 1 - normaliza.fit_transform(dfCause[merrors])
        
        # en: now for R2, which is goodness of fit
        # es: ahora la de R2, que es de bondad
        # ca: ara la de R2, que es de bondat


        dfCause["R2_norm"] = normaliza.fit_transform(dfCause[["R2"]])
        
        # en: Choose the weights
        # es: Escojo los pesos
        # ca: Escullo els pesos


        pesos = {"MAE_norm": 0.3, "RMSE_norm": 0.3, "R2_norm": 0.25, "SMAPE_norm": 0.15}
        
        # en: Calculate the weighted score
        # es: Calculo el score ponderado
        # ca: Calculo el score ponderat


        dfCause["Score_ponderat"] = sum([dfCause[x]*y for x,y in pesos.items()])
        millor_score= dfCause["Score_ponderat"].max()
        dfCause["Best_score"] = [True if x==millor_score else False for x in dfCause["Score_ponderat"]]
        
        # en: Add the diagnosis
        # es: Añado el diagnóstico
        # ca: Afegeixo el diagnòstic


        dfCause["ICD-10"]=diagnostic
        causes_table.append(dfCause)

    # en: Combine the tables already before returning
    # es: Combino ya las tablas antes de retornar
    # ca: Combino ja les taules abans retornar

    
    return pd.concat(causes_table,axis=0,ignore_index=True)      
                                   


# %% [markdown]
# Posem la funció per pasar els hiperparàmetres a string

# %%
def params_a_text(cause, model):
    els_parametres=[parametre+"="+f'{valor}' for parametre, valor in models_and_params[cause][model]["params"].items()]
    return ", ".join(els_parametres)


# %%
def selecciona_models():
    # en: first I add to metrics_table the parameter field used in each model
    # es: primero añado a metrics_table el campo de parámetros usados en cada modelo
    # ca: primer afegeixo a metrics_table el camp de paràmetres fets servir en cada model


    t1=metrics_table.copy()
    t1['Parameters'] = t1.apply(lambda x: params_a_text(x["ICD-10"],x['Model']), axis=1)
    # en: finally sort the fields
    # es: ordenamos los campos finalmente
    # ca: ordenem els camps finalment


    camps_inicials = ["ICD-10", 'Model','Best_score']
    t1 = t1[camps_inicials + [c for c in t1.columns if c not in camps_inicials]]
    t1.sort_values(by=['Best_score',"ICD-10",'Model'], ascending=[False,True,True],inplace=True)
    # en: I will better return the models with parameters to use them later
    # es: devolveré mejor los modelos con parámetros para usarlos más tarde
    # ca: retornaré millor els models amb paràmetres per fer-ho servir més tard


    t2=t1[t1['Best_score']][["ICD-10",'Model','Parameters']]
    selected_models={}
    for i, registre in t2.iterrows():
        selected_models[registre["ICD-10"]]={'nom':registre['Model'],
                                               'params':models_and_params[registre["ICD-10"]][registre['Model']]['params'],
                                               'model':models_and_params[registre["ICD-10"]][registre['Model']]['best_model']}
        
    try:
        t1.to_csv(f"analisi_models.csv")
        t1.to_excel(f"analisi_models.xlsx", index=False)
        joblib.dump(selected_models, f"selected_models.pkl")
        # en: new in version 10 (save the data used in features and targets)
        # es: nuevo en la versión 10 (guardar los datos usados en features y targets)
        # ca: nou a la versió 10 (desar les dades fetes servir en features i targets)

        # versió amb dades per explicabilitat (afegim de moment train i test de features)
        
        dadesModel =(df[feature_cols].copy(), df[target_vars].copy(),
                     train_df[feature_cols].copy(), test_df[feature_cols].copy(),
                     train_df[target_vars].copy(), test_df[target_vars].copy())
        joblib.dump(dadesModel, f"dadesModel.pkl",compress=True)
        print("Models and data saved \\ Guardados modelos y datos \\ Desats els models i les dades")
    
    except Exception as e:
        # en: Returns false if there is an error
        # es: Devuelve false si hay un error
        # ca: Retorna false si hi ha un error

        print(e)
        return False
    return t2


# %% [markdown]
# Comencem creant els dataframes  
# _aillem els predits que siguin per cada tipus el seu best model_

# %%
# Comencem amb un bloc total

def bloc_full():
    millors_metriques = ponderacio(dfAval)
    millors_metriques = millors_metriques[millors_metriques["Best_score"]==True].copy()
    # en: variables by cause of death
    # es: variables por causa de muerte
    # ca: variables per causa de mort


    in_causes = ['C', 'G_H', 'I', 'J', 'K']
    # ca: ara afegirem els centroides
    # en: now we add the centroids
    # es: ahora añadiremos los centroides
    
    in_causes.extend(['centroid_x','centroid_y'])

    # en: we keep these dfAval3 columns in dfGlob
    # es: nos quedamos en dfGlob estas columnas de dfAval3
    # ca: ens quedem en dfGlob aquestes columnes de dfAval3


    dfGlob = dfAval[in_causes].copy()
    for indi, regis in millors_metriques.iterrows():
        dfGlob[regis["ICD-10"]+"_pred"]=dfAval[regis["ICD-10"]+"_"+regis['Model']].copy()
    
    # en: now we concatenate causes_ref rendering a merge, this code is recovered from above
    # es: ahora concatenamos causes_ref haciendo un merge, este código está recuperado de arriba
    # ca: ara concatenem causes_ref fent un merge, aquest codi està recuperat d'amunt


    causes_ref=pd.read_csv(f"mv_nuts2_xy.csv")


    dfGlob=dfGlob.merge(causes_ref[["NUTS_ID", "NUTS_NAME", "NUTS1_ID", "NUTS1_NAME",
                                        "NUTS0_ID","NUTS0_NAME","centroid_x","centroid_y"]], how="left", on= ["centroid_x","centroid_y"]).copy()
    # ca: comprovem buits
    # en: check for missing values
    # es: comprobamos valores faltantes

    if dfGlob.isna().any().any():
        raise Exception("Error unint els valors del conjunt")

    # ca: eliminem els centroides, que ja no ens fan falta
    # en: remove the centroids, which we no longer need
    # es: eliminamos los centroides, que ya no necesitamos

    dfGlob.drop(["centroid_x","centroid_y"], axis=1, inplace=True)

    dfGlob.reset_index(inplace=True)


    return dfGlob


# %% [markdown]
# _PREPAREM LES MÈTRIQUES_

# %%
def calcula_metriques(perBloc, complert=False, nuts0=False):

    in_causes = ['C', 'G_H', 'I', 'J', 'K']

    # ca: creo diccionari per tornar cada camp
    # en: create dictionary to return each field
    # es: creo diccionario para devolver cada campo

    res_met={}
    for cada in in_causes:
        res_met[cada] = sum(perBloc[cada])
        res_met[cada+"_pred"] = np.around(sum(perBloc[cada+"_pred"]),2)

        if complert or nuts0:
            res_met[cada +"_MAE"] = np.around(mean_absolute_error(perBloc[cada], perBloc[cada+"_pred"]),2)
            res_met[cada +"_RMSE"] = np.around(root_mean_squared_error(perBloc[cada], perBloc[cada+"_pred"]),2)
        if complert:
            res_met[cada +"_R2"] = np.around(r2_score(perBloc[cada], perBloc[cada+"_pred"]),2)
        res_met[cada +"_SMAPE"] = np.around(smapeMVM(perBloc[cada], perBloc[cada+"_pred"]),2)

    # ca: Retorno les sèries del diccionari
    # en: Return the series from the dictionary
    # es: Retorno las series del diccionario

    return pd.Series(res_met)


# %% [markdown]
# <!-- en: --><b> once we have the complete block, we start with NUTS2</b></br>
# <!-- es: --><i> una vez tenemos el bloque completo, empezamos con los NUTS2</i></br>
# <!-- ca: --><i> un cop tenim el bloc complet, comencem amb els NUTS2</i></br></br>
# 

# %%
def bloc_NUTS2():
    in_causes = ['C', 'G_H', 'I', 'J', 'K']
    columnes = ['NUTS_NAME','NUTS_ID']
    columnes.extend([cada for cause in in_causes for cada in (cause, cause+"_pred")])
    dfAbloc=dfBloc_full[columnes].copy()
    return dfAbloc.groupby(['NUTS_NAME','NUTS_ID']).apply(calcula_metriques, include_groups=False).reset_index()



# %% [markdown]
# <!-- en: --><b> Now, the NUTS1 level</b></br>
# <!-- es: --><i> Ahora, el nivel NUTS1</i></br>
# <!-- ca: --><i> Ara, el nivell NUTS1</i></br></br>

# %%
def bloc_NUTS1():
    in_causes = ['C', 'G_H', 'I', 'J', 'K']
    columnes = ['NUTS1_NAME', 'NUTS1_ID']
    columnes.extend([cada for cause in in_causes for cada in (cause, cause+"_pred")])
    dfAbloc=dfBloc_full[columnes].copy()
    return dfAbloc.groupby(['NUTS1_NAME', 'NUTS1_ID']).apply(calcula_metriques, include_groups=False).reset_index()

# %% [markdown]
# <!-- en: --><b> Finally the NUTS0 level (Countries)</b></br>
# <!-- es: --><i> Finalmente el nivel NUTS0 (Países)</i></br>
# <!-- ca: --><i> Finalment el nivell NUTS0 (Països)</i></br></br>
# 

# %%
def bloc_NUTS0():
    in_causes = ['C', 'G_H', 'I', 'J', 'K']
    columnes = ['NUTS0_NAME', 'NUTS0_ID']
    columnes.extend([cada for cause in in_causes for cada in (cause, cause+"_pred")])
    dfAbloc=dfBloc_full[columnes].copy()

    # en: There are countries without sectorizations, so R^2 cannot be calculated
    # es: Hay países sin sectorizaciones, por lo tanto no se puede hacer el R^2
    # ca: Hi ha països sense sectoritzacions, per tant no es pot fer el R^2


    return dfAbloc.groupby(['NUTS0_NAME', 'NUTS0_ID']).apply(calcula_metriques, nuts0=True, include_groups=False).reset_index()


# %% [markdown]
# Ara fem el global

# %%
def bloc_total():
    in_causes = ['C', 'G_H', 'I', 'J', 'K']
    columnes = ['NUTS1_NAME', 'NUTS1_ID']
    columnes.extend([cada for cause in in_causes for cada in (cause, cause+"_pred")])

    dfAbloc=dfBloc_full[columnes].copy()

    dfAbloc = []
    for cada in in_causes:
        registres={"ICD-10": cada}
        registres["Deaths" ]=sum(dfBloc_full[cada])
        registres["Prediction" ]=np.around(sum(dfBloc_full[cada+"_pred"]),2)
        registres["Error"] = round(registres["Prediction"]-registres["Deaths"],2)
        registres["MAE"]=np.around(mean_absolute_error(dfBloc_full[cada], dfBloc_full[cada+"_pred"]),2)
        registres["RMSE"] = np.around(root_mean_squared_error(dfBloc_full[cada], dfBloc_full[cada+"_pred"]),2)
        registres["R2"] = np.around(r2_score(dfBloc_full[cada], dfBloc_full[cada+"_pred"]),2)
        registres["SMAPE"] = np.around(smapeMVM(dfBloc_full[cada], dfBloc_full[cada+"_pred"]),2)
        dfAbloc.append(registres)
    dfAbloc = pd.DataFrame(dfAbloc)


    return pd.DataFrame(dfAbloc)


# %% [markdown]
# **Comencem a fer l'informe Excel**

# %%
# ca: Per Calcular els pitjors SMAPE i mostrar-los
# en: To calculate the worst SMAPE and display them
# es: Para calcular los peores SMAPE y mostrarlos

def pitjors_SMAPE(eldataframe):
    dfTemp = eldataframe.copy()
    nomCol = dfTemp.columns[0]
    in_causes = ['C', 'G_H', 'I', 'J', 'K']
    dfAbloc = []
    for cada in in_causes:
        elPitjor = dfTemp[dfTemp[cada+"_SMAPE"]==dfTemp[cada+"_SMAPE"].max()].iloc[0]
        registres ={"Worst SMAPE by type": cada}
        registres[nomCol] = elPitjor[nomCol]
        registres["Valor (%)"] = elPitjor[cada+"_SMAPE"]
        dfAbloc.append(registres)

    return pd.DataFrame(dfAbloc)



# %%

def genera_informe():
    # en: Save today's date
    # es: Guardo la fecha de hoy
    # ca: Guardo data d'avui

    
    today= date.today().strftime("%Y%m%d")
    # today = str(date.today())

    # en: Generate the file name we will give to the report
    # es: Genero el nombre del archivo que le daremos al informe
    # ca: Genero el nom del fitxer que li donarem a l'informe


    nom_analisi=f"informe_analisi_modelitzacio_"+today+".xlsx"

    # en: To label the metrics
    # es: Para etiquetar las métricas
    # ca: Per etiquetar les mètriques


    eti_metriques = {
        "MAE": "Mean Absolute Error",
        "RMSE": "Root Mean Squared Error",
        "R2": "Coefficient of Determination",
        "SMAPE": "Symmetric Mean Absolute Percentage Error (0-200%)"
    }

    eti_ICD_10 ={
        "C": "Malignant neoplasms (C00-C97)",
        "G-H": "Diseases of the nervous system and the sense organs (G00-H95)",
        "I": "Diseases of the circulatory system (I00-I99)",
        "J": "Diseases of the respiratory system (J00-J99)",
        "K": "Diseases of the digestive system (K00-K93)"
    }


    # en: We create the ExcelWriter object with 'typical' structure
    # es: Creamos el objeto ExcelWriter con estructura 'típica'
    # ca: Creem l'objecte ExcelWriter amb estructura 'típica'


    with pd.ExcelWriter(nom_analisi, engine="xlsxwriter") as writer:
        llibre  = writer.book
        full = llibre.add_worksheet("Summary")
        writer.sheets["Summary"] = full

        # en: Add the formats to use
        # es: Añado los formatos para usar
        # ca: Afegeixo els formats per fer servir


        format_titol_full = llibre.add_format({"bold": True, "font_size": 20, "indent": 5, "valign": "vcenter"})
        format_titol_full_trans = llibre.add_format({"font_size": 18, "indent": 5, "valign": "vcenter", "bg_color": "white"})

        format_titol_seccio = llibre.add_format({"bold": True, "font_size": 16, "valign": "vcenter", "underline": True})
        format_titol_seccio_trans = llibre.add_format({"font_size": 14, "valign": "vcenter"})

        format_capceleres = llibre.add_format({"bold": True, "font_size": 12, "align": "center", "valign": "vcenter",
                                            "font_color": "white", "bg_color": "black", "border": 1})

        format_resaltat = llibre.add_format({"bold": True, "font_size": 11, "valign": "vcenter",
                                            "indent": 1, "border": 1})
        format_normal = llibre.add_format({"font_size": 11, "valign": "vcenter", "text_wrap": False})
        format_comentari = llibre.add_format({"italic": True, "font_size": 10, "valign": "vcenter"})
        format_etiqueta = llibre.add_format({"bold": True, "font_size": 12, "valign": "vcenter",
                                            "indent": 1, "text_wrap": True})

        format_graella = llibre.add_format({ "font_size": 11, "valign": "vcenter", "text_wrap": True, "border": 1})
        format_wrap = llibre.add_format({"font_size": 11, "valign": "vcenter", "text_wrap": True})

        # en: I put row and column pointers to control position, (I will do it on each sheet)
        # es: Pongo punteros de fila y columna para controlar la posición, (lo haré en cada hoja)
        # ca: Poso punters de fila i columna per controlar la posició, (ho faré en cada full)


        fila, columna = 0, 0

        # en: I change the width of the first column since descriptives will go there
        # es: Cambio el ancho de la primera columna ya que irán descriptivos
        # ca: Canvio l'amplada de la primera columna ja que aniran descriptius


        full.set_column(0, 0, 30)
        
        # en: Finish the first sheet changing the width of the 3 columns where the table is
        # es: Termino la primera hoja cambiando el ancho de las 3 columns donde está la tabla
        # ca: Acabo el primer full canviant l'ample de les 3 columnes on està la taula


        full.set_column(columna+1, columna+10, 18, format_wrap)

        # en: Add the title
        # es: Añado el título
        # ca: Afegeixo el titol

        
        titolGeneral = "Report of the optimized predictive model and results obtained"
        titolGeneral_es = "Informe del modelo predictivo optimizado y resultados obtenidos"
        titolGeneral_ca = "Informe del model predictiu optimitzat i resultats obtinguts"
        full.write(fila, columna, titolGeneral, format_titol_full)
        fila += 1
        full.write(fila, columna, titolGeneral_es, format_titol_full_trans)
        fila += 1
        full.write(fila, columna, titolGeneral_ca, format_titol_full_trans)
        fila += 3

        full.write(fila, columna, "Generation date: ", format_etiqueta)
        full.write(fila, columna + 1, str(date.today()), format_normal)
        fila += 1
        full.write(fila, columna, "Time spent on analysis: ", format_etiqueta)
        full.write(fila, columna + 1, analysisTime, format_normal)
        fila += 1
        full.write(fila, columna, "Models evaluated: ", format_etiqueta)
        full.write(fila, columna + 1, ", ".join( models.keys()), format_normal)
        fila += 1
        full.write(fila, columna +1, "Each with optimized hyperparameters ", format_comentari)
        fila += 1
        full.write(fila, columna, "NUTS2 (basic regions) evalutated: ", format_etiqueta)
        full.write(fila, columna + 1, str(dfBloc_NUTS2.shape[0]), format_resaltat)
        fila += 1
        full.write(fila, columna +1, "It may vary from the actual number, depending on the data cross-tabulation. ", format_comentari)
        fila += 2

        filatemp = fila
        full.write(fila, columna, "Metrics used to evaluate results ", format_titol_seccio)
        fila += 1
        full.write(fila, columna, "Métricas utilizadas para evaluar los resultados ", format_titol_seccio_trans)
        fila += 1
        full.write(fila, columna, "Mètriques utilitzades per avaluar els resultats ", format_titol_seccio_trans)
        fila += 2
        for nom, descrip in eti_metriques.items():
            full.write(fila, columna, nom, format_etiqueta)
            full.write(fila, columna + 1, descrip, format_normal)
            fila += 1
        fila += 1

        fila = filatemp
        columna = 4
        full.write(fila, columna, "ICD-10 description of the groups analyzed", format_titol_seccio)
        fila += 1
        full.write(fila, columna, "Descripción CIM-10 de los grupos analizados", format_titol_seccio_trans)
        fila += 1
        full.write(fila, columna, "Descripció CIM-10 dels grups analitzats", format_titol_seccio_trans)
        fila += 2
        for nom, descrip in eti_ICD_10.items():
            full.write(fila, columna, nom, format_etiqueta)
            full.write(fila, columna + 1, descrip, format_normal)
            fila += 1
        fila += 1

        columna = 0
        
        full.write(fila, columna, "Total results by type of death ", format_titol_seccio)
        fila += 1
        full.write(fila, columna, "Resultados totales por tipos de muerte ", format_titol_seccio_trans)
        fila += 1
        full.write(fila, columna, "Resultats totals per tipus de mort ", format_titol_seccio_trans)
        fila += 2
        
        # en: to write the dataframes, I will save records and columns before
        # es: para escribir los dataframes, guardaré antes registros y columnas
        # ca: per escriure els dataframes, guardarè abans registres i columnes


        registres, columnes = dfBloc_total.shape
        
        # en: place the table
        # es: ponemos la tabla
        # ca: posem la taula


        dfBloc_total.to_excel(writer, sheet_name="Summary", startrow=fila, startcol=columna+1, index=False, header=True)
        
        # en: apply formatting to rows, starting with headers
        # es: aplico el formato a las filas, empiezo por la cabecera
        # ca: aplico el format a les files, començo per la capcelera


        full.conditional_format(fila, columna+1, fila, columna+columnes,
                                {"type": "no_errors", "format":format_capceleres})
        # en: now, to records
        # es: ahora, a los registros
        # ca: ara, als registres


        full.conditional_format(fila+1, columna+2, fila+registres, columna+columnes,
                                    {"type": "no_errors", "format":format_graella})
        # en: and now, to columns
        # es: y ahora, las columnas
        # ca: i ara, la columnes


        full.conditional_format(fila+1, columna+1, fila+registres, columna+1,
                                {"type": "no_errors", "format":format_resaltat})
        fila +=registres+2
        
        
        # en: Replicate for the next table
        # es: Replico para la siguiente tabla
        # ca: Replico per la següent tabla


        full.write(fila, columna, "Optimal model applied by type of death", format_titol_seccio)
        fila += 1
        full.write(fila, columna, "Model óptimo aplicado por tipo de muerte", format_titol_seccio_trans)
        fila += 1
        full.write(fila, columna, "Model òptim aplicat per tipus de mort", format_titol_seccio_trans)
        fila += 2
        registres, columnes = dfBloc_models.shape
        dfBloc_models.to_excel(writer, sheet_name="Summary", startrow=fila, startcol=columna+1, index=False, header=True)
        full.conditional_format(fila, columna+1, fila, columna+columnes,
                                {"type": "no_errors", "format":format_capceleres})
        full.conditional_format(fila+1, columna+2, fila+registres, columna+columnes,
                                    {"type": "no_errors", "format":format_graella})
        full.conditional_format(fila+1, columna+1, fila+registres, columna+1,
                                {"type": "no_errors", "format":format_resaltat})

        # en: WE GO FOR THE SECOND SHEET, NUTS 0 LEVEL
        # es: VAMOS A LA SEGUNDA HOJA, NIVEL NUTS 0
        # ca: ANEM PEL SEGON FULL, NIVELL NUTS 0

        
        full2 = llibre.add_worksheet("NUTS 0")
        writer.sheets["NUTS 0"] = full2
        fila, columna = 0, 0
        registres, columnes = dfBloc_NUTS0.shape
        full2.set_column(columna, columna, 30)
        full2.set_column(columna+1, columna+columnes-1, 18, format_wrap)
        
        # en: Add the title
        # es: Añado el título
        # ca: Afegeixo el titol


        titolGeneral = "NUTS level 0 results"
        full2.write(fila, columna, titolGeneral, format_titol_full)
        fila += 1
        full2.write(fila, columna +1, " At this level of aggregation (NUTS0), although the original work also included the R² metric,", format_normal)
        fila += 1
        full2.write(fila, columna +1, " it has been excluded in this adaptation due to the nature of the test data,", format_normal)
        fila += 1
        full2.write(fila, columna +1, " where some territorial units have only a single observation.", format_normal)
        fila += 3
        
        # en: Add the visualization of the SMAPE metric
        # es: Añado la visualización de la métrica SMAPE
        # ca: Afegeixo la visualització de la mètrica SMAPE


        dfPitjorSMAPE = pitjors_SMAPE(dfBloc_NUTS0)
        registres, columnes =dfPitjorSMAPE.shape
        dfPitjorSMAPE.to_excel(writer, sheet_name="NUTS 0", startrow=fila, startcol=columna, index=False, header=True)
        full2.conditional_format(fila, columna, fila, columna+columnes-1,
                                {"type": "no_errors", "format":format_capceleres})
        full2.conditional_format(fila+1, columna+1, fila+registres, columna+columnes-1,
                                    {"type": "no_errors", "format":format_graella})
        full2.conditional_format(fila+1, columna, fila+registres, columna,
                                {"type": "no_errors", "format":format_resaltat})
        fila +=registres+3

        registres, columnes = dfBloc_NUTS0.shape
        dfBloc_NUTS0.to_excel(writer, sheet_name="NUTS 0", startrow=fila, startcol=columna, index=False, header=True)
        full2.conditional_format(fila, columna, fila, columna+columnes-1,
                                {"type": "no_errors", "format":format_capceleres})
        full2.conditional_format(fila+1, columna+1, fila+registres, columna+columnes-1,
                                    {"type": "no_errors", "format":format_graella})
        full2.conditional_format(fila+1, columna, fila+registres, columna,
                                {"type": "no_errors", "format":format_resaltat})


        # en: WE GO FOR THE THIRD SHEET, NUTS 1 LEVEL
        # es: VAMOS A LA TERCERA HOJA, NIVEL NUTS 1
        # ca: ANEM PEL TERCER FULL, NIVELL NUTS 1


        full3 = llibre.add_worksheet("NUTS 1")
        writer.sheets["NUTS 1"] = full3
        fila, columna = 0, 0
        registres, columnes = dfBloc_NUTS1.shape
        full3.set_column(columna, columna, 48)
        full3.set_column(columna+1, columna+columnes-1, 18, format_wrap)
        
        # en: Add the title
        # es: Añado el título
        # ca: Afegeixo el titol


        titolGeneral = "Nuts level 1 results"
        full3.write(fila, columna, titolGeneral, format_titol_full)
        fila += 1
        full3.write(fila, columna +1, " We only apply the SMAPE metric, as it\'s a level at risk of only having one result in the aggregation,", format_normal)
        fila += 1
        full3.write(fila, columna +1, " which is why the rest of the metrics might return an error when evaluating them.", format_normal)
        fila += 3
        
        # en: Add the visualization of the SMAPE metric
        # es: Añado la visualización de la métrica SMAPE
        # ca: Afegeixo la visualització de la mètrica SMAPE


        dfPitjorSMAPE = pitjors_SMAPE(dfBloc_NUTS1)
        registres, columnes =dfPitjorSMAPE.shape
        dfPitjorSMAPE.to_excel(writer, sheet_name="NUTS 1", startrow=fila, startcol=columna, index=False, header=True)
        full3.conditional_format(fila, columna, fila, columna+columnes-1,
                                {"type": "no_errors", "format":format_capceleres})
        full3.conditional_format(fila+1, columna+1, fila+registres, columna+columnes-1,
                                    {"type": "no_errors", "format":format_graella})
        full3.conditional_format(fila+1, columna, fila+registres, columna,
                                {"type": "no_errors", "format":format_resaltat})
        fila +=registres+3

        registres, columnes = dfBloc_NUTS1.shape
        dfBloc_NUTS1.to_excel(writer, sheet_name="NUTS 1", startrow=fila, startcol=columna, index=False, header=True)
        full3.conditional_format(fila, columna, fila, columna+columnes-1,
                                {"type": "no_errors", "format":format_capceleres})
        full3.conditional_format(fila+1, columna+1, fila+registres, columna+columnes-1,
                                    {"type": "no_errors", "format":format_graella})
        full3.conditional_format(fila+1, columna, fila+registres, columna,
                                {"type": "no_errors", "format":format_resaltat})



        # en: Finally NUTS 2 level
        # es: Finalmente el nivel NUTS 2
        # ca: Finalment el nivell NUTS 2


        full5 = llibre.add_worksheet("NUTS 2")
        writer.sheets["NUTS 2"] = full5
        fila, columna = 0, 0
        registres, columnes = dfBloc_NUTS2.shape
        full5.set_column(columna, columna, 48)
        full5.set_column(columna+1, columna+columnes-1, 18, format_wrap)
        
        # en: Add the title
        # es: Añado el título
        # ca: Afegeixo el titol

        
        titolGeneral = "Nuts level 2 results"
        full5.write(fila, columna, titolGeneral, format_titol_full)
        fila += 1
        full5.write(fila, columna +1, " We only apply the SMAPE metric, as it's a level at risk of only having one result in the aggregation,", format_normal)
        fila += 1
        full5.write(fila, columna +1, " which is why the rest of the metrics might return an error when evaluating them.", format_normal)
        fila += 3

        dfPitjorSMAPE = pitjors_SMAPE(dfBloc_NUTS2)
        registres, columnes =dfPitjorSMAPE.shape
        dfPitjorSMAPE.to_excel(writer, sheet_name="NUTS 2", startrow=fila, startcol=columna, index=False, header=True)
        full5.conditional_format(fila, columna, fila, columna+columnes-1,
                                {"type": "no_errors", "format":format_capceleres})
        full5.conditional_format(fila+1, columna+1, fila+registres, columna+columnes-1,
                                    {"type": "no_errors", "format":format_graella})
        full5.conditional_format(fila+1, columna, fila+registres, columna,
                                {"type": "no_errors", "format":format_resaltat})
        fila +=registres+3


        registres, columnes = dfBloc_NUTS2.shape
        dfBloc_NUTS2.to_excel(writer, sheet_name="NUTS 2", startrow=fila, startcol=columna, index=False, header=True)
        full5.conditional_format(fila, columna, fila, columna+columnes-1,
                                {"type": "no_errors", "format":format_capceleres})
        full5.conditional_format(fila+1, columna+1, fila+registres, columna+columnes-1,
                                    {"type": "no_errors", "format":format_graella})
        full5.conditional_format(fila+1, columna, fila+registres, columna,
                                {"type": "no_errors", "format":format_resaltat})



# %% [markdown]
# Creem una funció principal (com una 'main()')

# %%
def principal(**arguments):
    # en: pass the global variables
    # es: pasamos las variables globales
    # ca: passem les variables globals


    global basicAnalisys, metrics_table, dfBloc_models, dfBloc_full
    global dfBloc_NUTS2, dfBloc_NUTS1, dfBloc_NUTS0, dfBloc_total
    
    # en: check if the files used exist
    # es: comprobamos si existen los archivos utilizados
    # ca: comprovem si existeixen els fitxers que s'utilitzen


    fitxers=["mv_nuts2_xy.csv", "dades_to_train_test.csv"]
    # en: then, the files
    # es: después, los archivos
    # ca: després, els fitxers


    for fitxer in fitxers:
        if not os.path.exists(f"{fitxer}"):
            raise Exception(f"No existeix el fitxer: {fitxer}")
        
    # en: change the analysis type if needed, otherwise leave it
    # es: cambiamos el tipo de análisis si es necesario, si no, dejamos el que está
    # ca: canviem el tipus d'anàlisi si cal, si no, deixem el que està


    basicAnalisys= arguments.get("basic", basicAnalisys)

    # en: only for testing in notebook
    # es: solo para testing en notebook
    # ca: sols per test en notebook

    # basicAnalisys = True

    # en: read first files and prepare groups
    # es: leemos primeros archivos y preparamos grupos
    # ca: llegim primers fitxers i preparem grups


    phase_one()
    # en: we model
    # es: modelizamos
    # ca: modelitzem


    phase_two()
    metrics_table = ponderacio(dfAval)
    
    # en: do each block
    # es: hacemos cada bloque
    # ca: fem cada bloc


    dfBloc_models = selecciona_models()
    dfBloc_full = bloc_full()
    dfBloc_NUTS2=bloc_NUTS2()
    dfBloc_NUTS1=bloc_NUTS1()
    dfBloc_NUTS0=bloc_NUTS0()
    dfBloc_total=bloc_total()
    
    # en: finally, generate the report
    # es: finalmente, generamos el informe
    # ca: finalment, generem l'informe


    genera_informe()



def retorna_arguments():
    arguments = argparse.ArgumentParser(description="""Predictive model generator
/ (es) Generador de modelo predictor
/ (ca) Generador de model predictor""")
    arguments.add_argument("-b","--basic", action="store_true", help="Do a basic analysis / (es) Haz un análisis básico / (ca) Fes un anàlisi bàsic")
    # en: we pass them as known in case we are running the notebook
    # es: los pasamos como known por si estamos ejecutando el notebook
    # ca: els passem com a known per si estem executant el notebook


    return vars(arguments.parse_known_args()[0])

if __name__ == "__main__":
    arguments = retorna_arguments()
    principal(**arguments)




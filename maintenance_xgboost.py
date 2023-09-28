""" Classe de maintenance du modele de ML avec mlflow"""

import os
import sys
from collections import Counter
from datetime import datetime, timedelta
from random import random
import logging

import pandas as pd
import numpy as np
import pickle
import hashlib

#### machine learning :

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate, train_test_split
from sklearn.metrics import confusion_matrix, f1_score, fbeta_score, make_scorer, classification_report, fbeta_score
from sklearn.svm import LinearSVC

from lightgbm import LGBMClassifier

from xgboost import XGBClassifier


from mlflow import autolog, log_metric, log_params, start_run, set_tag, set_tracking_uri
from mlflow.models import infer_signature
from mlflow.sklearn import log_model
from mlflow.xgboost import log_model
from mlflow.lightgbm import log_model
from mlflow.client import MlflowClient

from pathlib import Path


def fReadDataFrameFile(path, file, use_dict=False, encoding='utf8', sep=';'):
    """
    Fonction permettant la lecture d'un fichier au format csv encodé en utf8 contenant un DataFrame avec ou sans précision des types de données

    Args:
        path (str): path complet du fichier.
        file (str): nom du fichier sans extention, l'extention doit être csv
        use_dict (bool, optional): utilisation d'un dictionnaire pour les type des colonnes. Le dictionnaire doit avoir le même nom que le fichier avec une extention dict. Defaults to False.
        encoding (str): Encodage du fichier. Defaults to 'utf8'.
        sep (char): Séparateur de coonne. Defaults to ';'.
        

    Returns:
        DataFrame: Jeu de données
    """
    datetime_is_used = False
    list_var_datetime = []
    
    timedelta_is_used = False
    list_var_timedelta = []
    
    # Utilisation d'un dictionnaire pour préciser les types de données du fichier csv en lecture
    if use_dict:
        path_dict = os.path.join(path, f'{file}.dict')
        if os.path.exists(path_dict):
            with open(path_dict, 'r') as reader:
                liste_columns_type = eval(reader.read())
    
            list_var_datetime = []
            list_var_timedelta = []
            for k, v in liste_columns_type.items():
                if 'time' in str(v):
                    liste_columns_type[k] = 'object'
                    
                    if 'datetime' in str(v):
                        list_var_datetime = list_var_datetime + [k]
                        datetime_is_used = True
                    if 'timedelta' in str(v):
                        list_var_timedelta = list_var_timedelta + [k]
                        timedelta_is_used = True
                    
    path_file = os.path.join(path, f'{file}.csv')
    if os.path.exists(path_file):
        if use_dict:
            df = pd.read_csv(path_file, sep=sep, encoding=encoding, dtype=liste_columns_type)
            if datetime_is_used:
                for var in list_var_datetime:
                    df[var] = pd.to_datetime(df[var])
            if timedelta_is_used:
                for var in list_var_timedelta:
                    df[var] = pd.to_timedelta(df[var])
        else:
            df = pd.read_csv(path_file, sep=sep, encoding=encoding)
    
    return df


def fGetRessources(path_file):
    """Chargement et mise en cache des ressources pour ne les charger qu'une seule fois

    Args:
        path_file (str): chemin d'accès de la ressources à charger

    Returns:
        object: ressources chargées, elle peut être de différentes natures
    """    
    if os.path.exists(path_file):
        return pickle.load(open(path_file, "rb"))


def custom_scoring(y_true, y_pred, beta=None):
    """Scoring spécifique pour influer sur l'importance d'une classe d'erreur spécifique
    
    Args:
        y_true (array of int): Classification réelle
        y_pred (array of int): Classification prédite
        beta (float, optional): Paramètre permettant de réduirre ou de privilégier une classe d'erreur par rapport à une autre. Defaults to None.

    Returns:
        float: Scoring de la pertience entre les données réelles et celles prédites 
    """    
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0,1]).ravel()
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)

    if beta == None :
        c = Counter(y_true)
        if c[0] > c[1]:
            beta = c[1] > c[0]
        else:
            beta = c[0] > c[1]
        
    fb = (1 + beta**2) * (precision * recall) / ((beta**2 * precision) + recall)
    return fb


def custom_scoring_min_fn(y_true, y_pred):
    """Scoring spécifique pour reduire ne nombre de faux négatif 

    Args:
        y_true (array of int): Classification réelle
        y_pred (array of int): Classification prédite

    Returns:
        float: Scoring de la pertience entre les données réelles et celles prédites 
    """    
    # Minimiser les accords de prêt qui seront éventuellement en défaut de paiement
    return custom_scoring(y_true, y_pred, beta=2)


def fSplitDataSetForModelingTesting(df, target_label='TARGET', train_target_values=[0, 1], ratio_sampling=1.):
    """Découpage du jeux d'essais pour les test de modélisation tout en reduisant la taille du jeux d'essai de manière cohérante

    Args:
        df (DataFrame): Jeu d'essais
        target_label (str, optional): Nom de la colonne comportant les target. Defaults to 'TARGET'.
        train_target_values (list, optional): Liste des valeurs possible pour les individues d'entraienement. Defaults to [0, 1].
        ratio_sampling (float, optional): Taux de fractionnement des individus du jeux d'essais. Defaults to 1..

    Returns:
        Datafraime: Jeu de données d'entrainement
        Series: Target du jeux d'entrainement
        DataFrame: Jeu de données de test
    """    
    train = df.loc[df[target_label].isin(train_target_values)].copy()
    target = train[target_label]
    train.drop(columns=target_label, inplace=True)
    train.reset_index(drop=True)
    
    test = df.loc[~df[target_label].isin(train_target_values)].copy()
    test.drop(columns=target_label, inplace=True)
    test.reset_index(drop=True)
    
    if ratio_sampling < 1.:
        train = pd.DataFrame([])
        for value in train_target_values:
            train = pd.concat([train, 
                               df.loc[df[target_label] == value].sample(n=int(df.loc[df[target_label] == value].shape[0] * ratio_sampling), random_state=72)
                              ])
        target = train[target_label]
        train.drop(columns=target_label, inplace=True)
        train.reset_index(drop=True)

        test = df.loc[~df[target_label].isin(train_target_values)].sample(n=int(df.loc[~df[target_label].isin(train_target_values)].shape[0] * ratio_sampling), random_state=72)
        test.drop(columns=target_label, inplace=True)
        test.reset_index(drop=True)

    return train, target, test

def fGetExecuteTime(start_time):
    """Affichage structuré des délais intermédiaires d'exécution

    Args:
        start_time (DateTime): DateTime de référence pour le calcul du délais

    Returns:
        str: Print le temps et le délais d'exécution
    """    
    if start_time:
        duration = float(datetime.now().timestamp() - start_time)
        s_duration = f"{str(timedelta(seconds=duration))[:-5]}"
        print(f"\n{datetime.now().strftime('%d/%m/%Y, %H:%M:%S')}, durée d'exécution : {s_duration}")
    
    return datetime.now().timestamp()


def fGetDataSets(path, file, use_dict=False, encoding='utf8', sep=';', 
                 features_index=None, features_to_keep=None, features_target_label='TARGET', features_target_values=[0, 1], 
                 ratio_sampling=1.0, test_size=0.2, random_state=72): 
    """_summary_

    Args:
        path (str): Chemin d'accès au repertoire du fichier de données
        file (str): Nom du fichier à charger
        use_dict (bool, optional): Utilise un dictionnaire pour spécifier les types des données. Defaults to False.
        encoding (str, optional): Encoding du fichier. Defaults to 'utf8'.
        sep (str, optional): Caractère de séparation des données. Defaults to ';'.
        features_index (str, optional): Nom de la colone de l'index. Defaults to None.
        features_to_keep (list, optional): features à conserver. Defaults to None.
        features_target_label (str, optional): Nom de la colonne des target. Defaults to 'TARGET'.
        features_target_values (list, optional): Liste des valeurs possible pour les targets. Defaults to [0, 1].
        ratio_sampling (float, optional): Taux de fractionnement des individus du jeux d'essais. Defaults to 1..
        test_size (float, optional): Taux de fractionnement de données réservées pour la phase de test. Defaults to 0.2.
        random_state (int, optional): Graine de conservation d'état. Defaults to 72.

    Returns:
        DataFrame: DataFrame et Target d'entrainement et de validation
    """    
    start_time = fGetExecuteTime(None)
    
    df = fReadDataFrameFile(path, file, use_dict=use_dict, encoding=encoding, sep=sep)
    if features_index:
        df.set_index(features_index, inplace=True)
    
    
    if features_to_keep:
        df = df[features_to_keep].copy()

    df_train, target, df_test = fSplitDataSetForModelingTesting(df, target_label=features_target_label, train_target_values=features_target_values, ratio_sampling=ratio_sampling)
            
    X_train, X_valid, y_train, y_valid = train_test_split(df_train, target, test_size=test_size, random_state=random_state)
            
    print(f"Shape : Dataset initial {df.shape} | X_train {X_train.shape} | y_train {y_train.shape} | X_valid {X_train.shape} | y_valid {y_train.shape} | X_test {df_test.shape}\n")
    print(f"list of features dataset initial : {list(df)}\n")
    print(f"list of df_train : {list(df_train)}\n")
    print(f"list of target : {target.name}\n")
    print(f"list of df_test : {list(df_test)}\n")
    
    _ = fGetExecuteTime(start_time)
    
    return X_train, X_valid, y_train, y_valid
    

def fRunTrainValideModel(model, experiment_label, experiment_tags, X_train, X_valid, y_train, y_valid, scorer="custom", best_score=0):
    """Entrainement et validation des modeles dans mlflow

    Args:
        model (object): Modele à entrainer ou valider
        experiment_label (str): Nom de l'experiment mlflow
        experiment_tags (dict): Dictionnaire de tags à tracer pendant le run
        X_train (array): Matrice des données d'entrainnement
        X_valid (array): Matrice des données de validation
        y_train (array): Listes des target d'entrainement
        y_valid (array): Listes des target de validation
        scorer (str, optional): Méthode de scoring ou nom du score standard à utiliser. Defaults to "custom".
        best_score (float, optional): Valeur du mailleur score obtenu. Defaults to 0.

    Returns:
        float: Meilleur score tracé.
    """    
    start_time = fGetExecuteTime(None)
    set_tracking_uri("http://127.0.0.1:5000")
    #mlflow ui --backend-store-uri "file:///Users/petx698/OneDrive - LA POSTE GROUPE/Documents/MyDev/Python/Projets/ocr/ocr-ds-projet7/mlflow_tracking"
    #set_tracking_uri("file:///Users/petx698/OneDrive - LA POSTE GROUPE/Documents/MyDev/Python/Projets/ocr/ocr-ds-projet7/mlflow_tracking")
    
    experiment = MlflowClient().get_experiment_by_name(experiment_label)
    
    if experiment is None:
        experiment_id = MlflowClient().create_experiment(experiment_label, artifact_location=Path.cwd().joinpath("mlruns").as_uri(), tags=experiment_tags)
    else:
        experiment_id = experiment.experiment_id
        
    autolog()
    with start_run(experiment_id=experiment_id) as run:
        
        model_name = str(model).split("(")[0].strip()
        print(f"Model : {model_name}")
        
        params = model.get_params()
        print(f"Params : {params}\n")
        log_params(params)
        
        model_params_id = hashlib.md5('-'.join([model_name, str(params)]).encode()).hexdigest()
        
        set_tag("model_params_id", model_params_id)

        # Cross validation
        cv_start_time = datetime.now().timestamp()
    
        if scorer == "custom":
            # Instanciation du Scorer
            custom_scorer = make_scorer(custom_scoring_min_fn, greater_is_better=True, needs_proba=False)
            print(f"Custom scorer : {custom_scorer}\n")
            scores = cross_validate(model, X_train, y_train, cv=10, scoring=custom_scorer, return_estimator=True)
        else: #'roc_auc'
            scores = cross_validate(model, X_train, y_train, cv=10, scoring=scorer, return_estimator=True)
            print(f"standard scorer : {scorer}\n")
        
        cv_timedelta = timedelta(seconds=(float(datetime.now().timestamp() - cv_start_time))).seconds
        if int(cv_timedelta) > 60:
            cv_duration = f"{int(cv_timedelta / 60)} minutes"
        else:
            cv_duration = f"{int(cv_timedelta)} secondes"
 
        best_iteration = scores['test_score'].argmax(axis=0)
        best_model = scores['estimator'][best_iteration]
                
        print(f"Best score : {scores['test_score'][best_iteration]:0.4f}, Mean score : {scores['test_score'].mean():0.4f}, with a standard deviation of {scores['test_score'].std():0.4f}, during {cv_duration}.")

        log_metric(f"best iteration score", scores['test_score'][best_iteration])
        log_metric(f"mean score", scores['test_score'].mean())
        log_metric(f"standard deviation", scores['test_score'].std())
        log_metric(f"execution delay in sec", cv_timedelta)

        y_pred = best_model.predict(X_valid)

        print(classification_report(y_valid, y_pred))
        print()
        print("f1-score : ", f1_score(y_valid, y_pred), " <==> f2-score : ", fbeta_score(y_valid, y_pred, beta=2))
        print()
        
        log_metric("f1-score", f1_score(y_valid, y_pred))
        log_metric("f2-score", fbeta_score(y_valid, y_pred, beta=2))
        
        if scores['test_score'][best_iteration] > best_score:  
            signature = infer_signature(X_valid, y_pred)
            
            log_model(best_model, "model", signature=signature, registered_model_name=f"{model_name}_{model_params_id}")
            best_score = scores['test_score'][best_iteration]

        print(f"Run ID: {run.info.run_id} - Model-Params ID: {model_params_id}")            
        print()
    
    _ = fGetExecuteTime(start_time)
    
    return best_score

    
def Exploration():
    """Lancement de l'exploration du meilleur modele
    """    
    ##logging.basicConfig(filename='mlflow.log', encoding='utf-8', level=logging.DEBUG)
    #logger = logging.getLogger("mlflow")
    #logger.setLevel(logging.DEBUG)
    #
    ## create file handler which logs even debug messages
    #fh = logging.FileHandler('mlflow.log')
    #fh.setLevel(logging.DEBUG)
    #logger.addHandler(fh)
    
    # Avant execution : export/set MLFLOW_TRACKING_URI=http://127.0.0.1:5000
    print("Lancement du script.")
    print("Chargement des données.")
    best_score = 0
    
    X_train, X_valid, y_train, y_valid = fGetDataSets(os.path.join("data", "cleaned"), 
                                                                       'features_smote_03', 
                                                                       use_dict=True, 
                                                                       encoding='utf8', 
                                                                       sep=';', 
                                                                       features_index="SK_ID_CURR",
                                                                       features_target_label='TARGET', 
                                                                       features_target_values=[0, 1],
                                                                       ratio_sampling=0.4, 
                                                                       test_size=0.25, 
                                                                       random_state=72
                                                                       )
   
    #Paramètres par modele sur lesquels itérer
    list_tests = { 
        "LogisticRegression": [{"max_iter": 1000, "random_state": 72},
                               {"C": 0.0001, "max_iter": 1000, "random_state": 72},
                               {"C": 0.001, "max_iter": 1000, "random_state": 72},
                               {"C": 0.01, "max_iter": 1000, "random_state": 72}],
        "RandomForestClassifier": [{"random_state": 72},
                                   {"n_estimators": 1, "random_state": 72, "verbose": 0, "n_jobs": -1},
                                   {"n_estimators": 10, "random_state": 72, "verbose": 0, "n_jobs": -1},
                                   {"n_estimators": 100, "random_state": 72, "verbose": 0, "n_jobs": -1},
                                   {"n_estimators": 1000, "random_state": 72, "verbose": 0, "n_jobs": -1}],
        "LinearSVC": [{"random_state": 72}],
        "LGBMClassifier": [{"random_state": 72},
                           {"boosting_type": "gbdt", "num_leaves": 31, "max_depth": -1, "learning_rate": 0.05, "objective": "binary", 
                            "min_split_gain": 0.0, "min_child_weight": 0.001, "min_child_samples": 20, "subsample": 0.8, "subsample_freq": 0, 
                            "colsample_bytree": 1.0, "reg_alpha": 0.1, "reg_lambda": 0.1, "random_state": 72, "n_jobs": -1, "importance_type": "split"}],
        "XGBClassifier": [{'verbosity': 0, 'subsample': 0.8, 'random_state': 72},
                          {'booster': 'gbtree', 'verbosity': 0, 'subsample': 0.8, 'random_state': 72},
                          {'booster': 'dart', 'verbosity': 0, 'subsample': 0.8, 'random_state': 72},
                          {'booster': 'dart', "max_depth": 6, "eta": 0.05, "min_child_weight": 3, "subsample": 0.8, "silent": 0, "objective": "reg:linear", "num_round": 200, 'verbosity': 0, 'random_state': 72},
                          {'booster': 'dart', 'eta': 0.45, 'gamma': 0, 'max_depth': 6, 'min_child_weight': 0, 'max_delta_step': 3, 'lambda': 0.55, 'alpha': 0.0001, 'num_parallel_tree': 1, 'verbosity': 0, 'subsample': 0.8, 'random_state': 72},
                          {'booster': 'dart', 'eta': 0.55, 'gamma': 0, 'max_depth': 6, 'min_child_weight': 0, 'max_delta_step': 3, 'lambda': 0.55, 'alpha': 0.0001, 'num_parallel_tree': 1, 'verbosity': 0, 'subsample': 0.8, 'random_state': 72},
                          {'booster': 'dart', 'eta': 0.5, 'gamma': 0.1, 'max_depth': 6, 'min_child_weight': 0, 'max_delta_step': 3, 'lambda': 0.55, 'alpha': 0.0001, 'num_parallel_tree': 1, 'verbosity': 0, 'subsample': 0.8, 'random_state': 72},
                          {'booster': 'dart', 'eta': 0.5, 'gamma': 0.001, 'max_depth': 6, 'min_child_weight': 0, 'max_delta_step': 3, 'lambda': 0.55, 'alpha': 0.0001, 'num_parallel_tree': 1, 'verbosity': 0, 'subsample': 0.8, 'random_state': 72},
                          {'booster': 'dart', 'eta': 0.5, 'gamma': 0, 'max_depth': 5, 'min_child_weight': 0, 'max_delta_step': 3, 'lambda': 0.55, 'alpha': 0.0001, 'num_parallel_tree': 1, 'verbosity': 0, 'subsample': 0.8, 'random_state': 72},
                          {'booster': 'dart', 'eta': 0.5, 'gamma': 0, 'max_depth': 7, 'min_child_weight': 0, 'max_delta_step': 3, 'lambda': 0.55, 'alpha': 0.0001, 'num_parallel_tree': 1, 'verbosity': 0, 'subsample': 0.8, 'random_state': 72},
                          {'booster': 'dart', 'eta': 0.5, 'gamma': 0, 'max_depth': 6, 'min_child_weight': 1, 'max_delta_step': 3, 'lambda': 0.55, 'alpha': 0.0001, 'num_parallel_tree': 1, 'verbosity': 0, 'subsample': 0.8, 'random_state': 72},
                          {'booster': 'dart', 'eta': 0.5, 'gamma': 0, 'max_depth': 6, 'min_child_weight': 0, 'max_delta_step': 2, 'lambda': 0.55, 'alpha': 0.0001, 'num_parallel_tree': 1, 'verbosity': 0, 'subsample': 0.8, 'random_state': 72},
                          {'booster': 'dart', 'eta': 0.5, 'gamma': 0, 'max_depth': 6, 'min_child_weight': 0, 'max_delta_step': 4, 'lambda': 0.55, 'alpha': 0.0001, 'num_parallel_tree': 1, 'verbosity': 0, 'subsample': 0.8, 'random_state': 72},
                          {'booster': 'dart', 'eta': 0.5, 'gamma': 0, 'max_depth': 6, 'min_child_weight': 0, 'max_delta_step': 3, 'lambda': 0.55, 'alpha': 0, 'num_parallel_tree': 1, 'verbosity': 0, 'subsample': 0.8, 'random_state': 72},
                          {'booster': 'dart', 'eta': 0.5, 'gamma': 0, 'max_depth': 6, 'min_child_weight': 0, 'max_delta_step': 3, 'lambda': 0.55, 'alpha': 0.01, 'num_parallel_tree': 1, 'verbosity': 0, 'subsample': 0.8, 'random_state': 72},
                          {'booster': 'dart', 'eta': 0.5, 'gamma': 0, 'max_depth': 6, 'min_child_weight': 0, 'max_delta_step': 3, 'lambda': 0.55, 'alpha': 0.0001, 'num_parallel_tree': 10, 'verbosity': 0, 'subsample': 0.8, 'random_state': 72},
                          {'booster': 'dart', 'eta': 0.5, 'gamma': 0, 'max_depth': 6, 'min_child_weight': 0, 'max_delta_step': 3, 'lambda': 0.55, 'alpha': 0.0001, 'num_parallel_tree': 100, 'verbosity': 0, 'subsample': 0.8, 'random_state': 72},
                          {'booster': 'dart', 'eta': 0.5, 'gamma': 0, 'max_depth': 6, 'min_child_weight': 0, 'max_delta_step': 3, 'lambda': 0.55, 'alpha': 0.0001, 'num_parallel_tree': 1, 'verbosity': 0, 'subsample': 0.8, 'random_state': 72}]
        }
        
    print("Lancement des tests.")
    
    experiment_label = "Tests Models Experiments"
    experiment_tags = {"Typologie": "Exploration", "Campagne": "01"}

    nb_tests = 0
    for _, values in list_tests.items():
        nb_tests = nb_tests + len(values)
    
    compteur = 1
    for model_name, list_params in list_tests.items():    
        for params in list_params:
            print(f"Itération {compteur}/{nb_tests} : {model_name}")
                
            if model_name == "LogisticRegression":
                model = LogisticRegression(**params)
            elif model_name == "RandomForestClassifier":
                model = RandomForestClassifier(**params)
            elif model_name == "LinearSVC":
                model = LinearSVC(**params)
            elif model_name == "LGBMClassifier":
                model = LGBMClassifier(**params)
            elif model_name == "XGBClassifier":
                model = XGBClassifier(**params)
            
            best_score = fRunTrainValideModel(model, experiment_label, experiment_tags, X_train, X_valid, y_train, y_valid, scorer="custom", best_score=best_score)
            compteur += 1

def Validation():
    """Lancement de la validation du modele, vérification de la performance avec des seed différents
    """    
    ##logging.basicConfig(filename='mlflow.log', encoding='utf-8', level=logging.DEBUG)
    #logger = logging.getLogger("mlflow")
    #logger.setLevel(logging.DEBUG)
    #
    ## create file handler which logs even debug messages
    #fh = logging.FileHandler('mlflow.log')
    #fh.setLevel(logging.DEBUG)
    #logger.addHandler(fh)
    
    # Avant execution : export/set MLFLOW_TRACKING_URI=http://127.0.0.1:5000
    print("Lancement du script.")
    print("Chargement des données.")
    best_score = 0   

    
    list_tests = {"XGBClassifier": [
        {'booster': 'dart', 'eta': 0.5, 'gamma': 0, 'max_depth': 6, 'min_child_weight': 0, 'max_delta_step': 3, 'lambda': 0.55, 'alpha': 0.0001, 'num_parallel_tree': 1, 'verbosity': 0, 'subsample': 0.8},
        {'booster': 'dart', 'eta': 0.5, 'gamma': 0, 'max_depth': 6, 'min_child_weight': 0, 'max_delta_step': 3, 'lambda': 0.55, 'alpha': 0.0001, 'num_parallel_tree': 1, 'verbosity': 0, 'subsample': 0.8},
        {'booster': 'dart', 'eta': 0.5, 'gamma': 0, 'max_depth': 6, 'min_child_weight': 0, 'max_delta_step': 3, 'lambda': 0.55, 'alpha': 0.0001, 'num_parallel_tree': 1, 'verbosity': 0, 'subsample': 0.8}]
        }
        
    print("Lancement de la validation.")
    
    experiment_label = "Validation Models Experiments"
    experiment_tags = {"Typologie": "Validation", "Campagne": "01"}

    nb_tests = 0
    for _, values in list_tests.items():
        nb_tests = nb_tests + len(values)
    
    compteur = 1
    for model_name, list_params in list_tests.items():    
        for params in list_params:
            fPrintTitleSouligne(f"Itération {compteur}/{nb_tests} : {model_name}")
            
            seed = int(random() *1000)
            
            params["random_state"] = seed
            
            X_train, X_valid, y_train, y_valid = fGetDataSets(os.path.join("data", "cleaned"), 
                                                                       'features_smote_03', 
                                                                       use_dict=True, 
                                                                       encoding='utf8', 
                                                                       sep=';', 
                                                                       features_index="SK_ID_CURR",
                                                                       features_target_label='TARGET', 
                                                                       features_target_values=[0, 1],
                                                                       ratio_sampling=0.6, 
                                                                       test_size=0.25, 
                                                                       random_state=seed
                                                                       )

                
            if model_name == "LogisticRegression":
                model = LogisticRegression(**params)
            elif model_name == "RandomForestClassifier":
                model = RandomForestClassifier(**params)
            elif model_name == "LinearSVC":
                model = LinearSVC(**params)
            elif model_name == "LGBMClassifier":
                model = LGBMClassifier(**params)
            elif model_name == "XGBClassifier":
                model = XGBClassifier(**params)
            
            best_score = fRunTrainValideModel(model, experiment_label, experiment_tags, X_train, X_valid, y_train, y_valid, scorer="custom", best_score=best_score)
            compteur += 1

def Maintenance(ratio, campagne):
    """Suivi des performance du modele lors de maintenance avec une évolution du jeux de données

    Args:
        ratio (float): Fractionnement du jeu de donnée pour simuler un apport nouveau de données
        campagne (str): Nom de la campagne de maintenance
    """    
    ##logging.basicConfig(filename='mlflow.log', encoding='utf-8', level=logging.DEBUG)
    #logger = logging.getLogger("mlflow")
    #logger.setLevel(logging.DEBUG)
    #
    ## create file handler which logs even debug messages
    #fh = logging.FileHandler('mlflow.log')
    #fh.setLevel(logging.DEBUG)
    #logger.addHandler(fh)
    
    # Avant execution : export/set MLFLOW_TRACKING_URI=http://127.0.0.1:5000
    print("Lancement du script.")
    print("Chargement des données.")
    best_score = 0   
    
    X_train, X_valid, y_train, y_valid = fGetDataSets(os.path.join("data", "cleaned"), 
                                                                       'features_smote_03', 
                                                                       use_dict=True, 
                                                                       encoding='utf8', 
                                                                       sep=';', 
                                                                       features_index="SK_ID_CURR",
                                                                       features_target_label='TARGET', 
                                                                       features_target_values=[0, 1],
                                                                       ratio_sampling=ratio, 
                                                                       test_size=0.25,
                                                                       random_state=72
                                                                       )

    list_tests = {"XGBClassifier": [
        {'booster': 'dart', 'eta': 0.5, 'gamma': 0, 'max_depth': 6, 'min_child_weight': 0, 'max_delta_step': 3, 'lambda': 0.55, 'alpha': 0.0001, 'num_parallel_tree': 1, 'verbosity': 0, 'subsample': 0.8}]
        }
        
    print("Lancement de la validation.")
    
    experiment_label = "Maintenance Models Experiments"
    experiment_tags = {"Typologie": "Maintenance", "Campagne": f"{campagne:02d}"}

    nb_tests = 0
    for _, values in list_tests.items():
        nb_tests = nb_tests + len(values)
    
    compteur = 1
    for model_name, list_params in list_tests.items():    
        for params in list_params:
            fPrintTitleSouligne(f"Itération {compteur}/{nb_tests} : {model_name}")
                
            if model_name == "LogisticRegression":
                model = LogisticRegression(**params)
            elif model_name == "RandomForestClassifier":
                model = RandomForestClassifier(**params)
            elif model_name == "LinearSVC":
                model = LinearSVC(**params)
            elif model_name == "LGBMClassifier":
                model = LGBMClassifier(**params)
            elif model_name == "XGBClassifier":
                model = XGBClassifier(**params)
            
            best_score = fRunTrainValideModel(model, experiment_label, experiment_tags, X_train, X_valid, y_train, y_valid, scorer="custom", best_score=best_score)
            compteur += 1
      
            
def main():
    
    if len( sys.argv ) > 2:
        print( "Loan Validation App" )
        print( "\tusage: python3 maintenance_xgboost.py intValue" )
        print( "Args => 1 for Exploration, 2 for Validation and 3 for Maintenance." )
    else:
        strParam = sys.argv[2]
        try:
            param = int(strParam)
            
            print()
            if param == 1:          
                print("Exploration")
                Exploration()
            elif param == 2 :            
                print("Validation")
                Validation()
            elif param == 3 :            
                print("Maintenance")
                for campagne, ratio in enumerate([0.4, 0.6, 0.8]):
                    print(f"Campagne {campagne+1:02d}")
                    Maintenance(ratio, campagne+1)
            else:
                raise(ValueError)
                    
        except ValueError: 
            print( "Bad parameter value: %s" % strParam, file=sys.stderr )


if __name__ == '__main__':
    main()
    
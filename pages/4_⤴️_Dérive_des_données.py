import os
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import streamlit as st
import streamlit.components.v1 as components

from sqlalchemy import text
from sqlalchemy import create_engine

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split

from mlflow import autolog, log_metric, log_params, start_run, set_tag, set_tracking_uri
from mlflow.models import infer_signature
from mlflow.sklearn import log_model
from mlflow.xgboost import log_model
from mlflow.lightgbm import log_model
from mlflow.client import MlflowClient
import mlflow.pyfunc


from evidently import ColumnMapping

from evidently.report import Report
from evidently.metrics.base_metric import generate_column_metrics
from evidently.metric_preset import DataDriftPreset, TargetDriftPreset, DataQualityPreset, RegressionPreset, ClassificationPreset
from evidently.metrics import *

from evidently.test_suite import TestSuite
from evidently.tests.base_test import generate_column_tests
from evidently.test_preset import DataStabilityTestPreset, NoTargetPerformanceTestPreset, RegressionTestPreset, BinaryClassificationTestPreset
from evidently.tests import *

import base64

import warnings
warnings.filterwarnings('ignore')
warnings.simplefilter('ignore')


def fCreateDataBase(url_db):
    global engine
    engine = create_engine(url_db)

@st.cache_data
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
        _type_: _description_
    """
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
                    if 'timedelta' in str(v):
                        list_var_timedelta = list_var_timedelta + [k]
                    
    path_file = os.path.join(path, f'{file}.csv')
    if os.path.exists(path_file):
        if use_dict:
            df = pd.read_csv(path_file, sep=sep, encoding=encoding, dtype=liste_columns_type)
            if list_var_datetime:
                for var in list_var_datetime:
                    df[var] = pd.to_datetime(df[var])
                for var in list_var_timedelta:
                    df[var] = pd.to_timedelta(df[var])
        else:
            df = pd.read_csv(path_file, sep=sep, encoding=encoding)
    
    return df


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
    df = fReadDataFrameFile(path, file, use_dict=use_dict, encoding=encoding, sep=sep)
    if features_index:
        df.set_index(features_index, inplace=True)
    
    
    if features_to_keep:
        df = df[features_to_keep].copy()

    df_train, target, df_test = fSplitDataSetForModelingTesting(df, target_label=features_target_label, train_target_values=features_target_values, ratio_sampling=ratio_sampling)
            
    X_train, X_valid, y_train, y_valid = train_test_split(df_train, target, test_size=test_size, random_state=random_state)
                
    return X_train, X_valid, y_train, y_valid
    
    
def get_rapport_name(file_name):
    if str(file_name).startswith("TrackingDataDriftReport"):
        return f"Rapport du {datetime.strptime(file_name.split('_')[1][:-5], '%Y%m%d-%H%M%S').strftime('%d/%m/%Y  à %H:%M:%S')}"
    else:
        return None


if 'engine' not in globals(): 
    fCreateDataBase("sqlite:///loan_validation.db")

st.set_page_config(
    page_title="Loan Validation",
    page_icon=":white_check_mark:",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'About': "# Application de validation des demandes de prêt"
    }
)


df_current = pd.read_sql_query(sql=f"select * from prospects;", con=engine)


st.title(":white_check_mark: Application de validation des prêts")
st.header(f":arrow_heading_up: *Data Drift*")



DATA_DRIFT_REPORT_DIR = os.path.join("TrackingDataDriftReport")
liste_report = sorted([f for f in os.listdir(DATA_DRIFT_REPORT_DIR) if os.path.isfile(os.path.join(DATA_DRIFT_REPORT_DIR, f))], reverse=True)
max_rapport = len(liste_report)
if max_rapport > 20: max_rapport = 20

if max_rapport < 10: default_nb_rapport = max_rapport
else: default_nb_rapport = 10

nb_rapport = st.sidebar.slider("**Nombre de rapport**", 1, max_rapport, value=10),
liste_report_name = [get_rapport_name(file) for file in liste_report[:nb_rapport[0]]]

report_4_consult = st.sidebar.radio(f"**Liste des {nb_rapport[0]} derniers rapports disponibles**", liste_report_name, horizontal=True) 

st.divider()

path_dataDriftReport = os.path.join(".", "TrackingDataDriftReport", liste_report[liste_report_name.index(report_4_consult)])
html_dataDriftReport = ""    
if os.path.exists(path_dataDriftReport):
    with open(path_dataDriftReport, 'r', encoding='utf-8') as reader:
        html_dataDriftReport = reader.read()

components.html(html_dataDriftReport, height=10000, scrolling=False)

    

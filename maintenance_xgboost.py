import os
from collections import Counter
from datetime import datetime, date, time, timedelta, timezone
import pandas as pd
import numpy as np
import pickle
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import confusion_matrix, f1_score, fbeta_score, adjusted_rand_score, accuracy_score, auc, roc_auc_score, roc_curve, make_scorer, classification_report, fbeta_score
from xgboost import XGBClassifier
from sklearn.model_selection import KFold, cross_validate, cross_val_score, train_test_split, GridSearchCV





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
    datetime_is_used = False
    list_var_datetime = []
    
    timedelta_is_used = False
    list_var_timedelta = []
    
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
    if os.path.exists(path_file):
        return pickle.load(open(path_file, "rb"))


def custom_scoring(y_true, y_pred, beta=None):
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
    # Accord de prêt alors qu'ils vont être en défaut de paiement
    return custom_scoring(y_true, y_pred, beta=2)


def fSplitDataSetForModelingTesting(df, target_label='TARGET', train_target_values=[0, 1], ratio_sampling=1.):
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



def main():
    
    
    list_of_best_features = ['app_IS_MASCULIN', 'app_REGION_RATING_CLIENT', 'app_REG_CITY_NOT_WORK_CITY', 'app_NAME_INCOME_TYPE_Working', 
                            'app_NAME_EDUCATION_TYPE_Higher education', 'agg_client_bureau_CREDIT_ACTIVE_Active_sum_sum', 
                            'agg_client_bureau_DAYS_CREDIT_ENDDATE_count_nunique', 'agg_client_bureau_AMT_CREDIT_SUM_DEBT_mean_nunique', 
                            'agg_client_credit_AMT_DRAWINGS_ATM_CURRENT_mean_nunique']

    dict_rename_features = {'app_IS_MASCULIN': "Genre", 
                            'app_REGION_RATING_CLIENT': "Evaluation de la région d'habitation", 
                            'app_REG_CITY_NOT_WORK_CITY': "Ecart adresse pro/perso", 
                            'app_NAME_INCOME_TYPE_Working': "A une activité pro", 
                            'app_NAME_EDUCATION_TYPE_Higher education': "Scolarité universitaire", 
                            'agg_client_bureau_CREDIT_ACTIVE_Active_sum_sum': "Risque de l'ensemble des crédits", 
                            'agg_client_bureau_DAYS_CREDIT_ENDDATE_count_nunique': "Crédit CB en cours", 
                            'agg_client_bureau_AMT_CREDIT_SUM_DEBT_mean_nunique': "Nombre de mensualitées", 
                            'agg_client_credit_AMT_DRAWINGS_ATM_CURRENT_mean_nunique': "Nombre de CB pour retrait"
                            }

    df = fReadDataFrameFile(os.path.join("data", "cleaned"), 'features_smote_03')
    df['SK_ID_CURR'] = df['SK_ID_CURR'].astype(int)


    ratio_sampling=1.0
    df_train, target, df_test = fSplitDataSetForModelingTesting(df, ratio_sampling=ratio_sampling)
    
    df_train.set_index('SK_ID_CURR', inplace=True)
    df_train = df_train[list_of_best_features].rename(columns=dict_rename_features)

    df_test.set_index('SK_ID_CURR', inplace=True)
    df_test = df_test[list_of_best_features].rename(columns=dict_rename_features)
    
    X_train, X_valid, y_train, y_valid = train_test_split(df_train, target, test_size=0.2, random_state=72)
    
    list_of_best_features_new_names = list(X_train)

    

    params = {
        'booster': 'dart',
        'verbosity': 0,
        'eta': 0.5,
        'gamma': 0,
        'max_depth': 6,
        'min_child_weight': 0, 
        'max_delta_step': 3, 
        'subsample': 0.8,
        'lambda': 0.55, 
        'alpha': 0.0001, 
        'num_parallel_tree': 1, 
        'random_state': 72
    }
    print(f"Params : {params}\n")
    
    model = XGBClassifier(**params)
    print(f"Model : {model}\n")
       
    custom_scorer = make_scorer(custom_scoring_min_fn, greater_is_better=True, needs_proba=False)
    print(f"Custom scorer : {custom_scorer}\n")
 
    cv_start_time = datetime.now().timestamp()

    # Cross validation
    scores = cross_validate(model, X_train, y_train, cv=10, scoring=custom_scorer, return_estimator=True)

    cv_timedelta = timedelta(seconds=(float(datetime.now().timestamp() - cv_start_time))).seconds
    if int(cv_timedelta) > 60:
        cv_duration = f"{int(cv_timedelta / 60)} minutes"
    else:
        cv_duration = f"{int(cv_timedelta)} secondes"

        
    best_iteration = scores['test_score'].argmax(axis=0)
    best_model = scores['estimator'][best_iteration]
        
    print(f"Best score : {scores['test_score'][best_iteration]:0.4f}, Mean score : {scores['test_score'].mean():0.4f}, with a standard deviation of {scores['test_score'].std():0.4f}, during {cv_duration}.")
    print()
    

if __name__ == '__main__':
    main()
    
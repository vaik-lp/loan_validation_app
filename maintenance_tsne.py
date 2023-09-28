""" Classe de maintenance des données t-SNE par rapport à un modelet un jeu de données spécifiques"""

import os
from datetime import datetime, timedelta
import pandas as pd
import pickle

from sklearn.manifold import TSNE
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


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
    """Chargement des ressources 

    Args:
        path_file (string): chemin d'accès de la ressources à charger

    Returns:
        object: ressources chargées, elle peut être de différentes natures
    """    
    if os.path.exists(path_file):
        return pickle.load(open(path_file, "rb"))



def fGetExecuteTime(start_time):
    """Affichage structuré des délais intermédiaires d'exécution

    Args:
        start_time (DateTime): DateTime de référence pour le calcul du délais

    Returns:
        String: Print le temps et le délais d'exécution
    """    
    if start_time:
        duration = float(datetime.now().timestamp() - start_time)
        s_duration = f"{str(timedelta(seconds=duration))[:-5]}"
        print(f"\n{datetime.now().strftime('%d/%m/%Y, %H:%M:%S')}, durée d'exécution : {s_duration}")
    else:
        print(f"\n{datetime.now().strftime('%d/%m/%Y, %H:%M:%S')}")
    
    return datetime.now().timestamp()


def main():
    
    print()
    start_time = fGetExecuteTime(None)

    list_of_best_features = ['app_IS_MASCULIN', 'app_REGION_RATING_CLIENT', 'app_REG_CITY_NOT_WORK_CITY', 'app_NAME_INCOME_TYPE_Working', 
                            'app_NAME_EDUCATION_TYPE_Higher education', 'agg_client_bureau_CREDIT_ACTIVE_Active_sum_sum', 
                            'agg_client_bureau_DAYS_CREDIT_ENDDATE_count_nunique', 'agg_client_bureau_AMT_CREDIT_SUM_DEBT_mean_nunique', 
                            'agg_client_credit_AMT_DRAWINGS_ATM_CURRENT_mean_nunique']

    # renomage des colonnes pour donner du sens aux utilisateurs 
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

    # Chargement des données 
    df = fReadDataFrameFile(os.path.join("data", "cleaned"), 'features_03')
    df['SK_ID_CURR'] = df['SK_ID_CURR'].astype(int)    
    df.set_index('SK_ID_CURR', inplace=True)
    target = df['TARGET']
    df = df[list_of_best_features].rename(columns=dict_rename_features).copy()

    print(df.shape)
    print(df.head())
    
    # Calcul du t-SNE ordonnancé par un pipeline
    tsne = Pipeline([
    ("scaler", StandardScaler()),
    ("tsne", TSNE(n_components=2, verbose=0, perplexity=40, n_iter=300, random_state=72))])

    X_tsne = tsne.fit_transform(df.values)
    
    df_tsne = df.copy()

    df_tsne['TARGET'] = target
    df_tsne['tsne_1'] = X_tsne[:,0]
    df_tsne['tsne_2'] = X_tsne[:,1]

  
    #Sauvegarde des données pour réutilisation dans le dashboard
    path_file_data = os.path.join("data", "cleaned", 'tsne_data_new.pkl')
    pickle.dump(df_tsne, open(path_file_data, "wb"))
    
    
    print()
    start_time = fGetExecuteTime(start_time)

if __name__ == '__main__':
    main()
    
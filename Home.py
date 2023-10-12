import os
import pandas as pd
import numpy as np
import streamlit as st
import streamlit.components.v1 as components
from sqlalchemy import text
from sqlalchemy import create_engine
import pickle
import shap
import matplotlib.pyplot as plt

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
        DataFrame: Données chargées
    """
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
                    if 'timedelta' in str(v):
                        list_var_timedelta = list_var_timedelta + [k]
                    
    path_file = os.path.join(path, f'{file}.csv')
    
    # Lecture des données
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
    
            if 'SK_ID_CURR' in list(df):
                df.rename(columns={'SK_ID_CURR': 'ID'}, inplace=True)
            
            if 'ID' in list(df):
                df['ID'] = df['ID'].astype(int)
                df.set_index("ID")

    return df


@st.cache_data
def fGetScoresFromMLModel(df):
    """calcul des scores d'un jeu de données avec le modele de ML sauvegardé

    Args:
        df (DataFrame): Dataset sans target

    Returns:
        DataFrame: dataframe comprenant le score de risque de non paiement des échéances deprêt
    """    
    # Chargement de modele de ML précédement calculé
    model = fGetRessources(os.path.join("data", "cleaned", "xgboost_model.pkl"))

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
    
    df.rename(columns={'SK_ID_CURR': 'ID'}, inplace=True)
    df["ID"] = df["ID"].astype(int)
    df.set_index('ID', inplace=True)
    df = df.loc[df['TARGET'] == -999, list_of_best_features].rename(columns=dict_rename_features).copy()
    idx_df = df.index
    
    # Pédiction avec un score de probabilité (pas de classification)
    target_pred = model.predict_proba(df)[:, 1]
    
    # Submission dataframe
    
    df_submission = pd.DataFrame(idx_df.values.reshape(len(idx_df.values), 1), columns=['ID'])
    df_submission['TARGET'] = target_pred 
    df_submission['ID'] = df_submission['ID'].astype(int)
    df_submission.set_index('ID', inplace=True)

    return df_submission
    

@st.cache_data
def fCreateProspectDataFrame(df_clients, df_scors):
    """
    Création d'un DataFrame de prospect pour une utilisation dans le tableau de bord. 

    Args:
        df_clients (DataFrame): Données clients
        df_scors (DataFrame): Scorring des prospects
    """

    list_of_best_features = ['app_IS_MASCULIN', 'app_REGION_RATING_CLIENT', 'app_REG_CITY_NOT_WORK_CITY', 'app_NAME_INCOME_TYPE_Working', 
                         'app_NAME_EDUCATION_TYPE_Higher education', 'agg_client_bureau_CREDIT_ACTIVE_Active_sum_sum', 
                         'agg_client_bureau_DAYS_CREDIT_ENDDATE_count_nunique', 'agg_client_bureau_AMT_CREDIT_SUM_DEBT_mean_nunique', 
                         'agg_client_credit_AMT_DRAWINGS_ATM_CURRENT_mean_nunique']

    dict_rename_features = {
                        'app_IS_MASCULIN': "Genre", 
                        'app_REGION_RATING_CLIENT': "Evaluation de la région d'habitation", 
                        'app_REG_CITY_NOT_WORK_CITY': "Ecart adresse pro/perso", 
                        'app_NAME_INCOME_TYPE_Working': "A une activité pro", 
                        'app_NAME_EDUCATION_TYPE_Higher education': "Scolarité universitaire", 
                        'agg_client_bureau_CREDIT_ACTIVE_Active_sum_sum': "Risque de l'ensemble des crédits", 
                        'agg_client_bureau_DAYS_CREDIT_ENDDATE_count_nunique': "Crédit CB en cours", 
                        'agg_client_bureau_AMT_CREDIT_SUM_DEBT_mean_nunique': "Nombre de mensualitées", 
                        'agg_client_credit_AMT_DRAWINGS_ATM_CURRENT_mean_nunique': "Nombre de CB pour retrait"
                        }

    df_prospects = df_clients.loc[df_clients['TARGET'] == -999, list_of_best_features].rename(columns=dict_rename_features).copy()
    df_prospects['Ecart adresse pro/perso'] = df_prospects['Ecart adresse pro/perso'].astype(int)
    df_prospects["Risque de l'ensemble des crédits"] = df_prospects["Risque de l'ensemble des crédits"].astype(int)
    df_prospects["Crédit CB en cours"] = df_prospects["Crédit CB en cours"].astype(int)
    df_prospects["Nombre de mensualitées"] = df_prospects["Nombre de mensualitées"].astype(int)
    df_prospects["Nombre de CB pour retrait"] = df_prospects["Nombre de CB pour retrait"].astype(int)
    df_prospects['Risque en %'] = df_scors['TARGET']
    df_prospects['Risque en %'] = df_prospects['Risque en %'].astype(float)
    df_prospects['Etat'] = np.nan
    list_index_to_valide_loan = df_prospects.loc[df_prospects['Risque en %'] < 0.6].sample(frac=0.32).index
    df_prospects.loc[list_index_to_valide_loan,'Etat'] = [1.0] * len(list_index_to_valide_loan)
    
    # By pass du rechargement de la table pour générer la couche de persistance entre les différentes pages et du statut du traitement des dossiers
    try:
        df_prospects.to_sql(name='prospects', con=engine, if_exists='fail', index=True, chunksize=1000)
    except Exception as e: 
        pass    
    

def fReset_states(df_prospects, reset_id=True):
    """
    Réinitialistion des filtres des widgets

    Args:
        df_prospects (DataFrame): DataFrame des prostects pour récupérer les listes des valeurs initiales.
        reset_id (bool, optional): Forcage pour réinitialiser le dossier client au dossier par défaut (premier dossier non traité). Defaults to True.
    """    
    if "w_dossier_unique" not in st.session_state: st.session_state.w_dossier_unique = False
    if reset_id: st.session_state.id = df_prospects.loc[df_prospects["Etat"].isna()].index[0]
    st.session_state.w_genre = "Tous"
    st.session_state.w_eval_reg = list(set(df_prospects["Evaluation de la région d'habitation"].astype(int)))
    st.session_state.w_ecart_adr = "Tous"
    st.session_state.w_activite_pro = "Tous"
    st.session_state.w_scol_univ = "Tous"
    st.session_state.w_risque_all_loan = (min(df_prospects["Risque de l'ensemble des crédits"]), max(df_prospects["Risque de l'ensemble des crédits"]))
    st.session_state.w_current_credit_cb = "Tous"
    st.session_state.w_nb_mens = (min(df_prospects["Nombre de mensualitées"]), max(df_prospects["Nombre de mensualitées"]))
    st.session_state.w_nb_cb = (min(df_prospects["Nombre de CB pour retrait"]), max(df_prospects["Nombre de CB pour retrait"]))
    st.session_state.w_risque_calc = (int(min(df_prospects["Risque en %"])*100), int(max(df_prospects["Risque en %"])*100+1))
    if st.session_state.w_dossier_unique == True:
        st.session_state.w_etat = ["A traiter", "Refusé", "Validé"]
        st.session_state.w_nombre_prospects = 1
    else:
        st.session_state.w_etat = ["A traiter"]
        st.session_state.w_nombre_prospects = 10
        
    st.experimental_set_query_params(**st.session_state)


@st.cache_resource
def fGetRessources(path_file):
    """Chargement et mise en cache des ressources pour ne les charger qu'une seule fois

    Args:
        path_file (string): chemin d'accès de la ressources à charger

    Returns:
        object: ressources chargées, elle peut être de différentes natures
    """    
    if os.path.exists(path_file):
        return pickle.load(open(path_file, "rb"))


#@st.cache_resource
def fCreateDataBase(url_db):
    """Chargement et mise en cache de la base de donnée popur ne la charger qu'une seule fois

    Args:
        url_db (string): url d'accès à la base de données. 
    """    
    global engine
    engine = create_engine(url_db)



def fGetUrl(action, id):
    """Construction de l'url pour réaliser des actions entre les pages en méthode GET

    Args:
        action (string): clé accociée aux actions à lancer
        id (string): identifiant du dossier a propager entre les différentes pages de l'application

    Returns:
        url: url permettant de lancer les actions souhaitées, notamment la persistance de la validation ou du refus d'un dossier
    """    
    url = f"./?id={id}&action={action}"
    for item in st.session_state.items():
        key, value = item
        if key not in ["id", "action"]:
            if (type(value) is list) | (type(value) is tuple):
                for val in value:
                    url = f"{url}&{key}={val}"
            else:
                url = f"{url}&{key}={value}"
    
    return url

        
def fReloadPage(df):
    """Gestion et mapping des données à transmettre entre pages

    Args:
        df (DataFrame): Données des dossiers en cours d'utilisation/manipulation

    Returns:
        DataFrame: Données filtrées selon l'utilsiation des widgets 
    """    
    if len(st.session_state.keys()) > 0:
        # Gestion différentes si on sélectionne un dossier unique ou si l'on travail sur une liste de dossier filtrés
        if st.session_state.w_dossier_unique:
            del st.session_state.w_nombre_prospects
            st.session_state.w_nombre_prospects = 1
            del st.session_state.w_etat
            st.session_state.w_etat = ["A traiter", "Refusé", "Validé"]
            df_wip = df.loc[df["ID"] == int(st.session_state.id)]
        
        else:
            if st.session_state.w_genre == "Homme": genre_lst = [1]
            elif st.session_state.w_genre == "Femme": genre_lst = [0]
            else: genre_lst = [0, 1]

            if st.session_state.w_ecart_adr == "Non": ecart_adr_lst = [0]
            elif st.session_state.w_ecart_adr == "Oui": ecart_adr_lst = [1]
            else: ecart_adr_lst = [0, 1]
        
            if st.session_state.w_activite_pro == "Non": activite_pro_lst = [0]
            elif st.session_state.w_activite_pro == "Oui": activite_pro_lst = [1]
            else: activite_pro_lst = [0, 1]
            
            if st.session_state.w_scol_univ == "Non": scol_univ_lst = [0]
            elif st.session_state.w_scol_univ == "Oui": scol_univ_lst = [1]
            else: scol_univ_lst = [0, 1]
        
            if st.session_state.w_current_credit_cb == "Non": current_credit_cb_lst = [1]
            elif st.session_state.w_current_credit_cb == "Oui": current_credit_cb_lst = [2]
            else: current_credit_cb_lst = [1, 2]
            
            dict_etat_lst = {"Refusé": 0, "Validé": 1}
            etat_lst = [dict_etat_lst.get(val, np.nan) for val in st.session_state.w_etat]
            
            risque_all_loan_min, risque_all_loan_max = st.session_state.w_risque_all_loan
            nb_mens_min, nb_mens_max = st.session_state.w_nb_mens
            nb_cb_min, nb_cb_max = st.session_state.w_nb_cb
            risque_min, risque_max = st.session_state.w_risque_calc
            
            df_wip = df.loc[
                (df["Genre"].isin(genre_lst)) & 
                (df["Evaluation de la région d'habitation"].isin(st.session_state.w_eval_reg)) & 
                (df["Ecart adresse pro/perso"].isin(ecart_adr_lst)) & 
                (df["A une activité pro"].isin(activite_pro_lst)) & 
                (df["Scolarité universitaire"].isin(scol_univ_lst)) & 
                (df["Risque de l'ensemble des crédits"] >= risque_all_loan_min) & 
                (df["Risque de l'ensemble des crédits"] <= risque_all_loan_max) & 
                (df["Crédit CB en cours"].isin(current_credit_cb_lst)) & 
                (df["Nombre de mensualitées"] >= nb_mens_min) & 
                (df["Nombre de mensualitées"] <= nb_mens_max) & 
                (df["Nombre de CB pour retrait"] >= nb_cb_min) & 
                (df["Nombre de CB pour retrait"] <= nb_cb_max) & 
                (df["Risque en %"] >= risque_min/100) & 
                (df["Risque en %"] <= risque_max/100) & 
                (df["Etat"].isin(etat_lst))
                ].head(st.session_state.w_nombre_prospects)

            if int(st.session_state.id) not in df_wip["ID"].values:
                if df_wip.shape[0] == 0:
                    st.session_state.id = df["ID"].values[0]
                else:
                    st.session_state.id = df_wip["ID"].values[0]
                
        st.experimental_set_query_params(**st.session_state)
        return df_wip


def fPrintTable(df_table, seuil_risque_min, seuil_risque_max):
    """Génération du tableau d'affichage des dossier en  cours de traitement

    Args:
        df_table (DataFrame): Données à afficher
        seuil_risque_min (int): seuil minimum du risque intermédiaire pour affichage de la coloration dans le tableau
        seuil_risque_max (_type_): seuil maximum du risque intermédiaire pour affichage de la coloration dans le tableau

    Returns:
        string: tableau en format html
    """    
    # Feuille de style du tableau
    css_table = """
    <style>
        td,
        th {
        border: 1px solid rgb(190, 190, 190);
        padding: 10px;
        text-align: center;
        }

        td {
        text-align: center;
        }

        tr.low {
        background-color: #B4FFB4;
        }

        tr.middle {
        background-color: #FFD8B4;
        }

        tr.high {
        background-color: #FFB4B4;
        }

        tr.noHover {
        pointer-events: none;
        }
        
        a {
        text-decoration:none;
        }
        
        a:hover {
        text-decoration:none;
        }

        tr:hover {
        border: 5px solid rgb(255, 0, 0);
        font-weight: bold;
        }

        #tr:nth-child(even) {
        #background-color: #eee;
        #}

        th[scope='col'] {
        background-color: #696969;
        color: #fff;
        }

        th[scope='row'] {
        background-color: #d7d9f2;
        }
        
        th[scope='row'].selected {
        background-color: #f7d9f2;
        }

        caption {
        padding: 10px;
        caption-side: bottom;
        }

        table {
        border-collapse: collapse;
        border: 2px solid rgb(200, 200, 200);
        letter-spacing: 1px;
        font-family: Verdana, Arial, Helvetica, sans-serif;
        font-size: 0.7rem;
        }
    </style>
    """
    html_table = f"{css_table}<div><table><tr class='noHover'>"
    
    list_cols = list(df_table)
    
    
    for col in list_cols:
        html_table = f"{html_table}<th scope='col'>{col}</th>"
    html_table = f"{html_table}</tr>"
    
    for idx in df_table.index:
        id_value = df_table["ID"][idx]
            
        if df_table["Risque en %"][idx] == "":
            html_table = f"{html_table}<tr>"
        elif df_table["Risque en %"][idx] < seuil_risque_min/100:
            html_table = f"{html_table}<tr class='low'>"
        elif df_table["Risque en %"][idx] > seuil_risque_max/100:
            html_table = f"{html_table}<tr class='high'>"
        else:
            html_table = f"{html_table}<tr class='middle'>"
        for col in list_cols:
            value = df_table[col][idx]
            if col == "ID":
                if value == int(st.session_state.id):
                    html_table = f"{html_table}<th scope='row' class='selected'><span style='white-space: nowrap;'><a target='_self'' href='{fGetUrl('select', id_value)}'>&#128270;</a> {value}</span></th>" 
                else:
                    html_table = f"{html_table}<th scope='row'><span style='white-space: nowrap;'><a target='_self'' href='{fGetUrl('select', id_value)}'>&#128270;</a> {value}</span></th>" 
            else:                    
                if col == "Genre":
                    if int(value) == 1: value = "H"
                    else: value = "F"
                elif col in ["Ecart adresse pro/perso", "A une activité pro", "Scolarité universitaire"]:
                    if int(value) == 0: value = "Non"
                    else: value = "Oui"
                elif col in ["Crédit CB en cours"]:
                    if int(value) == 1: value = "Non"
                    else: value = "Oui"
                elif col in ["Evaluation de la région d'habitation"]:
                    value = int(value)
                elif col in ["Risque en %"]:
                    value = f"{value:.0%}"
                elif col == "Etat":
                    if value == 0: value = "&#10060;"
                    elif value == 1: value = "&#10004;"
                    else: value = f"<span style='white-space: nowrap;'><a target='_self'' href='{fGetUrl('refus', id_value)}'>&#10060;</a><a target='_self'' href='{fGetUrl('valid', id_value)}'>&#10004;</a></span>"
                    
                html_table = f"{html_table}<td>{value}</td>" 
        html_table = f"{html_table}</tr>"
    
    return f"{html_table}</table></div>"


def main():
    """
    Affichage de la page d'accueil et du tableau de bord
    """
    # Paramétrage du menu par défaut de streamlit
    st.set_page_config(
        page_title="Loan Validation",
        page_icon=":white_check_mark:",
        layout="wide",
        initial_sidebar_state="expanded",
        menu_items={
            #'Get Help': 'http://localhost:8501/',
            #'Report a bug': "http://localhost:8501/",
            'About': "# Application de validation des demandes de prêt"
        }
    )

    st.title(":white_check_mark: Application de validation des prêts")

    #Chargement et constitution des données de l'application
    df_clients = fReadDataFrameFile(os.path.join("data", "cleaned"), 'features_03')
    df_scors = fGetScoresFromMLModel(df_clients)

    if 'engine' not in globals(): 
        fCreateDataBase("sqlite:///loan_validation.db")

    if ~("df_prospects" in locals()):
        fCreateProspectDataFrame(df_clients, df_scors)
    
    df_prospects = pd.read_sql_query(sql=text("select * from prospects;"), con=engine)

    # Activation de la gestion des multipage de streamlit
    input_params = st.experimental_get_query_params()
    
    # Gestion des variables de session
    dossier_unique=False
    for item in input_params.items():
        key, value = item
        if key in ["w_genre", "w_ecart_adr", "w_activite_pro", "w_scol_univ", "w_current_credit_cb"]:
            if type(value[0]) is str:
                st.session_state[key] = value[0]
            else:
                st.session_state[key] = value
        elif key in ["w_eval_reg", "w_risque_all_loan", "w_nb_mens", "w_seuil_risque", "w_risque_calc", "w_nb_cb"]:
            st.session_state[key] = [int(v) for v in value]
        elif key in ["id"]:
            if type(value) is list:
                st.session_state[key] = value[0]
            else:
                st.session_state[key] = value
        elif key in ["w_dossier_unique"]:
            if type(value) is list:
                st.session_state[key] = eval(value[0])
            else:
                st.session_state[key] = eval(value)
            dossier_unique = st.session_state[key]
            
        elif key in ["w_etat"]:
            st.session_state[key] = [v for v in value]
        elif key in ["w_nombre_prospects"]:
            if type(value[0]) is int:
                st.session_state[key] = value
            else:
                st.session_state[key] = int(value[0])
        elif key in ["action"]:
            if value[0] == "refus":
                df_prospects.loc[df_prospects["ID"] == int(st.session_state.id), "Etat"] = 0.0
                with engine.begin() as connection:
                    connection.execute(text(f"update prospects set Etat=0.0 where ID={int(st.session_state.id)};"))
                
            elif value[0] == "valid":
                df_prospects.loc[df_prospects["ID"] == int(st.session_state.id), "Etat"] = 1.0
                with engine.begin() as connection:
                    connection.execute(text(f"update prospects set Etat=1.0 where ID={int(st.session_state.id)};"))
        else:
            st.session_state[key] = value

    df_wip = df_prospects.copy()
    # Initialisation des données
    if ("id" not in st.session_state) | ("w_seuil_risque" not in st.session_state) | ("w_dossier_unique" not in st.session_state):
        fReset_states(df_prospects)
    
    
    if "simul" in st.session_state:
        del st.session_state.simul
        if "calc_simul" in st.session_state:
            del st.session_state.calc_simul
            
    # Affichage du cartouche d'entête
    st.header(f":inbox_tray: Dossiers à traiter : {df_prospects.loc[df_prospects['Etat'].isna()].shape[0]}")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.subheader(f":heavy_check_mark: {df_prospects.loc[df_prospects['Etat'] == 1.0].shape[0]} dossiers validés")

    with col2:
        st.subheader(f":x: {df_prospects.loc[df_prospects['Etat'] == 0.0].shape[0]} dossiers refusés")

    with col3:
        st.subheader(f":dart: {df_prospects.loc[~df_prospects['Etat'].isna()].shape[0] / df_prospects.shape[0]:.0%} d'avancement")
    
    st.divider()

	# Menu de gestion des filtres    
    # Seuils du risque
    w_seuil_risque_min, w_seuil_risque_max = st.sidebar.slider("**Seuils de coloration du risque**", 0, 100, (8, 20))#, 
    st.session_state.w_seuil_risque = (w_seuil_risque_min, w_seuil_risque_max)
    
    dossier_unique = st.sidebar.toggle("**Accéder à un dossier spécifique**", value=dossier_unique)
    
    if dossier_unique:
        st.session_state.w_dossier_unique = True
        
        idx_dossier = list(df_prospects["ID"].values).index(int(st.session_state.id))
        
        st.session_state.id = st.sidebar.selectbox("Dossier n°", options=df_prospects, label_visibility="collapsed", index=idx_dossier) #, format_func=lambda x: x) #key="request_id"
        fReset_states(df_prospects, reset_id=False)
        
    else:
        if st.session_state.w_dossier_unique == True:
            st.session_state.w_dossier_unique = False
            fReset_states(df_prospects, reset_id=True)
            

        if (("w_risque_all_loan" in st.session_state) & ("w_nb_mens" in st.session_state) & ("w_nb_cb" in st.session_state) & 
            ("w_risque_calc" in st.session_state)):
            if ((len(st.session_state.w_risque_all_loan) != 2) | (len(st.session_state.w_nb_mens) != 2) | 
                (len(st.session_state.w_nb_cb) != 2) |  (len(st.session_state.w_risque_calc) != 2)):
                fReset_states(df_prospects, reset_id=False)
        else:
            fReset_states(df_prospects, reset_id=False)

#        st.write(st.session_state)
        
        list_choice_genre = ["Homme", "Femme", "Tous"]
        
        st.sidebar.radio("**Genre**", list_choice_genre, horizontal=True, 
                        on_change=fReloadPage, args=[df_prospects], 
                        key="w_genre")

        # Evaluation de la région d'habitation
        st.sidebar.multiselect("**Evaluation de la région d'habitation**", set(df_prospects["Evaluation de la région d'habitation"].astype(int)), 
                            on_change=fReloadPage, args=[df_prospects], 
                            key="w_eval_reg")

		# Ecart adresse pro/perso
        list_choice_not = ["Non", "Oui", "Tous"]
        st.sidebar.radio("**Ecart adresse pro/perso**", list_choice_not, horizontal=True, 
                        on_change=fReloadPage, args=[df_prospects], 
                        key="w_ecart_adr")
        
        list_choice_not = ["Non", "Oui", "Tous"]
        st.sidebar.radio("**A une activité pro**", list_choice_not, horizontal=True, 
                        on_change=fReloadPage, args=[df_prospects], 
                        key="w_activite_pro")
        list_choice_not = ["Non", "Oui", "Tous"]
        st.sidebar.radio("**Scolarité universitaire**", list_choice_not, horizontal=True, 
                        on_change=fReloadPage, args=[df_prospects], 
                        key="w_scol_univ")
        st.sidebar.slider("**Niveau de risque de l'ensemble des crédits**", 
                        min(df_prospects["Risque de l'ensemble des crédits"]), max(df_prospects["Risque de l'ensemble des crédits"]), 
                        on_change=fReloadPage, args=[df_prospects], 
                        key="w_risque_all_loan")
        list_choice_not = ["Non", "Oui", "Tous"]
        st.sidebar.radio("**Crédit CB en cours**", list_choice_not, horizontal=True, 
                        on_change=fReloadPage, args=[df_prospects], 
                        key="w_current_credit_cb")
        st.sidebar.slider("**Nombre de mensualitées**", 
                        min(df_prospects["Nombre de mensualitées"]), max(df_prospects["Nombre de mensualitées"]),
                        on_change=fReloadPage, args=[df_prospects], 
                        key="w_nb_mens")

		# Nombre de CB pour retrait
        st.sidebar.slider("**Nombre de CB pour retrait**", 
                        min(df_prospects["Nombre de CB pour retrait"]), max(df_prospects["Nombre de CB pour retrait"]),
                        on_change=fReloadPage, args=[df_prospects], 
                        key="w_nb_cb")
        st.sidebar.slider("**Risque en %**", int(min(df_prospects["Risque en %"]*100)), int(max(df_prospects["Risque en %"]*100+1)), 
                        on_change=fReloadPage, args=[df_prospects], 
                        key="w_risque_calc")
        st.sidebar.multiselect("**Etat des demandes**", ["A traiter", "Refusé", "Validé"], 
                            on_change=fReloadPage, args=[df_prospects], 
                            key="w_etat")
        st.sidebar.number_input("**Nombre de dossier à afficher**", min_value=1, max_value=df_prospects.shape[0],
                                on_change=fReloadPage, args=[df_prospects], 
                                key="w_nombre_prospects")
        st.sidebar.button("Réinitialisation des filtres", on_click=fReset_states, args=[df_prospects])

    # Application des filtres
    df_wip = fReloadPage(df_prospects)
    
    # Affichage du tableau
    st.markdown(fPrintTable(df_wip.copy(), w_seuil_risque_min, w_seuil_risque_max), unsafe_allow_html=True)
    
    st.divider()
    
    # Affichage des explication SHAP du dossier en cours    
    if ("id" in st.session_state) & (int(st.session_state.id) > 0):
        shap_values = fGetRessources(os.path.join("data", "cleaned", "shap_values.pkl"))[...,1]
        shap.initjs()
        
        if "ID" in list(df_clients):
            id_shap = list(df_clients["ID"].astype(int)).index(int(st.session_state.id))
        else:
            id_shap = list(df_clients.index).index(int(st.session_state.id))
            
        st.set_option('deprecation.showPyplotGlobalUse', False)
        
        fig = shap.plots.waterfall(shap_values[id_shap])
        plt.gcf().set_size_inches(8, 4)
        plt.tight_layout()
        st.pyplot(fig)
               
    
if __name__ == '__main__':
    main()
    
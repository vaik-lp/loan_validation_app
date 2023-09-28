import os
import pandas as pd
import numpy as np
import streamlit as st
from sqlalchemy import text
from sqlalchemy import create_engine


def fCreateDataBase(url_db):
    """Chargement et mise en cache de la base de donnée popur ne la charger qu'une seule fois

    Args:
        url_db (string): url d'accès à la base de données. 
    """    
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
        DataFrame: Données chargées
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


# Paramétrage du menu par défaut de streamlit
st.set_page_config(
    page_title="Loan Validation",
    page_icon=":white_check_mark:",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'About': "# Application de validation des demandes de prêt"
    }
)


#Chargement et constitution des données de l'application
if 'engine' not in globals(): 
    fCreateDataBase("sqlite:///loan_validation.db")

# Récupération des données du dossier en cours d'étude
df_current = pd.read_sql_query(sql=f"select * from prospects where ID={st.session_state.id};", con=engine)
_, genre, eval_reg, ecart_adr, activ_pro, scol_univ, risque_all_cred, credit_cb, nb_mens, nb_cb, risque_calc, etat = df_current.iloc[0]

# Affichage du cartouche d'entête
st.title(":white_check_mark: Application de validation des prêts")
st.header(f":1234: Détail des données du demandeur n° *{st.session_state.id}*")
st.subheader(f"*Risque de défaut de paiement de {risque_calc:.0%}*")
st.divider()

if genre == 0: genre = "Femme"
else: genre = "Homme"

if ecart_adr == 0: ecart_adr = "Non"
else: ecart_adr = "Oui"

if activ_pro == 0: activ_pro = "Non"
else: activ_pro = "Oui"

if scol_univ == 0: scol_univ = "Non"
else: scol_univ = "Oui"


col1, col2, col3 = st.columns(3)
with col1:
    st.markdown(f"<p style='font-size:20px;'><b>Genre :</b> <i>{genre}</i></p>", unsafe_allow_html=True)
    st.markdown(f"<p style='font-size:20px;'><b>A une activité pro :</b> <i>{activ_pro}</i></p>", unsafe_allow_html=True)
    st.markdown(f"<p style='font-size:20px;'><b>Crédit CB en cours :</b> <i>{credit_cb}</i></p>", unsafe_allow_html=True)

with col2:
    st.markdown(f"<p style='font-size:20px;'><b>Evaluation de la région d'habitation :</b> <i>{int(eval_reg)}</i></p>", unsafe_allow_html=True)
    st.markdown(f"<p style='font-size:20px;'><b>Scolarité universitaire :</b> <i>{scol_univ}</i></p>", unsafe_allow_html=True)
    st.markdown(f"<p style='font-size:20px;'><b>Nombre de mensualités :</b> <i>{nb_mens}</i></p>", unsafe_allow_html=True)

with col3:
    st.markdown(f"<p style='font-size:20px;'><b>Ecart adresse pro/perso :</b> <i>{ecart_adr}</i></p>", unsafe_allow_html=True)
    st.markdown(f"<p style='font-size:20px;'><b>Risque de l'ensemble des crédits :</b> <i>{risque_all_cred}</i></p>", unsafe_allow_html=True)
    st.markdown(f"<p style='font-size:20px;'><b>Nombre de CB pour retrait :</b> <i>{nb_cb}</i></p>", unsafe_allow_html=True)

st.divider()

df_application = fReadDataFrameFile(os.path.join("data", "source"), 'application_test', use_dict=False, sep=',')
df_bureau = fReadDataFrameFile(os.path.join("data", "source"), 'bureau', use_dict=False, sep=',')
df_cash = fReadDataFrameFile(os.path.join("data", "source"), 'POS_CASH_balance', use_dict=False, sep=',')
df_credit = fReadDataFrameFile(os.path.join("data", "source"), 'credit_card_balance', use_dict=False, sep=',')
df_previous = fReadDataFrameFile(os.path.join("data", "source"), 'previous_application', use_dict=False, sep=',')
df_installments = fReadDataFrameFile(os.path.join("data", "source"), 'installments_payments', use_dict=False, sep=',')

# Menu de gestion des filtres    
file_choice = st.sidebar.radio("Données", ["Application", "Bureau", "Cash", "Credit", "Previous", "Installments payments"])

if file_choice == "Application":
    st.dataframe(df_application.loc[df_application["SK_ID_CURR"] == int(st.session_state.id)].set_index("SK_ID_CURR").T)
elif file_choice == "Bureau":
    st.dataframe(df_bureau.loc[df_bureau["SK_ID_CURR"] == int(st.session_state.id)].T)
elif file_choice == "Cash":
    st.dataframe(df_cash.loc[df_cash["SK_ID_CURR"] == int(st.session_state.id)].T)
elif file_choice == "Credit":
    st.dataframe(df_credit.loc[df_credit["SK_ID_CURR"] == int(st.session_state.id)].T)
elif file_choice == "Previous":
    st.dataframe(df_previous.loc[df_previous["SK_ID_CURR"] == int(st.session_state.id)].T)
elif file_choice == "Installments payments":
    st.dataframe(df_installments.loc[df_installments["SK_ID_CURR"] == int(st.session_state.id)].T)

import os
import pandas as pd
import numpy as np
import streamlit as st
from sqlalchemy import text
from sqlalchemy import create_engine
import pickle
from xgboost import XGBClassifier
import requests


REST_URL = "http://localhost:8000/"

def fCreateDataBase(url_db):
    """Chargement et mise en cache de la base de donnée popur ne la charger qu'une seule fois

    Args:
        url_db (string): url d'accès à la base de données. 
    """    
    global engine
    engine = create_engine(url_db)
    

def fReloadPage():
    """Gestion et mapping des données à transmettre entre pages

    Args:
        df (DataFrame): Données des dossiers en cours d'utilisation/manipulation

    Returns:
        DataFrame: Données filtrées selon l'utilsiation des widgets 
    """    
    if len(st.session_state.keys()) > 0:
        #print(st.session_state)
        st.experimental_set_query_params(**st.session_state)
        

def fRender_message_score():
    """Appel au web service de simulation du score de risque
    """    
    fReloadPage()
    
    if st.session_state.w_genre == "Femme": ws_genre = 0
    else: ws_genre = 1

    if st.session_state.w_ecart_adr == "Non": ws_ecart_adr = 0
    else: ws_ecart_adr = 1

    if st.session_state.w_activite_pro == "Non": ws_activite_pro = 0
    else: ws_activite_pro = 1

    if st.session_state.w_scol_univ == "Non": ws_scol_univ = 0
    else: ws_scol_univ = 1
    
    params = {"genre": ws_genre,
              "eval_region": st.session_state.w_eval_reg,
              "ecart_adr": ws_ecart_adr,
              "activite_pro": ws_activite_pro, 
              "scol_univ": ws_scol_univ, 
              "risque_all_loan": st.session_state.w_risque_all_loan, 
              "current_credit_cb": st.session_state.w_current_credit_cb, 
              "nb_mens" : st.session_state.w_nb_mens, 
              "nb_cb": st.session_state.w_nb_cb}
    print(params)
    
    resp = requests.get(REST_URL + "risk_simulation", params=params).json()
    
    print(resp)
    if resp["status"] == "success":
        st.success(f"La nouvelle simulation, en ligne, donne un risque de défaut de paiement de **{resp['score_simulation']}%**.")
    else:
        st.error(f"{resp['message']}")
    


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
_, genre, eval_reg, ecart_adr, activite_pro, scol_univ, risque_all_loan, current_credit_cb, nb_mens, nb_cb, risque_calc, _ = df_current.iloc[0]

if genre == 0: genre = "Femme"
else: genre = "Homme"

if ecart_adr == 0: ecart_adr = "Non"
else: ecart_adr = "Oui"

if activite_pro == 0: activite_pro = "Non"
else: activite_pro = "Oui"

if scol_univ == 0: scol_univ = "Non"
else: scol_univ = "Oui"

# Gestion des variables de session
if (("simul" not in st.session_state) | ("w_genre" not in st.session_state) | ("w_eval_reg" not in st.session_state) | 
    ("w_ecart_adr" not in st.session_state) | ("w_activite_pro" not in st.session_state) | ("w_scol_univ" not in st.session_state) | 
    ("w_risque_all_loan" not in st.session_state) | ("w_current_credit_cb" not in st.session_state) | ("w_nb_mens" not in st.session_state) | 
    ("w_nb_cb" not in st.session_state)):
    _, st.session_state.w_genre, st.session_state.w_eval_reg, st.session_state.w_ecart_adr, st.session_state.w_activite_pro, st.session_state.w_scol_univ, st.session_state.w_risque_all_loan, st.session_state.w_current_credit_cb, st.session_state.w_nb_mens, st.session_state.w_nb_cb, _, _ = df_current.iloc[0]
    st.session_state.simul = 0

    if st.session_state.w_genre == 0: st.session_state.w_genre = "Femme"
    else: st.session_state.w_genre = "Homme"

    if st.session_state.w_ecart_adr == 0: st.session_state.w_ecart_adr = "Non"
    else: st.session_state.w_ecart_adr = "Oui"

    if st.session_state.w_activite_pro == 0: st.session_state.w_activite_pro = "Non"
    else: st.session_state.w_activite_pro = "Oui"

    if st.session_state.w_scol_univ == 0: st.session_state.w_scol_univ = "Non"
    else: st.session_state.w_scol_univ = "Oui"


# Affichage du cartouche d'entête
st.title(":white_check_mark: Application de validation des prêts")
st.header(f":arrows_counterclockwise: Adaptation de la simulation du demandeur n° *{st.session_state.id}*")
st.subheader(f"*Risque de défaut de paiement de {risque_calc:.0%}*")
st.divider()

col1, col2, col3 = st.columns(3)
with col1:
    st.markdown(f"<p style='font-size:20px;'><b>Genre :</b> <i>{genre}</i></p>", unsafe_allow_html=True)
    st.markdown(f"<p style='font-size:20px;'><b>A une activité pro :</b> <i>{activite_pro}</i></p>", unsafe_allow_html=True)
    st.markdown(f"<p style='font-size:20px;'><b>Crédit CB en cours :</b> <i>{current_credit_cb}</i></p>", unsafe_allow_html=True)

with col2:
    st.markdown(f"<p style='font-size:20px;'><b>Evaluation de la région d'habitation :</b> <i>{int(eval_reg)}</i></p>", unsafe_allow_html=True)
    st.markdown(f"<p style='font-size:20px;'><b>Scolarité universitaire :</b> <i>{scol_univ}</i></p>", unsafe_allow_html=True)
    st.markdown(f"<p style='font-size:20px;'><b>Nombre de mensualités :</b> <i>{nb_mens}</i></p>", unsafe_allow_html=True)

with col3:
    st.markdown(f"<p style='font-size:20px;'><b>Ecart adresse pro/perso :</b> <i>{ecart_adr}</i></p>", unsafe_allow_html=True)
    st.markdown(f"<p style='font-size:20px;'><b>Risque de l'ensemble des crédits :</b> <i>{risque_all_loan}</i></p>", unsafe_allow_html=True)
    st.markdown(f"<p style='font-size:20px;'><b>Nombre de CB pour retrait :</b> <i>{nb_cb}</i></p>", unsafe_allow_html=True)

st.divider()

# Formulaire de saisie des valeurs de simulation de la modélisation
col1, col2, col3 = st.columns(3)
with col1:
    st.radio("**Genre**", ["Homme", "Femme"], horizontal=True, on_change=fReloadPage, key="w_genre")
    if st.session_state.w_genre == "Homme": genre_simul = 1
    if st.session_state.w_genre == "Femme": genre_simul = 0
    
    st.radio("**A une activité pro**", ["Non", "Oui"], horizontal=True, on_change=fReloadPage, key="w_activite_pro")
    if st.session_state.w_activite_pro == "Non": activite_pro_simul = 0
    if st.session_state.w_activite_pro == "Oui": activite_pro_simul = 1

    st.number_input("**Crédit CB en cours**", min_value=0, on_change=fReloadPage, key="w_current_credit_cb")

with col2:
    st.number_input("**Evaluation de la région d'habitation**", min_value=1, on_change=fReloadPage, key="w_eval_reg")

    st.radio("**Scolarité universitaire**", ["Non", "Oui"], horizontal=True, on_change=fReloadPage, key="w_scol_univ")
    if st.session_state.w_scol_univ == "Non": scol_univ_simul = 0
    if st.session_state.w_scol_univ == "Oui": scol_univ_simul = 1

    st.number_input("**Nombre de mensualités**", min_value=0, on_change=fReloadPage, key="w_nb_mens")

with col3:
    st.radio("**Ecart adresse pro/perso**", ["Non", "Oui"], horizontal=True, on_change=fReloadPage, key="w_ecart_adr")
    if st.session_state.w_ecart_adr == "Non": ecart_adr_simul = 0
    if st.session_state.w_ecart_adr == "Oui": ecart_adr_simul = 1

    st.number_input("**Risque de l'ensemble des crédits**", min_value=0, on_change=fReloadPage, key="w_risque_all_loan")

    st.number_input("**Nombre de CB pour retrait**", min_value=0, on_change=fReloadPage, key="w_nb_cb")

st.write(" ")

col4, col5 = st.columns(2)
# création bouton
with col4:
    st.button(":arrows_counterclockwise: Lancer la simulation en local", key="calc_simul")
    resultat = st.button(":arrows_counterclockwise: Lancer la simulation en ligne")
    if resultat:
        fRender_message_score()

with col5:
    # Reinitialisation des valeurs par défaut
    if st.button("Réinitialisation des valeurs par défaut"):
        del st.session_state.simul
        st.experimental_rerun()

if st.session_state.calc_simul:
    # calcul de la simulation par appel au model chargé en local
    model = fGetRessources(os.path.join("data", "cleaned", "xgboost_model.pkl"))
    
    current_simul = pd.DataFrame([[genre_simul, st.session_state.w_eval_reg, ecart_adr_simul, activite_pro_simul, scol_univ_simul, st.session_state.w_risque_all_loan, st.session_state.w_current_credit_cb, st.session_state.w_nb_mens, st.session_state.w_nb_cb]], 
                                        columns=["Genre", "Evaluation de la région d'habitation", "Ecart adresse pro/perso", "A une activité pro", "Scolarité universitaire", "Risque de l'ensemble des crédits", "Crédit CB en cours", "Nombre de mensualitées", "Nombre de CB pour retrait"])
        
    score = model.predict_proba(current_simul)
#   
    resultat = f"La nouvelle simulation donne un risque de défaut de paiement de **{score[0][1]:.0%}**."
    st.success(resultat)
        
st.divider()


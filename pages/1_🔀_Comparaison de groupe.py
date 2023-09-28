import os
import pandas as pd
import numpy as np
import streamlit as st
import streamlit.components.v1 as components
from sqlalchemy import text
from sqlalchemy import create_engine
import pickle
from sklearn.neighbors import NearestNeighbors
import shap
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

from sklearn.metrics import confusion_matrix, f1_score, fbeta_score, adjusted_rand_score, accuracy_score, auc, roc_auc_score, roc_curve, make_scorer, classification_report, fbeta_score
from xgboost import XGBClassifier
from sklearn.model_selection import KFold, cross_validate, cross_val_score, train_test_split, GridSearchCV



@st.cache_data
def fReadDataFrameFile(path, file, use_dict=False, encoding='utf8', sep=';'):
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


@st.cache_data
def fReadBinaryDataFrame(path_file):
    if os.path.exists(path_file):
        return pd.read_pickle(open(path_file, "rb"))


def fCreateDataBase(url_db):
    global engine
    engine = create_engine(url_db)


def fReloadPage():
    if len(st.session_state.keys()) > 0:
        st.experimental_set_query_params(**st.session_state)


@st.cache_resource
def fGetRessources(path_file):
    if os.path.exists(path_file):
        return pickle.load(open(path_file, "rb"))


def st_shap(plot, height=None):
    shap_html = f"<head>{shap.getjs()}</head><body>{plot.html()}</body>"
    components.html(shap_html, height=height)


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
    return custom_scoring(y_true, y_pred, beta=2)


custom_scorer = make_scorer(custom_scoring_min_fn, greater_is_better=True, needs_proba=False)


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


if "wc_groupe" not in st.session_state:
    st.session_state.wc_groupe = 100
else:
    if type(st.session_state.wc_groupe) is list:
        st.session_state.wc_groupe = int(st.session_state.wc_groupe[0])

st.sidebar.slider("**Nombre de voisins**", 1, 100, key="wc_groupe")


file_choice = st.sidebar.radio("Vision", ["Globale", "Explicative", "Comparative"])

if st.sidebar.button("Réinitialisation des filtres."):
    del st.session_state.wc_groupe
    st.experimental_rerun()


st.title(":white_check_mark: Application de validation des prêts")
st.header(f":twisted_rightwards_arrows: Les données du demandeur n° *{st.session_state.id}* par rapport aux demandes les plus proches")
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

df = fReadDataFrameFile(os.path.join("data", "cleaned"), 'features_03')
df['SK_ID_CURR'] = df['SK_ID_CURR'].astype(int)
df.set_index('SK_ID_CURR', inplace=True)
target = df["TARGET"]
df = df[list_of_best_features].rename(columns=dict_rename_features)
df["Risque en %"] = target

knn = fGetRessources(os.path.join("data", "cleaned", "knn_model.pkl"))

df_voisins = pd.DataFrame([], columns=["Distances", "Voisins_idx"])

distances, voisins_idx = knn.kneighbors(df_current.iloc[:, 1:-2], n_neighbors=int(st.session_state.wc_groupe), return_distance=True)


df_voisins["Distances"] = distances[0]
df_voisins["Voisins_idx"] = voisins_idx[0]
df_voisins["id"] = df.index[df_voisins["Voisins_idx"].values]
df_voisins.sort_values(by=["Distances"], inplace=True)

html_value = ""
cpt = 0
for idx in df_voisins["id"].values:
    if (cpt % 12) == 0:
        html_value = f"{html_value}</tr><tr style='border:0px'>"
    html_value = f"{html_value}<td style='border:0px'>{str(idx)} ({cpt:03d})</td>"
    cpt += 1
html_value = f"<b>Liste des plus proches voisins</b><br><table><tr style='border:0px'>{html_value}</tr></table>"
    
st.markdown(html_value, unsafe_allow_html=True)

st.divider()
    
    
if file_choice == "Explicative":
    shap_values = fGetRessources(os.path.join("data", "cleaned", "shap_values.pkl"))[...,1]
    
    st_shap(shap.plots.force(shap_values[df_voisins["Voisins_idx"].values]), 400)

elif file_choice == "Comparative":
    
    df_wip = df.iloc[df_voisins["Voisins_idx"].values].copy()
    df_wip = pd.concat([df_current.set_index("ID").iloc[:, :-1], df_wip])
    for col in list(df_wip)[:-1]:
        min_value = df[col].min()
        max_delta_value = df[col].max() - min_value
        if max_delta_value > 0:
            df_wip[col] = df_wip[col].apply(lambda x: float(x) - float(min_value))
            df_wip[col] = df_wip[col].apply(lambda x: float(x)/float(max_delta_value))
        else:
            df_wip[col] = 0
            
    subjects=list(df_wip)

    df_wip["Genre_"] = df_wip["Genre"]
    
    plt.style.use('seaborn-v0_8-whitegrid')
    
    angles=np.linspace(0,2*np.pi,len(subjects), endpoint=False)
    angles=np.concatenate((angles,[angles[0]]))

    subjects.append(subjects[0])
    
    colonnes = 2
    lignes = (df_wip.shape[0] - 1) // colonnes
    if  ((df_wip.shape[0] - 1) % colonnes) != 0:
        lignes += 1
    
    fig = plt.figure(figsize=(7 * colonnes, 6 * lignes))    
    plt.gcf().subplots_adjust(left=0.05,
                          bottom=0.05,
                          right=0.95,
                          top=0.95,
                          wspace=0.05,
                          hspace=0.05)


    category_colors = plt.get_cmap('jet')(np.linspace(0.15, 0.85, 2))

    for index_axe in range(df_wip.shape[0] - 1):
        
        ax = fig.add_subplot(lignes, colonnes, (index_axe + 1), polar=True)

        df_4_plot = df_wip.T.iloc[:, [0, index_axe + 1]]
        
        ax.plot(angles, df_4_plot.iloc[:, [0]].values, 'o-', color=category_colors[0], linewidth=1, label=list(df_4_plot)[0], zorder=0)
        ax.plot(angles, df_4_plot.iloc[:, [1]].values, 'o-', color=category_colors[1], linewidth=1, label=list(df_4_plot)[1], zorder=1)
        
        ax.set_ylim(ymin=-0.1)
        ax.set_ylim(ymax=1.0)
        ax.set_xticks(angles)
        ax.set_xticklabels('')

        for angle in angles:

            if 0 <= angle * 180 / np.pi < 90:
                ha, distance_ax = 'left', 1.05
            elif angle * 180 / np.pi == 90:
                ha, distance_ax = 'center', 1.05
            elif 90 < (angle * 180 / np.pi) < 270:
                ha, distance_ax = 'right', 1.05
            elif angle * 180 / np.pi == 270:
                ha, distance_ax = 'center', 1.05
            else:
                ha, distance_ax = 'left', 1.05    

            ax.text(angle, .05 + distance_ax, subjects[list(angles).index(angle)], 
                    **{'size': 8, 'horizontalalignment': ha, 'verticalalignment': 'center', 'fontweight': 'bold'})


        plt.grid(True)
        plt.tight_layout()
        plt.legend(loc='upper right', bbox_to_anchor=(1.65,1.15))
        plt.title(f"{index_axe:03d} : {list(df_4_plot)[0]} vs {list(df_4_plot)[1]}", 
                **{'size': 14, 'verticalalignment': 'top', 'fontweight': 'bold', 'y': 1.15})
        
    st.pyplot(fig)

    

elif file_choice == "Globale":

    df_tsne = fReadBinaryDataFrame(os.path.join("data", "cleaned", "tsne_data.pkl"))
        
    df_current_tsne = df_tsne.loc[[int(st.session_state.id)]]
    df_current_tsne["TARGET"] = round(risque_calc, 2)
    df_tmp = df_tsne.iloc[df_voisins["Voisins_idx"].values].copy()
    df_tmp = pd.concat([df_current_tsne, df_tmp])
    
    df_tmp["TARGET"] = df_tmp["TARGET"].astype(str)
    df_tmp["ID"] = df_tmp.index
    df_tmp["Genre"] = df_tmp["Genre"].apply(lambda x: "Homme" if x == 1 else "Femme")
    df_tmp["Ecart adresse pro/perso"] = df_tmp["Ecart adresse pro/perso"].apply(lambda x: "Oui" if x == 1 else "Non")
    df_tmp["A une activité pro"] = df_tmp["A une activité pro"].apply(lambda x: "Oui" if x == 1 else "Non")
    df_tmp["Scolarité universitaire"] = df_tmp["Scolarité universitaire"].apply(lambda x: "Oui" if x == 1 else "Non")

    
    fig_scatter = px.scatter(
        data_frame=df_tmp,
        x="tsne_1", y="tsne_2",
        width=1200, height=600,
        color="TARGET", color_discrete_sequence=["orange", "green", "red"],
        hover_data=["ID", "Genre", "Evaluation de la région d'habitation", "Ecart adresse pro/perso", "A une activité pro", "Scolarité universitaire",
                    "Risque de l'ensemble des crédits", "Crédit CB en cours", "Nombre de mensualitées", "Nombre de CB pour retrait"],
        title=f"Distances des {st.session_state.wc_groupe} voisin(s) de {st.session_state.id}"
    )
    
    fig_scatter.update_layout(hovermode="closest")


    st.plotly_chart(fig_scatter)


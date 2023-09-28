"""Webservice réalisant une simulation de risque et restituant un score de ce risque"""
import os
import pandas as pd
import numpy as np
import pickle
from fastapi import FastAPI
from xgboost import XGBClassifier


# Load the model
current_dir = os.path.dirname(__file__)
model_file = os.path.join(current_dir, "data", "cleaned", "xgboost_model.pkl")
if os.path.exists(model_file):
    model = pickle.load(open(model_file, "rb"))

# Define the application
app = FastAPI()


@app.get("/risk_simulation/")
def risk_simulation(genre: int = 0, eval_region: int = 1, ecart_adr: int = 0, activite_pro: int = 0, scol_univ: int = 0, 
                    risque_all_loan: int = 0, current_credit_cb: int = 0, nb_mens: int = 0, nb_cb: int = 0) -> dict:
    """_summary_

    Args:
        genre (int, optional): Genre de la personne, 1 pour homme, 0 pour femme. Defaults to 0.
        eval_region (int, optional): Evaluation du niveau de risque de la région noté de 1 à 3. Defaults to 1.
        ecart_adr (int, optional): Ecart entre l'adresse personnelle et profesionnelle. Defaults to 0.
        activite_pro (int, optional): A une activité profesionnelle. Defaults to 0.
        scol_univ (int, optional): Est allé à l'université. Defaults to 0.
        risque_all_loan (int, optional): Niveau de risque de l'ensemble des crédits. Defaults to 0.
        current_credit_cb (int, optional): Crédit CB en cours. Defaults to 0.
        nb_mens (int, optional): Nombre de mensualités. Defaults to 0.
        nb_cb (int, optional): Nombre de CB pour retrait. Defaults to 0.

    Raises:
        ValueError: Paramètres hors de la plage de données possible

    Returns:
        dict: Dictionnaire donnant le status de la réponse, le score et le message associé aux erreurs
    """    
    status = "success"
    message = ""
    score_simulation = 999

    
    try: 
        if int(genre) not in [0, 1]:
            current_message = "La donnée genre doit être 0 ou 1."
            if message == "" : message = current_message 
            else: message = " | ".join([message, current_message])
            
        elif int(eval_region) not in [1, 2, 3]:
            current_message = "La donnée eval_region doit être entre 1 et 3."
            if message == "" : message = current_message 
            else: message = " | ".join([message, current_message])

        elif int(ecart_adr) not in [0, 1]:
            current_message = "La donnée ecart_adr doit être 0 ou 1."
            if message == "" : message = current_message 
            else: message = " | ".join([message, current_message])
        elif int(activite_pro) not in [0, 1]:
            current_message = "La donnée activite_pro doit être 0 ou 1."
            if message == "" : message = current_message 
            else: message = " | ".join([message, current_message])

        elif int(scol_univ) not in [0, 1]:
            current_message = "La donnée scol_univ doit être 0 ou 1."
            if message == "" : message = current_message 
            else: message = " | ".join([message, current_message])
            
        elif int(risque_all_loan) not in range(31):
            current_message = "La donnée risque_all_loan doit être entre 0 et 30."
            if message == "" : message = current_message 
            else: message = " | ".join([message, current_message])
            
        elif int(current_credit_cb) not in [0, 1]:
            current_message = "La donnée current_credit_cb doit être 0 ou 1."
            if message == "" : message = current_message 
            else: message = " | ".join([message, current_message])
        elif int(nb_mens) not in range(31):
            current_message = "La donnée nb_mens doit être entre 0 et 30."
            if message == "" : message = current_message 
            else: message = " | ".join([message, current_message])

        elif int(nb_cb) not in range(31):
            current_message = "La donnée nb_cb doit être entre 0 et 30."
            if message == "" : message = current_message 
            else: message = " | ".join([message, current_message])
            
        if message != "":
            raise ValueError() 

        current_simul = pd.DataFrame([[genre, eval_region, ecart_adr, activite_pro, scol_univ, risque_all_loan, current_credit_cb, nb_mens, nb_cb]], 
                                            columns=["Genre", "Evaluation de la région d'habitation", "Ecart adresse pro/perso", "A une activité pro", "Scolarité universitaire", "Risque de l'ensemble des crédits", "Crédit CB en cours", "Nombre de mensualitées", "Nombre de CB pour retrait"])
            
        score = model.predict_proba(current_simul)
        
    except Exception as e: 
        status = "fail"
        message = " | ".join([message, "Une erreur a été détectée !"])
        score = [[0, 999]]
    
    return {"status": status, "score_simulation": int(np.round(score[0][1]*100, 0)), "message": message}
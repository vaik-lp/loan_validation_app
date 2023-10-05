"""Classe de test pour le fichier risk_simulation.py"""
#import os
#import pandas as pd
#import numpy as np
#import pickle
#from fastapi import FastAPI
#from xgboost import XGBClassifier

import unittest

import risk_simulation as app


class Test(unittest.TestCase):
    
    def setUp(self):
        print( "\nLancement du test pour le fichier risk_simulation.py" )

    def tearDown(self):
        print( "Fin des tests pour le fichier risk_simulation.py" )

    def test_valide(self, genre: int = 0, eval_region: int = 1, ecart_adr: int = 0, activite_pro: int = 0, scol_univ: int = 0, 
                    risque_all_loan: int = 0, current_credit_cb: int = 0, nb_mens: int = 0, nb_cb: int = 0):
        
        score = app.risk_simulation(genre=genre, eval_region=eval_region, ecart_adr=ecart_adr, activite_pro=activite_pro, scol_univ=scol_univ, 
                                risque_all_loan=risque_all_loan, current_credit_cb=current_credit_cb, nb_mens=nb_mens, nb_cb=nb_cb)

        assert score.get("status") == "success"
        assert type(score.get("score_simulation")) == int
        assert score.get("score_simulation") in range(101)
        assert score.get("message") == ""
        


    #@unittest.expectedFailure
    def testBadParameters(self, genre: int = 2, eval_region: int = 0, ecart_adr: int = 2, activite_pro: int = 2, scol_univ: int = 2, 
                    risque_all_loan: int = 100, current_credit_cb: int = 100, nb_mens: int = 100, nb_cb: int = 100):
        
        score = app.risk_simulation(genre=genre, eval_region=eval_region, ecart_adr=ecart_adr, activite_pro=activite_pro, scol_univ=scol_univ, 
                                risque_all_loan=risque_all_loan, current_credit_cb=current_credit_cb, nb_mens=nb_mens, nb_cb=nb_cb)
        
        assert score.get("status") == "fail"
        assert score.get("score_simulation") == 999
        assert score.get("message") != ""

    

if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()

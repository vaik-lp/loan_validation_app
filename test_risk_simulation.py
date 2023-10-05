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
        

    def test_valide_parameters(self):
        print("test_valide_parameters ==> ", end="\r")
        test_cpt = 0
        for genre in range(2):
            for eval_region in range(1, 4):
                for ecart_adr in range(2):
                    for activite_pro in range(2):
                        for scol_univ in range(2):
                            for risque_all_loan in [0, 10, 20, 30]:
                                for current_credit_cb in [0, 10, 20, 30]:
                                    for nb_mens in [0, 10, 20, 30]:
                                        for nb_cb in [0, 10, 20, 30]:
                                            self.test_valide(genre=genre, eval_region=eval_region, ecart_adr=ecart_adr, 
                                                             activite_pro=activite_pro, scol_univ=scol_univ, 
                                                            risque_all_loan=risque_all_loan, current_credit_cb=current_credit_cb, 
                                                            nb_mens=nb_mens, nb_cb=nb_cb)
                                            test_cpt += 1
                                            print(f"test_valide_parameters ==> Avancement : {test_cpt/(2*3*2*2*2*30*30*30*30):.0%}", end="\r")

    #@unittest.expectedFailure
    def testBadParameters(self, genre: int = 2, eval_region: int = 0, ecart_adr: int = 2, activite_pro: int = 2, scol_univ: int = 2, 
                    risque_all_loan: int = 100, current_credit_cb: int = 100, nb_mens: int = 100, nb_cb: int = 100):
        
        score = app.risk_simulation(genre=genre, eval_region=eval_region, ecart_adr=ecart_adr, activite_pro=activite_pro, scol_univ=scol_univ, 
                                risque_all_loan=risque_all_loan, current_credit_cb=current_credit_cb, nb_mens=nb_mens, nb_cb=nb_cb)
        
        assert score.get("status") == "fail"
        assert score.get("score_simulation") == 999
        assert score.get("message") != ""

    def test_invalide_parameters(self):
        
        print("test_invalide_parameters ==> ", end="\r")
        test_cpt = 0
        for genre in [-1, 0.5, 2]:
            for eval_region in [-1, 0, 1, 0.5, 5]:
                for ecart_adr in [-1, 0, 0.5, 2]:
                    for activite_pro in [-1, 0, 0.5, 2]:
                        for scol_univ in [-1, 0, 0.5, 2]:
                            for risque_all_loan in [-1, 0.5, 1, 31]:
                                for current_credit_cb in [-1, 0.5, 1, 31]:
                                    for nb_mens in [-1, 0.5, 1, 31]:
                                        for nb_cb in [-1, 0.5, 1, 31]:
                                            self.testBadParameters(genre=genre, eval_region=eval_region, ecart_adr=ecart_adr, 
                                                             activite_pro=activite_pro, scol_univ=scol_univ, 
                                                            risque_all_loan=risque_all_loan, current_credit_cb=current_credit_cb, 
                                                            nb_mens=nb_mens, nb_cb=nb_cb)

                                            test_cpt += 1
                                            print(f"test_invalide_parameters ==> Avancement : {test_cpt/(3*5*4*4*4*4*4*4*4):.0%}", end="\r")

if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()


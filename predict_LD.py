# -*- coding: utf-8 -*-
"""
Created on Thu Aug  4 15:03:44 2022

@author: rraja14
"""

from data_loader import DataLoader
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from DataSample import DataSample
import pickle

class ModelPredictor():
    def __init__(self, algoritm, test_size=0.30):  
        if(algoritm == 'decision tree'):
            self.data_loader = DataLoader(scale = False)
            self.model = DecisionTreeClassifier()

    def train(self):
         X_train, X_test, y_train, y_test = self.data_loader.load_preprocess()
         self.model.fit(X_train, y_train)
         #save the model to disk
         pickle.dump(self.model, open('DTClassifier.pkl', 'wb'))
         #load the model from disk
         loaded_model = pickle.load(open('DTClassifier.pkl', 'rb'))
         result = loaded_model.score(X_test, y_test)
         return result

    def predict(self, data: DataSample):
        loaded_model = pickle.load(open('DTClassifier.pkl', 'rb'))
        prepared_sample = self.data_loader.prepare_sample(data)
        prediction = loaded_model.predict(prepared_sample)
        probability = loaded_model.predict_proba(prepared_sample).max()
        print({'prediction': prediction[0],'probability': probability})
        return prediction
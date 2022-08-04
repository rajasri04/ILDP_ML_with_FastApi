# -*- coding: utf-8 -*-
"""
Created on Thu Aug  4 14:48:55 2022

@author: rraja14
"""

import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pickle
from DataSample import DataSample

class DataLoader():
    def __init__(self, test_size = 0.30, scale = True):
        self.test_size = test_size
        self.scale = scale
        self.scaler = StandardScaler()

    def load_preprocess(self):
        data = pd.read_csv("Liver_Patient_Dataset.csv")
        
        data = self._feature_engineering_pipeline(data)

        data = data.replace({'Diseased': 2}, 0)
        
        if data.duplicated().any():
             data = data.drop_duplicates()
             
        X = data.drop(['Diseased'], axis=1)
        y = data.Diseased
        
        X = X.drop(["Age", "Total_Bilirubin", "Albumin_and_Globulin_Ratio", "Gender"], axis=1)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= self.test_size,random_state=33)

        return X_train, X_test, y_train, y_test

    def prepare_sample(self, raw_sample: DataSample):
        print('Preparing Sample')
        sample = [raw_sample.direct_Bilirubin, raw_sample.alkaline_Phosphotase, 
                  raw_sample.alamine_Aminotransferase,
                  raw_sample.aspartate_Aminotransferase, 
                  raw_sample.total_Proteins, raw_sample.albumin]
        print(sample)
        sample = np.array([np.asarray(sample)]).reshape(-1, 1)

        if(self.scale):
            StandardScaler().fit_transform(sample)
            print(sample)
        sample = sample.reshape(1, -1)
        print(sample)
        return sample

    def _feature_engineering_pipeline(self, data):
        ### Filling Normal Values in empty places
        data['Total_Bilirubin'].fillna(0.9, inplace=True)
        data['Direct_Bilirubin'].fillna(0.3, inplace=True)
        data['Alkaline_Phosphotase'].fillna(240, inplace=True)
        data['Alanine_Aminotransferase'].fillna(20, inplace=True)
        data['Aspartate_Aminotransferase'].fillna(22, inplace=True)
        data['Total_Proteins'].fillna(7.1, inplace=True)
        data['Albumin'].fillna(4.4, inplace=True)
        data['Albumin_and_Globulin_Ratio'].fillna(1.2, inplace=True)
        
        for col in ['Gender', 'Age']:
            data[col].ffill(inplace=True)
        
        numerical = data.select_dtypes(exclude=['object'])
        categorical = data.select_dtypes(include=['object'])
        le = preprocessing.LabelEncoder()
        label_encoded_categorical = categorical.apply(le.fit_transform)
        data = pd.concat([numerical, label_encoded_categorical], axis=1)

        return data
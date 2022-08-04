# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from predict_LD import ModelPredictor
from DataSample import DataSample

origins = [
    "http://localhost:8000",
    "http://localhost:4200"
]

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.model = ModelPredictor('decision tree')

@app.post("/train")
async def train():
    print("Model Training Started")
    accuracy = app.model.train()
    return accuracy

@app.post("/predict")
async def predict(data:DataSample):
    try: 
        print("Predicting")
        diseases_map = {0: 'No Disease', 1: 'Disease'}
        results = app.model.predict(data)
        return diseases_map[results[0]]
        #return {prediction: results[0], probability: results[]}
    except Exception as e:
        print(e.__class__)

        
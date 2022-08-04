# -*- coding: utf-8 -*-
"""
Created on Thu Aug  4 14:42:48 2022

@author: rraja14
"""

from pydantic import BaseModel

class DataSample(BaseModel):
    age: float                        
    gender: int    
    total_Bilirubin: float             
    direct_Bilirubin:float             
    alkaline_Phosphotase: float         
    alamine_Aminotransferase: float     
    aspartate_Aminotransferase: float
    total_Proteins: float 
    albumin: float                
    albumin_and_Globulin_Ratio: float

    def __getitem__(self, item):
        return getattr(self, item)
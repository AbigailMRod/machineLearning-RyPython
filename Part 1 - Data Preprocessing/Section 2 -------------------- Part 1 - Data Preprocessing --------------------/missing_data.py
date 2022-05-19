# -*- coding: utf-8 -*-
"""
Created on Wed May 18 11:47:43 2022

@author: Abigail
"""
#Plantilla de Pre Procesado - Datos faltantes

#Cómo importar las librerías

import numpy as np #matemáticas
import matplotlib.pyplot as plt #graficas y representación visual
import pandas as pd #carga de datos

#importar el dataset
dataset = pd.read_csv('Data.csv')
#iloc sirve para localizar los elemntos (filas y columnas) de un sataset
#por posicion i=index
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 3].values

#Tratamiento de los NAs
from sklearn.impute import SimpleImputer
#axis(verbose)=0 se aplica por columna, axis =1 se aplica por fila
imputer = SimpleImputer(missing_values = np.nan, strategy = "mean", verbose=0)
imputer = imputer.fit(X[:,1:3]) 
X[:, 1:3] = imputer.transform(X[:,1:3])
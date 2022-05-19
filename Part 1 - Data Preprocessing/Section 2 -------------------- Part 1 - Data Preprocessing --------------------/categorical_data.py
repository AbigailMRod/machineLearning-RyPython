# -*- coding: utf-8 -*-
"""
Created on Wed May 18 11:46:32 2022

@author: Abigail
"""
#Plantilla de Pre Procesado - Datos categóricos

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

#Codificar datos categóricos

from sklearn.preprocessing import LabelEncoder
labelEncoder_X = LabelEncoder()
labelEncoder_X.fit_transform(X[:,0])
#para afectar el data set con los valores obtenido
X[:,0] = labelEncoder_X.fit_transform(X[:,0])

#Varaibles dummy, es para categorizar variables que no tiene un orden
#en vez de una columna con categorías, se pasa a tantas columnas como categorías
# tenga la variable

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer

labelencoder_X = LabelEncoder()
X[:, 0] = labelencoder_X.fit_transform(X[:, 0])

ct = ColumnTransformer(
    [('one_hot_encoder', OneHotEncoder(categories='auto'), [0])],   
    remainder='passthrough')

X = np.array(ct.fit_transform(X), dtype=np.float)
labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)
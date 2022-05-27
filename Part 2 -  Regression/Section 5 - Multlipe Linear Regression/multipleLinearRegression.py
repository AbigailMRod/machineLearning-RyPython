# -*- coding: utf-8 -*-
"""
Created on Thu May 19 10:40:10 2022

@author: Abigail
"""
#Regresión lineal múltiple

#Cómo importar las librerías

import numpy as np #matemáticas
import matplotlib.pyplot as plt #graficas y representación visual
import pandas as pd #carga de datos

#importar el dataset
dataset = pd.read_csv('50_Startups.csv')
#iloc sirve para localizar los elemntos (filas y columnas) de un sataset
#por posicion i=index
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 4].values

#Codificar datos categóricos


from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import make_column_transformer

labelencoder_X = LabelEncoder()
X[:, 3] = labelencoder_X.fit_transform(X[:, 3])
onehotencoder = make_column_transformer((OneHotEncoder(), [3]), remainder = "passthrough")
X = onehotencoder.fit_transform(X)

#Evitar la trampa de las variables ficticias
X = X[:,1:]

#Dividir el data set en conjunto d eentrenamiento y conjunto de test
#size=0.2 implica que un 20% de los datos se destinaran a testing, max0.3
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


#Ajustar el modelo de regresión lineal múltiple con el conjunto de ENTRENAMIENTO
from sklearn.linear_model import LinearRegression
regression = LinearRegression()
regression.fit(X_train, y_train)


#Predicción de los resultados en el conjutno de TESTING
y_pred= regression.predict(X_test)

"""Construir el modelo de óptimo de regresión lineal múltipl utilizando 
la eliminación hacia atras 
"""
import statsmodels.api as sm
X = np.append(arr= np.ones((50,1)).astype(int), values=X, axis=1)

SL = 0.05

X_opt = X[:, [0, 1, 2, 3, 4, 5]]
regression_OLS = sm.OLS(endog = y, exog = X_opt.tolist()).fit()
regression_OLS.summary()

#quitando la variable con P más grande
X_opt = X[:, [0, 1, 3, 4, 5]]
regression_OLS = sm.OLS(endog = y, exog = X_opt.tolist()).fit()
regression_OLS.summary()

#quitando la variable con P más grande
X_opt = X[:, [0, 3, 4, 5]]
regression_OLS = sm.OLS(endog = y, exog = X_opt.tolist()).fit()
regression_OLS.summary()

#quitando la variable con P más grande
X_opt = X[:, [0, 3, 5]]
regression_OLS = sm.OLS(endog = y, exog = X_opt.tolist()).fit()
regression_OLS.summary()

#quitando la variable con P más grande
X_opt = X[:, [0, 3,]]
regression_OLS = sm.OLS(endog = y, exog = X_opt.tolist()).fit()
regression_OLS.summary()

### Eliminación hacia atras automática##############
#" Eliminación hacia atras utilizando solo p-valores"
import statsmodels.api as sm
def backwardElimination(x, sl):    
    numVars = len(x[0])    
    for i in range(0, numVars):        
        regressor_OLS = sm.OLS(y, x.tolist()).fit()        
        maxVar = max(regressor_OLS.pvalues).astype(float)        
        if maxVar > sl:            
            for j in range(0, numVars - i):                
                if (regressor_OLS.pvalues[j].astype(float) == maxVar):                    
                    x = np.delete(x, j, 1)    
    regressor_OLS.summary()    
    return x 
 
SL = 0.05
X_opt = X[:, [0, 1, 2, 3, 4, 5]]
X_Modeled = backwardElimination(X_opt, SL)


#Eliminación hacia atras utilizando p-valores y el valor de Rcuadrado ajustado
import statsmodels.api as sm
def backwardElimination(x, SL):    
    numVars = len(x[0])    
    temp = np.zeros((50,6)).astype(int)    
    for i in range(0, numVars):        
        regressor_OLS = sm.OLS(y, x.tolist()).fit()        
        maxVar = max(regressor_OLS.pvalues).astype(float)        
        adjR_before = regressor_OLS.rsquared_adj.astype(float)        
        if maxVar > SL:            
            for j in range(0, numVars - i):                
                if (regressor_OLS.pvalues[j].astype(float) == maxVar):                    
                    temp[:,j] = x[:, j]                    
                    x = np.delete(x, j, 1)                    
                    tmp_regressor = sm.OLS(y, x.tolist()).fit()                    
                    adjR_after = tmp_regressor.rsquared_adj.astype(float)                    
                    if (adjR_before >= adjR_after):                        
                        x_rollback = np.hstack((x, temp[:,[0,j]]))                        
                        x_rollback = np.delete(x_rollback, j, 1)     
                        print (regressor_OLS.summary())                        
                        return x_rollback                    
                    else:                        
                        continue    
    regressor_OLS.summary()    
    return x 
 
SL = 0.05
X_opt = X[:, [0, 1, 2, 3, 4, 5]]
X_Modeled = backwardElimination(X_opt, SL)


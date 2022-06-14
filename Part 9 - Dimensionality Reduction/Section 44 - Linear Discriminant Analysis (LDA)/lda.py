# -*- coding: utf-8 -*-
"""
Created on Tue Jun 14 17:48:06 2022

@author: Abigail
"""

##Análisis discriminante lineal 



import numpy as np #matemáticas
import matplotlib.pyplot as plt #graficas y representación visual
import pandas as pd #carga de datos

#importar el dataset
dataset = pd.read_csv('Wine.csv')
#iloc sirve para localizar los elemntos (filas y columnas) de un sataset
#por posicion i=index
X = dataset.iloc[:, 0:13].values
y = dataset.iloc[:, 13].values


#Dividir el data set en conjunto d eentrenamiento y conjunto de test
#size=0.2 implica que un 20% de los datos se destinaran a testing, max0.3
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)



#Escalado de variables 
#Se hace a menudo pero no siempre
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
#solo transform para que detecte la transformaci+on que se hizo en el de entrenamiento
X_test = sc_X.transform(X_test)



###################Reducir la dimensión el LDA

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
#pca = PCA(n_components = None) para saber el numero de componentes a elegir
lda= LDA(n_components=2)
X_train= lda.fit_transform(X_train, y_train)
X_test= lda.transform(X_test)



## Ajustar el modelo de regresión logistica en el conjunto de entrenamiento
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state=(0))
classifier.fit(X_train, y_train)

#Predicción de los resultados con el conjunto de testing
y_pred = classifier.predict(X_test)


# Elaborar una matriz de confusión
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

#Representación gráfica de los resultados del algoritmo en conjunto de entrenamiento

from matplotlib.colors import ListedColormap
X_set, y_set = X_train, y_train
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green', 'blue')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green', 'blue'))(i), label = j)
plt.title('Clasificador (Conjunto de Entrenamiento)')
plt.xlabel('CP1')
plt.ylabel('CP2')
plt.legend()
plt.show()


# Representación gráfica de los resultados del algoritmo en el Conjunto de Testing
X_set, y_set = X_test, y_test
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green', 'blue')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green', 'blue'))(i), label = j)
plt.title('Clasificador (Conjunto de Test)')
plt.xlabel('CP1')
plt.ylabel('CP2')
plt.legend()
plt.show()



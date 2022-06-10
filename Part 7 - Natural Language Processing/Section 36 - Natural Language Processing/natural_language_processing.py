# -*- coding: utf-8 -*-
"""
Created on Thu Jun  9 18:01:12 2022

@author: Abigail 
"""

#Procesamiento del leguaje natural 

#importar librerias básicas
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#importar el dataset
#"quoting=3 ignora las commillas dobles"
dataset = pd.read_csv("Restaurant_Reviews.tsv", delimiter="\t", quoting= 3)


#Limpiar el texto 
######################## ejemplo para solo una reseña
#re es de expresiones regulares 
import re
#quitar los elementos que no queremos con la expresion regular
#review = re.sub('[^a-zA-Z]', ' ', dataset['Review'][0])   
    
#convertir las cadenas a minúsculas 
#review = review.lower()

#eliminar palabras irrelevantes 
import nltk
#nltk.download('stopwords')
from nltk.corpus import stopwords

#de la cadena de texto, separa las palabras
#review = review.split()

#convertir las palabras a su infinitivo 
from nltk.stem.porter import PorterStemmer
#ps = PorterStemmer()
#review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))] 

#de las palabras que quedan, pasar de nuevoa a cadena de texto
#review = ' '.join(review)


##### bucle para que todas la criticas se limpien 
corpus = []
for i in range(0,1000):
    review = re.sub('[^a-zA-Z]', ' ', dataset['Review'][i]) 
    review = review.lower()
    review = review.split()
    ps = PorterStemmer()
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    review = ' '.join(review)
    corpus.append(review)


#crear el bag of words
from sklearn.feature_extraction.text import CountVectorizer
#max_features limita a las 11500 palabras mas reelevantes
cv = CountVectorizer(max_features=1500 )
X = cv.fit_transform(corpus).toarray()
y= dataset.iloc[:, 1].values


#en procesamiento de lenguaje natural los modelos más utilizados son
# las maquinas de soporte vectorial, naive Bayes o árboles de decisión

################### Naive Bayes 73% exactitiud

#Dividir el data set en conjunto d eentrenamiento y conjunto de test
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)


## Ajustar el clasificador en el conjunto de entrenamiento
from sklearn.naive_bayes import GaussianNB
classifier= GaussianNB()
classifier.fit(X_train, y_train)

#Predicción de los resultados con el conjunto de testing
y_pred = classifier.predict(X_test)


# Elaborar una matriz de confusión
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)


####################### regresion logistica 71% exactitud
#Dividir el data set en conjunto d eentrenamiento y conjunto de test
#size=0.2 implica que un 20% de los datos se destinaran a testing, max0.3
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)



## Ajustar el modelo de regresión logistica en el conjunto de entrenamiento
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state=(0))
classifier.fit(X_train, y_train)

#Predicción de los resultados con el conjunto de testing
y_pred = classifier.predict(X_test)

# Elaborar una matriz de confusión
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)


########KNN 58% de exactitud

#Dividir el data set en conjunto d eentrenamiento y conjunto de test
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)


## Ajustar el clasificador en el conjunto de entrenamiento
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors=5, metric="minkowski", p=2)
classifier.fit(X_train, y_train)



#Predicción de los resultados con el conjunto de testing
y_pred = classifier.predict(X_test)


# Elaborar una matriz de confusión
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)


############SVM 72% exactitud

#Dividir el data set en conjunto d eentrenamiento y conjunto de test
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)


## Ajustar el clasificador en el conjunto de entrenamiento
from sklearn.svm import SVC
classifier = SVC(kernel="linear", random_state=0)
classifier.fit(X_train, y_train)


#Predicción de los resultados con el conjunto de testing
y_pred = classifier.predict(X_test)


# Elaborar una matriz de confusión
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)


############### Kernel SVM 73.5% exactitud
#Dividir el data set en conjunto d eentrenamiento y conjunto de test
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)


## Ajustar el clasificador en el conjunto de entrenamiento
from sklearn.svm import SVC
classifier = SVC(kernel="rbf", random_state=0)
classifier.fit(X_train, y_train)


#Predicción de los resultados con el conjunto de testing
y_pred = classifier.predict(X_test)


# Elaborar una matriz de confusión
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)


############### Arbol de desición 71% exactitud

#Dividir el data set en conjunto d eentrenamiento y conjunto de test
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)


## Ajustar el clasificador en el conjunto de entrenamiento
from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier(criterion="entropy", random_state=0)
classifier.fit(X_train, y_train)

#Predicción de los resultados con el conjunto de testing
y_pred = classifier.predict(X_test)


# Elaborar una matriz de confusión
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)


###random forest 72% exactitud

#Dividir el data set en conjunto d eentrenamiento y conjunto de test
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)


## Ajustar el clasificador en el conjunto de entrenamiento
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators=10, criterion= "entropy", random_state=0)
classifier.fit(X_train, y_train)


#Predicción de los resultados con el conjunto de testing
y_pred = classifier.predict(X_test)


# Elaborar una matriz de confusión
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

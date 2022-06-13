#Redes neuronales artificiales 

#Importar el data set
dataset = read.csv('Churn_Modelling.csv')
dataset = dataset[, 4:14]


#Codificar los factores para la red neuronal 
dataset$Geography = as.numeric(factor(dataset$Geography, 
                                      levels = c("France", "Spain", "Germany"),
                                      labels = c(1,2,3)))

dataset$Gender =as.numeric(factor(dataset$Gender,
                                 levels = c("Female", "Male"),
                                 labels = c(1,2)))


#Dividir los datos en conjunto de entrenamiento 
install.packages("caTools")
library(caTools)
set.seed(123) #semilla aleatoria
split = sample.split(dataset$Exited, SplitRatio = 0.80)
training_set = subset(dataset, split == TRUE)
testing_set = subset(dataset, split == FALSE)


##Escalado de valores (normalizaci?n)
training_set [, -11] = scale(training_set[,-11])
testing_set [, -11]= scale(testing_set[, -11])


#Ajustar el clasificador  con el conjunto de enrenamiento
#################### Crear la red neuronal
#neural net sirve para regresiones
#nnet para una sola capa oculta
#deepnet también para redes

#install.packages("h2o")
library(h2o)
#hay que conectarse a una intancia de h2o para inicializarla
#especificar el numero de nucleos que se le de va a dedicar a esta red 
#en este caso todos menos 1, ya que se necesita muhco poder 
h2o.init(nthreads = -1)


##
#hidde, tamaño de la capa oculta y nodos
#y variable dependiente
#activation, formula de activacion
#epochs el numero de iteraciones
#train_samples_per_iteration numero de muestras totales que elige por iteacion 
classifier = h2o.deeplearning(y= "Exited", 
                              training_frame = as.h2o(training_set),
                              activation ="Rectifier", hidden = c(6,6),
                              epochs = 100,
                              train_samples_per_iteration = -2)




#Predicción con los resultados con el conjunto de testing
prob_pred = h2o.predict(classifier, newdata = as.h2o(testing_set[, -11]))
y_pred = (prob_pred > 0.5)
y_pred = as.vector(y_pred)

#Crear la matriz de confusión
cm = table(testing_set[,11], y_pred)


#cerrar la sesión de h2o
h2o.shutdown()



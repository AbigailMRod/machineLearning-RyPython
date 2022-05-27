#Regresión lineal multiple

#Importar el data set

dataset = read.csv('50_Startups.csv')

#Codificar las variables categÃ³ricas
dataset$State = factor(dataset$State,
                         levels = c("New York", "California", "Florida"),
                         labels = c(1,2,3))


#Dividir los datos en conjunto de entrenamiento 
#install.packages("caTools")

library(caTools)
set.seed(123) #semilla aleatoria
#0.8 es el porcentaje que sive para entrenar (train)
split = sample.split(dataset$Profit, SplitRatio = 0.8)
training_set = subset(dataset, split == TRUE)
testing_set = subset(dataset, split == FALSE)


#Ajustar el modelo de regresion lineal multiple con el conjunto de ENTRENAMIENTO
regression = lm(formula= Profit ~ . ,
                 data = training_set)
summary(regression)

#Predecir los resultados con el conjunto de TESTING
y_pred = predict(regression,newdata = testing_set)


# Construir un modelo optimo con la eliminacion hacia atras
SL=0.05
regression = lm(formula= Profit ~ R.D.Spend + Administration + Marketing.Spend + State ,
                data = dataset)
summary(regression)

#Cuando se van eliminando las variables con mayor p valor
regression = lm(formula= Profit ~ R.D.Spend + Administration + Marketing.Spend,
                data = dataset)
summary(regression)

#Cuando se van eliminando las variables con mayor p valor
regression = lm(formula= Profit ~ R.D.Spend + Marketing.Spend,
                data = dataset)
summary(regression)

#Cuando se van eliminando las variables con mayor p valor
regression = lm(formula= Profit ~ R.D.Spend,
                data = dataset)
summary(regression)

install.packages("https://cran.r-project.org/src/contrib/Archive/ElemStatLearn/ElemStatLearn_2015.6.26.2.tar.gz",repos=NULL, type="source")

#Implementación automática de la eliminación hacia atras
backwardElimination <- function(x, sl) {
  numVars = length(x)
  for (i in c(1:numVars)){
    regressor = lm(formula = Profit ~ ., data = x)
    maxVar = max(coef(summary(regressor))[c(2:numVars), "Pr(>|t|)"])
    if (maxVar > sl){
      j = which(coef(summary(regressor))[c(2:numVars), "Pr(>|t|)"] == maxVar)
      x = x[, -j]
    }
    numVars = numVars - 1
  }
  return(summary(regressor))
}

SL = 0.05
dataset = dataset[, c(1,2,3,4,5)]
backwardElimination(training_set, SL)


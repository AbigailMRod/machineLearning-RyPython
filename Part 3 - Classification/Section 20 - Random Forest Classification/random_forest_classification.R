#Random forest 


#Importar el data set
dataset = read.csv('Social_Network_Ads.csv')
dataset = dataset[, 3:5]

#Codificar la variable de clasificación como factor
dataset$Purchased = factor(dataset$Purchased, 
                           levels = c(0,1))

#Dividir los datos en conjunto de entrenamiento 
install.packages("caTools")
library(caTools)
set.seed(123) #semilla aleatoria
split = sample.split(dataset$Purchased, SplitRatio = 0.75)
training_set = subset(dataset, split == TRUE)
testing_set = subset(dataset, split == FALSE)


#Ajustar el clasificador  con el conjunto de enrenamiento
library(randomForest)
classifier = randomForest(x = training_set[,-3],
                          y = training_set$Purchased,
                          ntree = 10)


#Predicción con los resultados con el conjunto de testing
y_pred = predict(classifier, newdata = testing_set[, -3])
#type class es para que convierta la probabilidad solo en 2 factores

#Crear la matriz de confusión
cm = table(testing_set[,3], y_pred)


#Visualización del conjunto de entrenamiento
#install.packages("ElemStatLearn")
library(ElemStatLearn)
set = training_set
X1 = seq(min(set[, 1]) - 1, max(set[, 1]) + 1, by = 0.1)
X2 = seq(min(set[, 2]) - 1, max(set[, 2]) + 1, by = 500)
grid_set = expand.grid(X1, X2)
colnames(grid_set) = c('Age', 'EstimatedSalary')
y_grid = predict(classifier, newdata = grid_set)
plot(set[, -3],
     main = 'Random forest (Conjunto de Entrenamiento)',
     xlab = 'Edad', ylab = 'Sueldo Estimado',
     xlim = range(X1), ylim = range(X2))
contour(X1, X2, matrix(as.numeric(y_grid), length(X1), length(X2)), add = TRUE)
points(grid_set, pch = '.', col = ifelse(y_grid == 1, 'springgreen3', 'tomato'))
points(set, pch = 21, bg = ifelse(set[, 3] == 1, 'green4', 'red3'))


# Visualización del conjunto de testing
set = testing_set
X1 = seq(min(set[, 1]) - 1, max(set[, 1]) + 1, by = 1)
X2 = seq(min(set[, 2]) - 1, max(set[, 2]) + 1, by = 500)
grid_set = expand.grid(X1, X2)
colnames(grid_set) = c('Age', 'EstimatedSalary')
y_grid = predict(classifier, newdata = grid_set)
plot(set[, -3],
     main = 'Random forest (Conjunto de Testing)',
     xlab = 'Edad', ylab = 'Sueldo Estimado',
     xlim = range(X1), ylim = range(X2))
contour(X1, X2, matrix(as.numeric(y_grid), length(X1), length(X2)), add = TRUE)
points(grid_set, pch = '.', col = ifelse(y_grid == 1, 'springgreen3', 'tomato'))
points(set, pch = 21, bg = ifelse(set[, 3] == 1, 'green4', 'red3'))






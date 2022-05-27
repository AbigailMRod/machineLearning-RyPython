
#Regresión lineal simple
#Importar el data set

dataset = read.csv('Salary_Data.csv')


#Dividir los datos en conjunto de entrenamiento 
install.packages("caTools")

library(caTools)
set.seed(123) #semilla aleatoria
#0.8 es el porcentaje que sive para entrenar (train)
split = sample.split(dataset$Salary, SplitRatio = 2/3)
training_set = subset(dataset, split == TRUE)
testing_set = subset(dataset, split == FALSE)


#Ajustar el modelo de regresión lineal simple con el conjunto de ENTRENAMIENTO  
regressor = lm(formula = Salary ~ YearsExperience, data = training_set)

summary(regressor)

#Ajustar el modelo de regresión lineal con el conjunto de TEST
y_pred = predict(regressor, newdata = testing_set)


#Visualización de los resultados en el conjunto de ENTRENAMIENTO
install.packages("ggplot2")
library(ggplot2)

ggplot()+
  geom_point(aes(x= training_set$YearsExperience, 
                 y = training_set$Salary),
             colour="red") +
  geom_line(aes(x =training_set$YearsExperience, 
                y= predict(regressor, newdata = training_set)),
              colour="blue") +
  ggtitle("Sueldo vs Años de experiencia (Conjunto de entrenamiento)") +
  xlab("Años de experiencia") +
  ylab("Sueldo en US")



#Visualización de los resultados en el conjunto de TEST
install.packages("ggplot2")
library(ggplot2)

ggplot()+
  geom_point(aes(x= testing_set$YearsExperience, 
                 y = testing_set$Salary),
             colour="black") +
  geom_line(aes(x =training_set$YearsExperience, 
                y= predict(regressor, newdata = training_set)),
            colour="blue") +
  ggtitle("Sueldo vs Años de experiencia (Conjunto de testing)") +
  xlab("Años de experiencia") +
  ylab("Sueldo en US")









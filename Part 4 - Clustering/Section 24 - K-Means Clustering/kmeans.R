#K-means

#importar el dataset
dataset = read.csv("Mall_Customers.csv")
X = dataset[, 4:5]

#método del codo para saber el número de clusters
set.seed(6)
wcss=vector()
for (i in 1:10) {
  wcss[i] <- sum(kmeans(X,i)$withinss)
}

plot(1:10, wcss, type='b', main = "Método del codo",
     xlab = "Número de clusters (k)", ylab = "WCSS(k)")
#WCSS (suma de los cuadrados con respecto al centro)


#aplicar el algoritmo de k-means con k-optimo
set.seed(29)
kmeans <- kmeans(X, 5, iter.max = 300, nstart = 10)


#"visualización de clusters"
library(cluster)
clusplot(X, kmeans$cluster,
         lines = 0,
         shade = TRUE,
         color = TRUE,
         labels = 4,
         plotchar = FALSE,
         span = TRUE,
         main = "Clustering de clientes",
         xlab = "Ingresos anuales en miles USD",
         ylab = "Puntuación de gastos (1-100)")




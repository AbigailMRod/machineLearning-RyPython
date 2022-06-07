#Clustering jerarquico 

#importar el dataset
dataset= read.csv("Mall_Customers.csv")
X = dataset[,4:5]

#utilizar el dendrograma para encontrar el numero optimo de clusters
dendrogram = hclust(dist(X, method = "euclidean"), 
                    method = "ward.D")
plot(dendrogram,
     main= "Dendrograma",
     xlab= "Clientes del centro comercial",
     ylab= "Distancia Euclídea")


#ajustar el clustering jerarquico a nuestro dataset
hc = hclust(dist(X, method = "euclidean"), 
                    method = "ward.D")
y_hc = cutree(hc, k=5)



#"visualización de clusters"
library(cluster)
clusplot(X, y_hc,
         lines = 0,
         shade = TRUE,
         color = TRUE,
         labels = 4,
         plotchar = FALSE,
         span = TRUE,
         main = "Clustering de clientes",
         xlab = "Ingresos anuales en miles USD",
         ylab = "Puntuación de gastos (1-100)")


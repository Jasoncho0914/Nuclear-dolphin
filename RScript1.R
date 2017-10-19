#R script for Competition I 
#lt342, [bc454], [insert netid]

library(readr)
Extracted_features <- read_csv("C:/Users/bj091/Desktop/CS 4786/comp1/Extracted_features.csv", 
                               col_names = FALSE)
Extracted_features <- matrix(as.matrix(Extracted_features),ncol=1084, nrow=10000,dimnames=NULL)

initial_centroids <- read_csv("C:/Users/bj091/Desktop/CS 4786/Kaggle 1/Nuclear-dolphin/centriods.csv",
                              col_names = FALSE)

initial_centroids <- matrix(as.matrix(initial_centroids),ncol=1084, nrow=10,dimnames=NULL)

rownum<-matrix(c(1:10000),nrow=10000,ncol=1)

#running k-means algorithm (Lloyd algorithm) without the initial clustering from "Graph"

cluster_wo<-kmeans(Extracted_features,algorithm = "Lloyd",centers=10,iter.max = 10000)$cluster


cluster_with<-kmeans(Extracted_features,algorithm = "Lloyd",centers=initial_centroids,iter.max = 10000)$cluster





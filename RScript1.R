#R script for Competition I 
#lt342, [bc454], [insert netid]

#Seed <- read_csv("C:/Users/bj091/Desktop/CS 4786/comp1/Seed.csv",col_names = FALSE)

#Seed <- matrix(as.matrix(Seed),ncol=2, dimnames = NULL)

#running k-means algorithm (Lloyd algorithm) without the initial clustering from "Graph"

#cluster_wo<-kmeans(Extracted_features,algorithm = "Lloyd",centers=10,iter.max = 10000)$cluster


#cluster_with<-kmeans(Extracted_features,algorithm = "Lloyd",centers=initial_centroids,iter.max = 10000)$cluster

#matching datapoint and seed

#seed_rownum<-merge(rownum,Seed,all.x=TRUE)

#datapoints<-cbind(seed_rownum,cluster_with,cluster_wo)

#write.table(cluster_wo, file = "cluster_wo.csv",row.names=FALSE, na="",col.names=FALSE, sep=",")

#write.table(cluster_with, file = "cluster_with.csv",row.names=FALSE,na="",col.names=FALSE, sep=",")

#Extracted_features<-cbind(Extracted_features,KNNLabels1)

#eigein<-prcomp(Extracted_features)$rotation[,1:30]

#for(i in 1:1084){
#Extracted_features[,i]<-(Extracted_features[,i]-mean(as.matrix(Extracted_features[,i])))
#}

#Extracted_features_pca<-as.data.frame(as.matrix(Extracted_features)%*%eigein)
library(readr)
Extracted_features <- read_csv("C:/Users/bj091/Desktop/CS 4786/Kaggle 1/Nuclear-dolphin/data/Extracted_features.csv", 
                               col_names = FALSE)

initial_centroids <- read_csv("C:/Users/bj091/Desktop/CS 4786/Kaggle 1/Nuclear-dolphin/Scripts/Results/centriods.csv",
                              col_names = FALSE)

initial_centroids <- matrix(as.matrix(initial_centroids),ncol=1084, nrow=10,dimnames=NULL)

rownum<-matrix(c(1:10000),nrow=10000,ncol=1)

library(ggplot2)

Extracted_features_pca<- read_csv("C:/Users/bj091/Desktop/CS 4786/Kaggle 1/Nuclear-dolphin/Scripts/Results/pca30.csv", 
                                  col_names = FALSE)

AgglomerativePred_200 <- read_csv("C:/Users/bj091/Desktop/CS 4786/Kaggle 1/Nuclear-dolphin/Scripts/Results/AgglomerativePred_200.csv", col_names = TRUE)

Kmeans_pca30_d1 <- read_csv("C:/Users/bj091/Desktop/CS 4786/Kaggle 1/Nuclear-dolphin/Scripts/Results/Kmeans_pca30_d1.csv", col_names = TRUE)

color_agglomerative<-as.factor(as.matrix(AgglomerativePred_200$Label))

color_kmeans<-as.factor(as.matrix(Kmeans_pca30_d1$Label))

#write.table(Extracted_features_pca, file = "Extracted_features_pca.csv",row.names=FALSE,na="",col.names=FALSE, sep=",")


#agglomerative
agglomerative<-ggplot(data = Extracted_features_pca[6001:10000,])+
  geom_point(aes(x = as.matrix(Extracted_features_pca)[6001:10000,1], y= as.matrix(Extracted_features_pca)[6001:10000,2],colour=color_agglomerative))+
  xlab("First Principal Component") + ylab("Second Principal Component")+
  labs(colour = "Label") + 
  scale_color_manual(values=c("RED", "ORANGE","YELLOW","GREEN", "BLUE","BLACK","#E69F00","GREY","BROWN","PINK"))

#Kmeans with 30
kmeans<-ggplot(data = Extracted_features_pca[6001:10000,])+
  geom_point(aes(x = as.matrix(Extracted_features_pca)[6001:10000,1], y= as.matrix(Extracted_features_pca)[6001:10000,2],colour=color_kmeans))+
  xlab("First Principal Component") + ylab("Second Principal Component")+
  labs(colour = "Label") + 
  scale_color_manual(values=c("RED", "ORANGE","YELLOW","GREEN", "BLUE","BLACK","#E69F00","GREY","BROWN","PINK"))

ggsave("AgglomerativePred_200.png", plot = agglomerative)
ggsave("Kmeans_with_30.png", plot = kmeans)

import numpy as np
import pandas as pd
import csv
import time
import sys
from sklearn.cluster import AgglomerativeClustering

from Utility import assignClusters
from Utility import createPred
from Utility import writeCSV
from Utility import createConnectivityMatrix


#=======================================================================
#Agglomerative
#Usage: 
#1. Go to Nuclear-Dolphin 
#2. run 'python ./Scripts/kagglomerativeClustering.py ./data/Extracted_features.csv result'
#./data/Extracted_features.csv can be replaced by any data file (for instance pca output)
#This will create a prediciton file named result.csv
#=======================================================================
def loadData(path):
    seed = np.array(pd.read_csv("./data/Seed.csv", header=None))
    similarity = np.array(pd.read_csv("./data/Graph.csv", header=None))
    train = np.array(pd.read_csv(path, header=None))
    return seed, similarity, train

def runClustering(train):
    ac = AgglomerativeClustering(n_clusters=10).fit(train)
    labels = ac.labels_
    return labels

if __name__ == '__main__':
    path = sys.argv[1]
    output = sys.argv[2]

    print 'Loading data...'
    seed, similarity, train = loadData(path)

    # print 'Creating connectivity matrix...'
    # connectivity = createConnectivityMatrix(similarity)

    print 'Running clustering algorithm...'
    labels = runClustering(train)

    print 'Creating CSV File...'
    name = 'AgglomerativeLabels'
    writeCSV(labels, name)

    print 'Visualing clusters...'
    result = np.array(pd.read_csv("./Scripts/Results/" + name + '.csv', header=None))
    cluster = assignClusters(seed, result)

    pred = createPred(seed, cluster, result)
    writeCSV(pred, output, ['Id', 'Label'])
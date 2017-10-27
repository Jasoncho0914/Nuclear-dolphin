import numpy as np
import pandas as pd
import csv
import time
import sys
from sklearn.cluster import KMeans
from Utility import assignClusters
from Utility import createPred
from Utility import writeCSV

def loadData(path):
    seed = np.array(pd.read_csv("./data/Seed.csv", header=None))
    train = np.array(pd.read_csv(path, header=None))
    return seed, train

def runKNN(train):
    centriods = np.array(pd.read_csv("./Scripts/Results/centriods.csv", header=None))
    kmeans = KMeans(n_clusters=10, random_state=0).fit(train)
    labels = kmeans.labels_
    return labels

if __name__ == '__main__':
    path = sys.argv[1]

    print 'Loading data...'
    seed, train = loadData(path)
    print 'Running KNN...'
    labels = runKNN(train)
    print 'Creating CSV File...'
    name = 'KNNLabels1'
    writeCSV(labels, name)
    print 'Visualing clusters...'
    result = np.array(pd.read_csv("./Scripts/Results/" + name + '.csv', header=None))
    cluster = assignClusters(seed, result)

    pred = createPred(seed, cluster, result)
    writeCSV(pred[6001:], 'KNNPred1', ['Id', 'Label'])

import numpy as np
import pandas as pd
import csv
import time
import sys
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import Birch
from sklearn.cluster import FeatureAgglomeration
from sklearn.cluster import MiniBatchKMeans
from sklearn.mixture import GaussianMixture
from Utility import assignClusters
from Utility import createPred
from Utility import writeCSV

def loadData(path):
    seed = np.array(pd.read_csv("./data/Seed.csv", header=None))
    train = np.array(pd.read_csv(path, header=None))
    return seed, train

def runClustering(train):
    # kmeans = KMeans(n_clusters=10).fit(train)
    ac = AgglomerativeClustering(n_clusters=10).fit(train)
    # b = Birch(n_clusters=10).fit(train)
    # db = DBSCAN().fit(train)
    # fa = FeatureAgglomeration(n_clusters=10).fit(train)
    # mb = MiniBatchKMeans(n_clusters=10).fit(train)
    labels = ac.labels_
    return labels

if __name__ == '__main__':
    path = sys.argv[1]

    print 'Loading data...'
    seed, train = loadData(path)

    
    preds = []

    for _ in range(1):
        print 'Running clustering algorithm...'
        labels = runClustering(train)

        print 'Creating CSV File...'
        name = 'ClusteringLabels'
        writeCSV(labels, name)

        print 'Visualing clusters...'
        result = np.array(pd.read_csv("./Scripts/Results/" + name + '.csv', header=None))
        cluster = assignClusters(seed, result)

        pred = createPred(seed, cluster, result)
        # if pred:
        #     preds.append(pred)

    # for i in range(len(preds[0])):
    #     digit = [0]*11
    #     for j in range(len(preds)):
    #         digit[preds[j][i][1]]+=1
    #     pred[i] = [pred[i][0], digit.index(max(digit))]

    writeCSV(pred[6000:], 'ClusteringPred', ['Id', 'Label'])
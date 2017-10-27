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
from Utility import initClusters
from Utility import calculateCentriods

#=======================================================================
#Playground script for testing and trying out different scripts
#=======================================================================
def loadData(path):
    seed = np.array(pd.read_csv("./data/Seed.csv", header=None))
    train = np.array(pd.read_csv(path, header=None))
    graph = np.array(pd.read_csv("./data/Graph.csv", header=None))
    return seed, train, graph

def runClustering(train, centriods=None):
    kmeans = KMeans(n_clusters=10, init=centriods).fit(train)
    # ac = AgglomerativeClustering(n_clusters=10).fit(train)
    # labels = ac.labels_
    labels = kmeans.labels_
    return labels

if __name__ == '__main__':
    path = sys.argv[1]
    output = sys.argv[2]

    print 'Loading data...'
    seed, train, graph = loadData(path)

    print 'Creating initial clusters...'
    clusters = initClusters(seed, graph, 1)
    centriods = calculateCentriods(train, clusters)

    print 'Running clustering algorithm...'
    labels = runClustering(train, centriods)

    print 'Creating CSV File...'
    name = 'ClusteringLabels'
    writeCSV(labels, name)

    print 'Visualing clusters...'
    result = np.array(pd.read_csv("./Scripts/Results/" + name + '.csv', header=None))
    cluster = assignClusters(seed, result)

    print 'Creating prediction...'
    pred = createPred(seed, cluster, result)
    writeCSV(pred[6000:], output, ['Id', 'Label'])
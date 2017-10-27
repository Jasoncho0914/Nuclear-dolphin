import numpy as np
import pandas as pd
import csv
import time
import sys
from sklearn.cluster import KMeans
from sklearn.cluster import SpectralClustering

from Utility import assignClusters
from Utility import createPred
from Utility import writeCSV

def loadData(path):
    seed = np.array(pd.read_csv("./data/Seed.csv", header=None))
    train = np.array(pd.read_csv(path, header=None))
    return seed, train

def runSpectralClustering(train):
    sc = SpectralClustering(n_clusters=10, affinity='nearest_neighbors').fit(train)
    labels = sc.labels_
    return labels

if __name__ == '__main__':
    start = time.time()

    path = sys.argv[1]

    print 'Loading data...'
    seed, train = loadData(path)

    print 'Running Spectral Clustering...'

    labels = runSpectralClustering(train)
    print 'Creating CSV File...'
    name = 'SCLabels'
    writeCSV(labels, name)

    print 'Visualing clusters...'
    result = np.array(pd.read_csv("./Scripts/Results/" + name + '.csv', header=None))
    cluster = assignClusters(seed, result)

    # pred = createPred(seed, cluster, result)
    # writeCSV(pred[6000:], 'SCPred', ['Id', 'Label'])
    print time.time() - start
import numpy as np
import pandas as pd
import csv
import time
import sys
from sklearn.cluster import KMeans
from sklearn.cluster import SpectralClustering
from calculateAccuracy import assignClusters

def writeCSV(data, name):
    data = list(data)
    with open("./Scripts/Results/" + name + ".csv",'wb') as resultFile:
        wr = csv.writer(resultFile)
        for x in data:
            wr.writerow([x])

def loadData(path):
    seed = np.array(pd.read_csv("./data/Seed.csv", header=None))
    train = np.array(pd.read_csv(path, header=None))
    return seed, train

def runSpectralClustering(train):
    sc = SpectralClustering(n_clusters=10, random_state=0, n_jobs=2).fit(train)
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
    name = 'SCLabels1'
    writeCSV(labels, name)
    print 'Visualing clusters...'
    result = np.array(pd.read_csv("./Scripts/Results/" + name + '.csv', header=None))
    assignClusters(seed, result)
    print time.time() - start
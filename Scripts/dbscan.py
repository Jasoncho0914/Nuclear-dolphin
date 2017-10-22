import numpy as np
import pandas as pd
import csv
import time
import sys
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import Birch
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

def runDBSCAN(train):
    db = Birch(n_clusters=10).fit(train)
    labels = db.labels_
    return labels

if __name__ == '__main__':
    path = sys.argv[1]

    print ('Loading data...')
    seed, train = loadData(path)
    print ('Running DBSCAN...')
    labels = runDBSCAN(train)
    print ('Creating CSV File...')
    name = 'DBSCANLabels1'
    writeCSV(labels, name)
    print ('Visualing clusters...')
    result = np.array(pd.read_csv("./Scripts/Results/" + name + '.csv', header=None))
    assignClusters(seed, result)
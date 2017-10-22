import sys
import numpy as np
import pandas as pd
import csv
import time

# Run this file by running 
# python ./Scripts/calculateAccuracy.py [path to result] from root directory

def assignClusters(seed, result):
    j = 0
    cluster = {}
    for i in range(len(result)):
        if j < len(seed) and seed[j][0] - 1 == i:
            l = cluster.setdefault(result[i][0], [])
            l.append(seed[j][1])
            cluster[result[i][0]] = l
            j+=1
    print (cluster)
    return cluster

#mathces by frequency. If cluster 1 gets mapped to digit 2 the most, cluster 1 = 2. 
def createPred(seed, cluster, result):
    matchedInt = set([])
    for key, val in cluster.items():
        maxCount = 0
        for n in range(10):
            if maxCount < val.count(n) and n not in matchedInt:
                cluster[key] = n
                maxCount = val.count(n)
        # print cluster[key]
        # matchedInt.add(cluster[key])
    result = [[i+1, cluster[x[0]]] for i, x in enumerate(result)]
    return result

if __name__ == "__main__":
    resultPath = sys.argv[1]

    seed = np.array(pd.read_csv('./data/Seed.csv', header=None))
    result = np.array(pd.read_csv(resultPath, header=None))

    assignClusters(seed, result)




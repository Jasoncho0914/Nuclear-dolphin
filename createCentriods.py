import numpy as np
import pandas as pd
import csv
import time

def createSets(data):
    clusterList = []
    for i in range(len(data)):
        found = False
        skip = False

        for j in range(len(clusterList)):
            if data[i][1] in clusterList[j]:
                skip = True
                found = True
                break

        if not skip:
            for j in range(len(clusterList)):
                if data[i][0] in clusterList[j]:
                    clusterList[j].add(data[i][1])
                    found = True
                    break

        if not found:
            clusterList.append(set([data[i][0], data[i][1]]))

    return clusterList

def calculateCentriods(rawData, clusterSets):
    centriods = []
    clusterSets.sort(key=len)
    for cs in clusterSets[:10]:
        s = 0
        for idx in cs:
            s += rawData[idx-1]
        s = s/float(len(cs))
        centriods.append(s)
    return centriods

def writeData(data, name):
    for i in range(len(data)):
        data[i] = list(data[i])

    with open(name + ".csv",'wb') as resultFile:
        wr = csv.writer(resultFile)
        wr.writerows(data)

if __name__ == "__main__":
    Graph = np.array(pd.read_csv("fa2017_competition_1/Graph.csv"))
    train = train = np.array(pd.read_csv("fa2017_competition_1/Extracted_features.csv"))

    clusters = createSets(Graph)
    centriods = calculateCentriods(train, clusters)
    writeData(centriods, 'centriods')




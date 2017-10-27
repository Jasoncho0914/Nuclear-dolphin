import sys
import numpy as np
import pandas as pd
import scipy as sp
import csv
import time


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
        add = False
        for n in range(10):
            if maxCount < val.count(n):
                cluster[key] = n
                maxCount = val.count(n)
                add = True
        if add:
            matchedInt.add(cluster[key])
        # print cluster[key]
        # matchedInt.add(cluster[key])
    for i in range(10):
        if i not in cluster:
            for j in range(10):
                if j not in matchedInt:
                    cluster[i] = j

    print cluster
    input = raw_input("Change anything?: ")
    if input != "":
        done = False
    else:
        done = True
    while not done:
        key = int(raw_input("Enter key: "))
        val = int(raw_input("Enter val: "))
        if key in cluster and val in cluster:
            cluster[key] = val
        else:
            break
        cont = raw_input("Continue change?: ")
        if cont == "":
            done = True
    print cluster

    try: 
        result = [[i+1, cluster[x[0]]] for i, x in enumerate(result)]
    except Exception:
        return None
    return result


#Writes data into a csv file
def writeCSV(data, name, header=None):
    data = list(data)
    with open("./Scripts/Results/" + name + ".csv",'wb') as resultFile:
        wr = csv.writer(resultFile)
        if header:
            if type(header) == type([]):
                wr.writerow(header)
            else:
                wr.writerow([header])
            header = None
        
        for x in data:
            if type(x) == type([]):
                wr.writerow(list(x))
            else:
                wr.writerow([x])


def createConnectivityMatrix(data, size=10000):
    connectivityMatrix = [[0]*size for _ in range(size)]
    for i, j in data:
        connectivityMatrix[i-1][j-1] = 1

    return sp.sparse.csr_matrix(connectivityMatrix)


def initClusters(seed, graph, depth=1):
    #Graph -> dictionary 
    graphDict = {}
    for x, y in graph:
        l = graphDict.setdefault(x, [])
        l.append(y)
        graphDict[x] = l

    clusters = {}
    for x, y in seed:
        l = clusters.setdefault(y, [])
        l.append(x)
        clusters[y] = l

    for _ in range(depth-1):
        newList = []
        for key, val in clusters.items():
            for node in val:
                newList += graphDict[node]
                newList += [node]
            clusters[key] = list(set(newList))

    return clusters


def calculateCentriods(data, cluster):
    centriods = []
    for key, val in cluster.items():
        s = 0
        for i in val:
            s += data[i-1]
        s = s/float(len(val))
        centriods.append(s)
    return np.array(centriods)


def combineResults():
    data1 = np.array(pd.read_csv("./Scripts/Results/Kmeans_d2.csv"))
    data2 = np.array(pd.read_csv("./Scripts/Results/AgglomerativePred.csv"))
    data3 = np.array(pd.read_csv("./Scripts/Results/Kmeans_pca30_d1.csv"))
    newResult = []
    for i in range(len(data1)):
        if data1[i][1] == 7 or data1[i][1] == 4:
            newResult.append([data1[i][0], data1[i][1]])
        elif data2[i][1] == 6:
            newResult.append([data2[i][0], data2[i][1]])
        elif data3[i][1] == 1:
            newResult.append([data3[i][0], data3[i][1]])

        else:
            # a, b, c = data1[i][1], data2[i][1], data3[i][1]
            # if a == b or a == c:
            #     newResult.append([data1[i][0], data1[i][1]])
            # else:
            #     newResult.append([data2[i][0], data2[i][1]])
            newResult.append([data2[i][0], data2[i][1]])
    # newResult = np.array(newResult)
    # print newResult
    writeCSV(newResult, 'Combined', ['Id', 'Label'])


if __name__ == "__main__":
    combineResults()




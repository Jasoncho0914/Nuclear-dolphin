import numpy as np
import pandas as pd
import csv
import time
import sys
import random
import json
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import Birch
from sklearn.cluster import FeatureAgglomeration
from sklearn.cluster import MiniBatchKMeans
from sklearn.mixture import GaussianMixture
from sklearn.manifold import SpectralEmbedding
from sklearn.cluster import SpectralClustering
from sklearn.metrics.pairwise import pairwise_kernels
from sklearn.mixture import GaussianMixture
from sklearn.cross_decomposition import CCA


from Utility import assignClusters
from Utility import createPred
from Utility import writeCSV
from Utility import initClusters
from Utility import calculateCentriods
from sklearn.semi_supervised import LabelSpreading
from sklearn.semi_supervised import LabelPropagation
from sklearn.ensemble import RandomForestClassifier


def loadData(path):
    seed = np.array(pd.read_csv("./data/Seed.csv", header=None))
    data = np.array(pd.read_csv(path, header=None))
    graph = np.array(pd.read_csv("./data/Graph.csv", header=None))
    return seed, data, graph


def distance(a, b):
    return np.linalg.norm(a-b)


def mostOccurance(l):
    max = 0
    res = -1
    for val in l:
        if val != -1:
            count = l.count(val)
            if count > max:
                max = count
                res = val
    return res


def calculateCentriod(points, data):
    s = 0.0
    for idx in points:
        s+=np.array(data[idx-1])
    s/= len(points)

    minDist = float('inf')
    new_s = None
    for idx in points:
        dist = distance(data[idx-1], s)
        if dist < minDist:
            minDist = dist
            new_s = idx
    # print minDist
    if new_s == None:
        print 'oops'
    return new_s


def createAffinityMatrix(graph, size=6000):
    affinityMatrix = [[0]*size for _ in range(size)]
    for i, j in graph:
        # connectivityMatrix[i-1][j-1] = distance(data[i], data[j])
        affinityMatrix[i-1][j-1] = 1

    return np.array(affinityMatrix)


def assignClusters(seed, result):
    #Maps cluster output by the clustering algorithm to actual digit value
    j = 0
    cluster = {}
    for i in range(len(result)):
        if j < len(seed) and seed[j][0] - 1 == i:
            l = cluster.setdefault(result[i], [])
            l.append(seed[j][1])
            cluster[result[i]] = l
            j+=1
    print (cluster)
    return cluster


def createLabel(seed, data, graph):
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


    # Creating embedding and affinity matrices for later usage

    # affinityMatrix = createAffinityMatrix(graph)
    # clf = SpectralEmbedding(n_components=1000, affinity="precomputed")
    # clf.fit(affinityMatrix)
    # affinity = clf.affinity_matrix_
    # embedding = clf.embedding_
    # writeCSV(affinity, 'affinity')
    # writeCSV(embedding, 'embedding')


    # affinity = np.array(pd.read_csv("./Scripts/Results/affinity.csv", header=None))
    # embedding = np.array(pd.read_csv("./Scripts/Results/embedding.csv", header=None))


    # X = [x[:35] for x in embedding]
    # X = data[:6000]

    #spectral clustering
    clf = SpectralClustering(n_clusters=10, affinity='precomputed')
    labels = clf.fit_predict(affinity)

    #kmeans clustering
    # clusters = initClusters(seed, graph, 1)
    # clusters = {}
    # centriods = [18, 37, 9, 38, 89, 33, 11, 29, 74, 64]
    # for i, val in enumerate(centriods):
    #     clusters[i] = [val]
    # centriods = calculateCentriods(data[:6000], clusters)

    # X = _CCA(data[:6000], graph, 40)

    # clf = GaussianMixture(n_components=10, init_params="random")
    # clf.fit(X)
    # labels = clf.predict(X)

    # kmeans = KMeans(n_clusters=10).fit(data[:6000])
    # labels = kmeans.labels_

    #agglomerative clustering
    # ac = AgglomerativeClustering(n_clusters=10).fit(data[:6000])
    # labels = ac.labels_

    assignClusters(seed, labels)


    # res = raw_input("Select number to assign: ").split(' ')
    # res2 = raw_input("Select digits to assign it to: ").split(' ')
    # if res[0] == "" or res2[0] == "":
    #     print 'nothing'
    #     return

    # assignment = {}
    # for i, val in enumerate(res):
    #     assignment[int(val)] = int(res2[i])
    # print assignment 

    # res = []
    # for i, label in enumerate(list(labels)):
    #     if label in assignment:
    #         res.append([i+1, assignment[label]])
    # print len(res)

    # name = ''
    # for num in res2:
    #     name+=str(num)
    # writeCSV(res, 'TrainLabel_(' + name + ')')


# def runRandomForest(train, test, ntree):
#     clf = RandomForestClassifier(ntree)
#     clf.fit(train[:, :-1], train[:, -1])
#     pred = clf.predict(test)

#     result = []

#     d = {}
#     for i in range(6000, len(pred)):
#         c = d.setdefault(int(pred[i]), 0)
#         d[int(pred[i])] = c + 1
#         result.append([i+1, int(pred[i])])

#     print d
#     return result


def runLabelSpreading(data, assignment):
    lp_model = LabelSpreading(kernel='knn', n_neighbors=10)
    labels = [-1]*len(data)
    for x, y in assignment:
        labels[x-1] = y
    labels = np.array(labels)
    lp_model.fit(data, labels)
    pred = lp_model.transduction_

    result = []
    d = {}
    for i in range(6000, len(pred)):
        c = d.setdefault(int(pred[i]), 0)
        d[int(pred[i])] = c + 1
        result.append([i+1, int(pred[i])])
    print d
    return result


def createTrain(data, assignment):
    train = []
    for idx, digit in assignment:
        row = list(data[idx-1])
        row.append(digit)
        train.append(row)
    return np.array(train)


def makeAssignment(labels):
    assignment = {}
    for x, y in labels:
        l = assignment.setdefault(x, [])
        l.append(y)
        assignment[x] = l
    return assignment


def combineLabels(data):
    label0 = np.array(pd.read_csv("./Scripts/Results/TrainLabel_(0).csv", header=None))
    assignment0 = makeAssignment(label0)
    print '0 - ' + str(len(assignment0))
    label1 = np.array(pd.read_csv("./Scripts/Results/TrainLabel_(1).csv", header=None))
    assignment1 = makeAssignment(label1)
    print '1 - ' + str(len(assignment1))
    label2 = np.array(pd.read_csv("./Scripts/Results/TrainLabel_(2).csv", header=None))
    assignment2 = makeAssignment(label2)
    print '2 - ' + str(len(assignment2))
    label3 = np.array(pd.read_csv("./Scripts/Results/TrainLabel_(3).csv", header=None))
    assignment3 = makeAssignment(label3)
    print '3 - ' + str(len(assignment3))
    label6 = np.array(pd.read_csv("./Scripts/Results/TrainLabel_(6).csv", header=None))
    assignment6 = makeAssignment(label6)
    print '6 - ' + str(len(assignment6))

    label5_1 = np.array(pd.read_csv("./Scripts/Results/TrainLabel_(5)1.csv", header=None))
    label5_2 = np.array(pd.read_csv("./Scripts/Results/TrainLabel_(5)2.csv", header=None))
    label5_3 = np.array(pd.read_csv("./Scripts/Results/TrainLabel_(5)3.csv", header=None))

    print 'Assignment 5'
    assignment5 = {}
    for x, y in label5_1:
        l = assignment5.setdefault(x, [])
        l.append(y)
        assignment5[x] = l

    for x, y in label5_2:
        l = assignment5.setdefault(x, [])
        l.append(y)
        assignment5[x] = l

    for x, y in label5_3:
        l = assignment5.setdefault(x, [])
        l.append(y)
        assignment5[x] = l

    print '5 -  ' + str(len(assignment5))

    # for key, val in assignment5.items():
    #     if len(val) < 2:
    #         del assignment5[key]

    # print len(assignment5)


    label7_1 = np.array(pd.read_csv("./Scripts/Results/TrainLabel_(7).csv", header=None))
    label7_2 = np.array(pd.read_csv("./Scripts/Results/TrainLabel_(7)1.csv", header=None))
    label7_3 = np.array(pd.read_csv("./Scripts/Results/TrainLabel_(7)2.csv", header=None))
    label7_4 = np.array(pd.read_csv("./Scripts/Results/TrainLabel_(7)3.csv", header=None))
    print 'Assignment 7'
    assignment7 = {}
    for x, y in label7_1:
        l = assignment7.setdefault(x, [])
        l.append(y)
        assignment7[x] = l

    for x, y in label7_2:
        l = assignment7.setdefault(x, [])
        l.append(y)
        assignment7[x] = l

    for x, y in label7_3:
        l = assignment7.setdefault(x, [])
        l.append(y)
        assignment7[x] = l

    for x, y in label7_4:
        l = assignment7.setdefault(x, [])
        l.append(y)
        assignment7[x] = l

    print len(assignment7)

    for key, val in assignment7.items():
        if len(val) < 3:
            del assignment7[key]

    print '7 -  ' + str(len(assignment7))


    label8_1 = np.array(pd.read_csv("./Scripts/Results/TrainLabel_(8)1.csv", header=None))
    label8_2 = np.array(pd.read_csv("./Scripts/Results/TrainLabel_(8)2.csv", header=None))
    label8_3 = np.array(pd.read_csv("./Scripts/Results/TrainLabel_(8)3.csv", header=None))
    print 'Assignment 8'
    assignment8 = {}
    for x, y in label8_1:
        l = assignment8.setdefault(x, [])
        l.append(y)
        assignment8[x] = l

    for x, y in label8_2:
        l = assignment8.setdefault(x, [])
        l.append(y)
        assignment8[x] = l

    for x, y in label8_3:
        l = assignment8.setdefault(x, [])
        l.append(y)
        assignment8[x] = l

    for key, val in assignment8.items():
        if len(val) < 2:
            del assignment8[key]

    print '8 -  ' + str(len(assignment8))

    label4_1 = np.array(pd.read_csv("./Scripts/Results/TrainLabel_(4)1.csv", header=None))
    label4_2 = np.array(pd.read_csv("./Scripts/Results/TrainLabel_(4)2.csv", header=None))
    label9_1 = np.array(pd.read_csv("./Scripts/Results/TrainLabel_(9)1.csv", header=None))
    label49 = np.array(pd.read_csv("./Scripts/Results/TrainLabel_(49).csv", header=None))

    print 'Assignment 4 and 9'
    assignment49 = {}
    for x, y in label4_1:
        l = assignment49.setdefault(x, [])
        l.append(y)
        assignment49[x] = l

    for x, y in label4_2:
        l = assignment49.setdefault(x, [])
        l.append(y)
        assignment49[x] = l

    for x, y in label9_1:
        l = assignment49.setdefault(x, [])
        l.append(y)
        assignment49[x] = l

    for x, y in label49:
        l = assignment49.setdefault(x, [])
        l.append(y)
        assignment49[x] = l

    assignment4 = {}
    assignment9 = {}
    for key, val in assignment49.items():
        if val.count(4) == 2 and val.count(9) == 2:
            a = random.randint(0,1)
            if a == 0:
                val.append(4)
            else:
                val.append(9)

        if val.count(4) >= 2:
            l = assignment4.setdefault(key, [])
            l.append(val)
            assignment4[key] = l
        elif val.count(9) >= 2:
            l = assignment9.setdefault(key, [])
            l.append(val)
            assignment9[key] = l

    print '4 -  ' + str(len(assignment4))
    print '9 -  ' + str(len(assignment9))


    assignments = [assignment0, assignment1, assignment2, assignment3, assignment4, assignment5, assignment6, assignment7, assignment8, assignment9]
    assignment = {}

    for i in range(len(data)):
        for digit in range(10):
            if i+1 in assignments[digit]:
                assignment[i+1] = digit

    d = {}
    res = []
    for key, val in assignment.items():
        res.append([key, val])
        count = d.setdefault(val, 0)
        d[val] = count+1

    writeCSV(res, 'assignment3')


def combineLabels2(data, seed):
    label0 = np.array(pd.read_csv("./Scripts/Results/TrainLabel_(0).csv", header=None))
    assignment0 = makeAssignment(label0)
    print '0 - ' + str(len(assignment0))

    label1 = np.array(pd.read_csv("./Scripts/Results/TrainLabel_(1).csv", header=None))
    assignment1 = makeAssignment(label1)
    print '1 - ' + str(len(assignment1))

    label2 = np.array(pd.read_csv("./Scripts/Results/TrainLabel_(2).csv", header=None))
    assignment2 = makeAssignment(label2)
    print '2 - ' + str(len(assignment2))

    label3_1 = np.array(pd.read_csv("./Scripts/Results/TrainLabel_(3)1.csv", header=None))
    label3_2 = np.array(pd.read_csv("./Scripts/Results/TrainLabel_(3)2.csv", header=None))
    assignment3 = {}
    for x, y in label3_1:
        l = assignment3.setdefault(x, [])
        l.append(y)
        assignment3[x] = l

    for x, y in label3_2:
        l = assignment3.setdefault(x, [])
        l.append(y)
        assignment3[x] = l

    for key, val in assignment3.items():
        if len(val) < 2:
            del assignment3[key]

    print '3 - ' + str(len(assignment3))

    label4_1 = np.array(pd.read_csv("./Scripts/Results/TrainLabel_(4)1.csv", header=None))
    label4_2 = np.array(pd.read_csv("./Scripts/Results/TrainLabel_(4)2.csv", header=None))
    assignment4 = {}
    for x, y in label4_1:
        l = assignment4.setdefault(x, [])
        l.append(y)
        assignment4[x] = l

    for x, y in label4_2:
        l = assignment4.setdefault(x, [])
        l.append(y)
        assignment4[x] = l

    for key, val in assignment4.items():
        if len(val) < 2:
            del assignment4[key]

    print '4 - ' + str(len(assignment4))

    label5_1 = np.array(pd.read_csv("./Scripts/Results/TrainLabel_(5)1.csv", header=None))
    label5_2 = np.array(pd.read_csv("./Scripts/Results/TrainLabel_(5)2.csv", header=None))
    assignment5 = {}
    for x, y in label5_1:
        l = assignment5.setdefault(x, [])
        l.append(y)
        assignment5[x] = l

    for x, y in label5_2:
        l = assignment5.setdefault(x, [])
        l.append(y)
        assignment5[x] = l

    for key, val in assignment5.items():
        if len(val) < 2:
            del assignment5[key]

    print '5 - ' + str(len(assignment5))


    label6 = np.array(pd.read_csv("./Scripts/Results/TrainLabel_(6).csv", header=None))
    assignment6 = makeAssignment(label6)
    print '6 - ' + str(len(assignment6))

    label7_1 = np.array(pd.read_csv("./Scripts/Results/TrainLabel_(7)1.csv", header=None))
    label7_2 = np.array(pd.read_csv("./Scripts/Results/TrainLabel_(7)2.csv", header=None))
    assignment7 = {}
    for x, y in label7_1:
        l = assignment7.setdefault(x, [])
        l.append(y)
        assignment7[x] = l

    for x, y in label7_2:
        l = assignment7.setdefault(x, [])
        l.append(y)
        assignment7[x] = l

    for key, val in assignment7.items():
        if len(val) < 2:
            del assignment7[key]

    print '7 - ' + str(len(assignment7))

    label8_1 = np.array(pd.read_csv("./Scripts/Results/TrainLabel_(8)1.csv", header=None))
    label8_2 = np.array(pd.read_csv("./Scripts/Results/TrainLabel_(8)2.csv", header=None))
    assignment8 = {}
    for x, y in label8_1:
        l = assignment8.setdefault(x, [])
        l.append(y)
        assignment8[x] = l

    for x, y in label8_2:
        l = assignment8.setdefault(x, [])
        l.append(y)
        assignment8[x] = l

    for key, val in assignment8.items():
        if len(val) < 2:
            del assignment8[key]

    print '8 - ' + str(len(assignment8))

    label9_1 = np.array(pd.read_csv("./Scripts/Results/TrainLabel_(9)1.csv", header=None))
    label9_2 = np.array(pd.read_csv("./Scripts/Results/TrainLabel_(9)2.csv", header=None))
    assignment9 = {}
    for x, y in label9_1:
        l = assignment9.setdefault(x, [])
        l.append(y)
        assignment9[x] = l

    for x, y in label9_2:
        l = assignment9.setdefault(x, [])
        l.append(y)
        assignment9[x] = l

    for key, val in assignment9.items():
        if len(val) < 2:
            del assignment9[key]

    print '9 - ' + str(len(assignment9))

    assignments = [assignment0, assignment1, assignment2, assignment3, assignment4, assignment5, assignment6, assignment7, assignment8, assignment9]
    assignment = {}

    for i in range(len(data)):
        for digit in range(10):
            if i+1 in assignments[digit]:
                assignment[i+1] = digit

    for x, y in seed:
        assignment[x] = y

    d = {}
    res = []
    for key, val in assignment.items():
        res.append([key, val])
        count = d.setdefault(val, 0)
        d[val] = count+1

    print d

    writeCSV(res, 'assignment5')


def _CCA(data, graph, n):
    cca = CCA(n_components=n)
    adjacencyMatrix = createAffinityMatrix(graph)
    cca.fit(data, adjacencyMatrix)
    X_c, Y_c = cca.transform(data, adjacencyMatrix)

    writeCSV(X_c, 'CCA_X')
    writeCSV(Y_c, 'CCA_Y')
    # return X_c

if __name__ == "__main__":
    path = sys.argv[1]
    output = sys.argv[2]

    print 'Loading data...'
    seed, data, graph = loadData(path)

    print 'Creating assignment...'
    # data2 = np.array(pd.read_csv('./data/Extracted_Features.csv', header=None))
    start = time.time()
    createLabel(seed, data, graph)
    # combineLabels2(data, seed)
    # assignment = np.array(pd.read_csv("./Scripts/Results/assignment5.csv", header=None))
    print time.time() - start 
    print 'Training...'
    # train = createTrain(data, assignment)
    # pred = runRandomForest(train, data, 50)
    # pred = runLabelSpreading(data, assignment)
    # writeCSV(pred, output, header=['Id', 'Label'])
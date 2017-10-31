# EXAMPLE USAGE: python3 Scripts/clustering_info_meta.py Scripts/Results/pca30.csv

# goal of this file: get useful info for spectral clustering
# - We need a graph representation of all 10,000 datapoints in order to cluster them,
#   but we're only given a graph for the first 6,000.
# - Hypothesis: There is some underlying logic used to generate the graph, and we can
#   reverse engineer that logic by investigating the difference between datapoints that
#   are 'connected', and those that aren't

# after running this script, it became apparent that there's no real correlation between
# whether two datapoints are connected and their cosine similarity

import numpy as np
import sys
import os
import math
import matplotlib.pyplot as plt
import random

# a and b are numpy arrays
def cos_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def load_graph():
    adj = {}
    for line in open(os.path.realpath("./data/Graph.csv"), 'r'):
        line = line.rstrip()
        if len(line) == 0:
            continue
        else:
            a, b = tuple([int(x) for x in line.split(',')])
            # convert to indices
            a = a-1
            b = b-1
            if a not in adj:
                adj[a] = set()
                adj[a].add(b)
            else:
                adj[a].add(b)

            if b not in adj:
                adj[b] = set()
                adj[b].add(a)
            else:
                adj[b].add(a)
    return adj

def save_bar_chart(values, increment, fname):
    # plot bar chart to show distribution with respect to cosine similarity:
    cap = max(values)
    base = min(values)
    print("base: {} cap: {}".format(base, cap))
    buckets = []
    labels = []
    val_to_bucket = {}
    for xi, x in enumerate(range(int(math.ceil(base / increment)), int(math.ceil(cap / increment)) + 1)):
        val_to_bucket[x] = xi
        buckets.append(0)
        labels.append("{}-{}".format(round((x - 1) * increment, 2), round(x * increment), 2))
    for similarity in values:
        val = int(math.ceil(similarity / increment))
        bucket = val_to_bucket[val]
        buckets[bucket] += 1

    ys = np.arange(len(buckets))
    plt.bar(ys, buckets, align='center')
    plt.xticks(ys, labels, rotation='vertical')
    plt.tight_layout()  # makes room for labels
    plt.savefig(fname)
    plt.close()

def main(matrix_path):
    data_matrix = np.loadtxt(open(matrix_path, "rb"), delimiter=",")
    data_matrix = data_matrix[:6000] # we only care about first 6000, because that's all the graph contains

    connected_similarities = []
    disconnected_similarities = []

    # gather all similarity values
    adj = load_graph()
    for a, bs in adj.items():
        for b in bs:
            connected_similarities.append(cos_similarity(data_matrix[a], data_matrix[b]))

    # gather the same number of dissimilarity values as there are similarity values
    while len(disconnected_similarities) < len(connected_similarities):
        a = random.randint(0,6000-1)
        b = random.randint(0,6000-1)
        if a != b and not (a in adj and b in adj[b]):
            disconnected_similarities.append(cos_similarity(data_matrix[a], data_matrix[b]))

    save_bar_chart(connected_similarities, .05, 'graphs/connections_distribution.png')
    save_bar_chart(disconnected_similarities, .05, 'graphs/unconnected_distribution.png')


# pass the file to run on
if __name__ == '__main__':
    main(sys.argv[1])
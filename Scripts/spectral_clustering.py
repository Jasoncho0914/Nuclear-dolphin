import numpy as np
from clustering_info_meta import load_graph
from sklearn.cluster import SpectralClustering

# returns the matrix to be used for spectral clustering
def get_affinity_matrix():

    original_graph = load_graph(False)

    # initialize matrix as 10,000x10,000 zeroes
    matrix = []
    dim = 6000
    for i in range(dim):
        matrix.append([0]*dim)

    # set edges that are in the graph to '1'
    for a, bs in original_graph.items():
        for b in bs:
            matrix[a][b] = 1

    return np.asmatrix(matrix)

def grade_against_seed(label_list):
    # get a mapping of seed to datapoints
    seeds = {}
    datapoints = []
    for line in open("data/Seed.csv"):
        line = line.rstrip()
        datapoint, cluster = tuple([int(x) for x in line.split(',')])
        datapoints.append(datapoint)
        if cluster in seeds:
            seeds[cluster].add(datapoint)
        else:
            seeds[cluster] = set([datapoint])

    # get mapping of calculated label to datapoints
    labels = {}
    for datapoint in datapoints:
        label = label_list[datapoint]
        if label not in labels:
            labels[label] = set([datapoint])
        else:
            labels[label].add(datapoint)

    # get a mapping of seed label to calculated label
    taken_seed_labels = set()
    correct = 0
    for calculated_label, calculated_set in labels.items():
        scores = []
        for seed_label, seed_set in seeds.items():
            if seed_label in taken_seed_labels:
                continue
            scores.append( (len(seed_set.intersection(calculated_set)), seed_label) )
        score, seed_label = max(scores)
        print("matching seed {} to label {} gives score {}".format(seed_label, calculated_label, score))
        taken_seed_labels.add(seed_label)
        correct += score

    for label, labelset in labels.items():
        print("Label: {} -> {}".format(label, labelset))
    for seed, seedset in seeds.items():
        print("Seed: {} -> {}".format(seed, seedset))


    print("Score: {}%".format(round(correct*100/60, 2)))

def main():
    data_matrix = np.ndarray.tolist(np.loadtxt(open("Scripts/Results/pca30.csv", "rb"), delimiter=","))
    print("Creating spectral matrix...")
    # spectral_matrix = get_affinity_matrix()
    print('Running Clustering...')
    clusterer = SpectralClustering(n_clusters=10, affinity='poly') # affinity='precomputed'
    labels = clusterer.fit_predict(data_matrix)

    grade_against_seed(labels)
    print("DONE")


if __name__ == '__main__':
    main()
import numpy as np
import pandas as pd
import sys
from sklearn.decomposition import PCA
import os
from matplotlib import pyplot

from Utility import writeCSV

# load txt suggested on stackoverflow
# data_matrix = np.asmatrix(np.loadtxt(open("data/Extracted_features.csv", "rb"), delimiter=","))
# pca = PCA(n_components=K)
# new_matrix = pca.fit_transform(data_matrix)
# np.savetxt('pca_data/pca{}.csv'.format(K), new_matrix, delimiter=',')
if __name__ == "__main__":
	K = int(sys.argv[1])
	data_matrix = np.asmatrix(np.loadtxt(open(os.path.realpath("../data/Extracted_features.csv"), "rb"), delimiter=","))
	pca = PCA(n_components=K)
	new_matrix = pca.fit_transform(data_matrix)
	np.savetxt('Results/pca{}.csv'.format(K), new_matrix, delimiter=',')

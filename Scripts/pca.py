import numpy as np
import pandas as pd
import sys
from sklearn.decomposition import PCA
import os
from matplotlib import pyplot

from Utility import writeCSV

#=======================================================================
#PCA 
#1. Go to Nuclear-Dolphin directory
#2. run 'python ./Scripts/pca.py [number of dimension to reduce to]'
#3. Output file name will be pca[# dimentsions].csv
#=======================================================================
if __name__ == "__main__":
	K = int(sys.argv[1])
	data_matrix = np.asmatrix(np.loadtxt(open(os.path.realpath("../data/Extracted_features.csv"), "rb"), delimiter=","))
	pca = PCA(n_components=K)
	new_matrix = pca.fit_transform(data_matrix)
	np.savetxt('Results/pca{}.csv'.format(K), new_matrix, delimiter=',')



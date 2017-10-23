import numpy as np
import pandas as pd
import sys
from sklearn.decomposition import PCA
from Utility import writeCSV

# load txt suggested on stackoverflow
# data_matrix = np.asmatrix(np.loadtxt(open("data/Extracted_features.csv", "rb"), delimiter=","))
# pca = PCA(n_components=K)
# new_matrix = pca.fit_transform(data_matrix)
# np.savetxt('pca_data/pca{}.csv'.format(K), new_matrix, delimiter=',')
if __name__ == "__main__":
	K = int(sys.argv[1])
	print K
	train = pd.read_csv('./data/Extracted_features.csv', header=None)
	pca = PCA(n_components=K)
	pca_train = pca.fit_transform(train.values)
	writeCSV(pca_train.tolist(), 'pca_Extracted_features' + str(K))

# data_mu = np.average(data_matrix, axis=1)
#
# cov_matrix = np.asmatrix(np.cov(data_matrix, rowvar=False))
#
# eigvals, eigvecs = np.linalg.eig(cov_matrix)
# eiglist = [(eigval, eigvec,) for eigval, eigvec in zip(eigvals, eigvecs)]
# eiglist.sort() # low to high
# eiglist.reverse() # high to low
#
# # print(len(eiglist))
# # print([x[0] for x in eiglist])
#
# k_eigvecs_matrix = np.zeros( (data_matrix.shape[1], K,) )
# for col_i, val_vec in enumerate(eiglist):
#     eigenvector = val_vec[1]
#     if col_i >= K:
#         break
#     np.copyto(k_eigvecs_matrix[:,col_i], eigenvector[0])
#     # print("should be equal:")
#     # print(eigenvector[0])
#     # print(k_eigvecs_matrix[:,col_i])
#
# Y = (data_matrix-data_mu) * k_eigvecs_matrix
# print(Y.shape)
# np.savetxt('reduced_features.csv', Y, delimiter=',')
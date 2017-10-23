import numpy as np
from sklearn.decomposition import PCA
import os
from matplotlib import pyplot

K = 20

# load txt suggested on stackoverflow
try:
    os.mkdir('pca_data')
except FileExistsError:
    pass
data_matrix = np.asmatrix(np.loadtxt(open("data/Extracted_features.csv", "rb"), delimiter=","))
pca = PCA(n_components=K)
new_matrix = pca.fit_transform(data_matrix)
np.savetxt('pca_data/pca{}.csv'.format(K), new_matrix, delimiter=',')


# data_mu = np.average(data_matrix, axis=1)
#
cov_matrix = np.asmatrix(np.cov(data_matrix, rowvar=False))

eigvals, eigvecs = np.linalg.eig(cov_matrix)
eiglist = [(eigval, eigvec,) for eigval, eigvec in zip(eigvals, eigvecs)]
eiglist.sort() # low to high
eiglist.reverse() # high to low

y = [val for val, _ in eiglist]
pyplot.scatter([x for x in range(len(y))], y)
pyplot.show()

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
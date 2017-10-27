import numpy as np
import os
from matplotlib import pyplot


# load txt suggested on stackoverflow
data_matrix = np.asmatrix(np.loadtxt(open("data/Extracted_features.csv", "rb"), delimiter=","))

cov_matrix = np.asmatrix(np.cov(data_matrix, rowvar=False))

eigvals, eigvecs = np.linalg.eig(cov_matrix)
eiglist = [(eigval, eigvec,) for eigval, eigvec in zip(eigvals, eigvecs)]
eiglist.sort() # low to high
eiglist.reverse() # high to low

y = [val for val, _ in eiglist]
pyplot.scatter([x for x in range(len(y))], y)
pyplot.show()
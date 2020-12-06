'''
File name: glove_template.py
Author: Fatih Mutlu, Loïc Busson, Nicolas Thierry d’Argenlieu
Date created: 23/11/2020
Date last modified: 06/12/2020
Python Version: 3.8.5

glove_template.py is a python file that takes a previously computed co-occurence matrix
and returns the matrix xs containing the word vectors computed with GloVe. 

GloVe uses the following loss function: J = \sum_{i,j}^{V} f(M_{ij})(w_i^Tw_j + b_i + c_j - log(M_{ij}))^2
For now, we took the biais terms b_i = c_j = 0 --> room to improvement

Details on the computation of the gradient can be found in the report.
'''

#!/usr/bin/env python3
from scipy.sparse import *
import numpy as np
import pickle
import random

DATA_PATH = "../data/"

def main():
    print("loading cooccurrence matrix")
    with open(DATA_PATH + 'cooc.pkl', 'rb') as f:
        M = pickle.load(f)
    print("{} nonzero entries".format(M.nnz))

    nmax = 100
    print("using nmax =", nmax, ", cooc.max() =", M.max())

    print("initializing embeddings")
    embedding_dim = 20
    xs = np.random.normal(size=(M.shape[0], embedding_dim))
    ys = np.random.normal(size=(M.shape[1], embedding_dim))

    eta = 0.001
    alpha = 3 / 4

    epochs = 10

    for epoch in range(epochs):
        print("epoch {}".format(epoch))
        for ix, jy, M_ij in zip(M.row, M.col, M.data):

            #Computing grad_xi and grad_yj
			e_ij = np.dot(x[ix, :], y[jy, :]) - np.log(n)
            fM_ij = min(1.0, (M_ij / nmax) ** alpha)
            grad_xi = 2 * fM_ij * e_ij * y[jy, :]
            grad_yj = 2 * fM_ij * e_ij * x[ix, :]

            #Updating the weights
            xs[ix, :] -= eta * grad_xi
            ys[jy, :] -= eta * grad_yj

    np.save(DATA_PATH + 'embeddings', xs)


if __name__ == '__main__':
    main()

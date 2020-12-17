"""Used to split and concatenate two matrices of embeddings (separated for storage issues)"""

import numpy as np

DATA_PATH = "../precomputed_data/"

def split(file, numbersplit):
    M = np.load(DATA_PATH + file)
    n,p = M.shape

    L = np.array_split(M,numbersplit)
    
    for i, submatrix in enumerate(L):
        np.save(DATA_PATH + 'embeddings_full_22epoch_{}dim_part{}'.format(p,i+1), submatrix)


def concatenate(list_of_files, data_path):
    L=[]
    for file in list_of_files:
        L.append(np.load(data_path + file, allow_pickle=True))

    return np.concatenate(L, axis = 0)


if __name__ == "__main__":
    split("embeddings_full_22epoch_250dim.npy",3)

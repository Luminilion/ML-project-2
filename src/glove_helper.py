"""Used to split and concatenate two matrices of embeddings (separated for storage issues)"""

DATA_PATH = "../data/"

def split(file, numbersplit):
    M = np.load(DATA_PATH + file)
    n,p = M.shape

    L = np.array_split(M,numbersplit)
    
    for i, submatrix in enumerate(L):
        np.save(DATA_PATH + 'embeddings_full_10epoch_{}dim_part{}'.format(p,i+1), submatrix)


def concatenate(list_of_files):
    L=[]
    for file in list_of_files:
        L.append(np.load(DATA_PATH + file))

    return np.concatenate(L, axis = 0)

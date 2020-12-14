"""Used to split and concatenate two matrices of embeddings (separated for storage issues)"""

DATA_PATH = "../data/"

def split(file):
    M = np.load(DATA_PATH + file)
    n,p = M.shape
    A, B = np.array_split()

    np.save(DATA_PATH + 'embeddings_full_10epoch_{}dim_firsthalf'.format(p), A)
    np.save(DATA_PATH + 'embeddings_full_10epoch_{}dim_secondhalf'.format(p), B)

    return A,B


def concatenate(file1, file2):
    A = np.load(DATA_PATH + file1)
    B = np.load(DATA_PATH + file2)

    return np.concatenate((A,B), axis = 0)

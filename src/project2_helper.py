"""Contains all helper functions of the main notebook."""

import ctypes, os
# Checking if the process is executed as admin (multiprocessing does not work otherwise) 
def isAdmin():
    """
    Verifies if the process is executed as administrator.
    """
    try:
        is_admin = (os.getuid() == 0)
    except AttributeError:
        is_admin = ctypes.windll.shell32.IsUserAnAdmin() != 0
    return is_admin


import numpy as np
from multiprocessing import cpu_count, Pool
 
def parallelize(data, func, partitions=9999):
    """
    Uses all CPU cores available to compute the function on each element of the data.
    """
    cores = cpu_count() #Number of CPU cores on your system
    partitions = partitions if partitions<=cores else cores
    print(f"Computing function on {partition} cores.")
    data_split = np.array_split(data, partitions)
    pool = Pool(cores)
    data = pd.concat(pool.map(func, data_split))
    pool.close()
    pool.join()
    return data
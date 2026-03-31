import numpy as np

def matrix_transpose(A):
    """
    Return the transpose of matrix A (swap rows and columns).
    """
    # Write code here
    arr = np.array(A)
    final_lst = []
    for i in range(arr.shape[1]):
        final_lst.append(arr[:,i])
    return np.array(final_lst)      


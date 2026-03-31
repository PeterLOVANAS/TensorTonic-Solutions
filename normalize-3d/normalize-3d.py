import numpy as np

def normalize_3d(v):
    """
    Normalize 3D vector(s) to unit length.
    """
    # Your code here
    vec = np.array(v, dtype=np.float64)
    axis = 1
    if vec.shape == (len(v),):
        axis = 0
    vec_norm = np.linalg.norm(vec, axis=axis, keepdims=True)
    vec = np.where(vec_norm > 10**(-10) , vec / vec_norm, 0)

        
    return vec
import numpy as np

def expected_value_discrete(x, p):
    """
    Returns: float expected value
    """
    # Write code here
    x = np.array(x, dtype = np.float64)
    p = np.array(p, dtype = np.float64)

    if np.sum(p) != 1:
        raise ValueError

    return np.sum(x*p)

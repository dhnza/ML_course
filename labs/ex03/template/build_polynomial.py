# -*- coding: utf-8 -*-
"""implement a polynomial basis function."""

import numpy as np
    
def build_poly(x, degree):
    """polynomial basis functions for input data x, for j=0 up to j=degree."""
    if (degree < 0): raise ValueError("degree must be positive")
    
    phi = np.zeros((len(x), degree+1))
    
    phi[:,0] = 1
    for d in range(1,degree+1):
        phi[:,d] = phi[:,d-1] * x

    return phi
# -*- coding: utf-8 -*-
"""Function used to compute the loss."""

import numpy as np

################################################################################
# MSE
################################################################################
def compute_loss(y, tx, w):
    """Calculate the loss.

    You can calculate the loss using mse.
    """
    N = len(y)
    e = y - (tx @ w)
    L = (0.5 / N) * np.dot(e, e)
    return L


################################################################################
# MAE
################################################################################
# def compute_loss(y, tx, w):
#     """Calculate the loss.
#
#     You can calculate the loss using mae.
#     """
#     N = len(y)
#     e = y - (tx @ w)
#     # The MAE is just the L1 norm of the error vector scaled by 1/N
#     L = (1.0 / N) * np.linalg.norm(e, 1)
#     return L

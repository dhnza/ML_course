# -*- coding: utf-8 -*-
"""Stochastic Gradient Descent"""

def compute_stoch_gradient(y, tx, w, batch_size):
    """Compute a stochastic gradient from just few examples n and their corresponding y_n labels."""
    for mb_y, mb_tx in batch_iter(y, tx, batch_size):
        grad = compute_gradient(mb_y, mb_tx, w)
    return grad


def stochastic_gradient_descent(
        y, tx, initial_w, batch_size, max_iters, gamma):
    """Stochastic gradient descent algorithm."""
    # Define parameters to store w and loss
    ws = [initial_w]
    losses = []
    w = initial_w
    for n_iter in range(max_iters):
        # compute gradient and loss
        grad = compute_stoch_gradient(y, tx, w, batch_size)
        loss = compute_loss(y, tx, w)

        # update w by gradient
        w = w - gamma*grad

        # store w and loss
        ws.append(w)
        losses.append(loss)
        print("Gradient Descent({bi}/{ti}): loss={l}, w0={w0}, w1={w1}".format(
              bi=n_iter, ti=max_iters - 1, l=loss, w0=w[0], w1=w[1]))

    return losses, ws

#!/usr/bin/env python3

import random
from typing import List, Tuple, Callable

import numpy as np
from sklearn.decomposition import NMF

Matrix = np.ndarray
eps = 1e-5

def matrixDivergence(m1 : Matrix, m2 : Matrix) -> float:
    m1 = np.maximum(m1, eps)
    m2 = np.maximum(m2, eps)
    return np.sum(m1 * np.log(m1 / m2) - m1 + m2)

def nmf(m : Matrix, k : int) -> Tuple[Matrix, Matrix]:

    model = NMF(n_components=k, init='random')
    w = model.fit_transform(np.array(m))
    h = model.components_

    # m = np.maximum(np.array(m), eps)
    # def nmf_iter(v : Matrix, w : Matrix, h : Matrix) -> Tuple[Matrix, Matrix]:
    #     h_prime = h * ((np.tranpose(w) @ v) / (np.tranpose(w) @ w @ h))
    #     w_prime = w * ((v @ np.tranpose(h_prime))
    #                    /
    #                    (w @ h_prime @ np.tranpose(h_prime)))
    #     return w_prime, h_prime

    # w = np.random.rand(m.shape[0], k)
    # h = np.random.rand(k, m.shape[1])

    # old_w = None
    # old_h = None
    # iters = 0

    # while not(matrixDivergence(m, w@h) < 0.001) \
    #       and ((w != old_w).any() or (h != old_h).any()):
    #     if iters % 100 == 0:
    #         # print("iter #{}:\nw:\n{},\nh:\n{},\nwh:\n{}\n"
    #         #       .format(iters, w, h, w @ h))
    #         print("Distance: {}".format(matrixDivergence(m, w @ h)))
    #     old_w = w
    #     old_h = h
    #     w, h = nmf_iter(m, w, h)
    #     iters += 1

    return w, h

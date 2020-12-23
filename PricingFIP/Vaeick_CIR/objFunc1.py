# -*- coding: utf-8 -*-
import numpy as np


def objFunc1(params, y, x1):

    y = np.array(y)
    x1 = np.array(x1)
    a0 = params[0]
    b1 = params[1]

    n = len(y)

    mae = (np.square(y - a0 - b1 * x1)).sum() / n

    return mae

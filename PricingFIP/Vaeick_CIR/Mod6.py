import numpy as np


def zero_coupon(tau, r0, kappa, theta, sigma, model):
    ''' Calculates the zero-coupon price based on the Vasicek/CIR model
        using the parameters provided

    :param tau:
    :type tau:
    :param r0:
    :type r0:
    :param kappa:
    :type kappa:
    :param theta:
    :type theta:
    :param sigma:
    :type sigma:
    :param model:
    :type model:
    '''
    if model == 'Vasicek':
        B = (1 - np.exp(-kappa * tau)) / kappa
        A = (theta - sigma ** 2 / (2 * kappa ** 2)) * \
            (B - tau) - (sigma ** 2 / (4 * kappa)) * B ** 2
    elif model == 'CIR':
        g = np.sqrt(kappa ** 2 + 2 * sigma ** 2)
        tmp = 2 * kappa * theta / sigma ** 2
        tmp1 = kappa * tau / 2
        tmp2 = g * tau / 2

        A = tmp * np.log(np.exp(tmp1) / (np.cosh(tmp2) +
                                         (kappa / g) * np.sinh(tmp2)))
        # B = 2. / (kappa + g * (1. / np.tanh(g * tau / 2)))
        tanh = np.tanh(g * tau / 2)
        B = 2. * tanh / (kappa * tanh + g)
    else:
        print('zero_coupon: model must be "Vasicek" or "CIR"!')
        return -1

    n2 = len(A)
    A_ = A
    B_ = B

    r_ = np.repeat(r0, n2)

    p = np.exp(A_ - B_ * r_)
    return p


def swapRates(tau, p, mat):
    '''Computes interpolated values for the swap
    rates using the parameters provided
    :param tau:
    :type tau:
    :param p:
    :type p:
    :param mat:
    :type mat:
    '''
    if len(tau) == 1:
        return -1
    tmax = mat[-1]

    ttemp = np.arange(0.5, tmax + 0.5, 0.5)
    ptemp = np.interp(ttemp, tau, p)

    dis = np.cumsum(ptemp)
    # dis = dis(:);

    # linear interpolation
    pmat = np.interp(mat, tau, p)

    index = (2 * mat).astype(int) - 1
    S = 100 * 2 * (1 - pmat) / dis[index]

    return S


def liborRates(tau, p, mat):
    '''Computes interpolated values of the libor
     rates between the given extreme values

    :param tau:
    :type tau:
    :param p:
    :type p:
    :param mat:
    :type mat:
    '''
    if len(tau) == 1:
        return -1
    pmat = np.interp(mat, tau, p)
    L = 100 * (1. / pmat - 1) / mat
    return L


def objFunc1(params, tau, LIBOR, SWAP, model):
    '''Utilizes a gradient-based approach to reduce the difference between
       the model values and the actual values to tune the parameters

    :param params:
    :type params:
    :param tau:
    :type tau:
    :param LIBOR:
    :type LIBOR:
    :param SWAP:
    :type SWAP:
    :param model:
    :type model:
    '''
    # unpack params
    r0 = params[0]
    kappa = params[1]
    theta = params[2]
    sigma = params[3]
    if r0 < 0:
        return -1
    if sigma < 0:
        return -2
    p = zero_coupon(tau, r0, kappa, theta, sigma, model)
    # now that we have zero-coupon bond prices p(t,T)
    # now it is time to calculate MODEL LIBOR rates and SWAP rates
    S = swapRates(tau, p, SWAP[:, 0])
    L = liborRates(tau, p, LIBOR[:, 0])

    # the goal is to minimize the distance between model rates and market rates
    rel1 = (S - SWAP[:, 1]) / SWAP[:, 1]
    rel2 = (L - LIBOR[:, 1]) / LIBOR[:, 1]

    # rel1 = (S-SWAP(:,2))
    # rel2 = (L-LIBOR(:,2))

    # mae = (sum(abs(rel1))+sum(abs(rel2)))
    mae = np.sum(rel1 ** 2) + np.sum(rel2 ** 2)

    return mae

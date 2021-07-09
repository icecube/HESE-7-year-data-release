import numpy as np
import scipy as sp
import scipy.special

import autodiff
import weighter


def gammaPriorPoissonLikelihood(k, alpha, beta):
    """Poisson distribution marginalized over the rate parameter, priored with
       a gamma distribution that has shape parameter alpha and inverse rate
       parameter beta.
    Parameters
    ----------
    k : int
        The number of observed events
    alpha : tuple
        Zeroeth element contains Gamma distribution shape parameter
        First element contains respective gradients
    beta : tuple
        Zeroeth element contains Gamma distribution inverse rate parameter
        First element contains respective gradients
    Returns
    -------
    2D tuple
        Zeroeth element contains the log likelihood
        First element contains respective gradients
    """
    val = autodiff.mul_grad(alpha, autodiff.log(beta))
    val = autodiff.plus_grad(val, autodiff.lgamma(autodiff.plus(alpha, k)))
    val = autodiff.minus(val, sp.special.loggamma(k + 1.0))
    val = autodiff.minus_grad(
        val, autodiff.mul_grad(autodiff.plus(alpha, k), autodiff.log1p(beta))
    )
    val = autodiff.minus_grad(val, autodiff.lgamma(alpha))

    return val


def poissonLikelihood(k, weight_sum):
    """Computes Log of the Poisson Likelihood.
    Parameters
    ----------
    k : int
        the number of observed events
    weight_sum : 2D tuple
        Zeroeth element contains the sum of the weighted MC event counts
        First element contains respective gradients
    Returns
    -------
    2D tuple
        Zeroeth element contains the log likelihood
        First element contains respective gradients
    """

    logw = autodiff.log(weight_sum)

    klogw = autodiff.mul_r(k, logw)

    klogw_minus_w = autodiff.minus_grad(klogw, weight_sum)

    llh = autodiff.minus(klogw_minus_w, sp.special.loggamma(k + 1))

    return llh


def LEff(k, weight_sum, weight_sq_sum):
    """Computes Log of the L_Eff Likelihood.
       This is the poisson likelihood, using a poisson distribution with
       rescaled rate parameter to describe the Monte Carlo expectation, and
       assuming a uniform prior on the rate parameter of the Monte Carlo.
       This is the main result of the paper arXiv:1901.04645
    Parameters
    ----------
    k : int
        the number of observed events
    weight_sum : 2D tuple
        Zeroeth element contains the sum of the weighted MC event counts
        First element contains respective gradients
    weight_sq_sum : 2D tuple
        Zeroeth element containsthe sum of the square of the weighted MC event counts
        First element contains respective gradients
    Returns
    -------
    2D tuple
        Zeroeth element contains the log likelihood
        First element contains respective gradients
    """

    # Return -inf for an ill formed likelihood or 0 without observation
    if weight_sum[0] <= 0 or weight_sq_sum[0] < 0:
        if k == 0:
            return np.array((0.0, np.zeros(len(weight_sum[1]))))
        else:
            return np.array((-np.inf, np.zeros(len(weight_sum[1]))))

    # Return the poisson likelihood in the appropriate limiting case
    if weight_sq_sum[0] == 0:
        return poissonLikelihood(k, weight_sum)

    alpha = autodiff.plus(
        autodiff.div_grad(autodiff.pow(weight_sum, 2), weight_sq_sum), 1.0
    )
    beta = autodiff.div_grad(weight_sum, weight_sq_sum)
    L = gammaPriorPoissonLikelihood(k, alpha, beta)
    return L


def computeLEff(k, weights):
    """Computes Log of the L_Eff Likelihood from a list of weights.
       This is the poisson likelihood, using a poisson distribution with
       rescaled rate parameter to describe the Monte Carlo expectation, and
       assuming a uniform prior on the rate parameter of the Monte Carlo.
       This is the main result of the paper arXiv:1901.04645
    Parameters
    ----------
    k : int
        the number of observed events
    weights : 2D tuple
        Zeroeth element contains list of the weighted MC events
        First element contains list of respective gradients
    Returns
    -------
    2D tuple
        Zeroeth element contains the log likelihood
        First element contains respective gradients
    """
    weight_sum = autodiff.sum(weights)
    weight_sq_sum = autodiff.sum(autodiff.pow(weights, 2))

    return LEff(k, weight_sum, weight_sq_sum)


def calcEffLLH(data, weights, bin_slices):
    """
    Computes and returns the effective log likelihood
    Parameters
    -----------
    data: array-like
        list of observed events in each analysis bin.
    weights: array-like
        list of sorted weights.
    bin_slices: array-like
        list of bin slices, where each slice picks out the elements in weights
        corresponding to an analysis bin.

    Returns
    --------
    tuple:
        Zeroth element is the effective log likelihood
        First element is the gradient of the effective log likelihood
    """
    llhs = []

    for i, bin_slice in enumerate(bin_slices):
        if bin_slice.stop - bin_slice.start == 0:
            continue
        llhs.append(
            computeLEff(data[i], (weights[0][bin_slice], weights[1][bin_slice]))
        )

    llhs = (np.array([llh[0] for llh in llhs]), np.array([llh[1] for llh in llhs]))
    llh = autodiff.sum(llhs)

    return llh


def calcLLH(
    params, parameter_names, priors, bin_slices, data, weighter_maker, livetime
):
    """
    Computes and returns the total negative log likelihood

    Returns
    --------
    tuple:
        Zeroth element is the total negative log likelihood
        First element is the gradient of the -llh
    """

    weights = weighter_maker.get_weights(livetime, parameter_names, params)

    PriorLLH = [0.0, np.zeros(shape=len(params)).astype(float)]

    # The loop calculates and adds the prior llh to the effective llh
    for i, (param, prior) in enumerate(zip(params, priors)):

        mu, sigma, low, high = prior

        if param > high or param < low:
            PriorLLH = (-np.inf, np.zeros(shape=len(params)).astype(float))
            break
        if mu == None:
            continue

        grad = np.zeros(shape=len(params)).astype(float)
        grad[i] = 1.0

        LLH = autodiff.normal_log_pdf((param, grad), mu, sigma)
        PriorLLH = autodiff.plus_grad(PriorLLH, LLH)

    EffLLH = calcEffLLH(data, weights, bin_slices)

    # Combine the effective llh and prior llh, and take the negative
    return autodiff.mul(autodiff.plus_grad(PriorLLH, EffLLH), -1.0)

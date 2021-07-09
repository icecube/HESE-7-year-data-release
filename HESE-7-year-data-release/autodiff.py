import numpy as np
import scipy as sp

# This file defines the functions used to calculate the values and gradients of
# various arithmetic operations
# The inputs and outputs of these functions come in two varieties:
# 1. a lone 'array_like' object that represents a value without a gradient
# 2. a tuple of two 'array_like' objects where the first element represents a
#    value and the second element represents the gradient of the value. In this
#    case the second element has the same shape as the first plus one extra
#    dimension

# Adds an extra dimension to a quantity so it may represent a gradient
def up(x):
    x = np.atleast_1d(x)
    return np.reshape(x, newshape=(x.shape + (1,)))


# Apply a slice to a value gradient tuple
def slice(xg, bin_slice):
    x, g = xg
    return x[bin_slice], g[bin_slice]


# Assign masked elements from one value gradient tuple to another
def assign_mask(xg0, xg1, mask):
    x0, grad0 = xg0
    x1, grad1 = xg1
    x0[mask] = x1[mask]
    xg0[mask] = xg1[mask]


# Add a value gradient tuple and a value
def plus(xg0, x1):
    x0, grad0 = xg0
    return x0 + x1, grad0


# Add a value and a value gradient tuple
def plus_r(x0, xg1):
    x1, grad1 = xg1
    return x1 + x0, grad1


# Add two value gradient tuples
def plus_grad(xg0, xg1):
    x0, grad0 = xg0
    x1, grad1 = xg1
    return x0 + x1, grad0 + grad1


# Compute the sum of a value gradient tuple
def sum(xg, axis=(0,)):
    x, g = xg
    return x.sum(axis=axis), g.sum(axis=axis)


# Subtract a value gradient tuple and a value
def minus(xg0, x1):
    return plus(xg0, -x1)


# Subtract a value and a value gradient tuple
def minus_r(x0, xg1):
    x1, grad1 = xg1
    return plus_r(x0, (-x1, -grad1))


# Subtract two value gradient tuples
def minus_grad(xg0, xg1):
    x1, grad1 = xg1
    return plus_grad(xg0, (-x1, -grad1))


# Multiply a value gradient tuple and a value
def mul(xg0, x1):
    x0, grad0 = xg0
    return x0 * x1, grad0 * up(x1)


# Multiply a value and a value gradient tuple
def mul_r(x0, xg1):
    x1, grad1 = xg1
    return x0 * x1, grad1 * up(x0)


# Multiply two value gradient tuples
def mul_grad(xg0, xg1):
    x0, grad0 = xg0
    x1, grad1 = xg1
    return x0 * x1, up(x1) * grad0 + up(x0) * grad1


# Divide a value gradient tuple and a value
def div(xg0, x1):
    x0, grad0 = xg0
    return x0 / x1, grad0 / up(x1)


# Divide a value and a value gradient tuple
def div_r(x0, xg1):
    x1, grad1 = xg1
    val = x0 / x1
    x0, x1 = up(x0), up(x1)
    grad = -up(val) / x1 * grad1
    return val, grad


# Divide two value gradient tuples
def div_grad(xg0, xg1):
    x0, grad0 = xg0
    x1, grad1 = xg1
    val = x0 / x1
    x0, x1 = up(x0), up(x1)
    grad = grad0 / x1 - up(val) / x1 * grad1
    return val, grad


# Take the power of a value gradient tuple to a value
def pow(xg0, x1):
    x0, grad0 = xg0
    val = x0 ** x1
    x0, x1 = up(x0), up(x1)
    grad = x1 * x0 ** (x1 - 1) * grad0
    return val, grad


# Take the power of a value to a value gradient tuple
def pow_r(x0, xg1):
    x1, grad1 = xg1
    val = x0 ** x1
    grad = up(val) * np.log(up(x0)) * grad1
    return val, grad


# Take the power of a value gradient tuple to another value gradient tuple
def pow_grad(xg0, xg1):
    x0, grad0 = xg0
    x1, grad1 = xg1
    val = x0 ** x1
    x0, x1 = up(x0), up(x1)
    grad = x1 * x0 ** (x1 - 1) * grad0 + up(val) * np.log(x0) * grad1
    return val, grad


# Take the natural log of a value gradient tuple
def log(xg0):
    x0, grad0 = xg0
    return np.log(x0), grad0 / up(x0)


# Take the base 10 log of a value gradient tuple
def log10(xg0):
    x0, grad0 = xg0
    return np.log10(x0), grad0 / (up(x0) * np.log(10.0))


# Take the base 2 log of a value gradient tuple
def log2(xg0):
    x0, grad0 = xg0
    return np.log2(x0), grad0 / (up(x0) * np.log(2.0))


# Take the square root of a value gradient tuple
def sqrt(xg0):
    x0, grad0 = xg0
    val = (np.sqrt(x0),)
    grad = grad0 / (2.0 * up(val))
    return val, grad


# Take the loggamma of a value gradient tuple
def lgamma(xg0):
    x0, grad0 = xg0
    val = sp.special.loggamma(x0)
    grad = sp.special.digamma(x0) * grad0
    return val, grad


# Compute the log of one plus a value gradient tuple
def log1p(xg0):
    x0, grad0 = xg0
    return np.log1p(x0), grad0 / up(x0 + 1.0)


# Compute the log of the pdf of a normal distribution evaluated at a value gradient tuple
def normal_log_pdf(xg0, mu, sigma):
    return minus_r(
        -0.5 * np.log(2.0 * np.pi) - np.log(sigma),
        div(pow(minus(xg0, mu), 2), 2.0 * sigma ** 2),
    )

# Methods for computing "Feldman Cousin's" style intervals for Poisson distributed signals
# Copied from https://github.com/austinschneider/feldman_cousins
# Author: Austin Schneider

import numpy as np
from scipy.stats import poisson
import collections

from functools import wraps


class memodict_(collections.OrderedDict):
    def __init__(self, f, maxsize=1):
        collections.OrderedDict.__init__(self)
        self.f = f
        self.maxsize = maxsize

    def __missing__(self, key):
        if len(self) == self.maxsize:
            self.popitem(last=False)
        ret = self[key] = self.f(*key)
        return ret

    def __call__(self, *args):
        return self.__getitem__(args)


def memodict(f, maxsize=1):
    """ Memoization decorator for a function taking a single argument """
    m = memodict_(f, maxsize)
    return m


lam = lambda k, mu: poisson.pmf(k, mu) if (k != 0 or mu != 0) else 1.0
poisson_pmf = memodict(lam, 2000)


def gen_next(info, last, pmf, r):
    inc = -1 + 2 * last
    info[last, 0] = info[last, 0] + inc
    info[last, 1] = pmf(info[last, 0])
    info[last, 2] = r(info[last, 0])

    return info, np.argmax(info[:, 2])


def gen_next_zero(info, pmf, r):
    info[1, 0] = info[1, 0] + 1
    info[1, 1] = pmf(info[1, 0])
    info[1, 2] = r(info[1, 0])

    return info, 1


def construct_poisson_interval(mu, bkg=None, alpha=0.9):
    if bkg is None:
        bkg = 0
    if mu + bkg == 0:
        return 0, 0
    pmf = lambda k: poisson_pmf(k, mu + bkg)
    r = memodict(lambda k: pmf(k) / poisson_pmf(k, max(0, k - bkg) + bkg), 200)

    # Search for the largest likelihood ratio
    # Start at the k closest to mu
    closest = max(np.around(mu + bkg), 1)

    # Walk right searching for largest r
    i = closest
    last_r = r(i)
    while True:
        i += 1
        next_r = r(i)
        if next_r < last_r:
            break
        last_r = next_r
    best_k_right = i - 1
    best_r_right = last_r

    # Walk left searching for largest r
    i = closest
    last_r = r(i)
    while True:
        i -= 1
        if i < 0:
            break
        next_r = r(i)
        if next_r < last_r:
            break
        last_r = next_r
    best_k_left = i + 1
    best_r_left = last_r

    best_k = max(
        [(best_k_left, best_r_left), (best_k_right, best_r_right)], key=lambda x: x[1]
    )[0]

    get_info = lambda k: np.array([k, pmf(k), r(k)])
    mm = lambda l: [min(l), max(l)]
    get_minmax = lambda minmax, new: mm(list(minmax) + [new])

    info = np.array([get_info(best_k), get_info(best_k + 1)])

    points = []
    points.append(info[0, 0])
    tot = info[0, 1]
    last = 0
    minmax = [info[0, 0], info[0, 0]]

    while minmax[0] != 0 and tot < alpha:
        inc = -1 + 2 * last
        info[last] = get_info(info[last, 0] + inc)
        last = np.argmax(info[:, 2])
        minmax = get_minmax(minmax, info[last, 0])
        points.append(info[last, 0])
        tot += info[last, 1]

    if tot < alpha and last == 0:
        last = 1
        minmax = get_minmax(minmax, info[last, 0])
        points.append(info[last, 0])
        tot += info[last, 1]

    while tot < alpha:
        info[last] = get_info(info[last, 0] + 1)
        minmax = get_minmax(minmax, info[last, 0])
        points.append(info[last, 0])
        tot += info[last, 1]

    return minmax


def walk_right(x, step, cond, lim=None):
    while cond(x) and (lim is None or x < lim):
        x += step
    if lim is not None and x > lim:
        x = lim
    return x


def walk_left(x, step, cond, lim=None):
    while cond(x) and (lim is None or x > lim):
        x -= step
    if lim is not None and x < lim:
        x = lim
    return x


def poisson_interval(k, bkg=None, alpha=0.9, epsilon=1e-10):
    is_inside = lambda x, i: x >= i[0] and x <= i[1]
    exit_condition = lambda x: is_inside(
        k, construct_poisson_interval(x, bkg=bkg, alpha=alpha)
    )
    entry_condition = lambda x: not is_inside(
        k, construct_poisson_interval(x, bkg=bkg, alpha=alpha)
    )
    step = 1.0
    left_bound = int(k)
    while True:
        right_bound = walk_right(left_bound, step, exit_condition)
        if abs(right_bound - left_bound) <= epsilon:
            break
        step /= 2.0
        left_bound = walk_left(right_bound, step, entry_condition, lim=0)
        if abs(right_bound - left_bound) <= epsilon:
            break
        step /= 2.0

    right = (left_bound + right_bound) / 2.0

    if k == 0:
        return (0.0, right)

    step = 1.0
    right_bound = int(k)
    while True:
        left_bound = walk_left(right_bound, step, exit_condition, lim=0)
        if abs(right_bound - left_bound) <= epsilon:
            break
        step /= 2.0
        right_bound = walk_right(left_bound, step, entry_condition)
        if abs(right_bound - left_bound) <= epsilon:
            break
        step /= 2.0

    left = (left_bound + right_bound) / 2.0

    return (left, right)

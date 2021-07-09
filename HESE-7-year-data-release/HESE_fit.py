"""
This file runs a fit of the data over a single power-law astrophysical flux
model. The initial value of the fits can be modified within the file or through
command line arguments. One can also choose to fix certain parameters in the
fit, where the fixed value is kept at the initial value
"""

import sys
import numpy as np
from scipy.optimize import fmin_l_bfgs_b
import argparse
import time

import weighter
import binning
import data_loader
import autodiff
import likelihood
import det_sys_weights

parser = argparse.ArgumentParser()

parser.add_argument(
    "--cr_delta_gamma", default=-0.05, type=float, help="set initial cosmic ray slope"
)
parser.add_argument(
    "--nunubar_ratio",
    default=1.0,
    type=float,
    help="set initial neutrino/antineutrino ratio",
)
parser.add_argument(
    "--anisotropy_scale",
    default=1.0,
    type=float,
    help="set initial ice anisotropy scale",
)
parser.add_argument(
    "--astro_gamma",
    default=2.5,
    type=float,
    help="set initial astrophysical spectral index",
)
parser.add_argument(
    "--astro_norm",
    default=6.0,
    type=float,
    help="set initial astrophysical six-neutrino flux normalization",
)
parser.add_argument(
    "--conv_norm",
    default=1.0,
    type=float,
    help="set initial atmospheric conventional neutrino flux normalization",
)
parser.add_argument(
    "--epsilon_dom",
    default=0.99,
    type=float,
    help="set initial DOM absolute energy scale",
)
parser.add_argument(
    "--epsilon_head_on",
    default=0.0,
    type=float,
    help="set initial DOM angular response",
)
parser.add_argument(
    "--muon_norm",
    default=1.0,
    type=float,
    help="set initial atmospheric muon flux normalization",
)
parser.add_argument(
    "--kpi_ratio",
    default=1.0,
    type=float,
    help="set initial kaon/pion ratio correction",
)
parser.add_argument(
    "--prompt_norm",
    default=1.0,
    type=float,
    help="set initial atmospheric prompt neutrino flux normalization",
)

parser.add_argument(
    "--fix_cr_delta_gamma", action="store_true", help="fix cosmic ray slope in fit"
)
parser.add_argument(
    "--fix_nunubar_ratio",
    action="store_true",
    help="fix neutrino/antineutrino ratio in fit",
)
parser.add_argument(
    "--fix_anisotropy_scale",
    action="store_true",
    help="fix ice anisotropy scale in fit",
)
parser.add_argument(
    "--fix_astro_gamma",
    action="store_true",
    help="fix astrophysical spectral index in fit",
)
parser.add_argument(
    "--fix_astro_norm",
    action="store_true",
    help="fix astrophysical six-neutrino flux normalization in fit",
)
parser.add_argument(
    "--fix_conv_norm",
    action="store_true",
    help="fix atmospheric conventional neutrino flux normalization in fit",
)
parser.add_argument(
    "--fix_epsilon_dom",
    action="store_true",
    help="fix DOM absolute energy scale in fit",
)
parser.add_argument(
    "--fix_epsilon_head_on", action="store_true", help="fix DOM angular response in fit"
)
parser.add_argument(
    "--fix_muon_norm",
    action="store_true",
    help="fix atmospheric muon flux normalization in fit",
)
parser.add_argument(
    "--fix_kpi_ratio", action="store_true", help="fix kaon/pion ratio correction in fit"
)
parser.add_argument(
    "--fix_prompt_norm",
    action="store_true",
    help="fix atmospheric prompt neutrino flux normalization in fit",
)

args = parser.parse_args()

livetime = 227708167.68

parameter_names = [
    "cr_delta_gamma",
    "nunubar_ratio",
    "anisotropy_scale",
    "astro_gamma",
    "astro_norm",
    "conv_norm",
    "epsilon_dom",
    "epsilon_head_on",
    "muon_norm",
    "kpi_ratio",
    "prompt_norm",
]

params = np.array(
    [
        args.cr_delta_gamma,
        args.nunubar_ratio,
        args.anisotropy_scale,
        args.astro_gamma,
        args.astro_norm,
        args.conv_norm,
        args.epsilon_dom,
        args.epsilon_head_on,
        args.muon_norm,
        args.kpi_ratio,
        args.prompt_norm,
    ]
)

# Priors used in the fit. Each parameter has either a Gaussian or uniform prior.
# Zeroth column is the mean for a Gaussian Prior, None for a uniform prior
# First column is the standard deviation for a Gaussian Prior, None for a
# uniform prior
# Second column is the lower bound
# Third column is the upper bound
priors = [
    (-0.05, 0.05, -np.inf, np.inf),
    (1.0, 0.1, 0.0, 2.0),
    (1.0, 0.2, 0.0, 2.0),
    (None, None, -np.inf, np.inf),
    (None, None, 0.0, np.inf),
    (1.0, 0.4, 0.0, np.inf),
    (0.99, 0.1, 0.8, 1.25),
    (0.0, 0.5, -3.82, 2.18),
    (1.0, 0.5, 0.0, np.inf),
    (1.0, 0.1, 0.0, np.inf),
    (None, None, 0.0, np.inf),
]

# Check that all initial parameters are within prior bounds
for param_name, param, prior in zip(parameter_names, params, priors):
    param_min = prior[2]
    param_max = prior[3]
    if param < param_min or param > param_max:
        error_message = (
            "Given value for {}, {}, is outside of prior range [{},{}]".format(
                param_name, param, param_min, param_max
            )
        )
        raise ValueError(error_message)

# is_fixed dictates what parameters will be kept fixed during the fit. By
# default all values are set to False.
is_fixed = [
    args.fix_cr_delta_gamma,
    args.fix_nunubar_ratio,
    args.fix_anisotropy_scale,
    args.fix_astro_gamma,
    args.fix_astro_norm,
    args.fix_conv_norm,
    args.fix_epsilon_dom,
    args.fix_epsilon_head_on,
    args.fix_muon_norm,
    args.fix_kpi_ratio,
    args.fix_prompt_norm,
]

if np.any(is_fixed):
    print("Fixing parameters")
    for b, name, val in zip(is_fixed, parameter_names, params):
        if b:
            print(name + " = ", val)

is_fitted = [not b for b in is_fixed]

# Load MC and data file, and return an array of events within energy and length
# bounds.
mc_filenames = [
    "./resources/data/HESE_mc_observable.json",
    "./resources/data/HESE_mc_flux.json",
    "./resources/data/HESE_mc_truth.json",
]
mc = data_loader.load_mc(mc_filenames)
data = data_loader.load_data("./resources/data/HESE_data.json")

# bin_data takes an MC/data numpy array as input, and returns
# 0: the events rearranged such that events are grouped by analysis bins.
# 1: the list of bin slices for each analysis bin.
sorted_mc, mc_bin_slices = binning.bin_data(mc)
sorted_data, data_bin_slices = binning.bin_data(data)

# Counts the number of events in each analysis bin, to give the total observed
# events in each bin
binned_data = np.array([len(sorted_data[data_bin]) for data_bin in data_bin_slices])

# Sets up the Weighter class, that manages all the weight calculations
weight_maker = weighter.Weighter(sorted_mc)

# A wrapper function that handles fits with fixed parameters
def calcLLH_fitted_func(is_fitted, params):
    def func(
        fitted_params,
        parameter_names,
        priors,
        mc_bin_slices,
        binned_data,
        weights,
        livetime,
    ):
        params[:][is_fitted] = fitted_params
        llh, grads = likelihood.calcLLH(
            params,
            parameter_names,
            priors,
            mc_bin_slices,
            binned_data,
            weights,
            livetime,
        )
        return llh, np.array(grads[0])[is_fitted]

    return func


calcLLH = calcLLH_fitted_func(is_fitted, np.copy(params))

bounds_list = []
fitted_params_list = []
llh_list = []
info_list = []

# It has been observed that the log likelihood space is bimodal as a function
# of the DOM efficiency. To account for this, we split the allowed boundaries
# of the DOM efficiency in the fit, and separately fit for both sets of
# boundaries.
if args.fix_epsilon_dom:
    # If the DOM efficiency paramter is fixed, don't split the boundaries
    bounds = np.array([(prior[2], prior[3]) for prior in priors])
    bounds_list.append(bounds)
else:
    # If the DOM efficiency is fitted, split the allowed boundaries in DOM
    # Efficiency space and create two sets of boundaries.
    bounds = np.array([(prior[2], prior[3]) for prior in priors])
    index = parameter_names.index("epsilon_dom")
    bounds_low = np.copy(bounds)
    bounds_high = np.copy(bounds)
    bounds_low[index] = [0.8, 0.99]
    bounds_high[index] = [0.99, 1.25]

    bounds_list.append(bounds_low)
    bounds_list.append(bounds_high)

start = time.time()
print("Running fit")

for bounds in bounds_list:
    # Function that runs the fit.
    fitted_params, llh, info = fmin_l_bfgs_b(
        calcLLH,
        x0=params[is_fitted],
        args=(
            parameter_names,
            priors,
            mc_bin_slices,
            binned_data,
            weight_maker,
            livetime,
        ),
        bounds=bounds[is_fitted],
        m=10,
        pgtol=1e-18,
        factr=1e4,
    )

    fitted_params_list.append(fitted_params)
    llh_list.append(llh)
    info_list.append(info)

# Pick out the information from the fit with the lowest log likelihood.
min_index = np.argmin(llh_list)
BF_fitted_params = fitted_params_list[min_index]
BF_llh = llh_list[min_index]
BF_info = info_list[min_index]

end = time.time()

print("Fit took " + str(end - start) + " seconds")
BF_params = params[:]
BF_params[is_fitted] = BF_fitted_params

print("Best Fit -LLH: ", BF_llh)
print("Best Fit Paramters:")
for param, BF_param in zip(parameter_names, BF_params):
    print("\t{}: \t{}".format(param, BF_param))

print(BF_info)
print(BF_llh)
print(BF_params.tolist())

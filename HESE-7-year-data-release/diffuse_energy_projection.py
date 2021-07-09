import sys
import os
import os.path

base_path = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, base_path + "/resources/external/")
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.style

matplotlib.style.use("./resources/mpl/paper.mplstyle")
from matplotlib.font_manager import FontProperties
import scipy.stats

import data_loader
import weighter
import binning
import fc

# Load data/MC. By default load_mc loads events at energies >60 TeV, but we want to plot all events.
mc_filenames = [
    "./resources/data/HESE_mc_observable.json",
    "./resources/data/HESE_mc_flux.json",
    "./resources/data/HESE_mc_truth.json",
]
mc = data_loader.load_mc(mc_filenames, emin=10.0e3)
data = data_loader.load_data("./resources/data/HESE_data.json", emin=10.0e3)

e_edges, _, _ = binning.get_bins(emin=10.0e3)
bin_centers = 10.0 ** (0.5 * (np.log10(e_edges[:-1]) + np.log10(e_edges[1:])))

n_events, _ = np.histogram(data["recoDepositedEnergy"], bins=e_edges)

weight_maker = weighter.Weighter(mc)

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

# We plot using the best fit parameters found by HESE_fit.py
params = np.array(
    [
        -0.05309302,
        0.99815326,
        1.000683,
        2.87375956,
        6.36488608,
        1.00621679,
        0.95192328,
        -0.0548763,
        1.18706341,
        1.00013744,
        0.0,
    ]
)

livetime = 227708167.68

params_dict = dict(zip(parameter_names, params))

params_zeroed = params_dict.copy()

component_order = [
    ("muon_norm", "Atmo. Muons"),
    ("conv_norm", "Atmo. Conv."),
    ("prompt_norm", "Atmo. Prompt"),
    ("astro_norm", "Astro."),
]

for (zeroed_norm, _) in component_order:
    params_zeroed[zeroed_norm] = 0.0

# We want to separate the histogram by components, so we separately get the weights
# where the all normalization parameters but one are set to zero
weights = []
colors = []
labels = []
cm = plt.get_cmap("inferno")
color_scale = [cm(x) for x in [0.2, 0.55, 0.75, 0.9]]
for i, (zeroed_norm, zeroed_label) in enumerate(component_order):
    if params_dict[zeroed_norm] == 0.0:
        continue
    p_copy = params_zeroed.copy()
    p_copy[zeroed_norm] = params_dict[zeroed_norm]
    weights.append(
        weight_maker.get_weights(livetime, p_copy.keys(), p_copy.values())[0]
    )
    colors.append(color_scale[i])
    labels.append(zeroed_label)

fit, ax = plt.subplots(figsize=(7, 5))
plt.loglog()
plt.xlim(10.0e3, 10.0e6)
plt.ylim(1.0e-1, 4.0e1)
plt.xlabel("Deposited Energy [GeV]")
plt.ylabel("Events per 2635 days")

xerr = [bin_centers - e_edges[:-1], e_edges[1:] - bin_centers]
# The error bars are obtained using fc.py, a function writen by Austin Schneider
# and found at https://github.com/austinschneider/feldman_cousins/blob/master/fc.py
one_sigma_proportion = scipy.special.erf(1.0 / np.sqrt(2.0))
yerr = np.array(
    [
        (lambda x: [k - x[0], x[1] - k])(
            fc.poisson_interval(k, alpha=one_sigma_proportion)
        )
        for k in n_events
    ]
).T

plt.errorbar(
    bin_centers,
    n_events,
    xerr=xerr,
    yerr=yerr,
    color="black",
    marker=None,
    label="Data",
    linestyle="None",
    capsize=3,
    elinewidth=1,
)

plt.hist(
    len(weights) * [mc["recoDepositedEnergy"]],
    weights=weights,
    bins=e_edges,
    histtype="bar",
    stacked=True,
    label=labels,
    color=colors,
)

# This applies the visual filter that covers events that are not used
# in the analysis
mask_color = "#7ab9f3"
ax.fill_between(
    [10e3, 60e3],
    [1e3, 1e3],
    [0, 0],
    edgecolor="none",
    linewidth=0.0,
    facecolor=mask_color,
    zorder=3,
    alpha=0.7,
)
plt.axvline(x=60e3, linestyle="dashed")

font = FontProperties()
font.set_size("medium")
font.set_family("sans-serif")
font.set_weight("bold")

# Simply reverses the order of the legend labels.
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles[::-1], labels[::-1])

plt.tight_layout()
plt.show()

import sys
import os
import os.path
import matplotlib
import matplotlib.style

matplotlib.use("Agg")
matplotlib.style.use("./resources/mpl/paper.mplstyle")
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.collections import LineCollection
import functools

import json

outdir = "./effective_areas/"


def center(x):
    x = np.asarray(x)
    return (x[1:] + x[:-1]) / 2.0


def get_particle_masks(particleType):
    """
    Get a dictionary containing masks by particle type.
    """
    particle_dict = {
        "eminus": 11,
        "eplus": -11,
        "muminus": 13,
        "muplus": -13,
        "tauminus": 15,
        "tauplus": -15,
        "nue": 12,
        "nuebar": -12,
        "numu": 14,
        "numubar": -14,
        "nutau": 16,
        "nutaubar": -16,
    }
    abs_particle_dict = {
        "e": 11,
        "mu": 13,
        "tau": 15,
        "2nue": 12,
        "2numu": 14,
        "2nutau": 16,
    }
    other_particle_dict = {
        "nu": lambda x: (
            lambda xx: functools.reduce(
                np.logical_or, [(xx == 12), (xx == 14), (xx == 16)], np.zeros(xx.shape)
            )
        )(abs(np.array(x))),
        "all": lambda x: np.ones(np.array(x).shape).astype(bool),
    }
    masks = {}
    for name, id in particle_dict.items():
        mask = particleType == id
        if np.any(mask):
            masks[name] = mask
    for name, id in abs_particle_dict.items():
        mask = abs(particleType) == id
        if np.any(mask):
            masks[name] = mask
    for name, id in other_particle_dict.items():
        mask = id(particleType)
        if np.any(mask):
            masks[name] = mask
    return masks


mc_filenames = [
    "./resources/data/HESE_mc_observable.json",
    "./resources/data/HESE_mc_flux.json",
    "./resources/data/HESE_mc_truth.json",
]


def plot_effective_areas(json_files=mc_filenames):
    try:
        os.makedirs(outdir)
    except:
        pass

    # Load the MC
    json_data = dict()
    for filename in json_files:
        json_data.update(json.load(open(filename, "r")))

    # Get the MC generation information
    weight_over_flux_over_livetime = np.array(json_data["weightOverFluxOverLivetime"])

    # Choose the energy binning
    energy_bins = np.logspace(2, 7, 5 * 20 + 1)  # 1e2 to 1e7 with 20 bins per decade
    energy_bin_widths = np.diff(energy_bins)

    # Get neutrino interaction information from the file
    primaryEnergy = np.array(json_data["primaryEnergy"])
    interactionType = np.array(json_data["interactionType"])
    primaryType = np.array(json_data["primaryType"])

    # Get some masks that correspond to our chosen energy bins
    nu_energy_mapping = np.digitize(primaryEnergy, bins=energy_bins) - 1
    nu_energy_masks = [nu_energy_mapping == i for i in range(len(energy_bins) - 1)]

    # Get some masks that sort by interaction type
    interaction_types = [1, 2, 3]
    interaction_masks = [interactionType == i for i in interaction_types]
    CC_mask, NC_mask, GR_mask = interaction_masks

    # Get some masks that sort by primary particle type
    # Remember these are the relevant entries in the dictionary:
    """
        'nue',   'nuebar',   '2nue',
        'numu',  'numubar',  '2numu',
        'nutau', 'nutaubar', '2nutau',
        'mu', 'nu', 'all',
    """
    particle_masks = get_particle_masks(primaryType)

    ## Now we have what we need to compute the effective area ##

    # Choose the color map
    cm = plt.get_cmap("plasma")

    # Choose some line styles
    line_styles = ["-", "--", ":", ":", "-", "--", ":", ":"]

    # 3 flavors in the MC
    n_flavors = 3

    # We are going to average our effective area over the whole sky
    total_angular_width = 4.0 * np.pi

    bin_widths = energy_bin_widths * total_angular_width

    # A meter is 100cm
    meter = 100

    # How to compute and plot the effective area (in a histogram style with errors)
    def plot_line(ax, masks, color, line_style, label, factor=1.0):
        # Each entry in masks corresponds to an energy bin
        # The mask should define the events that contribute to the effective area calcualtion in that bin

        # Effective area is the sum of weightOverFluxOverLivetime, divided by bin width
        effective_area_cm2 = np.array(
            [
                np.sum(weight_over_flux_over_livetime[mask]) / bin_width
                for mask, bin_width in zip(masks, bin_widths)
            ]
        ) * factor
        # An additional factor may be needed if we are computing an average
        # effective area for multiple particle types

        # Compute the error on this quantity
        effective_area_cm2_error = np.array(
            [
                np.sqrt(np.sum(weight_over_flux_over_livetime[mask] ** 2)) / bin_width
                for mask, bin_width in zip(masks, bin_widths)
            ]
        ) * factor

        # Convert to meters^2
        effective_area_m2 = effective_area_cm2 / (meter ** 2)
        effective_area_m2_error = effective_area_cm2_error / (meter ** 2)

        # Plot things only if they will appear on the plot
        if np.any(effective_area_m2 > 1e-4):
            # Make plot of effective area
            ax.step(
                energy_bins[1:],
                effective_area_m2,
                color=color,
                linestyle=line_style,
                lw=2,
                label=label,
            )
            # Add the errorbars to the plot
            ax.errorbar(
                10 ** center(np.log10(energy_bins)),
                effective_area_m2,
                yerr=effective_area_m2_error,
                color=color,
                linestyle="none",
            )

    # How to format the axis
    def format_axis(ax):
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim((1e4, 1e7))
        ax.set_ylim((1e-4, 2e3))
        ax.set_xlabel(r"$\textmd{Neutrino Energy}~[\textmd{GeV}]$")
        ax.set_ylabel(r"$\textmd{Effective Area}~[\textmd{m}^2]$")

        # Override the yaxis tick settings
        major = 10.0 ** np.arange(-3, 5)
        minor = np.arange(2, 10) / 10.0
        locmaj = matplotlib.ticker.FixedLocator(10.0 ** np.arange(-2, 4))
        locmin = matplotlib.ticker.FixedLocator(
            np.tile(minor, len(major)) * np.repeat(major, len(minor))
        )
        locmaj = matplotlib.ticker.LogLocator(base=10.0, subs=(1,), numticks=12)
        locmin = matplotlib.ticker.LogLocator(base=10.0, subs=minor, numticks=12)
        ax.yaxis.set_major_locator(locmaj)
        ax.yaxis.set_minor_locator(locmin)

        ax.legend(frameon=True, loc="upper left")

    # How to save the figure with nice spacing
    def save(fig, name):
        fig.tight_layout()
        fig.savefig(outdir + "/" + name + ".pdf")
        fig.savefig(outdir + "/" + name + ".png", dpi=400)
        fig.clf()

    # Let's make an effective area vs. energy plot split by neutrino flavor
    fig, ax = plt.subplots(figsize=(7, 5))
    for flavor_index, flavor in enumerate(["e", "mu", "tau"]):
        color = cm((float(flavor_index) / float(n_flavors)) * 0.8 + 0.1)
        line_style = line_styles[0]
        flavor_string = flavor if flavor == "e" else "\\" + flavor
        label = (
            r"$"
            + r"\nu_{"
            + flavor_string
            + r"} + \bar\nu_{"
            + flavor_string
            + "}"
            + r"$"
        )
        particle_key = "2nu" + flavor
        particle_mask = particle_masks[particle_key]
        masks = np.logical_and(particle_mask[None, :], nu_energy_masks)
        # The factor of 0.5 is needed so that we compute the average
        # neutrino/antineutrino effective area. This is in contrast to the
        # effective area plot (FIG. 33) in PhysRevD.104.022002 which plots the
        # sum of the neutrino and antineutrino effective areas.
        plot_line(ax, masks, color, line_style, label, factor=0.5)
    format_axis(ax)
    save(fig, "effective_area_energy_2nu_flavor")

    # Make an effective area vs. energy plot split by neutrino type
    fig, ax = plt.subplots(figsize=(7, 5))
    for flavor_index, flavor in enumerate(["e", "mu", "tau"]):
        color = cm((float(flavor_index) / float(n_flavors)) * 0.8 + 0.1)
        for antiparticle_index, (antiparticle_suffix, is_antiparticle) in enumerate(
            zip(["", "bar"], [False, True])
        ):
            line_style = line_styles[antiparticle_index]
            label = (
                r"$"
                + (r"\bar" if is_antiparticle else "")
                + r"\nu_{"
                + (flavor if flavor == "e" else "\\" + flavor)
                + "}"
                + r"$"
            )
            particle_key = "nu" + flavor + antiparticle_suffix
            particle_mask = particle_masks[particle_key]
            masks = np.logical_and(particle_mask[None, :], nu_energy_masks)
            plot_line(ax, masks, color, line_style, label)
    format_axis(ax)
    save(fig, "effective_area_energy_nu_nubar_flavor")

    # Make an effective area vs. energy plot
    for antiparticle_index, (antiparticle_suffix, is_antiparticle) in enumerate(
        zip(["", "bar"], [False, True])
    ):
        fig, ax = plt.subplots(figsize=(7, 5))
        for flavor_index, flavor in enumerate(["e", "mu", "tau"]):
            color = cm((float(flavor_index) / float(n_flavors)) * 0.8 + 0.1)
            line_style = "-"
            label = (
                r"$"
                + (r"\bar" if is_antiparticle else "")
                + r"\nu_{"
                + (flavor if flavor == "e" else "\\" + flavor)
                + "}"
                + r"$"
            )
            particle_key = "nu" + flavor + antiparticle_suffix
            particle_mask = particle_masks[particle_key]
            masks = np.logical_and(particle_mask[None, :], nu_energy_masks)
            plot_line(ax, masks, color, line_style, label)
        format_axis(ax)
        save(
            fig,
            "effective_area_energy_nu" + ("bar" if is_antiparticle else "") + "_flavor",
        )

    # Make an effective area vs. energy plot split by neutrino type and interaction type
    for flavor_index, flavor in enumerate(["e", "mu", "tau"]):
        for antiparticle_index, (antiparticle_suffix, is_antiparticle) in enumerate(
            zip(["", "bar"], [False, True])
        ):
            line_style = "-"
            fig, ax = plt.subplots(figsize=(7, 5))
            for interaction_index, interaction in enumerate(["CC", "NC", "GR"]):
                color = cm((float(interaction_index) / float(3)) * 0.8 + 0.1)
                label = (
                    r"$"
                    + (r"\bar" if is_antiparticle else "")
                    + r"\nu_{"
                    + (flavor if flavor == "e" else "\\" + flavor)
                    + r"} \textmd{"
                    + interaction
                    + r"}$"
                )
                particle_key = "nu" + flavor + antiparticle_suffix
                particle_mask = particle_masks[particle_key]
                particle_mask = np.logical_and(
                    particle_mask, interaction_masks[interaction_index]
                )
                masks = np.logical_and(particle_mask[None, :], nu_energy_masks)
                plot_line(ax, masks, color, line_style, label)
            format_axis(ax)
            save(
                fig,
                "effective_area_energy_nu"
                + flavor
                + ("bar" if is_antiparticle else ""),
            )

    # Make an effective area vs. energy plot split by flavor and interaction type
    fig, ax = plt.subplots(figsize=(7, 5))
    for flavor_index, flavor in enumerate(["e", "mu", "tau"]):
        color = cm((float(flavor_index) / float(n_flavors)) * 0.8 + 0.1)
        for interaction_index, interaction in enumerate(["CC"]):
            line_style = line_styles[interaction_index]
            flavor_string = flavor if flavor == "e" else "\\" + flavor
            label = (
                r"$"
                + r"\nu_{"
                + flavor_string
                + r"} + \bar\nu_{"
                + flavor_string
                + r"} \textmd{"
                + interaction
                + r"}$"
            )
            particle_key = "2nu" + flavor
            particle_mask = particle_masks[particle_key]
            particle_mask = np.logical_and(
                particle_mask, interaction_masks[interaction_index]
            )
            masks = np.logical_and(particle_mask[None, :], nu_energy_masks)
            # The factor of 0.5 is needed so that we compute the average
            # neutrino/antineutrino effective area. This is in contrast to the
            # effective area plot (FIG. 33) in PhysRevD.104.022002 which plots the
            # sum of the neutrino and antineutrino effective areas.
            plot_line(ax, masks, color, line_style, label, factor=0.5)

    color = "#4d4d4d"
    line_style = line_styles[1]
    flavor_string = flavor if flavor == "e" else "\\" + flavor
    label = r"$\textmd{NC All Flavor}$"
    masks = np.logical_and(interaction_masks[1][None, :], nu_energy_masks)
    # The factor of 0.5 is needed so that we compute the average
    # neutrino/antineutrino effective area. This is in contrast to the
    # effective area plot (FIG. 33) in PhysRevD.104.022002 which plots the
    # sum of the neutrino and antineutrino effective areas.
    plot_line(ax, masks, color, line_style, label, factor=0.5)

    line_style = line_styles[2]
    color = cm((float(0) / float(n_flavors)) * 0.8 + 0.1)
    flavor = "e"
    flavor_string = flavor
    label = (
        r"$"
        + r"\nu_{"
        + flavor_string
        + r"} + \bar\nu_{"
        + flavor_string
        + r"} \textmd{GR}$"
    )
    particle_key = "2nu" + flavor
    particle_mask = particle_masks[particle_key]
    particle_mask = np.logical_and(particle_mask, interaction_masks[2])
    masks = np.logical_and(particle_mask[None, :], nu_energy_masks)
    # The factor of 0.5 is needed so that we compute the average
    # neutrino/antineutrino effective area. This is in contrast to the
    # effective area plot (FIG. 33) in PhysRevD.104.022002 which plots the
    # sum of the neutrino and antineutrino effective areas.
    plot_line(ax, masks, color, line_style, label, factor=0.5)

    format_axis(ax)
    save(fig, "effective_area_energy_2nu_flavor_interaction")


plot_effective_areas()

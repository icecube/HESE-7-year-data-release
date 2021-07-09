import numpy as np
import functools


def get_bins(
    emin=60.0e3,
    emax=1.0e7,
    ewidth=0.111,
    eedge=60.0e3,
    lmin=10.0,
    lmax=1000.0,
    lwidth=0.1,
    ledge=10.0,
    nzenith=10,
):
    """
    Get the analysis bins.
    Using argument defaults will return the default analysis binning used in the paper
    """

    # Store the target range
    target_emin = emin
    target_emax = emax
    target_lmin = lmin
    target_lmax = lmax

    # Walk outwards from the energy edge until target range is achieved
    n_edges = 1
    energy_edge = eedge
    emin = energy_edge
    while emin > target_emin:
        n_edges += 1
        emin /= 10.0 ** ewidth
    emax = energy_edge
    while emax < target_emax:
        n_edges += 1
        emax *= 10.0 ** ewidth

    # Compute the set of energy bin edges
    energy_bins = np.logspace(np.log10(emin), np.log10(emax), n_edges)

    # Compute the set of zenith bin edges
    zenith_bins = np.arccos(np.linspace(-1, 1, nzenith + 1))[::-1]

    # Walk outwards from the length edge until target range is achieved
    n_edges = 1
    length_edge = ledge
    lmin = length_edge
    while lmin > target_lmin:
        n_edges += 1
        lmin /= 10.0 ** lwidth
    lmax = length_edge
    while lmax < target_lmax:
        n_edges += 1
        lmax *= 10.0 ** lwidth

    # Compute the set of length bin edges
    length_bins = np.logspace(np.log10(lmin), np.log10(lmax), n_edges)

    # Return the bin edges for each observable
    return energy_bins, zenith_bins, length_bins


def get_bin_masks(
    morphology, energy, zenith, length, energy_bins, zenith_bins, length_bins
):
    """
    Get masks for all analysis bins
    Returns 3 sets of masks: cascade bins, track bins, double cascade bins
    """

    # Define the masks for the three topologies
    mask_0 = morphology == 0
    mask_1 = morphology == 1
    mask_2 = morphology == 2

    def make_bin_masks(energies, zeniths, energy_bins, zenith_bins):
        """
        Takes energy/zenith quantities and their bins
        Returns masks correponding to those bins
        """

        assert len(energies) == len(zeniths)

        n_energy_bins = len(energy_bins) - 1
        n_zenith_bins = len(zenith_bins) - 1

        energy_mapping = np.digitize(energies, bins=energy_bins) - 1
        zenith_mapping = np.digitize(zeniths, bins=zenith_bins) - 1
        bin_masks = []
        for j in range(n_zenith_bins):
            for k in range(n_energy_bins):
                mask = zenith_mapping == j
                mask = np.logical_and(mask, energy_mapping == k)
                bin_masks.append(mask)
        return bin_masks

    # Cascades are binned in energy and zenith
    masks_0 = np.logical_and(
        make_bin_masks(energy, zenith, energy_bins, zenith_bins), mask_0
    )

    # Tracks are binned in energy and zenith
    masks_1 = np.logical_and(
        make_bin_masks(energy, zenith, energy_bins, zenith_bins), mask_1
    )
    # Double Cascades are binned in energy and length
    masks_2 = np.logical_and(
        make_bin_masks(energy, length, energy_bins, length_bins), mask_2
    )

    return masks_0, masks_1, masks_2


def sort_by_bin(data, masks):
    """Returns the data/MC events sorted by masks, and returns the slices for each mask
    Parameters
    ----------
    data : array_like
        list of data/MC events
    masks : array_like
        list of masks
        masks should be the same length as data

    Return
    ---------
    sorted_data: array_like
        list of 'data', sorted by bin

    bin_slices: array_like
        list of bin slices. Each element in bin_slices, gives the slice that returns the data elements in a particular bin
    """

    no_mask = ~functools.reduce(np.logical_or, masks)
    masks = list(masks) + [no_mask]

    sorted_data = np.empty(data.shape, dtype=data.dtype)
    bin_edge = 0
    bin_slices = []
    for mask in masks:
        n_events = np.sum(mask)
        bin_slices.append(slice(bin_edge, bin_edge + n_events))
        sorted_data[bin_edge : bin_edge + n_events] = data[mask]
        bin_edge += n_events

    return sorted_data, bin_slices[:-1]


def bin_data(data):
    """Returns the data/MC events sorted by bins, and returns the bin slices for each bin

    Parameters
    ----------
    data : array_like
        list of data/MC events

    Return
    ---------
    sorted_data: array_like
        list of 'data', sorted by bin

    bin_slices: array_like
        list of bin slices. Each element in bin_slices, gives the slice that returns the data elements in a particular bin
    """

    # Get the bin edges
    energy_bins, zenith_bins, length_bins = get_bins()

    # Get the corresponding masks: data[mask] --> events in the bin corresponding to mask
    cascade_masks, track_masks, dcascades_masks = get_bin_masks(
        data["recoMorphology"],
        data["recoDepositedEnergy"],
        data["recoZenith"],
        data["recoLength"],
        energy_bins,
        zenith_bins,
        length_bins,
    )

    # Combine masks in one array
    masks = np.concatenate([cascade_masks, track_masks, dcascades_masks], axis=0)

    # Sort the data by bin
    sorted_data, bin_slices = sort_by_bin(data, masks)

    return sorted_data, bin_slices

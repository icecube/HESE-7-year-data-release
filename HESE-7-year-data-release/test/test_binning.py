import numpy as np

from .. import binning
from .. import data_loader


def test_get_bins():

    bins = binning.get_bins()

    assert len(bins) == 3

    energy_bins, zenith_bins, length_bins = bins

    assert len(energy_bins) == 22
    assert len(zenith_bins) == 11
    assert len(length_bins) == 21

    eedge = 10.0 ** (np.log10(60.0e3))

    assert np.all(np.diff(energy_bins) > 0.0)
    assert np.all(np.abs(np.diff(np.log10(energy_bins)) - 0.111) / 0.111 < 1.0e-10)
    assert np.any(np.abs(energy_bins - eedge) / eedge < 1.0e-10)
    assert (
        np.abs(np.amin(energy_bins) - 10.0 ** (np.log10(60.0e3)))
        / 10.0 ** (np.log10(60.0e3))
        < 1.0e-10
    )
    assert np.sum(energy_bins >= 1.0e7) == 1
    assert np.sum(energy_bins <= 10.0 ** (np.log10(60.0e3))) == 1

    assert np.all(np.diff(zenith_bins) > 0.0)
    assert np.all(np.abs(np.diff(np.cos(zenith_bins)) + 0.2) / 0.2 < 1.0e-10)
    assert np.any(np.abs(zenith_bins) < 1.0e-10)
    assert np.abs(np.amin(np.cos(zenith_bins)) + 1) < 1.0e-10
    assert np.abs(np.amax(np.cos(zenith_bins)) - 1) < 1.0e-10
    assert np.sum(np.cos(zenith_bins) >= 1.0) == 1
    assert np.sum(np.cos(zenith_bins) <= -1.0) == 1

    assert np.all(np.diff(length_bins) > 0.0)
    assert np.all(np.abs(np.diff(np.log10(length_bins)) - 0.1) / 0.1 < 1.0e-10)
    assert np.any(np.abs(length_bins - 10.0) / 10.0 < 1.0e-10)
    assert np.abs(np.amin(length_bins) - 10.0) / 10.0 < 1.0e-10
    assert np.abs(np.amax(length_bins) - 1000.0) / 1000.0 < 1.0e-10
    assert np.sum(length_bins >= 1000.0) == 1
    assert np.sum(length_bins <= 10.0) == 1


test_get_bins()

# HESE Data Release
Data release for IceCube's HESE 7.5 year analysis, as detailed in
[doi:10.1103/PhysRevD.104.022002][PRD] / [arXiv:2011.03545][arXiv].

Provided in this release are several json files. The first contains the 102 data
events that pass the HESE selection criterion. The others file contain information for
the MC events used to compute the expected data event rates. The contents of these
files will be described below.

As an instructive example, we provide a python3 script which
reproduces the fit of the data to a single power-law astrophysical flux in the
same manner as described in the text. The primary goal of these scripts are to
provide a working example utilizing the information provided in the data
files, and we encourage readers to use these files as a jumping point into
their own analyses.

## Data

For each of the 102 data events, we provide the following variables:
- `recoDepositedEnergy` - The reconstructed deposited energy of the event,
given in GeV.
- `recoMorphology` - The reconstructed morphology of the event, where 0, 1, 2,
correspond to cascades, tracks, and double cascades, respectively.
- `recoZenith` - The reconstructed zenith direction of the event, given in
radians.
- `recoLength` - The reconstructed length of the event, given in meters.

The HESE MC events are stored in the `HESE_mc_truth.json`, `HESE_mc_observable.json`,
and `HESE_mc_flux.json` files. For each event, we provide the following variables:
 - `primaryType` - The simulated initial particle flavor, given in the Monte
Carlo numbering scheme outlined by the [Particle Data Group][pdg].
 - `primaryEnergy` - The simulated true energy of the initial particle
 (neutrino or muon), given in GeV.
 - `primaryZenith` - The simulated true zenith direction of the initial
 particle (neutrino or muon).
 - `trueLength` - The simulated true length of the event, given in meters.
 - `interactionType` - The simulated neutrino interaction of the initial
 particle. Values of `1, 2, 3` correspond to CC, NR, and GR
 interactions, respectively. For simulated atmospheric muons a value of
 `0` is given.
 - `weightOverFluxOverLivetime` - The MC weight of the neutrino event, divided
 by the simulated flux and detector livetime, given in units of
 GeV sr cm^2. This is set to zero for atmospheric muon events.
 - `muonWeightOverLivetime` - The MC weight for each muon event, divided by the
 detector livetime. This is set to zero for neutrino events.
 - `pionFlux` - The nominal conventional atmospheric neutrino flux from pion
 decay, as described in the text. Fluxes are given in units of
 GeV^-1 s^-1 sr^-1 cm^-2.
 - `kaonFlux` - The nominal conventional atmospheric neutrino flux from kaon
 decay, as described in the text.
 - `promptFlux` - The nominal prompt atmospheric flux neutrino flux, as
 described in the text.
 - `conventionalSelfVetoCorrection` - The veto passing fraction for
 conventional neutrinos, as described in the text.
 - `promptSelfVetoCorrection` - The veto passing fraction for prompt
 neutrinos, as described in the text.
 - `recoDepositedEnergy` - The simulated reconstructed deposited energy of the
 event, given in GeV.
 - `recoMorphology` - The reconstructed morphology of the event. `0, 1, 2`
 correspond to cascades, tracks, and double cascades, respectively.
 - `recoLength` - The reconstructed length of the event, given in meters.
 - `recoZenith` - The reconstructed zenith angle of the event, given in radians.

## Prerequisites

In order to run the example scripts, the `PHOTOSPLINE` software package is
required. The package can be found [here][photospline]. In addition to the
instructions provided in the photospline `README.md`, the user must add the
`-DPYTHON_EXECUTABLE=<full path of python3>` option when running `cmake`. The
 python3 path can be found by running `which python3` on the command line.

## Code organization

Out of the box, the provided scripts simply fit the parameters of a single
power-law astrophysical flux model to the data. The goal is that these scripts
provide an example on how to use the provided information, as well as a
starting point for more complex analyses.

The provided files are summarized below:
- `HESE_fit.py` - This is the main script in the provided example. Simply
running `python3 HESE_fit.py` will fit the model to the data and print the
best fit parameters. Command line options exist where the user can selectively
choose to keep select parameters fixed, and to what value. Comments within
the file provide more detailed description.
- `data_loader.py` - This file defines the functions used to read in the data
and MC `json` files. Energy and length cuts are defined within this file, and
they can be modified.
- `binning.py` - This file defines the analysis binning, and can sort the
data/MC to group together events that are in the same analysis bins.
- `weighter.py` - This file handles the calculations of the weights for the MC
events. This file includes, for example, the calculation of the astrophysical
neutrino flux. If the user wishes to fit to a double power law astrophysical
flux, the new parameters and necessary functions would be added here.
- `det_sys_weights.py` - This file handles the calculations of the detector
systematic corrections to the weights. The corrections are calculated using
interpolating b-splines. These are stored as `fits` files in the `splines`
directory.
- `likelihood.py` - This file defines the functions that take in the final
weights and data events, and returns the negative log likelihood. The
statistical treatment used in this analysis derives from
[arXiv:1901.04645][arxivstat].
- `autodiff.py` - This file defines a set of functions allowing easy tracking
of mathematical operations on values and their gradients.

As an exercise for the reader, we suggest using the provided scripts to recreate
Fig. XXX from the paper, shown below. This plot shows the distribution of observed
and predicted event counts as a function of energy, broken up by the different
components and assuming the single power-law astrophysical flux model.
A script, `diffuse_energy_projection.py`, is provided which will make
this plot.

![spectrum](/HESE-7-year-data-release/resources/images/diffuse_energy_projection_all.png)

Another exercise for the reader is to recreate the HESE contours of Fig. XXX from the paper, shown below.
This plot shows the 68.3% and 95.4% confidence regions for the astrophysical components assuming the single
power-law astrophysical flux model, using the asymptotic approximation given by Wilks' theorem.
To make such a plot, one should scan over the 2D space in AstroNorm and AstroGamma, and run a fit at each
point while keeping AstroNorm and AstroGamma fixed. We provide a text file, `astro_scan_data.txt`,
that gives the resulting -LLH at each point, and a script, `astro_scan.py`, to plot these results.
The provided plotting script utilizes the `meander` package, which can be installed by running
`pip3 install meander`. (Note: A fit at each point can take ~10 minutes to complete,
so this exercise is not recommended without the use of multiprocessing or a computer cluster.)

![contour](/HESE-7-year-data-release/resources/images/spl_freq_scan_paper.png)

Finally, the effective area can be computed using the simulation information provided in `HESE_mc.json`.
To compute the effective area of the astrophysical neutrino flux, one must compute the sum of the `weightOverFluxOverLivetime`
quantity for events within a range of true MC parameters, then this must be divided by the bin widths in direction, and energy.
An example of this is provided in `plot_effective_areas.py` which produces several effective area plots, including a modified version
of FIG. 33 from the paper.

[arXiv]: https://arxiv.org/abs/2011.03545
[PRD]: https://doi.org/10.1103/PhysRevD.104.022002
[photospline]: https://github.com/IceCubeOpenSource/photospline
[pdg]: https://journals.aps.org/prd/abstract/10.1103/PhysRevD.98.030001
[arxivstat]: https://arxiv.org/abs/1901.04645

import numpy as np
import photospline
import autodiff


class SysWeighter:
    """
    This class handles the weight modifications due to detector systematic
    effects.
    Interpolating b-splines are used to perform these systematic corrections.
    There are different spline tables for each systematic type, particle type, and
    reconstructed morphology.
    """

    def __init__(
        self,
        mc,
        nominal_hole_ice_forward=0.0,
        nominal_dom_eff=0.99,
        nominal_anisotropy=1.0,
    ):

        self.mc = mc
        self.nominal_hole_ice_forward = nominal_hole_ice_forward
        self.nominal_dom_eff = nominal_dom_eff
        self.nominal_anisotropy = nominal_anisotropy

        particle_masks = []

        for particle_id in np.unique(self.mc["primaryType"]):
            for morphology_id in np.unique(self.mc["recoMorphology"]):
                indices = np.flatnonzero(
                    (self.mc["primaryType"] == particle_id)
                    & (self.mc["recoMorphology"] == morphology_id)
                )
                if indices.size > 0:
                    particle_masks.append([(particle_id, morphology_id), indices])

        # This member variable keeps track of the indices for each combination
        # of particle type and morphology type in the weights
        self.particle_masks = np.array(particle_masks)

        sys_name_list = [
            ("HoleIce", self.nominal_hole_ice_forward),
            ("DOMEff", self.nominal_dom_eff),
            ("Anisotropy", self.nominal_anisotropy),
        ]
        flux_component_list = ["Astro", "Conv", "Prompt"]

        # This member variable is a dictionary that keeps track of the events
        # whose values lie within the spline table's extents
        self.in_extents_dict = dict()

        # This member variable is a dictionary that stores the nominal
        # parameterized expectations for each systematic. These values
        # correspond to the denominators in equations E13 and E14 in
        # PhysRevD.104.022002
        self.nominal_vals_dict = dict()

        # This member variable is a dictionary that stores the splines
        self.splines = dict()

        for (p_id, m_id), mask in particle_masks:
            # Skip muons
            if np.abs(p_id) == 13:
                continue
            for (systematic_type, nominal_systematic_val) in sys_name_list:
                for flux_component in flux_component_list:

                    spline_table = self.get_spline(
                        systematic_type, flux_component, p_id, m_id
                    )
                    extents = np.array(spline_table.extents)

                    # Limit the extent of the anisotropy correction
                    if systematic_type == "Anisotropy" and m_id == 2:
                        extents[0, 1] = 2.3

                    def in_extents(x):
                        return np.all(
                            np.logical_and(x >= extents[:, 0], x <= extents[:, 1])
                        )

                    log10E = self.mc["log10E"][mask]
                    n = len(log10E)

                    if systematic_type == "Anisotropy":
                        if m_id != 2:
                            nominal_coords = np.array(
                                [log10E, np.full(n, nominal_systematic_val)]
                            ).T
                        else:
                            log10L = self.mc["log10L"][mask]
                            assert n == len(log10L)
                            nominal_coords = np.array(
                                [log10L, np.full(n, nominal_systematic_val)]
                            ).T

                    elif systematic_type == "HoleIce" and m_id == 2:
                        log10L = self.mc["log10L"][mask]
                        assert n == len(log10L)
                        nominal_coords = np.array(
                            [log10E, log10L, np.full(n, nominal_systematic_val)]
                        ).T

                    else:
                        cosZenith = self.mc["cosZenith"][mask]
                        assert n == len(cosZenith)
                        nominal_coords = np.array(
                            [log10E, cosZenith, np.full(n, nominal_systematic_val)]
                        ).T

                    # Save the indices of the coordinates that lie within the spline extents
                    in_extent_coords = np.flatnonzero(
                        np.array([in_extents(coord) for coord in nominal_coords])
                    )

                    self.in_extents_dict[
                        (p_id, m_id, systematic_type, flux_component)
                    ] = in_extent_coords

                    nominal_vals = spline_table.evaluate_simple(
                        nominal_coords[in_extent_coords].T, 0
                    )

                    self.nominal_vals_dict[
                        (p_id, m_id, systematic_type, flux_component)
                    ] = nominal_vals

    def get_spline(self, sys_name, flux_component, primary, morphology):
        if flux_component == "Astro":
            key = (sys_name, flux_component, primary, morphology)
        else:
            key = (sys_name, flux_component, morphology)

        if key in self.splines:
            return self.splines[key]

        morphology_string = {0: "shower", 1: "track", 2: "doublebang"}[morphology]

        if sys_name == "HoleIce":
            systematic_string = "HoleIceSplines/holeice_"
        elif sys_name == "DOMEff":
            systematic_string = "DOMEffSplines/domefficiency_"
        elif sys_name == "Anisotropy":
            systematic_string = "AnisotropySplines/tauanisotropy_"
        else:
            raise ValueError("sys_name must be either HoleIce, DOMEff, Anisotropy")

        if flux_component == "Astro":
            primary_string = {16: "tau", 14: "mu", 12: "e"}[np.abs(primary)]
            flux_component_string = "diffuseAstro_" + primary_string

        elif flux_component == "Conv":
            flux_component_string = "atmConv"

        elif flux_component == "Prompt":
            flux_component_string = "atmPrompt"

        else:
            raise ValueError("flux_component must be either Astro, Conv, or Prompt")

        spline_filename = (
            "./resources/splines/"
            + systematic_string
            + flux_component_string
            + "_"
            + morphology_string
            + ".fits"
        )

        spline = photospline.SplineTable(spline_filename)
        self.splines[key] = spline
        return spline

    def get_weights_grad(
        self,
        particle_masks,
        mc,
        flux_component,
        systematic_type,
        systematic_value_grad,
        nominal_systematic_val,
    ):
        """
        Return
        --------
        weights: array-like
            array of the systematic corrections applied to the weights
        gradients: array-like
            array of the gradients of the corrections
        """

        weights = np.ones(len(mc))
        gradients = np.zeros((len(mc), len(systematic_value_grad[1])))
        for (p_id, m_id), p_mask in particle_masks:
            # Skip muons
            if np.abs(p_id) == 13:
                continue

            spline_table = self.get_spline(systematic_type, flux_component, p_id, m_id)

            systematic_val, systematic_grad = systematic_value_grad

            # The list of indices that lie within the photospline extent.
            in_extent_mask = self.in_extents_dict[
                (p_id, m_id, systematic_type, flux_component)
            ]

            mask = p_mask[in_extent_mask]

            log10E = mc["log10E"][mask]
            n = len(log10E)

            if systematic_type == "Anisotropy":
                if m_id != 2:
                    coords = np.array([log10E, np.full(n, systematic_val)]).T
                else:
                    log10L = mc["log10L"][mask]
                    assert n == len(log10L)
                    coords = np.array([log10L, np.full(n, systematic_val)]).T

            elif systematic_type == "HoleIce" and m_id == 2:
                log10L = mc["log10L"][mask]
                assert n == len(log10L)
                coords = np.array([log10E, log10L, np.full(n, systematic_val)]).T

            else:
                cosZenith = mc["cosZenith"][mask]
                assert n == len(cosZenith)
                coords = np.array([log10E, cosZenith, np.full(n, systematic_val)]).T

            nominal_vals = self.nominal_vals_dict[
                (p_id, m_id, systematic_type, flux_component)
            ]

            vals = spline_table.evaluate_simple(coords.T, 0)

            if systematic_type == "Anisotropy":
                deriv_coord = 1 << 1
            else:
                deriv_coord = 1 << 2

            grads = spline_table.evaluate_simple(coords.T, deriv_coord)

            weights[mask], gradients[mask] = autodiff.pow_r(
                10.0,
                autodiff.minus(
                    (vals, systematic_grad * autodiff.up(grads)), nominal_vals
                ),
            )

        return weights, gradients

    def get_hole_ice_weights(self, flux_component, hole_ice_forward_val_grad):

        return self.get_weights_grad(
            self.particle_masks,
            self.mc,
            flux_component,
            "HoleIce",
            hole_ice_forward_val_grad,
            self.nominal_hole_ice_forward,
        )

    def get_dom_eff_weights(self, flux_component, dom_eff_val_grad):

        return self.get_weights_grad(
            self.particle_masks,
            self.mc,
            flux_component,
            "DOMEff",
            dom_eff_val_grad,
            self.nominal_dom_eff,
        )

    def get_anisotropy_weights(self, flux_component, anisotropy_val_grad):

        return self.get_weights_grad(
            self.particle_masks,
            self.mc,
            flux_component,
            "Anisotropy",
            anisotropy_val_grad,
            self.nominal_anisotropy,
        )

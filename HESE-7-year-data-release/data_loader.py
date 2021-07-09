import numpy as np
import json


def load_mc(filenames, emin=60.0e3, emax=1.0e7, lmin=10.0, lmax=1000.0):
    """
    Loads the MC json files and returns the MC events that lie within
    energy and length bounds.
    Energy range applies to all events.
    Length range applies only to double bang events.

    Return
    ---------
    data: array_like
        list of MC events within bounds
    """

    # Information is spread across multiple files
    # Merge dictionaries from each file
    file_contents = dict()
    for filename in filenames:
        file_contents.update(json.load(open(filename, "r")))

    expected_fields = [
        ("primaryEnergy", "<f8"),
        ("primaryZenith", "<f8"),
        ("primaryType", "<i8"),
        ("recoDepositedEnergy", "<f8"),
        ("recoZenith", "<f8"),
        ("recoLength", "<f8"),
        ("recoMorphology", "<i8"),
        ("weightOverFluxOverLivetime", "<f8"),
        ("muonWeightOverLivetime", "<f8"),
        ("pionFlux", "<f8"),
        ("kaonFlux", "<f8"),
        ("promptFlux", "<f8"),
        ("conventionalSelfVetoCorrection", "<f8"),
        ("promptSelfVetoCorrection", "<f8"),
    ]

    found_fields = np.array(
        [field in file_contents.keys() for field, field_type in expected_fields]
    )

    if not np.all(found_fields):
        raise RuntimeError(
            "MC files are missing " + str(expected_fields[~found_fields])
        )

    length_of_fields = np.unique([len(value) for value in file_contents.values()])

    if not len(length_of_fields) == 1:
        raise RuntimeError("Fields in json files have differing lengths")

    convenience_data = [("log10E", "<f8"), ("cosZenith", "<f8"), ("log10L", "<f8")]

    data = np.empty(
        length_of_fields, dtype=np.dtype(expected_fields + convenience_data)
    )

    for field_name, _ in expected_fields:
        data[field_name] = file_contents[field_name]

    data["log10E"] = np.log10(data["recoDepositedEnergy"])
    data["cosZenith"] = np.cos(data["recoZenith"])
    data["log10L"] = np.log10(data["recoLength"])

    energy_mask = np.logical_and(
        data["recoDepositedEnergy"] >= emin, data["recoDepositedEnergy"] <= emax
    )

    # Ignore errors when encoutering NaN length values
    with np.errstate(invalid="ignore"):
        length_mask = np.logical_and(
            data["recoLength"] >= lmin, data["recoLength"] <= lmax
        )

    double_cascade_length_mask = np.logical_or(data["recoMorphology"] != 2, length_mask)

    bounds_mask = np.logical_and(energy_mask, double_cascade_length_mask)

    return data[bounds_mask]


def load_data(filename, emin=60.0e3, emax=1.0e7, lmin=10.0, lmax=1000.0):
    """
    Loads the data json file and returns the data events that lie within
    energy and length bounds.
    Energy range applies to all events.
    Length range applies only to double bang events.

    Return
    ---------
    data: array_like
        list of data events within bounds
    """

    file_contents = json.load(open(filename, "r"))

    expected_fields = [
        ("recoDepositedEnergy", "<f8"),
        ("recoZenith", "<f8"),
        ("recoLength", "<f8"),
        ("recoMorphology", "<i8"),
    ]

    found_fields = np.array(
        [field in file_contents.keys() for field, field_type in expected_fields]
    )

    if not np.all(found_fields):
        raise RuntimeError(
            filename + " is missing " + str(expected_fields[~found_fields])
        )

    length_of_fields = np.unique([len(value) for value in file_contents.values()])

    if not len(length_of_fields) == 1:
        raise RuntimeError("Fields in json file have differing lengths")

    data = np.empty(length_of_fields, dtype=np.dtype(expected_fields))

    for field_name, _ in expected_fields:
        data[field_name] = file_contents[field_name]

    energy_mask = np.logical_and(
        data["recoDepositedEnergy"] >= emin, data["recoDepositedEnergy"] <= emax
    )

    with np.errstate(invalid="ignore"):
        length_mask = np.logical_and(
            data["recoLength"] >= lmin, data["recoLength"] <= lmax
        )

    double_cascade_length_mask = np.logical_or(data["recoMorphology"] != 2, length_mask)

    bounds_mask = np.logical_and(energy_mask, double_cascade_length_mask)

    return data[bounds_mask]


if __name__ == "__main__":
    mc_filenames = [
        "./resources/data/HESE_mc_observable.json",
        "./resources/data/HESE_mc_flux.json",
        "./resources/data/HESE_mc_truth.json",
    ]
    load_mc(mc_filenames)
    load_data("./resources/data/HESE_data.json")

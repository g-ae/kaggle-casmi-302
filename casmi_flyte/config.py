"""Data location and chemistry constants. Only stdlib imports."""

import os


# The training data, on SWITCH's S3 (not AWS, and not the cluster's own object store). Tasks read it
# with their own credentials, from the two Flyte secrets below; endpoint and region are not secret.
TRAIN_URI = os.getenv("CASMI_TRAIN_URI", "s3://302-data/kaggle_CASMI2026/train.parquet")
SOURCE_S3_ENDPOINT = os.getenv("CASMI_S3_ENDPOINT", "https://zhw-a.s3.cloud.switch.ch")
SOURCE_S3_REGION = os.getenv("CASMI_S3_REGION", "ch")
SOURCE_S3_SECRETS = ("casmi-s3-access-key-id", "casmi-s3-secret-access-key")  # `flyte create secret ...`

# RDKit version used by the competition metric.
RDKIT_VERSION = "2026.3.3"

# The ten adducts present in the test set, with the mass shift (Da) and multiplier for M:
#   precursor_mz = n_mol * M + shift   (charge is always 1)
PROTON = 1.007276
TEST_ADDUCTS: dict[str, tuple[int, float]] = {
    "[M+H]+": (1, PROTON),
    "[M+NH4]+": (1, 18.033823),
    "[M-H2O+H]+": (1, PROTON - 18.010565),
    "[M-2H2O+H]+": (1, PROTON - 2 * 18.010565),
    "[M+Na]+": (1, 22.989218),
    "[M+K]+": (1, 38.963158),
    "[M-H]-": (1, -PROTON),
    "[M-H2O-H]-": (1, -PROTON - 18.010565),
    "[M+CH2O2-H]-": (1, 44.998201),
    "[M+Cl]-": (1, 34.969402),
}


def neutral_mass(precursor_mz: float, adduct: str) -> float:
    n_mol, shift = TEST_ADDUCTS[adduct]
    return (precursor_mz - shift) / n_mol

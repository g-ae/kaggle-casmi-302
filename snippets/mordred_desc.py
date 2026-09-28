"""Mordred 2D descriptors (~1,600 float32 per molecule, NaN where a descriptor can't be computed).

Needs: `mordred==1.2.0` (CLASS.md, R1) and whatever it takes to install it: finding that is yours.
Returns ((n, n_descriptors) float32, valid). Time it on a few hundred real molecules before a full run.
"""

import numpy as np


def featurize_smiles(smiles: list[str]) -> tuple[np.ndarray, np.ndarray]:
    from mordred import Calculator, descriptors
    from rdkit import Chem, RDLogger

    RDLogger.DisableLog("rdApp.*")
    calc = Calculator(descriptors, ignore_3D=True)
    out = np.full((len(smiles), len(calc.descriptors)), np.nan, dtype=np.float32)
    valid = np.zeros(len(smiles), dtype=bool)
    for i, s in enumerate(smiles):
        mol = Chem.MolFromSmiles(s)
        if mol is None:
            continue
        out[i] = np.array(list(calc(mol).fill_missing(np.nan).values()), dtype=np.float64).clip(-1e30, 1e30)
        valid[i] = True
    return out, valid

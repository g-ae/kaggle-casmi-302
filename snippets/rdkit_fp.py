"""RDKit fingerprints and the monoisotopic mass (the mass is what the evaluation filters candidates on).

Needs: `pip install rdkit==2026.3.3 numpy` (the RDKit version the competition metric uses).
Returns ({"morgan2", "atompair", "torsion": (n, 2048), "maccs": (n, 167)} uint8 bits, exact_mass, valid).
"""

import numpy as np

N_BITS = 2048


def featurize_smiles(smiles: list[str]) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray]:
    """Returns ({representation: (n, d) array}, exact_mass (n,), valid (n,)). Importable for local tests."""
    from rdkit import Chem, RDLogger
    from rdkit.Chem import Descriptors, MACCSkeys, rdFingerprintGenerator

    RDLogger.DisableLog("rdApp.*")
    generators = {
        "morgan2": rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=N_BITS),
        "atompair": rdFingerprintGenerator.GetAtomPairGenerator(fpSize=N_BITS),
        "torsion": rdFingerprintGenerator.GetTopologicalTorsionGenerator(fpSize=N_BITS),
    }
    n = len(smiles)
    out = {name: np.zeros((n, N_BITS), dtype=np.uint8) for name in generators}
    out["maccs"] = np.zeros((n, 167), dtype=np.uint8)
    mass = np.full(n, np.nan)
    valid = np.zeros(n, dtype=bool)
    for i, s in enumerate(smiles):
        mol = Chem.MolFromSmiles(s)
        if mol is None:
            continue
        for name, gen in generators.items():
            out[name][i] = gen.GetFingerprintAsNumPy(mol)
        out["maccs"][i] = np.array(MACCSkeys.GenMACCSKeys(mol), dtype=np.uint8)
        mass[i] = Descriptors.ExactMolWt(mol)
        valid[i] = True
    return out, mass, valid

"""CheMeleon embeddings (https://github.com/JacksonBurns/chemeleon), loaded from a local weights file.

Needs: `pip install chemprop` (brings torch and RDKit) and the weights file `chemeleon_mp.pt`
from https://zenodo.org/records/15460715. Returns ((n, 2048) float32 embeddings, valid).
"""

import numpy as np

CHEMELEON_WEIGHTS = "/path/to/chemeleon_mp.pt"  # wherever your image puts the weights


class CheMeleonFingerprint:
    """Adapted from chemeleon_fingerprint.py in the CheMeleon repository (MIT): loads local weights."""

    def __init__(self, weights: str = CHEMELEON_WEIGHTS):
        import torch
        from chemprop import featurizers, nn
        from chemprop.models import MPNN
        from chemprop.nn import RegressionFFN

        self.featurizer = featurizers.SimpleMoleculeMolGraphFeaturizer()
        ckpt = torch.load(weights, weights_only=True)
        mp = nn.BondMessagePassing(**ckpt["hyper_parameters"])
        mp.load_state_dict(ckpt["state_dict"])
        self.model = MPNN(message_passing=mp, agg=nn.MeanAggregation(), predictor=RegressionFFN(input_dim=mp.output_dim))
        self.model.eval()
        self.dim = mp.output_dim

    def __call__(self, mols: list) -> np.ndarray:
        import torch
        from chemprop.data import BatchMolGraph

        bmg = BatchMolGraph([self.featurizer(m) for m in mols])
        with torch.no_grad():
            return self.model.fingerprint(bmg).numpy(force=True)


def featurize_smiles(smiles: list[str], batch_size: int = 256) -> tuple[np.ndarray, np.ndarray]:
    from rdkit import Chem, RDLogger

    RDLogger.DisableLog("rdApp.*")
    model = CheMeleonFingerprint()
    mols = [Chem.MolFromSmiles(s) for s in smiles]
    valid = np.array([m is not None and m.GetNumAtoms() > 0 for m in mols], dtype=bool)  # "" parses to 0 atoms
    out = np.zeros((len(smiles), model.dim), dtype=np.float32)  # also right for a shard with nothing valid
    idx = np.flatnonzero(valid)
    for start in range(0, len(idx), batch_size):
        batch = idx[start : start + batch_size]
        out[batch] = model([mols[i] for i in batch])
        if start // batch_size % 20 == 0:
            print(f"{start + len(batch):,}/{len(idx):,}")
    return out, valid

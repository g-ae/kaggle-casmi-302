"""CDK fingerprints from Python through JPype (CDK is a Java library).

Needs: a Java runtime, the CDK jar (https://github.com/cdk/cdk/releases), `pip install jpype1 numpy`.
Returns ({"pubchem": (n, 881), "klekota_roth": (n, 4860), "cdk_substructure": (n, 307)} uint8 bits, valid).
"""

import numpy as np

CDK_JAR = "/path/to/cdk-2.13.jar"  # wherever your image puts the jar


def featurize_smiles(smiles: list[str], jar: str = CDK_JAR) -> tuple[dict[str, np.ndarray], np.ndarray]:
    import jpype

    if not jpype.isJVMStarted():
        jpype.startJVM(classpath=[jar])
    cdk = jpype.JPackage("org").openscience.cdk
    builder = cdk.silent.SilentChemObjectBuilder.getInstance()
    parser = cdk.smiles.SmilesParser(builder)
    aromaticity = cdk.aromaticity.Aromaticity(
        cdk.aromaticity.ElectronDonation.cdk(), cdk.graph.Cycles.cdkAromaticSet()
    )
    fingerprinters = {
        "pubchem": cdk.fingerprint.PubchemFingerprinter(builder),
        "klekota_roth": cdk.fingerprint.KlekotaRothFingerprinter(),
        "cdk_substructure": cdk.fingerprint.SubstructureFingerprinter(),
    }
    n = len(smiles)
    out = {name: np.zeros((n, fp.getSize()), dtype=np.uint8) for name, fp in fingerprinters.items()}
    valid = np.zeros(n, dtype=bool)
    for i, s in enumerate(smiles):
        try:
            mol = parser.parseSmiles(s)
            cdk.tools.manipulator.AtomContainerManipulator.percieveAtomTypesAndConfigureAtoms(mol)
            aromaticity.apply(mol)
            for name, fp in fingerprinters.items():
                bits = fp.getBitFingerprint(mol).getSetbits()
                out[name][i, np.asarray(bits, dtype=np.int64)] = 1
            valid[i] = True
        except Exception as e:  # Java (unparsable SMILES, fingerprinter limits, ...) or Python side
            for bits in out.values():
                bits[i] = 0  # a failure halfway leaves no partial fingerprint behind
            print(f"CDK failed on {s}: {e}")
    return out, valid

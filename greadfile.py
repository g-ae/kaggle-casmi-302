import asyncio
from pathlib import Path

import flyte
from flyte.io import File
import os
import pyarrow.fs

from casmi_flyte.config import TRAIN_URI, SOURCE_S3_ENDPOINT, SOURCE_S3_REGION
from casmi_flyte.tables import write_table, read_table, matrix_column

from snippets.rdkit_fp import featurize_smiles as rdkit_featurize_smiles
from snippets.cdk_jpype import featurize_smiles as cdk_featurize_smiles
from snippets.mordred_desc import featurize_smiles as mordred_featurize_smiles

# IMAGES
base_image = (
    flyte.Image.from_debian_base(
        registry="registry.86.119.83.247.sslip.io/agile-badger",
        name="goncalo-base",
        platform=("linux/amd64",),
    )
    .with_pip_packages("pyarrow", "numpy")
    .with_source_folder(Path("casmi_flyte"))
    .with_source_folder(Path("snippets"))
)

rdkit = (
    flyte.Image.from_debian_base(
        registry="registry.86.119.83.247.sslip.io/agile-badger",
        name="goncalo-rdkit",
        platform=("linux/amd64",),
    )
    .with_pip_packages("pyarrow", "numpy", "rdkit==2026.3.3")
    .with_source_folder(Path("casmi_flyte"))
    .with_source_folder(Path("snippets"))
)

cdk = (
    flyte.Image.from_debian_base(
        registry="registry.86.119.83.247.sslip.io/agile-badger",
        name="goncalo-cdk",
        platform=("linux/amd64",),
    )
    .with_apt_packages("default-jre-headless", "curl")
    .with_pip_packages("pyarrow", "numpy", "jpype1")
    .with_commands(
        "curl -L -o /opt/cdk-2.13.jar https://github.com/cdk/cdk/releases/download/cdk-2.13/cdk-2.13.jar"
    )
    .with_source_folder(Path("casmi_flyte"))
    .with_source_folder(Path("snippets"))
)

mordred = (
    flyte.Image.from_debian_base(
        registry="registry.86.119.83.247.sslip.io/agile-badger",
        name="goncalo-mordred",
        platform=("linux/amd64",),
    )
    .with_pip_packages("pyarrow", "numpy<2", "mordred==1.2.0", "rdkit", "setuptools")  # mordred pas compatible avec numpy >=2
    .with_source_folder(Path("casmi_flyte"))
    .with_source_folder(Path("snippets"))
)

chemeleon = (
    flyte.Image.from_debian_base(
        registry="registry.86.119.83.247.sslip.io/agile-badger",
        name="goncalo-chemeleon",
        platform=("linux/amd64",),
    )
    .with_pip_packages("pyarrow", "numpy", "chemprop", "rdkit==2026.3.3", extra_index_urls=["https://download.pytorch.org/whl/cpu"])    # pytorche tourne uniquement sous CPU (plus léger et en accord avec les requirements du labo)
    .with_apt_packages("curl")
    .with_source_folder(Path("casmi_flyte"))
    .with_source_folder(Path("snippets"))
    .with_commands(
        "curl -L -o /opt/chemeleon_mp.pt https://zenodo.org/records/15460715/files/chemeleon_mp.pt?download=1"
    )
)

# ENVIRONNEMENTS
rdkitEnv = flyte.TaskEnvironment(
    name="feat-rdkit",         
    image=rdkit,
    secrets=[
        flyte.Secret(key="casmi-s3-secret-access-key", as_env_var="SECRET_KEY"),
        flyte.Secret(key="casmi-s3-access-key-id", as_env_var="ACCESS_KEY"),
    ]
)

cdkEnv = flyte.TaskEnvironment(
    name="feat-cdk",
    image=cdk,
    secrets=[
        flyte.Secret(key="casmi-s3-secret-access-key", as_env_var="SECRET_KEY"),
        flyte.Secret(key="casmi-s3-access-key-id", as_env_var="ACCESS_KEY"),
    ]
)

mordredEnv = flyte.TaskEnvironment(
    name="feat-mordred",
    image=mordred,
    secrets=[
        flyte.Secret(key="casmi-s3-secret-access-key", as_env_var="SECRET_KEY"),
        flyte.Secret(key="casmi-s3-access-key-id", as_env_var="ACCESS_KEY"),
    ]
)

chemeleonEnv = flyte.TaskEnvironment(
    name="feat-chemeleon",
    image=chemeleon,
    secrets=[
        flyte.Secret(key="casmi-s3-secret-access-key", as_env_var="SECRET_KEY"),
        flyte.Secret(key="casmi-s3-access-key-id", as_env_var="ACCESS_KEY"),
    ]
)

baseEnv = flyte.TaskEnvironment(
    name="base",         
    image=base_image,
    depends_on=[rdkitEnv, cdkEnv, mordredEnv, chemeleonEnv],
    secrets=[
        flyte.Secret(key="casmi-s3-secret-access-key", as_env_var="SECRET_KEY"),
        flyte.Secret(key="casmi-s3-access-key-id", as_env_var="ACCESS_KEY"),
    ]
)

# filesystem
def get_s3_filesystem() -> pyarrow.fs.S3FileSystem:
    return pyarrow.fs.S3FileSystem(
        endpoint_override=SOURCE_S3_ENDPOINT,
        region=SOURCE_S3_REGION,
        access_key=os.environ["ACCESS_KEY"],
        secret_key=os.environ["SECRET_KEY"],
    )

# Base from https://stackoverflow.com/a/66711534
def drop_duplicates(table: pyarrow.Table, column_name: str) -> pyarrow.Table:
    unique_values = pyarrow.compute.unique(table[column_name])
    indices = pyarrow.compute.index_in(unique_values, table[column_name])
    return table.take(indices)

@baseEnv.task(cache="auto")
async def extract_unique_molecules() -> File:
    import pyarrow.parquet as pq

    s3 = get_s3_filesystem()
    path = TRAIN_URI.removeprefix("s3://")
    table = pq.read_table(path, filesystem=s3, columns=["inchikey14", "normalized_smiles"])
    
    # deduplicate by inchikey14
    unique_table = drop_duplicates(table, "inchikey14")
    
    # print count, checkpoint 1
    print(f"Total spectra: {len(table):,}, Unique molecules: {len(unique_table):,}")
    
    return await write_table(unique_table, "unique_molecules.parquet")

@rdkitEnv.task(cache="auto")
async def featurize_rdkit(unique_molecules: File, sample_size: int | None = 1000) -> File:
    print("Launching RDKIT featurization on sample size of", sample_size or "whole dataset")
    table = await read_table(unique_molecules, columns=["inchikey14", "normalized_smiles"])
    inchikeys = table["inchikey14"].to_pylist()
    smiles_list = table["normalized_smiles"].to_pylist()

    # prendre sample
    if sample_size is not None:
        inchikeys = inchikeys[:sample_size]
        smiles_list = smiles_list[:sample_size]

    print("Number of unique smiles read:", len(smiles_list))
    
    features, mass, valid = rdkit_featurize_smiles(smiles_list)
    print("Number of valid molecules:", int(valid.sum()))

    rdkit_table = pyarrow.Table.from_arrays(
        [
            pyarrow.array(inchikeys),
            pyarrow.array(valid),
            pyarrow.array(mass),
            matrix_column(features["morgan2"]),
            matrix_column(features["atompair"]),
            matrix_column(features["torsion"]),
            matrix_column(features["maccs"]),
        ],
        names=["inchikey14", "valid", "exact_mass", "morgan2", "atompair", "torsion", "maccs"],
    )

    # write table for R1
    return await write_table(rdkit_table, "rdkit_features.parquet")

@cdkEnv.task(cache="auto")
async def featurize_cdk(unique_molecules: File, sample_size: int | None = 1000) -> File:
    from casmi_flyte.tables import read_table, write_table, matrix_column

    print("Launching CDK featurization on sample size of", sample_size or "whole dataset")
    table = await read_table(unique_molecules, columns=["inchikey14", "normalized_smiles"])
    inchikeys = table["inchikey14"].to_pylist()
    smiles_list = table["normalized_smiles"].to_pylist()

    # prendre sample
    if sample_size is not None:
        inchikeys = inchikeys[:sample_size]
        smiles_list = smiles_list[:sample_size]

    print("Number of unique smiles read:", len(smiles_list))

    out, valid = cdk_featurize_smiles(smiles_list)
    print("Number of valid molecules:", int(valid.sum()))

    cdk_table = pyarrow.Table.from_arrays(
        [
            pyarrow.array(inchikeys),
            pyarrow.array(valid),
            matrix_column(out["pubchem"]),
            matrix_column(out["klekota_roth"]),
            matrix_column(out["cdk_substructure"])
        ],
        names=["inchikey14", "valid", "pubchem", "klekota_roth", "cdk_substructure"],
    )

    # write table for R1
    return await write_table(cdk_table, "cdk_features.parquet")


@mordredEnv.task(cache="auto")
async def featurize_mordred(unique_molecules: File, sample_size: int | None = 1000) -> File:
    from casmi_flyte.tables import read_table, write_table, matrix_column

    print("Launching Mordred featurization on sample size of", sample_size or "whole dataset")
    table = await read_table(unique_molecules, columns=["inchikey14", "normalized_smiles"])
    inchikeys = table["inchikey14"].to_pylist()
    smiles_list = table["normalized_smiles"].to_pylist()

    # prendre sample
    if sample_size is not None:
        inchikeys = inchikeys[:sample_size]
        smiles_list = smiles_list[:sample_size]

    print("Number of unique smiles read:", len(smiles_list))

    out, valid = mordred_featurize_smiles(smiles_list)
    print("Number of valid molecules:", int(valid.sum()))

    # shape of out: (n_smiles, n_descriptors)

    # put all 2d descriptors in array
    mordred_table = pyarrow.Table.from_arrays(
        [
            pyarrow.array(inchikeys),
            pyarrow.array(valid),
            *[pyarrow.array(out[:, i]) for i in range(out.shape[1])]
        ],
        names=["inchikey14", "valid", *["desc_" + str(i) for i in range(out.shape[1])]],
    )

    # write table for R1
    return await write_table(mordred_table, "mordred_features.parquet")

@chemeleonEnv.task(cache="auto")
async def featurize_chemeleon(unique_molecules: File, sample_size: int | None = 1000) -> File:
    from casmi_flyte.tables import read_table, write_table, matrix_column
    
    print("Launching Chemeleon featurization on sample size of", sample_size or "whole dataset")
    table = await read_table(unique_molecules, columns=["inchikey14", "normalized_smiles"])
    inchikeys = table["inchikey14"].to_pylist()
    smiles_list = table["normalized_smiles"].to_pylist()

    # prendre sample
    if sample_size is not None:
        inchikeys = inchikeys[:sample_size]
        smiles_list = smiles_list[:sample_size]

    print("Number of unique smiles read:", len(smiles_list))

    out, valid = mordred_featurize_smiles(smiles_list)
    print("Number of valid molecules:", int(valid.sum()))

    # shape of out: (n_smiles, 2048)

    # put all 2d descriptors in array
    chem_table = pyarrow.Table.from_arrays(
        [
            pyarrow.array(inchikeys),
            pyarrow.array(valid),
            matrix_column(out)
        ],
        names=["inchikey14", "valid", "chemeleon"],
    )

    # write table for R1
    return await write_table(chem_table, "chemeleon_features.parquet")

@baseEnv.task
async def main(sample_size: int | None = 1000) -> list[File]:
    unique_molecules_file = await extract_unique_molecules()
    rdkit_file, cdk_file, mordred_file, chemeleon_file = await asyncio.gather(
        featurize_rdkit(unique_molecules_file, sample_size=sample_size),
        featurize_cdk(unique_molecules_file, sample_size=sample_size),
        featurize_mordred(unique_molecules_file, sample_size=sample_size),
        featurize_chemeleon(unique_molecules_file, sample_size=sample_size),
    )

    return [rdkit_file, cdk_file, mordred_file, chemeleon_file]
# Hands-on Flyte 2: a molecular feature factory for CASMI 2026

**Format:** one day (~6 h), your laptop plus a Flyte cluster on Switch Cloud shared by the whole
class, CPU only.
**You get:** requirements, a starter kit, and checkpoints. **You don't get:** a recipe.

## The story

The [Enveda CASMI 2026](https://www.kaggle.com/competitions/enveda-CASMI26-molecule-id-mass-spectra)
Kaggle competition asks: given the MS/MS spectra of an unknown molecule, return up to 25 candidate
structures (SMILES), best first. It is scored with MRR@25, and a guess counts as correct when its
InChIKey14 matches the answer.

One classic approach: **predict a molecular representation from the spectrum, then retrieve the
library molecules of the right mass whose representation is most similar.** Which representation
works best is an open question, and nobody can answer it until the representations exist.

Your job today is to build the pipeline that computes the molecular representations for the whole training set and scores
each one.

## Getting connected

You work in a **group**, and you write and launch pipelines on your laptops. They run on the class
cluster, in your group's Flyte project. Your instructor tells you your group's name (below:
`<group>`, e.g. `brave-otter`); it is also your group's registry project. You log in with your own GitHub
account (below: `<github-user>`). Do steps 1–2 **before the class**.

1. **GitHub.** Send your GitHub username to your instructor, then accept the invitation to the
   class organization `isc-302-flyte`. Your GitHub account is how you log in to Flyte and to the
   image registry.
2. **Registry login.** Open <https://registry.86.119.83.247.sslip.io>, click **Login via OIDC provider** and
   sign in with GitHub. That creates your registry account; your instructor then adds you to your
   group's registry project (named `<group>`). Open the user menu (top right) › **User Profile**
   and copy your **CLI secret**: it is your password for `docker login`, not your GitHub password.
3. **Flyte config.** Put this in `~/.flyte/config.yaml`, with your group's project (the same for
   everyone in the group, so you share runs and cache):
   ```yaml
   admin:
     endpoint: dns:///flyte.86.119.83.247.sslip.io
     insecure: false
     authType: Pkce
   image:
     builder: local
   task:
     domain: development
     project: <group>
   ```
   The first `flyte` command opens a browser for the GitHub login. The UI is at
   `https://flyte.86.119.83.247.sslip.io/v2`.
4. **Images.** You build images on your laptop with Docker and push them to the class registry
   (Harbor), into your group's project, **in lowercase** (the registry rejects upper-case names):
   ```bash
   docker login registry.86.119.83.247.sslip.io -u <github-user>
   ```
   (the password is the CLI secret from step 2)
   ```
   registry.86.119.83.247.sslip.io/<group>/<image-name>:<tag>
   ```
   You can push only to your group's project, which has a storage quota (40 GB). You can browse and
   delete your group's images in the registry's web UI. Your images are **public**: the cluster pulls them
   without credentials, and anyone can download them. One more reason never to put a key in an
   image.

You get no `kubectl` access: the Flyte UI, `flyte` CLI and task logs are your window on the cluster.

## The data

`train.parquet` has 2.5M spectra with known structures, 3 GB. It lives in an S3 bucket on
[SWITCH](https://www.switch.ch/) (S3-compatible, not AWS):

```
CASMI_TRAIN_URI = s3://302-data/kaggle_CASMI2026/train.parquet
endpoint        = https://zhw-a.s3.cloud.switch.ch      (region: ch)
```

Reading it needs an access key. It is **not** the storage the cluster uses for its own data, so
your tasks need to be given the credentials. A read-only key is already stored on the cluster as
two Flyte secrets, `casmi-s3-access-key-id` and `casmi-s3-secret-access-key`: see `flyte.Secret`
for how a task gets them (don't create your own). Never put keys in code or images.

Columns you will care about: `normalized_smiles`, `inchikey14`, `adduct`, `precursor_mz`,
`ms2_mzs`, `ms2_normalized_intensities`, `collision_energy_ev`, `ingest_lib`. The
[competition data page](https://www.kaggle.com/competitions/enveda-CASMI26-molecule-id-mass-spectra/data)
documents all of them. Look at the data before you design anything.

## Requirements

**R1. Features.** For each featurizer below, produce one parquet table with **one row per distinct
training structure**, keyed by `inchikey14`. Include a boolean `valid` column and one column per
representation.

| Featurizer | Version (pinned) | Representations |
|---|---|---|
| [RDKit](https://www.rdkit.org) | `rdkit==2026.3.3`, the version the competition metric uses | Morgan radius 2 (2048 bits), MACCS keys, atom pairs (2048), topological torsions (2048), monoisotopic mass |
| [CDK](https://cdk.github.io) | CDK 2.13 (Java), driven from Python with JPype | PubChem (881 bits), Klekota-Roth (4860 bits), CDK substructure keys (307 bits) |
| [Mordred](https://github.com/mordred-descriptor/mordred) | `mordred==1.2.0`, the original release | all 2D descriptors |
| [CheMeleon](https://github.com/JacksonBurns/chemeleon) | weights `chemeleon_mp.pt` from [Zenodo record 15460715](https://zenodo.org/records/15460715) | 2048-d learned embedding |

**R2. Evaluation.** Build a train/hold-out split of the spectra, then run the provided evaluation
(`baseline.py`) once per representation. The run must end with a report (the task's *Report*
tab in the UI) ranking the representations by MRR@25. The hold-out should look like the
Kaggle test set:
- about 400 molecules measured on the timsTOF (`ingest_lib` `enveda-180` or `enveda-np-examples`)
- only the 10 test adducts
- **no spectrum of a hold-out molecule in training**

**R3. Constraints.**
- CPU only, on the shared class cluster. The other groups' pipelines run next to yours: ask for the
  resources you need, not more. Full runs are launched in waves; your instructor tells you when.
- A **development run** (on a sample of the data) takes under 5 minutes once images are built.
- The **full run** takes under ~2 hours. If a featurizer can't cover every molecule within
  that budget, cover what you can and report its coverage.
- Rerunning with unchanged code and inputs recomputes nothing.
- One molecule a toolkit can't parse must not fail the run.

**R4. Deliverable.** A single `flyte.run(...)` entry point with parameters for the development
and full runs, a link to a successful full run, and the report table.

## Starter kit

The starter kit is
[isc-302-flyte/kaggle-casmi-flyte-starter](https://github.com/isc-302-flyte/kaggle-casmi-flyte-starter).
Start your own repository from it (**Use this template**), or just clone it. Everything unrelated
to orchestration is provided, so you can spend the day on the pipeline:

- `casmi_flyte/metric.py`: MRR@25.
- `casmi_flyte/config.py`: the 10 test adducts and `neutral_mass(precursor_mz, adduct)`.
- `casmi_flyte/baseline.py`: `bin_spectra()` and the evaluation logic (train an MLP to predict a
  representation from spectra, retrieve candidates by mass, compute MRR@25). You decide where
  and how it runs.
- `casmi_flyte/tables.py`: parquet ↔ `flyte.io.File` helpers, fingerprint matrix ↔ column, and
  polars helpers to merge parquet files (streaming) and scan them lazily. The evaluation needs polars.
- `snippets/`: the featurizer calls, so you don't spend the day on library parameters.
  `rdkit_fp.py` (the four fingerprints and the monoisotopic mass), `mordred_desc.py` (all 2D
  descriptors), `cdk_jpype.py` (CDK fingerprints through JPype) and `chemeleon.py` (CheMeleon
  embeddings from a local weights file). Each is a plain function, SMILES in, arrays and a `valid`
  mask out: making them run somewhere (images, dependencies, resources) is your job.
- `tests/`: tests for the metric and the helpers. Add your own.

## Checkpoints

Show the checkpoint to an instructor before moving on. The times are a guide, not a rule.

| # | By | Checkpoint |
|---|---|---|
| 1 | ~0:45 | You can say how many spectra and how many *distinct structures* there are, and what that means for the featurization work. A first task reads the data on the cluster. |
| 2 | ~1:45 | RDKit features for a sample of molecules, computed on the cluster, as a parquet table that meets R1. |
| 3 | ~3:00 | All four featurizers run on the cluster in the same run. You can explain why they can't share one environment. |
| 4 | ~4:15 | The work fans out. You can show, with numbers, how wall-clock time changes with the degree of parallelism, where it stops improving, and why. A rerun is instant. |
| 5 | ~5:30 | The full pipeline, evaluation included, works on a sample. The full run is launched. |
| 6 | ~6:00 | Wrap-up: your report table, and what you would do next for the Kaggle competition. |

## Using AI assistants

Claude and other assistants are allowed, **in moderation**. Use them like a knowledgeable
colleague, not an autopilot:

- **Give it Flyte 2, not Flyte 1.** Most of what assistants have seen about Flyte is Flyte 1:
  `flytekit`, `@workflow`, `ImageSpec`, `pyflyte`. None of that applies today. Point your
  assistant at the Flyte 2 docs (`https://www.union.ai/docs/v2/flyte/llms.txt`), or connect it to
  the MCP server that ships with the SDK, limited to its docs and examples search tools:
  ```bash
  uv run --with 'flyte[mcp]' flyte-mcp --transport stdio --tools search_flyte_sdk_examples,search_flyte_docs_examples,search_full_docs
  ```
  With Claude Code, for example:
  ```bash
  claude mcp add flyte-docs -- uv run --with 'flyte[mcp]' flyte-mcp --transport stdio --tools search_flyte_sdk_examples,search_flyte_docs_examples,search_full_docs
  ```
  (Without `--tools`, the server can also launch and abort runs in your project.)
- **The cluster is the ground truth.** Version pins, library names and resource numbers an
  assistant suggests are hypotheses until a run on your cluster confirms them.
- **Checkpoints need evidence you produced**: numbers from your runs, links to your runs, and an
  explanation of *why* your pipeline looks the way it does, in your own words. "The assistant
  said so" doesn't pass a checkpoint.

## If you finish early

- Make `evaluate` better than the provided MLP. Does the ranking of representations change?
- Aggregate the evidence from *all* of a molecule's spectra more cleverly than by averaging.
- Write `make_submission`: `test.parquet` → `submission.csv`, as the competition expects.
- Candidate libraries beyond the training set (PubChem, COCONUT): what would that do to the pipeline?

## Questions worth asking yourself

- What does each featurizer actually need from the 3 GB?
- What happens to a task (and its pod) that needs something the cluster's machines don't have?
- Your laptop builds the images, the cluster runs them. Are they the same kind of machine?
- What in your pipeline has to be identical across tasks, and how do you guarantee it?
- Who waits for whom? What could run at the same time?
- How many CPUs are you asking for, and how many are free while everyone else is running too?

# NoiSDE: Positive Noise Learning in Energy-Guided Diffusion Bridges for Boltzmann Transport

NoiSDE is a research implementation of utility-supervised positive-noise learning for energy-guided diffusion bridges between Boltzmann distributions. The learned object is an energy-conditioned low-rank diffusion factor, selected by hard validation utility coordinates and corrected through a probability-flow companion density for density-dependent diagnostics.

This repository contains the core PyTorch implementation, result data files, PF-companion audit logs, and figure-generation scripts for the NoiSDE manuscript. Large external sample stores, raw observables, and MBAR matrices must be restored under `data/external/` when running the full molecular workflows.

---

## Environment

The implementation is based on PyTorch, NumPy, pandas, Matplotlib, OpenMM, and PyMBAR. A conda-first setup is recommended for molecular dependencies:

```bash
conda env create -f environment.yml
conda activate noisde
pip install -e .
```

Alternatively, use a Python virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

Python `3.11` is the tested target. OpenMM and PyMBAR are required for the full molecular and MBAR entry points; the toy, PF-companion, figure, and core-contract paths use the lighter project stack.

---

## Data preparation

Publication-facing data files are included in the repository:

- `data/processed/seed_level/` contains seed-level result tables.
- `data/metadata/` contains run indexes paired with released result tables.
- `results/audits/pf_companion/` contains PF-companion audit logs.
- `results/figures/` and `results/statistics/figures/` contain figure source data, rendered assets, and summaries.

Large or environment-specific inputs are intentionally omitted from Git. Place them under `data/external/` before running the corresponding workflow. Typical full-workflow inputs include:

- `data/external/alanine_dipeptide/source_samples_alphaR.npz`
- `data/external/alanine_dipeptide/target_samples_beta.npz`
- `data/external/alanine_dipeptide/alanine_dipeptide.pdb`
- `data/external/free_energy/mbar_u_kn.npz`
- `data/external/raw_observables_chi2_demarcation.csv`

The figure source CSV files already stored in `results/figures/*/*_source.csv` are sufficient for the supported figure-regeneration commands.

`config` values in `data/metadata/run_index_*.csv` are provenance pointers for the original experiment registry. Most full-run configuration snapshots are not included in this lightweight release. Executable configs for included workflows live under `configs/`.

---

## Run figure reproduction

Regenerate every supported manuscript figure from repository data files:

```bash
python scripts/generate_figures.py --all
```

Regenerate one figure:

```bash
python scripts/generate_figures.py --figure main_result_overview
python scripts/generate_figures.py --figure chi2_demarcation_overlay
python scripts/generate_figures.py --figure component_gains_by_family
```

Figure configs live in `configs/figures/`. The current supported outputs are:

- `results/figures/main_result_overview/main_result_overview.pdf`
- `results/figures/chi2_demarcation_overlay/chi2_demarcation_overlay.pdf`
- `results/figures/component_gains_by_family/component_gains_by_family.pdf`

---

## Run optional full workflows

Train with OpenMM-backed reduced-potential and force calls after restoring the alanine sample stores and structure file:

```bash
python scripts/train_openmm_bridge.py --config configs/experiments/alanine_transport/default_alanine_gbn2.yaml
```

Run an MBAR estimate from an external reduced-potential matrix:

```bash
python scripts/run_mbar_estimate.py --config configs/experiments/free_energy/default_mbar.yaml
```

These commands require local external inputs and the optional molecular stack. If an input path is missing, the script stops instead of substituting synthetic data.

---

## Configuration and variants

`configs/config_index.csv` maps each experiment family to its config namespace, result table, run index, and current status. The main executable training template is:

```text
configs/templates/noisde_training.yaml
```

The component-attribution registry is:

```text
configs/experiments/core_ablation/variant_matrix.yaml
```

It defines the manuscript variant family:

| Variant | Role |
|---|---|
| `F0` | deterministic backbone |
| `F1` | energy-guided drift only |
| `F2` | positive noise without energy-guided drift |
| `A2` | fixed isotropic noise sweep |
| `A3_prime` | learned isotropic noise without utility |
| `A3` | learned isotropic noise with utility |
| `A4_prime` | full architecture without utility |
| `A4` | full NoiSDE |
| `A5` | over-noise stress variant |
| `B1` | non-PN learnable stochastic control |

Checkpoint selection uses the safe Pareto-knee rule implemented in `src/noisde/utility/pareto.py`.

---

## Repository layout

```text
.
|-- src/noisde/                  # Models, SDE sampler, training, evaluation, PF companion, figures
|-- configs/                     # Figure, experiment, and template configs
|-- data/
|   |-- external/                # Restored local inputs, kept out of Git
|   |-- metadata/                # Run indexes
|   `-- processed/seed_level/    # Seed-level result tables
|-- results/
|   |-- audits/                  # PF-companion audit logs
|   |-- figures/                 # Figure assets and source data
|   `-- statistics/              # Figure summaries
|-- scripts/                     # Executable figure, OpenMM, and MBAR entry points
`-- tests/                       # Core algorithm and config-contract checks
```

Run the local contract tests with:

```bash
pytest
```

---

## Contact

For questions, contact Yiming Zuo at `zuoyiming@stu.ouc.edu.cn`.

---

## License

Code is released under the MIT License. Result data files, figure source data, and derived summaries are released under CC BY 4.0 unless a specific file states otherwise.

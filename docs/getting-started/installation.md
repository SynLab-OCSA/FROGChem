# Installation Guide

FROG needs **Python 3.10 or newer**. The recommended setup is a dedicated conda environment, because
`rdkit`, `ase` and `scipy` install most reliably from conda-forge.

## 1. Recommended: conda environment + editable install

```bash
# 1. create and activate an environment
conda create -n frog python=3.11 -y
conda activate frog

# 2. compiled dependencies from conda-forge
conda install -c conda-forge numpy scipy matplotlib pandas seaborn periodictable ase rdkit mdtraj pyyaml -y

# 3. FROG itself, from a clone of the repository
git clone https://github.com/SynLab-OCSA/frog.git
cd frog
pip install -e .
```

`pip install -e .` installs the remaining dependencies listed in `pyproject.toml` (including the AIMNet2
machine-learning potential, fetched from GitHub, so step 3 needs internet access) and puts all the `frog-*`
commands on your `PATH`. "Editable" (`-e`) means that pulling a new version with `git pull` takes effect
immediately, without reinstalling.

Without a clone you can also install directly:

```bash
pip install "git+https://github.com/SynLab-OCSA/frog.git"
```

## 2. Check that it works

```bash
python -c "import frogchem as frog; print('frogchem', frog.__version__)"
frog                 # lists all tools with a one-line description
frog-rrho --help     # options, outputs and examples of one tool
```

If `frog` is "command not found", the environment where you installed it is not active
(`conda activate frog`).

## 3. Optional: web GUI

The interactive web GUI of the reaction plotter additionally needs `pip install streamlit`.

## 4. Building this documentation locally

The documentation is built with **Sphinx** (MyST Markdown, Read-the-Docs theme):

```bash
pip install -r docs/requirements.txt
sphinx-build -b html docs docs/_build/html
```

Then open `docs/_build/html/index.html` in a browser.

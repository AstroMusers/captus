# captus
`captus` is a pipeline to estimate the galactic abundance of primordial black holes captured by main-sequence stars and neutron stars.

`captus` uses the N-body integrator [Rebound](https://github.com/hannorein/rebound).

## Installation

### Clone captus

```bash
git clone https://github.com/AstroMusers/captus.git
cd captus
```
### Create environment

```bash
conda create -n captus python=3.11
conda activate captus
```
### Install

```bash
python -m pip install -e .
```
#### PBHbounds dependency

Some `captus` functionality uses a modified version of [PBHbounds](https://github.com/bradkav/PBHbounds) for primordial black hole observational constraints.

`captus` currently relies on a fork of PBHbounds containing compatibility changes required by the `captus` analysis and plotting routines:

```bash
git clone https://github.com/aesar77/PBHbounds.git
```

PBHbounds is kept as a separate repository and is not installed automatically with `captus`.

### Configure the PBHbounds path

After cloning PBHbounds, set the `PBHBOUNDS_PATH` environment variable to the location of the cloned repository:

```bash
export PBHBOUNDS_PATH=/path/to/PBHbounds
```

For example:

```bash
export PBHBOUNDS_PATH=$HOME/repo/PBHbounds
```

You can verify that the variable is set with:

```bash
echo $PBHBOUNDS_PATH
```

If you are using a Conda environment and would like the variable to be set automatically whenever the environment is activated, you can instead run:

```bash
conda activate captus

conda env config vars set PBHBOUNDS_PATH=/path/to/PBHbounds

conda deactivate
conda activate captus
```

Then verify:

```bash
echo $PBHBOUNDS_PATH
```

### Using PBHbounds from Jupyter

Jupyter must be launched from an environment that has access to `PBHBOUNDS_PATH`. After activating the `captus` environment, verify the variable and launch Jupyter:

```bash
conda activate captus
echo $PBHBOUNDS_PATH
jupyter lab
```

Inside Python or a notebook, you can verify that `captus` can see the variable with:

```python
import os

print(os.environ.get("PBHBOUNDS_PATH"))
```

### Reproducibility

For reproducible `captus` results, use the PBHbounds fork linked above rather than the unmodified upstream repository. The `captus`-compatible fork contains changes required by the current integration.

For a fully reproducible release, the specific PBHbounds commit or release tag used with a given `captus` release should be checked out before running the analysis:

```bash
cd PBHbounds
git checkout captus-v1
```

The corresponding PBHbounds revision is documented with each `captus` release.

### Attribution

PBHbounds is originally developed by Bradley J. Kavanagh and collaborators. If you use PBHbounds functionality or observational constraints in your work, please also follow the citation instructions provided by the original PBHbounds project:

https://github.com/bradkav/PBHbounds



## Testing

```bash
pip install -e ".[dev]"
pytest
```


# Acknowledgements

This research was supported by the National Aeronautics and Space Administration (NASA) under grant number 80NSSC24K0233 issued by the Astrophysics Division of the Science Mission Directorate (SMD) and the McDonnell Center for the Space Sciences at Washington University in St. Louis.

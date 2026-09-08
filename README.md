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

## Testing

```bash
pip install -e ".[dev]"
pytest
```


# Acknowledgements

This research was supported by the National Aeronautics and Space Administration (NASA) under grant number 80NSSC24K0233 issued by the Astrophysics Division of the Science Mission Directorate (SMD) and the McDonnell Center for the Space Sciences at Washington University in St. Louis.

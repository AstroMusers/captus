# captus
Captus follows two ways a primordial black hole (PBH) can become bound to a star. Gravitational waves can carry energy away during a close neutron-star encounter. In a three-body encounter around a main-sequence star, the other bodies redistribute the orbital energy. We use Captus to calculate capture rates and follow the systems that survive.

For the three-body channel, we use the N-body integrator REBOUND. The gravitational-wave module handles the separate two-body route around compact stars.

## Three-body binary-planet formation

We use `captus.three_body_capture` to integrate direct Newtonian encounters among
three initially unbound, equal-mass planets. A seeded sampler draws encounter
positions uniformly inside the interaction volume and draws velocities from an
isotropic Maxwellian. We count an outcome as a binary only when a bound pair
and an energetically unbound, outward-moving third planet pass the configured
hierarchy threshold. Each result includes energy and angular-momentum errors.

To inspect one seeded encounter, run this command.

```bash
PYTHONPATH=src captus-binary-capture --mass-mjup 1 --velocity-dispersion-kms 0.3 --chi1 10 --seed 29 --checkpoints 20 --escape-separation-factor 5
```

For the rate calculation, we implement the equal-mass hard-binary fit from
Atallah et al. (2024, arXiv:2402.12429). It gives

```text
Gamma = 0.081 G^5 m^5 n^3 sigma^-9 (a_max / a_fs)^3 sqrt(chi1)
a_fs = G m / (3 sigma^2)
```

Here `n` is the number density of each equal-mass planet population, `sigma` is their shared one-dimensional velocity dispersion, and `chi1` must be at least 10. The fit describes hard binaries with `a_max <= a_fs`. Its assumptions exclude wide binaries, unequal masses, finite-size collisions, and cluster evolution.

Install PCAT for posterior inference on the Poisson rate normalization.

```bash
python -m pip install -e '.[inference]'
```

The notebook `notebooks/binary_planet_three_body_capture.ipynb` walks through
the rate calculation and PCAT posterior with synthetic counts. The numbers are
there to make the calculation concrete. Applying it to a real population also
requires selection, disruption, and contamination models.

## Installation

### Clone Captus

```bash
git clone https://github.com/AstroMusers/captus.git
cd captus
```
### Create an environment

```bash
conda create -n captus python=3.11
conda activate captus
```
### Install Captus

```bash
python -m pip install -e .
```

Set `CAPTUS_PATH` to the checkout directory when Captus is used from another location. If it is unset, Captus uses its checkout root. The legacy `CAPTUS_REPO_ROOT` remains a fallback. Keep local inputs in the ignored `data/` directory and generated figures directly in the ignored `visuals/` directory.

TDpy supplies shared plot styling and is installed with Captus. Install the
optional plotting tools, including `corner` and Bokeh, with this command.

```bash
python -m pip install -e '.[plotting]'
```

For interactive three-body results, `Plots.plot_metric_wrt_v_bokeh()` writes a
standalone HTML version of the metric-versus-velocity plot with hover values,
pan and zoom tools, and a legend that can hide individual series. Pass
`output_path` to place the HTML figure in a chosen `visuals/` directory. It
uses the same analysis values and exclusions as `plot_metric_wrt_v()`.
After `analysis_list_a` has been assembled in
`notebooks/threebody_capture_paper.ipynb`, create an interactive capture
cross-section plot with this call.

```python
from captus.paths import get_visuals_path

plotobj = pl.Plots(name="Sun_Jupiter_Set_A_paper", analysis=analysis_list_a)
plotobj.plot_metric_wrt_v_bokeh(
	metric_list=["dsigma_au2"],
	metric_ylabel=r"$\sigma_{cap}$ [AU$^2$]",
	analysis_name=analysis_list_a[0].name,
	metric_labels=[r"$\sigma_{cap}$"],
	output_path=get_visuals_path() / "captus_sun_jupiter_capture_cross_section.html",
)
```

Plots use white backgrounds with black axes by default. The GW grid functions
accept `typeplotback='dark'` for black backgrounds with white axes. For the
three-body plots, select a theme with `Plots(..., typeplotback='dark')` or
override it on an individual plotting call. With no explicit selection,
these functions inherit an enclosing TDpy `plot_style()` theme. Plotting calls
restore the caller's settings when they return.

### PBHbounds

Some `captus` calculations use PBHbounds for observational constraints on PBHs. We use a Captus-compatible fork with patches required by the package's analysis and plotting routines.

Clone the fork separately.

```bash
git clone https://github.com/aesar77/PBHbounds.git
```

Install PBHbounds separately from `captus`.

### Configure the PBHbounds path

Set `PBHBOUNDS_PATH` to the fork's directory.

```bash
export PBHBOUNDS_PATH=/path/to/PBHbounds
```

To persist this variable in Conda, run these commands.

```bash
conda activate captus

conda env config vars set PBHBOUNDS_PATH=/path/to/PBHbounds

conda deactivate
conda activate captus
```

### Using PBHbounds from Jupyter

Launch Jupyter from the activated `captus` environment so it inherits `PBHBOUNDS_PATH`.

```bash
conda activate captus
echo $PBHBOUNDS_PATH
jupyter lab
```

Print the active path from Python or a notebook with this snippet.

```python
import os

print(os.environ.get("PBHBOUNDS_PATH"))
```

### Reproducibility

Use the Captus-compatible PBHbounds fork when reproducing these calculations. It carries the compatibility changes required by Captus.

For a release-level reproduction, check out the PBHbounds revision recorded for that Captus release.

```bash
cd PBHbounds
git checkout captus-v1
```

### Attribution

Bradley J. Kavanagh and collaborators developed PBHbounds. Follow the project's citation instructions when using its constraints.

`github.com/bradkav/PBHbounds`



## Testing

```bash
pip install -e ".[dev]"
pytest
```

## Paper Walkthroughs

Our Saricaoglu-Daylan paper walkthroughs begin with the Milky Way's density and
velocity field, move to three-body capture around Sun-Jupiter and Beta Pictoris,
then turn to gravitational-wave capture by a neutron star. Open them in Jupyter
or VS Code from an environment with Captus installed. The guide is in
[examples/README.md](examples/README.md).

One example uses MulensModel to generate matched synthetic single- and
binary-lens light curves, then uses PCAT to sample the binary-lens parameters
and compare the models' posterior probabilities. The event trajectory is fixed
in this example. It explores model comparison and makes no prediction about
Roman detection yields or the origin of a black-hole population. Install it with
`python -m pip install -e '.[inference]'`.

# Acknowledgements

This research was supported by the National Aeronautics and Space Administration (NASA) under grant number 80NSSC24K0233 issued by the Astrophysics Division of the Science Mission Directorate (SMD) and the McDonnell Center for the Space Sciences at Washington University in St. Louis.

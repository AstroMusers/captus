# Captus Paper Walkthroughs

We start with the Milky Way's density and velocity field, move to three-body
capture around two planetary hosts, then visit gravitational-wave capture by a
neutron star. Together, the notebooks trace several results from Saricaoglu and
Daylan's *Captured Primordial Black Holes in Microlensing Surveys*. Open them in
Jupyter or VS Code with a Python environment where Captus is installed.

`paper_environment.ipynb` recalculates the Milky Way density, circular speed,
and velocity dispersion in Table 1. A separate seeded Maxwellian draw shows how
the velocity model behaves in a reproducible toy sample. The paper's reported
results come from the full encounter simulations.

`paper_threebody_table.ipynb` plots selected 8 kpc entries from Table 2 for
Sun-Jupiter and Beta Pictoris. We transcribe the paper's reported central values
and 2-sigma uncertainties. Reproducing those estimates requires the separate
120,000-system encounter archives.

`paper_gw_capture.ipynb` calculates capture cross-section and coalescence-time
grids for a typical neutron star. Its 8-by-8 grid makes the trends quick to
inspect. Set `grid_size` to 100 to match the manuscript's resolution. With
`PBHBOUNDS_PATH` set to the Captus-compatible PBHbounds checkout, the notebook
also calculates the equilibrium-abundance grid.

`paper_microlensing_pcat.py` uses MulensModel to generate matched single- and
binary-lens light curves, then uses PCAT to compare their posterior model
probabilities and sample the binary-lens parameters. The light curves are
synthetic, and the event trajectory is fixed. A Roman detection yield would
also require a target population and observing strategy. Install the optional
dependencies from the Captus repository root with `python -m pip install -e '.[inference]'`. Run the short,
121-epoch example with `python examples/paper_microlensing_pcat.py --smoke`.
The default run uses a 72-day baseline at 15-minute cadence. Its JSON summary
reports the relative Monte Carlo error for each log-evidence estimate, which
describes numerical uncertainty in the importance-sampling calculation. For
each posterior chain, it records the largest R-hat and smallest effective
sample size across parameters. R-hat compares between-chain and within-chain
spread, while effective sample size discounts correlated draws. PCAT's
convergence flag uses finite values, R-hat at most 1.05, and effective sample
size of at least 200 for every parameter.

From the paper directory, `python reproduce_paper.py --smoke` writes the
Milky Way environment table, an 8-by-8 neutron-star GW-capture figure, the two
selected Table 2 plots, the PCAT injection-recovery figure and samples, and a
reproduction manifest. Without `--smoke`, the GW grid uses 100 points per axis
and the PCAT run uses the full 15-minute cadence. The Table 2 plots use values
transcribed from the paper. Reproducing the underlying 120,000-system results
requires the original Monte Carlo and REBOUND archives or a new production run.
The exact three-body notebook workflow is available through the runner's
explicit `--stages full-notebook` mode. It is excluded from the lightweight
default because it can take days and write about 3 TB.

Figures are written directly beneath Captus's ignored `visuals/` directory.
The environment notebook is a prose/table walkthrough and does not write a
figure. Posterior or intermediate data products belong in the ignored `data/`
directory. Export a notebook from the repository root with
`jupyter nbconvert --to html examples/paper_environment.ipynb`.
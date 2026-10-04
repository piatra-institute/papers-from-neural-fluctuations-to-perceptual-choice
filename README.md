# From Neural Fluctuations to Perceptual Choice

A Fluctuation–Response Test of Evidence Accumulation.

For a decision variable that accumulates evidence to a bound, the probability of a choice is a logistic function of the evidence divided by D/(MB), where D is the diffusion of the decision variable, M its mobility and B the bound, so the psychometric slope is MB/D. The ratio survives any affine change of a neural decoder's coordinate and the choice of time unit, so diffusion, mobility and bound measured from a recorded population, on trials whose choices are never used, predict the slope fitted to choices on other trials. The paper derives the exact factors by which leak, instability, drift variability and start-point bias change the slope (leak at most doubles it), shows that the equilibrium fluctuation-response test along a trained decoder is uninformative because it holds along the Fisher discriminant of any stationary linear system, and builds estimators of D and M on optional stopping and split-half cross moments. In a simulated population of 200 Poisson neurons the naive increment variance predicts a slope of 1.40 against a fitted 10.02, a variogram of undecided trials 12.29, and the stopped split-half estimators 9.89; the prediction tracks behaviour under leak, instability and drift variability and misses it by about 10% under urgency or a choice-irrelevant signal on the coding axis. An equivalence test with a ±25% margin needs about 8,700 trials; the paper ends with an analysis plan for a public macaque LIP dataset. No data were analyzed.

## Build

```bash
uv run build.py          # -> paper/PAPER.pdf  (vendored canonical recipe)
```

Requires `pandoc` and `xelatex` on PATH. From the workspace you can also run `papers build from-neural-fluctuations-to-perceptual-choice`.

## Simulation

```bash
cd simulation && uv run run_all.py   # -> output/results.json and output/figures/*.png, 19 checks
```

Part of [piatra-papers](https://github.com/piatra-institute). See the workspace docs for the research and writing pipelines.

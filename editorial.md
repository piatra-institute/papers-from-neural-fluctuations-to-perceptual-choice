# Editorial record

Status: full read and revision complete (2026-10-04).

## Contribution and reader

For neuroscientists who record decision variables in populations, for modelers of perceptual decisions, and for physicists interested in effective temperatures of living systems: a test in which a quantity measured from neural dynamics without choices predicts the stochasticity of choices, with the relation and its invariances stated, the exact effect of the standard departures from the integrator, a proof that the equilibrium fluctuation-response construction is uninformative along a trained decoder, estimators that are unbiased under absorbing bounds and observation noise, their behaviour on a simulated population with known ground truth, the sample size of the test, and an analysis plan for a public dataset. Confidence is that of derivations for stated stochastic models and of simulations with known generating processes; no data were analyzed.

## Material worth preserving

From the origin material: the question of a measured rather than fitted temperature; the evidence-in-log-likelihood framing; the diffusion/mobility/bound ratio and its scale invariance; randomized pulses; the separation of process from observation noise with disjoint parts of the population; the warning against choice-conditioned trajectories; the development/confirmation split by monkey; the hierarchy from response geometry to scalar; the removal of "temperature" and "Einstein" from the title.

## Reverse outline

- Section 1: fitted temperatures, measured temperature in physics and its nonequilibrium versions, LIP and bounded accumulation, the test, results.
- Section 2: the model; Proposition 1 (exit probability); Proposition 2 (invariance); the Einstein and Boltzmann reading.
- Section 3: Proposition 3 (leak, drift variability, start point); the Kramers limit; Table 1; collapsing bounds by simulation.
- Section 4: Proposition 4 (fluctuation-response ratio along the Fisher discriminant of a stationary linear system); why the equilibrium test is uninformative.
- Section 5: Proposition 5 (optional stopping); the estimators; the simulated population; naive estimators; sensitivity to non-decision time, population size and the Gaussian approximation.
- Section 6: six generating models (Table 2); what agreement shows; two diagnostics.
- Section 7: sample size of the equivalence test.
- Section 8: analysis plan for the public dataset and the reading of each outcome.
- Section 9: limits.

## Strongest rival

That the comparison adds nothing to fitting a drift-diffusion model to choices and reaction times, since the slope MB/D is a property of that model. The paper's answer is that the three neural quantities are measured without choices, so the comparison tests the claim that the recorded population carries the variable that generates the choices; the simulations show what the test can and cannot distinguish (it does not single out the pure integrator). A second rival is the view that LIP does not carry the decision variable at all (stepping, Latimer et al.; task-dependent geometry, Okazawa et al.); the test would then fail, and the diagnostics say how.

## Decisions and remaining checks

- Argument: the equilibrium fluctuation-dissipation construction of the origin material was replaced after Proposition 4 showed it uninformative along a trained decoder; the estimators were rebuilt on optional stopping; mobility is estimated from sustained evidence as well as pulses because pulse estimates are noisier; a passed test is stated to support bounded accumulation with an effective ratio and not the pure integrator.
- Structure: five propositions with proofs; two tables, neither split across pages; five figures, placement checked on the final PDF.
- Voice: no em-dashes; signature words reduced; no roadmap paragraph; math set in LaTeX.
- Sources: literature sentences checked against abstracts, and the dataset paper against its full text (sources.md); items cited for standard results are used only for what their titles and standard content establish.
- Executed: one recorded run (verification/model.json, PASS, 19 of 19 internal checks); 208 claims bound (152 to the run); refs PASS (59 of 59); voice OK.
- Visual: all pages inspected after the final build.

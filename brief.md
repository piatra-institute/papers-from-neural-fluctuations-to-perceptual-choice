# Brief

## Question

The origin material asks whether perception has a measurable state variable in the sense that temperature is one: a quantity defined by measurement, not fitted to the behaviour it is meant to explain. Its final form is concrete. For a bounded accumulator of evidence, the probability of a choice depends on the diffusion D of the decision variable, its mobility M (the change of drift per unit of evidence) and the bound B only through D / (M B), so the psychometric slope should equal M B / D. All three quantities can in principle be measured from a recorded neural decision variable on trials whose choices are never used, and the slope can be measured from choices on other trials. Is the comparison meaningful, how should D, M and B be estimated from spiking populations, what does agreement or disagreement mean, and how many trials does the test need?

## Claim

Results about stated models, each derived and checked numerically, and a simulated test with known ground truth.

1. For the bounded integrator the choice probability is the logistic function of h / Θ with Θ = D / (M B). The slope M B / D is unchanged by any affine change of the decoded coordinate and by the choice of time unit, and it changes with the evidence unit exactly as the fitted slope does, so a neural estimate in arbitrary decoder units can be compared with behaviour.
2. Departures from the integrator change the slope at zero evidence by exact factors: leak and instability by (e^Λ - 1) / (Λ ∫₀¹ e^{Λu²} du), with Λ = λB²/(2D), which rises to 2 in the strong-leak limit and falls to 0 with instability; trial-to-trial drift variability by 4 E[σ'(sZ)], s = ηB/D; a start-point offset z by 1 - z²/B²; a uniform start-point spread by 1 - s_z²/(3B²). Collapsing bounds are matched approximately by the mean bound at the time of decision.
3. For any stationary linear system, the fluctuation-response ratio -C'(τ)/R(τ) is constant in lag along the Fisher discriminant of the stationary response, with or without detailed balance; at detailed balance that readout is the one conjugate to the evidence, and other readouts give a lag-dependent ratio even at equilibrium. A collapse of this ratio along a trained decoder therefore tests nothing about equilibrium or about a temperature.
4. Optional stopping gives unbiased estimators of D and M from all trials despite absorbing bounds; a split-half cross moment removes independent observation noise. In a simulated population of 200 Poisson neurons, the naive one-bin increment variance predicts a slope of 1.40 against a fitted 10.02 and the variogram of undecided trials 12.29, while the stopped split-half estimators predict 9.89, with no detectable bias.
5. The neural prediction tracks the slope fitted to choices under leak, instability and drift variability (mean log ratios between -0.044 and 0.051), because the stopped estimators measure effective diffusion and mobility over the decision window; urgency and a slow signal along the coding axis that does not drive choice raise the log ratio to 0.108 and 0.097. A passed test supports bounded accumulation with the measured effective ratio and does not single out the pure integrator.
6. With 8,000 trials the run-to-run SD of the log ratio between fitted and predicted slopes is 0.080; an equivalence test with a ±25% margin reaches a power of 0.8 at about 8,700 trials and a ±10% margin needs about 48,000.

## Kind

Formal model with simulation. has_simulation: true; claims bound in claims.yaml to simulation/output/results.json through a recorded run.

## Cornerstone literature

Bounded accumulation and the diffusion model (Ratcliff; Ratcliff and McKoon; Palmer, Huk and Shadlen; Bogacz et al.; Wald), first-passage theory (Redner; Risken), the Einstein relation and fluctuation-dissipation theorem (Einstein; Callen and Welton; Kubo; Kramers), effective temperatures and nonequilibrium response (Cugliandolo, Kurchan and Peliti; Cugliandolo; Martin, Hudspeth and Jülicher; Harada and Sasa; Baiesi, Maes and Wynants; Seifert; Gnesotto et al.; Mizuno et al.; Lynn et al.), LIP and evidence accumulation (Shadlen and Newsome; Roitman and Shadlen; Huk and Shadlen; Kiani, Hanks and Shadlen; Gold and Shadlen; Yang and Shadlen; Kira, Yang and Shadlen; Churchland et al.; Latimer et al.; Shadlen and Kiani; Okazawa et al.; Stine et al.; Steinemann et al.), single-trial decoding (Kiani et al.; Peixoto et al.), urgency and collapsing bounds (Churchland, Kiani and Shadlen; Ditterich; Drugowitsch et al.; Hanks, Kiani and Shadlen; Cisek et al.), the noise in accumulation (Brunton, Botvinick and Brody; Zylberberg, Fetsch and Shadlen), neural fluctuation and response (Arieli et al.; Churchland et al. 2010; Sarracino et al.), fitted temperatures and precision (Daw et al.; Feldman and Friston; Friston; Aguilera et al.; Isomura et al.), methods (Smith and Brown; Kalman; Wichmann and Hill; Schuirmann; Lakens; Efron and Tibshirani), and the public datasets (Steinemann et al. data; International Brain Laboratory).

## Discipline

No data were analyzed: every number about neurons is a property of a simulated population with a stated generating model, and the public dataset is described only from its paper. "Temperature" and "Einstein relation" are used only for the analogy the integrator makes exact, not as claims about brains. The free-energy principle is not used; precision is mentioned once as a quantity a measured ratio could be compared with. No number in the manuscript that is not bound to a JSON pointer in a recorded execution or to a checked source.

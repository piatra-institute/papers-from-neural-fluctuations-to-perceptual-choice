# Audit

Dated log of editorial passes and verification runs. Newest first.
See the workspace docs (run `papers docs`): writing-pipeline.md §7 and refresh-pipeline.md.

## 2026-10-04 — initial publication

Scope: first publication of the paper from the origin conversation (chats/, not committed).
Changes:
  - brief, research log and frozen bibliography (59 references resolved through doi.org on 2026-10-04; one wrong identifier replaced)
  - simulation/: first-passage theory and departure factors with Monte Carlo checks, collapsing bounds, stationary
    fluctuation-response ratios along four readouts, a simulated spiking population with stopped split-half estimators,
    six generating models, power of the equivalence test, sensitivity to non-decision time, population size and the
    Gaussian approximation; 19 internal checks
  - PAPER.md with Propositions 1 to 5, five figures, two tables and an analysis plan for a public dataset
Verification:
  - run: verification/model.json PASS (19/19 checks)
  - voice: OK (0 errors, 0 warns)
  - refs: 59 cited, 59 entries, PASS
  - claims: 208 bound (152 computation), PASS
  - build: 18 pages, no missing-character warnings

<!--
## YYYY-MM-DD — <pass name>
Scope: <the bounded list of changes>
Changes:
  - ...
Verification:
  - voice: <errors / warns>
  - refs: <missing / unused>
  - claims: <reconciliation result>
  - build: <pages, missing-char warnings>
  - check => PASS/FAIL
-->

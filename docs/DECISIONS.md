# Decisions

Decisions fixed before the search runs, so results cannot steer them.

## 2026-11-23: Search walls

NSGA-II and random search fly every candidate on the same 10 training flights (train index, gust level), the same for every candidate, generation, seed and method:

| Train index | Gusts | Origin |
|---|---|---|
| 0 | none | original 5 |
| 1 | moderate | original 5 |
| 2 | moderate | original 5 |
| 3 | strong | original 5 |
| 4 | strong | original 5 |
| 5 | strong | added |
| 6 | strong | added |
| 7 | strong | added |
| 8 | moderate | added |
| 9 | none | added |

The added walls are the next five train indices in order (5 to 9). No walls were skipped and none were picked by how they fly. The gust levels are assigned by hand so that strong gusts get the most weight, since strong gusts are the hardest condition and the one where a fast setting is most likely to touch the facade. The 10 flights are 2 with no gusts, 3 moderate and 5 strong. Validation and test walls are never used in the search.

## 2026-11-23: Search setup

- Objectives, both minimized: minus the mean coverage over the 10 flights, and the mean of time / T_max over the 10 flights.
- Constraint: the number of failed flights out of 10 must be 0 (pymoo inequality constraint, feasibility first).
- NSGA-II: population 40, 50 generations (2,000 candidates, 20,000 flights), SBX crossover (prob 0.9, eta 15), polynomial mutation (eta 20), duplicates removed, random initial population (hand-tuned settings not included).
- Random search: 2,000 candidates drawn uniformly within BOUNDS, flown in batches of 40 so its log lines up with NSGA-II generations.
- Search seeds 0, 1, 2 for both methods.
- Each method's Pareto front is the non-dominated set of all candidates with no failures, taken over every evaluation in the run.

## 2026-11-23: Choosing the final setting on validation

For each method (NSGA-II and random search, applied the same way to each), every setting on the Pareto front is flown on all 50 validation walls at their own gust levels. Then:

1. Keep the settings with zero validation failures and validation mean coverage of at least 96.7% (the hand-tuned validation coverage). Of those, pick the fastest (lowest mean validation time).
2. If no setting has zero validation failures, pick the one with the fewest validation failures, breaking ties by highest validation coverage and then by lowest validation time.
3. If some settings have zero validation failures but none reaches 96.7% coverage, pick the zero-failure setting with the highest validation coverage, breaking ties by lowest validation time.

The chosen settings, together with HAND_TUNED, are flown once on the 300 test flights at the very end. The test set is not used for search, selection or tuning.

## 2026-11-24: Validation rule, time measure and ties

Fixed before any Pareto setting was flown on validation.

- "Fastest" and "validation time" in the rule above mean the mean validation time ratio (time / T_max, averaged over the 50 validation flights), the same measure as the search objective, not the mean time in seconds. This applies to rule 1 and to the time tie-breaks in rules 2 and 3. Mean time in seconds is still reported, and it is noted when it would pick a different setting.
- In rule 1, settings tied on time ratio are separated by higher validation coverage, then by lower candidate id. In rules 2 and 3, any tie left after the rule's own keys is broken by lower candidate id.
- The 96.7% threshold is fixed and is not recomputed from the validation flights of HAND_TUNED.
- Coverage is the scored coverage: 0 for a failed flight. A flight that ends in a simulator error counts as a failure.

## 2026-11-24: New validation and test sets, and the selection rule

Written before any code changes and before any Pareto setting was flown on validation.

### Why val and test were redrawn

While the controller was being debugged, the held-out test set (code 4, the 300 flights from `final_test_flights`) was flown and what it showed was used to fix the controller. The Known issues list in the README (test[45], test[34], test[2], test[49], test[98], test[86], test[76], test[39]) comes from those test walls. Validation walls (val[19] and others) were also traced during debugging. Both sets have shaped the controller, so neither can give an honest score any more.

The NSGA-II and random search runs for seed 0 used only the 10 training flights, so they are not affected and are kept. The harder set (code 5) was never flown and stays held out.

### The new sets

| Code | Name | Walls | Recipe | Gusts | Flights | Role |
|---|---|---|---|---|---|---|
| 6 | val2 | 50 | normal | one random level per wall, as in val | 50 | validation |
| 7 | test2 | 100 | normal | each wall at none, moderate and strong, as in test | 300 | final test |

- Same seed scheme as the other sets: each wall's facade, gust and noise seeds come from `np.random.SeedSequence([set_code, index, stream])`, and gust level, glass dropout, hose weight and pump position from `np.random.default_rng([set_code, index, 642])`. `check_disjoint()` covers the new codes.
- Old val (code 1) and old test (code 4) become dev sets, renamed `dev_val` and `dev_test`, free to debug on. Their walls do not change. Codes 2 and 3 stay dev sets. Harder stays code 5 and stays held out.
- val2 is used only for the selection below. test2 and harder are flown once, at the very end.

### Coverage threshold

The validation coverage threshold is HAND_TUNED's mean val2 coverage, from HAND_TUNED's flights in the same val2 run. It replaces the fixed 96.7%. "At least hand-tuned's" means validation coverage >= that value.

### Candidates

- For each method, the candidates are the union of the Pareto fronts (`pareto.csv`) of every search seed run for that method. The union is not filtered again for dominance.
- A candidate is named by seed and candidate id (`s<seed>:<cand>`). "Lower cand id" in a tie-break means lower seed, then lower candidate id.
- The picks that go to the final test come from a val2 run that pools seeds 0, 1 and 2. A val2 run on fewer seeds is provisional and is not used for the final test.
- Time ratio, scored coverage and simulator errors are measured as in the section above. Mean time in seconds is still reported.

### Selection

Within each method, the pool is the candidates with zero val2 failures. If no candidate has zero failures, the pool is the candidates with the fewest val2 failures. All picks below are made from that pool.

NSGA-II gets three picks:

1. Fast: of the pool settings with val2 coverage at least hand-tuned's, the lowest val2 mean time ratio. If none reaches hand-tuned's coverage, fast is the thorough pick.
2. Thorough: the highest val2 mean coverage, ties by lower time ratio.
3. Balanced: scale val2 coverage and val2 time ratio to 0..1 by min-max over the whole pool (a column with one value scales to 0). Among the pool settings that are neither fast nor thorough, pick the lowest Euclidean distance to (coverage 1, time ratio 0). If no such setting is left, there is no balanced pick.

Random search gets one pick, by the same rule as NSGA-II fast.

Any tie left after a rule's own keys is broken by higher coverage, then by lower cand id.

choice.json records each pick and the rule branch that chose it (zero-failure pool or fewest-failure pool, and for fast whether the coverage threshold was met or it fell back to thorough).

### Final test

Five controllers: NSGA-II fast, balanced and thorough, the random search pick, and HAND_TUNED. Each is flown once on test2 (300 flights) and on harder (50 flights). If two picks are the same setting, it is flown once and reported under both names; a missing balanced pick is reported as missing. Search seeds are reported separately only for the NSGA-II versus random search comparison (hypervolume against flights flown); everything else uses the pooled picks.

### What this replaces

This section replaces the rule in "Choosing the final setting on validation" (2026-11-23) and the fixed 96.7% threshold bullet in "Validation rule, time measure and ties" (2026-11-24). The other bullets of that section still apply.

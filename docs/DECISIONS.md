# Decisions

This is the frozen rulebook for the wall sweep drone project (MSML 642, Group 10).
`scoring.py`, `controller.py` and `nsga.py` must follow it. To change a rule, edit this
file first and say why.

## 1. Spraying and coverage

| Rule | Value |
|---|---|
| Spray footprint | 10 x 10 cm square, centered on the drone's position projected onto the wall |
| Coverage grid | 5 cm squares in the wall plane |
| Cleaned square | The footprint overlaps the square while spraying, and the true distance from the drone center to the main wall plane is between 0.25 m and 0.35 m |
| Source of truth | Scoring uses simulator truth, not sensors |

### Which squares count

| Cleanable | Excluded |
|---|---|
| Window glass | Main wall face |
| | Tops, fronts and undersides of sills, hoods and ledges |
| | Window side walls |
| | Anything below 0.15 m height |

Coverage = cleaned glass squares / all glass squares on that wall.
The drone holds 0.25 to 0.35 m from the main wall plane; the sprayer is assumed to reach recessed glass up to 0.40 m away.

## 2. Flight limits

| Rule | Value |
|---|---|
| Speed cap | 0.5 m/s |
| Time limit | T_max = 1.3 x (planned path length / 0.25 m/s) + 20 s, computed per wall |
| Failure | Any latched failure (facade_contact, ground, tumble, out_of_bounds) gives coverage 0 and time T_max |

## 3. Search

| Rule | Value |
|---|---|
| Objectives | Coverage (higher is better) and time (lower is better) |
| Constraints | Mean coverage at least 50 percent, and zero failures |
| Training walls | Each candidate flies the same 5 fixed training walls for the whole run |
| Budget | 10,000 flights per run |
| Repeats | NSGA-II: 3 runs. Random search: 3 runs. Each run uses a different optimizer seed |

### Training walls

Train walls 0 to 4, flown at gust levels none, moderate, moderate, strong, strong.
If fewer than 2 of them have a ledge, replace walls from index 5 upward until 2 do.
Record the final indices here before the first run.

| Slot | Gust level | Train index |
|---|---|---|
| 1 | none | to be recorded |
| 2 | moderate | to be recorded |
| 3 | moderate | to be recorded |
| 4 | strong | to be recorded |
| 5 | strong | to be recorded |

## 4. Success (for reporting)

A flight is a success only if coverage is at least 90 percent and there are zero failures.
We report the success rate for every controller.

## 5. Picking controllers

Selection uses the 50 validation walls. Five controllers in total:

| Controller | Source | How it is picked |
|---|---|---|
| Thorough | NSGA-II | Highest coverage with zero failures |
| Balanced | NSGA-II | Knee of the front |
| Fast | NSGA-II | Shortest time with at least 80 percent of the thorough coverage |
| Random balanced | Random search | Its balanced pick |
| Hand tuned | By hand | Tuned by hand on a flat wall without wind, as in the proposal, and fixed before the final test |

## 6. Final test

Run once, after all tuning is finished. No changes to any controller after this.

| Set | Walls | Gust levels | Flights per controller |
|---|---|---|---|
| test | 100 | none, moderate, strong | 300 |
| harder | 50 | strong (set default) | 50 |

## 7. Statistics

- Mann Whitney U test between each evolved controller and each baseline
  (random search balanced, hand tuned).
- 95 percent bootstrap confidence intervals.
- All results reported by gust level.

## 8. Changes from the proposal

| Proposal | Now |
|---|---|
| Two forward rays | One forward ray |
| 8 controller settings | 9 controller settings |
| Failure latch for wall contact only | Latch also covers ground, tumble and out of bounds |
| Original test sets | Test sets regenerated (old ones kept as dev and dev_harder) |
| 20 test flights per gust level per controller | 300 test flights per controller (100 walls x 3 gust levels) plus 50 harder |
| Hand tuned on a flat wall without wind | Unchanged |
| Wall and glass coverage | Windows only (team decision, Nov 18) |

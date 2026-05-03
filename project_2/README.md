# Project 2: Cost Allocation via the Nucleolus

## Running

To run all problems:
```
julia runner.jl
```

To run a single problem:
```
julia problem1.jl
julia problem2.jl
```

## File Structure

| File | Purpose |
|------|---------|
| `algorithms.jl` | Shared computation: `compute_cost`, `all_subsets`, `coalition_vector`, `compute_all_costs`, `nucleolus_sequential_lp` |
| `save_outputs.jl` | CSV writers for results |
| `problem1.jl` | P1: n=2, T=2, hardcoded INV, perfect anti-correlation |
| `problem2.jl` | P2: n=3 and n=10, Euclidean INV, seed=238 |
| `problem3.jl` | P3: n=2, T=168, bivariate generation (stub — not yet implemented) |
| `runner.jl` | Runs all implemented problems in sequence |
| `problem_1.jl` | Original P1 submission file (do not modify) |
| `problem_2.jl` | Original P2 submission file (do not modify) |
| `calibrate_coalition_cost.jl` | Parameter calibration script for P1 INV values |

## Outputs

Results are written to `results/` as CSVs:

```
results/
├── problem1/
│   ├── metadata.csv          n, T, C(N), timestamp
│   ├── line_costs.csv        investment cost per line
│   ├── coalition_costs.csv   C(s) and C(s)/C(N) for all coalitions
│   └── nucleolus.csv         x* and x*/C(N) per player
└── problem2/
    ├── n3/                   same files for n=3
    └── n10/                  same files for n=10
```

## Dependencies

```julia
using JuMP, HiGHS, CSV, DataFrames, Random, LinearAlgebra, Printf, Dates
```

All packages should already be installed in your Julia environment. If not:
```julia
import Pkg
Pkg.add(["JuMP", "HiGHS", "CSV", "DataFrames"])
```

## Random Seed

P2 uses `seed = 238` for node position sampling. Share this seed with other groups so coalition costs agree.

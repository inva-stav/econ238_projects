# Project 1

## Install

```julia
using Pkg
Pkg.add(["JuMP", "HiGHS", "Plots", "CSV", "DataFrames", "Random"])
```

## Run a single problem

```bash
julia problem1.jl   # or problem2.jl ... problem6.jl
```

Outputs (CSVs + PNGs) land in `results/problemN/`.

## Run all problems

```bash
julia runner.jl
```

Failures in any one problem (e.g. unimplemented P4/P5) are caught and reported; remaining problems still run.

## Layout

- `algorithms.jl` — `run_kelley`, `run_subgradient`, `save_outputs`
- `problemN.jl` — per-problem setup, calls `dispatch()`
- `runner.jl` — runs P1–P6 sequentially

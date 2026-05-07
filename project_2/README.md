# Project 2: Cost Allocation via the Nucleolus

ECON 138/238 — Optimization in Power Systems & Electricity Markets

## Running

Reproduce all figures and tables:
```
julia runner.jl
```

Run a single problem:
```
julia problem1.jl
julia problem2.jl
julia problem3.jl
julia problem4.jl          # single-month (January)
julia problem4_sweep.jl    # 12-month sweep
julia problem5.jl           # demand-side heterogeneity experiments
```

Generate plots (after Julia scripts have produced CSVs):
```
python3 plot_networks.py
python3 plot_coalition_analysis.py
python3 plot_p4.py
python3 plot_p5.py
python3 plot_p5_demand_comparison.py
```

## File Structure

### Core algorithms
| File | Purpose |
|------|---------|
| `algorithms.jl` | Coalition-cost LP (`compute_cost`), coalition enumeration (`all_subsets`, `coalition_vector`, `compute_all_costs`), sequential-LP nucleolus (`nucleolus_sequential_lp`) |
| `save_outputs.jl` | CSV writers for metadata, node positions, line costs, coalition costs, nucleolus |

### Problem scripts (one self-contained script per problem)
| File | Purpose |
|------|---------|
| `problem1.jl` | **P1**: n=2, T=2, perfect anti-correlation, calibrated INV. Verifies x*=(55,65) |
| `problem2.jl` | **P2**: n=3 and n=10, T=n, Euclidean INV, seed=238 |
| `problem3.jl` | **P3**: n=2, T=168 hours, correlation sweep ρ∈[-1,+1] using Gaussian copula with Beta(2,5) marginals |
| `problem4.jl` | **P4**: n=2, real renewables.ninja PV+wind data (Tehachapi, CA, 2019) |
| `problem4_sweep.jl` | **P4 extension**: 12-month sweep, overlay on P3 synthetic curve |
| `problem5.jl` | **P5**: Demand-side heterogeneity — synthetic load profiles with real generation |
| `problem5_synthetic_sweep.jl` | **P5 extension**: Correlation sweep with demand (P3-style, with load) |

### Plotting scripts (Python/Matplotlib)
| File | Purpose |
|------|---------|
| `plot_networks.py` | Network diagrams for P1, P2 (n=3), P2 (n=10) |
| `plot_coalition_analysis.py` | Coalition cost analysis plots for P2 (n=3, n=10) |
| `plot_p4.py` | P4 overlay on P3 curve, monthly bars, generation scatter, annual time series |
| `plot_p5.py` | P5 pairing comparison, scale sweep, monthly comparison, demand profiles |
| `plot_p5_demand_comparison.py` | P5 synthetic sweep comparison (with vs. without demand) |

### Other files
| File | Purpose |
|------|---------|
| `runner.jl` | Master script — runs all problems and plots in sequence |
| `calibrate_coalition_cost.jl` | Parameter calibration for P1 INV values |
| `p4_data/` | Renewables.ninja CSV data (PV and wind capacity factors, 2019) |

## Outputs

Results are written to `results/` as CSVs and PNGs:

```
results/
├── problem1/
│   ├── metadata.csv, line_costs.csv, coalition_costs.csv, nucleolus.csv
│   └── network_diagram.png
├── problem2/
│   ├── n3/   (same CSVs + network_diagram.png, coalition_cost_vs_size.png, ...)
│   └── n10/  (same CSVs + network_diagram.png, coalition_cost_vs_size.png, ...)
├── problem3/
│   ├── sweep.csv
│   └── savings_vs_rho.png, shares_vs_rho.png, rho_sanity_check.png
├── problem4/
│   ├── summary.csv, monthly_sweep.csv, annual_generation.csv
│   └── overlay_on_p3.png, monthly_bars.png, generation_scatter.png, annual_timeseries.png
└── problem5/
    ├── pairings.csv, scale_sweep.csv, monthly_with_demand.csv, monthly_no_demand.csv
    ├── demand_profiles.csv, synthetic_sweep_with_demand.csv
    └── pairing_comparison.png, scale_sweep.png, monthly_comparison.png, ...
```

## Dependencies

Julia:
```julia
using JuMP, HiGHS, CSV, DataFrames, Random, Distributions, LinearAlgebra, Statistics, Printf, Plots, Dates
```

Python:
```
matplotlib, numpy
```

Install Julia packages if needed:
```julia
import Pkg
Pkg.add(["JuMP", "HiGHS", "CSV", "DataFrames", "Distributions", "Plots"])
```

## Random Seeds

- **P2**: `seed = 238` for node position sampling
- **P3**: `seed = 238` for copula generation

# Crossover summary: FAI (on L) vs sinc (on L0)

Ratio = best sinc time / best FAI time (> 1: FAI faster). F* = smallest F from which FAI stays faster.


## N = 5, solve_time

| F | baseline [s] | best sinc [s] (M) | best FAI [s] (key) | FAI(L0) [s] | ratio | total_iters sinc / FAI | winner |
|---|---|---|---|---|---|---|---|
| 1 | 0.22 | 0.08 (10) | 0.15 (fai_mu_1.0e7_gdim_4) | 0.31 | 0.56 | 18 / 29 | sinc |
| 2 | 0.18 | 0.19 (5) | 0.15 (fai_mu_1.0e7_gdim_4) | 0.34 | 1.21 | 44 / 30 | FAI |
| 4 | 0.24 | 0.20 (5) | 0.16 (fai_mu_1.0e7_gdim_4) | 0.56 | 1.23 | 48 / 34 | FAI |
| 10 | 0.20 | 0.42 (5) | 0.17 (fai_mu_1.0e7_gdim_4) | 0.48 | 2.48 | 57 / 37 | FAI |
| 30 | 0.64 | 0.53 (4) | 0.17 (fai_mu_1.0e7_gdim_4) | 0.61 | 3.08 | 105 / 43 | FAI |
| 100 | 0.78 | 0.56 (4) | 0.19 (fai_mu_1.0e7_gdim_4) | 1.06 | 2.98 | 111 / 45 | FAI |

F = 1.0 winner: **sinc**. F* = **2.0**.

## N = 5, total_time

| F | baseline [s] | best sinc [s] (M) | best FAI [s] (key) | FAI(L0) [s] | ratio | total_iters sinc / FAI | winner |
|---|---|---|---|---|---|---|---|
| 1 | 0.22 | 0.10 (7) | 908.05 (fai_mu_1.0e7_gdim_4) | 6513.48 | 0.00 | 20 / 29 | sinc |
| 2 | 0.18 | 0.19 (5) | 1676.17 (fai_mu_1.0e7_gdim_4) | 6615.04 | 0.00 | 44 / 30 | sinc |
| 4 | 0.24 | 0.21 (5) | 1876.47 (fai_mu_1.0e7_gdim_4) | 8120.07 | 0.00 | 48 / 34 | sinc |
| 10 | 0.20 | 0.43 (5) | 1963.67 (fai_mu_1.0e7_gdim_4) | 6929.67 | 0.00 | 57 / 37 | sinc |
| 30 | 0.64 | 0.54 (4) | 2103.56 (fai_mu_1.0e7_gdim_4) | 6825.95 | 0.00 | 105 / 43 | sinc |
| 100 | 0.78 | 0.56 (4) | 2286.64 (fai_mu_1.0e7_gdim_4) | 8398.01 | 0.00 | 111 / 45 | sinc |

F = 1.0 winner: **sinc**. F* = **not reached**.

## N = 10, solve_time

| F | baseline [s] | best sinc [s] (M) | best FAI [s] (key) | FAI(L0) [s] | ratio | total_iters sinc / FAI | winner |
|---|---|---|---|---|---|---|---|
| 1 | 0.73 | 0.67 (7) | 0.99 (fai_mu_1.0e7_gdim_8) | 1.25 | 0.67 | 54 / 54 | sinc |
| 2 | 1.19 | 1.24 (5) | 1.08 (fai_mu_1.0e7_gdim_4) | 1.30 | 1.15 | 108 / 70 | FAI |
| 4 | 1.06 | 1.33 (5) | 1.14 (fai_mu_1.0e7_gdim_4) | 1.37 | 1.17 | 116 / 81 | FAI |
| 10 | 1.12 | 1.53 (5) | 1.10 (fai_mu_1.0e7_gdim_4) | 1.75 | 1.39 | 140 / 90 | FAI |
| 30 | 14.62 | 1.81 (5) | 1.09 (fai_mu_1.0e7_gdim_4) | 12.46 | 1.66 | 157 / 102 | FAI |
| 100 | 10.56 | 13.66 (5) | 10.12 (fai_mu_1.0e7_gdim_4) | 14.30 | 1.35 | 213 / 131 | FAI |

F = 1.0 winner: **sinc**. F* = **2.0**.

## N = 10, total_time

| F | baseline [s] | best sinc [s] (M) | best FAI [s] (key) | FAI(L0) [s] | ratio | total_iters sinc / FAI | winner |
|---|---|---|---|---|---|---|---|
| 1 | 0.73 | 0.69 (7) | 206.31 (fai_mu_1.0e7_gdim_6) | 514.91 | 0.00 | 54 / 57 | sinc |
| 2 | 1.19 | 1.25 (5) | 1246.85 (fai_mu_1.0e7_gdim_10) | 529.77 | 0.00 | 108 / 58 | sinc |
| 4 | 1.06 | 1.35 (5) | 2777.08 (fai_mu_1.0e7_gdim_4) | 510.36 | 0.00 | 116 / 81 | sinc |
| 10 | 1.12 | 1.54 (5) | 2922.95 (fai_mu_1.0e7_gdim_4) | 531.06 | 0.00 | 140 / 90 | sinc |
| 30 | 14.62 | 1.83 (5) | 3317.72 (fai_mu_1.0e7_gdim_4) | 530.35 | 0.00 | 157 / 102 | sinc |
| 100 | 10.56 | 13.68 (5) | 3788.72 (fai_mu_1.0e7_gdim_4) | 526.97 | 0.00 | 213 / 131 | sinc |

F = 1.0 winner: **sinc**. F* = **not reached**.

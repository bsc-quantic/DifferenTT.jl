# [Tutorials](@id tutorials_overview)

The tutorials are runnable Literate scripts, rendered into this site and executed during its build. Each one is also a
standalone script, which runs with `julia --project=examples examples/<file>.jl` from the root of the repository. They
use small grids so that every script takes under two minutes, and every random choice is seeded. We order them from the
representation of a function to the incompressible flow solvers, and the first seven are meant to be read in sequence.

| Tutorial | What it teaches |
|---|---|
| [Functions in the QTT format](@ref) | Grids and architectures (blocked, interleaved, reversed); encoding a function by cross interpolation; `sample`, `samples` and `integrate`; the behavior of bond dimensions. |
| [The Poisson equation in one dimension](@ref) | Boundary conditions and the finite-difference operator MPO; solving with ResidualALS; the error against the exact solution as the grid is refined. |
| [The Poisson equation in N dimensions](@ref) | Poisson in the TT format with ``N = 5`` and ``N = 20`` variables, one site of dimension ``d = 10`` per variable; ranks; ALS versus AMEn. |
| [Factorized approximate inverse preconditioners](@ref) | The paper in miniature: why truncated direct inverses lose definiteness, how `fai_preconditioner` avoids it, the choice of ``\mu``, and its effect inside ResidualALS. |
| [Solver statistics](@ref) | Local iterations per site with `KrylovSweepStats`, local conditioning with `KrylovCondStats` and `KrylovSpectrumStats`. |
| [Channel flow past a cylinder](@ref) | Sec. 5.2 of the paper in miniature: a QTT channel with a cylinder mask, and the pressure Poisson solve with and without a reused FAI. |
| [Lid-driven cavity](@ref) | The vorticity–streamfunction formulation: its faces and a short cavity run, compared with the data of Ghia, Ghia & Shin (1982). |

The section **Boundary conditions** contains two longer tutorials that derive, step by step, every boundary closure of the
library, with the dense matrices of a small grid. The docstrings of the boundary-condition functions link to them, and so
do the manual pages on [Grids, fields and boundary conditions](@ref), [Finite-difference operators](@ref) and
[Navier–Stokes](@ref).

| Tutorial | What it teaches |
|---|---|
| [Velocity–pressure formulation](@ref bc_velocity_pressure) | The closures of the operators of a field, the Laplacians and the pressure system of the Chorin projection: ghost nodes, lifts, the Neumann weight and the mean removal. |
| [Vorticity–streamfunction formulation](@ref bc_vorticity_streamfunction) | The faces of the formulation, the grid they imply, the stream-function system, the wall vorticity and the closures of the transport of the vorticity. |

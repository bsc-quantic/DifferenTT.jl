# [Tutorials](@id tutorials_overview)

The tutorials are runnable Literate scripts, rendered into this site and executed during its build. Each one is also a
standalone script, which runs with `julia --project=examples examples/<file>.jl` from the root of the repository. They
use small grids so that every script takes under two minutes, and every random choice is seeded.

We plan the following sequence, from the representation of a function to the incompressible flow solvers:

| Tutorial | What it teaches |
|---|---|
| QTT basics | Grids and architectures (blocked, interleaved, reversed); encoding a function by cross interpolation; `sample`, `samples` and `integrate`; the behavior of bond dimensions. |
| Poisson in 1D | Boundary conditions and the finite-difference operator MPO; solving with ResidualALS; the error against the exact solution as the grid is refined. |
| Poisson in N dimensions | Poisson in the TT format with N = 5 and d = 10; ranks; ALS versus AMEn. |
| FAI preconditioner | The paper in miniature: why truncated direct inverses lose definiteness, how `fai_preconditioner` avoids it, and its effect inside ResidualALS. |
| Solver statistics | Local iterations per site with `KrylovSweepStats`, local conditioning with `KrylovCondStats` and `KrylovSpectrumStats`. |
| Chorin projection around a cylinder | Sec. 5.2 of the paper in miniature: a QTT channel with a cylinder mask, and the pressure Poisson solve with and without a reused FAI. |
| Lid-driven cavity | The vorticity–streamfunction faces and a short cavity run. |

The section **Boundary conditions** contains two longer tutorials that derive, step by step, every boundary closure
of the library: the velocity–pressure formulation of the Chorin projection and the vorticity–streamfunction formulation.
The docstrings of the boundary-condition functions link to them.

!!! note "Status"
    The scripts are written and reviewed separately from the library, and this build renders those that are present in
    the `examples` folder of the repository. The list above is the plan, and a tutorial that is missing from the
    sidebar has not been merged yet.

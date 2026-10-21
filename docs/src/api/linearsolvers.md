# [Linear solvers](@id api_linearsolvers)

Local (DMRG-type) and global (Krylov) linear solvers in the TT format, preconditioners, and solver statistics.

Exported names and names declared `public` are listed here; the other names of the area are on the
[Internals](@ref) page.

```@autodocs
Modules = [DifferenTT]
Pages = ["LinearSolvers/linearsolvers.jl",
         "LinearSolvers/preconditionedproblem.jl",
         "LinearSolvers/exactupdaters.jl",
         "LinearSolvers/als.jl",
         "LinearSolvers/expandedals.jl",
         "LinearSolvers/residualals.jl",
         "LinearSolvers/amen.jl",
         "LinearSolvers/preconditioners.jl",
         "LinearSolvers/KrylovSolvers/cg.jl",
         "LinearSolvers/KrylovSolvers/gmres.jl",
         "LinearSolvers/KrylovSolvers/krylovstats.jl",
         "LinearSolvers/KrylovSolvers/krylovupdaters.jl",
         "LinearSolvers/KrylovSolvers/preconditionedlinsolve.jl",
         "LinearSolvers/KrylovSolvers/ttkrylov.jl",
         "LinearSolvers/KrylovSolvers/ttcg.jl",
         "LinearSolvers/KrylovSolvers/ttgmres.jl"]
Private = false
```

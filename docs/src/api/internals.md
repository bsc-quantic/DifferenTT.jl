# Internals

The names below are neither exported nor declared `public`. They are not part of the API covered by semantic
versioning and may change in any release; we list them because the docstrings of the public API link to them, and
because they document how the library is built. Each section collects the internal names of one area of the
API reference, grouped by source file.

## Internal names: tensor trains

```@autodocs
Modules = [DifferenTT]
Pages = ["TensorTrains/architecture.jl",
         "TensorTrains/constructors.jl",
         "TensorTrains/elementwise.jl",
         "TensorTrains/operations.jl",
         "TensorTrains/qttconstructors.jl",
         "TensorTrains/qttoperations.jl",
         "TensorTrains/tci.jl",
         "TensorTrains/utils.jl"]
Public = false
```

## Internal names: PDE problem

```@autodocs
Modules = [DifferenTT]
Pages = ["PdeProblem/bcoperators.jl",
         "PdeProblem/boundaryconditions.jl",
         "PdeProblem/diffoperators.jl",
         "PdeProblem/grid.jl",
         "PdeProblem/linearsyscorrections.jl",
         "PdeProblem/scalarfields.jl",
         "PdeProblem/vectorfields.jl"]
Public = false
```

## Internal names: physics

```@autodocs
Modules = [DifferenTT]
Pages = ["Physics/NavierStokes/chorin.jl",
         "Physics/NavierStokes/insexperiment.jl",
         "Physics/NavierStokes/vorticity.jl",
         "Physics/equationterms.jl",
         "Physics/fdstencils.jl"]
Public = false
```

## Internal names: time evolution

```@autodocs
Modules = [DifferenTT]
Pages = ["TimeEvolution/explicitevol.jl",
         "TimeEvolution/implicitevol.jl",
         "TimeEvolution/timevol.jl"]
Public = false
```

## Internal names: linear solvers

```@autodocs
Modules = [DifferenTT]
Pages = ["LinearSolvers/KrylovSolvers/cg.jl",
         "LinearSolvers/KrylovSolvers/gmres.jl",
         "LinearSolvers/KrylovSolvers/krylovstats.jl",
         "LinearSolvers/KrylovSolvers/krylovupdaters.jl",
         "LinearSolvers/KrylovSolvers/preconditionedlinsolve.jl",
         "LinearSolvers/KrylovSolvers/ttcg.jl",
         "LinearSolvers/KrylovSolvers/ttgmres.jl",
         "LinearSolvers/KrylovSolvers/ttkrylov.jl",
         "LinearSolvers/als.jl",
         "LinearSolvers/amen.jl",
         "LinearSolvers/exactupdaters.jl",
         "LinearSolvers/expandedals.jl",
         "LinearSolvers/linearsolvers.jl",
         "LinearSolvers/preconditionedproblem.jl",
         "LinearSolvers/preconditioners.jl",
         "LinearSolvers/residualals.jl"]
Public = false
```

## Internal names: utilities

```@autodocs
Modules = [DifferenTT]
Pages = ["Utils/logging.jl",
         "Utils/visualization.jl"]
Public = false
```


module DifferenTT

using Adapt
using ITensors
using Statistics
using Printf
using Random
using ITensorMPS
using KrylovKit
using TensorCrossInterpolation
using LinearAlgebra

# Include Architecture
include("TensorTrains/architecture.jl")
include("Physics/fdstencils.jl")

# Include Base PDE Problem related files
include("PdeProblem/boundaryconditions.jl")
include("PdeProblem/grid.jl")
include("PdeProblem/scalarfields.jl")
include("PdeProblem/vectorfields.jl")
include("PdeProblem/bcoperators.jl")
include("PdeProblem/diffoperators.jl")

include("Physics/equationterms.jl")

include("TensorTrains/utils.jl")
include("TensorTrains/constructors.jl")
include("TensorTrains/operations.jl")
include("TensorTrains/qttconstructors.jl")
include("TensorTrains/qttoperations.jl")
include("TensorTrains/elementwise.jl")
include("TensorTrains/tci.jl")

# Include Time Evolution related files
include("TimeEvolution/timevol.jl")
include("TimeEvolution/explicitevol.jl")
include("TimeEvolution/implicitevol.jl")

# Include Linear Solvers related files
include("LinearSolvers/preconditionedproblem.jl")
include("LinearSolvers/linearsolvers.jl")
include("LinearSolvers/exactupdaters.jl")
include("LinearSolvers/als.jl")
include("LinearSolvers/expandedals.jl")
include("LinearSolvers/residualals.jl")
include("LinearSolvers/amen.jl")
include("LinearSolvers/preconditioners.jl")

include("LinearSolvers/KrylovSolvers/cg.jl")
include("LinearSolvers/KrylovSolvers/gmres.jl")
include("LinearSolvers/KrylovSolvers/krylovupdaters.jl")
include("LinearSolvers/KrylovSolvers/preconditionedlinsolve.jl")
include("LinearSolvers/KrylovSolvers/ttkrylov.jl")
include("LinearSolvers/KrylovSolvers/ttcg.jl")
include("LinearSolvers/KrylovSolvers/ttgmres.jl")

# Include Navier Stokes
include("Physics/NavierStokes/insexperiment.jl")
include("Physics/NavierStokes/corrections.jl")
include("Physics/NavierStokes/chorin.jl")

# Export types and functions from TimeEvolution
export TimeEvol, AbstractTimeStepper, AbstractExplicitStepper, AbstractImplicitStepper
export AbstractRHS, AbstractImplicitSystem
export ExplicitEuler, RK4, timestep
export ImplicitEuler, CrankNicolson

# Export types and functions
export Architecture, QTTArchitecture, TTArchitecture
export nsites_per_variable, total_sites, site_dims, variable_dims
export backend_type
export BoundaryCond, PeriodicBC, OpenBC, DirichletBC, NeumannBC, BoundaryCond1d, BoundaryConditions
export ScalarField, VectorField, Grid
export contract_mps, contract_mpo
export tt_mul, elementwise_function

# Export types and functions from Physics
export advection, diffusion, divergence, gradient, symmetrized_laplacian
export INSExperimentBase, INSExperiment, check_cfl, check_kolmogorov_scale
export divergence_dirichlet_correction, pressure_dirichlet_correction
export chorin_evolve!, NavierStokesPredictorRHS

# Export types and functions from Linear Solvers
export AbstractLinearSolver, AbstractDMRGSolver, AbstractGlobalSolver, linearsolve
export ALS, ResidualALS, ExpandedALS, AMEn, GlobalCG, GlobalGMRES, LocalUpdaterArgs
export TTCG, TTGMRES
export Preconditioner, FactorizedPreconditioner, rank1_preconditioner, newtoniter_preconditioner, linsolve_preconditioner
export fai_dg, fai_preconditioner, fsai_g, fsai_build_gl

end


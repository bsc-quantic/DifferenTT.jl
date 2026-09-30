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
using JLD2
using TimerOutputs

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
include("PdeProblem/linearsyscorrections.jl")

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
include("LinearSolvers/KrylovSolvers/krylovstats.jl")
include("LinearSolvers/KrylovSolvers/krylovupdaters.jl")
include("LinearSolvers/KrylovSolvers/preconditionedlinsolve.jl")
include("LinearSolvers/KrylovSolvers/ttkrylov.jl")
include("LinearSolvers/KrylovSolvers/ttcg.jl")
include("LinearSolvers/KrylovSolvers/ttgmres.jl")

# Include Navier Stokes
include("Physics/NavierStokes/insexperiment.jl")
include("Physics/NavierStokes/chorin.jl")

# Include Utils
include("Utils/logging.jl")
include("Utils/visualization.jl")


# Export types and functions from TimeEvolution
export TimeEvolArgs, AbstractTimeStepper, AbstractExplicitStepper
export AbstractRHS
export ExplicitEuler, RK4, timestep
# export AbstractImplicitStepper, AbstractImplicitSystem, ImplicitEuler, CrankNicolson # -> This is still a WIP

# Export types and functions from TensorTrains
export QTTArchitecture, TTArchitecture
export nsites_per_variable, total_sites, sited_dim, site_dims, variable_dims
export backend_type
export ndim_operator_mpo, diagmpo, diagmps
export identity_mpo, zeros_mpo, zeros_mps, ones_mpo, ones_mps, vec2tensor, vec2mps, mat2tensor, mat2mpo
export contract_mps, contract_mpo
export tt_mul, elementwise_function, sample, samples, integrate

# Export types and functions from PdeProblem
export BoundaryCond, PeriodicBC, OpenBC, DirichletBC, NeumannBC, BoundaryCond1d, BoundaryConditions
export ScalarField, VectorField, Grid
export DifferentialOpsScalar, DifferentialOps
export BoundaryNullScalar, BoundaryNull
export divergence_dirichlet_correction, scalar_dirichlet_correction, symmetrize_rhs

# Export types and functions from Physics
export advection, diffusion, divergence, gradient, symmetrized_laplacian, equicorrelated_laplacian
export INSExperiment, check_cfl, check_kolmogorov_scale
export ChorinCache, chorin_step, chorin_evolve!, NavierStokesPredictorRHS

# Export types and functions from Linear Solvers
export AbstractLinearSolver, AbstractDMRGSolver, AbstractGlobalSolver, linearsolve
export ALS, ResidualALS, ExpandedALS, AMEn, GlobalCG, GlobalGMRES, ResidualObserver
export LocalUpdaterArgs, SweepingArgs, GlobalUpdaterArgs
export Preconditioner, FactorizedPreconditioner
export rank1_preconditioner, newtoniter_preconditioner, linsolve_preconditioner, fai_preconditioner
export sincquadrature_preconditioner, inverse_laplacian_preconditioner
export KrylovSweepStats, KrylovCondStats, positions_mean, positions_mean_cond

# Include and Export Logging

export Logger, SimpleLogger, init_logger, log_data, timeit_to, enable_debug_timings!

# Include and Export Visualization
export visualize_scalar, visualize_vector, visualize_vorticity

end


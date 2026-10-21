module DifferenTT

# Public APIs of the dependencies
using Adapt
using ITensors
using ITensorMPS
using JLD2
using KrylovKit
using LinearAlgebra
using Logging: NullLogger, with_logger
using PrecompileTools: @compile_workload, @setup_workload
using Printf
using Random
using TensorCrossInterpolation
using TimerOutputs
using ITensors.NDTensors: matrix
# TensorCrossInterpolation also exports a `contract`; ours is the ITensors one
using ITensors: contract
using LinearAlgebra: svd as la_svd

# Non-exported internals of ITensors (and NDTensors), ITensorMPS and KrylovKit. They are not part of the public APIs of
# these packages and may change in any release, which is why their [compat] entries in Project.toml are pinned to
# minor versions. The sweep machinery of the local solvers (ALS, ResidualALS, AMEn, ExpandedALS) is built on the
# ITensorMPS ones, and the preconditioned Krylov solvers on the KrylovKit ones.
using ITensors: Arrow, OneITensor, TagSets, primestring
using ITensors.NDTensors.BackendSelection: @Algorithm_str, Algorithm
using ITensorMPS:
    AbstractProjMPO,
    ReducedConstantTerm,
    ReducedLinearProblem,
    TDVPOrder,
    alternating_update,
    constant_term,
    makeL!,
    makeR!,
    operator,
    orderings,
    set_nsite!,
    site_range,
    sub_time_steps
using KrylovKit:
    ConvergenceInfo,
    EACHITERATION_LEVEL,
    Givens,
    SILENT_LEVEL,
    STARTSTOP_LEVEL,
    WARN_LEVEL,
    add!!,
    apply_scalartype,
    linselector,
    normres2string,
    scale,
    scale!!,
    zerovector

# Functions to which DifferenTT adds methods for its own types without qualifying the name
import ITensorMPS: apply

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
include("Physics/NavierStokes/vorticity.jl")

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
export nsites_per_variable, total_sites, site_dim, site_dims, variable_dims
export backend_type
export ndim_operator_mpo, diagmpo, diagmps
export identity_mpo, zeros_mpo, zeros_mps, ones_mpo, ones_mps, vec2tensor, vec2mps, mat2tensor, mat2mpo
export contract_mps, contract_mpo
export tt_mul, elementwise_function, sample, samples, integrate
export tt_mul_fallbacks, reset_tt_mul_fallbacks!

# Export types and functions from PdeProblem
export BoundaryCond, PeriodicBC, OpenBC, DirichletBC, NeumannBC, BoundaryCond1d, BoundaryConditions
export ScalarField, VectorField, Grid, interior_grid
export DifferentialOpsScalar, DifferentialOps
export BoundaryNullScalar, BoundaryNull
export divergence_dirichlet_correction,
    scalar_dirichlet_correction, symmetrize_rhs, neumann_weight_mpo, face_dirichlet_lift

# Export types and functions from Physics
export advection, diffusion, divergence, gradient, symmetrized_laplacian, kronecker_laplacian, equicorrelated_laplacian
export INSExperiment, check_cfl, check_kolmogorov_scale
export ChorinCache, chorin_step, chorin_evolve!, NavierStokesPredictorRHS
export INSVorticityExperiment, VorticityCache, VorticityTransportRHS, vorticity_step, vorticity_evolve!
export poisson_streamfunction, streamfunction_velocity
export VorticityFace, NoSlipWall, SlipWall, InflowFace, OutflowFace, PeriodicFace
export vorticity_boundary_conditions, vorticity_grid, cavity_faces

# Export types and functions from Linear Solvers
export AbstractLinearSolver, AbstractDMRGSolver, AbstractGlobalSolver, linearsolve
export ALS, ResidualALS, ExpandedALS, AMEn, GlobalCG, GlobalGMRES, ResidualObserver
export LocalUpdaterArgs, SweepingArgs, GlobalUpdaterArgs
export Preconditioner, FactorizedPreconditioner
export rank1_preconditioner, newtoniter_preconditioner, linsolve_preconditioner, fai_preconditioner
export sincquadrature_preconditioner, inverse_laplacian_preconditioner
export KrylovSweepStats, KrylovCondStats, positions_mean, positions_mean_cond
export KrylovSpectrumStats, spectrum_table, NotSPDError

# Include and Export Logging

export Logger, SimpleLogger, init_logger, log_data, timeit_to, enable_debug_timings!, disable_debug_timings!

# Include and Export Visualization
export visualize_scalar, visualize_vector, visualize_vorticity

# Documented API that is not exported: reachable as `DifferenTT.name`, and covered by semantic versioning
# Tensor-train utilities
public upper_triang_mpo, lower_triang_mpo, triang_projector_mpo, ndim_operator_sum_mpo, toeplitz_mpo
public row_pick_mpo, column_pick_mpo, entry_mpo, arrays2mps, arrays2mpo, point_sequence, sequence_to_point, iszero_tt
public site_indices, link_indices, replace_mpo_siteinds!, replace_mpo_siteinds, transpose_mpo, reverse_mpo
# Operators and physics
public onedim_differential_operator, derivative_coefficients, wall_vorticity_coefficients
public poisson_rhs, vorticity_transport, wall_vorticity_source, face_profile_mps, remove_mean
public regularization_check, mask_correction, cfl_spectral_radii
# Solvers and statistics
public TTCG, TTGMRES, KrylovUpdater, QRUpdater, positions_sum, krylov_numiters, krylov_conds
public dense_projected_operator, preconditioned_linsolve
public AbstractUpdater, bind_observer, default_differenTT_checkdone
public AbstractALSExpansion, NullExpansion, RandomSPSDexpansion, RandomLTexpansion, ResidualExpansion

function __init__()
    _register_plots_error_hint()
    return nothing
end

include("precompile.jl")

end

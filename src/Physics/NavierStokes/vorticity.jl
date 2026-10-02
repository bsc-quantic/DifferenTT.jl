"""
    struct INSVorticityExperiment{N} <: INSExperimentBase

Defines a 2D wall-bounded Incompressible Navier-Stokes experiment in the vorticity-streamfunction formulation
(Kiffner & Jaksch, arXiv:2303.03010): the lid-driven cavity. It describes the physical problem only: the time
integrator and the Poisson solver are passed to the solving scheme (e.g. `vorticity_evolve!`).

The walls are ghost nodes one grid spacing outside the stored nodes (build `grid` with [`interior_grid`](@ref)),
so that the stream function Poisson operator is the plain Dirichlet Laplacian. The top wall (axis 2, right side)
moves in `+x` with velocity `U_lid`; the other three walls are at rest.

# Fields
- `grid::Grid{N}`: Grid of the interior nodes.
- `omega_field::ScalarField{N}`: Vorticity `ω = ∂ₓv - ∂ᵧu`. `DirichletBC` on every face: the wall values come from `wall_formula`.
- `psi_field::ScalarField{N}`: Stream function `ψ`, with `u = ∂ᵧψ`, `v = -∂ₓψ` and `ψ = 0` on every wall (`DirichletBC`).
- `Re::Number`: Reynolds number.
- `U_lid::Float64`: Velocity of the lid.
- `wall_formula::Symbol`: Wall vorticity formula, `:jensen` (second order, as in the paper) or `:thom` (first order).
- `stencil_order::Int`: Order of the finite difference stencils (only 2 is supported).
- `maxdim::Int`: Maximum bond dimension allowed throughout the simulation.
- `cutoff::Float64`: SVD cutoff performed throughout the simulation.
"""
struct INSVorticityExperiment{N} <: INSExperimentBase
    grid::Grid{N}
    omega_field::ScalarField{N}
    psi_field::ScalarField{N}
    Re::Number
    U_lid::Float64
    wall_formula::Symbol
    stencil_order::Int
    maxdim::Int
    cutoff::Float64

    """
        INSVorticityExperiment(grid::Grid{N}, omega_field::ScalarField{N}, psi_field::ScalarField{N}, Re::Number, U_lid::Real; wall_formula::Symbol=:jensen, stencil_order::Int=2, maxdim::Int=100, cutoff::Float64=1e-8) where N

    Construct an INSVorticityExperiment. Checks that the problem is 2D, that the stencils are second order, that
    both fields share their site indices and that every face of both fields is a `DirichletBC` wall.

    # Arguments
    - `grid::Grid{N}`: Grid of the interior nodes (see [`interior_grid`](@ref)).
    - `omega_field::ScalarField{N}`: Initial vorticity (usually zero, fluid at rest).
    - `psi_field::ScalarField{N}`: Stream function field.
    - `Re::Number`: Reynolds number.
    - `U_lid::Real`: Velocity of the lid.
    - `wall_formula::Symbol`: `:jensen` or `:thom`, see [`wall_vorticity_coefficients`](@ref).
    - `stencil_order::Int`: Finite difference stencil order. Must be 2: the wall vorticity formulas and the ghost
      walls only reach one node beyond the grid.
    - `maxdim::Int`: Maximum bond dimension for TT arithmetic in the evolution. Not the solver bond dimension.
    - `cutoff::Float64`: SVD cutoff for that same TT arithmetic. Not the solver truncation or stopping tolerance.
    """
    function INSVorticityExperiment(
        grid::Grid{N},
        omega_field::ScalarField{N},
        psi_field::ScalarField{N},
        Re::Number,
        U_lid::Real;
        wall_formula::Symbol=:jensen,
        stencil_order::Int=2,
        maxdim::Int=100,
        cutoff::Float64=1e-8
    ) where N

        N == 2 || throw(ArgumentError("INSVorticityExperiment is only implemented in 2D, got N = $N"))
        stencil_order == 2 || throw(ArgumentError("INSVorticityExperiment only supports stencil_order = 2, got $stencil_order"))
        wall_vorticity_coefficients(wall_formula) # Throws for an unknown formula

        if omega_field.usites != psi_field.usites
            throw(ArgumentError("Physical sites for vorticity and stream function fields do not match"))
        end

        for (name, field) in (("omega_field", omega_field), ("psi_field", psi_field)), i in 1:N
            bc = field.bcu[i]
            if !(bc.left isa DirichletBC && bc.right isa DirichletBC)
                throw(ArgumentError("$name must have DirichletBC (wall) boundary conditions on every face, got $(typeof(bc)) on axis $i"))
            end
        end

        check_kolmogorov_scale(grid, Re)

        return new{N}(
            grid,
            omega_field,
            psi_field,
            Re,
            Float64(U_lid),
            wall_formula,
            stencil_order,
            maxdim,
            cutoff
        )
    end
end

"""
    wall_vorticity_operator(field::ScalarField{N}, grid::Grid{N}; formula::Symbol=:jensen, cutoff::Float64=1e-15, maxdim::Int=typemax(Int64)) where N

Build the MPO `S` of the ghost contribution of the wall vorticity of stationary walls to the Laplacian of the
vorticity: on the node next to each `DirichletBC` wall, `(S ψ) = ω_wall / h² = (c₁ ψ₁ + c₂ ψ₂) / h⁴`. Built as a
sum of 1D [`onedim_wall_vorticity_correction`](@ref) operators lifted to N dimensions, so a node next to two
walls receives both contributions. Returns `nothing` if `field` has no `DirichletBC` face.
"""
function wall_vorticity_operator(field::ScalarField{N}, grid::Grid{N}; formula::Symbol=:jensen, cutoff::Float64=1e-15, maxdim::Int=typemax(Int64)) where N
    array_type = backend_type(field)
    ops = ntuple(i -> begin
            onedim_op = onedim_wall_vorticity_correction(field.arch, i, field.bcu[i], array_type; formula=formula, cutoff=cutoff, maxdim=maxdim)
            isnothing(onedim_op) && return nothing
            onedim_op = onedim_op isa MPO ? onedim_op * (1 / grid.dx[i]^4) : onedim_op .* (1 / grid.dx[i]^4)
            return ndim_operator_mpo(onedim_op, i, field.usites, field.arch)
        end, N)
    return _nothing_sum_handler(ops...; cutoff=cutoff, maxdim=maxdim)
end

"""
    lid_vorticity_source(field::ScalarField{N}, grid::Grid{N}, U_lid::Real; formula::Symbol=:jensen, cutoff::Float64=1e-15, maxdim::Int=typemax(Int64)) where N

Build the MPS `ℓ` of the ghost contribution of the lid's moving-wall term to the Laplacian of the vorticity:
`c_U U_lid / h_y³` on the top row of nodes (axis 2, last index) and zero elsewhere.
"""
function lid_vorticity_source(field::ScalarField{N}, grid::Grid{N}, U_lid::Real; formula::Symbol=:jensen, cutoff::Float64=1e-15, maxdim::Int=typemax(Int64)) where N
    N == 2 || throw(ArgumentError("lid_vorticity_source is only implemented in 2D, got N = $N"))
    array_type = backend_type(field)
    _, _, c_U = wall_vorticity_coefficients(formula)

    lid_axis = 2
    n = variable_dims(field.arch)[lid_axis]
    onedim_selector = onedim_entry_operator(field.arch, lid_axis, [(n, n, 1.0)], array_type; cutoff=cutoff, maxdim=maxdim)
    selector = ndim_operator_mpo(onedim_selector, lid_axis, field.usites, field.arch)
    lid_row = apply(selector, ones_mps(field.usites, 1, array_type); cutoff=cutoff, maxdim=maxdim)

    return (c_U * U_lid / grid.dx[lid_axis]^3) * lid_row
end

"""
    VorticityCache

Operators of the vorticity-streamfunction scheme that only depend on the experiment and are built once per run.

# Fields
- `diff_ops`: First and second derivative operators of the stream function (`DifferentialOpsScalar`). The vorticity
  has the same boundary conditions, so they also apply to it.
- `laplacian::MPO`: Dirichlet Laplacian `Δ` (ghost walls with value 0).
- `poisson_op::MPO`: Operator `A = -Δ` of the stream function Poisson system `A ψ = ω`. It is symmetric positive definite.
- `wall_op::MPO`: Ghost contribution of the stationary walls to `Δω`, see [`wall_vorticity_operator`](@ref).
- `lid_source::MPS`: Ghost contribution of the lid velocity to `Δω`, see [`lid_vorticity_source`](@ref).
"""
struct VorticityCache{D}
    diff_ops::D
    laplacian::MPO
    poisson_op::MPO
    wall_op::MPO
    lid_source::MPS
end

"""
    VorticityCache(experiment::INSVorticityExperiment{N}; cutoff::Float64=1e-15, maxdim::Int=typemax(Int64)) where N

Build the operators of the vorticity-streamfunction scheme for `experiment`. The operators are built with their
own tight `cutoff`, not with `experiment.cutoff`, so that the wall entries are not truncated away.
"""
function VorticityCache(experiment::INSVorticityExperiment{N}; cutoff::Float64=1e-15, maxdim::Int=typemax(Int64)) where N
    psi_field = experiment.psi_field
    grid = experiment.grid
    stencil_order = experiment.stencil_order

    @timeit_debug timeit_to "psi_diff_ops" begin
        diff_ops = DifferentialOpsScalar(grid, psi_field; stencil_order=stencil_order, cutoff=cutoff, maxdim=maxdim)
    end
    @timeit_debug timeit_to "laplacian" begin
        Δ = symmetrized_laplacian(psi_field, grid; stencil_order=stencil_order, cutoff=cutoff, maxdim=maxdim)
    end
    @timeit_debug timeit_to "wall vorticity operators" begin
        wall_op = wall_vorticity_operator(psi_field, grid; formula=experiment.wall_formula, cutoff=cutoff, maxdim=maxdim)
        lid_source = lid_vorticity_source(psi_field, grid, experiment.U_lid; formula=experiment.wall_formula, cutoff=cutoff, maxdim=maxdim)
    end

    return VorticityCache(diff_ops, Δ, -1.0 * Δ, wall_op, lid_source)
end

"""
    poisson_rhs(experiment::INSVorticityExperiment, cache::VorticityCache, omega::MPS)

Right-hand side of the stream function Poisson system `-Δψ = ω`: the vorticity itself. The walls are ghost nodes
with `ψ = 0`, so no boundary correction is needed.
"""
function poisson_rhs(experiment::INSVorticityExperiment, cache::VorticityCache, omega::MPS)
    return omega
end

"""
    poisson_streamfunction(experiment::INSVorticityExperiment{N}, cache::VorticityCache, omega::MPS, solver::AbstractLinearSolver) where N

Solve the stream function Poisson equation `-Δψ = ω`, using the previous stream function (if any, and nonzero) as
initial guess. If `omega` is identically zero (fluid at rest), returns `ψ = 0` without calling the solver.

# Arguments
- `experiment::INSVorticityExperiment{N}`: The experimental setup.
- `cache::VorticityCache`: Precomputed operators.
- `omega::MPS`: Vorticity.
- `solver::AbstractLinearSolver`: Linear solver for the Poisson system. The operator is symmetric positive definite,
  so CG-based solvers apply. A `ResidualObserver` without `A`/`b` in its `SweepingArgs` is bound to this system.

# Returns
- The stream function `ψ` and the l2 norm of the residual `||A ψ - ω||`.
"""
function poisson_streamfunction(experiment::INSVorticityExperiment{N}, cache::VorticityCache, omega::MPS, solver::AbstractLinearSolver) where N
    rhs = poisson_rhs(experiment, cache, omega)
    if iszero_tt(rhs)
        return zeros_mps(experiment.psi_field.usites, backend_type(rhs)), 0.0
    end
    A = cache.poisson_op

    previous_psi = last(experiment.psi_field)
    psi_guess = (isnothing(previous_psi) || iszero_tt(previous_psi)) ? nothing : previous_psi

    @timeit_debug timeit_to "poisson solver" begin
        psi = linearsolve(solver, A, rhs, psi_guess)
    end

    # l2 error of the linear solver = ||Ax - b||² = <psi|A†A|psi> - 2Re<psi|A†|b> + <b|b>
    term1 = ITensorMPS.inner(A, psi, A, psi)
    term2 = 2 * real(ITensorMPS.inner(rhs', A, psi))
    term3 = ITensorMPS.inner(rhs, rhs)

    res_l2norm = sqrt(max(0.0, term1 - term2 + term3))

    return psi, res_l2norm
end

"""
    streamfunction_velocity(cache::VorticityCache, psi::MPS; cutoff::Float64=1e-15, maxdim::Int=typemax(Int64))

Velocity `(u, v) = (∂ᵧψ, -∂ₓψ)` of the stream function `psi`, on the stored (interior) nodes. With the same central
differences, it is exactly discretely divergence-free: `∂ₓu + ∂ᵧv = (∂ₓ∂ᵧ - ∂ᵧ∂ₓ)ψ = 0`.
"""
function streamfunction_velocity(cache::VorticityCache, psi::MPS; cutoff::Float64=1e-15, maxdim::Int=typemax(Int64))
    d_ops = cache.diff_ops.d_ops
    u = apply(d_ops[2], psi; cutoff=cutoff, maxdim=maxdim)
    v = -1.0 * apply(d_ops[1], psi; cutoff=cutoff, maxdim=maxdim)
    return u, v
end

"""
    wall_vorticity_source(cache::VorticityCache, psi::MPS; cutoff::Float64=1e-15, maxdim::Int=typemax(Int64))

Ghost contribution `s = S ψ + ℓ` of the wall vorticity to the Laplacian of the vorticity on the nodes next to the
walls, i.e. the term that completes the truncated Laplacian `Δω` with the wall values given by the wall vorticity
formula.
"""
function wall_vorticity_source(cache::VorticityCache, psi::MPS; cutoff::Float64=1e-15, maxdim::Int=typemax(Int64))
    return +(apply(cache.wall_op, psi; cutoff=cutoff, maxdim=maxdim), cache.lid_source; cutoff=cutoff, maxdim=maxdim)
end

"""
    VorticityTransportRHS{E, C} <: AbstractRHS

Right-hand side of the vorticity transport equation in conservative form,

    ∂ₜω = -∂ₓ(u ω) - ∂ᵧ(v ω) + (1/Re) (Δω + s),

with the velocity `(u, v)` and the wall vorticity source `s` frozen over the stages of the time integrator. The
normal velocity vanishes on the walls, so the truncated stencils (ghost value 0) give the exact wall fluxes of the
advection term.

# Fields
- `experiment::E`: The experimental setup (`Re`, `cutoff`, `maxdim`).
- `cache::C`: Precomputed operators (`VorticityCache`).
- `u::MPS`, `v::MPS`: Velocity components.
- `source::MPS`: Wall vorticity source `s`, see [`wall_vorticity_source`](@ref).
"""
struct VorticityTransportRHS{E<:INSExperimentBase, C} <: AbstractRHS
    experiment::E
    cache::C
    u::MPS
    v::MPS
    source::MPS
end

function (rhs::VorticityTransportRHS)(omega::Tuple, t::Real, stage::Int)
    experiment = rhs.experiment
    cutoff = experiment.cutoff
    maxdim = experiment.maxdim
    ω = only(omega)
    d_ops = rhs.cache.diff_ops.d_ops

    # 1. Advection (conservative form)
    @timeit_debug timeit_to "advection" begin
        uω = tt_mul(rhs.u, ω; cutoff=cutoff, maxdim=maxdim)
        vω = tt_mul(rhs.v, ω; cutoff=cutoff, maxdim=maxdim)
        advection_term = +(apply(d_ops[1], uω; cutoff=cutoff, maxdim=maxdim), apply(d_ops[2], vω; cutoff=cutoff, maxdim=maxdim); cutoff=cutoff, maxdim=maxdim)
    end

    # 2. Diffusion, completed with the wall vorticity
    @timeit_debug timeit_to "diffusion" begin
        diffusion_term = +(apply(rhs.cache.laplacian, ω; cutoff=cutoff, maxdim=maxdim), rhs.source; cutoff=cutoff, maxdim=maxdim)
    end

    # 3. Combine RHS = - ∇⋅(u ω) + (1/Re) (Δω + s)
    return (+(-1.0 * advection_term, (1 / experiment.Re) * diffusion_term; cutoff=cutoff, maxdim=maxdim),)
end

"""
    vorticity_transport(experiment::INSVorticityExperiment{N}, cache::VorticityCache, omega::MPS, psi::MPS, stepper::AbstractExplicitStepper, t::Real) where N

Advance the vorticity `omega` by one time step `stepper.time_evol_kwargs.dt`, with the velocity and the wall
vorticity of the stream function `psi` frozen over the stages of `stepper`.

# Returns
- The vorticity at `t + dt`.
"""
function vorticity_transport(experiment::INSVorticityExperiment{N}, cache::VorticityCache, omega::MPS, psi::MPS, stepper::AbstractExplicitStepper, t::Real) where N
    cutoff = experiment.cutoff
    maxdim = experiment.maxdim

    @timeit_debug timeit_to "velocity" begin
        u, v = streamfunction_velocity(cache, psi; cutoff=cutoff, maxdim=maxdim)
    end
    @timeit_debug timeit_to "wall vorticity" begin
        source = wall_vorticity_source(cache, psi; cutoff=cutoff, maxdim=maxdim)
    end

    rhs_func = VorticityTransportRHS(experiment, cache, u, v, source)
    @timeit_debug timeit_to "timestep" begin
        omega_next = only(timestep(stepper, (omega,), rhs_func, t))
    end
    return omega_next
end

"""
    vorticity_step(experiment::INSVorticityExperiment{N}, cache::VorticityCache, omega::MPS, solver::AbstractLinearSolver, stepper::AbstractExplicitStepper, t::Real) where N

Advance the vorticity `omega` by one time step `stepper.time_evol_kwargs.dt` with the vorticity-streamfunction
scheme: one stream function Poisson solve, then the explicit vorticity transport with the resulting velocity and
wall vorticity frozen over the stages of `stepper`. Does not modify `experiment`; storing the results is left to
the caller.

# Arguments
- `experiment::INSVorticityExperiment{N}`: The experimental setup. Its last stream function (if any) is used as the Poisson initial guess.
- `cache::VorticityCache`: Precomputed operators.
- `omega::MPS`: Vorticity at time `t`.
- `solver::AbstractLinearSolver`: Linear solver for the stream function Poisson equation.
- `stepper::AbstractExplicitStepper`: Time integrator for the vorticity transport.
- `t::Real`: Time at the start of the step.

# Returns
- `(omega_next, psi, linsolve_error)`: The vorticity at `t + dt`, the stream function at `t` and the l2 residual norm of the Poisson solve.
"""
function vorticity_step(experiment::INSVorticityExperiment{N}, cache::VorticityCache, omega::MPS, solver::AbstractLinearSolver, stepper::AbstractExplicitStepper, t::Real) where N
    psi, linsolve_error = poisson_streamfunction(experiment, cache, omega, solver)
    omega_next = vorticity_transport(experiment, cache, omega, psi, stepper, t)
    return omega_next, psi, linsolve_error
end

"""
    vorticity_evolve!(experiment::INSVorticityExperiment{N}, solver::AbstractLinearSolver, stepper::AbstractExplicitStepper, save_every::Int=1, logger=nothing, log_filename::String="vorticity_log") where N

Run the time evolution of the vorticity-streamfunction experiment. Updates the fields in `experiment` in-place
(appends to time series): at each checkpoint, the stream function of the step and the vorticity at its end.

# Arguments
- `experiment::INSVorticityExperiment{N}`: The experiment object to evolve. Its vorticity must be initialized and its stream function empty.
- `solver::AbstractLinearSolver`: Linear solver for the stream function Poisson equation (`ALS`, `ResidualALS`, `GlobalCG`, ...).
- `stepper::AbstractExplicitStepper`: Time integrator for the vorticity transport (`ExplicitEuler`, `RK4`, ...). Its `TimeEvolArgs` set the time step and the number of steps.
- `save_every::Int`: MPS saving frequency. Defaults to 1 (save every step).
- `logger`: Optional logger for data persistence.
- `log_filename::String`: Filename prefix for logged data.
"""
function vorticity_evolve!(experiment::INSVorticityExperiment{N}, solver::AbstractLinearSolver, stepper::AbstractExplicitStepper, save_every::Int=1, logger=nothing, log_filename::String="vorticity_log") where N
    length(experiment.omega_field) == 0 && throw(ArgumentError("Vorticity field in experiment must be initialized."))
    length(experiment.psi_field) > 0 && throw(ArgumentError("Stream function field must not be initialized."))

    time_evol_kwargs = stepper.time_evol_kwargs
    check_cfl(experiment, stepper; vmax=experiment.U_lid)

    save_checkpoints = collect(1:save_every:time_evol_kwargs.nsteps)
    save_checkpoints = save_checkpoints[end] == time_evol_kwargs.nsteps ? save_checkpoints : push!(save_checkpoints, time_evol_kwargs.nsteps)

    cache = VorticityCache(experiment)
    linsolve_errorlog = Float64[]

    for s in 1:time_evol_kwargs.nsteps
        t = time_evol_kwargs.ti + (s - 1) * time_evol_kwargs.dt
        omega_next, psi, linsolve_error = vorticity_step(experiment, cache, last(experiment.omega_field), solver, stepper, t)

        if s in save_checkpoints
            push!(experiment.psi_field, psi)
            push!(experiment.omega_field, omega_next)
            push!(linsolve_errorlog, linsolve_error)
        else
            experiment.psi_field[length(experiment.psi_field)] = psi
            experiment.omega_field[length(experiment.omega_field)] = omega_next
        end
    end

    if !isnothing(logger)
        log_data(logger, log_filename, "omega_field", experiment.omega_field)
        log_data(logger, log_filename, "psi_field", experiment.psi_field)
        log_data(logger, log_filename, "linsolve_errorlog", linsolve_errorlog)
    end
end

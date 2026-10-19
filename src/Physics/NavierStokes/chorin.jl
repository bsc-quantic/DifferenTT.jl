"""
    ChorinCache

Operators of Chorin's projection method that only depend on the experiment and are built once per run.

# Fields

  - `u_diff_ops`: Velocity differential operators (`DifferentialOps`).
  - `p_diff_ops`: Pressure differential operators (`DifferentialOpsScalar`).
  - `u_boundary_null`: Velocity boundary nullifiers (`BoundaryNull`).
  - `poisson_op::MPO`: Operator `A` of the pressure Poisson system `A pdt = rhs`, i.e. `-Δ`, including the
    rank-1 pressure pinning when `regularize` is true.
  - `regularize::Bool`: Whether the Poisson problem needs regularization (no Dirichlet pressure boundaries).

Besides `ChorinCache(experiment)`, the fields can be given directly to reuse operators built elsewhere.
"""
struct ChorinCache{D,P,B}
    u_diff_ops::D
    p_diff_ops::P
    u_boundary_null::B
    poisson_op::MPO
    regularize::Bool
end

"""
    ChorinCache(experiment::INSExperiment{N}) where N

Build the operators of Chorin's projection method for `experiment`, using its stencil order and TT truncation.
"""
function ChorinCache(experiment::INSExperiment{N}) where {N}
    stencil_order = experiment.stencil_order
    cutoff = experiment.cutoff
    maxdim = experiment.maxdim

    @timeit_debug timeit_to "u_diff_ops" begin
        u_diff_ops = DifferentialOps(
            experiment.grid, experiment.u_fields; stencil_order=stencil_order, cutoff=cutoff, maxdim=maxdim
        )
    end
    @timeit_debug timeit_to "u_boundary_null" begin
        u_boundary_null = BoundaryNull(experiment.u_fields; stencil_order=stencil_order, cutoff=cutoff, maxdim=maxdim)
    end
    @timeit_debug timeit_to "p_diff_ops" begin
        p_diff_ops = DifferentialOpsScalar(
            experiment.grid, experiment.p_field; stencil_order=stencil_order, cutoff=cutoff, maxdim=maxdim
        )
    end
    @timeit_debug timeit_to "laplacian" begin
        Δ = symmetrized_laplacian(
            experiment.p_field, experiment.grid; stencil_order=stencil_order, maxdim=maxdim, cutoff=cutoff
        )
    end

    regularize = regularization_check(experiment)
    @info "regularization = $regularize"

    if regularize
        # Exact Pressure Pinning (Rank-1 Deflation)
        # TODO Implement pressure pinning
        N_sites = length(experiment.p_field.usites)
        c_shift = 1.0 / (2^N_sites)
        mean_pin_mpo = ones_mpo(experiment.p_field.usites, backend_type(Δ))
        Δ = +(Δ, -c_shift * mean_pin_mpo; cutoff=cutoff, maxdim=maxdim)
    end

    return ChorinCache(u_diff_ops, p_diff_ops, u_boundary_null, -1.0 * Δ, regularize)
end

"""
    poisson_rhs(experiment::INSExperiment{N}, cache::ChorinCache, masked_u_mpss::NTuple{N, MPS}, dt::Real) where N

Build the right-hand side of the pressure Poisson system `-Δ pdt = -∇ ⋅ u*`, with `pdt = dt * p`.
Includes the Dirichlet corrections of the divergence and the Laplacian, the Neumann symmetrization and,
if `cache.regularize`, the removal of the mean divergence.

# Arguments

  - `experiment::INSExperiment{N}`: The experimental setup.
  - `cache::ChorinCache`: Precomputed Chorin operators.
  - `masked_u_mpss::NTuple{N,MPS}`: Velocity field after applying mask/boundary corrections.
  - `dt::Real`: Time step.
"""
function poisson_rhs(experiment::INSExperiment{N}, cache::ChorinCache, masked_u_mpss::NTuple{N,MPS}, dt::Real) where {N}
    cutoff = experiment.cutoff
    maxdim = experiment.maxdim

    ∇u = divergence(masked_u_mpss, cache.u_diff_ops; cutoff=cutoff, maxdim=maxdim)

    # @debug_plots begin
    #     @show visualize_scalar(∇u, experiment.grid, experiment.u_fields.arch, "experiments/debug_plots/divu/divu_$(length(experiment.u_fields)).png"; npoints=200, size = (1200, 600))
    # end

    ∇u_correction = divergence_dirichlet_correction(
        experiment.u_fields; stencil_order=experiment.stencil_order, cutoff=cutoff, maxdim=maxdim
    )

    Δp_correction_mpo = scalar_dirichlet_correction(
        experiment.p_field, experiment.grid, 2; stencil_order=experiment.stencil_order, cutoff=cutoff, maxdim=maxdim
    )
    # We assume that the pressure field is contstant at the Dirichlet boundaries -> Identity MPS (one could encode an arbitrary field)
    Δp_correction = if (isnothing(Δp_correction_mpo))
        nothing
    else
        apply(
            Δp_correction_mpo,
            ones_mps(experiment.p_field.usites, 1, backend_type(Δp_correction_mpo));
            alg="naive",
            cutoff=cutoff,
            maxdim=maxdim,
        )
    end

    # @debug_plots begin
    #     @show visualize_scalar(Δp_correction, experiment.grid, experiment.u_fields.arch, "experiments/debug_plots/lapp_correction/lapp_correction_$(length(experiment.u_fields)).png"; npoints=200, size = (1200, 600))
    # end

    ∇u = isnothing(∇u_correction) ? ∇u : apply(∇u_correction, ∇u; cutoff=cutoff, maxdim=maxdim)

    # @debug_plots begin
    #     @show visualize_scalar(∇u, experiment.grid, experiment.u_fields.arch, "experiments/debug_plots/dirich_corrected_divu/dirich_corrected_divu_$(length(experiment.u_fields)).png"; npoints=200, size = (1200, 600))
    # end

    # TODO Clarify what is the right sign for this correction
    ∇u = isnothing(Δp_correction) ? ∇u : +(∇u, (-1.0 * dt) * Δp_correction; cutoff=cutoff, maxdim=maxdim)

    # @debug_plots begin
    #     @show visualize_scalar(∇u, experiment.grid, experiment.u_fields.arch, "experiments/debug_plots/rhs/rhs_$(length(experiment.u_fields)).png"; npoints=200, size = (1200, 600))
    # end

    rhs = symmetrize_rhs(experiment.p_field, -1.0 * ∇u; cutoff=cutoff, maxdim=maxdim)

    if cache.regularize
        # Project RHS to guarantee consistency by explicitly removing the mean divergence 
        # (This is mandatory for pure Neumann / Periodic systems)
        @timeit_debug timeit_to "mean correction integration" begin
            total_div = integrate(rhs) * prod(experiment.grid.dx)
        end
        vol = prod(experiment.grid.box_size)
        rhs = +(
            rhs,
            -(1 / vol) * total_div * ones_mps(experiment.p_field.usites, 1, backend_type(rhs));
            cutoff=cutoff,
            maxdim=maxdim,
        )
    end

    return rhs
end

"""
    poisson_pressure(experiment::INSExperiment{N}, cache::ChorinCache, masked_u_mpss::NTuple{N, MPS}, dt::Real, solver::AbstractLinearSolver) where N

Solve the pressure Poisson equation `-Δ pdt = -∇ ⋅ u*` for `pdt = dt * p`, using the previous pressure (if any) as initial guess.

# Arguments

  - `experiment::INSExperiment{N}`: The experimental setup.
  - `cache::ChorinCache`: Precomputed Chorin operators.
  - `masked_u_mpss::NTuple{N,MPS}`: Velocity field after applying mask/boundary corrections.
  - `dt::Real`: Time step.
  - `solver::AbstractLinearSolver`: Linear solver for the Poisson system. A `ResidualObserver` without `A`/`b` in its
    `SweepingArgs` is bound to this step's system.

# Returns

  - The physical pressure `p = pdt / dt` and the l2 norm of the residual `||A pdt - rhs||`.
"""
function poisson_pressure(
    experiment::INSExperiment{N},
    cache::ChorinCache,
    masked_u_mpss::NTuple{N,MPS},
    dt::Real,
    solver::AbstractLinearSolver,
) where {N}
    rhs = poisson_rhs(experiment, cache, masked_u_mpss, dt)
    A = cache.poisson_op

    previous_p = last(experiment.p_field)
    # p_field stores physical p = pdt / dt; the linear system is for pdt.
    pdt_guess = isnothing(previous_p) ? nothing : previous_p * dt

    @timeit_debug timeit_to "poisson solver" begin
        pdt = linearsolve(solver, A, rhs, pdt_guess)
    end

    # l2 error of the linear solver = ||Ax - b||² = <psi|A†A|psi> - 2Re<psi|A†|b> + <b|b>
    term1 = ITensorMPS.inner(A, pdt, A, pdt)
    term2 = 2 * real(ITensorMPS.inner(rhs', A, pdt))
    term3 = ITensorMPS.inner(rhs, rhs)

    res_l2norm = sqrt(max(0.0, term1 - term2 + term3))

    # @debug_plots begin
    #     @show visualize_scalar(pdt * (1 / dt), experiment.grid, experiment.u_fields.arch, "experiments/debug_plots/solp/solp_$(length(experiment.u_fields)).png"; npoints=200, size = (1200, 600))
    # end

    return pdt * (1 / dt), res_l2norm
end

struct NavierStokesPredictorRHS{E<:INSExperimentBase,D,B} <: AbstractRHS
    experiment::E
    u_diff_ops::D
    u_boundary_null::B
end

function (rhs::NavierStokesPredictorRHS)(u::Tuple, t::Real, stage::Int)
    experiment = rhs.experiment
    inv_Re = experiment.Re^-1
    cutoff = experiment.cutoff
    maxdim = experiment.maxdim
    N = length(u)

    # 1. Advection
    @timeit_debug timeit_to "advection" begin
        advection_terms = advection(u, rhs.u_diff_ops; cutoff=cutoff, maxdim=maxdim)
    end

    # 2. Diffusion
    @timeit_debug timeit_to "diffusion" begin
        diffusion_terms = diffusion(u, rhs.u_diff_ops; cutoff=cutoff, maxdim=maxdim)
    end

    # 3. Combine RHS = - (u.∇)u + (1/Re) ∇²u
    b_terms = ntuple(i -> +(-1.0 * advection_terms[i], inv_Re * diffusion_terms[i]; cutoff=cutoff, maxdim=maxdim), N)

    # 4. Apply Boundary Nullifiers
    b_corrected = ntuple(
        i -> begin
            deleter = rhs.u_boundary_null.components[i].u_row_deleter
            return isnothing(deleter) ? b_terms[i] : apply(deleter, b_terms[i]; cutoff=cutoff, maxdim=maxdim)
        end, N
    )

    return b_corrected
end

"""
    ∇p_correction(u_mpss::NTuple{N,MPS}, experiment::INSExperiment{N}, cache::ChorinCache, p::MPS, dt::Real) where N

Project the velocity field to be divergence-free by subtracting the pressure gradient: `u = u* - dt * ∇p`.
Includes Dirichlet boundary corrections for the pressure gradient.

# Arguments

  - `u_mpss::NTuple{N, MPS}`: The intermediate velocity field.
  - `experiment::INSExperiment{N}`: The experimental setup.
  - `cache::ChorinCache`: Precomputed Chorin operators.
  - `p::MPS`: The pressure field.
  - `dt::Real`: Time step.
"""
function ∇p_correction(
    u_mpss::NTuple{N,MPS}, experiment::INSExperiment{N}, cache::ChorinCache, p::MPS, dt::Real
) where {N}
    cutoff = experiment.cutoff
    maxdim = experiment.maxdim

    @timeit_debug timeit_to "divfree correction" begin
        u_corrected = ntuple(
            i -> begin
                dp = apply(cache.p_diff_ops.d_ops[i], p; cutoff=cutoff, maxdim=maxdim)
                # We correct dp/dxi per dimension, not mixing in corrections from other dimensions.
                dirichlet_correction_mpo = scalar_dirichlet_correction_dim(
                    experiment.p_field,
                    experiment.grid,
                    1,
                    i;
                    stencil_order=experiment.stencil_order,
                    cutoff=cutoff,
                    maxdim=maxdim,
                )

                # We assume that the pressure field is contstant at the Dirichlet boundaries -> Identity MPS (one could encode an arbitrary field)
                dirichlet_correction = if (isnothing(dirichlet_correction_mpo))
                    nothing
                else
                    apply(
                        dirichlet_correction_mpo,
                        ones_mps(experiment.p_field.usites, 1, backend_type(dirichlet_correction_mpo));
                        cutoff=cutoff,
                        maxdim=maxdim,
                    )
                end

                full_dp = if isnothing(dirichlet_correction)
                    dp # Correct according to -dt * ∇p
                else
                    +(dp, dirichlet_correction; cutoff=cutoff, maxdim=maxdim) # Correct according to -dt * ∇p
                end # Correct according to -dt * ∇p

                output = +(u_mpss[i], (-1.0 * dt) * full_dp; cutoff=cutoff, maxdim=maxdim)
                return output
            end,
            N,
        )
    end
    return u_corrected
end

"""
Apply a geometric mask to the velocity field (e.g., for obstacles) by element-wise multiplication. If isnothing(mask) == true, no correction is applied.

# Arguments

  - `u_mpss::NTuple{N, MPS}`: Velocity field.
  - `mask::MPS`: Mask MPS (1.0 = fluid, 0.0 = obstacle).
"""
function mask_correction(u_mpss::NTuple{N,MPS}, mask::MPS; cutoff::Float64=1e-15, maxdim::Int=typemax(Int64)) where {N}
    return ntuple(i -> tt_mul(u_mpss[i], mask; cutoff=cutoff, maxdim=maxdim), N)
end

function mask_correction(
    u_mpss::NTuple{N,MPS}, mask::Nothing; cutoff::Float64=1e-15, maxdim::Int=typemax(Int64)
) where {N}
    return u_mpss
end

"""
    regularization_check(experiment::INSExperiment{N}) where N

Check whether the pressure Poisson equation requires regularization (mean subtraction + small shift).
Returns `true` if no Dirichlet BCs are present on any pressure dimension (i.e. all periodic or Neumann).
"""
function regularization_check(experiment::INSExperiment{N}) where {N}
    for i in 1:N
        if experiment.p_field.bcu[i].left isa DirichletBC || experiment.p_field.bcu[i].right isa DirichletBC
            return false
        end
    end
    return true
end

"""
    chorin_step(experiment::INSExperiment{N}, cache::ChorinCache, u_mpss::NTuple{N,MPS}, solver::AbstractLinearSolver, stepper::AbstractExplicitStepper, t::Real) where N

Advance the velocity `u_mpss` by one time step `stepper.time_evol_kwargs.dt` with Chorin's projection method:
mask correction, pressure Poisson solve, divergence-free projection and explicit advection-diffusion predictor.
Does not modify `experiment`; storing the results is left to the caller.

# Arguments

  - `experiment::INSExperiment{N}`: The experimental setup. Its last pressure (if any) is used as the Poisson initial guess.
  - `cache::ChorinCache`: Precomputed Chorin operators.
  - `u_mpss::NTuple{N,MPS}`: Velocity field at time `t`.
  - `solver::AbstractLinearSolver`: Linear solver for the pressure Poisson equation.
  - `stepper::AbstractExplicitStepper`: Time integrator for the predictor.
  - `t::Real`: Time at the start of the step.

# Returns

  - `(u_next, p, linsolve_error)`: The velocity at `t + dt`, the pressure and the l2 residual norm of the Poisson solve.
"""
function chorin_step(
    experiment::INSExperiment{N},
    cache::ChorinCache,
    u_mpss::NTuple{N,MPS},
    solver::AbstractLinearSolver,
    stepper::AbstractExplicitStepper,
    t::Real,
) where {N}
    dt = stepper.time_evol_kwargs.dt

    @timeit_debug timeit_to "mask correction" begin
        masked_u_mpss = mask_correction(
            u_mpss, experiment.object_mask; cutoff=experiment.cutoff, maxdim=experiment.maxdim
        )
    end

    p, linsolve_error = poisson_pressure(experiment, cache, masked_u_mpss, dt, solver)

    div_free_u_mpss = ∇p_correction(masked_u_mpss, experiment, cache, p, dt)

    # Predictor step: Explict integration of advection-diffusion
    rhs_func = NavierStokesPredictorRHS(experiment, cache.u_diff_ops, cache.u_boundary_null)
    @timeit_debug timeit_to "timestep" begin
        u_next = timestep(stepper, div_free_u_mpss, rhs_func, t)
    end

    return u_next, p, linsolve_error
end

"""
    chorin_evolve!(experiment::INSExperiment{N}, solver::AbstractLinearSolver, stepper::AbstractExplicitStepper, save_every::Int=1, logger=nothing, log_filename::String="chorin_log") where N

Run the time evolution of the Incompressible Navier-Stokes experiment using Chorin's projection method.
Updates the fields in `experiment` in-place (appends to time series).

# Arguments

  - `experiment::INSExperiment{N}`: The experiment object to evolve.
  - `solver::AbstractLinearSolver`: Linear solver for the pressure Poisson equation (`ALS`, `ResidualALS`, `GlobalCG`, ...). Truncation and stopping criteria come from its `sweep_kwargs` / `updater_kwargs`.
  - `stepper::AbstractExplicitStepper`: Time integrator for the explicit advection-diffusion predictor (`ExplicitEuler`, `RK4`, ...). Its `TimeEvolArgs` set the time step and the number of steps.
  - `save_every::Int`: MPS saving frequency. Defaults to 1 (save every step).
  - `logger`: Optional logger for data persistence.
  - `log_filename::String`: Filename prefix for logged data.
"""
function chorin_evolve!(
    experiment::INSExperiment{N},
    solver::AbstractLinearSolver,
    stepper::AbstractExplicitStepper,
    save_every::Int=1,
    logger=nothing,
    log_filename::String="chorin_log",
) where {N}
    any(x -> length(x) == 0, [component.u for component in experiment.u_fields.components]) && throw(ArgumentError("Velocity fields in experiment must be initialized."))
    length(experiment.p_field.u) > 0 && throw(ArgumentError("Pressure field must not be initialized."))

    time_evol_kwargs = stepper.time_evol_kwargs
    check_cfl(experiment, stepper)

    save_checkpoints = collect(1:save_every:time_evol_kwargs.nsteps)
    save_checkpoints = if save_checkpoints[end] == time_evol_kwargs.nsteps
        save_checkpoints
    else
        push!(save_checkpoints, time_evol_kwargs.nsteps)
    end

    cache = ChorinCache(experiment)
    linsolve_errorlog = Float64[]

    for s in 1:time_evol_kwargs.nsteps
        t = time_evol_kwargs.ti + (s - 1) * time_evol_kwargs.dt
        u_next, p, linsolve_error = chorin_step(experiment, cache, last(experiment.u_fields), solver, stepper, t)

        if s in save_checkpoints
            push!(experiment.p_field, p)
            push!(experiment.u_fields, u_next)
            push!(linsolve_errorlog, linsolve_error)
        else
            experiment.p_field[length(experiment.p_field)] = p
            experiment.u_fields[length(experiment.u_fields)] = u_next
        end
    end

    if !isnothing(logger)
        log_data(logger, log_filename, "u_fields", experiment.u_fields)
        log_data(logger, log_filename, "p_field", experiment.p_field)
        log_data(logger, log_filename, "linsolve_errorlog", linsolve_errorlog)
    end
end

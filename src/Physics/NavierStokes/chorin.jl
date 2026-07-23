"""
    poisson_pressure(experiment::INSExperiment{N}, u_diff_ops::DifferentialOps{N}, Δ::MPO; solver::Function = gmres_linsolve, kwargs...) where N

Solve the pressure Poisson equation: `Δp = (1/dt) ∇ ⋅ u*`.

# Arguments
- `experiment::INSExperiment{N}`: The experimental setup.
- `masked_u_mpss::NTuple{N,MPS}`: Velocity field after applying mask/boundary corrections, used to build a consistent divergence right-hand side.
- `u_diff_ops::DifferentialOps{N}`: Velocity differential operators.
- `Δ::MPO`: The Laplacian operator.
- `M⁻¹::Union{Nothing,MPO}`: A preconditioner for a Krylov-based method. If nothing, no preconditioner will be used.
- `solver::Function`: Linear solver function to use.
- `kwargs...`: Arguments passed to the solver.
"""
function poisson_pressure(
    experiment::INSExperiment{N},
    masked_u_mpss::NTuple{N, MPS},
    u_diff_ops::DifferentialOps{N},
    Δ::MPO,
    solver::AbstractLinearSolver;
    regularize::Bool=false
) where N

    dt = experiment.time_evol_settings.dt
    cutoff = experiment.cutoff
    maxdim = experiment.maxdim

    ∇u = divergence(masked_u_mpss, u_diff_ops; cutoff=cutoff, maxdim=maxdim)

    # @debug_plots begin
    #     @show visualize_scalar(∇u, experiment.grid, experiment.u_fields.arch, "experiments/debug_plots/divu/divu_$(length(experiment.u_fields)).png"; npoints=200, size = (1200, 600))
    # end

    ∇u_correction = divergence_dirichlet_correction(experiment.u_fields; stencil_order=experiment.stencil_order, cutoff=cutoff, maxdim=maxdim)

    Δp_correction_mpo = pressure_dirichlet_correction(experiment.p_field, experiment.grid, 2; stencil_order=experiment.stencil_order, cutoff=cutoff, maxdim=maxdim)
    # We assume that the pressure field is contstant at the Dirichlet boundaries -> Identity MPS (one could encode an arbitrary field)
    Δp_correction = (isnothing(Δp_correction_mpo)) ? nothing : apply(Δp_correction_mpo, ones_mps(experiment.p_field.usites, 1, backend_type(Δp_correction_mpo)); alg="naive", cutoff=cutoff, maxdim=maxdim)

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

    previous_p = last(experiment.p_field)
    epsilon = 1e-10
    rhs = symmetrize_rhs(experiment.p_field, -1.0 * ∇u; cutoff=cutoff, maxdim=maxdim)

    # @timeit_debug timeit_to "mean correction integration" begin
    #     total_div = integrate(rhs) * prod(grid.dx)
    # end

    vol = prod(experiment.grid.box_size)

    # TODO Implement pressure pinning
    if regularize
        # 1. Project RHS to guarantee consistency by explicitly removing the mean divergence 
        # (This is mandatory for pure Neumann / Periodic systems)
        total_div = integrate(rhs) * prod(experiment.grid.dx)
        rhs = +(rhs, -(1 / vol) * total_div * ones_mps(experiment.p_field.usites, 1, backend_type(rhs)); cutoff=cutoff, maxdim=maxdim)

        # 2. Exact Pressure Pinning (Rank-1 Deflation)
        N_sites = length(experiment.p_field.usites)
        c_shift = 1.0 / (2^N_sites)
        mean_pin_mpo = ones_mpo(experiment.p_field.usites, backend_type(Δ))
        
        Δ = +(Δ, -c_shift * mean_pin_mpo; cutoff=cutoff, maxdim=maxdim)
    end


    # @timeit_debug timeit_to "poisson solver" begin 
        pdt = linearsolve(solver, -1.0 * Δ, rhs, previous_p)
    # end

    # l2 error of the linear solver = ||Ax - b||² = <psi|A†A|psi> - 2Re<psi|A†|b> + <b|b>
    term1 = ITensorMPS.inner(Δ, pdt, Δ, pdt)
    term2 = 2 * real(ITensorMPS.inner(rhs', -1.0 * Δ, pdt))
    term3 = ITensorMPS.inner(rhs, rhs)

    res_l2norm = sqrt(max(0.0, term1 - term2 + term3))

    # @debug_plots begin
    #     @show visualize_scalar(pdt * (1 / dt), experiment.grid, experiment.u_fields.arch, "experiments/debug_plots/solp/solp_$(length(experiment.u_fields)).png"; npoints=200, size = (1200, 600))
    # end

    return pdt * (1 / dt), res_l2norm
end

struct NavierStokesPredictorRHS{E<:INSExperimentBase, D, B} <: AbstractRHS
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
    advection_terms = advection(u, rhs.u_diff_ops; cutoff=cutoff, maxdim=maxdim)
    
    # 2. Diffusion
    diffusion_terms = diffusion(u, rhs.u_diff_ops; cutoff=cutoff, maxdim=maxdim)

    # 3. Combine RHS = - (u.∇)u + (1/Re) ∇²u
    b_terms = ntuple(i -> +(-1.0 * advection_terms[i], inv_Re * diffusion_terms[i]; cutoff=cutoff, maxdim=maxdim), N)

    # 4. Apply Boundary Nullifiers
    b_corrected = ntuple(i -> begin
            deleter = rhs.u_boundary_null.components[i].u_row_deleter
            return isnothing(deleter) ? b_terms[i] : apply(deleter, b_terms[i]; cutoff=cutoff, maxdim=maxdim)
        end, N)

    return b_corrected
end

"""
    ∇p_correction(u_mpss::NTuple{N,MPS}, experiment::INSExperiment{N}, p_diff_ops::DifferentialOpsScalar{N}, u_boundary_null::BoundaryNull{N}) where N

Project the velocity field to be divergence-free by subtracting the pressure gradient: `u = u* - dt * ∇p`.
Includes Dirichlet boundary corrections for the pressure gradient.

# Arguments
- `u_mpss::NTuple{N, MPS}`: The intermediate velocity field.
- `experiment::INSExperiment{N}`: The experimental setup.
- `p_diff_ops::DifferentialOpsScalar{N}`: Pressure differential operators.
- `u_boundary_null::BoundaryNull{N}`: Boundary nullification operators.
"""
function ∇p_correction(u_mpss::NTuple{N,MPS}, experiment::INSExperiment{N}, p_diff_ops::DifferentialOpsScalar{N}, u_boundary_null::BoundaryNull{N}) where N
    p = last(experiment.p_field)

    cutoff = experiment.cutoff
    maxdim = experiment.maxdim
    dt = experiment.time_evol_settings.dt

    u_corrected = ntuple(i -> begin
            dp = apply(p_diff_ops.d_ops[i], p; cutoff=cutoff, maxdim=maxdim)
            # We correct dp/dxi per dimension, not mixing in corrections from other dimensions.
            dirichlet_correction_mpo = pressure_dirichlet_correction_dim(experiment.p_field, experiment.grid, 1, i; stencil_order=experiment.stencil_order, cutoff=cutoff, maxdim=maxdim)

            # We assume that the pressure field is contstant at the Dirichlet boundaries -> Identity MPS (one could encode an arbitrary field)
            dirichlet_correction = (isnothing(dirichlet_correction_mpo)) ? nothing : apply(dirichlet_correction_mpo, ones_mps(experiment.p_field.usites, 1, backend_type(dirichlet_correction_mpo)); cutoff=cutoff, maxdim=maxdim)

            full_dp = isnothing(dirichlet_correction) ? dp : +(dp, dirichlet_correction; cutoff=cutoff, maxdim=maxdim) # Correct according to -dt * ∇p

            output = +(u_mpss[i], (-1.0 * dt) * full_dp; cutoff=cutoff, maxdim=maxdim)
            return output
        end, N)

    return u_corrected
end

"""
Apply a geometric mask to the velocity field (e.g., for obstacles) by element-wise multiplication. If isnothing(mask) == true, no correction is applied.

# Arguments
- `u_mpss::NTuple{N, MPS}`: Velocity field.
- `mask::MPS`: Mask MPS (1.0 = fluid, 0.0 = obstacle).
"""
function mask_correction(u_mpss::NTuple{N,MPS}, mask::MPS; cutoff::Float64=1e-15, maxdim::Int=typemax(Int64)) where N
    return ntuple(i -> tt_mul(u_mpss[i], mask; cutoff=cutoff, maxdim=maxdim), N)
end

function mask_correction(u_mpss::NTuple{N,MPS}, mask::Nothing; cutoff::Float64=1e-15, maxdim::Int=typemax(Int64)) where N
    return u_mpss
end

"""
    regularization_check(experiment::INSExperiment{N}) where N

Check whether the pressure Poisson equation requires regularization (mean subtraction + small shift).
Returns `true` if no Dirichlet BCs are present on any pressure dimension (i.e. all periodic or Neumann).
"""
function regularization_check(experiment::INSExperiment{N}) where N
    for i in 1:N
        if experiment.p_field.bcu[i].left isa DirichletBC || experiment.p_field.bcu[i].right isa DirichletBC
            return false
        end
    end
    return true
end

"""
    chorin_evolve!(experiment::INSExperiment{N}, save_every::Int=1, logger=nothing, log_filename::String="chorin_euler_log"; solver::Function=cg_linsolve, linsolve_kwargs) where N

Run the time evolution of the Incompressible Navier-Stokes experiment using Chorin's projection method.
Supports Euler and RK4 time integration. Updates the fields in `experiment` in-place (appends to time series).

# Arguments
- `experiment::INSExperiment{N}`: The experiment object to evolve.
- `save_every::Int`: MPS saving frequency. Defaults to 1 (save every step).
- `logger`: Optional logger for data persistence.
- `log_filename::String`: Filename prefix for logged data.
- `solver::AbstractLinearSolver`: Linear solver for the pressure Poisson equation.
"""
function chorin_evolve!(experiment::INSExperiment{N}, solver::AbstractLinearSolver, save_every::Int=1, logger=nothing, log_filename::String="chorin_euler_log") where N
    any(x -> length(x) == 0, [component.u for component in experiment.u_fields.components]) && throw(ArgumentError("Velocity fields in experiment must be initialized."))
    length(experiment.p_field.u) > 0 && throw(ArgumentError("Pressure field must not be initialized."))

    save_checkpoints = collect(1:save_every:experiment.time_evol_settings.nsteps)
    save_checkpoints = save_checkpoints[end] == experiment.time_evol_settings.nsteps ? save_checkpoints : push!(save_checkpoints, experiment.time_evol_settings.nsteps)

    u_diff_ops = DifferentialOps(experiment.grid, experiment.u_fields; stencil_order=experiment.stencil_order, cutoff=experiment.cutoff, maxdim=experiment.maxdim)
    u_boundary_nullifier = BoundaryNull(experiment.u_fields; stencil_order=experiment.stencil_order, cutoff=experiment.cutoff, maxdim=experiment.maxdim)
    p_diff_ops = DifferentialOpsScalar(experiment.grid, experiment.p_field; stencil_order=experiment.stencil_order, cutoff=experiment.cutoff, maxdim=experiment.maxdim)
    Δ_cache = symmetrized_laplacian(experiment.p_field, experiment.grid; stencil_order=experiment.stencil_order, maxdim=experiment.maxdim, cutoff=experiment.cutoff)

    regularization_needed = regularization_check(experiment)
    @info "regularization = $regularization_needed"
    linsolve_errorlog = Float64[]

    scheme_sym = experiment.time_evol_settings.scheme
    if scheme_sym == :euler
        stepper = ExplicitEuler(cutoff=experiment.cutoff, maxdim=experiment.maxdim)
    elseif scheme_sym == :rk4
        stepper = RK4(cutoff=experiment.cutoff, maxdim=experiment.maxdim)
    else
        throw(ArgumentError("Unknown scheme $scheme_sym"))
    end
    
    rhs_func = NavierStokesPredictorRHS(experiment, u_diff_ops, u_boundary_nullifier)

    for s in 1:experiment.time_evol_settings.nsteps
        try
            prev_u_mpss = last(experiment.u_fields)

            masked_u_mpss = mask_correction(prev_u_mpss, experiment.object_mask; cutoff=experiment.cutoff, maxdim=experiment.maxdim)

            divfree_p, linsolve_error = poisson_pressure(experiment, masked_u_mpss, u_diff_ops, Δ_cache, solver; regularize=regularization_needed)
            
            if s in save_checkpoints
                push!(experiment.p_field, divfree_p)
            else
                last_index = length(experiment.p_field)
                experiment.p_field[last_index] = divfree_p
            end

            div_free_u_mpss = ∇p_correction(masked_u_mpss, experiment, p_diff_ops, u_boundary_nullifier)

            # Predictor step: Explict integration of advection-diffusion
            timestepped_u_mpss = timestep(stepper, div_free_u_mpss, experiment.time_evol_settings.dt, rhs_func, s * experiment.time_evol_settings.dt)

            if s in save_checkpoints
                push!(experiment.u_fields, timestepped_u_mpss)
                push!(linsolve_errorlog, linsolve_error)
            else
                last_index = length(experiment.u_fields)
                experiment.u_fields[last_index] = timestepped_u_mpss
            end
            
        catch e
            throw(e)
            break
        end
    end

    if !isnothing(logger)
        # Note: logger logic depends on external save implementations
        # log_data(logger, log_filename, "u_fields", experiment.u_fields)
        # log_data(logger, log_filename, "p_field", experiment.p_field)
        # log_data(logger, log_filename, "linsolve_errorlog", linsolve_errorlog)
    end
end

"""
    ChorinCache(experiment::INSExperiment{N}) where N
    ChorinCache(u_diff_ops, p_diff_ops, u_boundary_null, poisson_op::MPO, regularize::Bool)

Operators of the Chorin projection that depend only on the experiment, built once per run.

The pressure operator is the same at every time step, since the obstacle only enters through the mask on the
velocity, so it is built once, together with the velocity and pressure derivatives and the boundary projectors. The
Poisson system for ``p\\,dt`` is

```math
A = -W\\Delta + c\\,\\mathbf{1}\\mathbf{1}^\\top,
```

where ``W\\Delta`` is [`symmetrized_laplacian`](@ref) and the rank-1 term, with ``c = 2^{-N_s}`` for ``N_s`` sites,
is only added when the pressure has no `DirichletBC` side ([`regularization_check`](@ref)). Without it, ``A`` would
have the constant null space; with it, ``A`` is SPD and the solution of a consistent system is the one of zero mean
(for any ``c > 0``). Note that ``2^{-N_s}`` is the inverse number of nodes in QTT only. The first method builds every
operator with the stencil order and the truncation of `experiment`; the second takes them directly, to reuse
operators built elsewhere.

# Fields

  - `u_diff_ops`: the velocity derivatives ([`DifferentialOps`](@ref)).
  - `p_diff_ops`: the pressure derivatives ([`DifferentialOpsScalar`](@ref)).
  - `u_boundary_null`: the velocity boundary projectors ([`BoundaryNull`](@ref)).
  - `poisson_op::MPO`: the operator ``A`` above.
  - `regularize::Bool`: whether the rank-1 term was added.

See also [`poisson_rhs`](@ref), [`chorin_step`](@ref), and the boundary-condition tutorial [Velocity–pressure
formulation](@ref bc_velocity_pressure).
"""
struct ChorinCache{D,P,B}
    u_diff_ops::D
    p_diff_ops::P
    u_boundary_null::B
    poisson_op::MPO
    regularize::Bool
end

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
    @debug "regularization = $regularize"

    if regularize
        # Rank-1 regularization of the constant null space (see the docstring)
        # TODO(release): implement pressure pinning (fixing p at one node) as an alternative to the rank-1 shift
        N_sites = length(experiment.p_field.usites)
        c_shift = 1.0 / (2^N_sites)
        mean_pin_mpo = ones_mpo(experiment.p_field.usites, backend_type(Δ))
        Δ = +(Δ, -c_shift * mean_pin_mpo; cutoff=cutoff, maxdim=maxdim)
    end

    return ChorinCache(u_diff_ops, p_diff_ops, u_boundary_null, -1.0 * Δ, regularize)
end

"""
    poisson_rhs(experiment::INSExperiment{N}, cache::ChorinCache, masked_u_mpss::NTuple{N,MPS}, dt::Real) where N -> MPS

Build the right-hand side of the pressure Poisson system of the Chorin projection, Eq. (34) of Torrente et al.,
*Factorized approximate inverse preconditioners in the tensor train format*, solved for ``p\\,dt``.

Imposing ``\\nabla \\cdot \\mathbf{u}_{n+1} = 0`` on the projected velocity gives
``\\Delta (p\\,dt) = \\nabla \\cdot \\mathbf{u}^*``. The truncated Laplacian misses the Dirichlet values of the pressure
on the ghost nodes, and the rows of the divergence next to the Dirichlet velocity faces are not consistent, so the
right-hand side of the system ``A\\,(p\\,dt) = \\mathbf{b}`` of [`ChorinCache`](@ref) is

```math
\\mathbf{b} = W \\left( -Z\\, \\nabla \\cdot \\mathbf{u}^* + dt\\, C\\mathbf{1} \\right),
```

where ``Z`` is the projector of [`divergence_dirichlet_correction`](@ref), ``C`` the Dirichlet lift operator of
[`scalar_dirichlet_correction`](@ref) for the second derivative (the ghost values of ``p\\,dt`` are ``dt`` times the
pressure values), and ``W`` the Neumann weight of [`symmetrize_rhs`](@ref). If `cache.regularize`, a constant is then
subtracted to make the singular system consistent, which is one of the known limitations of [`INSExperiment`](@ref)
(the constant is not the node mean).

# Arguments

  - `experiment::INSExperiment{N}`: the experiment, whose truncation is used.
  - `cache::ChorinCache`: the operators of the experiment.
  - `masked_u_mpss::NTuple{N,MPS}`: the velocity ``\\mathbf{u}^*``, already multiplied by the obstacle mask.
  - `dt::Real`: the time step.
"""
function poisson_rhs(experiment::INSExperiment{N}, cache::ChorinCache, masked_u_mpss::NTuple{N,MPS}, dt::Real) where {N}
    cutoff = experiment.cutoff
    maxdim = experiment.maxdim

    ∇u = divergence(masked_u_mpss, cache.u_diff_ops; cutoff=cutoff, maxdim=maxdim)

    ∇u_correction = divergence_dirichlet_correction(
        experiment.u_fields; stencil_order=experiment.stencil_order, cutoff=cutoff, maxdim=maxdim
    )

    Δp_correction_mpo = scalar_dirichlet_correction(
        experiment.p_field, experiment.grid, 2; stencil_order=experiment.stencil_order, cutoff=cutoff, maxdim=maxdim
    )
    # The pressure is constant on each Dirichlet side, so the lift operator acts on the all-ones MPS
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

    ∇u = isnothing(∇u_correction) ? ∇u : apply(∇u_correction, ∇u; cutoff=cutoff, maxdim=maxdim)

    # Δ_full (p dt) = Δ (p dt) + dt C 1, since the ghost values of p dt are dt times those of p, so -Δ (p dt) = -∇⋅u* +
    # dt C 1 (checked against the full stencil on the ghost-extended vector)
    ∇u = isnothing(Δp_correction) ? ∇u : +(∇u, (-1.0 * dt) * Δp_correction; cutoff=cutoff, maxdim=maxdim)

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
    poisson_pressure(experiment::INSExperiment{N}, cache::ChorinCache, masked_u_mpss::NTuple{N,MPS}, dt::Real,
                     solver::AbstractLinearSolver) where N -> Tuple{MPS,Float64}

Solve the pressure system ``A\\,(p\\,dt) = \\mathbf{b}`` of [`ChorinCache`](@ref) and [`poisson_rhs`](@ref) with
`solver`, starting from ``dt`` times the last stored pressure, if any (a `ResidualObserver` without `A` and `b` in the
`SweepingArgs` of `solver` is bound to this system). Returns the pressure ``p`` and the residual norm
``\\lVert A\\,(p\\,dt) - \\mathbf{b} \\rVert_2``, evaluated with inner products.
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

    return pdt * (1 / dt), res_l2norm
end

"""
    NavierStokesPredictorRHS(experiment::INSExperimentBase, u_diff_ops, u_boundary_null) <: AbstractRHS

Right-hand side of the predictor of the Chorin projection, Eq. (33) of Torrente et al., *Factorized approximate inverse
preconditioners in the tensor train format*,

```math
\\partial_t \\mathbf{u} = \\frac{1}{Re} \\nabla^2 \\mathbf{u} - (\\mathbf{u} \\cdot \\nabla)\\mathbf{u},
```

as a callable for the explicit time steppers: `rhs(u::Tuple, t, stage)` returns, for every component ``i``,
``Z_i \\left( \\frac{1}{Re} \\nabla^2 u_i - (\\mathbf{u} \\cdot \\nabla) u_i \\right)``, with the [`advection`](@ref)
and [`diffusion`](@ref) of `u_diff_ops` and the projector ``Z_i`` of `u_boundary_null` (see
[`BoundaryNullScalar`](@ref)), which keeps the Dirichlet boundary rows of the velocity fixed. It does not depend on `t`
or `stage`. The truncation is that of `experiment`.

# Fields

  - `experiment::E`: the experiment (`Re`, `cutoff`, `maxdim`).
  - `u_diff_ops::D`: the velocity derivatives ([`DifferentialOps`](@ref)).
  - `u_boundary_null::B`: the velocity boundary projectors ([`BoundaryNull`](@ref)).

See also [`AbstractRHS`](@ref), [`chorin_step`](@ref).
"""
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

Return the projected velocity ``u_i = u_i^* - dt\\, (\\partial_i p + C_i \\mathbf{1})``, where ``\\partial_i p`` uses
the pressure derivatives of `cache` and ``C_i`` the Dirichlet lift of the first derivative along axis ``i``
([`scalar_dirichlet_correction_dim`](@ref)). The correction acts on every row, including the Dirichlet boundary rows of
the velocity, which is one of the known limitations of [`INSExperiment`](@ref).
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

                # The pressure is constant on each Dirichlet side, so the lift operator acts on the all-ones MPS
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
    mask_correction(u_mpss::NTuple{N,MPS}, mask::MPS; cutoff::Float64=1e-15, maxdim::Int=typemax(Int64)) where N
    mask_correction(u_mpss::NTuple{N,MPS}, mask::Nothing; cutoff::Float64=1e-15, maxdim::Int=typemax(Int64)) where N

Return the velocity multiplied element-wise by the obstacle mask, ``u_i \\odot m`` for every component, which immerses
the obstacle in the fluid (Sec. 5.2 of Torrente et al., *Factorized approximate inverse preconditioners in the tensor
train format*). The mask is 1 in the fluid and 0 inside the obstacle, e.g. the smoothed indicator of a cylinder computed
by `crossinterpolate2`, and the products are computed by [`tt_mul`](@ref) with `cutoff` and `maxdim`. With
`mask = nothing`, `u_mpss` is returned unchanged.
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
    regularization_check(experiment::INSExperiment{N}) where N -> Bool

Return whether the pressure Poisson system of `experiment` is singular, i.e. whether no side of the pressure is a
`DirichletBC` (every side periodic, Neumann or open). The Laplacian then has the constant null space, which
[`ChorinCache`](@ref) removes with a rank-1 term and [`poisson_rhs`](@ref) compensates by subtracting a constant.
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
    chorin_step(experiment::INSExperiment{N}, cache::ChorinCache, u_mpss::NTuple{N,MPS}, solver::AbstractLinearSolver,
                stepper::AbstractExplicitStepper, t::Real) where N -> Tuple

Advance the velocity `u_mpss` by one time step ``dt`` = `stepper.time_evol_kwargs.dt` with the Chorin projection
(Chorin, 1967), as in the QTT pipeline of Peddinti et al. (2024) and Sec. 5.2 of Torrente et al., *Factorized
approximate inverse preconditioners in the tensor train format*.

The step proceeds as follows. We first multiply the velocity by the obstacle mask ([`mask_correction`](@ref)). We then
solve the pressure Poisson system of [`poisson_rhs`](@ref) with `solver`, starting from the last pressure stored in
`experiment`, and project the masked velocity, ``\\mathbf{u} = \\mathbf{u}^* - dt\\, \\nabla p`` (`∇p_correction`).
Finally, we advance the projected velocity over ``dt`` with the explicit predictor of
[`NavierStokesPredictorRHS`](@ref) and `stepper`. Note the order: each step projects the velocity it receives and then
predicts, so the velocity it returns is the provisional field ``\\mathbf{u}^*`` of the next step, which is not
projected, and the pressure is the one of the projection at the start of the step. This order follows the pipeline
of Peddinti et al. (2024), where the mask is applied right before the projection. We are aware that the order
predict, mask, project, which returns the projected velocity, is the more natural one; since it applies the same
sequence of operations, shifted by a projection of the initial velocity, it would solve the same pressure systems, and
it may replace the current order in a future version. The splitting is first-order
accurate in ``dt`` whatever `stepper` is, and the projected velocity is only approximately divergence-free (see
[`INSExperiment`](@ref), including its known limitations). `experiment` is not modified; storing the results is left
to the caller.

# Arguments

  - `experiment::INSExperiment{N}`: the experiment. Its last pressure, if any, is the initial guess of the solve.
  - `cache::ChorinCache`: the operators of the experiment.
  - `u_mpss::NTuple{N,MPS}`: the velocity at time `t`.
  - `solver::AbstractLinearSolver`: the solver of the pressure system (`ALS`, `ResidualALS`, `GlobalCG`, …).
  - `stepper::AbstractExplicitStepper`: the time integrator of the predictor.
  - `t::Real`: the time at the start of the step.

# Returns

  - `(u_next, p, linsolve_error)`: the velocity at `t + dt` (before its projection), the pressure, and the residual
    norm of the pressure solve.

# Examples

```julia
cache = ChorinCache(experiment)
stepper = RK4(; time_evol_kwargs=TimeEvolArgs(0.0, 1.0, 1000), cutoff=1e-10, maxdim=64)
u = last(experiment.u_fields)
for s in 1:10
    u, p, residual = chorin_step(experiment, cache, u, solver, stepper, (s - 1) * stepper.time_evol_kwargs.dt)
end
```
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
    # TODO(v0.2): reconsider the order of the step. Predict -> mask -> project, with chorin_evolve! masking and
    # projecting the initial velocity once, applies the same operations (step k becomes P(M(old step k))), solves the
    # same pressure systems, and returns the projected velocity instead of the provisional u*

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
    chorin_evolve!(experiment::INSExperiment{N}, solver::AbstractLinearSolver, stepper::AbstractExplicitStepper,
                   save_every::Int=1, logger=nothing, log_filename::String="chorin_log") where N

Run the time evolution of `experiment` with the Chorin projection, storing the velocity and the pressure in its fields.

We check the CFL condition ([`check_cfl`](@ref)), build the [`ChorinCache`](@ref) once, and advance the last stored
velocity `stepper.time_evol_kwargs.nsteps` times with [`chorin_step`](@ref). The fields keep the initial velocity and
the velocity and pressure of the steps `save_every`, `2 save_every`, … and of the last step, which is always stored;
the last stored state is also the input of the next step. As explained in `chorin_step`, the stored velocities are
the provisional fields of the next step. If `logger` is given, the fields and the residual norms of the stored pressure
solves (`linsolve_errorlog`) are written to `log_filename` with [`log_data`](@ref).

# Arguments

  - `experiment::INSExperiment{N}`: the experiment, with its initial velocity stored and its pressure empty.
  - `solver::AbstractLinearSolver`: the solver of the pressure system (`ALS`, `ResidualALS`, `GlobalCG`, …), with its
    truncation and stopping criteria in its `sweep_kwargs` and `updater_kwargs`.
  - `stepper::AbstractExplicitStepper`: the integrator of the predictor (`ExplicitEuler`, `RK4`); its `TimeEvolArgs`
    set the time step and the number of steps.
  - `save_every::Int=1`: the interval, in steps, between stored states.
  - `logger=nothing`: a [`SimpleLogger`](@ref), or `nothing` to write nothing.
  - `log_filename::String="chorin_log"`: the name of the JLD2 file in the directory of `logger`.

# Throws

  - `ArgumentError`: if the velocity has no stored state, if the pressure has one, or if `save_every < 1`.

# Known limitations

The Chorin projection of v0.1 has two known limitations, which are recorded as `@test_broken` in the test suite and
will be fixed in a later version:

  - **Mean removal.** When the pressure has no `DirichletBC` side, the Poisson system is singular and
    [`poisson_rhs`](@ref) makes its right-hand side consistent by subtracting a constant. That constant is
    ``\\sum_j b_j\\, \\prod_i h_i / \\prod_i L_i = \\sum_j b_j / \\prod_i (n_i - 1)`` instead of the node mean
    ``\\sum_j b_j / \\prod_i n_i``, so the mean of the right-hand side becomes ``1 - \\prod_i n_i/(n_i - 1)`` times its
    original value instead of zero, and the regularized system is not exactly consistent. This is inactive whenever the
    pressure has a Dirichlet side, as in the channel of Sec. 5.2 of the paper (Dirichlet outlet pressure).
  - **Wall rows.** The projection ``\\mathbf{u} = \\mathbf{u}^* - dt\\, \\nabla p`` (`∇p_correction`) is applied to
    every row, including the ``q/2`` rows next to the Dirichlet velocity faces that the time steppers otherwise keep
    fixed ([`BoundaryNull`](@ref)). The tangential pressure gradient therefore changes the boundary velocity a little at
    every step, which in the setup of Sec. 5.2 acts on the inlet rows.

See also [`INSExperiment`](@ref).
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

    save_every >= 1 || throw(ArgumentError("save_every must be at least 1, got $save_every"))
    # A new slot is pushed at the steps 1, 1 + save_every, …, which freezes the previous one (the initial state, then
    # the states of the steps save_every, 2 save_every, …). Between checkpoints the last slot is overwritten, so it
    # always holds the latest step, which is the input of the next one, and the last step is always stored
    save_checkpoints = 1:save_every:time_evol_kwargs.nsteps

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
            linsolve_errorlog[end] = linsolve_error
        end
    end

    if !isnothing(logger)
        log_data(logger, log_filename, "u_fields", experiment.u_fields)
        log_data(logger, log_filename, "p_field", experiment.p_field)
        log_data(logger, log_filename, "linsolve_errorlog", linsolve_errorlog)
    end
end

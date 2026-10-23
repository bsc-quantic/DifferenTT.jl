"""
    AbstractLinearSolver

Supertype of every solver of the linear system ``A \\mathbf{x} = \\mathbf{b}`` in the TT format, where `A` is an MPO
and ``\\mathbf{b}`` an MPS. Each concrete solver stores its own configuration and is run by [`linearsolve`](@ref).

We distinguish two classes of solvers, as in Sec. 2.1 of Torrente et al., *Factorized approximate inverse
preconditioners in the tensor train format*. The DMRG-type solvers ([`AbstractDMRGSolver`](@ref)) project the system
onto one or two sites at a time and solve each local problem with a classical iterative method, while the global
solvers ([`AbstractGlobalSolver`](@ref)) run a Krylov method on the whole tensor train and truncate its iterates.
"""
abstract type AbstractLinearSolver end

"""
    AbstractDMRGSolver <: AbstractLinearSolver

Supertype of the solvers that proceed in an alternating fashion along the tensor train: [`ALS`](@ref),
[`ResidualALS`](@ref), [`AMEn`](@ref) and [`ExpandedALS`](@ref).

At each step, we use the current solution guess, orthogonalized so that its orthogonality center lies on the sites being
optimized, to project the global system onto a local problem
``A_{\\text{loc}} \\mathbf{x}_{\\text{loc}} = \\mathbf{b}_{\\text{loc}}`` on one or two sites (Sec. 2.1, Fig. 2 of the
paper). The local problem is solved by an [`AbstractUpdater`](@ref), and the TT format is then recovered by an SVD that
also sets the new bond dimension. Since the local solve is exact within the current subspace, its convergence is not
affected by truncation errors, which is what makes these solvers robust for CG. Every concrete subtype has the fields
`sweep_kwargs::SweepingArgs`, `updater_kwargs::LocalUpdaterArgs` and `updater::AbstractUpdater`.
"""
abstract type AbstractDMRGSolver <: AbstractLinearSolver end

"""
    AbstractGlobalSolver <: AbstractLinearSolver

Supertype of the solvers that run a Krylov method on the whole tensor train: [`GlobalCG`](@ref) and
[`GlobalGMRES`](@ref).

Their iterates are sums and operator products of tensor trains, whose bond dimensions are truncated after every
operation in order to keep the cost bounded. These truncations perturb the Krylov recurrences: for CG in particular,
they can destroy the conjugacy of the search directions, so convergence is not guaranteed when the truncation is
severe (Sec. 2.1 of the paper). Every concrete subtype has the field `updater_kwargs::GlobalUpdaterArgs`.
"""
abstract type AbstractGlobalSolver <: AbstractLinearSolver end

"""
    linearsolve(solver::AbstractLinearSolver, A::MPO, b::MPS, x0::Union{MPS,Nothing}; kwargs...) -> MPS
    linearsolve(solver::AbstractLinearSolver, A::MPO, b::MPS, M⁻¹::AbstractPreconditioner, x0::Union{MPS,Nothing};
                kwargs...) -> MPS

Solve ``A \\mathbf{x} = \\mathbf{b}`` with `solver`, optionally preconditioned by `M⁻¹`, starting from `x0`.

The operator `A` must share its site indices with `b`, with prime level 0 for the input and 1 for the output. If `x0`
is `nothing`, we start from the zero MPS of bond dimension 1 ([`zeros_mps`](@ref)) on the sites of `b`. The second
method applies the preconditioner `M⁻¹`, a [`Preconditioner`](@ref) (an approximate inverse) or a
[`FactorizedPreconditioner`](@ref) (the factor ``G_L`` of ``M^{-1} = G_L^\\top G_L``). The DMRG-type solvers apply it
to the local problems, in the projected form of Sec. 4 of the paper, while the global solvers apply it to the whole
residual.

The keywords given at call time override the solver's stored configuration for this call only, which lets one solver
object be reused with different settings:

  - DMRG-type solvers ([`AbstractDMRGSolver`](@ref)): any field of [`SweepingArgs`](@ref) (`nsweeps`, `maxdim`,
    `cutoff`, `nsite`, `outputlevel`, `observer!`, `sweep_observer!`, `checkdone`), and the other keywords of
    `ITensorMPS.alternating_update` (e.g. `mindim`), except its time-evolution and disk options (`time_step`,
    `write_when_maxdim_exceeds`), which ITensorMPS does not support for linear systems; `noise` is accepted but ignored
    by [`ResidualALS`](@ref) and [`AMEn`](@ref). An observer without an operator
    or right-hand side, such as `ResidualObserver(rtol=1e-6)`, is bound to `A` and `b` of this call (see
    [`bind_observer`](@ref)). The preconditioned method also accepts `preconditioner_sites`, a collection of site
    indices in `1:length(b)`: only the local problems whose first site is in it are preconditioned, and the others are
    solved with `A` alone.
  - Global solvers ([`AbstractGlobalSolver`](@ref)): any field of [`GlobalUpdaterArgs`](@ref) (e.g. `rtol`,
    `maxiter`, `maxdim`). Other keywords are ignored.

# Returns

  - `MPS`: the approximate solution ``\\mathbf{x}``.

# Throws

  - `ArgumentError`: if `preconditioner_sites` has an entry outside `1:length(b)`, or a [`ResidualObserver`](@ref)
    has no operator and right-hand side when it is checked.

# Examples

```julia
A, sites = …                                  # SPD MPO, e.g. -1.0 * symmetrized_laplacian(field, grid)
b = random_mps(sites; linkdims=2)
solver = ResidualALS(;
    sweep_kwargs=SweepingArgs(; nsweeps=10, maxdim=50, (observer!)=ResidualObserver(; rtol=1e-6)),
    updater_kwargs=LocalUpdaterArgs(; ishermitian=true, isposdef=true, rtol=1e-8),
)
x = linearsolve(solver, A, b, nothing)
x = linearsolve(solver, A, b, fai_preconditioner(A; μ=1e7, solver=ResidualALS()), nothing; nsweeps=4)
```

See also [`ALS`](@ref), [`ResidualALS`](@ref), [`AMEn`](@ref), [`ExpandedALS`](@ref), [`GlobalCG`](@ref),
[`GlobalGMRES`](@ref).
"""
function linearsolve end

"""
    LocalUpdaterArgs(; ishermitian=false, isposdef=false, rtol=1e-8, krylovdim=30, maxiter=100, verbosity=0,
                     numiter_store=nothing, cond_store=nothing, pc_cond_store=nothing, spectrum_store=nothing,
                     spd_tol=nothing)

Configuration of the local solver of a DMRG-type solver ([`AbstractDMRGSolver`](@ref)), stored in its
`updater_kwargs` field and read by its [`AbstractUpdater`](@ref) at every local problem.

The first six fields set the Krylov method of [`KrylovUpdater`](@ref), which passes them to `KrylovKit.linsolve`.
KrylovKit selects CG when `ishermitian && isposdef`, and restarted GMRES otherwise, so an SPD system must set both
flags to be solved by CG. The other fields attach optional diagnostics. Each one has a single role and costs nothing
when it is `nothing`, since the dense local spectrum, of cost ``\\mathcal{O}(n_{\\text{loc}}^3)`` for a local block
of dimension ``n_{\\text{loc}}``, is computed only when `cond_store`, `pc_cond_store`, `spectrum_store` or `spd_tol`
asks for it. A local block that is not SPD raises no warning: it is detected only by `spd_tol`, which stops the
solve, or recorded by `spectrum_store`.

# Fields

  - `ishermitian::Bool=false`, `isposdef::Bool=false`: properties of the local operator, used to select CG.
  - `rtol::Float64=1e-8`: relative tolerance of the local solve. KrylovKit stops at
    ``\\|\\mathbf{r}_{\\text{loc}}\\| \\le \\max(10^{-12}, \\texttt{rtol}\\, \\|\\mathbf{b}_{\\text{loc}}\\|)``.
  - `krylovdim::Int=30`: Krylov dimension of GMRES before a restart.
  - `maxiter::Int=100`: maximum number of GMRES restarts. KrylovKit gives CG `krylovdim * maxiter` iterations (3000 by
    default).
  - `verbosity::Int=0`: KrylovKit verbosity; with `verbosity > 0` we also log the local condition numbers we compute.
  - `numiter_store=nothing`: a [`KrylovSweepStats`](@ref) that records the iteration count of every local solve.
  - `cond_store=nothing`: a [`KrylovCondStats`](@ref) that records ``\\kappa(A_{\\text{loc}})`` for every local
    problem.
  - `pc_cond_store=nothing`: a [`KrylovCondStats`](@ref) that records ``\\kappa(M^{-1}_{\\text{loc}} A_{\\text{loc}})``
    for every local problem of a preconditioned solve, and ``\\kappa(A_{\\text{loc}})`` for the local problems that
    `preconditioner_sites` leaves unpreconditioned, so that it stays aligned with `cond_store`.
  - `spectrum_store=nothing`: a [`KrylovSpectrumStats`](@ref) that records the sign-aware spectra of the local
    operator and preconditioner.
  - `spd_tol::Union{Nothing,Float64}=nothing`: if set, a local ``A_{\\text{loc}}`` or
    ``M^{-1}_{\\text{loc}} A_{\\text{loc}}`` with an eigenvalue whose real part is below
    ``-\\texttt{spd\\_tol} \\max|\\lambda|`` throws a [`NotSPDError`](@ref). It requires CG.

The condition numbers are ``\\kappa = \\max|\\lambda| / \\min|\\lambda|`` of the dense local operator, which is the
spectral condition number that bounds the convergence of CG and PCG. For GMRES it is only a heuristic indicator,
because it is a lower bound on ``\\kappa_2`` for non-normal operators.

[`QRUpdater`](@ref) reads only `verbosity`, `numiter_store` and `cond_store`.

# Throws

  - `ArgumentError`: if `spd_tol` is set without `ishermitian=true` and `isposdef=true`.
"""
Base.@kwdef struct LocalUpdaterArgs
    ishermitian::Bool = false
    isposdef::Bool = false
    rtol::Float64 = 1e-8
    krylovdim::Int = 30
    maxiter::Int = 100
    verbosity::Int = 0
    numiter_store::Any = nothing
    cond_store::Any = nothing
    pc_cond_store::Any = nothing
    spectrum_store::Any = nothing
    spd_tol::Union{Nothing,Float64} = nothing

    function LocalUpdaterArgs(
        ishermitian,
        isposdef,
        rtol,
        krylovdim,
        maxiter,
        verbosity,
        numiter_store,
        cond_store,
        pc_cond_store,
        spectrum_store,
        spd_tol,
    )
        if !isnothing(spd_tol) && (!ishermitian || !isposdef)
            throw(
                ArgumentError(
                    "spd_tol checks the local blocks of a CG solve, which requires ishermitian=true and isposdef=true."
                ),
            )
        end
        new(
            ishermitian,
            isposdef,
            rtol,
            krylovdim,
            maxiter,
            verbosity,
            numiter_store,
            cond_store,
            pc_cond_store,
            spectrum_store,
            spd_tol,
        )
    end
end

"""
    GlobalUpdaterArgs(; ishermitian=false, isposdef=false, krylovdim=30, maxiter=100, rtol=1e-8, verbosity=0,
                      maxdim=typemax(Int), cutoff=1e-8, orth_maxdim=typemax(Int), orth_cutoff=1e-12,
                      numiter_store=nothing, converged_store=nothing)

Configuration of a global solver ([`GlobalCG`](@ref) or [`GlobalGMRES`](@ref)), stored in its `updater_kwargs` field.
[`linearsolve`](@ref) copies it into a [`TTCG`](@ref) or [`TTGMRES`](@ref) algorithm, and any field can be overridden
at call time.

The two pairs of truncation parameters control the bond dimension of the iterates. Since every operator application
and every sum of tensor trains increases the bond dimension, we truncate after each of them, and these truncations
are the price paid for running a Krylov method on the whole tensor train (see [`AbstractGlobalSolver`](@ref)).

# Fields

  - `ishermitian::Bool=false`, `isposdef::Bool=false`: properties of `A`. [`GlobalCG`](@ref) requires both to be
    `true`; [`GlobalGMRES`](@ref) does not read them.
  - `krylovdim::Int=30`: Krylov dimension of TT-GMRES before a restart (not read by TT-CG).
  - `maxiter::Int=100`: maximum number of TT-CG iterations, or of TT-GMRES restart cycles.
  - `rtol::Float64=1e-8`: relative tolerance; the solve stops at
    ``\\|\\mathbf{b} - A\\mathbf{x}\\| \\le \\texttt{rtol}\\, \\|\\mathbf{b}\\|``, where the residual is the truncated
    one of the recurrence.
  - `verbosity::Int=0`: KrylovKit verbosity levels (1: warn on no convergence, 2: start and end, 3: every iteration).
  - `maxdim::Int=typemax(Int)`, `cutoff::Float64=1e-8`: truncation of the operator applications and of the updates of
    ``\\mathbf{x}`` and of the residual.
  - `orth_maxdim::Int=typemax(Int)`, `orth_cutoff::Float64=1e-12`: truncation of the new search direction (TT-CG) or of
    the Gram–Schmidt orthogonalization (TT-GMRES).
  - `numiter_store=nothing`: a `Ref{Int}` (or anything that supports `store[] = value`) that receives the number of
    iterations.
  - `converged_store=nothing`: a `Ref{Bool}` that receives whether the solve converged.

A solve that stops without converging always logs a warning, whether or not `converged_store` is set.
"""
Base.@kwdef struct GlobalUpdaterArgs
    ishermitian::Bool = false
    isposdef::Bool = false
    krylovdim::Int = 30
    maxiter::Int = 100
    rtol::Float64 = 1e-8
    verbosity::Int = 0
    maxdim::Int = typemax(Int)
    cutoff::Float64 = 1e-8
    orth_maxdim::Int = typemax(Int)
    orth_cutoff::Float64 = 1e-12
    numiter_store::Any = nothing
    converged_store::Any = nothing
end

"""
    default_differenTT_checkdone(; observer, sweep_observer, kwargs...) -> Bool

Default `checkdone` of [`SweepingArgs`](@ref). Return whether the sweep observer or the bond observer asks to stop,
by calling `ITensorMPS.checkdone!` on each of them that is not ITensorMPS's default (empty) observer. The keywords
`state`, `sweep`, `outputlevel`, … are forwarded to `checkdone!`, as `ITensorMPS.alternating_update` passes them at
the end of every sweep.
"""
function default_differenTT_checkdone(; kwargs...)
    obs = get(kwargs, :observer, get(kwargs, :observer!, ITensorMPS.default_observer()))
    sweep_obs = get(kwargs, :sweep_observer, get(kwargs, :sweep_observer!, ITensorMPS.default_sweep_observer()))

    clean_kwargs = filter(kv -> kv.first ∉ (:observer, :observer!, :sweep_observer, :sweep_observer!), kwargs)

    obs_done =
        (typeof(obs) === typeof(ITensorMPS.default_observer())) ? false : ITensorMPS.checkdone!(obs; clean_kwargs...)
    sweep_obs_done = if (typeof(sweep_obs) === typeof(ITensorMPS.default_sweep_observer()))
        false
    else
        ITensorMPS.checkdone!(sweep_obs; clean_kwargs...)
    end

    return obs_done || sweep_obs_done
end

"""
    SweepingArgs(; nsweeps=5, maxdim=30, cutoff=1e-10, nsite=2, outputlevel=0,
                 observer! = ITensorMPS.default_observer(), sweep_observer! = ITensorMPS.default_sweep_observer(),
                 checkdone = default_differenTT_checkdone)

Configuration of the sweeps of a DMRG-type solver ([`AbstractDMRGSolver`](@ref)), stored in its `sweep_kwargs` field.
[`linearsolve`](@ref) passes the fields to the sweeping loop as keywords, and any of them can be overridden at call
time.

A sweep is one forward half-sweep, from the first site to the last, followed by one reverse half-sweep. The solve
stops after `nsweeps` sweeps, or earlier when `checkdone`, evaluated at the end of every sweep, returns `true`. Note
that the residual is therefore checked only once per sweep, so a solve that narrowly misses its tolerance pays for a
whole extra sweep (Sec. 5.1 of Torrente et al.).

# Fields

  - `nsweeps::Int=5`: maximum number of sweeps.
  - `maxdim::Int=30`: maximum bond dimension of the solution after each SVD.
  - `cutoff::Float64=1e-10`: SVD cutoff. [`ALS`](@ref) follows the ITensors convention and discards the smallest
    singular values whose relative weight ``\\sum_{k>r} \\sigma_k^2 / \\sum_k \\sigma_k^2`` is at most `cutoff`.
    [`ResidualALS`](@ref) and [`AMEn`](@ref) choose the rank by the local residual instead, and use the cutoff, in the
    same convention, only as a floor on the rank.
  - `nsite::Int=2`: number of sites of the local problem. [`ALS`](@ref) accepts 1 or 2, but only the 2-site variant
    (MALS) can change the bond dimension. [`ResidualALS`](@ref) truncates by the residual only for `nsite = 2`;
    [`AMEn`](@ref) ignores it; [`ExpandedALS`](@ref) requires 1.
  - `outputlevel::Int=0`: printed output of the sweeping loop (1: one line per sweep).
  - `observer!`: observer updated after every local step, and checked by `checkdone`. A
    [`ResidualObserver`](@ref) stops the solve once the global relative residual is below its `rtol`.
  - `sweep_observer!`: observer updated after every sweep, and checked by `checkdone`.
  - `checkdone`: function of the keywords `state`, `sweep`, `outputlevel`, `observer` and `sweep_observer` that
    returns `true` to stop; the default is [`default_differenTT_checkdone`](@ref).
"""
Base.@kwdef struct SweepingArgs
    nsweeps::Int = 5
    maxdim::Int = 30
    cutoff::Float64 = 1e-10
    nsite::Int = 2
    outputlevel::Int = 0
    observer!::Any = ITensorMPS.default_observer()
    sweep_observer!::Any = ITensorMPS.default_sweep_observer()
    checkdone::Any = default_differenTT_checkdone
end

"""
    NamedTuple(args::LocalUpdaterArgs)

Return the fields of `args` as a `NamedTuple`, to be splatted as keywords into the local updater.
"""
function Base.NamedTuple(args::LocalUpdaterArgs)
    return (;
        ishermitian=args.ishermitian,
        isposdef=args.isposdef,
        rtol=args.rtol,
        krylovdim=args.krylovdim,
        maxiter=args.maxiter,
        verbosity=args.verbosity,
        numiter_store=args.numiter_store,
        cond_store=args.cond_store,
        pc_cond_store=args.pc_cond_store,
        spectrum_store=args.spectrum_store,
        spd_tol=args.spd_tol,
    )
end

"""
    NamedTuple(args::GlobalUpdaterArgs)

Return the fields of `args` as a `NamedTuple`, to be merged with the call-time keywords of a global solve.
"""
function Base.NamedTuple(args::GlobalUpdaterArgs)
    return (;
        ishermitian=args.ishermitian,
        isposdef=args.isposdef,
        rtol=args.rtol,
        krylovdim=args.krylovdim,
        maxiter=args.maxiter,
        verbosity=args.verbosity,
        maxdim=args.maxdim,
        cutoff=args.cutoff,
        orth_maxdim=args.orth_maxdim,
        orth_cutoff=args.orth_cutoff,
        numiter_store=args.numiter_store,
        converged_store=args.converged_store,
    )
end

"""
    NamedTuple(args::SweepingArgs)

Return the fields of `args` as a `NamedTuple`, to be merged with the call-time keywords of a DMRG-type solve.
"""
function Base.NamedTuple(args::SweepingArgs)
    return (;
        nsweeps=args.nsweeps,
        maxdim=args.maxdim,
        cutoff=args.cutoff,
        nsite=args.nsite,
        outputlevel=args.outputlevel,
        (observer!)=args.observer!,
        (sweep_observer!)=args.sweep_observer!,
        checkdone=args.checkdone,
    )
end

"""
    default_linearsolve_init(const_term::MPS) -> MPS

Return the initial guess used when `x0 = nothing`: the zero MPS of bond dimension 1 on the sites of `const_term`, with
its array backend.
"""
function default_linearsolve_init(const_term::MPS)
    return zeros_mps(siteinds(const_term), backend_type(const_term))
end

"""
    AbstractUpdater

Supertype of the local solvers of the DMRG-type solvers: [`KrylovUpdater`](@ref) (the default) and
[`QRUpdater`](@ref).

An updater is a callable `(updater)(problem, init; internal_kwargs, coefficients, kwargs...)`. It receives the local
problem `problem` (an `ITensorMPS.ReducedLinearProblem`, or its preconditioned counterpart), the current local tensor
`init` as initial guess, and the fields of [`LocalUpdaterArgs`](@ref) as `kwargs`. It solves
``(a_0 + a_1 A_{\\text{loc}}) \\mathbf{x}_{\\text{loc}} = \\mathbf{b}_{\\text{loc}}``, with `coefficients = (a₀, a₁)`
(always `(false, true)` in [`linearsolve`](@ref)), and returns the new local tensor and a `NamedTuple` with an `info`
entry.
"""
abstract type AbstractUpdater end

"""
    ResidualObserver(; A=nothing, b=nothing, rtol=1e-8) <: ITensorMPS.AbstractObserver
    ResidualObserver(A, b, rtol)

Observer that stops the sweeps of a DMRG-type solver once the global relative residual
``\\|A\\mathbf{x} - \\mathbf{b}\\| / \\|\\mathbf{b}\\|`` falls below `rtol`. Pass it as `observer!` in
[`SweepingArgs`](@ref).

The local problems are solved to their own tolerance, but nothing in the sweeps controls the global residual, so without
an observer [`ALS`](@ref) and [`ResidualALS`](@ref) always run `nsweeps` sweeps. We evaluate the residual at the end of
every sweep, from the expansion

```math
\\|A\\mathbf{x} - \\mathbf{b}\\|^2 = \\langle A\\mathbf{x}, A\\mathbf{x}\\rangle
    - 2\\,\\mathrm{Re} \\langle \\mathbf{b}, A\\mathbf{x}\\rangle + \\langle \\mathbf{b}, \\mathbf{b}\\rangle,
```

which costs three TT inner products and never forms ``A\\mathbf{x}``. Note that this expansion suffers from
cancellation: its rounding error is of order
``\\sqrt{\\epsilon}\\, \\|\\mathbf{b}\\| \\approx 1.5 \\cdot 10^{-8} \\|\\mathbf{b}\\|``, so a tolerance below about
``10^{-8}`` is checked against rounding noise ([#76](https://github.com/bsc-quantic/DifferenTT.jl/issues/76)).

`A` and `b` can be left as `nothing`: the DMRG-type [`linearsolve`](@ref) methods then bind them to the operator and
right-hand side of each call (always the unpreconditioned ones, never `M⁻¹`), so that one observer stored in
[`SweepingArgs`](@ref) can be reused across solves with different right-hand sides (see [`bind_observer`](@ref)). The
global solvers do not use observers; they stop on their own `rtol`.

# Fields

  - `A`: the operator, an MPO, or `nothing`.
  - `b::Union{MPS,Nothing}`: the right-hand side, or `nothing`.
  - `rtol::Float64=1e-8`: tolerance on the relative residual.

# Throws

  - `ArgumentError`: from `ITensorMPS.checkdone!`, if `A` or `b` is still `nothing` when the residual is checked.
"""
Base.@kwdef struct ResidualObserver <: AbstractObserver
    A::Any = nothing
    b::Union{MPS,Nothing} = nothing
    rtol::Float64 = 1e-8
end

"""
    bind_observer(observer, A, b)

Return `observer` with its unset operator and right-hand side taken from `A` and `b`. A [`ResidualObserver`](@ref)
keeps the `A` and `b` it already has; any other observer is returned unchanged. The input observer is never mutated.
"""
bind_observer(observer, A, b) = observer

function bind_observer(o::ResidualObserver, A, b)
    return ResidualObserver(isnothing(o.A) ? A : o.A, isnothing(o.b) ? b : o.b, o.rtol)
end

"""
    linearsolve_sweep_kwargs(solver, A, b, kwargs) -> NamedTuple

Merge the solver's [`SweepingArgs`](@ref) with the call-time `kwargs`, which take precedence, and bind both observers
to the system `A x = b` of this call with [`bind_observer`](@ref).
"""
function linearsolve_sweep_kwargs(solver, A, b, kwargs)
    sweep_kwargs = merge(NamedTuple(solver.sweep_kwargs), values(kwargs))
    return merge(
        sweep_kwargs,
        (;
            (observer!)=bind_observer(sweep_kwargs.observer!, A, b),
            (sweep_observer!)=bind_observer(sweep_kwargs.sweep_observer!, A, b),
        ),
    )
end

"""
    _checked_preconditioner_sites(preconditioner_sites, n) -> Union{Nothing,Vector{Int}}

Return `preconditioner_sites` as a `Vector{Int}` (or `nothing`, which preconditions every local problem), after
checking that every entry is an integer in `1:n`. Throws an `ArgumentError` otherwise: an index outside `1:n` would
match no local problem, so the solve would silently run unpreconditioned while the condition-number stores still
filled with ``\\kappa(A)``.
"""
function _checked_preconditioner_sites(preconditioner_sites::Nothing, n::Int)::Nothing
    return nothing
end

function _checked_preconditioner_sites(
    preconditioner_sites::Union{AbstractVector,AbstractRange,Tuple}, n::Int
)::Vector{Int}
    sites = Vector{Int}()
    for site in preconditioner_sites
        site isa Integer || throw(ArgumentError("preconditioner_sites entries must be integers, got $(repr(site))"))
        1 <= site <= n || throw(ArgumentError("preconditioner_sites index $site is outside 1:$n"))
        push!(sites, Int(site))
    end
    return sites
end

"""
    precond_linearsolve_kwargs(solver, A, b, kwargs) -> (updater_kwargs, sweep_kwargs)

Split the call-time `kwargs` of a preconditioned DMRG-type [`linearsolve`](@ref) into the keywords of the local updater
and those of the sweeps.

The optional `preconditioner_sites` (default `nothing`: precondition every local problem), checked by
[`_checked_preconditioner_sites`](@ref), goes to the local updater, which applies `M⁻¹` only to the local problems
whose first site is in it. The remaining `kwargs` are merged with the solver's [`SweepingArgs`](@ref) by
[`linearsolve_sweep_kwargs`](@ref).
"""
function precond_linearsolve_kwargs(
    solver::AbstractDMRGSolver, A::MPO, b::MPS, kwargs::Base.Pairs
)::Tuple{NamedTuple,NamedTuple}
    preconditioner_sites = _checked_preconditioner_sites(get(kwargs, :preconditioner_sites, nothing), length(b))
    updater_kwargs = (;
        coefficients=(false, true), preconditioner_sites=preconditioner_sites, NamedTuple(solver.updater_kwargs)...
    )
    kw = Dict(kwargs...)
    delete!(kw, :preconditioner_sites)
    sweep_kwargs = linearsolve_sweep_kwargs(solver, A, b, pairs((; kw...)))
    return updater_kwargs, sweep_kwargs
end

"""
    ITensorMPS.checkdone!(o::ResidualObserver; state, kwargs...) -> Bool

Return whether the relative residual ``\\|A\\mathbf{x} - \\mathbf{b}\\| / \\|\\mathbf{b}\\|`` of `state` is below
`o.rtol` (the absolute residual if ``\\mathbf{b} = 0``). Throws an `ArgumentError` if `o` has no operator or right-hand
side.
"""
function ITensorMPS.checkdone!(o::ResidualObserver; state, kwargs...)
    (isnothing(o.A) || isnothing(o.b)) && throw(
        ArgumentError(
            "ResidualObserver has no operator or right-hand side bound; pass A and b explicitly or use it through a sweep-based `linearsolve`.",
        ),
    )

    term1 = inner(o.A, state, o.A, state)
    term2 = -2.0 * real(inner(o.b', o.A, state))
    term3 = inner(o.b, o.b)

    res = sqrt(max(0.0, term1 + term2 + term3))
    rel_res = term3 > 0.0 ? res / sqrt(term3) : res

    if rel_res < o.rtol
        @debug "Residual early truncation condition met: ||Ax-b||/||b|| = $(rel_res) < $(o.rtol)"
        return true
    end
    return false
end

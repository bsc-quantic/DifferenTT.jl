"""
    abstract type AbstractExplicitStepper <: AbstractTimeStepper

Supertype of the explicit time steppers ([`ExplicitEuler`](@ref), [`RK4`](@ref)), which advance a state with
[`timestep`](@ref) from evaluations of an [`AbstractRHS`](@ref), and whose stability [`check_cfl`](@ref) checks.
"""
abstract type AbstractExplicitStepper <: AbstractTimeStepper end

"""
    abstract type AbstractRHS

Supertype of the right-hand sides ``\\mathbf{f}(\\mathbf{u}, t)`` of the explicit time steppers, for systems
``\\partial_t \\mathbf{u} = \\mathbf{f}(\\mathbf{u}, t)``.

A subtype is a callable struct that holds whatever it needs to evaluate the right-hand side (operators, boundary
terms, the experiment) and implements the interface

    (rhs::MyRHS)(u::Tuple, t::Real, stage::Int) -> Tuple

which takes the state `u`, a tuple of MPS with one entry per field component, the time `t`, and the stage `stage` of
the stepper (1 to 4 for [`RK4`](@ref), 1 for [`ExplicitEuler`](@ref); it may be ignored), and returns a tuple of MPS of
the same length as `u`. The library's right-hand sides are [`NavierStokesPredictorRHS`](@ref) and
[`VorticityTransportRHS`](@ref). The example below wraps a user-defined `rhs_custom_func` and keeps the Dirichlet
boundary rows fixed, as `NavierStokesPredictorRHS` does.

# Examples

```julia
struct SampleRHS{E,D,B} <: AbstractRHS
    experiment::E
    u_diff_ops::D
    u_boundary_null::B
end

function (rhs::SampleRHS)(u, t, stage)
    cutoff = rhs.experiment.cutoff
    maxdim = rhs.experiment.maxdim
    rhs_terms = rhs_custom_func(u, rhs.u_diff_ops; cutoff=cutoff, maxdim=maxdim)
    # Keep the boundary rows fixed
    return ntuple(length(u)) do i
        deleter = rhs.u_boundary_null[i].u_row_deleter
        isnothing(deleter) ? rhs_terms[i] : apply(deleter, rhs_terms[i]; cutoff=cutoff, maxdim=maxdim)
    end
end
```
"""
abstract type AbstractRHS end

"""
    ExplicitEuler(; time_evol_kwargs::TimeEvolArgs, kwargs...)

Forward (explicit) Euler scheme, ``\\mathbf{u}_{n+1} = \\mathbf{u}_n + dt\\, \\mathbf{f}(\\mathbf{u}_n, t_n)``,
first-order accurate in ``dt``. The keyword arguments other than `time_evol_kwargs` (e.g. `cutoff`, `maxdim`) are stored
as the TT truncation of the sums of [`timestep`](@ref). Its stability limits are those of [`check_cfl`](@ref).

# Fields

  - `time_evol_kwargs::TimeEvolArgs`: the time interval and step.
  - `tt_kwargs::K`: the TT truncation keywords, as a `NamedTuple`.

# Examples

```julia
stepper = ExplicitEuler(; time_evol_kwargs=TimeEvolArgs(0.0, 1.0, 100), cutoff=1e-10, maxdim=64)
```
"""
struct ExplicitEuler{K} <: AbstractExplicitStepper
    time_evol_kwargs::TimeEvolArgs
    tt_kwargs::K
end
ExplicitEuler(; time_evol_kwargs::TimeEvolArgs, kwargs...) = ExplicitEuler(time_evol_kwargs, NamedTuple(kwargs))

"""
    RK4(; time_evol_kwargs::TimeEvolArgs, kwargs...)

Classical fourth-order Runge–Kutta scheme,

```math
\\mathbf{u}_{n+1} = \\mathbf{u}_n + \\frac{dt}{6} (\\mathbf{k}_1 + 2\\mathbf{k}_2 + 2\\mathbf{k}_3 + \\mathbf{k}_4),
```

where ``\\mathbf{k}_1 = \\mathbf{f}(\\mathbf{u}_n, t_n)``,
``\\mathbf{k}_2 = \\mathbf{f}(\\mathbf{u}_n + \\frac{dt}{2}\\mathbf{k}_1, t_n + \\frac{dt}{2})``,
``\\mathbf{k}_3 = \\mathbf{f}(\\mathbf{u}_n + \\frac{dt}{2}\\mathbf{k}_2, t_n + \\frac{dt}{2})`` and
``\\mathbf{k}_4 = \\mathbf{f}(\\mathbf{u}_n + dt\\,\\mathbf{k}_3, t_n + dt)``. The keyword arguments other than
`time_evol_kwargs` (e.g. `cutoff`, `maxdim`) are stored as the TT truncation of every sum of [`timestep`](@ref). Its
stability limits are those of [`check_cfl`](@ref).

# Fields

  - `time_evol_kwargs::TimeEvolArgs`: the time interval and step.
  - `tt_kwargs::K`: the TT truncation keywords, as a `NamedTuple`.
"""
struct RK4{K} <: AbstractExplicitStepper
    time_evol_kwargs::TimeEvolArgs
    tt_kwargs::K
end
RK4(; time_evol_kwargs::TimeEvolArgs, kwargs...) = RK4(time_evol_kwargs, NamedTuple(kwargs))

"""
    timestep(stepper::ExplicitEuler, u::Tuple, rhs_func::AbstractRHS, t::Real=0.0) -> Tuple
    timestep(stepper::RK4, u::Tuple, rhs_func::AbstractRHS, t::Real=0.0) -> Tuple

Advance the state `u` from time `t` by one step ``dt`` = `stepper.time_evol_kwargs.dt` with the explicit scheme
`stepper` ([`ExplicitEuler`](@ref) or [`RK4`](@ref)), and return the new state.

Every stage evaluates `rhs_func(u, t, stage)`, and every linear combination of MPS is a sum truncated with the TT
keywords of `stepper` (`stepper.tt_kwargs`), so the accuracy of a step is bounded by that truncation as well as by
the order of the scheme. Note that the boundary conditions enter only through `rhs_func` (e.g. the rows that
[`BoundaryNullScalar`](@ref) keeps fixed).

# Arguments

  - `stepper::AbstractExplicitStepper`: the scheme, with the step and the TT truncation.
  - `u::Tuple`: the state, a tuple of MPS with one entry per field component.
  - `rhs_func::AbstractRHS`: the right-hand side, called as `rhs_func(u, t, stage)` (see [`AbstractRHS`](@ref)).
  - `t::Real=0.0`: the time of `u`.

# Examples

```julia
stepper = RK4(; time_evol_kwargs=TimeEvolArgs(0.0, 0.1, 10), cutoff=1e-12, maxdim=32)
u_next = timestep(stepper, (u,), rhs, 0.0)
```
"""
function timestep(stepper::ExplicitEuler, u::Tuple, rhs_func::AbstractRHS, t::Real=0.0)
    dt = stepper.time_evol_kwargs.dt
    k1 = rhs_func(u, t, 1)

    u_next = ntuple(length(u)) do i
        +(u[i], dt * k1[i]; stepper.tt_kwargs...)
    end

    return u_next
end

function timestep(stepper::RK4, u::Tuple, rhs_func::AbstractRHS, t::Real=0.0)
    dt = stepper.time_evol_kwargs.dt
    k1 = rhs_func(u, t, 1)

    u2 = ntuple(i -> +(u[i], 0.5 * dt * k1[i]; stepper.tt_kwargs...), length(u))
    k2 = rhs_func(u2, t + 0.5 * dt, 2)

    u3 = ntuple(i -> +(u[i], 0.5 * dt * k2[i]; stepper.tt_kwargs...), length(u))
    k3 = rhs_func(u3, t + 0.5 * dt, 3)

    u4 = ntuple(i -> +(u[i], dt * k3[i]; stepper.tt_kwargs...), length(u))
    k4 = rhs_func(u4, t + dt, 4)

    u_next = ntuple(length(u)) do i
        +(u[i], (1/6) * dt * k1[i], (1/3) * dt * k2[i], (1/3) * dt * k3[i], (1/6) * dt * k4[i]; stepper.tt_kwargs...)
    end

    return u_next
end

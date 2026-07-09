abstract type AbstractExplicitStepper <: AbstractTimeStepper end
abstract type AbstractRHS end

#################################################################

# For illustrative purposes
# struct SampleRHS{E, D, B} <: AbstractRHS
#     experiment::E
#     u_diff_ops::D
#     u_boundary_null::B
# end

# # Make the struct "callable" by defining its behavior:

# function (rhs::SampleRHS)(u, t, stage)
#     # stage is used for rk4 internal stage tracking, ut there is no need to use it internally in RHS
#     cutoff = rhs.experiment.cutoff
#     maxdim = rhs.experiment.maxdim
    
#     rhs_terms = rhs_custom_func(u, rhs.u_diff_ops; cutoff=cutoff, maxdim=maxdim)

#     rhs_corrected = ntuple(i -> begin
#         deleter = rhs.u_boundary_null.deleters[i]
#         return isnothing(deleter) ? rhs_terms[i] : apply(deleter, rhs_terms[i]; cutoff=cutoff, maxdim=maxdim)
#     end, length(u))

#     return rhs_corrected
# end

#################################################################


"""
    ExplicitEuler{K} <: AbstractExplicitStepper

Explicit Euler time integration scheme.
Stores TT-related truncation keyword arguments (e.g., `cutoff`, `maxdim`) in `tt_kwargs`.
"""
struct ExplicitEuler{K} <: AbstractExplicitStepper
    tt_kwargs::K
end
ExplicitEuler(; kwargs...) = ExplicitEuler(NamedTuple(kwargs))

"""
    RK4{K} <: AbstractExplicitStepper

Explicit Runge-Kutta 4th order time integration scheme.
Stores TT-related truncation keyword arguments (e.g., `cutoff`, `maxdim`) in `tt_kwargs`.
"""
struct RK4{K} <: AbstractExplicitStepper
    tt_kwargs::K
end
RK4(; kwargs...) = RK4(NamedTuple(kwargs))

"""
    timestep(scheme::AbstractExplicitStepper, u, dt, rhs_func, t=0.0)

Advances the state `u` by a time step `dt` using the specified explicit `scheme`.

# Arguments
- `scheme`: The explicit time integration scheme (e.g., `ExplicitEuler`, `RK4`).
- `u`: The current state, typically a Tuple of `MPS` for vector fields.
- `dt`: The time step size.
- `rhs_func`: A function `rhs_func(u, t, stage)` that computes the time derivative `du/dt = F(u, t)`. 
  The `stage` argument allows `rhs_func` to know which internal RK stage is being evaluated,
  useful for debugging or intermediate outputs.
- `t`: The current time.
"""
function timestep(scheme::ExplicitEuler, u::Tuple, dt::Real, rhs_func::AbstractRHS, t::Real=0.0)
    k1 = rhs_func(u, t, 1)
    
    u_next = ntuple(length(u)) do i
        +(u[i], dt * k1[i]; scheme.tt_kwargs...)
    end
    
    return u_next
end

function timestep(scheme::RK4, u::Tuple, dt::Real, rhs_func::AbstractRHS, t::Real=0.0)
    k1 = rhs_func(u, t, 1)
    
    u2 = ntuple(i -> +(u[i], 0.5 * dt * k1[i]; scheme.tt_kwargs...), length(u))
    k2 = rhs_func(u2, t + 0.5 * dt, 2)
    
    u3 = ntuple(i -> +(u[i], 0.5 * dt * k2[i]; scheme.tt_kwargs...), length(u))
    k3 = rhs_func(u3, t + 0.5 * dt, 3)
    
    u4 = ntuple(i -> +(u[i], dt * k3[i]; scheme.tt_kwargs...), length(u))
    k4 = rhs_func(u4, t + dt, 4)
    
    u_next = ntuple(length(u)) do i
        +(u[i], (1/6) * dt * k1[i], (1/3) * dt * k2[i], (1/3) * dt * k3[i], (1/6) * dt * k4[i]; scheme.tt_kwargs...)
    end
    
    return u_next
end

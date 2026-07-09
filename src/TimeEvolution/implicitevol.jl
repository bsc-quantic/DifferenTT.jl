#################

# WARNING THIS IS WIP

#################

# Implicit time evolution schemes will go here.
# For example, abstract type AbstractImplicitStepper <: AbstractTimeStepper end

abstract type AbstractImplicitStepper <: AbstractTimeStepper end
abstract type AbstractImplicitSystem end

"""
    ImplicitEuler{K} <: AbstractImplicitStepper

Implicit Euler time integration scheme.
Stores linear solver keyword arguments (e.g., `cutoff`, `maxdim`, `nsweeps`) in `linsolve_kwargs`.
"""
struct ImplicitEuler{K} <: AbstractImplicitStepper
    linsolve_kwargs::K
end
ImplicitEuler(; kwargs...) = ImplicitEuler(NamedTuple(kwargs))

"""
    CrankNicolson{K} <: AbstractImplicitStepper

Crank-Nicolson (semi-implicit) time integration scheme.
Stores linear solver keyword arguments (e.g., `cutoff`, `maxdim`, `nsweeps`) in `linsolve_kwargs`.
"""
struct CrankNicolson{K} <: AbstractImplicitStepper
    linsolve_kwargs::K
end
CrankNicolson(; kwargs...) = CrankNicolson(NamedTuple(kwargs))

#################################################################
# Unified Implicit API
#
# To use these implicit steppers, the user should define a struct 
# subtyping `AbstractImplicitSystem` which provides two methods:
# 1. `build_lhs_operator(sys, scheme, dt, t_next)` -> Returns the MPO (or tuple of MPOs) for the LHS matrix.
# 2. `build_rhs_vector(sys, scheme, u, dt, t)` -> Returns the MPS (or tuple of MPSs) for the RHS vector.
# 3. `solve_linear_system(sys, lhs_op, rhs_vec, u0, kwargs)` -> Solves the system (e.g. using DMRG `linsolve`).

"""
    timestep(scheme::AbstractImplicitStepper, u, dt, sys::AbstractImplicitSystem, t=0.0)

Advances the state `u` by a time step `dt` using the specified implicit `scheme`.
"""
function timestep(scheme::ImplicitEuler, u::Tuple, dt::Real, sys::AbstractImplicitSystem, t::Real=0.0)
    t_next = t + dt
    
    # For Implicit Euler: (I - dt * A) u_{n+1} = u_n + dt * b
    # 1. Build the LHS operator MPO: (I - dt * A)
    lhs_op = build_lhs_operator(sys, scheme, dt, t_next)
    
    # 2. Build the RHS vector MPS: u_n + dt * b
    rhs_vec = build_rhs_vector(sys, scheme, u, dt, t)
    
    # 3. Solve the linear system
    u_next = solve_linear_system(sys, lhs_op, rhs_vec, u, scheme.linsolve_kwargs)
    
    return u_next
end

function timestep(scheme::CrankNicolson, u::Tuple, dt::Real, sys::AbstractImplicitSystem, t::Real=0.0)
    t_next = t + dt
    
    # For Crank-Nicolson: (I - dt/2 * A) u_{n+1} = (I + dt/2 * A) u_n + dt/2 * (b_n + b_{n+1})
    # 1. Build the LHS operator MPO: (I - dt/2 * A)
    lhs_op = build_lhs_operator(sys, scheme, dt, t_next)
    
    # 2. Build the RHS vector MPS: (I + dt/2 * A) u_n + dt/2 * (b_n + b_{n+1})
    rhs_vec = build_rhs_vector(sys, scheme, u, dt, t)
    
    # 3. Solve the linear system
    u_next = solve_linear_system(sys, lhs_op, rhs_vec, u, scheme.linsolve_kwargs)
    
    return u_next
end

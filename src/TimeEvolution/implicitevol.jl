#################

# WARNING THIS IS WIP. Things to work on:
# - Develop into a more structure way how the implicit stepper interacts with the 
# <: AbstractLinearSolver structs
#
# - We will have coupling between the different field components within the implicit time
# time stepping scheme -> Need to add TT fusing via additional sites.
#

#################

# Implicit time evolution schemes will go here.
# For example, abstract type AbstractImplicitStepper <: AbstractTimeStepper end

abstract type AbstractImplicitStepper <: AbstractTimeStepper end
abstract type AbstractImplicitSystem end

"""
    ImplicitEuler{K} <: AbstractImplicitStepper

Implicit Euler time integration scheme.

# Fields

  - `time_evol_kwargs::TimeEvolArgs`: Time interval and step size configuration.
  - `linsolve_kwargs::K`: Linear solver keyword arguments (e.g., `cutoff`, `maxdim`, `nsweeps`).
"""
struct ImplicitEuler{K} <: AbstractImplicitStepper
    time_evol_kwargs::TimeEvolArgs
    linsolve_kwargs::K
end
ImplicitEuler(; time_evol_kwargs::TimeEvolArgs, kwargs...) = ImplicitEuler(time_evol_kwargs, NamedTuple(kwargs))

"""
    CrankNicolson{K} <: AbstractImplicitStepper

Crank-Nicolson (semi-implicit) time integration scheme.

# Fields

  - `time_evol_kwargs::TimeEvolArgs`: Time interval and step size configuration.
  - `linsolve_kwargs::K`: Linear solver keyword arguments (e.g., `cutoff`, `maxdim`, `nsweeps`).
"""
struct CrankNicolson{K} <: AbstractImplicitStepper
    time_evol_kwargs::TimeEvolArgs
    linsolve_kwargs::K
end
CrankNicolson(; time_evol_kwargs::TimeEvolArgs, kwargs...) = CrankNicolson(time_evol_kwargs, NamedTuple(kwargs))

#################################################################
# Unified Implicit API
#
# To use these implicit steppers, the user should define a struct 
# subtyping `AbstractImplicitSystem` which provides two methods:
# 1. `build_lhs_operator(sys, stepper, dt, t_next)` -> Returns the MPO (or tuple of MPOs) for the LHS matrix.
# 2. `build_rhs_vector(sys, stepper, u, dt, t)` -> Returns the MPS (or tuple of MPSs) for the RHS vector.
# 3. `solve_linear_system(sys, lhs_op, rhs_vec, u0, kwargs)` -> Solves the system (e.g. using DMRG `linsolve`).

"""
    timestep(stepper::AbstractImplicitStepper, u, sys::AbstractImplicitSystem, t=0.0)

Advances the state `u` by one time step `stepper.time_evol_kwargs.dt` using the specified implicit `stepper`.
"""
function timestep(stepper::ImplicitEuler, u::Tuple, sys::AbstractImplicitSystem, t::Real=0.0)
    dt = stepper.time_evol_kwargs.dt
    t_next = t + dt

    # For Implicit Euler: (I - dt * A) u_{n+1} = u_n + dt * b
    # 1. Build the LHS operator MPO: (I - dt * A)
    lhs_op = build_lhs_operator(sys, stepper, dt, t_next)

    # 2. Build the RHS vector MPS: u_n + dt * b
    rhs_vec = build_rhs_vector(sys, stepper, u, dt, t)

    # 3. Solve the linear system
    u_next = solve_linear_system(sys, lhs_op, rhs_vec, u, stepper.linsolve_kwargs)

    return u_next
end

function timestep(stepper::CrankNicolson, u::Tuple, sys::AbstractImplicitSystem, t::Real=0.0)
    dt = stepper.time_evol_kwargs.dt
    t_next = t + dt

    # For Crank-Nicolson: (I - dt/2 * A) u_{n+1} = (I + dt/2 * A) u_n + dt/2 * (b_n + b_{n+1})
    # 1. Build the LHS operator MPO: (I - dt/2 * A)
    lhs_op = build_lhs_operator(sys, stepper, dt, t_next)

    # 2. Build the RHS vector MPS: (I + dt/2 * A) u_n + dt/2 * (b_n + b_{n+1})
    rhs_vec = build_rhs_vector(sys, stepper, u, dt, t)

    # 3. Solve the linear system
    u_next = solve_linear_system(sys, lhs_op, rhs_vec, u, stepper.linsolve_kwargs)

    return u_next
end

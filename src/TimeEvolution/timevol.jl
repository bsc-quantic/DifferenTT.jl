"""
    struct TimeEvol

Configuration for the time interval and step size. The integrator is a separate
stepper (`ExplicitEuler`, `RK4`, ...) passed to `timestep` / `chorin_evolve!`.

# Fields
- `ti::Float64`: Initial time.
- `tf::Float64`: Final time.
- `dt::Float64`: Time step.
- `nsteps::Int`: Number of steps.
"""
struct TimeEvol
    ti::Float64
    tf::Float64
    dt::Float64
    nsteps::Int
end

function TimeEvol(ti::Float64, tf::Float64, nsteps::Int)
    dt = (tf - ti) / nsteps
    return TimeEvol(ti, tf, dt, nsteps)
end

function TimeEvol(ti::Float64, tf::Float64, dt::Float64)
    nsteps = round(Int, (tf - ti) / dt)
    return TimeEvol(ti, tf, dt, nsteps)
end

abstract type AbstractTimeStepper end

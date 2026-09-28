"""
    struct TimeEvolArgs

Configuration for the time interval and step size. It is stored in an `AbstractTimeStepper`
(`ExplicitEuler`, `RK4`, ...) as `time_evol_kwargs`, from which `timestep` reads the step size.

# Fields
- `ti::Float64`: Initial time.
- `tf::Float64`: Final time.
- `dt::Float64`: Time step.
- `nsteps::Int`: Number of steps.
"""
struct TimeEvolArgs
    ti::Float64
    tf::Float64
    dt::Float64
    nsteps::Int
end

function TimeEvolArgs(ti::Float64, tf::Float64, nsteps::Int)
    dt = (tf - ti) / nsteps
    return TimeEvolArgs(ti, tf, dt, nsteps)
end

function TimeEvolArgs(ti::Float64, tf::Float64, dt::Float64)
    nsteps = round(Int, (tf - ti) / dt)
    return TimeEvolArgs(ti, tf, dt, nsteps)
end

"""
    AbstractTimeStepper

Supertype of all time integration schemes. Every concrete stepper stores its time interval
configuration in a `time_evol_kwargs::TimeEvolArgs` field.
"""
abstract type AbstractTimeStepper end

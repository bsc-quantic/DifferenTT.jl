"""
    struct TimeEvol

Configuration for time evolution parameters.

# Fields
- `ti::Float64`: Initial time.
- `tf::Float64`: Final time.
- `dt::Float64`: Time step.
- `N::Int`: Number of steps.
- `scheme::Symbol`: Time integration scheme (currently only :euler).
"""
struct TimeEvol
    ti::Float64
    tf::Float64
    dt::Float64
    nsteps::Int
    scheme::Symbol
end

function check_available_schemes(scheme::Symbol)
    avail_schemes = (:euler, :rk4)
    scheme ∉ avail_schemes && throw(ArgumentError("Only $(avail_schemes) schemes are implemented."))
end

function TimeEvol(ti::Float64, tf::Float64, nsteps::Int, scheme::Symbol=:euler)
    dt = (tf - ti) / nsteps
    check_available_schemes(scheme)
    return TimeEvol(ti, tf, dt, nsteps, scheme)
end

function TimeEvol(ti::Float64, tf::Float64, dt::Float64, scheme::Symbol=:euler)
    nsteps = round(Int, (tf - ti) / dt)
    check_available_schemes(scheme)
    return TimeEvol(ti, tf, dt, nsteps, scheme)
end

abstract type AbstractTimeStepper end
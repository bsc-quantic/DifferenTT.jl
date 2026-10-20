"""
    TimeEvolArgs(ti::Float64, tf::Float64, dt::Float64, nsteps::Int)
    TimeEvolArgs(ti::Float64, tf::Float64, nsteps::Int)
    TimeEvolArgs(ti::Float64, tf::Float64, dt::Float64)

Time interval and step of an evolution, stored in an [`AbstractTimeStepper`](@ref) as `time_evol_kwargs`, from which
[`timestep`](@ref) reads the step and the evolution loops ([`chorin_evolve!`](@ref), [`vorticity_evolve!`](@ref)) read
the number of steps and the initial time.

Given the number of steps, the step is ``dt = (t_f - t_i)/n_\\text{steps}``. Given the step, the number of steps is
rounded, ``n_\\text{steps} = \\mathrm{round}((t_f - t_i)/dt)``, while `dt` and `tf` are kept as given, so
``t_i + n_\\text{steps}\\, dt`` can differ from ``t_f`` by up to ``dt/2``; the evolution loops run `nsteps` steps of
size `dt`. The times must be `Float64` and `nsteps` an `Int`, which is what tells the last two methods apart.

# Fields

  - `ti::Float64`: the initial time.
  - `tf::Float64`: the final time.
  - `dt::Float64`: the time step.
  - `nsteps::Int`: the number of steps.

# Examples

```jldoctest
julia> TimeEvolArgs(0.0, 1.0, 4)
TimeEvolArgs(0.0, 1.0, 0.25, 4)

julia> TimeEvolArgs(0.0, 1.0, 0.3)
TimeEvolArgs(0.0, 1.0, 0.3, 3)
```
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
    abstract type AbstractTimeStepper

Supertype of the time integration schemes. Every concrete stepper stores its time interval and step in a field
`time_evol_kwargs::TimeEvolArgs` (see [`TimeEvolArgs`](@ref)).
"""
abstract type AbstractTimeStepper end

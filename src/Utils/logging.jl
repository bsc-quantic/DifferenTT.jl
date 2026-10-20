"""
    timeit_to::TimerOutput

Global timer of DifferenTT. The solvers and the Navier–Stokes steppers time their main phases with
`TimerOutputs.@timeit_debug timeit_to "…"`, which costs nothing unless the debug timings are enabled with
[`enable_debug_timings!`](@ref). Print the collected timings with `show(timeit_to)`, and clear them with
`TimerOutputs.reset_timer!(timeit_to)`.
"""
const timeit_to = TimerOutput()

"""
    enable_debug_timings!() -> nothing

Turn on the `@timeit_debug` sections of DifferenTT, which then record their timings in [`timeit_to`](@ref).

This calls `TimerOutputs.enable_debug_timings(DifferenTT)`, which redefines a function inside the package, so the
timed methods are recompiled at their next call. Call it once, before the run to be profiled, and turn the timings off
again with [`disable_debug_timings!`](@ref).
"""
function enable_debug_timings!()
    # invokelatest: TimerOutputs reads the current switch before redefining it, and a caller running in an older world
    # (e.g. inside one testset or script block) would otherwise see a stale value
    Base.invokelatest(TimerOutputs.enable_debug_timings, @__MODULE__)
    return nothing
end

"""
    disable_debug_timings!() -> nothing

Turn off the `@timeit_debug` sections of DifferenTT that [`enable_debug_timings!`](@ref) turned on.
"""
function disable_debug_timings!()
    Base.invokelatest(TimerOutputs.disable_debug_timings, @__MODULE__)
    return nothing
end

"""
    abstract type Logger

Supertype of the data loggers of the evolutions ([`chorin_evolve!`](@ref), [`vorticity_evolve!`](@ref)), which store
their fields and residuals with [`log_data`](@ref). Not to be confused with `Logging.AbstractLogger`: these loggers
write data files, not log messages.
"""
abstract type Logger end

"""
    SimpleLogger(log_dir::String)

[`Logger`](@ref) that writes JLD2 files into the directory `log_dir`; build it with [`init_logger`](@ref), which
creates the directory.

Note that the standard library `Logging` also exports a `SimpleLogger` (a log-message handler). After
`using DifferenTT, Logging` the unqualified name is ambiguous and throws an `UndefVarError`, so it must be written
`DifferenTT.SimpleLogger` or `Logging.SimpleLogger`. The functions of DifferenTT only need the object that
`init_logger` returns, so most scripts never write the name.

# Fields

  - `log_dir::String`: the directory of the files.
"""
struct SimpleLogger <: Logger
    log_dir::String
end

"""
    init_logger(log_dir::String) -> SimpleLogger

Return a [`SimpleLogger`](@ref) that writes into the directory `log_dir`, creating it (and its parents) if it does not
exist.

# Examples

```julia
logger = init_logger(joinpath("results", "cavity"))
vorticity_evolve!(experiment, solver, stepper, 10, logger, "run1")  # writes results/cavity/run1.jld2
```
"""
function init_logger(log_dir::String)
    if !isdir(log_dir)
        mkpath(log_dir)
    end
    return SimpleLogger(log_dir)
end

"""
    log_data(logger::SimpleLogger, filename::String, key::String, value)
    log_data(logger::SimpleLogger, filename::String, data::Dict)
    log_data(logger::Nothing, filename::String, key::String, value)
    log_data(logger::Nothing, filename::String, data::Dict)

Write `value` under `key`, or every pair of `data`, into the JLD2 file `filename` (with the extension `.jld2` added
if it is missing) in the directory of `logger`. If the file exists, the other keys are kept and an existing key is
overwritten; otherwise the file is created. With `logger = nothing` nothing is written, so that the evolutions can
call it unconditionally.

# Examples

```julia
logger = init_logger(mktempdir())
log_data(logger, "run", "residuals", [1e-3, 1e-6])
log_data(logger, "run", Dict("Re" => 100.0, "dt" => 1e-3))   # same file, run.jld2
```
"""
function log_data(logger::SimpleLogger, filename::String, key::String, value)
    path = joinpath(logger.log_dir, filename)

    # Ensure filename ends with .jld2
    if !endswith(path, ".jld2")
        path *= ".jld2"
    end

    if isfile(path)
        jldopen(path, "r+") do file
            haskey(file, key) && delete!(file, key) # JLD2 does not overwrite a dataset in place
            file[key] = value
        end
    else
        jldsave(path; Dict(Symbol(key) => value)...)
    end
end

function log_data(logger::SimpleLogger, filename::String, data::Dict)
    path = joinpath(logger.log_dir, filename)

    if !endswith(path, ".jld2")
        path *= ".jld2"
    end

    if isfile(path)
        jldopen(path, "r+") do file
            for (k, v) in data
                haskey(file, k) && delete!(file, k)
                file[k] = v
            end
        end
    else
        symbol_data = Dict(Symbol(k) => v for (k, v) in data)
        jldsave(path; symbol_data...)
    end
end

function log_data(logger::Nothing, filename::String, key::String, value) end

function log_data(logger::Nothing, filename::String, data::Dict) end

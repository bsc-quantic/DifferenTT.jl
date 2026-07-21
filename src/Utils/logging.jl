const timeit_to = TimerOutput()

abstract type Logger end

struct SimpleLogger <: Logger
    log_dir::String
end

"""
    init_logger(log_dir::String)

Initialize a logger that stores data in the specified directory `log_dir`.
If the directory does not exist, it will be created.
"""
function init_logger(log_dir::String)
    if !isdir(log_dir)
        mkpath(log_dir)
    end
    return SimpleLogger(log_dir)
end

"""
    log_data(logger::SimpleLogger, filename::String, key::String, value)

Log a value with a specific key to a JLD2 file named `filename` in the logger's directory.
If the file exists, the value is appended/updated.
"""
function log_data(logger::SimpleLogger, filename::String, key::String, value)
    path = joinpath(logger.log_dir, filename)

    # Ensure filename ends with .jld2
    if !endswith(path, ".jld2")
        path *= ".jld2"
    end

    if isfile(path)
        jldopen(path, "r+") do file
            file[key] = value
        end
    else
        jldsave(path; Dict(Symbol(key) => value)...)
    end
end

"""
    log_data(logger::SimpleLogger, filename::String, data::Dict)

Log a dictionary of data to a JLD2 file named `filename`.
"""
function log_data(logger::SimpleLogger, filename::String, data::Dict)
    path = joinpath(logger.log_dir, filename)

    if !endswith(path, ".jld2")
        path *= ".jld2"
    end

    if isfile(path)
        jldopen(path, "r+") do file
            for (k, v) in data
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

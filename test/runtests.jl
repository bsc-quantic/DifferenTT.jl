# Run the whole suite with `julia --project -e 'using Pkg; Pkg.test()'`, or with
# `julia --project=test test/runtests.jl`. Select one module with the environment variable `GROUP`, e.g.
# `GROUP=LinearSolvers julia --project=test test/runtests.jl`; `GROUP=All` (the default) runs every module and Aqua.
using SafeTestsets
using Test

const GROUP = get(ENV, "GROUP", "All")
const MODULES = ["TensorTrains", "PdeProblem", "Physics", "TimeEvolution", "LinearSolvers", "Utils"]

GROUP in ("All", "Aqua", MODULES...) ||
    throw(ArgumentError("unknown GROUP = \"$GROUP\"; use \"All\", \"Aqua\" or one of $(join(MODULES, ", "))"))

@testset "DifferenTT.jl" begin
    for m in MODULES
        if GROUP == "All" || GROUP == m
            @testset "$m" begin
                include(joinpath(@__DIR__, m, "runtests.jl"))
            end
        end
    end
    if GROUP == "All" || GROUP == "Aqua"
        @safetestset "Aqua" begin
            include("aqua.jl")
        end
    end
end

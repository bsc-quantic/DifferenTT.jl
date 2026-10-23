# Run the whole suite with `julia --project -e 'using Pkg; Pkg.test()'`, or with
# `julia --project=test test/runtests.jl`. Select one module with the environment variable `GROUP`, e.g.
# `GROUP=LinearSolvers julia --project=test test/runtests.jl`; `GROUP=All` (the default) runs every module and Aqua.
using ITensors: ITensors
using SafeTestsets
using Test

# The tests densify operators of 7 and 8 sites, whose contractions reach order 16, and ITensors warns from order 14 on
# with a stack trace each. Raise the threshold just past them (the warning fires at order >= threshold), not further: a
# larger contraction in a test is a mistake (it may run out of memory) and should still warn. The setting is global, so it covers the safe
# testsets; files that change it restore it.
ITensors.set_warn_order(17)

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

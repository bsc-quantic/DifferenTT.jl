using DifferenTT
using Test

@testset "DifferenTT.jl" begin
    @testset "PdeProblem" begin
        include("PdeProblem/test_boundaryconditions.jl")
        include("PdeProblem/test_grid.jl")
        include("PdeProblem/test_scalarfields.jl")
        include("PdeProblem/test_vectorfields.jl")
    end
    @testset "TensorTrains" begin
        include("TensorTrains/test_architecture.jl")
        include("TensorTrains/test_tci.jl")
        include("TensorTrains/test_elementwise.jl")
    end
    @testset "TimeEvolution" begin
        include("TimeEvolution/test_explicitevol.jl")
    end
    @testset "LinearSolvers" begin
        include("LinearSolvers/test_linearsolvers.jl")
        include("LinearSolvers/test_expandedals.jl")
    end
end

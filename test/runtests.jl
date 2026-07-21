using DifferenTT
using Test

@testset "DifferenTT.jl" begin
    @testset "PdeProblem" begin
        include("PdeProblem/test_boundaryconditions.jl")
        include("PdeProblem/test_grid.jl")
        include("PdeProblem/test_scalarfields.jl")
        include("PdeProblem/test_vectorfields.jl")
        include("PdeProblem/test_operators.jl")
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
        include("LinearSolvers/test_preconditioners.jl")
        include("LinearSolvers/test_preconditionedproblem.jl")
    end
    @testset "Physics" begin
        include("Physics/test_chorin.jl")
    end
    @testset "Utils" begin
        include("Utils/test_logging.jl")
        include("Utils/test_visualization.jl")
    end
end

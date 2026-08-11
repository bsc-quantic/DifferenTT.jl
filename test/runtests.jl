using DifferenTT
using Test

@testset "DifferenTT.jl" begin
    @testset "PdeProblem" begin
        include("PdeProblem/test_boundaryconditions.jl")
        include("PdeProblem/test_grid.jl")
        include("PdeProblem/test_scalarfields.jl")
        include("PdeProblem/test_vectorfields.jl")
        include("PdeProblem/test_diffoperators.jl")
        include("PdeProblem/test_bcoperators.jl")
        include("PdeProblem/test_linearsyscorrections.jl")
    end
    @testset "TensorTrains" begin
        include("TensorTrains/test_architecture.jl")
        include("TensorTrains/test_tci.jl")
        include("TensorTrains/test_elementwise.jl")
        include("TensorTrains/test_constructors.jl")
        include("TensorTrains/test_operations.jl")
        include("TensorTrains/test_qttconstructors.jl")
        include("TensorTrains/test_qttoperations.jl")
        include("TensorTrains/test_utils.jl")
    end
    @testset "TimeEvolution" begin
        include("TimeEvolution/test_explicitevol.jl")
        include("TimeEvolution/test_timevol.jl")
    end
    @testset "LinearSolvers" begin
        include("LinearSolvers/test_linearsolvers.jl")
        include("LinearSolvers/test_expandedals.jl")
        include("LinearSolvers/test_preconditioners.jl")
        include("LinearSolvers/test_preconditionedproblem.jl")
        include("LinearSolvers/test_als.jl")
        include("LinearSolvers/test_amen.jl")
        include("LinearSolvers/test_exactupdaters.jl")
        include("LinearSolvers/test_projectedproblem.jl")
        include("LinearSolvers/test_residualals.jl")
        
        include("LinearSolvers/KrylovSolvers/test_cg.jl")
        include("LinearSolvers/KrylovSolvers/test_gmres.jl")
        include("LinearSolvers/KrylovSolvers/test_krylovstats.jl")
        include("LinearSolvers/KrylovSolvers/test_krylovupdaters.jl")
        include("LinearSolvers/KrylovSolvers/test_preconditionedlinsolve.jl")
        include("LinearSolvers/KrylovSolvers/test_ttcg.jl")
        include("LinearSolvers/KrylovSolvers/test_ttgmres.jl")
        include("LinearSolvers/KrylovSolvers/test_ttkrylov.jl")
    end
    @testset "Physics" begin
        include("Physics/test_chorin.jl")
        include("Physics/test_equationterms.jl")
        include("Physics/test_fdstencils.jl")
        include("Physics/test_insexperiment.jl")
        include("Physics/test_end2end.jl")
    end
    @testset "Utils" begin
        include("Utils/test_logging.jl")
        include("Utils/test_visualization.jl")
    end
end

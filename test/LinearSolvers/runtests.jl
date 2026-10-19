using SafeTestsets

@safetestset "linearsolvers" begin
    include("test_linearsolvers.jl")
end
@safetestset "solver matrix" begin
    include("test_solver_matrix.jl")
end
@safetestset "expandedals" begin
    include("test_expandedals.jl")
end
@safetestset "preconditioners" begin
    include("test_preconditioners.jl")
end
@safetestset "preconditionedproblem" begin
    include("test_preconditionedproblem.jl")
end
# TODO: expand minimalist test in a future PR
@safetestset "als" begin
    include("test_als.jl")
end
# TODO: expand minimalist test in a future PR
@safetestset "amen" begin
    include("test_amen.jl")
end
# TODO: expand minimalist test in a future PR
@safetestset "exactupdaters" begin
    include("test_exactupdaters.jl")
end
@safetestset "projectedproblem" begin
    include("test_projectedproblem.jl")
end
@safetestset "residualals" begin
    include("test_residualals.jl")
end
# TODO: expand minimalist test in a future PR
@safetestset "KrylovSolvers/cg" begin
    include("KrylovSolvers/test_cg.jl")
end
# TODO: expand minimalist test in a future PR
@safetestset "KrylovSolvers/gmres" begin
    include("KrylovSolvers/test_gmres.jl")
end
@safetestset "KrylovSolvers/krylovstats" begin
    include("KrylovSolvers/test_krylovstats.jl")
end
@safetestset "KrylovSolvers/krylovupdaters" begin
    include("KrylovSolvers/test_krylovupdaters.jl")
end
# TODO: expand minimalist test in a future PR
@safetestset "KrylovSolvers/preconditionedlinsolve" begin
    include("KrylovSolvers/test_preconditionedlinsolve.jl")
end
# TODO: expand minimalist test in a future PR
@safetestset "KrylovSolvers/ttcg" begin
    include("KrylovSolvers/test_ttcg.jl")
end
# TODO: expand minimalist test in a future PR
@safetestset "KrylovSolvers/ttgmres" begin
    include("KrylovSolvers/test_ttgmres.jl")
end
@safetestset "KrylovSolvers/ttkrylov" begin
    include("KrylovSolvers/test_ttkrylov.jl")
end
@safetestset "rng" begin
    include("test_rng.jl")
end

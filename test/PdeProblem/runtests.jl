using SafeTestsets

@safetestset "boundaryconditions" begin
    include("test_boundaryconditions.jl")
end
@safetestset "grid" begin
    include("test_grid.jl")
end
@safetestset "scalarfields" begin
    include("test_scalarfields.jl")
end
@safetestset "vectorfields" begin
    include("test_vectorfields.jl")
end
@safetestset "diffoperators" begin
    include("test_diffoperators.jl")
end
@safetestset "bcoperators" begin
    include("test_bcoperators.jl")
end
@safetestset "linearsyscorrections" begin
    include("test_linearsyscorrections.jl")
end

using SafeTestsets

@safetestset "logging" begin
    include("test_logging.jl")
end
@safetestset "visualization" begin
    include("test_visualization.jl")
end

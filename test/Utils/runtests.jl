using SafeTestsets

@safetestset "logging" begin
    include("test_logging.jl")
end
@safetestset "plotshint" begin
    include("test_plotshint.jl")
end
@safetestset "visualization" begin
    include("test_visualization.jl")
end

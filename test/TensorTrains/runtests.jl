using SafeTestsets

@safetestset "architecture" begin
    include("test_architecture.jl")
end
@safetestset "tci" begin
    include("test_tci.jl")
end
@safetestset "elementwise" begin
    include("test_elementwise.jl")
end
@safetestset "constructors" begin
    include("test_constructors.jl")
end
@safetestset "operations" begin
    include("test_operations.jl")
end
@safetestset "qttconstructors" begin
    include("test_qttconstructors.jl")
end
@safetestset "qttoperations" begin
    include("test_qttoperations.jl")
end
@safetestset "utils" begin
    include("test_utils.jl")
end

using SafeTestsets

@safetestset "chorin" begin
    include("test_chorin.jl")
end
@safetestset "vorticity" begin
    include("test_vorticity.jl")
end
@safetestset "equationterms" begin
    include("test_equationterms.jl")
end
@safetestset "fdstencils" begin
    include("test_fdstencils.jl")
end
@safetestset "insexperiment" begin
    include("test_insexperiment.jl")
end
@safetestset "end2end" begin
    include("test_end2end.jl")
end

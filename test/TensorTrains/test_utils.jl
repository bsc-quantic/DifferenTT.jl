using DifferenTT
using Test
using Adapt
using ITensors
using ITensorMPS

@testset "utils.jl" begin
    @testset "backend_type" begin
        cpu_array = rand(2, 2)
        cpu_itensor = ITensor(cpu_array, Index(2), Index(2))
        cpu_mps = random_mps(siteinds(2, 2))

        @test backend_type(cpu_array) == Array
        @test backend_type(cpu_itensor) == Array
        @test backend_type(cpu_mps) == Array

        try using CUDA
            if CUDA.functional()
                gpu_array = CuArray(rand(2, 2))
                @test backend_type(gpu_array) == CuArray
            else
                @warn "CUDA not functional, skipping GPU tests for backend_type"
            end
        catch e 
            @warn "CUDA not loaded, skipping GPU tests for backend_type"
        end
    end
end
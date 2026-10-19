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

        # import, not using: runtests.jl includes every file into Main, and CUDA's exports (e.g. CG) would
        # clash with the KrylovKit names of the later test files
        try
            using CUDA: CUDA
            if CUDA.functional()
                gpu_array = CUDA.CuArray(rand(2, 2))
                @test backend_type(gpu_array) == CUDA.CuArray
            else
                @warn "CUDA not functional, skipping GPU tests for backend_type"
            end
        catch e
            @warn "CUDA not loaded, skipping GPU tests for backend_type"
        end
    end

    @testset "ndim_index" begin
        # Blocked architecture: var 1 has 3 sites, var 2 has 4 sites
        arch = DifferenTT.QTTArchitecture((3, 4))

        # Axis 1: 3 bits → should fill positions mapped to axis 1, rest = 1
        bits_ax1 = [0, 1, 0]
        full_bits = DifferenTT.ndim_index(bits_ax1, 1, arch)
        @test length(full_bits) == 7 # total sites
        @test full_bits[1:3] == [0, 1, 0] # axis 1 sites
        @test all(full_bits[4:7] .== 1)   # axis 2 sites filled with 1

        # Axis 2: 4 bits
        bits_ax2 = [1, 0, 1, 1]
        full_bits2 = DifferenTT.ndim_index(bits_ax2, 2, arch)
        @test all(full_bits2[1:3] .== 1) # axis 1 sites filled with 1
        @test full_bits2[4:7] == [1, 0, 1, 1]

        # Interleaved architecture
        arch_int = DifferenTT.QTTArchitecture((2, 2); interleaved=true)
        # site_map should be [1, 2, 1, 2]
        bits_int = [0, 1]
        full_bits_int = DifferenTT.ndim_index(bits_int, 1, arch_int)
        @test length(full_bits_int) == 4
        @test full_bits_int[1] == 0  # site 1 → axis 1
        @test full_bits_int[2] == 1  # site 2 → axis 2 → filled with 1
        @test full_bits_int[3] == 1  # site 3 → axis 1
        @test full_bits_int[4] == 1  # site 4 → axis 2 → filled with 1

        # TTArchitecture
        tt_arch = DifferenTT.TTArchitecture((10, 20))
        idx_ax1 = [5]
        full_idx = DifferenTT.ndim_index(idx_ax1, 1, tt_arch)
        @test length(full_idx) == 2
        @test full_idx[1] == 5
        @test full_idx[2] == 1
    end

    @testset "Conversions" begin
        # base 2 (binary)
        @test DifferenTT.decimal_to_binary(5, 4) == [0, 1, 0, 1]
        @test DifferenTT.decimal_to_binary(0, 3) == [0, 0, 0]
        @test DifferenTT.decimal_to_binary(7, 3) == [1, 1, 1]

        @test DifferenTT.binary_to_decimal([0, 1, 0, 1]) == 5
        @test DifferenTT.binary_to_decimal([0, 0, 0]) == 0
        @test DifferenTT.binary_to_decimal([1, 1, 1]) == 7

        # base 3
        @test DifferenTT.decimal_to_base(14, 3, 3) == [1, 1, 2] # 1*9 + 1*3 + 2*1 = 14
        @test DifferenTT.base_to_decimal([1, 1, 2], 3) == 14

        # base 10
        @test DifferenTT.decimal_to_base(123, 10, 4) == [0, 1, 2, 3]
        @test DifferenTT.base_to_decimal([0, 1, 2, 3], 10) == 123
    end
end

using DifferenTT
using Test
using Adapt
using ITensors
using ITensorMPS

@testset "utils.jl" begin
    @testset "backend_type" begin
        cpu_array = zeros(2, 2)
        cpu_itensor = ITensor(cpu_array, Index(2), Index(2))
        cpu_mps = zeros_mps(siteinds(2, 2))

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

    @testset "ndim_index: reversed, permuted and errors" begin
        # var 2 sits on sites 3, 5 and 6 and is reversed: its bits are written least significant first
        arch = DifferenTT.QTTArchitecture((2, -3, 1), Dict(1 => 2, 2 => 3, 3 => 1); interleaved=true)
        @test arch.site_map == [3, 1, 2, 1, 2, 2]
        @test DifferenTT.ndim_index([0, 1, 1], 2, arch) == [1, 1, 1, 1, 1, 0]
        @test DifferenTT.ndim_index([0, 1], 1, arch) == [1, 0, 1, 1, 1, 1]
        @test DifferenTT.ndim_index([0], 3, arch) == [0, 1, 1, 1, 1, 1]
        @test_throws DimensionMismatch DifferenTT.ndim_index([0, 1], 2, arch)

        # TT with a permuted var_map: variable 1 lives on site 3
        tt = DifferenTT.TTArchitecture((4, 5, 6), Dict(1 => 3, 2 => 1, 3 => 2))
        @test DifferenTT.ndim_index([4], 1, tt) == [1, 1, 4]
        @test DifferenTT.ndim_index([2], 2, tt) == [2, 1, 1]
        @test_throws DimensionMismatch DifferenTT.ndim_index([1, 2], 1, tt)
    end

    @testset "Conversion round trips" begin
        for base in (2, 3, 5), ndigits in 1:4
            numbers = 0:(base^ndigits-1)
            @test all(
                DifferenTT.base_to_decimal(DifferenTT.decimal_to_base(n, base, ndigits), base) == n for n in numbers
            )
            @test all(length(DifferenTT.decimal_to_base(n, base, ndigits)) == ndigits for n in numbers)
            @test all(all(0 .<= DifferenTT.decimal_to_base(n, base, ndigits) .< base) for n in numbers)
        end
        for nbits in 1:5, n in 0:(2^nbits-1)
            @test DifferenTT.binary_to_decimal(DifferenTT.decimal_to_binary(n, nbits)) == n
        end
        # the digits are the most significant first, as in the site order of a QTT variable
        @test DifferenTT.decimal_to_binary(1, 3) == [0, 0, 1]
        # a number that needs more digits than asked for returns all of them
        @test DifferenTT.decimal_to_base(9, 2, 2) == [1, 0, 0, 1]
    end

    @testset "identity_array" begin
        @test DifferenTT.identity_array(3) == [1.0 0 0; 0 1 0; 0 0 1]
        @test DifferenTT.identity_array(1) isa Matrix{Float64}
    end
end

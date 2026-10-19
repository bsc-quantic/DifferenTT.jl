using DifferenTT
using Test
using ITensors
using ITensorMPS
using LinearAlgebra

include(joinpath(@__DIR__, "..", "testutils.jl"))
using .TestUtils

@testset "qttconstructors.jl" begin
    n = 6
    sites = siteinds(2, n)
    coeffs = Dict(i => 0.0 for i in -4:4)
    coeffs[0] = 2.0
    coeffs[1] = -1.0
    coeffs[-1] = -1.0

    short_coeffs = Dict(0 => 2.0, 1 => -1.0, -1 => -1.0)

    @testset "toeplitz_mpo" begin
        # Periodic
        bc_periodic = BoundaryCond1d(PeriodicBC(), PeriodicBC())

        eigth_order_mpo = DifferenTT.toeplitz_mpo(sites, coeffs, bc_periodic)
        @test eigth_order_mpo isa MPO
        @test length(eigth_order_mpo) == n
        @test linkdim(eigth_order_mpo, 1) == 3
        @test linkdim(eigth_order_mpo, n-1) == 9

        second_order_mpo = DifferenTT.toeplitz_mpo(sites, short_coeffs, bc_periodic)
        @test second_order_mpo isa MPO
        @test length(second_order_mpo) == n
        @test linkdim(second_order_mpo, 1) == 3
        @test linkdim(second_order_mpo, n-1) == 3

        @test_throws ArgumentError DifferenTT.toeplitz_mpo(sites, Dict(0 => 2.0, 1 => -1.0), bc_periodic)

        # Dirichlet
        bc_dirichlet = BoundaryCond1d(DirichletBC(0.0), DirichletBC(0.0))
        mpo_d = DifferenTT.toeplitz_mpo(sites, coeffs, bc_dirichlet)
        @test mpo_d isa MPO
        @test length(mpo_d) == n
    end

    @testset "column_pick_mpo" begin
        col_idx = 2
        mpo = DifferenTT.column_pick_mpo(sites, col_idx)
        @test mpo isa MPO
        @test length(mpo) == n

        # Test vector version
        mpo_sum = DifferenTT.column_pick_mpo(sites, [1, 2])
        @test mpo_sum isa MPO

        # Test MPO entries
        # Since col_idx = 2, reverse(digits(1, base=2, pad=6)) = [0, 0, 0, 0, 0, 1]
        # So sites 1-5 have matrix0 = [1.0 0.0; 1.0 0.0]
        # Site 6 has matrix1 = [0.0 1.0; 0.0 1.0]
        @test all(
            array(mpo[5], sites[5]', sites[5], linkinds(mpo)[4], linkinds(mpo)[5])[:, :, 1, 1] .== [1.0 0.0; 1.0 0.0]
        )
        @test all(array(mpo[6], sites[6]', sites[6], linkinds(mpo)[5])[:, :, 1] .== [0.0 1.0; 0.0 1.0])
    end

    @testset "row_pick_mpo" begin
        row_idx = 2
        mpo = DifferenTT.row_pick_mpo(sites, row_idx)
        @test mpo isa MPO
        @test length(mpo) == n

        # Test vector version
        mpo_sum = DifferenTT.row_pick_mpo(sites, [1, 2])
        @test mpo_sum isa MPO

        # Test MPO entries
        # Transpose of column_pick_mpo
        @test all(
            array(mpo[5], sites[5]', sites[5], linkinds(mpo)[4], linkinds(mpo)[5])[:, :, 1, 1] .== [1.0 1.0; 0.0 0.0]
        )
        @test all(array(mpo[6], sites[6]', sites[6], linkinds(mpo)[5])[:, :, 1] .== [0.0 0.0; 1.0 1.0])
    end

    @testset "corner_pick_mpo" begin
        coefficients = [1.0, 2.0, 3.0, 4.0]
        mpo = DifferenTT.corner_pick_mpo(coefficients, sites)
        @test mpo isa MPO
        @test length(mpo) == n
        @test linkdim(mpo, n - 2) == 1
        @test linkdim(mpo, n - 1) == 2

        # Content check: the default flag is :upper, which must place the corner block in the
        # top-left N_corner x N_corner corner and leave every other entry exactly zero. The
        # corner block itself must equal the upper-triangular Toeplitz matrix of `coefficients`
        # (this is the same block `corner_pick_matrix(coefficients, :upper)` builds for the dense
        # TT path, and the two are required to agree — see bcoperators.jl tests).
        full = DifferenTT.contract_mpo(mpo)
        N = 2^n
        expected_corner = DifferenTT.upper_toeplitz_matrix(coefficients)
        @test full[1:4, 1:4] ≈ expected_corner atol=1e-12
        @test full[1:4, 1:4] ≈ DifferenTT.corner_pick_matrix(coefficients, :upper) atol=1e-12
        @test all(iszero, full[1:4, 5:N])
        @test all(iszero, full[5:N, :])

        mpo_lower = DifferenTT.corner_pick_mpo(coefficients, sites, :lower)
        full_lower = DifferenTT.contract_mpo(mpo_lower)
        expected_corner_lower = DifferenTT.lower_toeplitz_matrix(coefficients)
        @test full_lower[(N-3):N, (N-3):N] ≈ expected_corner_lower atol=1e-12
        @test full_lower[(N-3):N, (N-3):N] ≈ DifferenTT.corner_pick_matrix(coefficients, :lower) atol=1e-12
        @test all(iszero, full_lower[(N-3):N, 1:(N-4)])
        @test all(iszero, full_lower[1:(N-4), :])
    end

    @testset "entry_mpo" begin
        N = 2^n
        for (row, col) in ((1, 1), (2, 5), (N, 1), (N, N - 1), (17, 42))
            mpo = DifferenTT.entry_mpo(sites, row, col)
            @test mpo isa MPO
            @test length(mpo) == n
            @test maxlinkdim(mpo) == 1

            # Single-entry matrix |row⟩⟨col|
            full = DifferenTT.contract_mpo(mpo)
            @test full[row, col] == 1.0
            @test count(!iszero, full) == 1
        end
    end
end

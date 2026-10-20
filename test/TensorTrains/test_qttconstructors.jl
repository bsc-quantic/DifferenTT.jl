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

        # Regression test: two sites with three coefficients threw a DimensionMismatch (one tensor too many), and too
        # few sites threw an ErrorException
        s2 = siteinds(2, 2)
        toeplitz4 = [2.0 -1.0 0.0 0.0; -1.0 2.0 -1.0 0.0; 0.0 -1.0 2.0 -1.0; 0.0 0.0 -1.0 2.0]
        @test contract_mpo(DifferenTT.toeplitz_mpo(s2, short_coeffs, bc_dirichlet)) ≈ toeplitz4
        circulant4 = copy(toeplitz4)
        circulant4[1, 4] = circulant4[4, 1] = -1.0
        @test contract_mpo(DifferenTT.toeplitz_mpo(s2, short_coeffs, bc_periodic)) ≈ circulant4
        @test_throws ArgumentError DifferenTT.toeplitz_mpo(siteinds(2, 1), short_coeffs, bc_dirichlet)
        @test_throws ArgumentError DifferenTT.toeplitz_mpo(siteinds(2, 3), coeffs, bc_dirichlet)
    end

    @testset "column_pick_mpo" begin
        col_idx = 2
        mpo = DifferenTT.column_pick_mpo(sites, col_idx)
        @test mpo isa MPO
        @test length(mpo) == n

        # Test vector version
        mpo_sum = DifferenTT.column_pick_mpo(sites, [1, 2])
        @test mpo_sum isa MPO

        @test_throws ArgumentError DifferenTT.column_pick_mpo(sites, 0)
        @test_throws ArgumentError DifferenTT.column_pick_mpo(sites, 2^n + 1)

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

    # Dense references: A[i, i+k] = c_k, wrapped around (circulant) for periodic boundary conditions. The coefficients
    # are random, so a transposed or shifted operator would not match
    @testset "toeplitz_mpo against dense Toeplitz and circulant matrices" begin
        r = rng()
        function dense_toeplitz(c::Dict, n::Int, periodic::Bool)
            A = zeros(n, n)
            for i in 1:n, (k, ck) in c
                j = i + k
                if periodic
                    A[i, mod1(j, n)] += ck
                elseif 1 <= j <= n
                    A[i, j] += ck
                end
            end
            return A
        end
        bcs = (
            BoundaryCond1d(PeriodicBC(), PeriodicBC()),
            BoundaryCond1d(DirichletBC(0.0), DirichletBC(0.0)),
            BoundaryCond1d(NeumannBC(0.0), DirichletBC(1.0)),
        )
        for (offsets, nsites) in ((-1:1, 2:5), (-4:4, 4:6)), nsite in nsites, bc in bcs
            c = Dict(k => randn(r) for k in offsets)
            periodic = bc.left isa PeriodicBC
            A = DifferenTT.toeplitz_mpo(siteinds(2, nsite), c, bc)
            @test todense(A) ≈ dense_toeplitz(c, 2^nsite, periodic)
            @test maxlinkdim(A) <= length(offsets)
            # the method that creates its own sites
            @test todense(DifferenTT.toeplitz_mpo(nsite, c, bc)) ≈ todense(A)
        end
        bc = bcs[1]
        @test_throws KeyError DifferenTT.toeplitz_mpo(siteinds(2, 3), Dict(-1 => 1.0, 0 => 1.0, 2 => 1.0), bc)
        @test_throws ArgumentError DifferenTT.toeplitz_mpo(siteinds(2, 4), Dict(k => 1.0 for k in -2:2), bc)
        @test_throws ArgumentError DifferenTT.toeplitz_mpo(1, Dict(k => 1.0 for k in -1:1), bc)
    end

    @testset "column_pick_mpo, row_pick_mpo and entry_mpo against dense references" begin
        s = siteinds(2, 4)
        N = 16
        e(k) = (v=zeros(N); v[k]=1.0; v)
        for c in (1, 6, 16)
            @test todense(DifferenTT.column_pick_mpo(s, c)) == ones(N) * e(c)'
            @test todense(DifferenTT.row_pick_mpo(s, c)) == e(c) * ones(N)'
        end
        cols = [2, 7, 7, 16]
        Cv = DifferenTT.column_pick_mpo(s, cols)
        @test todense(Cv) ≈ sum(ones(N) * e(c)' for c in cols)
        @test maxlinkdim(Cv) <= length(cols)
        @test todense(DifferenTT.row_pick_mpo(s, cols)) ≈ sum(e(c) * ones(N)' for c in cols)
        @test todense(DifferenTT.column_pick_mpo(s)) == ones(N) * e(1)'
        @test todense(DifferenTT.row_pick_mpo(s)) == e(1) * ones(N)'
        # applied to a vector: the column pick copies entry c everywhere, the row pick writes the sum into entry r
        x = randn(rng(), N)
        @test todense(DifferenTT.column_pick_mpo(s, 6)) * x ≈ fill(x[6], N)
        @test todense(DifferenTT.row_pick_mpo(s, 6)) * x ≈ sum(x) * e(6)
        # one site
        s1 = siteinds(2, 1)
        @test todense(DifferenTT.column_pick_mpo(s1, 2)) == [0.0 1.0; 0.0 1.0]
        @test todense(DifferenTT.entry_mpo(s1, 2, 1)) == [0.0 0.0; 1.0 0.0]

        @test_throws ArgumentError DifferenTT.column_pick_mpo(s, [1, 17])
        @test_throws ArgumentError DifferenTT.row_pick_mpo(s, 0)
        @test_throws ArgumentError DifferenTT.row_pick_mpo(s, [3, 0])
        @test_throws ArgumentError DifferenTT.entry_mpo(s, 1, 17)
        @test_throws ArgumentError DifferenTT.entry_mpo(s, 0, 1)
    end

    @testset "corner_pick_mpo on two sites and errors" begin
        c = [1.0, 2.0, 3.0, 4.0]
        s2 = siteinds(2, 2)
        @test todense(DifferenTT.corner_pick_mpo(c, s2)) ≈ [1 2 3 4; 0 1 2 3; 0 0 1 2; 0 0 0 1]
        @test todense(DifferenTT.corner_pick_mpo(c, s2, :lower)) ≈ [1 0 0 0; 2 1 0 0; 3 2 1 0; 4 3 2 1]
        @test_throws ArgumentError DifferenTT.corner_pick_mpo(c, s2, :middle)
        @test_throws ArgumentError DifferenTT.corner_pick_mpo(c, siteinds(2, 1))
    end
end

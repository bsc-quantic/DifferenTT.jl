using DifferenTT
using Test
using ITensors
using ITensorMPS
using LinearAlgebra

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
        bc_periodic = Pdetn.BoundaryCond1d(Pdetn.PeriodicBC(), Pdetn.PeriodicBC())
        
        eigth_order_mpo = Pdetn.toeplitz_mpo(sites, coeffs, bc_periodic)
        @test eigth_order_mpo isa MPO
        @test length(eigth_order_mpo) == n
        @test linkdim(eigth_order_mpo, 1) == 3
        @test linkdim(eigth_order_mpo, n-1) == 9

        second_order_mpo = Pdetn.toeplitz_mpo(sites, short_coeffs, bc_periodic)
        @test second_order_mpo isa MPO
        @test length(second_order_mpo) == n
        @test linkdim(second_order_mpo, 1) == 3
        @test linkdim(second_order_mpo, n-1) == 3

        @test_throws ArgumentError Pdetn.toeplitz_mpo(sites, Dict(0 => 2.0, 1 => -1.0), bc_periodic)

        # Dirichlet
        bc_dirichlet = Pdetn.BoundaryCond1d(Pdetn.DirichletBC(0.0), Pdetn.DirichletBC(0.0))
        mpo_d = Pdetn.toeplitz_mpo(sites, coeffs, bc_dirichlet)
        @test mpo_d isa MPO
        @test length(mpo_d) == n
    end

    @testset "column_pick_mpo" begin
        col_idx = 2
        mpo = Pdetn.column_pick_mpo(sites, col_idx)
        @test mpo isa MPO
        @test length(mpo) == n
    
        # Test vector version
        mpo_sum = Pdetn.column_pick_mpo(sites, [1, 2])
        @test mpo_sum isa MPO

        # Test MPO entries
        # Since col_idx = 2, reverse(digits(1, base=2, pad=6)) = [0, 0, 0, 0, 0, 1]
        # So sites 1-5 have matrix0 = [1.0 0.0; 1.0 0.0]
        # Site 6 has matrix1 = [0.0 1.0; 0.0 1.0]
        @test all(array(mpo[5], sites[5]', sites[5], linkinds(mpo)[4], linkinds(mpo)[5])[:, :, 1, 1] .== [1.0 0.0; 1.0 0.0])
        @test all(array(mpo[6], sites[6]', sites[6], linkinds(mpo)[5])[:, :, 1] .== [0.0 1.0; 0.0 1.0])
    end

    @testset "row_pick_mpo" begin
        row_idx = 2
        mpo = Pdetn.row_pick_mpo(sites, row_idx)
        @test mpo isa MPO
        @test length(mpo) == n
        
        # Test vector version
        mpo_sum = Pdetn.row_pick_mpo(sites, [1, 2])
        @test mpo_sum isa MPO

        # Test MPO entries
        # Transpose of column_pick_mpo
        @test all(array(mpo[5], sites[5]', sites[5], linkinds(mpo)[4], linkinds(mpo)[5])[:, :, 1, 1] .== [1.0 1.0; 0.0 0.0])
        @test all(array(mpo[6], sites[6]', sites[6], linkinds(mpo)[5])[:, :, 1] .== [0.0 0.0; 1.0 1.0])
    end

    @testset "corner_pick_mpo" begin
        coefficients = [1.0, 2.0, 3.0, 4.0]
        mpo = Pdetn.corner_pick_mpo(coefficients, sites)
        @test mpo isa MPO
        @test length(mpo) == n
        @test linkdim(mpo, n - 2) == 1
        @test linkdim(mpo, n - 1) == 2
    end
end
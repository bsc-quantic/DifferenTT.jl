using DifferenTT
using Test
using ITensors
using ITensorMPS
using LinearAlgebra
using Random

@testset "elementwise.jl" begin
    n = 4
    sites = siteinds("Qubit", n)
    
    @testset "tt_mul MPS-MPS" begin
        Random.seed!(1234)
        mps1 = random_mps(sites; linkdims=3)
        mps2 = random_mps(sites; linkdims=4)
        
        mps_prod = tt_mul(mps1, mps2)
        
        A1 = contract_mps(mps1)
        A2 = contract_mps(mps2)
        A_prod = contract_mps(mps_prod)
        
        @test A_prod ≈ A1 .* A2
        
        mps_prod_direct = DifferenTT._direct_tt_mul(mps1, mps2)
        @test contract_mps(mps_prod_direct) ≈ A1 .* A2
    end
    
    @testset "tt_mul MPO-MPO" begin
        Random.seed!(1234)
        mpo1 = random_mpo(sites)
        mpo2 = random_mpo(sites)
        
        mpo_prod = tt_mul(mpo1, mpo2)
        
        A1 = contract_mpo(mpo1)
        A2 = contract_mpo(mpo2)
        A_prod = contract_mpo(mpo_prod)
        
        @test A_prod ≈ A1 .* A2
    end
    
    @testset "tt_mul MPS-MPO and MPO-MPS" begin
        Random.seed!(1234)
        mps = random_mps(sites; linkdims=2)
        mpo = random_mpo(sites)
        
        mpo_prod1 = tt_mul(mps, mpo)
        mpo_prod2 = tt_mul(mpo, mps)
        
        T_prod1 = contract_mpo(mpo_prod1)
        T_prod2 = contract_mpo(mpo_prod2)
        
        @test size(T_prod1) == size(contract_mpo(mpo))
        @test size(T_prod2) == size(contract_mpo(mpo))
        
        @test T_prod1 ≈ T_prod2
        
        @test mpo_prod1 isa MPO
        @test mpo_prod2 isa MPO
    end
    
    @testset "elementwise_function" begin
        Random.seed!(1234)
        mps = random_mps(sites; linkdims=2)
        
        for i in 1:n
            mps[i] = ITensor(abs.(array(mps[i])) .+ 0.1, inds(mps[i]))
        end
        
        func = x -> x^2
        mps_mapped = elementwise_function(func, mps, true; maxdim=20, oversample=10)
        
        A_mps = contract_mps(mps)
        A_mapped = contract_mps(mps_mapped)
        
        @test isapprox(A_mapped, func.(A_mps); rtol=0.5, atol=0.5)
    end
end

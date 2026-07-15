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
        
        mps_prod = DifferenTT.tt_mul(mps1, mps2)
        
        A1 = array(contract(mps1))
        A2 = array(contract(mps2))
        A_prod = array(contract(mps_prod))
        
        @test A_prod ≈ A1 .* A2
        
        mps_prod_direct = DifferenTT._direct_tt_mul(mps1, mps2)
        @test array(contract(mps_prod_direct)) ≈ A1 .* A2
    end
    
    @testset "tt_mul MPO-MPO" begin
        Random.seed!(1234)
        mpo1 = random_mpo(sites)
        mpo2 = random_mpo(sites)
        
        mpo_prod = DifferenTT.tt_mul(mpo1, mpo2)
        
        T1 = contract(mpo1)
        T2 = contract(mpo2)
        T_prod = contract(mpo_prod)
        
        T2_aligned = permute(T2, inds(T1)...)
        T_prod_aligned = permute(T_prod, inds(T1)...)
        
        A1 = array(T1)
        A2 = array(T2_aligned)
        A_prod = array(T_prod_aligned)
        
        @test A_prod ≈ A1 .* A2
    end
    
    @testset "tt_mul MPS-MPO and MPO-MPS" begin
        Random.seed!(1234)
        mps = random_mps(sites; linkdims=2)
        mpo = random_mpo(sites)
        
        mpo_prod1 = DifferenTT.tt_mul(mps, mpo)
        mpo_prod2 = DifferenTT.tt_mul(mpo, mps)
        
        T_prod1 = contract(mpo_prod1)
        T_prod2 = contract(mpo_prod2)
        
        @test hasinds(T_prod1, inds(contract(mpo))...)
        @test hasinds(T_prod2, inds(contract(mpo))...)
        
        T_prod2_aligned = permute(T_prod2, inds(T_prod1)...)
        @test array(T_prod1) ≈ array(T_prod2_aligned)
        
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
        mps_mapped = DifferenTT.elementwise_function(func, mps, true; maxdim=20, oversample=10)
        
        A_mps = array(contract(mps))
        A_mapped = array(contract(mps_mapped))
        
        @test isapprox(A_mapped, func.(A_mps); rtol=0.5, atol=0.5)
    end
end

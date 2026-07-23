using DifferenTT
using Test
using ITensors
using LinearAlgebra
using ITensorMPS

@testset "qttoperations.jl" begin
    @testset "ndim_operator_mpo" begin
        N = 2
        var_nsites = (2, 3)
        
        # Blocked architecture
        arch = DifferenTT.QTTArchitecture(var_nsites)
        full_sites = siteinds(arch)
        
        sites_1 = siteinds(2, 2)
        mpo_1 = MPO(sites_1, "Id")
        lifted_mpo_1 = DifferenTT.ndim_operator_mpo(mpo_1, 1, full_sites, arch)
        @test length(lifted_mpo_1) == length(full_sites)
        mps_test = random_mps(full_sites)
        @test norm(apply(lifted_mpo_1, mps_test)) > 0
        
        # Reversal case
        arch_rev = DifferenTT.QTTArchitecture((-2, 3))
        lifted_mpo_rev = DifferenTT.ndim_operator_mpo(mpo_1, 1, full_sites, arch_rev)
        @test length(lifted_mpo_rev) == length(full_sites)
        
        # Interleaved architecture
        arch_int = DifferenTT.QTTArchitecture(var_nsites; interleaved=true)
        full_sites_int = siteinds(arch_int)
        lifted_mpo_int = DifferenTT.ndim_operator_mpo(mpo_1, 1, full_sites_int, arch_int)
        @test length(lifted_mpo_int) == length(full_sites_int)
        mps_test_int = random_mps(full_sites_int)
        @test norm(apply(lifted_mpo_int, mps_test_int)) > 0
    end
end
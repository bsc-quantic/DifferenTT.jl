using DifferenTT
using Test
using ITensorMPS: ITensorMPS, random_mpo, random_mps, siteinds
using ITensors
using DifferenTT: LocalUpdaterArgs, GlobalUpdaterArgs, SweepingArgs, default_linearsolve_init, ResidualALS, ResidualObserver

@testset "LinearSolvers configuration structs" begin
    @testset "LocalUpdaterArgs" begin
        args = LocalUpdaterArgs()
        nt = NamedTuple(args)
        @test nt.ishermitian == false
        @test nt.krylovdim == 30
    end

    @testset "GlobalUpdaterArgs" begin
        args = GlobalUpdaterArgs()
        nt = NamedTuple(args)
        @test nt.maxdim == typemax(Int)
        @test nt.orth_cutoff == 1e-12
    end

    @testset "SweepingArgs" begin
        args = SweepingArgs()
        nt = NamedTuple(args)
        @test nt.nsweeps == 5
    end
end

@testset "ResidualObserver and default_differenTT_checkdone" begin
    N = 4
    s = siteinds("S=1/2", N)
    A = random_mpo(s)
    b = random_mps(s)
    
    # Large tolerance should result in early termination
    obs_large = ResidualObserver(; A=A, b=b, tol=1e4)
    # Very small tolerance should not result in early termination for random MPS
    obs_small = ResidualObserver(; A=A, b=b, tol=1e-12)
    
    x = random_mps(s)
    
    # Test checkdone! directly
    @test ITensorMPS.checkdone!(obs_large; state=x) == true
    @test ITensorMPS.checkdone!(obs_small; state=x) == false
    
    # Test default_differenTT_checkdone properly parsing the kwargs
    @test DifferenTT.default_differenTT_checkdone(; observer=obs_large, state=x, sweep=1) == true
    @test DifferenTT.default_differenTT_checkdone(; observer=obs_small, state=x, sweep=1) == false
    
    # Test with EmptyObserver (should safely return false and not crash)
    empty_obs = ITensorMPS.EmptyObserver()
    @test DifferenTT.default_differenTT_checkdone(; observer=empty_obs, state=x, sweep=1) == false
end

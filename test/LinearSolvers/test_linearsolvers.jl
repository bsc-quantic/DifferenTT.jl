using DifferenTT
using Test
using ITensorMPS: ITensorMPS, random_mpo, random_mps, siteinds
using ITensors
using ITensorMPS: MPO, apply
using LinearAlgebra: norm
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
    obs_large = ResidualObserver(; A=A, b=b, rtol=1e4)
    # Very small tolerance should not result in early termination for random MPS
    obs_small = ResidualObserver(; A=A, b=b, rtol=1e-12)
    
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

# Counts the sweeps performed by an alternating update
mutable struct SweepCounter <: ITensorMPS.AbstractObserver
    nsweeps::Int
end
ITensorMPS.update_observer!(o::SweepCounter; kwargs...) = (o.nsweeps += 1; o)

relative_residual(A, x, b) = norm(apply(A, x) - b) / norm(b)

@testset "ResidualObserver binding in linearsolve" begin
    N = 4
    s = siteinds("S=1/2", N)
    A = 2.0 * MPO(s, "Id")
    b1 = random_mps(s; linkdims=2)
    b2 = random_mps(s; linkdims=2)
    rtol = 1e-6
    nsweeps = 10

    @testset "bind_observer" begin
        unbound = ResidualObserver(rtol=rtol)
        @test isnothing(unbound.A) && isnothing(unbound.b)

        bound = DifferenTT.bind_observer(unbound, A, b1)
        @test bound.A === A
        @test bound.b === b1
        @test bound.rtol == rtol
        # The original observer is left unbound
        @test isnothing(unbound.A) && isnothing(unbound.b)

        # Explicitly given operator and RHS are kept
        explicit = ResidualObserver(A, b2, rtol)
        rebound = DifferenTT.bind_observer(explicit, 3.0 * A, b1)
        @test rebound.A === A
        @test rebound.b === b2

        # Other observers are untouched
        empty_obs = ITensorMPS.EmptyObserver()
        @test DifferenTT.bind_observer(empty_obs, A, b1) === empty_obs
    end

    @testset "Unbound observer throws in checkdone!" begin
        @test_throws ArgumentError ITensorMPS.checkdone!(ResidualObserver(rtol=rtol); state=b1)
    end

    function counted_solver(solver_type; observer=true)
        counter = SweepCounter(0)
        sweep_kwargs = observer ?
            SweepingArgs(; nsweeps=nsweeps, maxdim=10, cutoff=1e-14, observer! = ResidualObserver(rtol=rtol), sweep_observer! = counter) :
            SweepingArgs(; nsweeps=nsweeps, maxdim=10, cutoff=1e-14, sweep_observer! = counter)
        return solver_type(; sweep_kwargs=sweep_kwargs), counter
    end

    @testset "Early stopping with $(solver_type)" for solver_type in (ALS, ResidualALS, AMEn)
        solver, counter = counted_solver(solver_type)
        x = linearsolve(solver, A, b1, nothing)
        @test relative_residual(A, x, b1) < rtol
        @test counter.nsweeps < nsweeps
        # The observer stored in the solver stays unbound
        @test isnothing(solver.sweep_kwargs.observer!.A) && isnothing(solver.sweep_kwargs.observer!.b)
    end

    @testset "No observer runs all sweeps" begin
        solver, counter = counted_solver(ALS; observer=false)
        linearsolve(solver, A, b1, nothing)
        @test counter.nsweeps == nsweeps
    end

    @testset "Preconditioned solve checks the unpreconditioned residual" begin
        solver, counter = counted_solver(ALS)
        M⁻¹ = rank1_preconditioner(A)
        x = linearsolve(solver, A, b1, M⁻¹, nothing)
        @test relative_residual(A, x, b1) < rtol
        @test counter.nsweeps < nsweeps
    end

    @testset "Consecutive solves bind their own RHS" begin
        solver, _ = counted_solver(ALS)
        x1 = linearsolve(solver, A, b1, nothing)
        x2 = linearsolve(solver, A, b2, nothing)
        @test relative_residual(A, x1, b1) < rtol
        @test relative_residual(A, x2, b2) < rtol
    end
end

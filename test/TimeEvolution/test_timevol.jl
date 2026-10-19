using DifferenTT
using Test

@testset "TimeEvolArgs Configuration" begin
    @testset "Constructor with nsteps" begin
        t = TimeEvolArgs(0.0, 1.0, 10)
        @test t.ti == 0.0
        @test t.tf == 1.0
        @test t.dt == 0.1
        @test t.nsteps == 10
    end

    @testset "Constructor with dt" begin
        t = TimeEvolArgs(0.0, 1.0, 0.1)
        @test t.ti == 0.0
        @test t.tf == 1.0
        @test t.dt == 0.1
        @test t.nsteps == 10
    end

    @testset "No integrator scheme field" begin
        @test !hasfield(TimeEvolArgs, :scheme)
    end

    @testset "Steppers store TimeEvolArgs" begin
        t = TimeEvolArgs(0.0, 1.0, 10)
        for stepper in
            (ExplicitEuler(time_evol_kwargs=t, cutoff=1e-8, maxdim=10), RK4(time_evol_kwargs=t, cutoff=1e-8, maxdim=10))
            @test stepper isa AbstractTimeStepper
            @test stepper.time_evol_kwargs === t
            @test stepper.tt_kwargs == (cutoff=1e-8, maxdim=10)
        end
        @test_throws UndefKeywordError RK4(cutoff=1e-8)
    end
end

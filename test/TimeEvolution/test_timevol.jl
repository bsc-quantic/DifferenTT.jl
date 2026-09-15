using DifferenTT
using Test

@testset "TimeEvol Configuration" begin
    @testset "Constructor with nsteps" begin
        t = TimeEvol(0.0, 1.0, 10)
        @test t.ti == 0.0
        @test t.tf == 1.0
        @test t.dt == 0.1
        @test t.nsteps == 10
    end

    @testset "Constructor with dt" begin
        t = TimeEvol(0.0, 1.0, 0.1)
        @test t.ti == 0.0
        @test t.tf == 1.0
        @test t.dt == 0.1
        @test t.nsteps == 10
    end

    @testset "No integrator scheme field" begin
        @test !hasfield(TimeEvol, :scheme)
    end
end

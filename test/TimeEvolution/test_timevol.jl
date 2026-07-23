using DifferenTT
using Test

@testset "TimeEvol Configuration" begin
    @testset "Constructor with nsteps" begin
        t = TimeEvol(0.0, 1.0, 10)
        @test t.ti == 0.0
        @test t.tf == 1.0
        @test t.dt == 0.1
        @test t.nsteps == 10
        @test t.scheme == :euler

        t_rk4 = TimeEvol(0.0, 1.0, 10, :rk4)
        @test t_rk4.scheme == :rk4
    end

    @testset "Constructor with dt" begin
        t = TimeEvol(0.0, 1.0, 0.1)
        @test t.ti == 0.0
        @test t.tf == 1.0
        @test t.dt == 0.1
        @test t.nsteps == 10
        @test t.scheme == :euler

        t_rk4 = TimeEvol(0.0, 1.0, 0.1, :rk4)
        @test t_rk4.scheme == :rk4
    end

    @testset "Invalid scheme" begin
        @test_throws ArgumentError TimeEvol(0.0, 1.0, 10, :unknown_scheme)
        @test_throws ArgumentError TimeEvol(0.0, 1.0, 0.1, :unknown_scheme)
    end
end

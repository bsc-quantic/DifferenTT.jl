using DifferenTT
using Test
using ITensorMPS: ITensorMPS
using DifferenTT: LocalUpdaterArgs, GlobalUpdaterArgs, SweepingArgs, default_linearsolve_init, ResidualALS

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

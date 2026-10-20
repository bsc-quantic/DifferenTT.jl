using ITensors
using ITensorMPS
using DifferenTT
using Test

include(joinpath(@__DIR__, "..", "testutils.jl"))
using .TestUtils

@testset "scalarfields.jl" begin
    # 2D BCs
    bcu1d = BoundaryCond1d(PeriodicBC(), PeriodicBC())
    bcu2d = BoundaryConditions(bcu1d, bcu1d)

    @testset "ScalarField with QTTArchitecture (2D)" begin
        var_nsites = (3, 4)
        arch = QTTArchitecture(var_nsites)

        n_sites = total_sites(arch)
        sites = [Index(2, "Site,n=$i") for i in 1:n_sites]
        u_init = random_mps(sites)

        field = ScalarField(u_init, arch, bcu2d)

        @test field isa ScalarField{2}
        @test field.arch == arch
        @test field.dims[1] == 8
        @test field.dims[2] == 16
        @test length(field.u) == 1
        @test field.u[1] == u_init

        @test mps_equal(last(field), u_init)
        @test field.u[1] == field[1]

        u_new = random_mps(sites)
        push!(field, u_new)
        @test mps_equal(last(field), u_new)

        bad_sites = [Index(2, "Site,n=$i") for i in 1:n_sites]
        u_bad = random_mps(bad_sites)

        # siteinds check might fail if indices match by ID but not object. 
        # Typically siteinds creates new indices.
        @test_throws ArgumentError push!(field, u_bad)
        @test_throws ArgumentError field[1] = u_bad

        @test length(field) == 2

        # push! and setindex! return the field, as Base's do
        @test push!(field, u_new) === field
        @test setindex!(field, u_init, 3) === field
        @test length(field) == 3

        field[1] = u_new
        @test mps_equal(field[1], u_new)

        empty_field = ScalarField(arch, bcu2d)
        @test empty_field.arch == arch
        @test length(empty_field) == 0

        field_from_sites = ScalarField(sites, arch, bcu2d)
        @test field_from_sites.arch == arch
        @test length(field_from_sites) == 0
        @test field_from_sites.usites == sites
    end

    @testset "ScalarField with TTArchitecture (2D)" begin
        var_dims = (8, 16)
        arch = TTArchitecture(var_dims)

        n_sites = total_sites(arch)
        sites = [Index(var_dims[i], "Site,n=$i") for i in 1:n_sites]
        u_init = random_mps(sites)

        field = ScalarField(u_init, arch, bcu2d)

        @test field isa ScalarField{2}
        @test field.arch == arch
        @test field.dims == var_dims
        @test length(field.u) == 1
        @test field.u[1] == u_init

        @test mps_equal(last(field), u_init)
        @test field.u[1] == field[1]

        u_new = random_mps(sites)
        push!(field, u_new)
        @test mps_equal(last(field), u_new)

        bad_sites = [Index(var_dims[i], "Site,n=$i") for i in 1:n_sites]
        u_bad = random_mps(bad_sites)

        @test_throws ArgumentError push!(field, u_bad)

        @test length(field) == 2

        empty_field = ScalarField(arch, bcu2d)
        @test empty_field.arch == arch
        @test length(empty_field) == 0

        field_from_sites = ScalarField(sites, arch, bcu2d)
        @test field_from_sites.arch == arch
        @test length(field_from_sites) == 0
        @test field_from_sites.usites == sites
    end
end

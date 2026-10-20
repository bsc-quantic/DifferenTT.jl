using ITensors
using ITensorMPS
using DifferenTT
using Test

include(joinpath(@__DIR__, "..", "testutils.jl"))
using .TestUtils

@testset "vectorfields.jl" begin
    bcu1d = BoundaryCond1d(PeriodicBC(), PeriodicBC())
    bcu2d = BoundaryConditions(bcu1d, bcu1d)
    bcs2d = (bcu2d, bcu2d)

    @testset "VectorField with QTTArchitecture (2D)" begin
        var_nsites = (3, 4)
        arch = QTTArchitecture(var_nsites)

        n_sites = total_sites(arch)
        sites = [Index(2, "Site,n=$i") for i in 1:n_sites]
        u_init = (random_mps(sites), random_mps(sites))

        field = VectorField(u_init, arch, bcs2d)

        @test field isa VectorField{2}
        @test field.arch == arch
        @test length(field.components) == 2
        @test length(field) == 1

        @test mps_equal(last(field), u_init)
        @test field[1] isa ScalarField{2}

        u_new = (random_mps(sites), random_mps(sites))
        push!(field, u_new)
        @test mps_equal(last(field), u_new)
        @test length(field) == 2

        bad_sites = [Index(2, "Site,n=$i") for i in 1:n_sites]
        u_bad = (random_mps(bad_sites), random_mps(bad_sites))

        @test_throws ArgumentError push!(field, u_bad)

        # A mismatch in a later component must not leave the earlier ones with one more state (regression: push!
        # and setindex! used to mutate component 1 before component 2 threw)
        u_half_bad = (random_mps(sites), random_mps(bad_sites))
        @test_throws ArgumentError push!(field, u_half_bad)
        @test length(field[1]) == length(field[2]) == 2
        @test_throws ArgumentError field[1] = u_half_bad
        @test mps_equal(last(field), u_new)

        # push! and setindex! return the field
        @test push!(field, u_new) === field
        @test length(field) == 3
        @test setindex!(field, u_init, 3) === field

        field[1] = u_new
        @test mps_equal(field.components[1][1], u_new[1])
        @test mps_equal(field.components[2][1], u_new[2])

        empty_field = VectorField(arch, bcs2d)
        @test empty_field.arch == arch
        @test length(empty_field) == 0

        field_from_sites = VectorField(sites, arch, bcs2d)
        @test field_from_sites.arch == arch
        @test length(field_from_sites) == 0
        @test field_from_sites.usites == sites
    end

    @testset "VectorField with TTArchitecture (2D)" begin
        var_dims = (8, 16)
        arch = TTArchitecture(var_dims)

        n_sites = total_sites(arch)
        sites = [Index(var_dims[i], "Site,n=$i") for i in 1:n_sites]
        u_init = (random_mps(sites), random_mps(sites))

        field = VectorField(u_init, arch, bcs2d)

        @test field isa VectorField{2}
        @test field.arch == arch
        @test length(field) == 1

        @test mps_equal(last(field), u_init)
        @test field[1] isa ScalarField{2}

        u_new = (random_mps(sites), random_mps(sites))
        push!(field, u_new)
        @test mps_equal(last(field), u_new)
        @test length(field) == 2

        bad_sites = [Index(var_dims[i], "Site,n=$i") for i in 1:n_sites]
        u_bad = (random_mps(bad_sites), random_mps(bad_sites))

        @test_throws ArgumentError push!(field, u_bad)

        empty_field = VectorField(arch, bcs2d)
        @test empty_field.arch == arch
        @test length(empty_field) == 0

        field_from_sites = VectorField(sites, arch, bcs2d)
        @test field_from_sites.arch == arch
        @test length(field_from_sites) == 0
        @test field_from_sites.usites == sites
    end

    @testset "VectorField from ScalarField components" begin
        arch = QTTArchitecture((2, 2))
        sites = siteinds(arch)
        sx = ScalarField(random_mps(sites), arch, bcu2d)
        sy = ScalarField(random_mps(sites), arch, bcu2d)
        field = VectorField((sx, sy))
        @test field isa VectorField{2}
        @test field.components === (sx, sy)
        @test field.usites == sites
        # The number of components must match the dimension, and the tuple must not be empty
        @test_throws DimensionMismatch VectorField((sx,))
        @test_throws MethodError VectorField(())
    end

    @testset "backend_type" begin
        arch = QTTArchitecture((2, 2))
        sites = siteinds(arch)
        sx = ScalarField(random_mps(rng(1), sites), arch, bcu2d)
        sy = ScalarField(random_mps(rng(2), sites), arch, bcu2d)
        @test DifferenTT.backend_type(sx) === Array
        @test DifferenTT.backend_type(VectorField((sx, sy))) === Array
        @test DifferenTT.backend_type(VectorField(arch, bcs2d)) === Array
        @test VectorField(arch, bcs2d).array_type === Array
        # The components of a vector field share one backend
        @test all(DifferenTT.backend_type(c) === Array for c in VectorField(arch, bcs2d).components)
    end
end

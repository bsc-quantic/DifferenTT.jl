using ITensors
using ITensorMPS
using DifferenTT
using Test

function Base.:(==)(a::ITensorMPS.AbstractMPS, b::ITensorMPS.AbstractMPS)
    return all(a .== b)
end

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
        
        @test last(field) == u_init
        @test field[1] isa ScalarField{2}
        
        u_new = (random_mps(sites), random_mps(sites))
        push!(field, u_new)
        @test last(field) == u_new
        @test length(field) == 2

        bad_sites = [Index(2, "Site,n=$i") for i in 1:n_sites]
        u_bad = (random_mps(bad_sites), random_mps(bad_sites))
        
        @test_throws ArgumentError push!(field, u_bad)

        field[1] = u_new
        @test field.components[1][1] == u_new[1]
        @test field.components[2][1] == u_new[2]

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
        
        @test last(field) == u_init
        @test field[1] isa ScalarField{2}
        
        u_new = (random_mps(sites), random_mps(sites))
        push!(field, u_new)
        @test last(field) == u_new
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
end

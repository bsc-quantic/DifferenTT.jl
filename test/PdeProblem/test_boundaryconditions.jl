using DifferenTT
using Test

include(joinpath(@__DIR__, "..", "testutils.jl"))
using .TestUtils

@testset "boundaryconditions.jl" begin
    bc_value = 5.0
    neumann_value = 0.0
    dirichlet_bc = quietly(() -> DirichletBC(bc_value))
    neumann_bc = NeumannBC(neumann_value)
    periodic_bc = PeriodicBC()
    open_bc = OpenBC()

    @testset "BC Types" begin
        @test dirichlet_bc.value == bc_value
        @test neumann_bc.value == neumann_value
        @test periodic_bc isa PeriodicBC
        @test open_bc isa OpenBC
        @test_throws ArgumentError NeumannBC(bc_value)
    end

    @testset "BoundaryCond1d" begin
        bc1d = BoundaryCond1d(dirichlet_bc, neumann_bc)
        bcopen1d = BoundaryCond1d(open_bc, open_bc)

        @test bc1d.left isa DirichletBC
        @test bc1d.right isa NeumannBC
        @test bcopen1d.left isa OpenBC
        @test bcopen1d.right isa OpenBC

        @test bc1d[1] == bc_value
        @test bc1d[2] == neumann_value
        @test isnothing(bcopen1d[1])
        @test isnothing(bcopen1d[2])

        @test_throws ArgumentError bc1d[3]
        @test_throws ArgumentError BoundaryCond1d(periodic_bc, neumann_bc)
    end

    @testset "BoundaryConditions (2D)" begin
        bc1d_x = BoundaryCond1d(dirichlet_bc, neumann_bc)
        bc1d_y = BoundaryCond1d(neumann_bc, dirichlet_bc)

        # New constructor: tuple or varargs
        bc2d = BoundaryConditions(bc1d_x, bc1d_y)

        @test bc2d isa BoundaryConditions{2}
        @test bc2d[1] isa BoundaryCond1d
        @test bc2d[2] isa BoundaryCond1d
        @test bc2d[1].left isa DirichletBC
        @test bc2d[2].right isa DirichletBC
    end

    @testset "BoundaryConditions (3D)" begin
        bc1d_x = BoundaryCond1d(dirichlet_bc, neumann_bc)
        bc1d_y = BoundaryCond1d(neumann_bc, dirichlet_bc)
        bc1d_z = BoundaryCond1d(dirichlet_bc, dirichlet_bc)

        bc3d = BoundaryConditions(bc1d_x, bc1d_y, bc1d_z)

        @test bc3d isa BoundaryConditions{3}
        @test bc3d[1] isa BoundaryCond1d
        @test bc3d[2] isa BoundaryCond1d
        @test bc3d[3] isa BoundaryCond1d
        @test bc3d[1].right isa NeumannBC
        @test bc3d[2].right isa DirichletBC
        @test bc3d[3].left isa DirichletBC
    end
end

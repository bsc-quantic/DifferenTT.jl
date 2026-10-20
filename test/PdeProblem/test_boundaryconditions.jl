using DifferenTT
using Test

include(joinpath(@__DIR__, "..", "testutils.jl"))
using .TestUtils

@testset "boundaryconditions.jl" begin
    bc_value = 5.0
    neumann_value = 0.0
    dirichlet_bc = DirichletBC(bc_value)
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

    @testset "Construction is silent and validates its input" begin
        # DirichletBC used to log an info note at construction; none of the types may log now
        @test_logs DirichletBC(1.0)
        @test_logs DirichletBC(2)
        @test_logs NeumannBC(0.0)
        @test_logs PeriodicBC()
        @test_logs OpenBC()
        @test_logs BoundaryCond1d(DirichletBC(0.0), NeumannBC(0.0))
        @test_logs BoundaryConditions(BoundaryCond1d(PeriodicBC(), PeriodicBC()))

        # The value keeps its type, and getindex returns it as is
        @test DirichletBC(2).value === 2
        @test DirichletBC(1.5 + 2im).value === 1.5 + 2im
        @test BoundaryCond1d(DirichletBC(2), DirichletBC(-0.5))[1] === 2
        @test BoundaryCond1d(DirichletBC(2), DirichletBC(-0.5))[2] === -0.5

        # Periodic conditions come in pairs: neither order is accepted with a non-periodic partner
        @test_throws ArgumentError BoundaryCond1d(dirichlet_bc, periodic_bc)
        @test_throws ArgumentError BoundaryCond1d(open_bc, periodic_bc)
        @test_throws ArgumentError BoundaryCond1d(periodic_bc, open_bc)
        @test BoundaryCond1d(periodic_bc, periodic_bc) isa BoundaryCond1d{PeriodicBC,PeriodicBC}
        @test isnothing(BoundaryCond1d(periodic_bc, periodic_bc)[1])

        # Only homogeneous Neumann conditions are implemented, whatever the number type
        @test_throws ArgumentError NeumannBC(-1.0)
        @test_throws ArgumentError NeumannBC(2)
        @test NeumannBC(0).value == 0

        # Only keys 1 and 2 address a side
        for key in (0, -1, 3)
            @test_throws ArgumentError BoundaryCond1d(dirichlet_bc, neumann_bc)[key]
        end
    end

    @testset "BoundaryConditions indexing and periodic_axes" begin
        wall = BoundaryCond1d(dirichlet_bc, neumann_bc)
        periodic = BoundaryCond1d(periodic_bc, periodic_bc)
        bcs = BoundaryConditions(wall, periodic, wall)
        @test bcs.conditions === (wall, periodic, wall)
        @test bcs[1] === wall && bcs[2] === periodic && bcs[3] === wall
        @test_throws BoundsError bcs[0]
        @test_throws BoundsError bcs[4]
        @test DifferenTT.periodic_axes(bcs) == (false, true, false)
        @test DifferenTT.periodic_axes(BoundaryConditions(periodic, periodic)) == (true, true)
        @test DifferenTT.periodic_axes(BoundaryConditions(wall)) == (false,)
        # A field's boundary conditions are the ones it was built with
        arch = QTTArchitecture((2, 2, 2))
        @test ScalarField(arch, bcs).bcu === bcs
    end
end

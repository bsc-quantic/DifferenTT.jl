using DifferenTT
using Test

@testset "grid.jl" begin
    @testset "Grid with QTTArchitecture (2D)" begin
        var_nsites = (3, 4)
        arch = QTTArchitecture(var_nsites)
        box_size = (1.0, 2.0)
        grid = Grid(arch, box_size)

        @test grid isa Grid{2}
        @test grid.box_size == box_size

        dims = variable_dims(arch)
        @test dims == (8, 16)

        expected_dx = (1.0 / 7, 2.0 / 15)
        @test grid.dx[1] ≈ expected_dx[1]
        @test grid.dx[2] ≈ expected_dx[2]

        @test length(grid.points[1]) == 8
        @test length(grid.points[2]) == 16

        @test grid.points[1][1] ≈ 0.0
        @test grid.points[1][end] ≈ 1.0
        @test grid.points[2][1] ≈ 0.0
        @test grid.points[2][end] ≈ 2.0
    end

    @testset "Grid with TTArchitecture (2D)" begin
        var_dims = (8, 16)
        arch = TTArchitecture(var_dims)
        box_size = (1.0, 2.0)
        grid = Grid(arch, box_size)

        @test grid isa Grid{2}
        @test grid.box_size == box_size

        expected_dx = (1.0 / 7, 2.0 / 15)
        @test grid.dx[1] ≈ expected_dx[1]
        @test grid.dx[2] ≈ expected_dx[2]

        @test length(grid.points[1]) == 8
        @test length(grid.points[2]) == 16

        @test grid.points[1][1] ≈ 0.0
        @test grid.points[1][end] ≈ 1.0
        @test grid.points[2][1] ≈ 0.0
        @test grid.points[2][end] ≈ 2.0
    end

    @testset "Grid with 1D Architecture" begin
        arch_qtt = QTTArchitecture((4,))
        grid_qtt = Grid(arch_qtt, 1.0)

        @test grid_qtt isa Grid{1}
        @test grid_qtt.box_size == (1.0,)
        @test grid_qtt.dx[1] ≈ (1.0 / 15)
        @test length(grid_qtt.points[1]) == 16

        arch_tt = TTArchitecture((16,))
        grid_tt = Grid(arch_tt, 1.0)

        @test grid_tt isa Grid{1}
        @test grid_tt.box_size == (1.0,)
        @test grid_tt.dx[1] ≈ (1.0 / 15)
        @test length(grid_tt.points[1]) == 16
    end

    @testset "interior_grid (walls as ghost nodes)" begin
        domain_size = (1.0, 2.0)
        for arch in (QTTArchitecture((3, 4)), TTArchitecture((8, 16)))
            grid = interior_grid(arch, domain_size)
            dims = variable_dims(arch)

            @test grid isa Grid{2}
            # n interior nodes per axis, h = L / (n + 1)
            @test all(grid.dx .≈ domain_size ./ (dims .+ 1))
            @test all(grid.box_size .≈ domain_size .* (dims .- 1) ./ (dims .+ 1))
            for i in 1:2
                @test length(grid.points[i]) == dims[i]
                # Grid coordinates are physical coordinates minus h: the nodes span [h, L - h]
                @test grid.points[i][1] ≈ 0.0
                @test grid.points[i][end] + 2 * grid.dx[i] ≈ domain_size[i]
            end
        end

        grid_1d = interior_grid(QTTArchitecture((4,)), (1.0,))
        @test grid_1d isa Grid{1}
        @test grid_1d.dx[1] ≈ 1 / 17
    end

    @testset "interior_grid with ghost sides per axis" begin
        domain_size = (1.0, 2.0)
        for arch in (QTTArchitecture((3, 4)), TTArchitecture((8, 16)))
            dims = variable_dims(arch)
            for g1 in 0:2, g2 in 0:2
                grid = interior_grid(arch, domain_size, (g1, g2))
                # n stored nodes and g ghost sides: h = L / (n - 1 + g)
                @test all(grid.dx .≈ domain_size ./ (dims .- 1 .+ (g1, g2)))
                @test all(grid.box_size .≈ grid.dx .* (dims .- 1))
            end
            # Two ghost sides per axis is the two-argument method
            @test interior_grid(arch, domain_size, (2, 2)).dx == interior_grid(arch, domain_size).dx
            @test_throws ArgumentError interior_grid(arch, domain_size, (3, 0))
            @test_throws ArgumentError interior_grid(arch, domain_size, (1, -1))
        end
    end

    @testset "periodic axes" begin
        for arch in (QTTArchitecture((3, 4)), TTArchitecture((8, 16)))
            dims = variable_dims(arch)
            L = (1.0, 2.0)
            grid = Grid(arch, L; periodic=(false, true))
            @test grid.periodic == (false, true)
            @test grid.dx[1] ≈ L[1] / (dims[1] - 1) # unchanged on the non-periodic axis
            @test grid.dx[2] ≈ L[2] / dims[2] # h = L/n: the node at L is the image of the node at 0
            @test grid.points[2][end] ≈ L[2] - grid.dx[2]
            @test length(grid.points[2]) == dims[2]
            @test grid.box_size[2] ≈ L[2] - grid.dx[2]
            # The default grid is bitwise unchanged
            default = Grid(arch, L)
            @test default.periodic == (false, false)
            @test default.box_size === L && default.points == Grid(arch, L; periodic=(false, false)).points
            # interior_grid with one ghost side gives the same spacing, and stores the flag
            igrid = interior_grid(arch, L, (2, 1); periodic=(false, true))
            @test igrid.periodic == (false, true)
            @test igrid.dx[2] ≈ grid.dx[2]
            @test_throws ArgumentError interior_grid(arch, L, (2, 2); periodic=(false, true))
        end
        @test Grid(TTArchitecture((10,)), 1.0; periodic=true).dx == (0.1,)
        @test Grid(QTTArchitecture((3,)), 1.0).periodic == (false,)

        periodic = BoundaryCond1d(PeriodicBC(), PeriodicBC())
        wall = BoundaryCond1d(DirichletBC(0.0), NeumannBC(0.0))
        @test DifferenTT.periodic_axes(BoundaryConditions(wall, periodic)) == (false, true)
        @test DifferenTT.periodic_axes(BoundaryConditions(periodic)) == (true,)

        # The wrapped second derivative of sin(2πx/L) is consistent only when the grid has period L
        n, L = 32, 1.0
        arch = TTArchitecture((n,))
        D2 = DifferenTT.onedim_differential_operator(2, arch, 1, periodic; stencil_order=8)
        function d2_error(grid)
            u = sin.(2π .* grid.points[1] ./ L)
            return maximum(abs, D2 * u ./ grid.dx[1]^2 .+ (2π / L)^2 .* u) / (2π / L)^2
        end
        @test d2_error(Grid(arch, L; periodic=true)) < 1e-8
        @test d2_error(Grid(arch, L)) > 1e-2 # period L + h
    end
end

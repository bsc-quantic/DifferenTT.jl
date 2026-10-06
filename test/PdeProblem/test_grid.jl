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
end

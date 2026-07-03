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
end

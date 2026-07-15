using DifferenTT
using ITensors
using ITensorMPS
using Test
using TensorCrossInterpolation

@testset "tci.jl" begin
    # 1. Setup Grids and Architectures
    box_size = (1.0, 2.0)
    
    # QTT Architecture (e.g. 2 bits per dimension) -> dims = (4, 4)
    qtt_arch = QTTArchitecture((2, 2))
    qtt_grid = Grid(qtt_arch, box_size)
    
    # TT Architecture (dim = 4 per dimension)
    tt_arch = TTArchitecture((4, 4))
    tt_grid = Grid(tt_arch, box_size)

    @testset "point_position" begin
        # dx = box_size ./ (dims .- 1) => (1.0/3, 2.0/3)
        
        point1 = (0.0, 0.0)
        @test DifferenTT.point_position(point1, qtt_grid) == [0, 0]
        
        point2 = (1/3, 4/3)
        @test DifferenTT.point_position(point2, qtt_grid) == [1, 2]

        point3 = (1.0, 2.0)
        @test DifferenTT.point_position(point3, qtt_grid) == [3, 3]
        
        # Test clamping out of bounds
        point_out = (1.5, -1.0)
        @test DifferenTT.point_position(point_out, qtt_grid) == [3, 0]
    end

    @testset "point_sequence and sequence_to_point (QTT)" begin
        for point in [(0.0, 0.0), (1/3, 4/3), (1.0, 2.0)]
            seq = DifferenTT.point_sequence(point, qtt_grid, qtt_arch)
            recovered_point = DifferenTT.sequence_to_point(seq, qtt_grid, qtt_arch)
            
            @test recovered_point[1] ≈ point[1]
            @test recovered_point[2] ≈ point[2]
        end
        
        # Test interleaved mapping
        qtt_arch_int = QTTArchitecture((2, 2); interleaved=true)
        for point in [(1/3, 4/3)]
            seq = DifferenTT.point_sequence(point, qtt_grid, qtt_arch_int)
            recovered_point = DifferenTT.sequence_to_point(seq, qtt_grid, qtt_arch_int)
            @test recovered_point[1] ≈ point[1]
            @test recovered_point[2] ≈ point[2]
        end
    end

    @testset "point_sequence and sequence_to_point (TT)" begin
        for point in [(0.0, 0.0), (1/3, 4/3), (1.0, 2.0)]
            seq = DifferenTT.point_sequence(point, tt_grid, tt_arch)
            recovered_point = DifferenTT.sequence_to_point(seq, tt_grid, tt_arch)
            
            @test recovered_point[1] ≈ point[1]
            @test recovered_point[2] ≈ point[2]
        end
    end

    @testset "TCI Methods (crossinterpolate2, crossinterpolate1, optfirstpivot)" begin
        # Smooth function
        func(x) = sin(x[1]) * cos(x[2])
        
        arch = TTArchitecture((8, 8))
        grid = Grid(arch, (pi, pi))
        
        pivot = DifferenTT.optfirstpivot(Float64, func, grid, arch; iters=10)
        @test length(pivot) == length(arch.site_map)
        @test all(1 .<= pivot .<= 8)
        
        tt2, err2 = DifferenTT.crossinterpolate2(Float64, func, grid, arch; initialpivots=[pivot], maxdim=10, tolerance=1e-2)
        mps2 = MPS(tt2)
        @test mps2 isa MPS
        @test length(mps2) == length(arch.site_map)
        
        tt1, err1 = DifferenTT.crossinterpolate1(Float64, func, grid, arch; initialpivots=[pivot], tolerance=1e-2)
        mps1 = MPS(tt1)
        @test mps1 isa MPS
        @test length(mps1) == length(arch.site_map)
    end

    @testset "sample and SampleCache" begin
        func(x) = x[1] + x[2]
        arch = TTArchitecture((4, 4))
        grid = Grid(arch, (1.0, 1.0))
        pivot = DifferenTT.optfirstpivot(Float64, func, grid, arch; iters=10)
        tt, _ = DifferenTT.crossinterpolate2(Float64, func, grid, arch; initialpivots=[pivot], maxdim=10, tolerance=1e-5)
        mps = MPS(tt)
        
        point = (1.0/3, 2.0/3)
        val = DifferenTT.sample(mps, point, grid, arch)
        @test val ≈ func(point) atol=1e-3
        
        cache = DifferenTT.SampleCache(2, Float64)
        @test !haskey(cache, point)
        val_cached = DifferenTT.sample(mps, point, grid, arch, cache)
        @test val_cached ≈ val
        @test haskey(cache, point)
        @test cache[point] == val_cached
        
        cache[(0.0, 0.0)] = 0.0
        @test cache[(0.0, 0.0)] == 0.0
    end

    @testset "integrate" begin
        s = siteinds(2, 2)
        A = ITensor(s[1])
        A[s[1] => 1] = 1.0
        A[s[1] => 2] = 2.0
        B = ITensor(s[2])
        B[s[2] => 1] = 3.0
        B[s[2] => 2] = 4.0
        mps_simple = MPS([A, B])
        
        # Integral simply sums all index entries
        # (1+2)*(3+4) = 3 * 7 = 21
        @test DifferenTT.integrate(mps_simple) ≈ 21.0
    end

    @testset "samples" begin
        func(x) = x[1] * x[2]
        arch = TTArchitecture((4, 4))
        grid = Grid(arch, (1.0, 1.0))
        pivot = DifferenTT.optfirstpivot(Float64, func, grid, arch; iters=10)
        tt, _ = DifferenTT.crossinterpolate2(Float64, func, grid, arch; initialpivots=[pivot], maxdim=10, tolerance=1e-5)
        mps = MPS(tt)
        
        ranges, data = DifferenTT.samples(mps, grid, arch, (3, 3))
        @test length(ranges) == 2
        @test size(data) == (3, 3)
        @test data[1, 1] ≈ func((ranges[1][1], ranges[2][1])) atol=1e-3
        @test data[2, 3] ≈ func((ranges[1][2], ranges[2][3])) atol=1e-3
        
        # Same npoints on all dims
        ranges2, data2 = DifferenTT.samples(mps, grid, arch, 3)
        @test ranges == ranges2
        @test data == data2
    end
end

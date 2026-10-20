using DifferenTT
using ITensors
using ITensorMPS
using Test
using TensorCrossInterpolation
using LinearAlgebra
include(joinpath(@__DIR__, "..", "testutils.jl"))
using .TestUtils

# All the multi-indices (0-based, one value per site) of an architecture, site 1 the most significant, i.e. in the
# order of the entries of contract_mps
function all_sequences(arch)
    d = Tuple(site_dims(arch))
    return [collect(Tuple(c)) .- 1 for c in vec(permutedims(collect(CartesianIndices(d)), length(d):-1:1))]
end

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
        # A function that vanishes on every multi-index tried has no usable pivot (was an ErrorException)
        @test_throws ArgumentError DifferenTT.optfirstpivot(Float64, x -> 0.0, grid, arch; iters=10)

        tt2, err2 = DifferenTT.crossinterpolate2(
            Float64, func, grid, arch; initialpivots=[pivot], maxbonddim=10, tolerance=1e-2
        )
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
        tt, _ = DifferenTT.crossinterpolate2(
            Float64, func, grid, arch; initialpivots=[pivot], maxbonddim=10, tolerance=1e-5
        )
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
        A[s[1]=>1] = 1.0
        A[s[1]=>2] = 2.0
        B = ITensor(s[2])
        B[s[2]=>1] = 3.0
        B[s[2]=>2] = 4.0
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
        tt, _ = DifferenTT.crossinterpolate2(
            Float64, func, grid, arch; initialpivots=[pivot], maxbonddim=10, tolerance=1e-5
        )
        mps = MPS(tt)

        ranges, data = DifferenTT.samples(mps, grid, arch, (4, 4))
        @test length(ranges) == 2
        @test size(data) == (4, 4)
        @test data[1, 1] ≈ func((ranges[1][1], ranges[2][1])) atol=1e-3
        @test data[2, 3] ≈ func((ranges[1][2], ranges[2][3])) atol=1e-3

        # Same npoints on all dims
        ranges2, data2 = DifferenTT.samples(mps, grid, arch, 4)
        @test ranges == ranges2
        @test data == data2
    end

    @testset "point_sequence and sequence_to_point are inverse bijections for every layout" begin
        layouts = [
            QTTArchitecture((2, 3)),
            QTTArchitecture((2, 3); interleaved=true),
            QTTArchitecture((2, 3), Dict(1 => 2, 2 => 1)),
            QTTArchitecture((-2, 3); interleaved=true),
            QTTArchitecture((1, -2), Dict(1 => 2, 2 => 1)),
            TTArchitecture((3, 5)),
            TTArchitecture((3, 5), Dict(1 => 2, 2 => 1)),
        ]
        for arch in layouts
            grid = Grid(arch, (1.0, 2.0))
            n = variable_dims(arch)
            seqs = all_sequences(arch)
            @test length(seqs) == prod(n)
            points = [DifferenTT.sequence_to_point(q, grid, arch) for q in seqs]
            # every node of the grid exactly once
            @test sort(points) == sort(vec([(i * grid.dx[1], j * grid.dx[2]) for i in 0:(n[1]-1), j in 0:(n[2]-1)]))
            @test all(DifferenTT.point_sequence(p, grid, arch) == q for (p, q) in zip(points, seqs))
            # a point between nodes is snapped to the closest one
            p = points[end÷2]
            @test DifferenTT.point_sequence(p .+ 0.3 .* grid.dx, grid, arch) == seqs[end÷2]
        end
        # the bits of a reversed variable are least significant first: node 1 of 8 is the sequence (1, 0, 0)
        arch = QTTArchitecture((-3,))
        grid = Grid(arch, (7.0,))
        @test DifferenTT.point_sequence((1.0,), grid, arch) == [1, 0, 0]
        @test DifferenTT.sequence_to_point([1, 0, 0], grid, arch) == (1.0,)
        @test DifferenTT.point_sequence((1.0,), grid, QTTArchitecture((3,))) == [0, 0, 1]
    end

    @testset "sample, samples and integrate against the dense vector" begin
        f(x) = 1 + x[1] - 2 * x[2]^2
        for arch in (TTArchitecture((3, 4), Dict(1 => 2, 2 => 1)), QTTArchitecture((2, -1); interleaved=true))
            grid = Grid(arch, (1.0, 2.0))
            seqs = all_sequences(arch)
            v = [f(DifferenTT.sequence_to_point(q, grid, arch)) for q in seqs]
            mps = DifferenTT.vec2mps(v, siteinds(arch))
            @test all(
                DifferenTT.sample(mps, DifferenTT.sequence_to_point(q, grid, arch), grid, arch) ≈ v[k] for
                (k, q) in enumerate(seqs)
            )
            @test DifferenTT.integrate(mps) ≈ sum(v)

            ranges, data = DifferenTT.samples(mps, grid, arch, (3, 5))
            @test size(data) == (3, 5)
            @test ranges[1] == range(0.0, 1.0; length=3) && ranges[2] == range(0.0, 2.0; length=5)
            # every value is that of the node closest to the sample point
            snap(x) = DifferenTT.sequence_to_point(DifferenTT.point_sequence(x, grid, arch), grid, arch)
            @test all(data[i, j] ≈ f(snap((ranges[1][i], ranges[2][j]))) for i in 1:3, j in 1:5)
            @test_throws DimensionMismatch DifferenTT.sample(mps, (0.0, 0.0), grid, QTTArchitecture((2, 2)))
        end
        # integrate on unequal site dimensions
        x = randn(rng(), 24)
        @test DifferenTT.integrate(DifferenTT.vec2mps(x, DifferenTT.site_indices([2, 3, 4]))) ≈ sum(x)
    end

    @testset "crossinterpolate1/2 against the dense function" begin
        f(x) = sin(x[1]) * cos(x[2]) + x[1] * x[2]
        for arch in (TTArchitecture((5, 4), Dict(1 => 2, 2 => 1)), QTTArchitecture((2, 3); interleaved=true))
            grid = Grid(arch, (2.0, 1.0))
            v = [f(DifferenTT.sequence_to_point(q, grid, arch)) for q in all_sequences(arch)]
            pivot = DifferenTT.optfirstpivot(Float64, f, grid, arch; iters=50, rng=rng())
            tt2, _ = DifferenTT.crossinterpolate2(Float64, f, grid, arch; initialpivots=[pivot], tolerance=1e-12)
            @test todense(MPS(tt2)) ≈ v rtol = 1e-10
            # one-site TCI, with a second pivot passed as additionalpivots, against TensorCrossInterpolation called
            # directly on the multi-indices. (TCI1 itself is not accurate here, even for a rank-1 function: see the
            # WP3a report.)
            pivots = [pivot, ones(Int, total_sites(arch))]
            tt1, ranks1, errs1 = DifferenTT.crossinterpolate1(
                Float64, f, grid, arch; initialpivots=pivots, tolerance=1e-12
            )
            g(i) = f(DifferenTT.sequence_to_point(i .- 1, grid, arch))
            ref1, ranks_ref, errs_ref = TensorCrossInterpolation.crossinterpolate1(
                Float64, g, collect(site_dims(arch)), pivots[1]; tolerance=1e-12, additionalpivots=pivots[2:2]
            )
            @test ranks1 == ranks_ref
            @test isequal(errs1, errs_ref)  # TCI1 reports NaN errors on some iterations
            @test todense(MPS(tt1)) ≈ todense(MPS(ref1))
            for ci in (DifferenTT.crossinterpolate1, DifferenTT.crossinterpolate2)
                @test_throws ArgumentError ci(Float64, f, grid, arch; initialpivots=Vector{Int}[])
                @test_throws DimensionMismatch ci(Float64, f, grid, arch; initialpivots=[[1]])
            end
        end
    end

    @testset "optfirstpivot and the pivot helpers" begin
        @test DifferenTT._tci_center_pivot([5, 4, 2, 1]) == [3, 2, 1, 1]
        r = rng()
        draws = [DifferenTT._tci_random_pivot([5, 2, 3]; rng=r) for _ in 1:200]
        @test all(2 <= p[1] <= 4 && 1 <= p[2] <= 2 && p[3] == 2 for p in draws)
        @test Set(p[1] for p in draws) == Set(2:4)
        @test DifferenTT._tci_random_pivot([5, 2, 3]; rng=rng(7)) == DifferenTT._tci_random_pivot([5, 2, 3]; rng=rng(7))

        # a function with its largest magnitude at one interior node is found by the random search
        arch = TTArchitecture((5, 5))
        grid = Grid(arch, (4.0, 4.0))
        g(x) = 1 / (1 + (x[1] - 1)^2 + (x[2] - 3)^2)
        @test DifferenTT.optfirstpivot(Float64, g, grid, arch; iters=100, rng=rng()) == [2, 4]
        # with no draw the center is returned; the center is the origin of a QTT layout
        @test DifferenTT.optfirstpivot(Float64, g, grid, arch; iters=0) == [3, 3]
        q = QTTArchitecture((2, 2))
        @test DifferenTT.optfirstpivot(Float64, x -> 1.0, Grid(q, (1.0, 1.0)), q; iters=0) == [1, 1, 1, 1]
    end
end

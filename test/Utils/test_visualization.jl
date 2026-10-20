ENV["GKSwstype"] = "100" # headless GR backend, set before Plots is first used

using Test
using DifferenTT
using Plots
using ITensors
using ITensorMPS

include(joinpath(@__DIR__, "..", "testutils.jl"))
using .TestUtils

# A TT field with n nodes along each of N axes, on the box (1, 2, …, N), from a function of the node coordinates.
# Returns the MPS (one site per axis), the dense array indexed [i₁, …, i_N], the architecture and the grid.
function tt_field(f, N, n, sites=nothing)
    arch = TTArchitecture(ntuple(_ -> n, N))
    grid = Grid(arch, ntuple(Float64, N))
    data = [f(p...) for p in Iterators.product(grid.points...)]
    sites = isnothing(sites) ? siteinds(arch) : sites # fields that are combined must share their site indices
    return MPS(ITensor(data, sites...), sites), data, arch, grid
end

# The matrix of the heatmap that is the first series of a plot
heat(p) = Matrix(p.series_list[1][:z].surf)

# The vectors (u, v) that the quiver series of a plot draws
quiver_vectors(p) = p.series_list[2][:quiver]

@testset "Visualization" begin
    f(x...) = sum(10.0^(i - 1) * xi for (i, xi) in enumerate(x)) # x₁ + 10 x₂ + 100 x₃ + …, every axis distinguishable

    @testset "visualize_scalar, N = 1: scatter of the samples" begin
        mps, data, arch, grid = tt_field(x -> x^2, 1, 8)
        p = visualize_scalar(mps, grid, arch; npoints=8)
        @test p isa Plots.Plot
        s = p.series_list[1]
        @test collect(s[:x]) ≈ grid.points[1]
        @test collect(s[:y]) ≈ data
        @test p[1][:title] == "Scalar Field"
    end

    @testset "visualize_scalar, N = 2: heatmap of the samples" begin
        mps, data, arch, grid = tt_field(f, 2, 6)
        p = visualize_scalar(mps, grid, arch; npoints=6)
        @test p isa Plots.Plot
        @test collect(p.series_list[1][:x]) ≈ grid.points[1]
        @test collect(p.series_list[1][:y]) ≈ grid.points[2]
        @test heat(p) ≈ data' # the rows of the heatmap are the y samples
        @test (p[1][:title], p[1][:xaxis][:guide], p[1][:yaxis][:guide]) == ("Scalar Field", "x", "y")

        # npoints coarser than the grid samples the nearest nodes
        coarse = visualize_scalar(mps, grid, arch; npoints=3)
        @test size(heat(coarse)) == (3, 3)
        # QTT, whatever the layout: the heatmap shows the samples of the MPS
        qarch = QTTArchitecture((3, 3); interleaved=true)
        qgrid = Grid(qarch, (1.0, 2.0))
        qmps = random_mps(rng(3), siteinds(qarch); linkdims=2)
        pq = visualize_scalar(qmps, qgrid, qarch; npoints=8)
        dense = [
            DifferenTT.sample(qmps, ((i - 1) * qgrid.dx[1], (j - 1) * qgrid.dx[2]), qgrid, qarch) for i in 1:8, j in 1:8
        ]
        @test heat(pq) ≈ dense'
    end

    @testset "visualize_scalar, N ≥ 3: slices" begin
        mps, data, arch, grid = tt_field(f, 3, 5)
        ranges = [range(0.0, grid.box_size[i]; length=5) for i in 1:3]

        # Default: the first two axes, every other axis at the middle sample ⌈n/2⌉ = 3
        p = visualize_scalar(mps, grid, arch; npoints=5)
        @test heat(p) ≈ data[:, :, 3]'
        @test p[1][:title] == "Scalar Field (x3 ≈ mid)"
        @test collect(p.series_list[1][:x]) ≈ ranges[1]
        @test collect(p.series_list[1][:y]) ≈ ranges[2]

        # Other pairs of axes
        p13 = visualize_scalar(mps, grid, arch; npoints=5, slice_dims=(1, 3))
        @test heat(p13) ≈ data[:, 3, :]'
        @test collect(p13.series_list[1][:y]) ≈ ranges[3]
        @test p13[1][:title] == "Scalar Field (x2 ≈ mid)"
        p23 = visualize_scalar(mps, grid, arch; npoints=5, slice_dims=(2, 3))
        @test heat(p23) ≈ data[3, :, :]'

        # slice_vals sets the other coordinates: the nearest sample along x₃ ∈ {0, 0.75, 1.5, 2.25, 3}
        pv = visualize_scalar(mps, grid, arch; npoints=5, slice_vals=(2.4,))
        @test heat(pv) ≈ data[:, :, 4]'
        @test pv[1][:title] == "Scalar Field (x3 ≈ 2.25)"
        # Fewer values than free axes: the missing ones fall back to the middle sample
        mps4, data4, arch4, grid4 = tt_field(f, 4, 4)
        p4 = visualize_scalar(mps4, grid4, arch4; npoints=4, slice_vals=(0.0,))
        @test heat(p4) ≈ data4[:, :, 1, 2]' # x3 at the first sample, x4 at the middle one (⌈4/2⌉ = 2)
        @test p4[1][:title] == "Scalar Field (x3 ≈ 0.0, x4 ≈ mid)"
        p4b = visualize_scalar(mps4, grid4, arch4; npoints=4, slice_dims=(2, 3), slice_vals=(4.0, 0.0))
        @test heat(p4b) ≈ data4[end, :, :, 1]'
    end

    @testset "The middle sample of an odd number of points (regression)" begin
        # div(npoints, 2) picked the sample before the middle (index 2 of 5) for odd npoints
        mps, data, arch, grid = tt_field(f, 3, 5)
        @test heat(visualize_scalar(mps, grid, arch; npoints=5)) ≈ data[:, :, 3]'
    end

    @testset "ScalarField methods: the last state, or the state `step`" begin
        mps1, data1, arch, grid = tt_field(f, 2, 6)
        mps2, data2, _, _ = tt_field((x, y) -> -f(x, y), 2, 6, siteinds(mps1))
        bc = BoundaryCond1d(PeriodicBC(), PeriodicBC())
        field = ScalarField(mps1, arch, BoundaryConditions(bc, bc))
        push!(field, mps2)
        @test heat(visualize_scalar(field, grid; npoints=6)) ≈ data2' # the last state
        @test heat(visualize_scalar(field, 1, grid; npoints=6)) ≈ data1'
        @test heat(visualize_scalar(field, 2, grid; npoints=6)) ≈ data2'
        @test_throws BoundsError visualize_scalar(field, 3, grid; npoints=6)
    end

    @testset "visualize_vector, N = 2: magnitude and quiver" begin
        u, udata, arch, grid = tt_field((x, y) -> x + 3y, 2, 6)
        v, vdata, _, _ = tt_field((x, y) -> -2x + y, 2, 6)
        p = visualize_vector((u, v), grid, arch; npoints=6, quiver_npoints=6, arrow_scale=0.5)
        @test p isa Plots.Plot
        @test length(p.series_list) == 2
        @test heat(p) ≈ sqrt.(udata .^ 2 .+ vdata .^ 2)'
        @test p[1][:title] == "Velocity Magnitude & Streamlines"
        # The quiver has one arrow per sample, scaled by arrow_scale
        qu, qv = quiver_vectors(p)
        @test qu ≈ 0.5 .* vec(udata)
        @test qv ≈ 0.5 .* vec(vdata)
        # quiver_npoints sets the number of arrows independently
        p3 = visualize_vector((u, v), grid, arch; npoints=6, quiver_npoints=3)
        @test length(quiver_vectors(p3)[1]) == 9
        @test size(heat(p3)) == (6, 6)
    end

    @testset "visualize_vector, N ≥ 3: slices of the magnitude" begin
        comps = [tt_field((x...) -> f(x...) * c, 3, 5) for c in (1.0, 2.0, 3.0)]
        arch, grid = comps[1][3], comps[1][4]
        mags = sqrt.(sum(c[2] .^ 2 for c in comps))
        mpss = Tuple(c[1] for c in comps)

        p = visualize_vector(mpss, grid, arch; npoints=5, quiver_npoints=5)
        @test heat(p) ≈ mags[:, :, 3]'
        @test p[1][:title] == "Velocity Magnitude (x3 ≈ mid)"
        # The quiver shows the two components of the slice, at the middle sample of the others
        @test quiver_vectors(p)[1] ≈ vec(comps[1][2][:, :, 3])
        @test quiver_vectors(p)[2] ≈ vec(comps[2][2][:, :, 3])

        p13 = visualize_vector(mpss, grid, arch; npoints=5, quiver_npoints=5, slice_dims=(1, 3))
        @test heat(p13) ≈ mags[:, 3, :]'
        @test quiver_vectors(p13)[1] ≈ vec(comps[1][2][:, 3, :])
        @test quiver_vectors(p13)[2] ≈ vec(comps[3][2][:, 3, :])

        # The documented limitation: slice_vals is ignored for vectors
        pvals = visualize_vector(mpss, grid, arch; npoints=5, quiver_npoints=5, slice_vals=(3.0,))
        @test heat(pvals) ≈ heat(p)
    end

    @testset "visualize_vector, N = 4" begin
        comps = [tt_field((x...) -> f(x...) + c, 4, 4) for c in (0.0, 1.0, 2.0, 3.0)]
        arch, grid = comps[1][3], comps[1][4]
        mags = sqrt.(sum(c[2] .^ 2 for c in comps))
        p = visualize_vector(Tuple(c[1] for c in comps), grid, arch; npoints=4, quiver_npoints=4, slice_dims=(2, 3))
        @test heat(p) ≈ mags[2, :, :, 2]' # axes 1 and 4 at the middle sample ⌈4/2⌉ = 2
        @test p[1][:title] == "Velocity Magnitude (x1 ≈ mid, x4 ≈ mid)"
    end

    @testset "visualize_vorticity, N = 2: rigid rotation" begin
        # u = (-y, x): ω = ∂ₓv - ∂ᵧu = 2 on the interior samples, 0 on the boundary samples
        u, _, arch, grid = tt_field((x, y) -> -y, 2, 6)
        v, _, _, _ = tt_field((x, y) -> x, 2, 6)
        p = visualize_vorticity((u, v), grid, arch; npoints=6, quiver_npoints=6)
        @test p isa Plots.Plot
        expected = zeros(6, 6)
        expected[2:5, 2:5] .= 2.0
        @test heat(p) ≈ expected atol = 1e-10
        @test p[1][:title] == "Vorticity Magnitude & Streamlines"
        # The quiver is the velocity itself
        @test quiver_vectors(p)[1] ≈ vec([-y for x in grid.points[1], y in grid.points[2]])
        @test quiver_vectors(p)[2] ≈ vec([x for x in grid.points[1], y in grid.points[2]])

        # A gradient field has no vorticity: u = (x, y)
        gu, _, _, _ = tt_field((x, y) -> x, 2, 6)
        gv, _, _, _ = tt_field((x, y) -> y, 2, 6)
        @test maximum(abs, heat(visualize_vorticity((gu, gv), grid, arch; npoints=6, quiver_npoints=6))) < 1e-10
        # Shear u = y², v = 0: ω = -2y on the interior (central differences are exact for quadratics)
        su, _, _, _ = tt_field((x, y) -> y^2, 2, 6)
        sv, _, _, _ = tt_field((x, y) -> 0.0, 2, 6)
        shear = heat(visualize_vorticity((su, sv), grid, arch; npoints=6, quiver_npoints=6))
        @test shear[2:5, 2:5] ≈ [-2y for y in grid.points[2][2:5], x in grid.points[1][2:5]] atol = 1e-10
    end

    @testset "visualize_vorticity, N = 3: rotation about z" begin
        # u = (-y, x, 0): ω = (0, 0, 2), |ω| = 2 on the interior, at the middle slice of x₃
        u, _, arch, grid = tt_field((x, y, z) -> -y, 3, 5)
        v, _, _, _ = tt_field((x, y, z) -> x, 3, 5)
        w, _, _, _ = tt_field((x, y, z) -> 0.0, 3, 5)
        p = visualize_vorticity((u, v, w), grid, arch; npoints=5, quiver_npoints=5)
        expected = zeros(5, 5)
        expected[2:4, 2:4] .= 2.0
        @test heat(p) ≈ expected atol = 1e-10
        @test p[1][:title] == "Vorticity Magnitude (slice) & Streamlines"
        # The slice through (x₁, x₃): the rotation does not depend on x₃, so every interior sample has |ω| = 2
        p13 = visualize_vorticity((u, v, w), grid, arch; npoints=5, quiver_npoints=5, slice_dims=(1, 3))
        @test heat(p13) ≈ expected atol = 1e-10
    end

    @testset "VectorField methods: the last state, or the state `step`" begin
        u1, _, arch, grid = tt_field((x, y) -> x + 3y, 2, 6)
        v1, _, _, _ = tt_field((x, y) -> -y, 2, 6, siteinds(u1))
        u2, _, _, _ = tt_field((x, y) -> 2x, 2, 6, siteinds(u1))
        v2, _, _, _ = tt_field((x, y) -> x, 2, 6, siteinds(u1))
        bc = BoundaryCond1d(PeriodicBC(), PeriodicBC())
        bcs = BoundaryConditions(bc, bc)
        field = VectorField((u1, v1), arch, (bcs, bcs))
        push!(field, (u2, v2))

        for visualize in (visualize_vector, visualize_vorticity)
            kwargs = (npoints=6, quiver_npoints=6)
            first_state = visualize(field, 1, grid; kwargs...)
            last_state = visualize(field, 2, grid; kwargs...)
            @test first_state isa Plots.Plot && last_state isa Plots.Plot
            @test heat(visualize(field, grid; kwargs...)) ≈ heat(last_state) # no step: the last state
            @test heat(first_state) ≈ heat(visualize((u1, v1), grid, arch; kwargs...))
            @test heat(last_state) ≈ heat(visualize((u2, v2), grid, arch; kwargs...))
            @test !(heat(first_state) ≈ heat(last_state))
        end
        # The magnitude of the second state is that of (2x, x)
        @test heat(visualize_vector(field, 2, grid; npoints=6, quiver_npoints=6)) ≈
            [sqrt(5) * x for x in grid.points[1], y in grid.points[2]]'
        @test_throws BoundsError visualize_vector(field, 3, grid)
    end

    @testset "N = 1 has no vector or vorticity" begin
        mps, _, arch, grid = tt_field(x -> x, 1, 8)
        @test_throws ArgumentError visualize_vector((mps,), grid, arch)
        @test_throws ArgumentError visualize_vorticity((mps,), grid, arch)
    end

    @testset "Saving: the file is written, in a directory that may not exist" begin
        mps, data, arch, grid = tt_field(f, 2, 6)
        u2, _, _, _ = tt_field((x, y) -> -y, 2, 6, siteinds(mps))
        v2, _, _, _ = tt_field((x, y) -> x, 2, 6, siteinds(mps))
        bc = BoundaryCond1d(PeriodicBC(), PeriodicBC())
        bcs = BoundaryConditions(bc, bc)
        sfield = ScalarField(mps, arch, bcs)
        vfield = VectorField((u2, v2), arch, (bcs, bcs))

        png_signature = UInt8[0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a]
        mktempdir() do dir
            calls = (
                "scalar_mps" => path -> visualize_scalar(mps, grid, arch, path; npoints=6),
                "scalar_field" => path -> visualize_scalar(sfield, grid, path; npoints=6),
                "scalar_step" => path -> visualize_scalar(sfield, 1, grid, path; npoints=6),
                "vector_mps" => path -> visualize_vector((u2, v2), grid, arch, path; npoints=6, quiver_npoints=4),
                "vector_field" => path -> visualize_vector(vfield, grid, path; npoints=6, quiver_npoints=4),
                "vector_step" => path -> visualize_vector(vfield, 1, grid, path; npoints=6, quiver_npoints=4),
                "vorticity_mps" => path -> visualize_vorticity((u2, v2), grid, arch, path; npoints=6, quiver_npoints=4),
                "vorticity_field" => path -> visualize_vorticity(vfield, grid, path; npoints=6, quiver_npoints=4),
                "vorticity_step" => path -> visualize_vorticity(vfield, 1, grid, path; npoints=6, quiver_npoints=4),
            )
            for (name, call) in calls
                path = joinpath(dir, "figures", name, "plot.png")
                @test !isdir(dirname(path))
                p = call(path)
                @test p isa Plots.Plot
                @test isfile(path)
                @test read(path, 8) == png_signature
                @test filesize(path) > 1000
            end
            # The extension selects the format
            svg = joinpath(dir, "figures", "plot.svg")
            visualize_scalar(mps, grid, arch, svg; npoints=6)
            @test startswith(read(svg, String), "<?xml")
            # Without a path, nothing is written
            visualize_scalar(mps, grid, arch; npoints=6)
            @test sort(readdir(dir)) == ["figures"]
        end
    end

    @testset "Plots keywords are passed through" begin
        mps, _, arch, grid = tt_field(f, 2, 6)
        p = visualize_scalar(mps, grid, arch; npoints=6, title="custom", size=(321, 234))
        @test p[1][:title] == "custom"
        @test p.attr[:size] == (321, 234)
    end
end

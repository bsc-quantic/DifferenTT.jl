using DifferenTT
using Test
using ITensors
using ITensorMPS
using LinearAlgebra
using Random

include(joinpath(@__DIR__, "..", "testutils.jl"))
using .TestUtils

@testset "linearsyscorrections.jl" begin
    n = 3
    qtt_arch = QTTArchitecture((n, n))
    grid = Grid(qtt_arch, (1.0, 1.0))

    sites = siteinds(qtt_arch)
    u_mps = random_mps(sites)

    # 1. Periodic BCs -> corrections should be nothing
    bc_periodic = BoundaryCond1d(PeriodicBC(), PeriodicBC())
    bcs_periodic = BoundaryConditions(bc_periodic, bc_periodic)
    sfield_per = ScalarField(u_mps, qtt_arch, bcs_periodic)
    vfield_per = VectorField((sfield_per, sfield_per))

    @testset "Periodic BCs Corrections" begin
        # For Periodic, onedim_boundary_nullifier returns nothing, so divergence_dirichlet_correction might return nothing
        div_corr = DifferenTT.divergence_dirichlet_correction(vfield_per; stencil_order=2)
        @test isnothing(div_corr)

        p_corr = DifferenTT.scalar_dirichlet_correction(sfield_per, grid, 2; stencil_order=2)
        @test isnothing(p_corr)
    end

    # 2. Dirichlet BCs
    bc_dirichlet = BoundaryCond1d(DirichletBC(1.0), DirichletBC(1.0))
    bcs_dirichlet = BoundaryConditions(bc_dirichlet, bc_dirichlet)
    sfield_dir = ScalarField(u_mps, qtt_arch, bcs_dirichlet)
    vfield_dir = VectorField((sfield_dir, sfield_dir))

    @testset "Dirichlet BCs Corrections" begin
        div_corr = DifferenTT.divergence_dirichlet_correction(vfield_dir; stencil_order=2)
        @test div_corr isa MPO

        p_corr_dim1 = DifferenTT.scalar_dirichlet_correction_dim(sfield_dir, grid, 2, 1; stencil_order=2)
        @test p_corr_dim1 isa MPO

        p_corr_dim2 = DifferenTT.scalar_dirichlet_correction_dim(sfield_dir, grid, 2, 2; stencil_order=2)
        @test p_corr_dim2 isa MPO

        p_corr_total = DifferenTT.scalar_dirichlet_correction(sfield_dir, grid, 2; stencil_order=2)
        @test p_corr_total isa MPO
    end

    # 3. Neumann weight W = W_x ⊗ W_y: 1/2 on the Neumann boundary rows of each axis, multiplied at corners
    @testset "Neumann weight and symmetrize_rhs ($label)" for (label, arch) in (
        ("QTT sequential", QTTArchitecture((3, 3))),
        ("QTT reversed", QTTArchitecture((3, -3))),
        ("QTT interleaved", QTTArchitecture((3, 3); interleaved=true)),
        ("TT", TTArchitecture((8, 8))),
    )
        n1, n2 = variable_dims(arch)
        w_grid = Grid(arch, (1.0, 1.0))
        w_sites = siteinds(arch)
        # Dirichlet-Neumann along x, Neumann on both sides along y
        bcs = BoundaryConditions(
            BoundaryCond1d(DirichletBC(0.0), NeumannBC(0.0)), BoundaryCond1d(NeumannBC(0.0), NeumannBC(0.0))
        )
        field = ScalarField(w_sites, arch, bcs)

        wx = ones(n1);
        wx[end] = 0.5
        wy = ones(n2);
        wy[1] = 0.5;
        wy[end] = 0.5
        expected = [wx[i] * wy[j] for i in 1:n1, j in 1:n2]
        sampled(mps) = [
            DifferenTT.sample(mps, ((i - 1) * w_grid.dx[1], (j - 1) * w_grid.dx[2]), w_grid, arch) for
            i in 1:n1, j in 1:n2
        ]

        W = neumann_weight_mpo(field)
        W_dense = DifferenTT.contract_mpo(W)

        # Diagonal, with the weights on the right nodes (0.25 on the corners shared by two Neumann faces)
        @test W_dense ≈ Diagonal(diag(W_dense))
        weights = apply(W, ones_mps(w_sites, 1))
        @test sampled(weights) ≈ expected
        @test diag(W_dense) ≈ DifferenTT.contract_mps(weights)
        arch isa TTArchitecture && @test maxlinkdim(W) == 1

        # symmetrize_rhs applies the same diagonal weight to any right-hand side
        @test sampled(symmetrize_rhs(field, ones_mps(w_sites, 1))) ≈ expected
        Random.seed!(7)
        rhs = random_mps(w_sites; linkdims=2)
        @test DifferenTT.contract_mps(symmetrize_rhs(field, rhs)) ≈ diag(W_dense) .* DifferenTT.contract_mps(rhs)

        # Without Neumann faces there is no weight and the right-hand side is returned untouched
        @test isnothing(neumann_weight_mpo(sfield_per))
        @test symmetrize_rhs(sfield_per, u_mps) === u_mps
    end

    # 4. Face profiles and the Dirichlet lift of a single face (ghost value g on one face)
    face_archs = (
        ("QTT sequential", QTTArchitecture((4, 4))),
        ("QTT reversed", QTTArchitecture((4, -4))),
        ("QTT interleaved", QTTArchitecture((4, 4); interleaved=true)),
        ("TT", TTArchitecture((16, 16))),
    )
    dirichlet_walls = BoundaryConditions(
        BoundaryCond1d(DirichletBC(0.0), DirichletBC(0.0)), BoundaryCond1d(DirichletBC(0.0), DirichletBC(0.0))
    )

    @testset "face_profile_mps ($label)" for (label, arch) in face_archs
        dims = variable_dims(arch)
        f_grid = Grid(arch, (1.0, 1.3))
        field = ScalarField(siteinds(arch), arch, dirichlet_walls)
        sampled(mps) = [
            DifferenTT.sample(mps, ((i - 1) * f_grid.dx[1], (j - 1) * f_grid.dx[2]), f_grid, arch) for
            i in 1:dims[1], j in 1:dims[2]
        ]

        Random.seed!(8)
        for axis in 1:2
            values = randn(dims[3-axis])
            # Constant along `axis`, equal to the profile along the other axis
            expected = axis == 1 ? repeat(values', dims[1]) : repeat(values, 1, dims[2])
            @test sampled(DifferenTT.face_profile_mps(field, axis, values)) ≈ expected
            @test sampled(DifferenTT.face_profile_mps(field, axis, 2.5)) ≈ fill(2.5, dims...)
            @test_throws ArgumentError DifferenTT.face_profile_mps(field, axis, randn(dims[3-axis] + 1))
        end
    end

    @testset "face_dirichlet_lift ($label)" for (label, arch) in face_archs
        dims = variable_dims(arch)
        f_grid = Grid(arch, (1.0, 1.3))
        field = ScalarField(siteinds(arch), arch, dirichlet_walls)
        sampled(mps) = [
            DifferenTT.sample(mps, ((i - 1) * f_grid.dx[1], (j - 1) * f_grid.dx[2]), f_grid, arch) for
            i in 1:dims[1], j in 1:dims[2]
        ]
        ops = DifferentialOpsScalar(f_grid, field; stencil_order=2)

        Random.seed!(9)
        f = random_mps(field.usites; linkdims=3)
        F = sampled(f)
        for axis in 1:2, side in (:left, :right), k in (1, 2)
            h = f_grid.dx[axis]
            g = randn(dims[3-axis])
            lift = face_dirichlet_lift(field, f_grid, k, axis, side, g)

            # (a) Only the row next to the face is nonzero: ∓g/(2h) for k = 1, g/h² for k = 2
            expected = zeros(dims...)
            value = k == 1 ? (side == :left ? -1 : 1) .* g ./ (2h) : g ./ h^2
            row = side == :left ? 1 : dims[axis]
            axis == 1 ? (expected[row, :] .= value) : (expected[:, row] .= value)
            @test sampled(lift) ≈ expected

            # (b) Truncated stencil + lift = full stencil with the ghost row g on that face (0 on the others)
            P = zeros(dims[1] + 2, dims[2] + 2)
            P[2:(end-1), 2:(end-1)] .= F
            ghost = side == :left ? 1 : dims[axis] + 2
            axis == 1 ? (P[ghost, 2:(end-1)] .= g) : (P[2:(end-1), ghost] .= g)
            lo, hi = if axis == 1
                (P[1:(end-2), 2:(end-1)], P[3:end, 2:(end-1)])
            else
                (P[2:(end-1), 1:(end-2)], P[2:(end-1), 3:end])
            end
            dense = k == 1 ? (hi .- lo) ./ (2h) : (hi .- 2F .+ lo) ./ h^2
            op = k == 1 ? ops.d_ops[axis] : ops.d2_ops[axis]
            @test sampled(+(apply(op, f), lift; cutoff=1e-15)) ≈ dense
        end

        # No lift for a zero or missing ghost value
        @test isnothing(face_dirichlet_lift(field, f_grid, 2, 1, :left, 0.0))
        @test isnothing(face_dirichlet_lift(field, f_grid, 2, 1, :left, zeros(dims[2])))
        @test isnothing(face_dirichlet_lift(field, f_grid, 2, 1, :left, nothing))
        @test_throws ArgumentError face_dirichlet_lift(field, f_grid, 2, 1, :top, 1.0)
    end

    @testset "remove_mean" begin
        for arch in (QTTArchitecture((3, 3)), TTArchitecture((8, 8)))
            Random.seed!(10)
            rhs = random_mps(siteinds(arch); linkdims=2)
            centred = DifferenTT.remove_mean(rhs)
            @test abs(DifferenTT.integrate(centred)) < 1e-12 * norm(rhs)
            # Only the constant mode changes
            shift = DifferenTT.contract_mps(rhs) .- DifferenTT.contract_mps(centred)
            @test shift ≈ fill(DifferenTT.integrate(rhs) / 64, 64)
        end
    end

    # 5. The Dirichlet lift of a whole field: truncated operator + lift = full stencil on the ghost-extended array
    # Ghost-extended dense reference: the stencil of order `order` along `axis`, spacing h, applied to the array F whose
    # ghost nodes hold g_left and g_right (a number for a Dirichlet side, :mirror for a Neumann side)
    function ghost_stencil(F, order, axis, q, h, g_left, g_right)
        coeffs = DifferenTT.derivative_coefficients(order; stencil_order=q)
        n = size(F, axis)
        out = zeros(size(F))
        for idx in CartesianIndices(F), (k, c) in coeffs
            j = idx[axis] + k
            ghost = j < 1 ? g_left : (j > n ? g_right : nothing)
            value = if isnothing(ghost)
                F[CartesianIndex(Base.setindex(Tuple(idx), j, axis))]
            elseif ghost === :mirror
                m = j < 1 ? 2 - j : 2n - j
                F[CartesianIndex(Base.setindex(Tuple(idx), m, axis))]
            else
                ghost
            end
            out[idx] += c * value
        end
        return out ./ h^order
    end

    @testset "scalar_dirichlet_correction(_dim): truncated operator + lift = full stencil ($label, q=$q)" for (
            label, arch
        ) in (
            ("QTT sequential", QTTArchitecture((4, 4))),
            ("QTT interleaved", QTTArchitecture((4, 4); interleaved=true)),
            ("QTT reversed", QTTArchitecture((4, -4))),
            ("TT", TTArchitecture((16, 16))),
        ),
        q in (2, 8)

        dims = variable_dims(arch)
        box = (1.0, 1.3)
        g_grid = Grid(arch, box)
        sites = siteinds(arch)
        values_of(mps) = [
            DifferenTT.sample(mps, ((i - 1) * g_grid.dx[1], (j - 1) * g_grid.dx[2]), g_grid, arch) for
            i in 1:dims[1], j in 1:dims[2]
        ]
        ones_field = ones_mps(sites, 1)
        p = random_mps(rng(21), sites; linkdims=3)
        P = values_of(p)

        # Different nonzero values on the four faces; one case with a Neumann side
        cases = (
            (
                "four Dirichlet faces",
                BoundaryConditions(
                    BoundaryCond1d(DirichletBC(0.7), DirichletBC(-1.3)),
                    BoundaryCond1d(DirichletBC(2.0), DirichletBC(0.5)),
                ),
                (0.7, -1.3, 2.0, 0.5),
            ),
            (
                "one zero face",
                BoundaryConditions(
                    BoundaryCond1d(DirichletBC(0.0), DirichletBC(1.1)),
                    BoundaryCond1d(DirichletBC(-0.4), DirichletBC(0.0)),
                ),
                (0.0, 1.1, -0.4, 0.0),
            ),
            (
                "Dirichlet-Neumann x, Neumann-Dirichlet y",
                BoundaryConditions(
                    BoundaryCond1d(DirichletBC(0.9), NeumannBC(0.0)), BoundaryCond1d(NeumannBC(0.0), DirichletBC(-0.6))
                ),
                (0.9, :mirror, :mirror, -0.6),
            ),
        )
        for (case, bcs, ghosts) in cases
            field = ScalarField(sites, arch, bcs)
            ops = DifferentialOpsScalar(g_grid, field; stencil_order=q)
            lifted_axes = Int[]
            for axis in 1:2
                gl, gr = ghosts[2axis-1], ghosts[2axis]
                for order in (1, 2)
                    op = order == 1 ? ops.d_ops[axis] : ops.d2_ops[axis]
                    C = DifferenTT.scalar_dirichlet_correction_dim(field, g_grid, order, axis; stencil_order=q)
                    nonzero = any(g -> g isa Number && !iszero(g), (gl, gr))
                    @test isnothing(C) == !nonzero
                    lift = nonzero ? values_of(apply(C, ones_field)) : zeros(dims...)
                    reference = ghost_stencil(P, order, axis, q, g_grid.dx[axis], gl, gr)
                    @test values_of(apply(op, p)) .+ lift ≈ reference
                end
                any(g -> g isa Number && !iszero(g), (gl, gr)) && push!(lifted_axes, axis)
            end
            # Summed over the axes
            for order in (1, 2)
                C = DifferenTT.scalar_dirichlet_correction(field, g_grid, order; stencil_order=q)
                @test isnothing(C) == isempty(lifted_axes)
                total = zeros(dims...)
                reference = zeros(dims...)
                for axis in 1:2
                    op = order == 1 ? ops.d_ops[axis] : ops.d2_ops[axis]
                    total .+= values_of(apply(op, p))
                    reference .+= ghost_stencil(P, order, axis, q, g_grid.dx[axis], ghosts[2axis-1], ghosts[2axis])
                end
                isnothing(C) || (total .+= values_of(apply(C, ones_field)))
                @test total ≈ reference
            end
        end

        # All values zero: nothing to lift
        zero_bcs = BoundaryConditions(
            BoundaryCond1d(DirichletBC(0.0), DirichletBC(0.0)), BoundaryCond1d(DirichletBC(0.0), DirichletBC(0.0))
        )
        zero_field = ScalarField(sites, arch, zero_bcs)
        @test isnothing(DifferenTT.scalar_dirichlet_correction(zero_field, g_grid, 2; stencil_order=q))
    end
end

using ITensors
using ITensorMPS
using DifferenTT
using Test
using LinearAlgebra

include(joinpath(@__DIR__, "..", "testutils.jl"))
using .TestUtils

# Helper function to generate expected dense matrix for 1D derivative
function expected_diff_matrix(derivative_order::Int, dim::Int, bcs::BoundaryCond1d; stencil_order::Int=8)
    coeffs = DifferenTT.derivative_coefficients(derivative_order; stencil_order=stencil_order)
    mat = zeros(Float64, dim, dim)
    for i in 1:dim
        for (offset, val) in coeffs
            j = i + offset
            if bcs.left isa PeriodicBC
                # Periodic wrap-around
                if j < 1
                    j += dim
                elseif j > dim
                    j -= dim
                end
                mat[i, j] += val
            else
                if 1 <= j <= dim
                    mat[i, j] += val
                end
            end
        end
    end

    if bcs.left isa NeumannBC
        arch = DifferenTT.TTArchitecture((dim,))
        mat -= DifferenTT.left_neumann_boundary_correction(
            derivative_order, arch, dim, Array; stencil_order=stencil_order
        )
    end
    if bcs.right isa NeumannBC
        arch = DifferenTT.TTArchitecture((dim,))
        mat -= DifferenTT.right_neumann_boundary_correction(
            derivative_order, arch, dim, Array; stencil_order=stencil_order
        )
    end
    return mat
end

# Dense reference by explicit ghost elimination (spacing 1, nodes 1:n): a stencil entry that lands on a ghost node is
# dropped (`:truncate`), moved to the mirror node (`:mirror`) or wrapped around (`:wrap`)
function ghost_operator(order::Int, n::Int, left::Symbol, right::Symbol; stencil_order::Int)
    coeffs = DifferenTT.derivative_coefficients(order; stencil_order=stencil_order)
    D = zeros(n, n)
    for i in 1:n, (k, c) in coeffs
        j = i + k
        if j < 1
            left == :mirror && (D[i, 2-j] += c)
            left == :wrap && (D[i, j+n] += c)
        elseif j > n
            right == :mirror && (D[i, 2n-j] += c)
            right == :wrap && (D[i, j-n] += c)
        else
            D[i, j] += c
        end
    end
    return D
end
side_symbol(bc) = bc isa NeumannBC ? :mirror : (bc isa PeriodicBC ? :wrap : :truncate)

@testset "diffoperators.jl" begin
    @testset "1D Differential Operators" begin
        nsite = 6
        dim = 2^nsite
        qtt_arch = QTTArchitecture((nsite,))
        tt_arch = TTArchitecture((dim,))

        for stencil_order in [2, 8]
            for order in [1, 2]
                for (name, bc1d) in [
                    ("Periodic", BoundaryCond1d(PeriodicBC(), PeriodicBC())),
                    ("Dirichlet", BoundaryCond1d(DirichletBC(0.0), DirichletBC(0.0))),
                    ("Neumann", BoundaryCond1d(NeumannBC(0.0), NeumannBC(0.0))),
                ]
                    @testset "Derivative $order, Stencil $stencil_order, BC $name" begin
                        expected_mat = expected_diff_matrix(order, dim, bc1d; stencil_order=stencil_order)

                        # Test TT Architecture
                        tt_mat = DifferenTT.onedim_differential_operator(
                            order, tt_arch, 1, bc1d, Array; stencil_order=stencil_order
                        )
                        @test tt_mat ≈ expected_mat atol=1e-12

                        # Test QTT Architecture
                        qtt_mpo = DifferenTT.onedim_differential_operator(
                            order, qtt_arch, 1, bc1d, Array; stencil_order=stencil_order
                        )
                        qtt_mat = DifferenTT.contract_mpo(qtt_mpo)
                        @test qtt_mat ≈ expected_mat atol=1e-10

                        # Eye padding tests
                        for n_eyepad in 0:2
                            padded_dim = dim
                            tt_unpadded_dim = dim - n_eyepad
                            tt_expected_unpadded = expected_diff_matrix(
                                order, tt_unpadded_dim, bc1d; stencil_order=stencil_order
                            )
                            expected_padded_tt = zeros(padded_dim, padded_dim)
                            expected_padded_tt[1:tt_unpadded_dim, 1:tt_unpadded_dim] .= tt_expected_unpadded
                            for i in (tt_unpadded_dim+1):padded_dim
                                expected_padded_tt[i, i] = 1.0
                            end

                            tt_padded = DifferenTT.onedim_differential_operator(
                                order, tt_arch, 1, n_eyepad, bc1d, Array; stencil_order=stencil_order
                            )
                            @test tt_padded ≈ expected_padded_tt

                            qtt_unpadded_dim = dim ÷ (2^n_eyepad)
                            qtt_expected_unpadded = expected_diff_matrix(
                                order, qtt_unpadded_dim, bc1d; stencil_order=stencil_order
                            )
                            expected_padded_qtt = kron(
                                qtt_expected_unpadded, Matrix{Float64}(I, 2^n_eyepad, 2^n_eyepad)
                            )
                            qtt_mpo_padded = DifferenTT.onedim_differential_operator(
                                order, qtt_arch, 1, n_eyepad, bc1d, Array; stencil_order=stencil_order
                            )
                            qtt_padded_mat = DifferenTT.contract_mpo(qtt_mpo_padded)
                            @test qtt_padded_mat ≈ expected_padded_qtt atol=1e-10
                        end
                    end
                end
            end
        end
    end

    @testset "Higher Dimensional Operator Containers" begin
        # Test DifferentialOpsScalar and BoundaryNullScalar using a minimal 2D setup
        nsite_x, nsite_y = 4, 4
        dim_x, dim_y = 2^nsite_x, 2^nsite_y
        qtt_arch = QTTArchitecture((nsite_x, nsite_y))
        grid = Grid(qtt_arch, (1.0, 1.0))
        bc1d_x = BoundaryCond1d(PeriodicBC(), PeriodicBC())
        bc1d_y = BoundaryCond1d(DirichletBC(0.0), NeumannBC(0.0))
        bcs = BoundaryConditions(bc1d_x, bc1d_y)

        # We need a ScalarField to create the ops
        sites = [Index(2, "Site,n=$i") for i in 1:(nsite_x+nsite_y)]
        u_init = random_mps(sites)
        sfield = ScalarField(u_init, qtt_arch, bcs)

        diff_ops = DifferenTT.DifferentialOpsScalar(grid, sfield)
        @test diff_ops isa DifferenTT.DifferentialOpsScalar{2}
        @test length(diff_ops.d_ops) == 2
        @test length(diff_ops.d2_ops) == 2

        # Boundary Null Scalar
        bn_scalar = DifferenTT.BoundaryNullScalar(sfield)
        @test bn_scalar isa DifferenTT.BoundaryNullScalar{2}
        # In this setup, y has a Dirichlet condition so u_row_deleter should not be nothing
        @test !isnothing(bn_scalar.u_row_deleter)

        # Test Vector fields diff ops
        vfield = VectorField((sfield, sfield))
        diff_ops_vec = DifferenTT.DifferentialOps(grid, vfield)
        @test diff_ops_vec isa DifferenTT.DifferentialOps{2}
        @test diff_ops_vec[1] isa DifferenTT.DifferentialOpsScalar{2}

        bn_vec = DifferenTT.BoundaryNull(vfield)
        @test bn_vec isa DifferenTT.BoundaryNull{2}
        @test bn_vec[1] isa DifferenTT.BoundaryNullScalar{2}
    end

    # Regression test for B3: the eye-padded DifferentialOpsScalar read the non-existent field grid.lengths. The padded
    # operator must equal the operator of the reduced grid, padded with the identity: in QTT the coarse grid of
    # 2^(n - p) nodes over the same box, in TT the first n - p nodes at the original spacing.
    @testset "DifferentialOpsScalar with eye padding (B3)" begin
        bcs_list = (
            ("periodic", BoundaryCond1d(PeriodicBC(), PeriodicBC())),
            ("Dirichlet", BoundaryCond1d(DirichletBC(0.0), DirichletBC(0.0))),
        )
        @testset "QTT, $name" for (name, bc) in bcs_list
            bcs = BoundaryConditions(bc)
            arch = QTTArchitecture((5,)) # toeplitz_mpo needs at least 4 non-padded sites
            ops = DifferenTT.DifferentialOpsScalar(Grid(arch, (1.0,)), ScalarField(arch, bcs), (1,); stencil_order=2)
            coarse = QTTArchitecture((4,))
            ref = DifferenTT.DifferentialOpsScalar(Grid(coarse, (1.0,)), ScalarField(coarse, bcs); stencil_order=2)
            @test todense(ops.d_ops[1]) ≈ kron(todense(ref.d_ops[1]), I(2))
            @test todense(ops.d2_ops[1]) ≈ kron(todense(ref.d2_ops[1]), I(2))
        end
        @testset "TT, $name" for (name, bc) in bcs_list
            bcs = BoundaryConditions(bc)
            arch = TTArchitecture((8,))
            grid = Grid(arch, (1.0,))
            h = grid.dx[1]
            ops = DifferenTT.DifferentialOpsScalar(grid, ScalarField(arch, bcs), (2,); stencil_order=2)
            reduced = TTArchitecture((6,))
            ref = DifferenTT.DifferentialOpsScalar(Grid(reduced, (5h,)), ScalarField(reduced, bcs); stencil_order=2)
            for (order, D, Dref) in ((1, ops.d_ops[1], ref.d_ops[1]), (2, ops.d2_ops[1], ref.d2_ops[1]))
                M = todense(D)
                @test M[1:6, 1:6] ≈ todense(Dref)
                @test iszero(M[1:6, 7:8]) && iszero(M[7:8, 1:6])
                # The 1/h^order scaling applies to the whole padded operator, the identity block included
                @test M[7:8, 7:8] ≈ I(2) / h^order
            end
        end
    end

    @testset "Eye padding against a hand ghost elimination ($name, q=$q, order $order)" for (name, bc) in (
            ("Dirichlet", BoundaryCond1d(DirichletBC(0.0), DirichletBC(0.0))),
            ("Neumann", BoundaryCond1d(NeumannBC(0.0), NeumannBC(0.0))),
            ("Neumann-Dirichlet", BoundaryCond1d(NeumannBC(0.0), DirichletBC(0.0))),
            ("periodic", BoundaryCond1d(PeriodicBC(), PeriodicBC())),
        ),
        q in (2, 8),
        order in (1, 2)

        # The operator acts on the first n - p nodes (TT) or on the coarse grid of 2^(n-p) nodes (QTT, kron with the
        # identity of the p padded bits); the padded rows are the identity
        left, right = side_symbol(bc.left), side_symbol(bc.right)
        n_tt, n_qtt = 20, 6 # toeplitz_mpo needs 4 non-padded sites
        for p in (1, 2)
            reduced = ghost_operator(order, n_tt - p, left, right; stencil_order=q)
            expected_tt = zeros(n_tt, n_tt)
            expected_tt[1:(n_tt-p), 1:(n_tt-p)] .= reduced
            expected_tt[(n_tt-p+1):n_tt, (n_tt-p+1):n_tt] .= I(p)
            tt = DifferenTT.onedim_differential_operator(
                order, TTArchitecture((n_tt,)), 1, p, bc, Array; stencil_order=q
            )
            @test tt ≈ expected_tt

            coarse = ghost_operator(order, 2^(n_qtt - p), left, right; stencil_order=q)
            qtt = DifferenTT.onedim_differential_operator(
                order, QTTArchitecture((n_qtt,)), 1, p, bc, Array; stencil_order=q
            )
            @test DifferenTT.contract_mpo(qtt) ≈ kron(coarse, Matrix(1.0I, 2^p, 2^p)) atol = 1e-10
        end
    end
end

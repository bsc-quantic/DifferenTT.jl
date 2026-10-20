using ITensors
using ITensorMPS
using DifferenTT
using Test
using LinearAlgebra

include(joinpath(@__DIR__, "..", "testutils.jl"))
using .TestUtils

# Helper function to generate expected boundary nullifier matrix
function expected_nullifier_matrix(dim::Int, bcs::BoundaryCond1d; stencil_order::Int=8)
    mat = Matrix{Float64}(I, dim, dim)
    if bcs.left isa DirichletBC
        rows = DifferenTT.stencil_rows(stencil_order)
        mat[rows, rows] .= 0.0
    end
    if bcs.right isa DirichletBC
        rows = dim + 1 .- reverse(DifferenTT.stencil_rows(stencil_order))
        mat[rows, rows] .= 0.0
    end
    return mat
end

# Dense reference of a one-dimensional operator by explicit ghost elimination, written term by term from the stencil
# coefficients (spacing 1, nodes 1:n). A stencil entry that lands on a ghost node j < 1 or j > n is dropped
# (`:truncate`) or moved to the mirror node (`:mirror`: u_{1-m} = u_{1+m} on the left, u_{n+m} = u_{n-m} on the right).
function ghost_operator(order::Int, n::Int, left::Symbol, right::Symbol; stencil_order::Int)
    coeffs = DifferenTT.derivative_coefficients(order; stencil_order=stencil_order)
    D = zeros(n, n)
    for i in 1:n, (k, c) in coeffs
        j = i + k
        if j < 1
            left == :mirror && (D[i, 2-j] += c)
        elseif j > n
            right == :mirror && (D[i, 2n-j] += c)
        else
            D[i, j] += c
        end
    end
    return D
end

# The three closures of one side that differ: Neumann mirrors, Dirichlet and open sides truncate
side_symbol(bc) = bc isa NeumannBC ? :mirror : :truncate

# Values of an MPS at the nodes of a two-dimensional grid of the unit box, whatever the layout of the sites
function node_values(mps, arch)
    grid = Grid(arch, (1.0, 1.0))
    dims = variable_dims(arch)
    return [
        DifferenTT.sample(mps, ((i - 1) * grid.dx[1], (j - 1) * grid.dx[2]), grid, arch) for
        i in 1:dims[1], j in 1:dims[2]
    ]
end

# Check that `op` is a diagonal operator whose diagonal, as a field on the nodes of `arch`, is `expected`
function check_diagonal(op::MPO, expected, arch)
    T = DifferenTT.contract_mpo(op)
    @test T ≈ Diagonal(diag(T)) atol=1e-12
    ones_on_sites = ones_mps(only.(siteinds(op; plev=0)), 1)
    @test node_values(apply(op, ones_on_sites), arch) ≈ expected atol=1e-12
end

@testset "bcoperators.jl" begin
    @testset "Toeplitz Matrices" begin
        coeffs = [1.0, 2.0, 3.0]
        L = DifferenTT.lower_toeplitz_matrix(coeffs)
        @test size(L) == (3, 3)
        @test L[1, 1] == 1.0
        @test L[2, 1] == 2.0
        @test L[3, 1] == 3.0
        @test L[1, 2] == 0.0
        @test L[2, 2] == 1.0

        U = DifferenTT.upper_toeplitz_matrix(coeffs)
        @test U == L'
    end

    @testset "1D Boundary Nullifiers" begin
        nsite = 5
        dim = 2^nsite
        qtt_arch = QTTArchitecture((nsite,))
        tt_arch = TTArchitecture((dim,))

        for stencil_order in [2, 8]
            for (name, bc1d, expect_nothing) in [
                ("Periodic", BoundaryCond1d(PeriodicBC(), PeriodicBC()), true),
                ("Dirichlet", BoundaryCond1d(DirichletBC(0.0), DirichletBC(0.0)), false),
                ("Dirichlet-Neumann", BoundaryCond1d(DirichletBC(0.0), NeumannBC(0.0)), false),
                ("Neumann-Dirichlet", BoundaryCond1d(NeumannBC(0.0), DirichletBC(0.0)), false),
            ]
                @testset "Nullifier Stencil $stencil_order, BC $name" begin
                    expected_mat = expected_nullifier_matrix(dim, bc1d; stencil_order=stencil_order)

                    tt_null = DifferenTT.onedim_boundary_nullifier(tt_arch, 1, bc1d, Array; stencil_order=stencil_order)
                    if expect_nothing
                        @test isnothing(tt_null)
                    else
                        @test tt_null ≈ expected_mat atol=1e-10
                    end

                    qtt_null_mpo = DifferenTT.onedim_boundary_nullifier(
                        qtt_arch, 1, bc1d, Array; stencil_order=stencil_order
                    )
                    if expect_nothing
                        @test isnothing(qtt_null_mpo)
                    else
                        qtt_null_mat = DifferenTT.contract_mpo(qtt_null_mpo)
                        @test qtt_null_mat ≈ expected_mat atol=1e-10
                    end
                end
            end
        end
    end

    @testset "corner_pick_matrix / corner_pick_mpo contents" begin
        # A simple, hand-checkable coefficient vector so the expected matrices can be written
        # out literally rather than re-derived from the code under test.
        coeffs = [1.0, 2.0, 3.0, 4.0]
        expected_upper = [
            1.0 2.0 3.0 4.0
             0.0 1.0 2.0 3.0
             0.0 0.0 1.0 2.0
             0.0 0.0 0.0 1.0
        ]
        expected_lower = [
            1.0 0.0 0.0 0.0
             2.0 1.0 0.0 0.0
             3.0 2.0 1.0 0.0
             4.0 3.0 2.0 1.0
        ]

        @testset "corner_pick_matrix (TT, dense)" begin
            @test DifferenTT.corner_pick_matrix(coeffs, :upper) == expected_upper
            @test DifferenTT.corner_pick_matrix(coeffs, :lower) == expected_lower
        end

        @testset "corner_pick_mpo (QTT) matches the dense block, and is zero elsewhere" begin
            nsite = 5
            N = 2^nsite
            sites = siteinds(2, nsite)

            full_upper = DifferenTT.contract_mpo(DifferenTT.corner_pick_mpo(coeffs, sites, :upper))
            @test full_upper[1:4, 1:4] ≈ expected_upper atol=1e-12
            @test all(iszero, full_upper[1:4, 5:N])   # no coupling into non-boundary columns
            @test all(iszero, full_upper[5:N, :])     # no coupling out of the corner block at all

            full_lower = DifferenTT.contract_mpo(DifferenTT.corner_pick_mpo(coeffs, sites, :lower))
            @test full_lower[(N-3):N, (N-3):N] ≈ expected_lower atol=1e-12
            @test all(iszero, full_lower[(N-3):N, 1:(N-4)])
            @test all(iszero, full_lower[1:(N-4), :])
        end
    end

    @testset "Dirichlet ghost-correction identity: (D_trunc + C) * ones == 0" begin
        # This is the identity every Dirichlet ghost correction must satisfy: truncating the
        # stencil at a non-periodic boundary is elimination of the exterior ghost points at value
        # 0 (`onedim_differential_operator`, diffoperators.jl), so the correction `C` that moves a
        # nonzero ghost value to the RHS must, on the constant field, supply exactly what
        # truncation removed. Concretely, on the row at distance `m` from the edge, `C * ones`
        # must equal the cumulative tail sum of dropped coefficients `sum(c_k for k in (m+1):w)`
        # (right) or `sum(c_k for k in -w:-(m+1))` (left). This is computed independently of
        # `onedim_{left,right}_dirichlet_correction` itself, so it catches sign, ordering, and
        # summation bugs rather than merely re-deriving what the code already computes.
        nsite = 5
        N = 2^nsite
        qtt_arch = QTTArchitecture((nsite,); interleaved=false)
        tt_arch = TTArchitecture((N,))
        ones_vec = ones(N)

        bc_right = BoundaryCond1d(NeumannBC(0.0), DirichletBC(1.0))
        bc_left = BoundaryCond1d(DirichletBC(1.0), NeumannBC(0.0))

        for stencil_order in (2, 8), derivative_order in (1, 2)
            coeffs = DifferenTT.derivative_coefficients(derivative_order; stencil_order=stencil_order)
            required_right = [sum(coeffs[k] for k in (m+1):4) for m in 3:-1:0]
            required_left = [sum(coeffs[k] for k in -4:(-(m+1))) for m in 0:3]

            @testset "stencil_order=$stencil_order, derivative_order=$derivative_order" begin
                # -- TT (dense) path --
                D_right = DifferenTT.onedim_differential_operator(
                    derivative_order, tt_arch, 1, bc_right, Array; stencil_order=stencil_order
                )
                C_right = DifferenTT.onedim_right_dirichlet_correction(
                    derivative_order, tt_arch, 1, Array; stencil_order=stencil_order
                )
                @test (C_right*ones_vec)[(end-3):end] ≈ required_right atol=1e-10
                @test (D_right*ones_vec .+ C_right*ones_vec)[(end-3):end] ≈ zeros(4) atol=1e-10

                D_left = DifferenTT.onedim_differential_operator(
                    derivative_order, tt_arch, 1, bc_left, Array; stencil_order=stencil_order
                )
                C_left = DifferenTT.onedim_left_dirichlet_correction(
                    derivative_order, tt_arch, 1, Array; stencil_order=stencil_order
                )
                @test (C_left*ones_vec)[1:4] ≈ required_left atol=1e-10
                @test (D_left*ones_vec .+ C_left*ones_vec)[1:4] ≈ zeros(4) atol=1e-10

                # -- QTT path, contracted to dense --
                D_right_q = DifferenTT.contract_mpo(
                    DifferenTT.onedim_differential_operator(
                        derivative_order, qtt_arch, 1, bc_right, Array; stencil_order=stencil_order
                    ),
                )
                C_right_q = DifferenTT.contract_mpo(
                    DifferenTT.onedim_right_dirichlet_correction(
                        derivative_order, qtt_arch, 1, Array; stencil_order=stencil_order
                    ),
                )
                @test (C_right_q*ones_vec)[(end-3):end] ≈ required_right atol=1e-10
                @test (D_right_q*ones_vec .+ C_right_q*ones_vec)[(end-3):end] ≈ zeros(4) atol=1e-10

                D_left_q = DifferenTT.contract_mpo(
                    DifferenTT.onedim_differential_operator(
                        derivative_order, qtt_arch, 1, bc_left, Array; stencil_order=stencil_order
                    ),
                )
                C_left_q = DifferenTT.contract_mpo(
                    DifferenTT.onedim_left_dirichlet_correction(
                        derivative_order, qtt_arch, 1, Array; stencil_order=stencil_order
                    ),
                )
                @test (C_left_q*ones_vec)[1:4] ≈ required_left atol=1e-10
                @test (D_left_q*ones_vec .+ C_left_q*ones_vec)[1:4] ≈ zeros(4) atol=1e-10

                # -- QTT and TT must agree exactly: this is the invariant that was broken before
                # the fix (the QTT and TT corrections were two different, independently wrong,
                # computations) and must hold going forward. --
                @test C_right_q ≈ Array(C_right) atol=1e-10
                @test C_left_q ≈ Array(C_left) atol=1e-10
            end
        end

        @testset "literal anchor values (stencil_order=8, derivative_order=1, right)" begin
            # Hardcoded against exact fractions of the known 8th-order stencil weights
            # (c1=4/5, c2=-1/5, c3=4/105, c4=-1/280), independent of `derivative_coefficients`,
            # so a future change to the coefficient table can't silently break this test too.
            C_right = DifferenTT.onedim_right_dirichlet_correction(1, tt_arch, 1, Array; stencil_order=8)
            expected = [-1 / 280, 4 / 105 - 1 / 280, -1 / 5 + 4 / 105 - 1 / 280, 4 / 5 - 1 / 5 + 4 / 105 - 1 / 280]
            @test (C_right*ones_vec)[(end-3):end] ≈ expected atol=1e-12
        end
    end

    @testset "Wall vorticity corrections" begin
        nsite = 4
        dim = 2^nsite
        qtt_arch = QTTArchitecture((nsite,))
        tt_arch = TTArchitecture((dim,))

        @testset "onedim_entry_operator" begin
            # The repeated entry must accumulate
            entries = [(1, 1, 2.0), (3, 7, -1.5), (dim, 2, 0.25), (3, 7, 0.5)]
            expected = zeros(dim, dim)
            for (row, col, value) in entries
                expected[row, col] += value
            end
            @test DifferenTT.contract_mpo(DifferenTT.onedim_entry_operator(qtt_arch, 1, entries)) ≈ expected atol=1e-12
            @test DifferenTT.onedim_entry_operator(tt_arch, 1, entries) ≈ expected

            no_entries = Tuple{Int,Int,Float64}[]
            @test isnothing(DifferenTT.onedim_entry_operator(qtt_arch, 1, no_entries))
            @test isnothing(DifferenTT.onedim_entry_operator(tt_arch, 1, no_entries))
        end

        bc_cases = [
            ("Dirichlet", BoundaryCond1d(DirichletBC(0.0), DirichletBC(0.0))),
            ("Dirichlet-Neumann", BoundaryCond1d(DirichletBC(0.0), NeumannBC(0.0))),
            ("Neumann-Dirichlet", BoundaryCond1d(NeumannBC(0.0), DirichletBC(0.0))),
            ("Neumann", BoundaryCond1d(NeumannBC(0.0), NeumannBC(0.0))),
            ("Periodic", BoundaryCond1d(PeriodicBC(), PeriodicBC())),
        ]

        @testset "onedim_wall_vorticity_correction, $formula, $name" for formula in (:jensen, :thom),
            (name, bc1d) in bc_cases

            c1, c2, _ = DifferenTT.wall_vorticity_coefficients(formula)
            # c₁ ψ₁ + c₂ ψ₂ on the node next to each wall, independent of the implementation
            expected = zeros(dim, dim)
            if bc1d.left isa DirichletBC
                expected[1, 1] = c1
                expected[1, 2] = c2
            end
            if bc1d.right isa DirichletBC
                expected[dim, dim] = c1
                expected[dim, dim-1] = c2
            end

            # Zero coefficients (c₂ of Thom's formula) are not stored as entries
            entries = DifferenTT.onedim_wall_vorticity_entries(bc1d, dim; formula=formula)
            @test length(entries) == count(!iszero, expected)

            qtt = DifferenTT.onedim_wall_vorticity_correction(qtt_arch, 1, bc1d; formula=formula)
            tt = DifferenTT.onedim_wall_vorticity_correction(tt_arch, 1, bc1d; formula=formula)
            if iszero(expected)
                @test isnothing(qtt)
                @test isnothing(tt)
            else
                @test DifferenTT.contract_mpo(qtt) ≈ expected atol=1e-12
                @test tt ≈ expected
            end
        end

        @testset "Wall vorticity corrections with explicit wall flags" begin
            c1, c2, _ = DifferenTT.wall_vorticity_coefficients(:jensen)
            # Only the flagged side gets entries, whatever the boundary conditions
            @test DifferenTT.onedim_wall_vorticity_entries((true, false), dim) == [(1, 1, c1), (1, 2, c2)]
            @test DifferenTT.onedim_wall_vorticity_entries((false, true), dim) == [(dim, dim, c1), (dim, dim - 1, c2)]
            @test isempty(DifferenTT.onedim_wall_vorticity_entries((false, false), dim))
            @test isnothing(DifferenTT.onedim_wall_vorticity_correction(qtt_arch, 1, (false, false)))

            # The BoundaryCond1d methods are the flag methods with every DirichletBC side as a wall
            for (name, bc1d) in bc_cases, formula in (:jensen, :thom)
                flags = (bc1d.left isa DirichletBC, bc1d.right isa DirichletBC)
                @test DifferenTT.onedim_wall_vorticity_entries(bc1d, dim; formula=formula) ==
                    DifferenTT.onedim_wall_vorticity_entries(flags, dim; formula=formula)
                any(flags) || continue
                @test DifferenTT.onedim_wall_vorticity_correction(tt_arch, 1, bc1d; formula=formula) ==
                    DifferenTT.onedim_wall_vorticity_correction(tt_arch, 1, flags; formula=formula)
                @test DifferenTT.contract_mpo(
                    DifferenTT.onedim_wall_vorticity_correction(qtt_arch, 1, bc1d; formula=formula)
                ) ≈ DifferenTT.contract_mpo(
                    DifferenTT.onedim_wall_vorticity_correction(qtt_arch, 1, flags; formula=formula)
                )
            end
        end

        @testset "onedim_wall_vorticity_correction on a reversed second axis" begin
            # The 1D operator is built in natural order on the axis' own number of nodes; ndim_operator_mpo handles
            # the reversal when lifting
            c1, c2, _ = DifferenTT.wall_vorticity_coefficients(:jensen)
            expected = zeros(16, 16)
            expected[1, 1:2] .= (c1, c2)
            expected[16, 15:16] .= (c2, c1)
            bc1d = BoundaryCond1d(DirichletBC(0.0), DirichletBC(0.0))
            qtt = DifferenTT.onedim_wall_vorticity_correction(QTTArchitecture((3, -4)), 2, bc1d)
            @test DifferenTT.contract_mpo(qtt) ≈ expected atol=1e-12
        end
    end

    @testset "Neumann closure against a hand ghost elimination" begin
        # `neumann_correction_matrix` is the corner block that turns the truncated operator into the mirrored one:
        # D_closed = D_truncated - M. The reference below drops no entry silently: every coefficient that reaches a
        # ghost node is moved by hand to its mirror node.
        n = 12
        nsite = 5
        N = 2^nsite
        @testset "q=$q, derivative order $order" for q in (2, 8), order in (1, 2)
            h = q ÷ 2
            D_trunc = ghost_operator(order, n, :truncate, :truncate; stencil_order=q)
            D_left = ghost_operator(order, n, :mirror, :truncate; stencil_order=q)
            D_right = ghost_operator(order, n, :truncate, :mirror; stencil_order=q)

            for (side, closed) in ((:left, D_left), (:right, D_right))
                M = DifferenTT.neumann_correction_matrix(order, side; stencil_order=q)
                @test size(M) == (q, q)
                # The correction lives in a q × q corner and nowhere else
                full = D_trunc - closed
                corner = side == :left ? (1:q, 1:q) : ((n-q+1):n, (n-q+1):n)
                outside = trues(n, n)
                outside[corner...] .= false
                @test all(iszero, full[outside])
                @test M ≈ full[corner...] atol=1e-13
            end
            # An unknown side is the zero block
            @test iszero(DifferenTT.neumann_correction_matrix(order, :top; stencil_order=q))

            # Embedded in TT (matrix) and QTT (MPO) operators
            tt = TTArchitecture((n,))
            qtt = QTTArchitecture((nsite,))
            qsites = siteinds(2, nsite)
            for (side, closed, N_, tt_arch, qtt_arch) in ((:left, D_left, n, tt, qtt), (:right, D_right, n, tt, qtt))
                tt_corr = if side == :left
                    DifferenTT.left_neumann_boundary_correction
                else
                    DifferenTT.right_neumann_boundary_correction
                end
                @test tt_corr(order, tt, n, Array; stencil_order=q) ≈ D_trunc - closed atol=1e-13
                # QTT: same block in a 2^nsite × 2^nsite operator, padded by zeros
                D_trunc_q = ghost_operator(order, N, :truncate, :truncate; stencil_order=q)
                closed_q = ghost_operator(
                    order, N, side == :left ? :mirror : :truncate, side == :right ? :mirror : :truncate; stencil_order=q
                )
                qtt_mpo = tt_corr(order, qtt, qsites, Array; stencil_order=q)
                @test qtt_mpo isa MPO
                @test DifferenTT.contract_mpo(qtt_mpo) ≈ D_trunc_q - closed_q atol=1e-13
            end
        end

        @testset "q=$q: stencil_order other than 2 and 8" for q in (4, 6)
            @test_throws ArgumentError DifferenTT.neumann_correction_matrix(2, :left; stencil_order=q)
            @test_throws ArgumentError DifferenTT.stencil_rows(q)
        end
    end

    @testset "onedim_differential_operator with every boundary-condition mix, against ghost elimination" begin
        # Truncated (Dirichlet, open), mirrored (Neumann) and wrapped (periodic) closures of a one-dimensional operator
        n = 32
        mixes = (
            ("Dirichlet-Dirichlet", BoundaryCond1d(DirichletBC(0.0), DirichletBC(0.0))),
            ("Neumann-Neumann", BoundaryCond1d(NeumannBC(0.0), NeumannBC(0.0))),
            ("Neumann-Dirichlet", BoundaryCond1d(NeumannBC(0.0), DirichletBC(0.0))),
            ("Dirichlet-Neumann", BoundaryCond1d(DirichletBC(0.0), NeumannBC(0.0))),
            ("Open-Neumann", BoundaryCond1d(OpenBC(), NeumannBC(0.0))),
        )
        tt = TTArchitecture((n,))
        qtt = QTTArchitecture((5,))
        @testset "$name, q=$q, order $order" for (name, bc) in mixes, q in (2, 8), order in (1, 2)
            expected = ghost_operator(order, n, side_symbol(bc.left), side_symbol(bc.right); stencil_order=q)
            @test DifferenTT.onedim_differential_operator(order, tt, 1, bc, Array; stencil_order=q) ≈ expected atol=1e-12
            qtt_mpo = DifferenTT.onedim_differential_operator(order, qtt, 1, bc, Array; stencil_order=q)
            # The MPO is truncated with the ITensors cutoff 1e-15, which leaves errors of order 1e-11 for q = 8
            @test DifferenTT.contract_mpo(qtt_mpo) ≈ expected atol=1e-10
        end
        @testset "periodic, q=$q, order $order" for q in (2, 8), order in (1, 2)
            coeffs = DifferenTT.derivative_coefficients(order; stencil_order=q)
            expected = zeros(n, n)
            for i in 1:n, (k, c) in coeffs
                expected[i, mod1(i + k, n)] += c
            end
            bc = BoundaryCond1d(PeriodicBC(), PeriodicBC())
            @test DifferenTT.onedim_differential_operator(order, tt, 1, bc, Array; stencil_order=q) ≈ expected atol=1e-12
            qtt_mpo = DifferenTT.onedim_differential_operator(order, qtt, 1, bc, Array; stencil_order=q)
            @test DifferenTT.contract_mpo(qtt_mpo) ≈ expected atol=1e-10
        end
    end

    @testset "Neumann symmetrization: W A is symmetric (q=$q, $name)" for q in (2, 8),
        (name, bc) in (
            ("Neumann-Neumann", BoundaryCond1d(NeumannBC(0.0), NeumannBC(0.0))),
            ("Neumann-Dirichlet", BoundaryCond1d(NeumannBC(0.0), DirichletBC(0.0))),
            ("Dirichlet-Neumann", BoundaryCond1d(DirichletBC(0.0), NeumannBC(0.0))),
        )
        # A is the mirrored second derivative, which is not symmetric (the mirror doubles the entry next to a Neumann
        # row, not the transposed one). Halving the Neumann rows restores the symmetry: W = I + C(I), with
        # C = -1/2 e_r e_r' A the symmetrization correction applied to the identity.
        n = 16
        A = ghost_operator(2, n, side_symbol(bc.left), side_symbol(bc.right); stencil_order=q)
        w = ones(n)
        bc.left isa NeumannBC && (w[1] = 0.5)
        bc.right isa NeumannBC && (w[n] = 0.5)

        # TT: dense matrices
        tt = TTArchitecture((n,))
        Id = Matrix{Float64}(I, n, n)
        C_id = DifferenTT.onedim_neumann_symmetrization_correction(Id, tt, 1, bc; stencil_order=q)
        @test Id + C_id ≈ Diagonal(w)
        C_A = DifferenTT.onedim_neumann_symmetrization_correction(A, tt, 1, bc; stencil_order=q)
        @test A + C_A ≈ Diagonal(w) * A
        WA = Diagonal(w) * A
        @test WA ≈ WA' atol=1e-12
        @test !(A ≈ A')     # the weight is needed

        # QTT: the same corrections as MPOs, acting on the MPO of A and on the identity
        qtt = QTTArchitecture((4,))
        qsites = siteinds(2, 4)
        A_mpo = mat2mpo(A, qsites)
        C_mpo = DifferenTT.onedim_neumann_symmetrization_correction(A_mpo, qtt, 1, bc; stencil_order=q)
        @test DifferenTT.contract_mpo(C_mpo) ≈ C_A atol=1e-12
        C_id_mpo = DifferenTT.onedim_neumann_symmetrization_correction(
            identity_mpo(qsites), qtt, 1, bc; stencil_order=q
        )
        @test DifferenTT.contract_mpo(C_id_mpo) ≈ C_id atol=1e-12
    end

    @testset "Neumann symmetrization correction: single sides and no Neumann side" begin
        n = 16
        A = Matrix(reshape(1.0:(n^2), n, n))
        tt = TTArchitecture((n,))
        qtt = QTTArchitecture((4,))
        qsites = siteinds(2, 4)
        # Only the first row, only the last row, and both: -1/2 of the Neumann boundary rows of A
        for (bc, rows) in (
            (BoundaryCond1d(NeumannBC(0.0), DirichletBC(0.0)), [1]),
            (BoundaryCond1d(DirichletBC(0.0), NeumannBC(0.0)), [n]),
            (BoundaryCond1d(NeumannBC(0.0), NeumannBC(0.0)), [1, n]),
        )
            expected = zeros(n, n)
            expected[rows, :] .= -0.5 .* A[rows, :]
            @test DifferenTT.onedim_neumann_symmetrization_correction(A, tt, 1, bc) ≈ expected
            C = DifferenTT.onedim_neumann_symmetrization_correction(mat2mpo(A, qsites), qtt, 1, bc)
            @test DifferenTT.contract_mpo(C) ≈ expected rtol=1e-10
        end
        # The side methods: the first row alone, the last row alone
        left = DifferenTT.left_neumann_symmetrization_correction(A, tt, 1)
        right = DifferenTT.right_neumann_symmetrization_correction(A, tt, 1)
        @test left[1, :] ≈ -0.5 .* A[1, :]
        @test iszero(left[2:end, :])
        @test right[end, :] ≈ -0.5 .* A[end, :]
        @test iszero(right[1:(end-1), :])
        @test A == Matrix(reshape(1.0:(n^2), n, n))   # the input is not modified

        for bc in (
            BoundaryCond1d(DirichletBC(0.0), DirichletBC(0.0)),
            BoundaryCond1d(PeriodicBC(), PeriodicBC()),
            BoundaryCond1d(OpenBC(), OpenBC()),
        )
            @test isnothing(DifferenTT.onedim_neumann_symmetrization_correction(A, tt, 1, bc))
            @test isnothing(DifferenTT.onedim_neumann_symmetrization_correction(mat2mpo(A, qsites), qtt, 1, bc))
        end
    end

    @testset "BoundaryNullScalar and BoundaryNull against the dense row mask" begin
        # Z_i = I - sum e_r e_r' over the q/2 rows next to each Dirichlet side of axis i; the projector of a field is
        # the product over its axes, i.e. the diagonal that is zero on a row next to any Dirichlet side
        dirichlet = BoundaryCond1d(DirichletBC(0.0), DirichletBC(0.0))
        neumann = BoundaryCond1d(NeumannBC(0.0), NeumannBC(0.0))
        periodic = BoundaryCond1d(PeriodicBC(), PeriodicBC())
        left_wall = BoundaryCond1d(DirichletBC(0.0), NeumannBC(0.0))

        function mask_diagonal(dims, bcs, q)
            keep = ones(dims...)
            for axis in 1:2
                rows = DifferenTT.stencil_rows(q)
                n = dims[axis]
                idx = Int[]
                bcs[axis].left isa DirichletBC && append!(idx, rows)
                bcs[axis].right isa DirichletBC && append!(idx, n + 1 .- rows)
                for r in idx
                    axis == 1 ? (keep[r, :] .= 0.0) : (keep[:, r] .= 0.0)
                end
            end
            return keep
        end

        @testset "$label, q=$q" for (label, arch) in (
                ("QTT", QTTArchitecture((4, 4))),
                ("QTT interleaved", QTTArchitecture((4, 4); interleaved=true)),
                ("TT", TTArchitecture((16, 16))),
            ),
            q in (2, 8)

            dims = variable_dims(arch)
            for bcs_xy in ((dirichlet, periodic), (periodic, dirichlet), (left_wall, dirichlet), (dirichlet, neumann))
                bcs = BoundaryConditions(bcs_xy...)
                field = ScalarField(siteinds(arch), arch, bcs)
                null = DifferenTT.BoundaryNullScalar(field; stencil_order=q)
                @test null isa DifferenTT.BoundaryNullScalar{2}
                @test !isnothing(null.u_row_deleter)
                check_diagonal(null.u_row_deleter, mask_diagonal(dims, bcs_xy, q), arch)
            end

            # No Dirichlet side: no projector at all
            for bcs_xy in ((periodic, periodic), (neumann, periodic), (neumann, neumann))
                field = ScalarField(siteinds(arch), arch, BoundaryConditions(bcs_xy...))
                @test isnothing(DifferenTT.BoundaryNullScalar(field; stencil_order=q).u_row_deleter)
            end

            # Each component of a vector field is projected with its own boundary conditions
            bc_u = BoundaryConditions(dirichlet, periodic)
            bc_v = BoundaryConditions(periodic, left_wall)
            vfield = VectorField(siteinds(arch), arch, (bc_u, bc_v))
            nulls = DifferenTT.BoundaryNull(vfield; stencil_order=q)
            @test nulls isa DifferenTT.BoundaryNull{2}
            @test nulls[1] === nulls.components[1]
            check_diagonal(nulls[1].u_row_deleter, mask_diagonal(dims, (dirichlet, periodic), q), arch)
            check_diagonal(nulls[2].u_row_deleter, mask_diagonal(dims, (periodic, left_wall), q), arch)
        end
    end

    @testset "divergence_dirichlet_correction keeps only the own axis of each component" begin
        # The divergence uses du_i/dx_i, so component i is projected with its boundary conditions along axis i only.
        # u has walls along x (its own axis) and periodic y; v has walls along x too (not its own axis) and periodic y.
        dirichlet = BoundaryCond1d(DirichletBC(0.0), DirichletBC(0.0))
        neumann = BoundaryCond1d(NeumannBC(0.0), NeumannBC(0.0))
        periodic = BoundaryCond1d(PeriodicBC(), PeriodicBC())
        @testset "$label, q=$q" for (label, arch) in (
                ("QTT", QTTArchitecture((4, 4))),
                ("QTT interleaved", QTTArchitecture((4, 4); interleaved=true)),
                ("TT", TTArchitecture((16, 16))),
            ),
            q in (2, 8)

            dims = variable_dims(arch)
            rows = DifferenTT.stencil_rows(q)
            function projector(axis_bcs...)
                keep = ones(dims...)
                for (axis, bc) in enumerate(axis_bcs)
                    idx = Int[]
                    bc.left isa DirichletBC && append!(idx, rows)
                    bc.right isa DirichletBC && append!(idx, dims[axis] + 1 .- rows)
                    for r in idx
                        axis == 1 ? (keep[r, :] .= 0.0) : (keep[:, r] .= 0.0)
                    end
                end
                return keep
            end
            sites = siteinds(arch)
            u = ScalarField(sites, arch, BoundaryConditions(dirichlet, periodic))
            v = ScalarField(sites, arch, BoundaryConditions(dirichlet, periodic))
            C = DifferenTT.divergence_dirichlet_correction(VectorField((u, v)); stencil_order=q)
            # Own axis of u is x (Dirichlet), own axis of v is y (periodic): only the x rows are removed
            check_diagonal(C, projector(dirichlet), arch)

            # Walls along both own axes: the product of both projectors
            v_wall = ScalarField(sites, arch, BoundaryConditions(periodic, dirichlet))
            C2 = DifferenTT.divergence_dirichlet_correction(VectorField((u, v_wall)); stencil_order=q)
            check_diagonal(C2, projector(dirichlet, dirichlet), arch)

            # Neumann or periodic along the own axes: nothing to remove
            p_u = ScalarField(sites, arch, BoundaryConditions(neumann, periodic))
            p_v = ScalarField(sites, arch, BoundaryConditions(dirichlet, neumann))
            @test isnothing(DifferenTT.divergence_dirichlet_correction(VectorField((p_u, p_v)); stencil_order=q))
        end
    end

    @testset "Periodic TT operator on an axis of at most q nodes (regression)" begin
        # A stencil of half-width 4 wraps onto itself when n <= 8: n = 8 maps the offsets -4 and +4 to the same node,
        # whose coefficients add up (the operator overwrote them), and n < 5 wrapped more than once (BoundsError)
        wrap = BoundaryCond1d(PeriodicBC(), PeriodicBC())
        @testset "n=$n, q=$q, order $order" for n in (8, 7, 6, 5, 4, 3), q in (2, 8), order in (1, 2)
            coeffs = DifferenTT.derivative_coefficients(order; stencil_order=q)
            expected = zeros(n, n)
            for i in 1:n, (k, c) in coeffs
                expected[i, mod1(i + k, n)] += c
            end
            D = DifferenTT.onedim_differential_operator(order, TTArchitecture((n,)), 1, wrap, Array; stencil_order=q)
            @test D ≈ expected
            # The constant is in the kernel, and the periodic operators are circulant
            @test D * ones(n) ≈ zeros(n) atol = 1e-12
            @test all(D[mod1(i + 1, n), mod1(j + 1, n)] ≈ D[i, j] for i in 1:n, j in 1:n)
        end
    end
end

using ITensors
using ITensorMPS
using DifferenTT
using Test
using LinearAlgebra

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
                ("Neumann-Dirichlet", BoundaryCond1d(NeumannBC(0.0), DirichletBC(0.0)), false)
            ]
                @testset "Nullifier Stencil $stencil_order, BC $name" begin
                    expected_mat = expected_nullifier_matrix(dim, bc1d; stencil_order=stencil_order)
                    
                    tt_null = DifferenTT.onedim_boundary_nullifier(tt_arch, 1, bc1d, Array; stencil_order=stencil_order)
                    if expect_nothing
                        @test isnothing(tt_null)
                    else
                        @test tt_null ≈ expected_mat atol=1e-10
                    end
                    
                    qtt_null_mpo = DifferenTT.onedim_boundary_nullifier(qtt_arch, 1, bc1d, Array; stencil_order=stencil_order)
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
        expected_upper = [1.0 2.0 3.0 4.0
                           0.0 1.0 2.0 3.0
                           0.0 0.0 1.0 2.0
                           0.0 0.0 0.0 1.0]
        expected_lower = [1.0 0.0 0.0 0.0
                           2.0 1.0 0.0 0.0
                           3.0 2.0 1.0 0.0
                           4.0 3.0 2.0 1.0]

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
            @test full_lower[N-3:N, N-3:N] ≈ expected_lower atol=1e-12
            @test all(iszero, full_lower[N-3:N, 1:N-4])
            @test all(iszero, full_lower[1:N-4, :])
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
            required_right = [sum(coeffs[k] for k in (m + 1):4) for m in 3:-1:0]
            required_left = [sum(coeffs[k] for k in -4:-(m + 1)) for m in 0:3]

            @testset "stencil_order=$stencil_order, derivative_order=$derivative_order" begin
                # -- TT (dense) path --
                D_right = DifferenTT.onedim_differential_operator(derivative_order, tt_arch, 1, bc_right, Array; stencil_order=stencil_order)
                C_right = DifferenTT.onedim_right_dirichlet_correction(derivative_order, tt_arch, 1, Array; stencil_order=stencil_order)
                @test (C_right * ones_vec)[end-3:end] ≈ required_right atol=1e-10
                @test (D_right * ones_vec .+ C_right * ones_vec)[end-3:end] ≈ zeros(4) atol=1e-10

                D_left = DifferenTT.onedim_differential_operator(derivative_order, tt_arch, 1, bc_left, Array; stencil_order=stencil_order)
                C_left = DifferenTT.onedim_left_dirichlet_correction(derivative_order, tt_arch, 1, Array; stencil_order=stencil_order)
                @test (C_left * ones_vec)[1:4] ≈ required_left atol=1e-10
                @test (D_left * ones_vec .+ C_left * ones_vec)[1:4] ≈ zeros(4) atol=1e-10

                # -- QTT path, contracted to dense --
                D_right_q = DifferenTT.contract_mpo(DifferenTT.onedim_differential_operator(derivative_order, qtt_arch, 1, bc_right, Array; stencil_order=stencil_order))
                C_right_q = DifferenTT.contract_mpo(DifferenTT.onedim_right_dirichlet_correction(derivative_order, qtt_arch, 1, Array; stencil_order=stencil_order))
                @test (C_right_q * ones_vec)[end-3:end] ≈ required_right atol=1e-10
                @test (D_right_q * ones_vec .+ C_right_q * ones_vec)[end-3:end] ≈ zeros(4) atol=1e-10

                D_left_q = DifferenTT.contract_mpo(DifferenTT.onedim_differential_operator(derivative_order, qtt_arch, 1, bc_left, Array; stencil_order=stencil_order))
                C_left_q = DifferenTT.contract_mpo(DifferenTT.onedim_left_dirichlet_correction(derivative_order, qtt_arch, 1, Array; stencil_order=stencil_order))
                @test (C_left_q * ones_vec)[1:4] ≈ required_left atol=1e-10
                @test (D_left_q * ones_vec .+ C_left_q * ones_vec)[1:4] ≈ zeros(4) atol=1e-10

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
            @test (C_right * ones_vec)[end-3:end] ≈ expected atol=1e-12
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

        @testset "onedim_wall_vorticity_correction, $formula, $name" for formula in (:jensen, :thom), (name, bc1d) in bc_cases
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
                @test DifferenTT.onedim_wall_vorticity_entries(bc1d, dim; formula=formula) == DifferenTT.onedim_wall_vorticity_entries(flags, dim; formula=formula)
                any(flags) || continue
                @test DifferenTT.onedim_wall_vorticity_correction(tt_arch, 1, bc1d; formula=formula) == DifferenTT.onedim_wall_vorticity_correction(tt_arch, 1, flags; formula=formula)
                @test DifferenTT.contract_mpo(DifferenTT.onedim_wall_vorticity_correction(qtt_arch, 1, bc1d; formula=formula)) ≈
                      DifferenTT.contract_mpo(DifferenTT.onedim_wall_vorticity_correction(qtt_arch, 1, flags; formula=formula))
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
end

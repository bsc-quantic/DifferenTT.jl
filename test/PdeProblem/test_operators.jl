using ITensors
using ITensorMPS
using DifferenTT
using Test
using LinearAlgebra

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
        mat -= DifferenTT.left_neumann_boundary_correction(derivative_order, dim, Array; stencil_order=stencil_order)
    end
    if bcs.right isa NeumannBC
        mat -= DifferenTT.right_neumann_boundary_correction(derivative_order, dim, Array; stencil_order=stencil_order)
    end
    return mat
end

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

@testset "operators.jl" begin
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
                    ("Neumann", BoundaryCond1d(NeumannBC(0.0), NeumannBC(0.0)))
                ]
                    @testset "Derivative $order, Stencil $stencil_order, BC $name" begin
                        expected_mat = expected_diff_matrix(order, dim, bc1d; stencil_order=stencil_order)
                        
                        # Test TT Architecture
                        tt_mat = DifferenTT.onedim_differential_operator(order, tt_arch, 1, bc1d, Array; stencil_order=stencil_order)
                        @test tt_mat ≈ expected_mat atol=1e-12

                        # Test QTT Architecture
                        qtt_mpo = DifferenTT.onedim_differential_operator(order, qtt_arch, 1, bc1d, Array; stencil_order=stencil_order)
                        qtt_mat = DifferenTT.contract_mpo(qtt_mpo)
                        @test qtt_mat ≈ expected_mat atol=1e-10
                        
                        # Eye padding tests
                        for n_eyepad in 0:2
                            padded_dim = dim
                            unpadded_dim = dim ÷ (2^n_eyepad)
                            expected_unpadded = expected_diff_matrix(order, unpadded_dim, bc1d; stencil_order=stencil_order)
                            expected_padded_tt = zeros(padded_dim, padded_dim)
                            expected_padded_tt[1:unpadded_dim, 1:unpadded_dim] .= expected_unpadded
                            for i in (unpadded_dim + 1):padded_dim
                                expected_padded_tt[i, i] = 1.0
                            end
                            
                            tt_padded = DifferenTT.onedim_differential_operator(order, tt_arch, 1, n_eyepad, bc1d, Array; stencil_order=stencil_order)
                            @test tt_padded ≈ expected_padded_tt atol=1e-12
                            
                            expected_padded_qtt = kron(expected_unpadded, Matrix{Float64}(I, 2^n_eyepad, 2^n_eyepad))
                            @show n_eyepad
                            qtt_mpo_padded = DifferenTT.onedim_differential_operator(order, qtt_arch, 1, n_eyepad, bc1d, Array; stencil_order=stencil_order)
                            qtt_padded_mat = DifferenTT.contract_mpo(qtt_mpo_padded)
                            @test qtt_padded_mat ≈ expected_padded_qtt atol=1e-10
                        end
                    end
                end
            end
        end
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
                        @test tt_null ≈ expected_mat atol=1e-12
                    end
                    
                    qtt_null_mpo = DifferenTT.onedim_boundary_nullifier(qtt_arch, 1, bc1d, Array; stencil_order=stencil_order)
                    if expect_nothing
                        @test isnothing(qtt_null_mpo)
                    else
                        qtt_null_mat = DifferenTT.contract_mpo(qtt_null_mpo)
                        @test qtt_null_mat ≈ expected_mat atol=1e-12
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
end

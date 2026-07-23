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
end

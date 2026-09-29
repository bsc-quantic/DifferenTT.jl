using Test
using ITensors
using ITensorMPS
using KrylovKit
using LinearAlgebra
using DifferenTT
using DifferenTT: _uses_cg_metric, _eig_condition, _warn_if_not_spd, _local_dense_matrix

@testset "KrylovUpdater instantiation" begin
    updater = DifferenTT.KrylovUpdater()
    @test updater isa DifferenTT.AbstractUpdater
end

@testset "KrylovUpdater condition metrics" begin
    @testset "_uses_cg_metric" begin
        @test _uses_cg_metric((ishermitian=true, isposdef=true)) == true
        @test _uses_cg_metric((ishermitian=true,)) == false
        @test _uses_cg_metric((isposdef=true,)) == false
        @test _uses_cg_metric(NamedTuple()) == false
    end

    @testset "_eig_condition" begin
        # SPD: κ_λ is the spectral condition number and matches κ₂
        κ, λ = _eig_condition(Matrix(Diagonal([1.0, 4.0, 10.0])))
        @test κ ≈ 10.0
        @test length(λ) == 3
        B = randn(6, 6)
        A_spd = B' * B + I
        @test first(_eig_condition(A_spd)) ≈ cond(A_spd)

        # Symmetric indefinite: ratio of moduli
        @test first(_eig_condition(Matrix(Diagonal([-4.0, 1.0, 2.0])))) ≈ 4.0

        # Complex spectrum (±i)
        @test first(_eig_condition([0.0 -1.0; 1.0 0.0])) ≈ 1.0

        # Non-normal: κ_λ is only a lower bound on κ₂
        M = [1.0 100.0; 0.0 2.0]
        @test first(_eig_condition(M)) ≈ 2.0
        @test first(_eig_condition(M)) < cond(M)

        # PCG: spec(P⁻¹A) = spec(U A Uᵀ) with P⁻¹ = UᵀU
        C = randn(6, 6)
        Pinv = C' * C + I
        U = cholesky(Symmetric(Pinv)).U
        κ_PC, λ_PA = _eig_condition(Pinv * A_spd)
        λ_sym = eigvals(Symmetric(U * A_spd * U'))
        @test κ_PC ≈ maximum(λ_sym) / minimum(λ_sym)
        @test maximum(abs.(imag.(λ_PA))) < 1e-8

        # Singular
        @test isinf(first(_eig_condition(Matrix(Diagonal([0.0, 1.0])))))
    end

    @testset "_warn_if_not_spd" begin
        @test_logs (:warn, r"TEST is not SPD") _warn_if_not_spd([-1.0, 2.0], "TEST", 1)
        @test_logs _warn_if_not_spd([1.0, 2.0], "TEST", 1)
    end

    @testset "_local_dense_matrix" begin
        i = Index(2, "i")
        j = Index(2, "j")
        op = randomITensor(i, i', j, j')
        b_tensor = randomITensor(i, j)

        A_dense, rowinds, colinds, rdim, cdim = _local_dense_matrix(op, b_tensor)
        @test rdim == 4
        @test cdim == 4
        @test size(A_dense) == (4, 4)
    end
end

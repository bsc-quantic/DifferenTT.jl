using Test
using ITensors
using ITensorMPS
using KrylovKit
using LinearAlgebra
using DifferenTT
# We are testing unexported internal utilities
using DifferenTT: _uses_cg_metric,
                  SPD_DEFINITE, SPD_AMBIGUOUS, SPD_INDEFINITE,
                  _classify_spd, _spd_tolerance, _hermitianize, _extremal_eigvals,
                  _spectral_condition, _svd_condition,
                  _extremal_hermitian_eigvals, _extremal_singular_values,
                  _report_spectral_κ, _EXTREMAL_KRYLOV_THRESHOLD,
                  _local_dense_matrix, _matrixfree_eigvals

@testset "KrylovUpdater instantiation" begin
    updater = DifferenTT.KrylovUpdater()
    @test updater isa DifferenTT.AbstractUpdater
end

@testset "KrylovUpdater condition metrics and SPD helpers" begin
    @testset "_uses_cg_metric" begin
        @test _uses_cg_metric((ishermitian=true, isposdef=true)) == true
        @test _uses_cg_metric((ishermitian=true,)) == false
        @test _uses_cg_metric((isposdef=true,)) == false
        @test _uses_cg_metric(NamedTuple()) == false
    end

    @testset "SPD Classification" begin
        # SPD_DEFINITE
        @test _classify_spd(1.0, 10.0) == SPD_DEFINITE
        @test _spectral_condition(1.0, 10.0) == 10.0

        # SPD_AMBIGUOUS
        @test _classify_spd(1e-15, 10.0) == SPD_AMBIGUOUS
        @test isnan(_spectral_condition(1e-15, 10.0))

        # SPD_INDEFINITE
        @test _classify_spd(-1.0, 10.0) == SPD_INDEFINITE
        @test isnan(_spectral_condition(-1.0, 10.0))
    end

    @testset "_svd_condition" begin
        @test _svd_condition(2.0, 10.0) == 5.0
        @test isinf(_svd_condition(0.0, 10.0))
        @test isnan(_svd_condition(-1.0, 10.0))
        @test isnan(_svd_condition(NaN, 10.0))
    end

    @testset "Extremal Eigenvalues (dense path)" begin
        # Dense Hermitian
        A = Hermitian([2.0 0.5; 0.5 2.0])
        λmin, λmax = _extremal_hermitian_eigvals(A)
        @test λmin ≈ 1.5
        @test λmax ≈ 2.5

        # Dense SVD
        B = [3.0 0.0; 0.0 -4.0]
        σmin, σmax = _extremal_singular_values(B)
        @test σmin ≈ 3.0
        @test σmax ≈ 4.0
    end

    @testset "Extremal Eigenvalues (sparse/KrylovKit path)" begin
        # We can force the Krylov path by passing krylov_threshold = 1
        A = Hermitian(Diagonal(1.0:5.0))
        λmin, λmax = _extremal_hermitian_eigvals(A; krylov_threshold=1)
        @test isapprox(λmin, 1.0; atol=1e-5)
        @test isapprox(λmax, 5.0; atol=1e-5)

        B = Diagonal(1.0:5.0)
        σmin, σmax = _extremal_singular_values(B; krylov_threshold=1)
        @test isapprox(σmin, 1.0; atol=1e-5)
        @test isapprox(σmax, 5.0; atol=1e-5)
    end

    @testset "_report_spectral_κ logging" begin
        # SPD - no warnings regardless of compute_cond
        @test _report_spectral_κ(1.0, 10.0; label="TEST", lpos=1, compute_cond=true) == 10.0
        @test _report_spectral_κ(1.0, 10.0; label="TEST", lpos=1, compute_cond=false) == 10.0

        # Indefinite - triggers error only if compute_cond=true
        @test_logs (:error, r"TEST IS NOT SPD — κ undefined") match_mode=:any begin
            @test isnan(_report_spectral_κ(-1.0, 10.0; label="TEST", lpos=1, compute_cond=true))
        end

        # When compute_cond=false, there should be no error logs
        @test_logs begin
            @test isnan(_report_spectral_κ(-1.0, 10.0; label="TEST", lpos=1, compute_cond=false))
        end

        # Ambiguous - triggers warn only if compute_cond=true
        @test_logs (:warn, r"TEST is numerically semidefinite/ambiguous — κ undefined") match_mode=:any begin
            @test isnan(_report_spectral_κ(1e-15, 10.0; label="TEST", lpos=1, compute_cond=true))
        end

        @test_logs begin
            @test isnan(_report_spectral_κ(1e-15, 10.0; label="TEST", lpos=1, compute_cond=false))
        end
    end

    @testset "Matrix Utilities" begin
        # _spd_tolerance
        @test _spd_tolerance(1.0, 10.0; rtol=1e-12) == 1e-11
        @test _spd_tolerance(-5.0, 2.0; rtol=1e-12) == 5e-12
        
        # _hermitianize
        M = [1.0 2.0; 3.0 4.0]
        H = _hermitianize(M)
        @test H == [1.0 2.5; 2.5 4.0]
        @test H isa Hermitian

        # _extremal_eigvals
        λmin, λmax = _extremal_eigvals(M)
        @test λmin ≈ (5 - sqrt(34))/2
        @test λmax ≈ (5 + sqrt(34))/2
    end

    @testset "ITensor Utilities" begin
        i = Index(2, "i")
        j = Index(2, "j")
        op = randomITensor(i, i', j, j')
        b_tensor = randomITensor(i, j)
        
        A_dense, rowinds, colinds, rdim, cdim = _local_dense_matrix(op, b_tensor)
        @test rdim == 4
        @test cdim == 4
        @test size(A_dense) == (4, 4)
        
        # _matrixfree_eigvals
        f(x) = x * 2.0
        x0 = randomITensor(i)
        λmin, λmax = _matrixfree_eigvals(f, x0; ishermitian=true)
        @test λmin ≈ 2.0
        @test λmax ≈ 2.0
    end
end

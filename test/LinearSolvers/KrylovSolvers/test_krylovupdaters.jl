using Test
using ITensors
using ITensorMPS
using KrylovKit
using LinearAlgebra
using DifferenTT
import Logging

# We are testing unexported internal utilities
using DifferenTT: _uses_cg_metric,
                  SPD_DEFINITE, SPD_AMBIGUOUS, SPD_INDEFINITE,
                  _classify_spd,
                  _spectral_condition, _svd_condition,
                  _extremal_singular_values,
                  _dense_extremal_eigvals,
                  _report_spectral_κ

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

    @testset "Extremal singular values (dense path)" begin
        B = [3.0 0.0; 0.0 -4.0]
        σmin, σmax = _extremal_singular_values(B)
        @test σmin ≈ 3.0
        @test σmax ≈ 4.0
    end

    @testset "Extremal singular values (KrylovKit path)" begin
        B = Diagonal(1.0:5.0)
        σmin, σmax = _extremal_singular_values(B; krylov_threshold=1)
        @test isapprox(σmin, 1.0; atol=1e-5)
        @test isapprox(σmax, 5.0; atol=1e-5)
    end

    @testset "_dense_extremal_eigvals (CG path)" begin
        A = [2.0 0.5; 0.5 2.0]
        λmin, λmax = _dense_extremal_eigvals(A)
        @test λmin ≈ 1.5
        @test λmax ≈ 2.5

        D = Diagonal(1.0:5.0)
        λmin, λmax = _dense_extremal_eigvals(Matrix(D))
        @test λmin ≈ 1.0
        @test λmax ≈ 5.0

        # Non-normal: spectral κ from eigenvalues, not singular values (GMRES uses the latter)
        N = [1.0 100.0; 0.0 2.0]
        λmin, λmax = _dense_extremal_eigvals(N)
        @test λmin ≈ 1.0
        @test λmax ≈ 2.0
        σmin, σmax = _extremal_singular_values(N)
        @test _svd_condition(σmin, σmax) > _spectral_condition(λmin, λmax)
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
        @test_logs min_level=Logging.Error begin
            @test isnan(_report_spectral_κ(-1.0, 10.0; label="TEST", lpos=1, compute_cond=false))
        end

        # Ambiguous - triggers warn only if compute_cond=true
        @test_logs (:warn, r"TEST is numerically semidefinite/ambiguous — κ undefined") match_mode=:any begin
            @test isnan(_report_spectral_κ(1e-15, 10.0; label="TEST", lpos=1, compute_cond=true))
        end

        @test_logs min_level=Logging.Warn begin
            @test isnan(_report_spectral_κ(1e-15, 10.0; label="TEST", lpos=1, compute_cond=false))
        end
    end
end

using Test
using LinearAlgebra
using KrylovKit
using DifferenTT

@testset "Preconditioned CG" begin
    # Setup symmetric positive definite matrix A
    A = [4.0 1.0; 1.0 3.0]
    b = [1.0, 2.0]
    x0 = [0.0, 0.0]
    # Simple preconditioner M
    M = [4.0 0.0; 0.0 3.0]
    Minv = inv(M)

    # Preconditioned CG
    alg = CG()
    x, info = DifferenTT.preconditioned_linsolve(A, b, x0, Minv, alg)

    @test info.converged > 0
    @test A * x ≈ b

    # Test a0 and a1
    x2, info2 = DifferenTT.preconditioned_linsolve(A, b, x0, Minv, alg, 1.0, -1.0)
    # solves (a₀ I + a₁ A) x = b with a₀ = 1 and a₁ = -1, i.e. (I - A) x = b
    @test info2.converged > 0
    @test (I - A) * x2 ≈ b
end

using StableRNGs

# A 60 × 60 SPD matrix, and the same matrix badly scaled by D = diag(10^0 … 10^3) (κ ≈ 1.3e6)
function spd_fixtures()
    r = StableRNG(7)
    n = 60
    B = randn(r, n, n)
    S = B' * B / n + I
    b = randn(r, n)
    D = Diagonal(exp10.(range(0, 3; length=n)))
    return S, D * S * D, b, zeros(n)
end

@testset "PCG with M⁻¹ = I reproduces KrylovKit's CG" begin
    S, _, b, x0 = spd_fixtures()
    alg = CG(; tol=1e-10, maxiter=500)
    x_kk, info_kk = KrylovKit.linsolve(S, b, x0, alg)
    @testset "M⁻¹ = $name" for (name, Minv) in (("I (matrix)", Matrix(1.0I, size(S))), ("identity", identity))
        x, info = DifferenTT.preconditioned_linsolve(S, b, x0, Minv, alg)
        @test info.converged == 1
        @test info.numiter == info_kk.numiter
        @test info.numops == info_kk.numops
        # Same iteration up to rounding (measured difference 1e-15)
        @test norm(x - x_kk) <= 1e-12 * norm(x_kk)
        @test norm(S * x - b) <= 1e-10
    end
end

@testset "A Jacobi preconditioner cuts the CG iterations on a badly scaled SPD matrix" begin
    _, Abad, b, x0 = spd_fixtures()
    alg = CG(; tol=1e-8 * norm(b), maxiter=5000)
    _, info_none = KrylovKit.linsolve(Abad, b, x0, alg)
    x, info = DifferenTT.preconditioned_linsolve(Abad, b, x0, Diagonal(1 ./ diag(Abad)), alg)
    @test info.converged == 1
    @test norm(Abad * x - b) <= 1e-7 * norm(b)
    # Measured: 403 iterations without preconditioner, 22 with Jacobi
    @test info.numiter < info_none.numiter / 5
end

@testset "PCG: early return, non-convergence and verbosity" begin
    S, _, b, x0 = spd_fixtures()
    Minv = Diagonal(1 ./ diag(S))
    x_exact = S \ b
    # From the solution, it returns without iterating
    _, info = DifferenTT.preconditioned_linsolve(S, b, x_exact, Minv, CG(; tol=1e-8))
    @test info.converged == 1 && info.numiter == 0
    # Without convergence it warns only with verbosity ≥ 1 (KrylovKit's default)
    _, info = @test_logs DifferenTT.preconditioned_linsolve(S, b, x0, Minv, CG(; tol=1e-14, maxiter=2, verbosity=0))
    @test info.converged == 0 && info.numiter == 2
    @test_logs (:warn, r"stopped without converging") DifferenTT.preconditioned_linsolve(
        S, b, x0, Minv, CG(; tol=1e-14, maxiter=2, verbosity=1)
    )
    @test_logs (:info, r"starts") (:info, r"in iteration") (:info, r"converged") match_mode = :any DifferenTT.preconditioned_linsolve(
        S, b, x0, Minv, CG(; tol=1e-8, verbosity=3)
    )
    @test_logs (:info, r"without any iterations") DifferenTT.preconditioned_linsolve(
        S, b, x_exact, Minv, CG(; tol=1e-8, verbosity=2)
    )
end

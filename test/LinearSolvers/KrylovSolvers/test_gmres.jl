using Test
using LinearAlgebra
using KrylovKit
using DifferenTT

@testset "Preconditioned GMRES" begin
    # Setup general matrix A
    A = [4.0 1.0; 2.0 3.0]
    b = [1.0, 2.0]
    x0 = [0.0, 0.0]
    # Simple preconditioner M
    M = [4.0 0.0; 0.0 3.0]
    Minv = inv(M)

    # Preconditioned GMRES
    alg = GMRES()
    x, info = DifferenTT.preconditioned_linsolve(A, b, x0, Minv, alg)

    @test info.converged > 0
    @test A * x ≈ b

    # Test a0 and a1
    x2, info2 = DifferenTT.preconditioned_linsolve(A, b, x0, Minv, alg, 1.0, -1.0)
    # solves (I - A)x = b
    @test info2.converged > 0
    @test (I - A) * x2 ≈ b
end

using StableRNGs

# A 60 × 60 nonsymmetric matrix, and an SPD one badly scaled by D = diag(10^0 … 10^3) (κ ≈ 1.3e6)
function gmres_fixtures()
    r = StableRNG(7)
    n = 60
    B = randn(r, n, n)
    S = B' * B / n + I
    b = randn(r, n)
    D = Diagonal(exp10.(range(0, 3; length=n)))
    N = S + 0.3 * triu(randn(r, n, n), 1)
    return N, D * S * D, b, zeros(n)
end

@testset "Preconditioned GMRES with M⁻¹ = I reproduces KrylovKit's GMRES" begin
    N, _, b, x0 = gmres_fixtures()
    alg = GMRES(; tol=1e-10, krylovdim=10, maxiter=50)
    x_kk, info_kk = KrylovKit.linsolve(N, b, x0, alg)
    @testset "M⁻¹ = $name" for (name, Minv) in (("I (matrix)", Matrix(1.0I, size(N))), ("identity", identity))
        x, info = DifferenTT.preconditioned_linsolve(N, b, x0, Minv, alg)
        @test info.converged == 1
        @test info.numiter == info_kk.numiter
        @test info.numops == info_kk.numops
        @test norm(x - x_kk) <= 1e-12 * norm(x_kk)
        @test norm(N * x - b) <= 1e-9
    end
end

@testset "A Jacobi preconditioner cuts the GMRES cycles on a badly scaled matrix" begin
    _, Abad, b, x0 = gmres_fixtures()
    alg = GMRES(; tol=1e-8 * norm(b), krylovdim=20, maxiter=500)
    _, info_none = KrylovKit.linsolve(Abad, b, x0, alg)
    x, info = DifferenTT.preconditioned_linsolve(Abad, b, x0, Diagonal(1 ./ diag(Abad)), alg)
    @test info.converged == 1
    @test norm(Abad * x - b) <= 1e-7 * norm(b)
    # Measured: no convergence in 500 cycles without preconditioner, 17 cycles with Jacobi
    @test info.numiter < info_none.numiter / 5
end

# A preconditioned solve that stops without converging returns the residual of the unpreconditioned system and its
# norm. (It used to return the norm of M⁻¹r, and an r that was not updated after the last restart.)
@testset "Non-converged preconditioned GMRES reports the unpreconditioned residual" begin
    _, Abad, b, x0 = gmres_fixtures()
    Minv = Diagonal(1 ./ diag(Abad))
    alg = GMRES(; tol=1e-12 * norm(b), krylovdim=3, maxiter=2)
    x, info = DifferenTT.preconditioned_linsolve(Abad, b, x0, Minv, alg)
    @test info.converged == 0
    @test info.residual ≈ b - Abad * x rtol = 1e-10
    @test info.normres ≈ norm(b - Abad * x) rtol = 1e-10
    # The preconditioned norm is far from it here, so the two cannot be confused
    @test !isapprox(norm(Minv * (b - Abad * x)), info.normres; rtol=0.5)
end

@testset "Preconditioned GMRES: early return and verbosity" begin
    N, _, b, x0 = gmres_fixtures()
    Minv = Diagonal(1 ./ diag(N))
    x_exact = N \ b
    _, info = DifferenTT.preconditioned_linsolve(N, b, x_exact, Minv, GMRES(; tol=1e-8))
    @test info.converged == 1 && info.numiter == 0
    @test_logs (:info, r"without any iterations") DifferenTT.preconditioned_linsolve(
        N, b, x_exact, Minv, GMRES(; tol=1e-8, verbosity=2)
    )
    @test_logs (:info, r"starts") (:info, r"in iteration") (:info, r"converged") match_mode = :any DifferenTT.preconditioned_linsolve(
        N, b, x0, Minv, GMRES(; tol=1e-8, krylovdim=5, verbosity=3)
    )
end

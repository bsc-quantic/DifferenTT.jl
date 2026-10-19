using Test
using DifferenTT
using ITensors
using ITensorMPS
using LinearAlgebra
using Logging
using DifferenTT: QRUpdater

include(joinpath(@__DIR__, "..", "testutils.jl"))
using .TestUtils

κ(M) = (λ=eigvals(M); maximum(abs, λ) / minimum(abs, λ))
cg_args(; kwargs...) = LocalUpdaterArgs(; ishermitian=true, isposdef=true, kwargs...)

# The identities of Sec. 3, Lemma 1 and Appendix B of Torrente et al. hold for the exact solution of Eq. (22). We get it
# with exact (QR) local solves on a 2⁴-point QTT Laplacian: the vectorized unknown has 4 sites of dimension 4, so
# 2-site ALS with bond dimension 16 represents it exactly.
@testset "FAI identities for the exact solution of Eq. (22)" begin
    A, sites, _, _ = qtt_laplacian((4,))
    Ad = todense(A)
    L = cholesky(Symmetric(Ad)).L
    exact = ALS(; updater=QRUpdater(), sweep_kwargs=SweepingArgs(; nsweeps=3, maxdim=16, cutoff=1e-15))
    G̃, D = DifferenTT.fai_dg(A, 1.0, exact, (; cutoff=1e-15); rng=rng(1))
    G̃d, Dd = todense(G̃), todense(D)

    # G̃_L is lower-triangular in the ordering of contract_mpo, and solves Eq. (10): (G̃_L A)_ij = δ_ij for i ≥ j.
    # (Before the fix of the vectorization, it solved (A G̃_L)_ij = δ_ij instead, the transposed FSAI.)
    @test norm(triu(G̃d, 1)) <= 1e-10 * norm(G̃d)
    @test norm(tril(G̃d * Ad) - I) <= 1e-10

    # Lemma 1(a): diag(G̃_L)_i = 1/(L_A)²_ii, and 1/A_ii ≤ g_i ≤ 1/λ_min(A)
    g = diag(G̃d)
    @test g ≈ 1 ./ diag(L) .^ 2 rtol = 1e-10
    @test all(1 ./ diag(Ad) .<= g .* (1 + 1e-10))
    @test all(g .<= (1 + 1e-10) / eigmin(Symmetric(Ad)))

    # D = diag(G̃_L)^(-1/2) (Eq. (23)) is diagonal and positive
    @test norm(Dd - Diagonal(diag(Dd))) == 0
    @test diag(Dd) ≈ g .^ (-1 / 2) rtol = 1e-8

    # G_L = D G̃_L = L_A⁻¹ (Appendix B.3), so G_Lᵀ G_L = A⁻¹
    G = Dd * G̃d
    @test G ≈ inv(L) rtol = 1e-8
    @test κ(G' * G * Ad) < 1 + 1e-6

    # The exact solution does not depend on μ (Appendix B.1)
    G̃_μ, _ = DifferenTT.fai_dg(A, 1e7, exact, (; cutoff=1e-15); rng=rng(1))
    @test todense(G̃_μ) ≈ G̃d rtol = 1e-8
end

@testset "fai_preconditioner" begin
    A, sites, _, _ = tt_laplacian((8, 8))
    Ad = todense(A)
    # The setting of the plan: μ = 1, 4 ALS sweeps with bond dimension 16 and a CG local solver, totalmaxdim 16.
    # Measured: κ(A) = 32.2, κ(GᵀGA) = 1.00 (it was 3.3 with the transposed FSAI).
    solver = ALS(; sweep_kwargs=SweepingArgs(; nsweeps=4, maxdim=16), updater_kwargs=cg_args())
    F = fai_preconditioner(A; μ=1.0, solver, totalmaxdim=16, rng=rng(1))
    G = todense(F.operator)
    M⁻¹ = G' * G
    @test F isa FactorizedPreconditioner
    @test maxlinkdim(F.operator) <= 16
    @test issymmetric(M⁻¹)
    @test eigmin(Symmetric(M⁻¹)) > 0
    @test κ(M⁻¹ * Ad) <= κ(Ad) / 4
    @test κ(M⁻¹ * Ad) < 1.1

    # totalmaxdim caps the bond dimension of G_L; with χ_G = 4 the preconditioner is still SPD and effective
    # (measured κ = 1.02)
    F4 = fai_preconditioner(A; μ=1.0, solver, totalmaxdim=4, rng=rng(1))
    G4 = todense(F4.operator)
    @test maxlinkdim(F4.operator) <= 4
    @test eigmin(Symmetric(G4' * G4)) > 0
    @test κ(G4' * G4 * Ad) <= κ(Ad) / 4

    # Reproducible with a seeded rng (the sketches of D are the only randomness)
    @test todense(fai_preconditioner(A; μ=1.0, solver, totalmaxdim=16, rng=rng(1)).operator) == G

    # The preconditioned build (rank-1 preconditioner of the projected system) gives a preconditioner of the same
    # quality
    Fp = fai_preconditioner(A; μ=1.0, solver, totalmaxdim=16, preconditioning=true, rng=rng(1))
    Gp = todense(Fp.operator)
    @test κ(Gp' * Gp * Ad) < 1.1

    # residual_tol stops the build once the relative residual of Eq. (22) is below it, which compute_residual logs
    loose = ALS(; sweep_kwargs=SweepingArgs(; nsweeps=10, maxdim=16), updater_kwargs=cg_args(; rtol=1e-6))
    logs, _ = Test.collect_test_logs(; min_level=Logging.Info) do
        fai_preconditioner(A; μ=1.0, solver=loose, residual_tol=1e-1, compute_residual=true, rng=rng(1))
    end
    residuals = [parse(Float64, match(r"rel residual = (\S+)", string(l.message))[1]) for l in logs]
    @test length(residuals) == 1
    @test only(residuals) < 1e-1

    # On a QTT Laplacian with κ(A) = 1712, a FAI of bond dimension 4 built by MALS (measured κ = 1.49)
    Aq, _, _, _ = qtt_laplacian((6,))
    Aqd = todense(Aq)
    Fq = fai_preconditioner(Aq; μ=1.0, solver, totalmaxdim=4, rng=rng(1))
    Gq = todense(Fq.operator)
    @test eigmin(Symmetric(Gq' * Gq)) > 0
    @test κ(Gq' * Gq * Aqd) <= κ(Aqd) / 100
end

using DifferenTT
using Test
using ITensors, ITensorMPS
using DifferenTT: QRUpdater, ALS

@testset "Exact Updaters" begin
    n = 4
    s = siteinds("S=1/2", n)
    A = random_mpo(s)
    b = random_mps(s)

    updater = QRUpdater()
    solver = ALS(; updater=updater, sweep_kwargs=DifferenTT.SweepingArgs(; nsweeps=1))

    x = linearsolve(solver, A, b, nothing)
    @test x isa MPS
    @test length(x) == n
end

include(joinpath(@__DIR__, "..", "testutils.jl"))
using .TestUtils
using LinearAlgebra: norm

# QRUpdater solves each local problem exactly, so 2-site ALS with it reaches the dense solution, and it records one
# iteration per local solve and a condition number per local problem
@testset "QRUpdater gives the dense solution" begin
    A, sites, _, _ = qtt_laplacian((4,))
    b = random_mps(rng(1), sites; linkdims=2)
    x_ref = todense(A) \ todense(b)
    numiter_store, cond_store = DifferenTT.KrylovSweepStats(), DifferenTT.KrylovCondStats()
    solver = ALS(;
        updater=QRUpdater(),
        sweep_kwargs=DifferenTT.SweepingArgs(; nsweeps=3, maxdim=8, cutoff=1e-14),
        updater_kwargs=DifferenTT.LocalUpdaterArgs(; numiter_store, cond_store),
    )
    x = linearsolve(solver, A, b, nothing)
    @test norm(todense(x) - x_ref) < 1e-10 * norm(x_ref)
    @test all(n == 1 for p in DifferenTT.krylov_numiters(numiter_store) for s in p for n in s)
    @test length.(DifferenTT.krylov_conds(cond_store)) == length.(DifferenTT.krylov_numiters(numiter_store))

    # The shifted local problem (a₀ + a₁A_loc)x = b_loc
    problem = ITensorMPS.ReducedLinearProblem(A, b)
    ITensorMPS.set_nsite!(problem, 2)
    x0 = DifferenTT.default_linearsolve_init(b)
    orthogonalize!(x0, 1)
    ITensorMPS.position!(problem, x0, 1)
    loc, (; info) = QRUpdater()(problem, x0[1] * x0[2]; internal_kwargs=(;), coefficients=(2.0, 3.0))
    @test info.numiter == 1
    @test info.residual < 1e-12
end

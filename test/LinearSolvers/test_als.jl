using DifferenTT
using Test
using ITensors, ITensorMPS
using DifferenTT: ALS

@testset "ALS Solver" begin
    n = 4
    s = siteinds("S=1/2", n)

    # Small random MPS/MPO
    A = random_mpo(s)
    b = random_mps(s)

    solver = ALS()
    x = linearsolve(solver, A, b, nothing; nsweeps=1)

    @test x isa MPS
    @test length(x) == n

    # With preconditioning
    M⁻¹ = DifferenTT.rank1_preconditioner(A)
    x_precond = linearsolve(solver, A, b, M⁻¹, nothing; nsweeps=1)
    @test x_precond isa MPS
    @test length(x_precond) == n

    # With local preconditioning
    x_local_precond = linearsolve(solver, A, b, M⁻¹, nothing; nsweeps=1, preconditioner_sites=2:3)
    @test x_local_precond isa MPS
    @test length(x_local_precond) == n

    # An index outside 1:length(b) would match no block and silently solve every block unpreconditioned
    @test_throws ArgumentError linearsolve(solver, A, b, M⁻¹, nothing; nsweeps=1, preconditioner_sites=0:2)
    @test_throws ArgumentError linearsolve(solver, A, b, M⁻¹, nothing; nsweeps=1, preconditioner_sites=5:6)

    # With local preconditioning and condition number tracking: the blocks outside `preconditioner_sites`
    # record their unpreconditioned κ(A) in `pc_cond_store` too, so both stores cover the same positions
    cond_store = DifferenTT.KrylovCondStats()
    pc_cond_store = DifferenTT.KrylovCondStats()
    cond_solver = ALS(
        updater_kwargs=DifferenTT.LocalUpdaterArgs(
            ishermitian=false, isposdef=false, cond_store=cond_store, pc_cond_store=pc_cond_store
        ),
    )
    linearsolve(cond_solver, A, b, M⁻¹, nothing; nsweeps=2, preconditioner_sites=2:3)

    conds = DifferenTT.krylov_conds(cond_store)
    pc_conds = DifferenTT.krylov_conds(pc_cond_store)
    @test length(pc_conds) == length(conds)
    @test all(length(pc_conds[pos]) == length(conds[pos]) for pos in eachindex(conds))
    for pos in eachindex(conds)
        if pos in 2:3
            # rank1_preconditioner of a random MPO is not a multiple of the identity, so a block that
            # applied it records κ(P⁻¹A) ≠ κ(A). Equality here would mean the window was ignored.
            @test !isempty(Iterators.flatten(pc_conds[pos]))
            @test pc_conds[pos] != conds[pos]
        else
            @test pc_conds[pos] == conds[pos]
        end
    end

    # Sign-aware local spectra: the identity preconditioner is SPD, -Id is negative definite, and the local blocks
    # are congruences of them, so their inertia must be recovered at every visited block
    spd_A = MPO(s, "Id")
    for (scale, negative) in ((1.0, false), (-1.0, true))
        spectrum_store = DifferenTT.KrylovSpectrumStats()
        spectrum_solver = ALS(updater_kwargs=DifferenTT.LocalUpdaterArgs(spectrum_store=spectrum_store))
        linearsolve(spectrum_solver, spd_A, b, Preconditioner(scale * MPO(s, "Id")), nothing; nsweeps=2)

        table = DifferenTT.spectrum_table(spectrum_store)
        @test !isempty(table)
        @test maximum(table["sweep"]) == 2
        @test all(table["eigmin_A"] .> 0)
        @test all((table["nneg_P"] .== table["blockdim"]) .== negative)
        @test all((table["nneg_PA"] .== table["blockdim"]) .== negative)
    end

    # spd_tol alone (no condition-number store) stops the solve at the first indefinite local P⁻¹A, and an SPD
    # preconditioner runs through
    spd_kwargs = (; ishermitian=true, isposdef=true, spd_tol=1e-8, krylovdim=1, maxiter=50)
    spd_solver = ALS(updater_kwargs=DifferenTT.LocalUpdaterArgs(; spd_kwargs...))
    @test_throws DifferenTT.NotSPDError linearsolve(
        spd_solver, spd_A, b, Preconditioner(-1.0 * MPO(s, "Id")), nothing; nsweeps=2
    )
    @test linearsolve(spd_solver, spd_A, b, Preconditioner(MPO(s, "Id")), nothing; nsweeps=2) isa MPS

    # Blocks above max_blockdim are recorded without a spectrum
    skip_store = DifferenTT.KrylovSpectrumStats(max_blockdim=1)
    skip_solver = ALS(updater_kwargs=DifferenTT.LocalUpdaterArgs(spectrum_store=skip_store))
    linearsolve(skip_solver, spd_A, b, nothing; nsweeps=1)
    @test all(isnan, DifferenTT.spectrum_table(skip_store)["eigmin_A"])
end

include(joinpath(@__DIR__, "..", "testutils.jl"))
using .TestUtils
using LinearAlgebra: norm

# Plain ALS (E5): the 2-site and the 1-site variants solve an SPD system to the dense solution
@testset "ALS solves an SPD system" begin
    A, sites, _, _ = qtt_laplacian((5,))
    b = random_mps(rng(1), sites; linkdims=2)
    x_ref = todense(A) \ todense(b)
    local_args = DifferenTT.LocalUpdaterArgs(; ishermitian=true, isposdef=true, rtol=1e-12)
    x2 = linearsolve(
        ALS(; sweep_kwargs=DifferenTT.SweepingArgs(; nsweeps=4, maxdim=16, cutoff=1e-14), updater_kwargs=local_args),
        A,
        b,
        nothing,
    )
    @test relres(A, x2, b) < 1e-9
    @test norm(todense(x2) - x_ref) < 1e-8 * norm(x_ref)

    # The 1-site variant keeps the bond dimension of its initial guess, so it needs one with enough rank
    x0 = random_mps(rng(2), sites; linkdims=8)
    x1 = linearsolve(
        ALS(; sweep_kwargs=DifferenTT.SweepingArgs(; nsweeps=6, nsite=1), updater_kwargs=local_args), A, b, x0
    )
    @test all(linkdims(x1) .<= linkdims(x0))
    @test relres(A, x1, b) < 1e-9
end

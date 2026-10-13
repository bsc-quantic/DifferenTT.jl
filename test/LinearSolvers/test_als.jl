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
    cond_solver = ALS(updater_kwargs = DifferenTT.LocalUpdaterArgs(ishermitian=false, isposdef=false, cond_store=cond_store, pc_cond_store=pc_cond_store))
    linearsolve(cond_solver, A, b, M⁻¹, nothing; nsweeps=2, preconditioner_sites=2:3)

    conds = DifferenTT.krylov_conds(cond_store)
    pc_conds = DifferenTT.krylov_conds(pc_cond_store)
    @test length(pc_conds) == length(conds)
    @test all(length(pc_conds[pos]) == length(conds[pos]) for pos in eachindex(conds))
    for pos in eachindex(conds)
        pos in 2:3 && continue
        @test pc_conds[pos] == conds[pos]
    end

    # Sign-aware local spectra: the identity preconditioner is SPD, -Id is negative definite, and the local blocks
    # are congruences of them, so their inertia must be recovered at every visited block
    spd_A = MPO(s, "Id")
    for (scale, negative) in ((1.0, false), (-1.0, true))
        spectrum_store = DifferenTT.KrylovSpectrumStats()
        spectrum_solver = ALS(updater_kwargs = DifferenTT.LocalUpdaterArgs(spectrum_store=spectrum_store))
        linearsolve(spectrum_solver, spd_A, b, Preconditioner(scale * MPO(s, "Id")), nothing; nsweeps=2)

        table = DifferenTT.spectrum_table(spectrum_store)
        @test !isempty(table)
        @test maximum(table["sweep"]) == 2
        @test all(table["eigmin_A"] .> 0)
        @test all((table["nneg_P"] .== table["blockdim"]) .== negative)
        @test all((table["nneg_PA"] .== table["blockdim"]) .== negative)
    end

    # With spd_tol set, the SPD check of compute_cond stops the solve at the first indefinite local P⁻¹A, and an SPD
    # preconditioner runs through
    spd_kwargs = (; ishermitian=true, isposdef=true, compute_cond=true, spd_tol=1e-8, krylovdim=1, maxiter=50)
    spd_solver = ALS(updater_kwargs = DifferenTT.LocalUpdaterArgs(; spd_kwargs...))
    @test_throws DifferenTT.NotSPDError linearsolve(spd_solver, spd_A, b, Preconditioner(-1.0 * MPO(s, "Id")), nothing; nsweeps=2)
    @test linearsolve(spd_solver, spd_A, b, Preconditioner(MPO(s, "Id")), nothing; nsweeps=2) isa MPS

    # Blocks above max_blockdim are recorded without a spectrum
    skip_store = DifferenTT.KrylovSpectrumStats(max_blockdim=1)
    skip_solver = ALS(updater_kwargs = DifferenTT.LocalUpdaterArgs(spectrum_store=skip_store))
    linearsolve(skip_solver, spd_A, b, nothing; nsweeps=1)
    @test all(isnan, DifferenTT.spectrum_table(skip_store)["eigmin_A"])
end

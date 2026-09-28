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
end

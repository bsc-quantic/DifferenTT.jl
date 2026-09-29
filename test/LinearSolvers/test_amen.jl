using DifferenTT
using Test
using ITensors: ITensors, siteinds, OpSum
using ITensorMPS
using DifferenTT: AMEn, ALS, linearsolve, SweepingArgs, LocalUpdaterArgs, KrylovUpdater, maxlinkdim
using DifferenTT: FactorizedPreconditioner
using LinearAlgebra: norm, normalize!
using Random: Random

# Counts the sweeps performed by an alternating update
mutable struct AMEnSweepCounter <: ITensorMPS.AbstractObserver
    nsweeps::Int
end
ITensorMPS.update_observer!(o::AMEnSweepCounter; kwargs...) = (o.nsweeps += 1; o)

# -Δ (Dirichlet) on a 2D QTT grid with `nx + ny` sites
function amen_poisson_2d(nx, ny)
    arch = QTTArchitecture((nx, ny); interleaved = false)
    grid = Grid(arch, (1.0, 1.0))
    bc1 = BoundaryCond1d(DirichletBC(0.0), DirichletBC(0.0))
    s = siteinds(arch)
    field = ScalarField(zeros_mps(s), arch, BoundaryConditions(bc1, bc1))
    return -1.0 * symmetrized_laplacian(field, grid), s
end

# Exact relative residual ‖A x - b‖ / ‖b‖ (no truncation in the product)
amen_relres(A, x, b) = norm(apply(A, x; alg = "naive", truncate = false) - b) / norm(b)

@testset "AMEn functionality" begin
    @testset "AMEn creation" begin
        sweep_kwargs = SweepingArgs(nsite=1)
        updater_kwargs = LocalUpdaterArgs()
        
        solver = AMEn(
            sweep_kwargs = sweep_kwargs,
            updater_kwargs = updater_kwargs,
            updater = KrylovUpdater(),
            enrichment_maxdim = 3,
            enrichment_cutoff = 1e-8
        )
        @test solver isa DifferenTT.AMEn
        @test solver.enrichment_maxdim == 3
    end

    @testset "AMEn residual-based defaults" begin
        solver = AMEn()
        @test solver.residual_tol == 1e-8
        @test solver.residual_damp == 1.1
        @test isnothing(solver.kick_tol)
        @test solver.local_stop
    end

    @testset "AMEn Integration with 1D Laplacian" begin
        # Build a 1D Laplacian MPO on 5 QTT sites with Dirichlet BCs
        N = 5
        arch = QTTArchitecture((N,))
        grid = Grid(arch, (1.0,))
        bc = BoundaryConditions(BoundaryCond1d(DirichletBC(0.0), DirichletBC(0.0)))
        s = siteinds(arch)
        dummy_mps = zeros_mps(s)
        field = ScalarField(dummy_mps, arch, bc)
        
        A = -1.0 * symmetrized_laplacian(field, grid)
        b = random_mps(s; linkdims=2)
        x0 = random_mps(s; linkdims=1) # start with low bond dimension
        
        # Solve with AMEn
        solver = AMEn(
            sweep_kwargs = SweepingArgs(nsite=1, nsweeps=3, maxdim=10, outputlevel=0),
            updater_kwargs = LocalUpdaterArgs(),
            updater = KrylovUpdater(),
            enrichment_maxdim = 2,
            enrichment_cutoff = 1e-8
        )
        
        x_sol = linearsolve(solver, A, b, x0)
        
        @test x_sol isa MPS
        @test length(x_sol) == N
        # We expect the max link dimension to have grown from 1 to >1
        @test maxlinkdim(x_sol) > 1
        
        # We can't use direct contract(A, x) - b because ITensor contract might return MPO * MPS as MPS or ITensor.
        # But we can just use inner to compute residual
        normb2 = norm(b)^2
        normAx2 = ITensors.inner(A, x_sol, A, x_sol)
        Ax_b = conj(ITensors.inner(b', A, x_sol))
        res2 = normAx2 - 2 * real(Ax_b) + normb2
        res1 = res2 > 0 ? sqrt(res2) : 0.0
        
        normAx02 = ITensors.inner(A, x0, A, x0)
        Ax0_b = conj(ITensors.inner(b', A, x0))
        res02 = normAx02 - 2 * real(Ax0_b) + normb2
        res0 = res02 > 0 ? sqrt(res02) : 0.0
        
        @test res1 < res0
    end

    @testset "Preconditioned linearsolve" begin
        n = 4
        sites = siteinds("S=1/2", n)
        A = MPO(sites, "Id")
        b = random_mps(sites; linkdims=2)

        solver = AMEn(
            sweep_kwargs = SweepingArgs(nsweeps=2, maxdim=4, cutoff=1e-8),
        )
        M_prec = rank1_preconditioner(A)
        x = linearsolve(solver, A, b, M_prec, nothing)
        @test x isa MPS
        @test norm(apply(A, x) - b) < 1e-6
    end

    @testset "AMEn on a 2D Poisson QTT problem" begin
        Random.seed!(1234)
        A, s = amen_poisson_2d(5, 5)
        b = random_mps(s; linkdims = 4)
        normalize!(b)
        updater_kwargs = LocalUpdaterArgs(ishermitian = true, isposdef = true, rtol = 1e-8, krylovdim = 1, maxiter = 2000)
        amen(; nsweeps = 10, maxdim = 100, cutoff = 1e-8, kwargs...) = AMEn(;
            sweep_kwargs = SweepingArgs(nsweeps = nsweeps, maxdim = maxdim, cutoff = cutoff),
            updater_kwargs, enrichment_maxdim = 3, residual_tol = 1e-6, kwargs...
        )

        @testset "Converges and returns an orthogonal state" begin
            x = linearsolve(amen(), A, b, nothing)
            @test amen_relres(A, x, b) < 1e-5
            @test ITensorMPS.isortho(x)
        end

        @testset "Truncation is independent of the scale of the solution" begin
            # The unknown of the Chorin pipeline is p·dt, so small solutions are the norm
            x1 = linearsolve(amen(), A, b, nothing)
            x2 = linearsolve(amen(), A, 1e-3 * b, nothing)
            x3 = linearsolve(amen(), A, 1e3 * b, nothing)
            @test amen_relres(A, x1, b) < 1e-5
            @test amen_relres(A, x2, 1e-3 * b) < 1e-5
            @test amen_relres(A, x3, 1e3 * b) < 1e-5
            @test maxlinkdim(x1) > 1
            @test maxlinkdim(x2) == maxlinkdim(x1)
            @test maxlinkdim(x3) == maxlinkdim(x1)
        end

        @testset "The rank does not grow once converged" begin
            solver = amen(nsweeps = 2, local_stop = false)
            x = linearsolve(amen(), A, b, nothing)
            x_more = linearsolve(solver, A, b, x)
            @test maxlinkdim(x_more) <= maxlinkdim(x) + solver.enrichment_maxdim
            @test amen_relres(A, x_more, b) < 1e-5
        end

        @testset "No enrichment: the rank cannot grow" begin
            x0 = random_mps(s; linkdims = 1)
            x = linearsolve(amen(kick_tol = Inf), A, b, x0)
            @test maxlinkdim(x) == 1
        end

        @testset "Local stopping rule" begin
            for (local_stop, stops_early) in ((true, true), (false, false))
                counter = AMEnSweepCounter(0)
                solver = AMEn(;
                    sweep_kwargs = SweepingArgs(nsweeps = 30, maxdim = 100, cutoff = 1e-8, sweep_observer! = counter),
                    updater_kwargs, enrichment_maxdim = 3, residual_tol = 1e-4, local_stop,
                )
                x = linearsolve(solver, A, b, nothing)
                @test (counter.nsweeps < 30) == stops_early
                @test amen_relres(A, x, b) < 1e-3
            end
        end

        @testset "Zero right-hand side" begin
            x = linearsolve(amen(), A, 0.0 * b, nothing)
            @test norm(x) < 1e-10
        end

        @testset "Preconditioned solves" begin
            M = rank1_preconditioner(A)
            x = linearsolve(amen(), A, b, M, nothing)
            @test amen_relres(A, x, b) < 1e-5
            @test ITensorMPS.isortho(x)

            # Only some of the blocks are preconditioned
            x = linearsolve(amen(), A, b, M, nothing; preconditioner_sites = 3:6)
            @test amen_relres(A, x, b) < 1e-5

            # Factorized preconditioner M⁻¹ = Gₗᵀ Gₗ (any Gₗ gives a valid solve, the environments of the
            # preconditioner just have to follow the sweeps)
            Gl = FactorizedPreconditioner(M.operator)
            x = linearsolve(amen(), A, b, Gl, nothing)
            @test amen_relres(A, x, b) < 1e-5
            @test ITensorMPS.isortho(x)
        end
    end

    @testset "AMEn honours maxdim and is competitive with 2-site ALS" begin
        Random.seed!(1)
        A, s = amen_poisson_2d(6, 6)
        b = random_mps(s; linkdims = 4)
        normalize!(b)
        maxdim = 24
        updater_kwargs = LocalUpdaterArgs(ishermitian = true, isposdef = true, rtol = 1e-7, krylovdim = 1, maxiter = 2000)
        sweep_kwargs = SweepingArgs(nsweeps = 8, maxdim = maxdim, cutoff = 0.0)

        x_amen = linearsolve(AMEn(; sweep_kwargs, updater_kwargs, enrichment_maxdim = 3, residual_tol = 1e-6, local_stop = false), A, b, nothing)
        x_als = linearsolve(ALS(; sweep_kwargs, updater_kwargs), A, b, nothing)
        @test maxlinkdim(x_amen) <= maxdim
        @test ITensorMPS.isortho(x_amen)
        @test amen_relres(A, x_amen, b) <= 2 * amen_relres(A, x_als, b)
    end
end

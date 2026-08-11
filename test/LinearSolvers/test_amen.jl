using DifferenTT
using Test
using ITensors: ITensors, siteinds, OpSum
using ITensorMPS: MPS, MPO, random_mps
using DifferenTT: AMEn, linearsolve, SweepingArgs, LocalUpdaterArgs, KrylovUpdater, maxlinkdim
using LinearAlgebra: norm

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
end

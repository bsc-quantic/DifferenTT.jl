using DifferenTT
using Test
using ITensors: ITensors, siteinds, OpSum
using ITensorMPS: MPS, MPO, random_mps
using DifferenTT:
    ExpandedALS,
    NullExpansion,
    RandomSPSDexpansion,
    RandomLTexpansion,
    ResidualExpansion,
    linearsolve,
    SweepingArgs,
    LocalUpdaterArgs,
    KrylovUpdater,
    QRUpdater

include(joinpath(@__DIR__, "..", "testutils.jl"))
using .TestUtils

@testset "ExpandedALS functionality" begin
    @testset "ExpandedALS creation" begin
        sweep_kwargs_1 = SweepingArgs(nsite=1)
        updater_kwargs = LocalUpdaterArgs()

        # Test creation with correct nsite
        solver = ExpandedALS(sweep_kwargs_1, updater_kwargs, NullExpansion(), KrylovUpdater())
        @test solver isa DifferenTT.ExpandedALS

        # Test creation throws if nsite != 1
        sweep_kwargs_2 = SweepingArgs(nsite=2)
        @test_throws ArgumentError ExpandedALS(sweep_kwargs_2, updater_kwargs, NullExpansion(), KrylovUpdater())
    end

    @testset "Expansions" begin
        # Create dummy state
        sites = siteinds(4, 3)
        state = random_mps(sites)

        # Test NullExpansion
        null_exp = NullExpansion()
        @test null_exp(state) === state

        # Testing RandomSPSDexpansion and RandomLTexpansion requires 
        # random_mpo and lower_triang_mpo from DifferenTT which should be available
        spsd_exp = RandomSPSDexpansion()
        spsd_mps = spsd_exp(state)
        @test spsd_mps isa MPS
        @test length(spsd_mps) == 3

        lt_exp = RandomLTexpansion()
        lt_mps = lt_exp(state)
        @test lt_mps isa MPS
        @test length(lt_mps) == 3

        # Test ResidualExpansion
        os_dummy = OpSum()
        for i in 1:2
            os_dummy += 1.0, "I", i, "I", i+1
        end
        A_dummy = MPO(os_dummy, sites)
        b_dummy = random_mps(sites)

        # Test creation and parameters
        res_exp = ResidualExpansion(A_dummy, b_dummy)
        @test res_exp.cutoff == 1e-8
        @test res_exp.maxdim == 5

        res_exp_custom = ResidualExpansion(A_dummy, b_dummy; cutoff=1e-5, maxdim=2)
        @test res_exp_custom.cutoff == 1e-5
        @test res_exp_custom.maxdim == 2

        # Test expansion functor
        expanded_mps = res_exp(state)
        @test expanded_mps isa MPS
        @test length(expanded_mps) == length(state)
    end

    @testset "ResidualExpansion Integration with 1D Laplacian" begin
        # Build a 1D Laplacian MPO on 5 QTT sites with Dirichlet BCs
        arch = QTTArchitecture((5,))
        grid = Grid(arch, (1.0,))
        bc = BoundaryConditions(BoundaryCond1d(DirichletBC(0.0), DirichletBC(0.0)))
        s = siteinds(arch)
        dummy_mps = zeros_mps(s)
        field = ScalarField(dummy_mps, arch, bc)

        A = -1.0 * symmetrized_laplacian(field, grid)
        b = random_mps(s; linkdims=2)
        x0 = random_mps(s; linkdims=2)

        # Solve with ExpandedALS and ResidualExpansion
        solver = ExpandedALS(
            sweep_kwargs=SweepingArgs(nsite=1, nsweeps=2, maxdim=10),
            updater_kwargs=LocalUpdaterArgs(),
            expander_func=ResidualExpansion(A, b),
            updater=KrylovUpdater(),
        )

        x_sol = linearsolve(solver, A, b, x0)

        @test x_sol isa MPS
        @test length(x_sol) == length(s)

        # Preconditioned path
        x_precond = linearsolve(solver, A, b, DifferenTT.rank1_preconditioner(A), x0)
        @test x_precond isa MPS
        @test length(x_precond) == length(s)
    end
end

using ITensorMPS: linkdims, maxlinkdim, siteinds as mps_siteinds
using LinearAlgebra: norm

# Every expansion returns an MPS on the same sites with larger bond dimensions; ResidualExpansion also keeps the vector
@testset "Expansions keep the sites and increase the bond dimension" begin
    sites = siteinds(4, 4)
    state = random_mps(rng(1), sites; linkdims=2)
    @testset "$name" for (name, make) in (
        ("RandomSPSDexpansion", r -> RandomSPSDexpansion(; rng=r)),
        ("RandomLTexpansion", r -> RandomLTexpansion(; rng=r)),
    )
        expanded = make(rng(2))(state)
        @test mps_siteinds(expanded) == sites
        @test all(linkdims(expanded) .> linkdims(state))
        # Seeded expansions are reproducible
        @test todense(make(rng(2))(state)) ≈ todense(expanded)
    end

    A = MPO(sites, "Id") + MPO(sites, "Id")
    b = random_mps(rng(4), sites; linkdims=3)
    expansion = ResidualExpansion(A, b; maxdim=2, cutoff=1e-12)
    expanded = expansion(state)
    @test mps_siteinds(expanded) == sites
    @test all(linkdims(state) .< linkdims(expanded) .<= linkdims(state) .+ 2)
    # The first core of the residual is zeroed, so the direct sum still represents the state
    @test todense(expanded) ≈ todense(state) rtol = 1e-12

    @test_throws ArgumentError ResidualExpansion(A, random_mps(rng(5), siteinds(4, 4)))
    @test_throws ArgumentError ExpandedALS(; sweep_kwargs=SweepingArgs(; nsite=2))
end

@testset "ExpandedALS with ResidualExpansion grows the rank of a rank-1 guess" begin
    A, sites, _, _ = qtt_laplacian((5,))
    b = random_mps(rng(1), sites; linkdims=3)
    x_ref = todense(A) \ todense(b)
    local_args = LocalUpdaterArgs(; ishermitian=true, isposdef=true, rtol=1e-10)
    plain = ExpandedALS(; sweep_kwargs=SweepingArgs(; nsite=1, nsweeps=6, maxdim=16), updater_kwargs=local_args)
    expanded = ExpandedALS(;
        sweep_kwargs=SweepingArgs(; nsite=1, nsweeps=6, maxdim=16),
        updater_kwargs=local_args,
        expander_func=ResidualExpansion(A, b; maxdim=3),
    )
    # From the zero guess of bond dimension 1, the one-site ALS cannot leave rank 1, while the expansion converges
    x_plain = linearsolve(plain, A, b, nothing)
    x_expanded = linearsolve(expanded, A, b, nothing)
    @test maxlinkdim(x_plain) == 1
    @test maxlinkdim(x_expanded) > 1
    @test norm(todense(x_expanded) - x_ref) < 1e-6 * norm(x_ref)
    @test norm(todense(x_plain) - x_ref) > 1e-2 * norm(x_ref)
end

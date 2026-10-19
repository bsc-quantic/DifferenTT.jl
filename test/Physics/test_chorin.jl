using DifferenTT
using Test
using ITensors
using ITensorMPS

include(joinpath(@__DIR__, "..", "testutils.jl"))
using .TestUtils

@testset "NavierStokes & Chorin" begin
    arch_2d = (4, 4)
    arch_3d = (4, 4, 4)

    # Create proper boundary conditions using actual types
    periodic_bc = PeriodicBC()
    bc1d = BoundaryCond1d(periodic_bc, periodic_bc)
    bc2d = BoundaryConditions(bc1d, bc1d)
    bc3d = BoundaryConditions(bc1d, bc1d, bc1d)

    @testset "Equation Terms 2D" begin
        # 
        sites = siteinds(TTArchitecture(arch_2d))
        ux = random_mps(sites)
        uy = random_mps(sites)
        p = random_mps(sites)

        u_fields = VectorField((ux, uy), TTArchitecture(arch_2d), (bc2d, bc2d))
        p_field = ScalarField(p, TTArchitecture(arch_2d), bc2d)
        grid = Grid(TTArchitecture(arch_2d), (1.0, 1.0))

        # Differential Operators
        u_diff_ops = DifferenTT.DifferentialOps(grid, u_fields; stencil_order=2)
        p_diff_ops = DifferenTT.DifferentialOpsScalar(grid, p_field; stencil_order=2)

        # Test gradient
        grad_p = gradient(p, p_diff_ops)
        @test length(grad_p) == 2
        @test grad_p[1] isa MPS
        @test grad_p[2] isa MPS

        # Test divergence
        div_u = divergence(last(u_fields), u_diff_ops)
        @test div_u isa MPS

        # Test advection
        adv_u = advection(last(u_fields), u_diff_ops)
        @test length(adv_u) == 2
        @test adv_u[1] isa MPS
        @test adv_u[2] isa MPS

        # Test diffusion
        diff_u = diffusion(last(u_fields), u_diff_ops)
        @test length(diff_u) == 2
        @test diff_u[1] isa MPS
        @test diff_u[2] isa MPS
    end

    @testset "symmetrized_laplacian 2D" begin
        # 
        sites = siteinds(TTArchitecture(arch_2d))
        n_sites = length(sites)
        ux = random_mps(sites)
        uy = random_mps(sites)

        grid = Grid(TTArchitecture(arch_2d), (1.0, 1.0))
        Re = 100.0

        u_fields = VectorField((ux, uy), TTArchitecture(arch_2d), (bc2d, bc2d))
        p_field = ScalarField(sites, TTArchitecture(arch_2d), bc2d)

        experiment = INSExperiment(grid, u_fields, p_field, Re, nothing; stencil_order=2)

        Δ = symmetrized_laplacian(experiment.p_field, experiment.grid; stencil_order=2)

        @test Δ isa MPO
        @test length(Δ) == n_sites
        @test first.(siteinds(Δ; plev=0)) == siteinds(ux)
    end

    @testset "symmetrized_laplacian 3D" begin
        sites = siteinds(TTArchitecture(arch_3d))
        n_sites = length(sites)
        ux = random_mps(sites)
        uy = random_mps(sites)
        uz = random_mps(sites)

        grid = Grid(TTArchitecture(arch_3d), (1.0, 1.0, 1.0))
        Re = 100.0

        u_fields = VectorField((ux, uy, uz), TTArchitecture(arch_3d), (bc3d, bc3d, bc3d))
        p_field = ScalarField(sites, TTArchitecture(arch_3d), bc3d)

        experiment = INSExperiment(grid, u_fields, p_field, Re, nothing; stencil_order=2)

        Δ = symmetrized_laplacian(experiment.p_field, experiment.grid; stencil_order=2)

        @test Δ isa MPO
        @test length(Δ) == n_sites
        @test first.(siteinds(Δ; plev=0)) == siteinds(ux)
    end

    @testset "mask_correction with nothing mask 2D" begin
        # 
        sites = siteinds(TTArchitecture(arch_2d))
        n_sites = length(sites)
        n_sites = length(sites)
        ux = random_mps(sites)
        uy = random_mps(sites)

        out = DifferenTT.mask_correction((ux, uy), nothing)

        @test length(out) == 2
        @test out[1] == ux
        @test out[2] == uy
        @test length(out[1]) == n_sites
        @test length(out[2]) == n_sites
    end

    @testset "regularization_check" begin
        # 
        sites = siteinds(TTArchitecture(arch_2d))
        ux = random_mps(sites)
        uy = random_mps(sites)

        grid = Grid(TTArchitecture(arch_2d), (1.0, 1.0))
        Re = 100.0

        u_fields = VectorField((ux, uy), TTArchitecture(arch_2d), (bc2d, bc2d))
        p_field_periodic = ScalarField(TTArchitecture(arch_2d), bc2d)

        exp_periodic = INSExperiment(grid, u_fields, p_field_periodic, Re, nothing; stencil_order=2)
        @test DifferenTT.regularization_check(exp_periodic) == true

        dirichlet_bc1d = BoundaryCond1d(quietly(() -> DirichletBC(0.0)), quietly(() -> DirichletBC(0.0)))
        dirichlet_bc2d = BoundaryConditions(dirichlet_bc1d, dirichlet_bc1d)
        p_field_dirichlet = ScalarField(TTArchitecture(arch_2d), dirichlet_bc2d)

        exp_dirichlet = INSExperiment(grid, u_fields, p_field_dirichlet, Re, nothing; stencil_order=2)
        @test DifferenTT.regularization_check(exp_dirichlet) == false
    end

    @testset "check_cfl 2D" begin
        sites = siteinds(TTArchitecture(arch_2d))
        grid = Grid(TTArchitecture(arch_2d), (1.0, 1.0))
        Re = 100.0
        u_fields = VectorField((random_mps(sites), random_mps(sites)), TTArchitecture(arch_2d), (bc2d, bc2d))
        p_field = ScalarField(TTArchitecture(arch_2d), bc2d)
        experiment = INSExperiment(grid, u_fields, p_field, Re, nothing; stencil_order=2)

        te_ok = TimeEvolArgs(0.0, 0.01, 0.001)
        te_bad = TimeEvolArgs(0.0, 100.0, 1)
        # Stable for RK4, but beyond the forward Euler advection-diffusion limit 2 / (diff_eig + vmax² Re) ≈ 0.02
        te_rk4_only = TimeEvolArgs(0.0, 0.05, 1)

        @test_logs min_level=Base.CoreLogging.Warn check_cfl(experiment, RK4(time_evol_kwargs=te_ok))
        @test_logs min_level=Base.CoreLogging.Warn check_cfl(experiment, ExplicitEuler(time_evol_kwargs=te_ok))

        @test_logs (:warn,) match_mode=:any check_cfl(experiment, RK4(time_evol_kwargs=te_bad))
        @test_logs (:warn,) match_mode=:any check_cfl(experiment, ExplicitEuler(time_evol_kwargs=te_bad))

        @test_logs min_level=Base.CoreLogging.Warn check_cfl(experiment, RK4(time_evol_kwargs=te_rk4_only))
        @test_logs (:warn,) match_mode=:any check_cfl(experiment, ExplicitEuler(time_evol_kwargs=te_rk4_only))
    end

    @testset "INSExperiment construction 2D" begin
        # 
        sites = siteinds(TTArchitecture(arch_2d))
        ux = random_mps(sites)
        uy = random_mps(sites)

        grid = Grid(TTArchitecture(arch_2d), (1.0, 1.0))
        Re = 100.0

        u_fields = VectorField((ux, uy), TTArchitecture(arch_2d), (bc2d, bc2d))
        p_field = ScalarField(TTArchitecture(arch_2d), bc2d)

        experiment = INSExperiment(grid, u_fields, p_field, Re, nothing; stencil_order=2)
        @test experiment isa INSExperiment{2}
        @test experiment.Re == Re
        @test experiment.stencil_order == 2
        @test isnothing(experiment.object_mask)
    end

    @testset "NavierStokesPredictorRHS" begin
        # 
        sites = siteinds(TTArchitecture(arch_2d))
        ux = random_mps(sites)
        uy = random_mps(sites)

        grid = Grid(TTArchitecture(arch_2d), (1.0, 1.0))
        Re = 100.0

        u_fields = VectorField((ux, uy), TTArchitecture(arch_2d), (bc2d, bc2d))
        p_field = ScalarField(TTArchitecture(arch_2d), bc2d)

        experiment = INSExperiment(grid, u_fields, p_field, Re, nothing; stencil_order=2)

        u_diff_ops = DifferenTT.DifferentialOps(
            experiment.grid, experiment.u_fields; stencil_order=experiment.stencil_order
        )
        u_boundary_nullifier = DifferenTT.BoundaryNull(experiment.u_fields; stencil_order=experiment.stencil_order)

        rhs_func = NavierStokesPredictorRHS(experiment, u_diff_ops, u_boundary_nullifier)

        rhs_out = rhs_func((ux, uy), 0.0, 1)
        @test length(rhs_out) == 2
        @test rhs_out[1] isa MPS
        @test rhs_out[2] isa MPS
    end
    @testset "Chorin building blocks" begin
        qarch = QTTArchitecture((4, 4))
        qsites = siteinds(qarch)
        qgrid = Grid(qarch, (1.0, 1.0))
        ux = random_mps(qsites)
        uy = random_mps(qsites)

        u_fields = VectorField((ux, uy), qarch, (bc2d, bc2d))
        p_field = ScalarField(qsites, qarch, bc2d)
        experiment = INSExperiment(qgrid, u_fields, p_field, 100.0, nothing; stencil_order=2, maxdim=20, cutoff=1e-12)
        stepper = ExplicitEuler(time_evol_kwargs=TimeEvolArgs(0.0, 0.001, 1), cutoff=1e-12, maxdim=20)
        dt = stepper.time_evol_kwargs.dt

        cache = ChorinCache(experiment)
        @test cache.regularize == DifferenTT.regularization_check(experiment) == true
        @test cache.poisson_op isa MPO
        @test first.(siteinds(cache.poisson_op; plev=0)) == qsites

        rhs = DifferenTT.poisson_rhs(experiment, cache, (ux, uy), dt)
        @test rhs isa MPS
        @test siteinds(rhs) == qsites
        # Periodic pressure: the mean divergence is removed from the RHS
        @test abs(integrate(rhs)) < 1e-8 * norm(rhs) * sqrt(2^length(qsites))

        p = random_mps(qsites)
        u_proj = DifferenTT.∇p_correction((ux, uy), experiment, cache, p, dt)
        @test length(u_proj) == 2
        @test all(u -> siteinds(u) == qsites, u_proj)
        # With dt = 0 the projection leaves the velocity unchanged
        u_same = DifferenTT.∇p_correction((ux, uy), experiment, cache, p, 0.0)
        @test norm(u_same[1] - ux) < 1e-8 * norm(ux)
        @test norm(u_same[2] - uy) < 1e-8 * norm(uy)

        solver = ALS(;
            sweep_kwargs=SweepingArgs(; nsweeps=2, maxdim=20),
            updater_kwargs=LocalUpdaterArgs(; ishermitian=true, isposdef=true),
        )
        n_u, n_p = length(experiment.u_fields), length(experiment.p_field)
        u_next, p_next, linsolve_error = chorin_step(experiment, cache, last(experiment.u_fields), solver, stepper, 0.0)
        @test length(u_next) == 2
        @test all(u -> siteinds(u) == qsites, u_next)
        @test p_next isa MPS
        @test linsolve_error >= 0.0
        # chorin_step leaves the experiment untouched
        @test length(experiment.u_fields) == n_u
        @test length(experiment.p_field) == n_p
    end

    @testset "symmetrize_rhs" begin
        arch = TTArchitecture((4,))
        sites = siteinds(arch)

        # Neumann boundaries on all sides
        neumann_bc = NeumannBC(0.0)
        bc1d = BoundaryCond1d(neumann_bc, neumann_bc)
        bc = BoundaryConditions(bc1d)

        p_field = ScalarField(sites, arch, bc)

        # Create an RHS with all ones
        rhs = ones_mps(sites, 1)

        # Symmetrize the RHS
        sym_rhs = symmetrize_rhs(p_field, rhs)

        # Contract to dense vector
        dense_rhs = contract_mps(rhs)
        dense_sym_rhs = contract_mps(sym_rhs)

        @test isapprox(dense_rhs[1], 1.0; atol=1e-14)
        @test isapprox(dense_sym_rhs[1], 0.5; atol=1e-14)
        @test isapprox(dense_sym_rhs[end], 0.5; atol=1e-14)
        @test isapprox(dense_sym_rhs[2], 1.0; atol=1e-14)
        @test isapprox(dense_sym_rhs[3], 1.0; atol=1e-14)

        # 2D test
        arch_2d = (4, 4)
        sites_2d = siteinds(TTArchitecture(arch_2d))
        bc2d = BoundaryConditions(bc1d, bc1d)
        p_field_2d = ScalarField(sites_2d, TTArchitecture(arch_2d), bc2d)
        rhs_2d = ones_mps(sites_2d, 1, Vector{Float64})
        sym_rhs_2d = symmetrize_rhs(p_field_2d, rhs_2d)
        dense_sym_rhs_2d = reshape(contract_mps(sym_rhs_2d), 4, 4)

        # Corners have 2 boundaries, so the weights multiply: 0.5 * 0.5 = 0.25
        @test isapprox(dense_sym_rhs_2d[1, 1], 0.25; atol=1e-14)
        @test isapprox(dense_sym_rhs_2d[1, end], 0.25; atol=1e-14)
        @test isapprox(dense_sym_rhs_2d[end, 1], 0.25; atol=1e-14)
        @test isapprox(dense_sym_rhs_2d[end, end], 0.25; atol=1e-14)

        # Edges have 1 boundary, so 1.0 - 0.5 = 0.5
        @test isapprox(dense_sym_rhs_2d[2, 1], 0.5; atol=1e-14)
        @test isapprox(dense_sym_rhs_2d[1, 2], 0.5; atol=1e-14)

        # Interior has 0 boundaries, so 1.0
        @test isapprox(dense_sym_rhs_2d[2, 2], 1.0; atol=1e-14)
    end
end

# Known limitations of the Chorin projection, kept as documented in v0.1 (decision D8 of the release plan): both are
# recorded as @test_broken until they are fixed in v0.2.
@testset "Chorin known limitations (D8)" begin
    arch = TTArchitecture((6, 6))
    sites = siteinds(arch)
    grid = Grid(arch, (1.0, 1.0))
    n1, n2 = variable_dims(arch)
    dirichlet = BoundaryCond1d(quietly(() -> DirichletBC(0.0)), quietly(() -> DirichletBC(0.0)))
    neumann = BoundaryCond1d(NeumannBC(0.0), NeumannBC(0.0))
    u_bcs = BoundaryConditions(dirichlet, dirichlet)
    p_bcs = BoundaryConditions(neumann, neumann)
    u = (random_mps(rng(1), sites; linkdims=3), random_mps(rng(2), sites; linkdims=3))
    u_fields = VectorField(u, arch, (u_bcs, u_bcs))
    p_field = ScalarField(sites, arch, p_bcs)
    experiment = quietly(() -> INSExperiment(grid, u_fields, p_field, 100.0, nothing; stencil_order=2, cutoff=1e-14))
    cache = quietly(() -> ChorinCache(experiment))
    @test cache.regularize
    dt = 1e-2

    # The mean removal of poisson_rhs subtracts Σrhs / Π(nᵢ - 1) instead of Σrhs / Π nᵢ, because it divides
    # Σrhs · Π hᵢ by the box volume Π (nᵢ - 1) hᵢ. The regularized right-hand side therefore keeps a
    # fraction 1 - Π nᵢ / (nᵢ - 1) of its mean.
    rhs = DifferenTT.poisson_rhs(experiment, cache, u, dt)
    @test_broken abs(DifferenTT.integrate(rhs)) < 1e-10 * sqrt(n1 * n2) * norm(rhs)

    # The projection u = u* - dt ∇p also changes the Dirichlet wall rows, so the tangential pressure gradient drags
    # the wall velocity a little at every step. The wall rows of the correction should be zero.
    p = random_mps(rng(3), sites; linkdims=3)
    u_corr = DifferenTT.∇p_correction(u, experiment, cache, p, dt)
    wall = [i in (1, n1) || j in (1, n2) for j in 1:n2, i in 1:n1] # [y, x], site 1 (x) is the most significant
    for c in 1:2
        δu = reshape(todense(u_corr[c]) - todense(u[c]), n2, n1)
        @test_broken maximum(abs, δu[wall]) < 1e-12
    end
end

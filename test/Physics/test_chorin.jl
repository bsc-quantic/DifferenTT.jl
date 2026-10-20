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

    @testset "mask_correction with a mask against the dense Hadamard product ($label)" for (label, arch) in (
        ("TT", TTArchitecture((6, 5))), ("QTT", QTTArchitecture((3, 3)))
    )
        sites = siteinds(arch)
        dims = variable_dims(arch)
        grid_m = Grid(arch, (1.0, 1.0))
        values_of(mps) = [
            DifferenTT.sample(mps, ((i - 1) * grid_m.dx[1], (j - 1) * grid_m.dx[2]), grid_m, arch) for
            i in 1:dims[1], j in 1:dims[2]
        ]
        # 0 inside a disc, 1 outside, like the cylinder of the paper's channel, stored as an MPS of the sites
        disc = [hypot(x - 0.5, y - 0.5) < 0.3 ? 0.0 : 1.0 for x in grid_m.points[1], y in grid_m.points[2]]
        mask = if arch isa TTArchitecture
            MPS(ITensor(disc, sites[1], sites[2]), sites)
        else
            vec2mps(vec(disc'), sites)  # site 1 (axis 1) is the most significant index
        end
        @test values_of(mask) ≈ disc

        u = (random_mps(rng(5), sites; linkdims=3), random_mps(rng(6), sites; linkdims=3))
        U = values_of.(u)
        masked = DifferenTT.mask_correction(u, mask)
        @test length(masked) == 2
        for c in 1:2
            @test values_of(masked[c]) ≈ U[c] .* disc atol = 1e-10
            # nothing survives inside the obstacle
            @test all(iszero, round.(values_of(masked[c])[disc .== 0]; digits=10))
        end
        # The truncation keywords are honoured, and the input is untouched
        capped = DifferenTT.mask_correction(u, mask; maxdim=2)
        @test all(maxlinkdim(c) <= 2 for c in capped)
        @test values_of(u[1]) ≈ U[1]
        # Applying the mask twice is the same as once, since the mask is 0 or 1
        twice = DifferenTT.mask_correction(masked, mask)
        @test values_of(twice[1]) ≈ values_of(masked[1]) atol = 1e-10
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

        dirichlet_bc1d = BoundaryCond1d(DirichletBC(0.0), DirichletBC(0.0))
        dirichlet_bc2d = BoundaryConditions(dirichlet_bc1d, dirichlet_bc1d)
        p_field_dirichlet = ScalarField(TTArchitecture(arch_2d), dirichlet_bc2d)

        exp_dirichlet = INSExperiment(grid, u_fields, p_field_dirichlet, Re, nothing; stencil_order=2)
        @test DifferenTT.regularization_check(exp_dirichlet) == false

        # One Dirichlet side anywhere is enough to remove the null space; Neumann or open sides alone are not
        neumann_bc1d = BoundaryCond1d(NeumannBC(0.0), NeumannBC(0.0))
        open_bc1d = BoundaryCond1d(OpenBC(), OpenBC())
        outlet_bc1d = BoundaryCond1d(NeumannBC(0.0), DirichletBC(0.0))
        for (bcs_p, singular) in (
            (BoundaryConditions(neumann_bc1d, neumann_bc1d), true),
            (BoundaryConditions(open_bc1d, neumann_bc1d), true),
            (BoundaryConditions(neumann_bc1d, outlet_bc1d), false),
            (BoundaryConditions(outlet_bc1d, bc1d), false),
        )
            p_field_mixed = ScalarField(TTArchitecture(arch_2d), bcs_p)
            exp_mixed = quietly(() -> INSExperiment(grid, u_fields, p_field_mixed, Re, nothing; stencil_order=2))
            @test DifferenTT.regularization_check(exp_mixed) == singular
        end
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
        u_before = (copy(ux), copy(uy))
        u_next, p_next, linsolve_error = chorin_step(experiment, cache, last(experiment.u_fields), solver, stepper, 0.0)
        @test length(u_next) == 2
        @test all(u -> siteinds(u) == qsites, u_next)
        @test p_next isa MPS
        @test linsolve_error >= 0.0
        # chorin_step leaves the experiment untouched
        @test length(experiment.u_fields) == n_u
        @test length(experiment.p_field) == n_p
        @test mps_equal(last(experiment.u_fields), (ux, uy))
        @test mps_equal((ux, uy), u_before)
        @test u_next[1] !== ux && u_next[2] !== uy
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

# The Poisson system of the projection against the full stencil on the ghost-extended array. For a pressure with a
# Dirichlet outlet of value g, a Neumann inlet and Neumann walls, the solution p dt of A (p dt) = b satisfies the full
# second-derivative stencil with ghost values g dt, and equals the divergence of the velocity on every row that is not
# next to a Dirichlet face of its component; the projection subtracts dt times the full-stencil gradient of p.
@testset "Chorin projection with a Dirichlet pressure against the ghost-extended stencil (q=$q)" for q in (2, 8)
    arch = TTArchitecture((10, 9))
    n1, n2 = variable_dims(arch)
    grid = Grid(arch, (1.0, 1.5))
    h = grid.dx
    sites = siteinds(arch)
    walls = BoundaryCond1d(DirichletBC(0.0), DirichletBC(0.0))
    mirror = BoundaryCond1d(NeumannBC(0.0), NeumannBC(0.0))
    wrap = BoundaryCond1d(PeriodicBC(), PeriodicBC())
    outlet_value = 0.8
    outlet = BoundaryCond1d(NeumannBC(0.0), DirichletBC(outlet_value))
    # u₁ has walls along x (its own axis, so the divergence rows next to them are removed) and wraps along y; u₂ is
    # Neumann along y (its own axis): nothing is removed there
    u_bcs = (BoundaryConditions(walls, wrap), BoundaryConditions(walls, mirror))
    p_bcs = BoundaryConditions(outlet, mirror)
    u = (random_mps(rng(31), sites; linkdims=9), random_mps(rng(32), sites; linkdims=9))
    u_fields = VectorField(u, arch, u_bcs)
    experiment = quietly(
        () -> INSExperiment(
            grid,
            u_fields,
            ScalarField(sites, arch, p_bcs),
            100.0,
            nothing;
            stencil_order=q,
            cutoff=1e-14,
            maxdim=256,
        ),
    )
    cache = quietly(() -> ChorinCache(experiment))
    @test !cache.regularize
    dt = 0.05

    ghost_stencil(F, order, axis, g_left, g_right) = begin
        coeffs = DifferenTT.derivative_coefficients(order; stencil_order=q)
        n = size(F, axis)
        out = zeros(size(F))
        for idx in CartesianIndices(F), (k, c) in coeffs
            j = idx[axis] + k
            ghost = j < 1 ? g_left : (j > n ? g_right : nothing)
            value = if isnothing(ghost)
                F[CartesianIndex(Base.setindex(Tuple(idx), j, axis))]
            elseif ghost === :mirror
                F[CartesianIndex(Base.setindex(Tuple(idx), j < 1 ? 2 - j : 2n - j, axis))]
            elseif ghost === :wrap
                F[CartesianIndex(Base.setindex(Tuple(idx), mod1(j, n), axis))]
            else
                ghost
            end
            out[idx] += c * value
        end
        return out ./ h[axis]^order
    end
    as_array(mps) = reshape(DifferenTT.contract_mps(mps), n2, n1)' # [i₁, i₂]: site 1 is the most significant index
    U = as_array.(u)

    # Divergence of u* with the closures of each component along its own axis (walls: truncated; Neumann: mirrored)
    divergence_dense = ghost_stencil(U[1], 1, 1, 0.0, 0.0) + ghost_stencil(U[2], 1, 2, :mirror, :mirror)
    keep = ones(n1, n2) # the projector Z: zero on the q/2 rows next to the x walls
    keep[1:(q÷2), :] .= 0.0
    keep[(n1-q÷2+1):n1, :] .= 0.0

    b = DifferenTT.poisson_rhs(experiment, cache, u, dt)
    A = todense(cache.poisson_op)
    pdt = A \ todense(b)
    PDT = reshape(pdt, n2, n1)'
    @test norm(A * pdt - todense(b)) < 1e-9 * norm(todense(b))

    # Full stencil of p dt, with the Dirichlet ghost value g dt beyond the outlet and the mirror elsewhere
    laplacian_full = ghost_stencil(PDT, 2, 1, :mirror, outlet_value * dt) + ghost_stencil(PDT, 2, 2, :mirror, :mirror)
    @test laplacian_full ≈ keep .* divergence_dense rtol = 1e-8 atol = 1e-8 * norm(divergence_dense)

    # The projected velocity: u - dt (grad p + lift), with the full-stencil gradient of p = (p dt) / dt
    p = MPS(ITensor(PDT ./ dt, sites[1], sites[2]), sites)
    P = PDT ./ dt
    projected = DifferenTT.∇p_correction(u, experiment, cache, p, dt)
    for (axis, expected_gradient) in
        ((1, ghost_stencil(P, 1, 1, :mirror, outlet_value)), (2, ghost_stencil(P, 1, 2, :mirror, :mirror)))
        @test as_array(projected[axis]) ≈ U[axis] .- dt .* expected_gradient rtol = 1e-8 atol = 1e-8 * norm(U[axis])
    end
end

# Known limitations of the Chorin projection, kept as documented in v0.1 (decision D8 of the release plan): both are
# recorded as @test_broken until they are fixed in v0.2.
@testset "Chorin known limitations (D8)" begin
    arch = TTArchitecture((6, 6))
    sites = siteinds(arch)
    grid = Grid(arch, (1.0, 1.0))
    n1, n2 = variable_dims(arch)
    dirichlet = BoundaryCond1d(DirichletBC(0.0), DirichletBC(0.0))
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

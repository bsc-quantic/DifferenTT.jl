using DifferenTT
using Test
using ITensors
using ITensorMPS
using Logging

include(joinpath(@__DIR__, "..", "testutils.jl"))
using .TestUtils

@testset "insexperiment.jl" begin
    n = 3
    tt_arch = TTArchitecture((2^n, 2^n))
    grid = Grid(tt_arch, (1.0, 1.0))
    time_evol_kwargs = TimeEvolArgs(0.1, 1.0, 10)
    time_stepper = RK4(time_evol_kwargs=time_evol_kwargs, cutoff=1e-5, maxdim=10)

    sites = [Index(2, "Site,n=$i") for i in 1:2]
    u_mps = random_mps(sites)

    bc_periodic = BoundaryCond1d(PeriodicBC(), PeriodicBC())
    bcs = BoundaryConditions(bc_periodic, bc_periodic)

    sfield = ScalarField(u_mps, tt_arch, bcs)
    vfield = VectorField((sfield, sfield))

    @testset "INSExperiment instantiation" begin
        exp = DifferenTT.INSExperiment(grid, vfield, sfield, 100.0, nothing; stencil_order=2, maxdim=10, cutoff=1e-5)
        @test exp.Re == 100.0
        @test !hasfield(DifferenTT.INSExperiment, :time_stepper)
        @test exp.stencil_order == 2
        @test exp.maxdim == 10
        @test exp.cutoff == 1e-5
        @test isnothing(exp.object_mask)

        # Test checks
        @test (DifferenTT.check_cfl(exp, time_stepper); true)
        @test (
            DifferenTT.check_cfl(exp, ExplicitEuler(time_evol_kwargs=time_evol_kwargs, cutoff=1e-5, maxdim=10)); true
        )
        DifferenTT.check_kolmogorov_scale(exp)
    end

    @testset "Grid periodicity must match the boundary conditions" begin
        periodic_grid = Grid(tt_arch, (1.0, 1.0); periodic=(true, true))
        exp = DifferenTT.INSExperiment(periodic_grid, vfield, sfield, 1.0, nothing; stencil_order=2)
        @test exp.grid.periodic == (true, true)
        walls = BoundaryConditions(BoundaryCond1d(DirichletBC(0.0), DirichletBC(0.0)), bc_periodic)
        wall_field = ScalarField(u_mps, tt_arch, walls)
        @test_throws ArgumentError DifferenTT.INSExperiment(periodic_grid, vfield, wall_field, 1.0, nothing)
        @test_throws ArgumentError DifferenTT.INSExperiment(
            periodic_grid, VectorField((wall_field, wall_field)), sfield, 1.0, nothing
        )
        # A PeriodicBC on a grid without the flag is still accepted (period L + h)
        @test DifferenTT.INSExperiment(grid, vfield, sfield, 1.0, nothing) isa DifferenTT.INSExperiment
    end

    @testset "INSExperiment validation" begin
        # Invalid object mask
        mask = random_mps(siteinds("S=1/2", 3)) # wrong sites
        @test_throws ArgumentError DifferenTT.INSExperiment(grid, vfield, sfield, 100.0, mask)
    end

    # Log records of a call: the warnings and the info records
    function logged(f)
        logger = Test.TestLogger(; min_level=Logging.Info)
        with_logger(f, logger)
        return logger.logs
    end
    warnings(logs) = filter(r -> r.level == Logging.Warn, logs)
    infos(logs) = filter(r -> r.level == Logging.Info, logs)

    # Hand values of the stencil sums Σ_k |c_k|: ±1/2 for the first derivative and 1, -2, 1 for the second with q = 2;
    # the eighth-order coefficients are the exact fractions below
    s1 = Dict(2 => 1.0, 8 => 2 * (4 / 5 + 1 / 5 + 4 / 105 + 1 / 280))
    s2 = Dict(2 => 4.0, 8 => 205 / 72 + 2 * (8 / 5 + 1 / 5 + 8 / 315 + 1 / 560))

    function cfl_experiment(; q=2, Re=100.0, box=(1.0, 2.0), n=16)
        arch = TTArchitecture((n, n))
        g = Grid(arch, box)
        s = siteinds(arch)
        u = VectorField((random_mps(rng(1), s), random_mps(rng(2), s)), arch, (bcs, bcs))
        return quietly(() -> DifferenTT.INSExperiment(g, u, ScalarField(s, arch, bcs), Re, nothing; stencil_order=q))
    end

    @testset "cfl_spectral_radii (q=$q)" for q in (2, 8)
        # λ_adv = Σ_i vmax s1 / h_i and λ_diff = Σ_i s2 / (Re h_i²), with h = (1/15, 2/15)
        experiment = cfl_experiment(; q=q, Re=50.0)
        h = (1 / 15, 2 / 15)
        adv, diff = DifferenTT.cfl_spectral_radii(experiment)
        @test adv ≈ s1[q] * sum(1 ./ h)
        @test diff ≈ s2[q] / 50.0 * sum(1 ./ h .^ 2)
        # vmax scales the advection only
        adv3, diff3 = DifferenTT.cfl_spectral_radii(experiment; vmax=3)
        @test adv3 ≈ 3 * adv
        @test diff3 == diff
    end

    @testset "check_cfl, RK4: limits 2√2/λ_adv and 2.785/λ_diff" begin
        rk4(dt) = RK4(time_evol_kwargs=TimeEvolArgs(0.0, dt, 1), cutoff=1e-8, maxdim=10)

        # Re = 100: the advection limit 2√2/30 binds (the diffusion limit 2.785/18 is larger)
        experiment = cfl_experiment(; Re=100.0, box=(1.0, 1.0))
        adv, diff = DifferenTT.cfl_spectral_radii(experiment)
        dt_adv, dt_diff = 2 * sqrt(2) / adv, 2.785 / diff
        @test adv ≈ 30.0 # 2 axes × 15
        @test diff ≈ 18.0 # 2 axes × 4 × 15² / 100
        @test dt_adv < dt_diff

        logs = logged(() -> check_cfl(experiment, rk4(0.9 * dt_adv)))
        @test isempty(warnings(logs))
        @test only(infos(logs)).message == "CFL condition satisfied"
        @test only(infos(logs)).kwargs[:dt_max] ≈ dt_adv

        logs = logged(() -> check_cfl(experiment, rk4(1.1 * dt_adv)))
        @test only(warnings(logs)).message == "CFL condition violated for advection term"
        @test only(warnings(logs)).kwargs[:dt_adv_max] ≈ dt_adv
        @test occursin("must set dt<=$dt_adv", only(infos(logs)).message)

        # Re = 1: the diffusion limit binds
        experiment = cfl_experiment(; Re=1.0, box=(1.0, 1.0))
        adv, diff = DifferenTT.cfl_spectral_radii(experiment)
        dt_adv, dt_diff = 2 * sqrt(2) / adv, 2.785 / diff
        @test dt_diff < dt_adv
        logs = logged(() -> check_cfl(experiment, rk4(1.1 * dt_diff)))
        @test only(warnings(logs)).message == "CFL condition violated for diffusion term"
        @test only(warnings(logs)).kwargs[:dt_diff_max] ≈ dt_diff
        logs = logged(() -> check_cfl(experiment, rk4(0.9 * dt_diff)))
        @test isempty(warnings(logs))

        # Beyond both limits: both warnings; vmax enters the advection limit only
        logs = logged(() -> check_cfl(experiment, rk4(1.1 * dt_adv)))
        @test length(warnings(logs)) == 2
        experiment100 = cfl_experiment(; Re=100.0, box=(1.0, 1.0))
        dt = 0.9 * 2 * sqrt(2) / DifferenTT.cfl_spectral_radii(experiment100)[1]
        @test isempty(warnings(logged(() -> check_cfl(experiment100, rk4(dt)))))
        @test length(warnings(logged(() -> check_cfl(experiment100, rk4(dt); vmax=2.0)))) == 1
    end

    @testset "check_cfl, forward Euler: limits 2/λ_diff and 2/(λ_diff + vmax² Re)" begin
        euler(dt) = ExplicitEuler(time_evol_kwargs=TimeEvolArgs(0.0, dt, 1), cutoff=1e-8, maxdim=10)
        experiment = cfl_experiment(; Re=100.0, box=(1.0, 1.0))
        _, diff = DifferenTT.cfl_spectral_radii(experiment)
        dt_diff = 2 / diff
        dt_comb = 2 / (diff + 1.0^2 * 100.0)
        @test dt_comb < dt_diff

        # Below the combined limit: nothing to warn about
        logs = logged(() -> check_cfl(experiment, euler(0.9 * dt_comb)))
        @test isempty(warnings(logs))
        @test only(infos(logs)).kwargs[:dt_max] ≈ dt_comb

        # Between the two limits: only the combined advection-diffusion warning (RK4 would be stable here)
        logs = logged(() -> check_cfl(experiment, euler(1.1 * dt_comb)))
        @test only(warnings(logs)).message == "CFL condition violated for the combined advection-diffusion term"
        @test only(warnings(logs)).kwargs[:dt_max] ≈ dt_comb

        # Beyond the diffusion limit: both warnings
        logs = logged(() -> check_cfl(experiment, euler(1.1 * dt_diff)))
        @test [w.message for w in warnings(logs)] == [
            "CFL condition violated for diffusion term",
            "CFL condition violated for the combined advection-diffusion term",
        ]
        @test warnings(logs)[1].kwargs[:dt_diff_max] ≈ dt_diff

        # vmax = 0 removes the advection: the limit is the diffusion limit
        logs = logged(() -> check_cfl(experiment, euler(1.1 * dt_comb); vmax=0.0))
        @test isempty(warnings(logs))
    end

    @testset "check_kolmogorov_scale" begin
        # The Kolmogorov scale of Re = 100 is 100^(-3/4) = 0.0316; a 16-node unit axis has h = 1/15 > η, a 64-node one
        # has 1/63 < η
        η = 100.0^(-3 / 4)
        coarse = Grid(TTArchitecture((16, 64)), (1.0, 1.0))
        fine = Grid(TTArchitecture((64, 64)), (1.0, 1.0))
        logs = logged(() -> DifferenTT.check_kolmogorov_scale(coarse, 100.0))
        @test [w.message for w in warnings(logs)] == [
            "Spacing in the 1-th dimension does not resolve to Kolmogorov scale dx = $(coarse.dx[1]) > Re^-(3/4) = $η"
        ]
        @test isempty(infos(logs))

        logs = logged(() -> DifferenTT.check_kolmogorov_scale(fine, 100.0))
        @test isempty(warnings(logs))
        @test only(infos(logs)).message == "Kolmogorov scale is resolved"

        # Every axis is checked on its own: an anisotropic box can be resolved along one axis only
        long = Grid(TTArchitecture((64, 64)), (1.0, 4.0))
        logs = logged(() -> DifferenTT.check_kolmogorov_scale(long, 100.0))
        @test length(warnings(logs)) == 1
        @test occursin("2-th dimension", only(warnings(logs)).message)

        # The experiment method uses its grid and Reynolds number; a lower Re has a larger scale
        arch = TTArchitecture((64, 64))
        s = siteinds(arch)
        u = VectorField((random_mps(rng(1), s), random_mps(rng(2), s)), arch, (bcs, bcs))
        for (Re, resolved) in ((100.0, true), (1e4, false))
            experiment = quietly(
                () -> DifferenTT.INSExperiment(fine, u, ScalarField(s, arch, bcs), Re, nothing; stencil_order=2)
            )
            logs = logged(() -> DifferenTT.check_kolmogorov_scale(experiment))
            @test isempty(warnings(logs)) == resolved
        end
        # The constructor performs the same check
        logs = logged(() -> DifferenTT.INSExperiment(fine, u, ScalarField(s, arch, bcs), 1e4, nothing))
        @test length(warnings(logs)) == 2
    end
end

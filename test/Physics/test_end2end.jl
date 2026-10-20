using ITensors
using ITensorMPS
using DifferenTT
using Test
using LinearAlgebra
using JLD2

include(joinpath(@__DIR__, "..", "testutils.jl"))
using .TestUtils

@testset "End-to-End Navier-Stokes Simulation" begin
    # Setup small problem for fast testing
    nodes = (4, 4) # very small grid for speed (16, 16)
    box_size = (1.0, 1.0)
    arch = QTTArchitecture(nodes)
    grid = Grid(arch, box_size)

    sites = siteinds(2, sum(abs, nodes))

    # 1 step RK4 to check pipeline works
    time_stepper = RK4(time_evol_kwargs=TimeEvolArgs(0.0, 0.001, 1), cutoff=1e-8, maxdim=10)
    Re = 100

    # Initialization
    v0 = 1.0
    mps_vx = v0 * ones_mps(sites, 1)
    mps_vy = 0.0 * ones_mps(sites, 1)

    # Periodic boundaries in both directions
    bcsvxx = BoundaryCond1d(PeriodicBC(), PeriodicBC())
    bcsvxy = BoundaryCond1d(PeriodicBC(), PeriodicBC())

    bcsvyx = BoundaryCond1d(PeriodicBC(), PeriodicBC())
    bcsvyy = BoundaryCond1d(PeriodicBC(), PeriodicBC())

    bcspx = BoundaryCond1d(PeriodicBC(), PeriodicBC())
    bcspy = BoundaryCond1d(PeriodicBC(), PeriodicBC())

    bcsvx = BoundaryConditions(bcsvxx, bcsvxy)
    bcsvy = BoundaryConditions(bcsvyx, bcsvyy)
    bcsp = BoundaryConditions(bcspx, bcspy)

    v_field = VectorField((mps_vx, mps_vy), arch, (bcsvx, bcsvy))
    p_field = ScalarField(sites, arch, bcsp)

    experiment = DifferenTT.INSExperiment(grid, v_field, p_field, Re, nothing; stencil_order=2, maxdim=10, cutoff=1e-8)

    updater_kwargs = DifferenTT.LocalUpdaterArgs(; ishermitian=true, isposdef=true)
    sweep_kwargs = DifferenTT.SweepingArgs(; nsweeps=1, maxdim=10)
    solver = DifferenTT.ALS(; sweep_kwargs=sweep_kwargs, updater_kwargs=updater_kwargs)

    @testset "Chorin Step Execution" begin
        # Do not save or log to file
        DifferenTT.chorin_evolve!(experiment, solver, time_stepper, 1, nothing, "")

        # Verify that fields have been updated
        @test length(experiment.u_fields) == 2
        @test length(experiment.p_field) == 1

        v_final_x, v_final_y = last(experiment.u_fields)
        p_final = last(experiment.p_field)

        @test v_final_x isa MPS
        @test v_final_y isa MPS
        @test p_final isa MPS

        # Test divergence free property roughly
        cutoff = experiment.cutoff
        maxdim = experiment.maxdim
        masked_u_mpss = (v_final_x, v_final_y)
        u_diff_ops = DifferenTT.DifferentialOps(
            experiment.grid, experiment.u_fields; stencil_order=experiment.stencil_order, cutoff=cutoff, maxdim=maxdim
        )

        ∇u = DifferenTT.divergence(masked_u_mpss, u_diff_ops; cutoff=cutoff, maxdim=maxdim)

        # The uniform flow (v0, 0) is a fixed point of the periodic Navier-Stokes equations: the pressure is
        # constant (zero), the velocity does not change and its divergence vanishes
        @test norm(v_final_x - mps_vx) < 1e-10 * norm(mps_vx)
        @test norm(v_final_y) < 1e-10
        @test norm(p_final) < 1e-10
        @test norm(∇u) < 1e-10 * norm(v_final_x)
    end

    @testset "Checkpoints and residual observer" begin
        # 3 steps saved every 2 -> checkpoints at steps 1 and 3
        stepper3 = RK4(time_evol_kwargs=TimeEvolArgs(0.0, 0.003, 3), cutoff=1e-8, maxdim=10)
        v_field3 = VectorField((v0 * ones_mps(sites, 1), 0.0 * ones_mps(sites, 1)), arch, (bcsvx, bcsvy))
        p_field3 = ScalarField(sites, arch, bcsp)
        experiment3 = DifferenTT.INSExperiment(
            grid, v_field3, p_field3, Re, nothing; stencil_order=2, maxdim=10, cutoff=1e-8
        )

        observed_sweep_kwargs = DifferenTT.SweepingArgs(; nsweeps=3, maxdim=10, (observer!)=ResidualObserver(rtol=1e-2))
        observed_solver = DifferenTT.ALS(; sweep_kwargs=observed_sweep_kwargs, updater_kwargs=updater_kwargs)

        DifferenTT.chorin_evolve!(experiment3, observed_solver, stepper3, 2, nothing, "")

        @test length(experiment3.u_fields) == 3
        @test length(experiment3.p_field) == 2

        # Regression test: 5 steps saved every 3 store the initial velocity and the steps 3 and 5. The step 4 was also
        # stored, because the last step was appended as an extra checkpoint
        stepper5 = RK4(time_evol_kwargs=TimeEvolArgs(0.0, 0.005, 5), cutoff=1e-8, maxdim=10)
        v_field5 = VectorField((v0 * ones_mps(sites, 1), 0.0 * ones_mps(sites, 1)), arch, (bcsvx, bcsvy))
        experiment5 = DifferenTT.INSExperiment(
            grid, v_field5, ScalarField(sites, arch, bcsp), Re, nothing; stencil_order=2, maxdim=10, cutoff=1e-8
        )
        DifferenTT.chorin_evolve!(experiment5, observed_solver, stepper5, 3, nothing, "")
        @test length(experiment5.u_fields) == 3
        @test length(experiment5.p_field) == 2
        # The observer in the solver was bound per step, never stored
        @test isnothing(observed_solver.sweep_kwargs.observer!.A)
        @test isnothing(observed_solver.sweep_kwargs.observer!.b)
    end

    @testset "A divergent velocity loses most of its divergence in one step (q=$q)" for q in (2, 8)
        # Periodic unit box, 16 × 16 nodes, a smooth velocity with divergence. On a two-site TT, a single two-site ALS
        # update is a direct solve of the pressure system
        tt_arch = TTArchitecture((16, 16))
        tt_grid = Grid(tt_arch, (1.0, 1.0))
        tt_sites = siteinds(tt_arch)
        per = BoundaryCond1d(PeriodicBC(), PeriodicBC())
        pbcs = BoundaryConditions(per, per)
        ux = [sin(2π * x) * cos(2π * y) + 0.3 * cos(2π * y) for x in tt_grid.points[1], y in tt_grid.points[2]]
        uy = [0.5 * sin(2π * x) * sin(2π * y) + sin(4π * y) for x in tt_grid.points[1], y in tt_grid.points[2]]
        as_mps(F) = MPS(ITensor(F, tt_sites[1], tt_sites[2]), tt_sites)
        u_div = VectorField((as_mps(ux), as_mps(uy)), tt_arch, (pbcs, pbcs))
        experiment_div = quietly(
            () -> INSExperiment(
                tt_grid,
                u_div,
                ScalarField(tt_sites, tt_arch, pbcs),
                100,
                nothing;
                stencil_order=q,
                maxdim=256,
                cutoff=1e-14,
            ),
        )
        solver_div = ALS(;
            sweep_kwargs=SweepingArgs(; nsweeps=1, maxdim=256, cutoff=1e-15),
            updater_kwargs=LocalUpdaterArgs(; ishermitian=true, isposdef=true, rtol=1e-13, maxiter=1000, krylovdim=1),
        )
        divergence_of(u) = norm(
            DifferenTT.divergence(
                u,
                DifferenTT.DifferentialOps(tt_grid, u_div; stencil_order=q, cutoff=1e-14, maxdim=256);
                cutoff=1e-14,
                maxdim=256,
            ),
        )
        initial_divergence = divergence_of(last(experiment_div.u_fields))
        @test initial_divergence > 10 # the velocity is far from solenoidal

        stepper_div = RK4(time_evol_kwargs=TimeEvolArgs(0.0, 1e-3, 1), cutoff=1e-14, maxdim=256)
        quietly(() -> chorin_evolve!(experiment_div, solver_div, stepper_div, 1, nothing, ""))
        # The stored velocity is the predicted field of the next step (the projection is applied at the next step, see
        # chorin_step), whose divergence is only that of the projection plus one explicit step. It is well below the
        # initial one: 0.18 (q = 2) and 0.09 (q = 8) of it were measured; the bound leaves a factor 1.6 of margin
        @test divergence_of(last(experiment_div.u_fields)) < 0.3 * initial_divergence
    end

    @testset "chorin_evolve! argument checks and logging" begin
        function fresh_experiment(; initialize_velocity=true, initialize_pressure=false)
            v = VectorField((v0 * ones_mps(sites, 1), 0.0 * ones_mps(sites, 1)), arch, (bcsvx, bcsvy))
            if !initialize_velocity
                v = VectorField(arch, (bcsvx, bcsvy))
            end
            p = ScalarField(sites, arch, bcsp)
            initialize_pressure && push!(p, ones_mps(sites, 1))
            return DifferenTT.INSExperiment(grid, v, p, Re, nothing; stencil_order=2, maxdim=10, cutoff=1e-8)
        end
        stepper2 = RK4(time_evol_kwargs=TimeEvolArgs(0.0, 0.002, 2), cutoff=1e-8, maxdim=10)

        @test_throws ArgumentError chorin_evolve!(
            fresh_experiment(; initialize_velocity=false), solver, stepper2, 1, nothing, ""
        )
        @test_throws ArgumentError chorin_evolve!(
            fresh_experiment(; initialize_pressure=true), solver, stepper2, 1, nothing, ""
        )
        for save_every in (0, -2)
            @test_throws ArgumentError chorin_evolve!(fresh_experiment(), solver, stepper2, save_every, nothing, "")
        end

        # With a logger, the fields and the residual norms of the stored pressure solves are written to the file
        mktempdir() do dir
            logged_experiment = fresh_experiment()
            chorin_evolve!(logged_experiment, solver, stepper2, 1, init_logger(dir), "chorin_run")
            path = joinpath(dir, "chorin_run.jld2")
            @test isfile(path)
            jldopen(path, "r") do file
                @test Set(keys(file)) == Set(["u_fields", "p_field", "linsolve_errorlog"])
                errorlog = file["linsolve_errorlog"]
                # One residual per stored pressure, none of them negative
                @test length(errorlog) == length(logged_experiment.p_field) == 2
                @test all(>=(0), errorlog)
                @test length(file["u_fields"]) == 3
                @test length(file["p_field"]) == 2
            end
        end
        # Without a logger nothing is written
        mktempdir() do dir
            chorin_evolve!(fresh_experiment(), solver, stepper2, 1, nothing, "never")
            @test isempty(readdir(dir))
        end
    end
end

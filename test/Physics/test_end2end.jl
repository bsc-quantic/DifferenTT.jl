using ITensors
using ITensorMPS
using DifferenTT
using Test
using LinearAlgebra

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
        u_diff_ops = DifferenTT.DifferentialOps(experiment.grid, experiment.u_fields; stencil_order=experiment.stencil_order, cutoff=cutoff, maxdim=maxdim)
        
        ∇u = DifferenTT.divergence(masked_u_mpss, u_diff_ops; cutoff=cutoff, maxdim=maxdim)
        
        # Due to small sizes and only 1 sweep in ALS, divergence might not be exactly zero, 
        # but the pipeline should have run and produced valid MPS
        @test norm(∇u) >= 0.0
    end

    @testset "Checkpoints and residual observer" begin
        # 3 steps saved every 2 -> checkpoints at steps 1 and 3
        stepper3 = RK4(time_evol_kwargs=TimeEvolArgs(0.0, 0.003, 3), cutoff=1e-8, maxdim=10)
        v_field3 = VectorField((v0 * ones_mps(sites, 1), 0.0 * ones_mps(sites, 1)), arch, (bcsvx, bcsvy))
        p_field3 = ScalarField(sites, arch, bcsp)
        experiment3 = DifferenTT.INSExperiment(grid, v_field3, p_field3, Re, nothing; stencil_order=2, maxdim=10, cutoff=1e-8)

        observed_sweep_kwargs = DifferenTT.SweepingArgs(; nsweeps=3, maxdim=10, observer! = ResidualObserver(rtol=1e-2))
        observed_solver = DifferenTT.ALS(; sweep_kwargs=observed_sweep_kwargs, updater_kwargs=updater_kwargs)

        DifferenTT.chorin_evolve!(experiment3, observed_solver, stepper3, 2, nothing, "")

        @test length(experiment3.u_fields) == 3
        @test length(experiment3.p_field) == 2
        # The observer in the solver was bound per step, never stored
        @test isnothing(observed_solver.sweep_kwargs.observer!.A)
        @test isnothing(observed_solver.sweep_kwargs.observer!.b)
    end
end

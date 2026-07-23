using DifferenTT
using Test
using ITensors
using ITensorMPS

@testset "insexperiment.jl" begin
    n = 3
    tt_arch = TTArchitecture((2^n, 2^n))
    grid = Grid(tt_arch, (1.0, 1.0))
    time_evol = TimeEvol(0.1, 1.0, 10)
    
    sites = [Index(2, "Site,n=$i") for i in 1:2]
    u_mps = random_mps(sites)
    
    bc_periodic = BoundaryCond1d(PeriodicBC(), PeriodicBC())
    bcs = BoundaryConditions(bc_periodic, bc_periodic)
    
    sfield = ScalarField(u_mps, tt_arch, bcs)
    vfield = VectorField((sfield, sfield))
    
    @testset "INSExperiment instantiation" begin
        exp = DifferenTT.INSExperiment(grid, time_evol, vfield, sfield, 100.0, nothing; stencil_order=2, maxdim=10, cutoff=1e-5)
        @test exp.Re == 100.0
        @test exp.stencil_order == 2
        @test exp.maxdim == 10
        @test exp.cutoff == 1e-5
        @test isnothing(exp.object_mask)
        
        # Test checks
        DifferenTT.check_cfl(exp)
        DifferenTT.check_kolmogorov_scale(exp)
    end
    
    @testset "INSExperiment validation" begin
        # Invalid object mask
        mask = random_mps(siteinds("S=1/2", 3)) # wrong sites
        @show !isnothing(mask)        
        @show vfield.usites != sfield.usites != siteinds(mask)
        @test_throws ArgumentError DifferenTT.INSExperiment(grid, time_evol, vfield, sfield, 100.0, mask)
    end
end

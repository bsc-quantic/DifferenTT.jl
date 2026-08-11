using DifferenTT
using Test
using ITensors
using ITensorMPS

@testset "linearsyscorrections.jl" begin
    n = 3
    qtt_arch = QTTArchitecture((n, n))
    grid = Grid(qtt_arch, (1.0, 1.0))
    
    sites = siteinds(qtt_arch)
    u_mps = random_mps(sites)
    
    # 1. Periodic BCs -> corrections should be nothing
    bc_periodic = BoundaryCond1d(PeriodicBC(), PeriodicBC())
    bcs_periodic = BoundaryConditions(bc_periodic, bc_periodic)
    sfield_per = ScalarField(u_mps, qtt_arch, bcs_periodic)
    vfield_per = VectorField((sfield_per, sfield_per))
    
    @testset "Periodic BCs Corrections" begin
        # For Periodic, onedim_boundary_nullifier returns nothing, so divergence_dirichlet_correction might return nothing
        div_corr = DifferenTT.divergence_dirichlet_correction(vfield_per; stencil_order=2)
        @test isnothing(div_corr)
        
        p_corr = DifferenTT.scalar_dirichlet_correction(sfield_per, grid, 2; stencil_order=2)
        @test isnothing(p_corr)
    end
    
    # 2. Dirichlet BCs
    bc_dirichlet = BoundaryCond1d(DirichletBC(1.0), DirichletBC(1.0))
    bcs_dirichlet = BoundaryConditions(bc_dirichlet, bc_dirichlet)
    sfield_dir = ScalarField(u_mps, qtt_arch, bcs_dirichlet)
    vfield_dir = VectorField((sfield_dir, sfield_dir))
    
    @testset "Dirichlet BCs Corrections" begin
        div_corr = DifferenTT.divergence_dirichlet_correction(vfield_dir; stencil_order=2)
        @test div_corr isa MPO
        
        p_corr_dim1 = DifferenTT.scalar_dirichlet_correction_dim(sfield_dir, grid, 2, 1; stencil_order=2)
        @test p_corr_dim1 isa MPO
        
        p_corr_dim2 = DifferenTT.scalar_dirichlet_correction_dim(sfield_dir, grid, 2, 2; stencil_order=2)
        @test p_corr_dim2 isa MPO
        
        p_corr_total = DifferenTT.scalar_dirichlet_correction(sfield_dir, grid, 2; stencil_order=2)
        @test p_corr_total isa MPO
    end
end

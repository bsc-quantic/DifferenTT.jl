using DifferenTT
using Test
using ITensors
using ITensorMPS

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
        te = TimeEvol(0.0, 0.001, 0.0001)
        Re = 100.0

        u_fields = VectorField((ux, uy), TTArchitecture(arch_2d), (bc2d, bc2d))
        p_field = ScalarField(sites, TTArchitecture(arch_2d), bc2d)

        experiment = INSExperiment(grid, te, u_fields, p_field, Re, nothing; stencil_order=2)

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
        te = TimeEvol(0.0, 0.001, 0.0001)
        Re = 100.0

        u_fields = VectorField((ux, uy, uz), TTArchitecture(arch_3d), (bc3d, bc3d, bc3d))
        p_field = ScalarField(sites, TTArchitecture(arch_3d), bc3d)

        experiment = INSExperiment(grid, te, u_fields, p_field, Re, nothing; stencil_order=2)

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
        te = TimeEvol(0.0, 0.001, 0.0001)
        Re = 100.0

        u_fields = VectorField((ux, uy), TTArchitecture(arch_2d), (bc2d, bc2d))
        p_field_periodic = ScalarField(TTArchitecture(arch_2d), bc2d)

        exp_periodic = INSExperiment(grid, te, u_fields, p_field_periodic, Re, nothing; stencil_order=2)
        @test DifferenTT.regularization_check(exp_periodic) == true

        dirichlet_bc1d = BoundaryCond1d(DirichletBC(0.0), DirichletBC(0.0))
        dirichlet_bc2d = BoundaryConditions(dirichlet_bc1d, dirichlet_bc1d)
        p_field_dirichlet = ScalarField(TTArchitecture(arch_2d), dirichlet_bc2d)

        exp_dirichlet = INSExperiment(grid, te, u_fields, p_field_dirichlet, Re, nothing; stencil_order=2)
        @test DifferenTT.regularization_check(exp_dirichlet) == false
    end

    @testset "check_cfl 2D" begin
        grid = Grid(TTArchitecture(arch_2d), (1.0, 1.0))
        te_ok = TimeEvol(0.0, 0.01, 0.001)
        Re = 100.0
        @test (check_cfl(grid, te_ok, Re); true)

        te_bad = TimeEvol(0.0, 100.0, 1)
        @test (check_cfl(grid, te_bad, Re); true)
    end

    @testset "INSExperiment construction 2D" begin
        # 
        sites = siteinds(TTArchitecture(arch_2d))
        ux = random_mps(sites)
        uy = random_mps(sites)

        grid = Grid(TTArchitecture(arch_2d), (1.0, 1.0))
        te = TimeEvol(0.0, 0.001, 0.0001)
        Re = 100.0

        u_fields = VectorField((ux, uy), TTArchitecture(arch_2d), (bc2d, bc2d))
        p_field = ScalarField(TTArchitecture(arch_2d), bc2d)

        experiment = INSExperiment(grid, te, u_fields, p_field, Re, nothing; stencil_order=2)
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
        te = TimeEvol(0.0, 0.001, 0.0001)
        Re = 100.0

        u_fields = VectorField((ux, uy), TTArchitecture(arch_2d), (bc2d, bc2d))
        p_field = ScalarField(TTArchitecture(arch_2d), bc2d)

        experiment = INSExperiment(grid, te, u_fields, p_field, Re, nothing; stencil_order=2)

        u_diff_ops = DifferenTT.DifferentialOps(experiment.grid, experiment.u_fields; stencil_order=experiment.stencil_order)
        u_boundary_nullifier = DifferenTT.BoundaryNull(experiment.u_fields; stencil_order=experiment.stencil_order)
        
        rhs_func = NavierStokesPredictorRHS(experiment, u_diff_ops, u_boundary_nullifier)
        
        rhs_out = rhs_func((ux, uy), 0.0, 1)
        @test length(rhs_out) == 2
        @test rhs_out[1] isa MPS
        @test rhs_out[2] isa MPS
    end
    @testset "symmetrize_rhs" begin
        arch = (4,)
        sites = siteinds(TTArchitecture(arch))
        
        # Neumann boundaries on all sides
        neumann_bc = NeumannBC(0.0)
        bc1d = BoundaryCond1d(neumann_bc, neumann_bc)
        bc = BoundaryConditions(bc1d)
        
        p_field = ScalarField(sites, TTArchitecture(arch), bc)
        
        # Create an RHS with all ones
        rhs = ones_mps(sites, 1, Vector{Float64})
        
        # Symmetrize the RHS
        sym_rhs = symmetrize_rhs(p_field, rhs)
        
        # Contract to dense vector
        dense_rhs = contract_mps(rhs)
        dense_sym_rhs = contract_mps(sym_rhs)
        
        @test dense_rhs[1] == 1.0
        @test dense_sym_rhs[1] == 0.5
        @test dense_sym_rhs[end] == 0.5
        @test dense_sym_rhs[2] == 1.0
        @test dense_sym_rhs[3] == 1.0
        
        # 2D test
        arch_2d = (4, 4)
        sites_2d = siteinds(TTArchitecture(arch_2d))
        bc2d = BoundaryConditions(bc1d, bc1d)
        p_field_2d = ScalarField(sites_2d, TTArchitecture(arch_2d), bc2d)
        rhs_2d = ones_mps(sites_2d, 1, Vector{Float64})
        sym_rhs_2d = symmetrize_rhs(p_field_2d, rhs_2d)
        dense_sym_rhs_2d = reshape(contract_mps(sym_rhs_2d), 4, 4)
        
        # Corners have 2 boundaries, so 1.0 - 0.5 - 0.5 = 0.0
        @test dense_sym_rhs_2d[1, 1] == 0.0
        @test dense_sym_rhs_2d[1, end] == 0.0
        @test dense_sym_rhs_2d[end, 1] == 0.0
        @test dense_sym_rhs_2d[end, end] == 0.0
        
        # Edges have 1 boundary, so 1.0 - 0.5 = 0.5
        @test dense_sym_rhs_2d[2, 1] == 0.5
        @test dense_sym_rhs_2d[1, 2] == 0.5
        
        # Interior has 0 boundaries, so 1.0
        @test dense_sym_rhs_2d[2, 2] == 1.0
    end
end


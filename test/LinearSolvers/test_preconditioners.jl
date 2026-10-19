using DifferenTT
using Test
using Logging
using ITensors
using ITensorMPS
using LinearAlgebra

include(joinpath(@__DIR__, "..", "testutils.jl"))
using .TestUtils

@testset "preconditioners.jl" begin
    n = 4
    sites = siteinds("S=1/2", n)

    @testset "rank1_preconditioner" begin
        # Build a well-conditioned SPD operator: identity MPO
        Id = MPO(sites, "Id")
        M⁻¹ = rank1_preconditioner(Id)

        @test M⁻¹ isa Preconditioner
        @test M⁻¹.operator isa MPO
        @test length(M⁻¹.operator) == n
        @test siteinds(M⁻¹.operator; plev=0) == siteinds(Id; plev=0)

        # For identity operator, preconditioner should also approximate identity
        # Apply M⁻¹ to a test vector and check it's close to the original
        test_mps = random_mps(sites)
        result = apply(M⁻¹.operator, test_mps)
        @test result isa MPS
        @test length(result) == n
        @test siteinds(result) == sites
    end

    @testset "rank1_preconditioner with random operator" begin
        A = random_mpo(sites)
        M⁻¹ = rank1_preconditioner(A)

        @test M⁻¹ isa Preconditioner
        @test M⁻¹.operator isa MPO
        @test length(M⁻¹.operator) == n

        # All bond dimensions should be 1 (rank-1 truncation of input)
        for i in 1:(n-1)
            @test linkdim(M⁻¹.operator, i) == 1
        end
    end

    @testset "newtoniter_preconditioner" begin
        A = MPO(sites, "Id")
        M⁻¹ = newtoniter_preconditioner(A, 2; maxdim=4, cutoff=1e-10)

        @test M⁻¹ isa Preconditioner
        @test M⁻¹.operator isa MPO
        @test length(M⁻¹.operator) == n

        # Test with custom init
        init_M⁻¹ = rank1_preconditioner(A).operator
        M⁻¹_custom = newtoniter_preconditioner(A, init_M⁻¹, 2)
        @test M⁻¹_custom isa Preconditioner
    end

    @testset "linsolve_preconditioner" begin
        A = MPO(sites, "Id")
        solver = ALS(sweep_kwargs=DifferenTT.SweepingArgs(nsweeps=1, maxdim=2, cutoff=1e-8))

        M⁻¹_1 = linsolve_preconditioner(A, solver; preconditioning=false, residual_tol=0.9)
        @test M⁻¹_1 isa Preconditioner
        @test M⁻¹_1.operator isa MPO

        # With initial guess
        init_x0 = rank1_preconditioner(A).operator
        M⁻¹_2_init = linsolve_preconditioner(A, solver; linsolve_x0=init_x0)
        @test M⁻¹_2_init isa Preconditioner

        # With preconditioning inside linsolve_preconditioner
        M⁻¹_2_prec = linsolve_preconditioner(A, solver; preconditioning=true)
        @test M⁻¹_2_prec isa Preconditioner
    end

    @testset "triang_mpo" begin
        # upper
        U = DifferenTT.upper_triang_mpo(sites)
        @test U isa MPO
        @test length(U) == n

        # lower
        L = DifferenTT.lower_triang_mpo(sites)
        @test L isa MPO
        @test length(L) == n

        # projector
        P_upper = DifferenTT.triang_projector_mpo(sites, :upper)
        @test P_upper isa MPO

        P_lower = DifferenTT.triang_projector_mpo(sites, :lower)
        @test P_lower isa MPO
    end

    @testset "fai_preconditioner" begin
        A = MPO(sites, "Id")
        solver = ALS(sweep_kwargs=DifferenTT.SweepingArgs(nsweeps=1, maxdim=2, cutoff=1e-8))
        D_kwargs = (; maxdim=2, cutoff=1e-8)

        M⁻¹ = fai_preconditioner(A, 0.5, solver, D_kwargs)
        @test M⁻¹ isa FactorizedPreconditioner
        @test M⁻¹.operator isa MPO
        @test length(M⁻¹.operator) == n

        # The residual of the projected system is computed and logged only on request
        @test_logs min_level=Logging.Info fai_preconditioner(A, 0.5, solver, D_kwargs)
        @test_logs (:info, r"rel residual") min_level=Logging.Info fai_preconditioner(
            A, 0.5, solver, D_kwargs; compute_residual=true
        )
    end

    @testset "SincQuadraturePreconditioner construction" begin
        arch = DifferenTT.TTArchitecture((4, 4))
        sites = ITensorMPS.siteinds(arch)

        A1 = [2.0 -1.0 0.0 0.0; -1.0 2.0 -1.0 0.0; 0.0 -1.0 2.0 -1.0; 0.0 0.0 -1.0 2.0]
        A2 = A1

        A_list = [A1, A2]
        M = 2

        prec = DifferenTT.sincquadrature_preconditioner(arch, sites, A_list, M)

        @test prec isa DifferenTT.Preconditioner
        @test prec.operator isa MPO
        @test maxlinkdim(prec.operator) == 2*M + 1
        @test length(prec.operator) == 2

        wrong_arch1 = DifferenTT.TTArchitecture((4, 4, 4))
        wrong_arch2 = DifferenTT.TTArchitecture((5, 4))
        @test_throws DimensionMismatch DifferenTT.sincquadrature_preconditioner(wrong_arch1, sites, A_list, M)
        @test_throws DimensionMismatch DifferenTT.sincquadrature_preconditioner(wrong_arch2, sites, A_list, M)

        A3 = [2.0 -1.0 0.0 0.0; -1.0 2.0 -1.0 0.0; 0.0 -1.0 2.0 -1.0]
        A4 = [2.0 -1.0 0.0; -1.0 2.0 -1.0; 0.0 -1.0 2.0]

        wrong_A_list1 = [A1, A3]
        wrong_A_list2 = [A1, A4]

        @test_throws DimensionMismatch DifferenTT.sincquadrature_preconditioner(arch, sites, wrong_A_list1, M)
        @test_throws DimensionMismatch DifferenTT.sincquadrature_preconditioner(arch, sites, wrong_A_list2, M)
        @test_throws ArgumentError DifferenTT.sincquadrature_preconditioner(arch, sites, [-A1, -A2], M)
    end

    @testset "SincQuadraturePreconditioner spectral normalization" begin
        arch = DifferenTT.TTArchitecture((4, 4))
        sites = ITensorMPS.siteinds(arch)
        dense(mpo) = reshape(Array(prod(mpo), sites[1]', sites[2]', sites[1], sites[2]), 16, 16)

        # 1D Dirichlet Laplacians with the 1/h² scaling that motivates the normalization
        h = 1 / 5
        A1 = Matrix(SymTridiagonal(fill(2.0, 4), fill(-1.0, 3))) / h^2
        A_list = [A1, 2 * A1]
        L0 = kron(Matrix(1.0I, 4, 4), A_list[1]) + kron(A_list[2], Matrix(1.0I, 4, 4))
        M = 3

        # Normalized B_M is homogeneous of degree -1 in the operator scaling
        B = dense(DifferenTT.sincquadrature_preconditioner(arch, sites, A_list, M).operator)
        B_scaled = dense(DifferenTT.sincquadrature_preconditioner(arch, sites, 10.0 .* A_list, M).operator)
        @test B_scaled ≈ B / 10

        κ(B) = (λ=eigvals(Symmetric((B * L0 + (B * L0)') / 2)); maximum(λ) / minimum(λ))
        B_raw = dense(
            DifferenTT.sincquadrature_preconditioner(arch, sites, A_list, M; normalize_spectrum=false).operator
        )
        @test κ(B) < 1.25
        @test κ(B) < κ(B_raw)
    end

    @testset "inverse_laplacian_preconditioner constructors" begin
        arch = QTTArchitecture((4,))
        test_sites = [Index(2, "Site,n=$i") for i in 1:4]
        grid = Grid(arch, (1.0,))

        bc_dd = BoundaryConditions(BoundaryCond1d(DirichletBC(0.0), DirichletBC(0.0)))
        bc_dn = BoundaryConditions(BoundaryCond1d(DirichletBC(0.0), NeumannBC(0.0)))

        # Dirichlet-Dirichlet case
        inv_dd_prec = DifferenTT.inverse_laplacian_preconditioner(test_sites, arch, grid, bc_dd)
        inv_dd_mat = DifferenTT.contract_mpo(inv_dd_prec.operator)

        u_init_dd = random_mps(test_sites)
        sfield_dd = ScalarField(u_init_dd, arch, bc_dd)
        lap_dd_mpo = DifferenTT.symmetrized_laplacian(sfield_dd, grid; stencil_order=2)
        lap_dd_mat = DifferenTT.contract_mpo(lap_dd_mpo)

        @test inv(lap_dd_mat) ≈ inv_dd_mat atol=1e-8

        # Dirichlet-Neumann case
        inv_dn_prec = DifferenTT.inverse_laplacian_preconditioner(test_sites, arch, grid, bc_dn)
        inv_dn_mat = DifferenTT.contract_mpo(inv_dn_prec.operator)

        u_init_dn = random_mps(test_sites)
        sfield_dn = ScalarField(u_init_dn, arch, bc_dn)
        lap_dn_mpo = DifferenTT.symmetrized_laplacian(sfield_dn, grid; stencil_order=2)
        lap_dn_mat = DifferenTT.contract_mpo(lap_dn_mpo)

        @test inv(lap_dn_mat) ≈ inv_dn_mat atol=1e-8
    end
end

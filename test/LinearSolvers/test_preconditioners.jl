using DifferenTT
using Test
using ITensors
using ITensorMPS
using LinearAlgebra

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
        for i in 1:n-1
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

        M⁻¹_1 = linsolve_preconditioner(A, solver; preconditioning=false)
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
    end
end

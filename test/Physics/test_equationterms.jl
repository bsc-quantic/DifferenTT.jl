using Test
using ITensors
using ITensorMPS
using LinearAlgebra
using DifferenTT

# Densely contracting 8-site MPOs yields tensors with up to 16 indices. Raise the ITensors
# order warning threshold for this file only, restoring the previous value afterwards.
previous_warn_order = ITensors.set_warn_order(17)

try
@testset "symmetrized_laplacian optimal MPO construction" begin
    ITensors.set_warn_order(17)
    # Helper to construct the exact dense matrix using directsum (the old inefficient way)
    # This serves as the ground truth to ensure the exact block MPO is mathematically identical.
    function directsum_symmetrized_laplacian(field::ScalarField{N}, grid::Grid{N}; asymmetry::Float64=1.0, stencil_order::Int=8) where N
        d2_ops = ntuple(i -> begin
                onedim_op = DifferenTT.onedim_differential_operator(2, field.arch, i, field.bcu[i], DifferenTT.backend_type(field); stencil_order=stencil_order, cutoff=0.0, maxdim=100)
                neumann_symmetrization = DifferenTT.onedim_neumann_symmetrization_correction(onedim_op, field.arch, i, field.bcu[i]; stencil_order=stencil_order, cutoff=0.0, maxdim=100)
                
                if !isnothing(neumann_symmetrization)
                    onedim_op = DifferenTT._nothing_sum_handler(onedim_op, neumann_symmetrization; cutoff=0.0, maxdim=100)
                end
    
                if onedim_op isa MPO
                    onedim_op = onedim_op * (1/grid.dx[i]^2)
                else
                    onedim_op = onedim_op .* (1/grid.dx[i]^2)
                end
                
                stacked_op = DifferenTT.ndim_operator_mpo(onedim_op, i, field.usites, field.arch)
                stacked_op = i == 1 ? asymmetry * stacked_op : stacked_op
                return stacked_op
            end, N)
    
        return sum([d2_ops...]; alg="directsum")
    end

    @testset "TT Architecture" begin
        nodes = (8, 8, 8)
        box_size = (1.0, 1.0, 1.0)
        arch = TTArchitecture(nodes)
        grid = Grid(arch, box_size)
        
        sites = siteinds(8, 3)
        bcs = BoundaryConditions(
            BoundaryCond1d(PeriodicBC(), PeriodicBC()),
            BoundaryCond1d(PeriodicBC(), PeriodicBC()),
            BoundaryCond1d(PeriodicBC(), PeriodicBC())
        )
        field = ScalarField(sites, arch, bcs)
        
        mpo_optimal = symmetrized_laplacian(field, grid; stencil_order=2)
        mpo_direct = directsum_symmetrized_laplacian(field, grid; stencil_order=2)
        
        # Test exact bond dimension for TT
        # For N dimensions, optimal bond dimension is 2
        @test maxlinkdim(mpo_optimal) == 2
        
        # Test mathematical equivalence
        dense_optimal = DifferenTT.contract_mpo(mpo_optimal)
        dense_direct = DifferenTT.contract_mpo(mpo_direct)
        @test norm(dense_optimal - dense_direct) < 1e-10
    end

    @testset "QTT Architecture (Sequential)" begin
        nodes = (4, 4) # 4 qubits per dimension -> 8 sites total
        box_size = (1.0, 1.0)
        arch = QTTArchitecture(nodes; interleaved=false)
        grid = Grid(arch, box_size)
        
        sites = siteinds(2, 8)
        bcs = BoundaryConditions(
            BoundaryCond1d(PeriodicBC(), PeriodicBC()),
            BoundaryCond1d(PeriodicBC(), PeriodicBC())
        )
        field = ScalarField(sites, arch, bcs)
        
        mpo_optimal = symmetrized_laplacian(field, grid; stencil_order=2)
        mpo_direct = directsum_symmetrized_laplacian(field, grid; stencil_order=2)
        
        # Test minimal bond dimension
        # Should be much smaller than direct sum (for large systems), but for tiny systems 
        # the exact block structure has exactly 2 extra states.
        @test maxlinkdim(mpo_optimal) <= maxlinkdim(mpo_direct) + 2
        
        # Test mathematical equivalence
        dense_optimal = DifferenTT.contract_mpo(mpo_optimal)
        dense_direct = DifferenTT.contract_mpo(mpo_direct)
        @test norm(dense_optimal - dense_direct) < 1e-10
    end

    @testset "QTT Architecture (Interleaved)" begin
        nodes = (4, 4) # 4 qubits per dimension -> 8 sites total
        box_size = (1.0, 1.0)
        arch = QTTArchitecture(nodes; interleaved=true)
        grid = Grid(arch, box_size)
        
        sites = siteinds(2, 8)
        bcs = BoundaryConditions(
            BoundaryCond1d(PeriodicBC(), PeriodicBC()),
            BoundaryCond1d(PeriodicBC(), PeriodicBC())
        )
        field = ScalarField(sites, arch, bcs)
        
        mpo_optimal = symmetrized_laplacian(field, grid; stencil_order=2)
        mpo_direct = directsum_symmetrized_laplacian(field, grid; stencil_order=2)
        
        # Test minimal bond dimension
        @test maxlinkdim(mpo_optimal) <= maxlinkdim(mpo_direct) + 2
        
        # Test mathematical equivalence
        dense_optimal = DifferenTT.contract_mpo(mpo_optimal)
        dense_direct = DifferenTT.contract_mpo(mpo_direct)
        @test norm(dense_optimal - dense_direct) < 1e-10
    end
end

@testset "ndim_operator_sum_mpo pairwise FSM" begin
    # Non-trivial variable -> site mapping: variable 2 on site 1, 3 on site 2, 1 on site 3
    arch = TTArchitecture((3, 4, 5), Dict(2 => 1, 3 => 2, 1 => 3))
    sites = [Index(arch.var_dims[arch.site_map[i]], "Site,n=$i") for i in 1:3]

    O = ntuple(c -> randn(arch.var_dims[c], arch.var_dims[c]), 3)
    P = ntuple(c -> randn(arch.var_dims[c], arch.var_dims[c]), 3)
    Q = ntuple(c -> randn(arch.var_dims[c], arch.var_dims[c]), 3)

    # Reference: explicit sum of lifted operators and products over physical site pairs i < j
    terms = MPO[ndim_operator_mpo(O[c], c, sites, arch) for c in 1:3]
    for i in 1:3, j in i+1:3
        ci, cj = arch.site_map[i], arch.site_map[j]
        push!(terms, apply(ndim_operator_mpo(P[ci], ci, sites, arch), ndim_operator_mpo(Q[cj], cj, sites, arch); alg="naive", truncate=false))
    end
    reference = DifferenTT.contract_mpo(sum(terms; alg="directsum"))

    mpo_pairs = DifferenTT.ndim_operator_sum_mpo(O, sites, arch; pair_ops=(P, Q))
    @test maxlinkdim(mpo_pairs) == 3
    @test norm(DifferenTT.contract_mpo(mpo_pairs) - reference) < 1e-10 * norm(reference)

    # Without pairs the Kronecker-sum FSM keeps bond dimension 2
    mpo_sum = DifferenTT.ndim_operator_sum_mpo(O, sites, arch)
    sum_reference = DifferenTT.contract_mpo(sum(terms[1:3]; alg="directsum"))
    @test maxlinkdim(mpo_sum) == 2
    @test norm(DifferenTT.contract_mpo(mpo_sum) - sum_reference) < 1e-10 * norm(sum_reference)
end

@testset "equicorrelated_laplacian" begin
    N = 3
    arch = TTArchitecture((6, 6, 6))
    grid = Grid(arch, (1.0, 1.0, 1.0))
    sites = siteinds(arch)
    dirichlet = BoundaryCond1d(DirichletBC(0.0), DirichletBC(0.0))
    field = ScalarField(sites, arch, BoundaryConditions(dirichlet, dirichlet, dirichlet))

    laplacian = DifferenTT.contract_mpo(symmetrized_laplacian(field, grid; stencil_order=2))

    for ρ in (0.0, 0.3, 0.9)
        mpo = equicorrelated_laplacian(field, grid; correlation=ρ, stencil_order=2)
        @test maxlinkdim(mpo) == (iszero(ρ) ? 2 : 3)

        L = DifferenTT.contract_mpo(mpo)
        @test norm(L - L') < 1e-10 * norm(L)

        if iszero(ρ)
            @test norm(L - laplacian) < 1e-10 * norm(laplacian)
        end

        # Spectral floor: λ_min(K) (-Δ) ⪯ -L ⪯ λ_max(K) (-Δ)
        λ = eigvals(Symmetric(-L), Symmetric(-laplacian))
        @test minimum(λ) >= (1 - ρ) - 1e-10
        @test maximum(λ) <= (1 + (N - 1) * ρ) + 1e-10
    end

    neumann = BoundaryCond1d(NeumannBC(0.0), NeumannBC(0.0))
    neumann_field = ScalarField(sites, arch, BoundaryConditions(neumann, dirichlet, dirichlet))
    @test_throws ArgumentError equicorrelated_laplacian(neumann_field, grid; correlation=0.5, stencil_order=2)

    qtt_arch = QTTArchitecture((3, 3); interleaved=false)
    qtt_field = ScalarField(siteinds(2, 6), qtt_arch, BoundaryConditions(dirichlet, dirichlet))
    @test_throws ArgumentError equicorrelated_laplacian(qtt_field, Grid(qtt_arch, (1.0, 1.0)); correlation=0.5)
end
finally
    ITensors.set_warn_order(previous_warn_order)
end
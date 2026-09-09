using Test
using ITensors
using ITensorMPS
using LinearAlgebra
using DifferenTT

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
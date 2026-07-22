using Test
using DifferenTT
using Plots
using ITensors
using ITensorMPS

@testset "Visualization Tests" begin
    # Test for N=2
    bc1d = BoundaryCond1d(PeriodicBC(), PeriodicBC())
    bcs2 = BoundaryConditions(bc1d, bc1d)
    arch2 = QTTArchitecture((2, 2))
    grid2 = Grid(arch2, (1.0, 1.0))
    sf2 = ScalarField(arch2, bcs2)
    vf2 = VectorField(arch2, (bcs2, bcs2))
    push!(sf2, random_mps(sf2.usites))
    push!(vf2, (random_mps(vf2.usites), random_mps(vf2.usites)))

    @testset "Scalar Field Visualization (N=2)" begin
        p = visualize_scalar(sf2, grid2; npoints=4)
        @test p isa Plots.Plot
    end

    @testset "Vector Field Visualization (N=2)" begin
        p = visualize_vector(vf2, grid2; npoints=4, quiver_npoints=4)
        @test p isa Plots.Plot
    end

    @testset "Vorticity Visualization (N=2)" begin
        p = visualize_vorticity(vf2, grid2; npoints=4, quiver_npoints=4)
        @test p isa Plots.Plot
    end

    # Test for N=3
    bcs3 = BoundaryConditions(bc1d, bc1d, bc1d)
    arch3 = QTTArchitecture((2, 2, 2))
    grid3 = Grid(arch3, (1.0, 1.0, 1.0))
    sf3 = ScalarField(arch3, bcs3)
    vf3 = VectorField(arch3, (bcs3, bcs3, bcs3))
    push!(sf3, random_mps(sf3.usites))
    push!(vf3, (random_mps(vf3.usites), random_mps(vf3.usites), random_mps(vf3.usites)))

    @testset "Scalar Field Visualization (N=3)" begin
        p = visualize_scalar(sf3, grid3; npoints=4)
        @test p isa Plots.Plot
    end

    @testset "Vector Field Visualization (N=3)" begin
        p = visualize_vector(vf3, grid3; npoints=4, quiver_npoints=4)
        @test p isa Plots.Plot
    end

    @testset "Vorticity Visualization (N=3)" begin
        p = visualize_vorticity(vf3, grid3; npoints=4, quiver_npoints=4)
        @test p isa Plots.Plot
    end

    # Test for N=4
    bcs4 = BoundaryConditions(bc1d, bc1d, bc1d, bc1d)
    arch4 = QTTArchitecture((2, 2, 2, 2))
    grid4 = Grid(arch4, (1.0, 1.0, 1.0, 1.0))
    sf4 = ScalarField(arch4, bcs4)
    vf4 = VectorField(arch4, (bcs4, bcs4, bcs4, bcs4))
    push!(sf4, random_mps(sf4.usites))
    push!(vf4, (random_mps(vf4.usites), random_mps(vf4.usites), random_mps(vf4.usites), random_mps(vf4.usites)))

    @testset "Scalar Field Visualization (N=4)" begin
        p = visualize_scalar(sf4, grid4; npoints=4, slice_dims=(1, 2))
        @test p isa Plots.Plot
    end

    @testset "Vector Field Visualization (N=4)" begin
        p = visualize_vector(vf4, grid4; npoints=4, quiver_npoints=4, slice_dims=(2, 3))
        @test p isa Plots.Plot
    end
end

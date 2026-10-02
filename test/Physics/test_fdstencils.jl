using DifferenTT
using Test

@testset "fdstencils.jl" begin
    @testset "First Derivative Coeffs" begin
        c2 = DifferenTT.first_deriv_coeffs(2)
        @test c2[1] ≈ 0.5
        @test c2[-1] ≈ -0.5
        @test sum(values(c2)) ≈ 0.0 atol=1e-14

        c8 = DifferenTT.first_deriv_coeffs(8)
        @test c8[1] ≈ 0.8
        @test c8[2] ≈ -0.2
        @test sum(values(c8)) ≈ 0.0 atol=1e-14
        
        @test_throws ArgumentError DifferenTT.first_deriv_coeffs(4)
    end
    
    @testset "Second Derivative Coeffs" begin
        c2 = DifferenTT.second_deriv_coeffs(2)
        @test c2[0] ≈ -2.0
        @test c2[1] ≈ 1.0
        @test sum(values(c2)) ≈ 0.0 atol=1e-14

        c8 = DifferenTT.second_deriv_coeffs(8)
        @test c8[0] ≈ -205.0 / 72.0
        @test sum(values(c8)) ≈ 0.0 atol=1e-14
        
        @test_throws ArgumentError DifferenTT.second_deriv_coeffs(4)
    end
    
    @testset "Fourth Derivative Coeffs" begin
        c2 = DifferenTT.fourth_deriv_coeffs(2)
        @test c2[0] ≈ 6.0
        @test c2[1] ≈ -4.0
        @test c2[2] ≈ 1.0
        @test sum(values(c2)) ≈ 0.0 atol=1e-14

        c8 = DifferenTT.fourth_deriv_coeffs(8)
        @test c8[0] ≈ 91.0 / 8
        @test sum(values(c8)) ≈ 0.0 atol=1e-14
        
        @test_throws ArgumentError DifferenTT.fourth_deriv_coeffs(4)
    end
    
    @testset "Derivative Coeffs Wrapper" begin
        @test DifferenTT.derivative_coefficients(1; stencil_order=2) == DifferenTT.first_deriv_coeffs(2)
        @test DifferenTT.derivative_coefficients(2; stencil_order=8) == DifferenTT.second_deriv_coeffs(8)
        @test DifferenTT.derivative_coefficients(4; stencil_order=2) == DifferenTT.fourth_deriv_coeffs(2)
        
        @test_throws ArgumentError DifferenTT.derivative_coefficients(3; stencil_order=2)
    end

    @testset "Wall Vorticity Coefficients" begin
        @test DifferenTT.wall_vorticity_coefficients(:jensen) == (-4.0, 0.5, -3.0)
        @test DifferenTT.wall_vorticity_coefficients(:thom) == (-2.0, 0.0, -2.0)
        @test_throws ArgumentError DifferenTT.wall_vorticity_coefficients(:unknown)

        # Independent check on a lid (top wall) with ψ = 0 and ψ_y = U on the wall: with s the distance from the
        # wall, ψ = -U s + a s² + b s³, so the exact wall vorticity is ω = -ψ_yy = -2a. Jensen's formula is exact
        # for cubics; Thom's formula has the first-order error -2b h.
        U, a, b, h = 1.3, 0.7, -2.1, 0.05
        ψ(s) = -U * s + a * s^2 + b * s^3
        function wall_vorticity(formula)
            c1, c2, cU = DifferenTT.wall_vorticity_coefficients(formula)
            return (c1 * ψ(h) + c2 * ψ(2h)) / h^2 + cU * U / h
        end
        @test wall_vorticity(:jensen) ≈ -2a
        @test wall_vorticity(:thom) ≈ -2a - 2b * h
    end
end

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
end

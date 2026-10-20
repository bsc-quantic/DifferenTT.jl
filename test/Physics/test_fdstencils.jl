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

    @testset "Exact fractions of the eighth-order stencils" begin
        c1 = DifferenTT.first_deriv_coeffs(8)
        @test [c1[k] for k in 1:4] ≈ [4 / 5, -1 / 5, 4 / 105, -1 / 280]
        c2 = DifferenTT.second_deriv_coeffs(8)
        @test [c2[k] for k in 0:4] ≈ [-205 / 72, 8 / 5, -1 / 5, 8 / 315, -1 / 560]
        c4 = DifferenTT.fourth_deriv_coeffs(8)
        @test [c4[k] for k in 0:4] ≈ [91 / 8, -122 / 15, 169 / 60, -2 / 5, 7 / 240]
    end

    @testset "Symmetry: the first derivative is antisymmetric, the even ones symmetric" begin
        for q in (2, 8)
            c1 = DifferenTT.derivative_coefficients(1; stencil_order=q)
            c2 = DifferenTT.derivative_coefficients(2; stencil_order=q)
            c4 = DifferenTT.derivative_coefficients(4; stencil_order=q)
            for k in 1:4
                @test c1[-k] == -c1[k]
                @test c2[-k] == c2[k]
                @test c4[-k] == c4[k]
            end
            @test iszero(c1[0])
            # Every stencil spans -4:4, with zeros beyond its support
            @test sort(collect(keys(c1))) == collect(-4:4)
            @test sort(collect(keys(c4))) == collect(-4:4)
        end
    end

    @testset "Moment conditions and order of accuracy (q=$q, derivative order $m)" for q in (2, 8), m in (1, 2, 4)
        c = DifferenTT.derivative_coefficients(m; stencil_order=q)
        # The stencil is exact on polynomials up to the order it claims: Σ c_k k^j = m! δ_{jm} for j below the order.
        # The fourth derivative with q = 8 only reaches sixth order (the widest stencil inside |k| ≤ 4)
        order = (q == 8 && m == 4) ? 6 : q
        for j in 0:(order+m-1)
            moment = sum(ck * k^j for (k, ck) in c)
            @test moment ≈ (j == m ? factorial(m) : 0.0) atol = 1e-10
        end
        # The first neglected moment does not vanish for the claimed order
        j = order + m
        iseven(j - m) && @test abs(sum(ck * k^j for (k, ck) in c)) > 1e-3

        # Halving h divides the error on sin(x + 0.3) by about 2^order
        f(x) = sin(x + 0.3)
        exact = Dict(1 => x -> cos(x + 0.3), 2 => x -> -sin(x + 0.3), 4 => x -> sin(x + 0.3))[m]
        x0 = 0.4
        err(h) = abs(sum(ck * f(x0 + k * h) for (k, ck) in c) / h^m - exact(x0))
        h1 = q == 2 ? 0.2 : 0.5
        ratio = err(h1) / err(h1 / 2)
        @test ratio ≈ 2.0^order rtol = 0.25
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

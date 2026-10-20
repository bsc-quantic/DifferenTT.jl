using DifferenTT
using Test
using ITensors
using ITensorMPS
using LinearAlgebra

include(joinpath(@__DIR__, "..", "testutils.jl"))
using .TestUtils

import Base: +, *

struct MockState
    val::Float64
end

function +(args::MockState...; kwargs...)
    @assert haskey(kwargs, :cutoff) "cutoff keyword argument is missing"
    @assert kwargs[:cutoff] == 1e-8 "cutoff keyword argument is not 1e-8"
    return MockState(sum(a.val for a in args))
end

*(a::Real, b::MockState) = MockState(a * b.val)
*(a::MockState, b::Real) = MockState(a.val * b)

struct MockRHS <: AbstractRHS
    lambda::Float64
end

function (rhs::MockRHS)(u::Tuple, t, stage)
    return ntuple(i -> MockState(rhs.lambda * u[i].val), length(u))
end

@testset "Explicit Evolution Schemes" begin
    @testset "ExplicitEuler" begin
        dt = 0.1
        stepper = ExplicitEuler(time_evol_kwargs=TimeEvolArgs(0.0, 1.0, dt), cutoff=1e-8, maxdim=10)
        u0 = (MockState(1.0),)
        rhs = MockRHS(1.0) # dy/dt = y

        u1 = timestep(stepper, u0, rhs, 0.0)

        # Expected: 1.0 + 0.1 * 1.0 = 1.1
        @test isapprox(u1[1].val, 1.1, atol=1e-12)
    end

    @testset "RK4" begin
        dt = 0.1
        stepper = RK4(time_evol_kwargs=TimeEvolArgs(0.0, 1.0, dt), cutoff=1e-8, maxdim=10)
        u0 = (MockState(1.0),)
        rhs = MockRHS(1.0)

        u1 = timestep(stepper, u0, rhs, 0.0)

        # Expected manual RK4 step for dy/dt = y
        # k1 = 1.0
        # k2 = 1.0 + 0.05 * 1.0 = 1.05
        # k3 = 1.0 + 0.05 * 1.05 = 1.0525
        # k4 = 1.0 + 0.1 * 1.0525 = 1.10525
        # u1 = 1.0 + (0.1/6) * (1.0 + 2*1.05 + 2*1.0525 + 1.10525)
        expected_val = 1.0 + (0.1 / 6.0) * (1.0 + 2.0 * 1.05 + 2.0 * 1.0525 + 1.10525)

        @test isapprox(u1[1].val, expected_val, atol=1e-12)
    end
end

# Right-hand side of the linear system u_i' = rate_i(t) u_i on MPS states, which records the (t, stage) of every call
struct LinearRHS{F} <: AbstractRHS
    rates::F
    calls::Vector{Tuple{Float64,Int}}
end
LinearRHS(rates) = LinearRHS(rates, Tuple{Float64,Int}[])
function (rhs::LinearRHS)(u::Tuple, t, stage)
    push!(rhs.calls, (t, stage))
    return ntuple(i -> rhs.rates[i](t) * u[i], length(u))
end

# Integrate to time T with step dt and return the error against the exact solution, in the 2-norm of the MPS
function integration_error(stepper_type, rates, exact_factors, u0::Tuple, T, nsteps)
    dt = T / nsteps
    stepper = stepper_type(; time_evol_kwargs=TimeEvolArgs(0.0, T, nsteps), cutoff=1e-14, maxdim=32)
    rhs = LinearRHS(rates)
    u = u0
    for s in 1:nsteps
        u = timestep(stepper, u, rhs, (s - 1) * dt)
    end
    return sqrt(sum(norm(u[i] - exact_factors[i] * u0[i])^2 for i in eachindex(u)))
end

@testset "Convergence order on a linear ODE with MPS states" begin
    sites = siteinds("S=1/2", 6)
    u0 = (random_mps(rng(1), sites; linkdims=3), random_mps(rng(2), sites; linkdims=3))
    T = 1.0

    @testset "autonomous u' = λu" begin
        λ = (-1.3, 0.7)
        rates = (t -> λ[1], t -> λ[2])
        exact = (exp(λ[1] * T), exp(λ[2] * T))
        for (stepper_type, order) in ((ExplicitEuler, 1), (RK4, 4))
            n0 = order == 1 ? 40 : 8
            errors = [integration_error(stepper_type, rates, exact, u0, T, n0 * 2^k) for k in 0:2]
            # Halving dt divides the error by 2^order
            @test errors[1] / errors[2] ≈ 2.0^order rtol = 0.1
            @test errors[2] / errors[3] ≈ 2.0^order rtol = 0.1
        end
        # The scheme is exact for a constant state: u' = 0
        zero_rates = (t -> 0.0, t -> 0.0)
        for stepper_type in (ExplicitEuler, RK4)
            @test integration_error(stepper_type, zero_rates, (1.0, 1.0), u0, T, 5) < 1e-12
        end
    end

    @testset "non-autonomous u' = t u uses the right stage times" begin
        # exact: exp(t²/2). A wrong stage time lowers the order of RK4
        rates = (t -> t, t -> -2t)
        exact = (exp(T^2 / 2), exp(-T^2))
        errors = [integration_error(RK4, rates, exact, u0, T, 4 * 2^k) for k in 0:2]
        @test errors[1] / errors[2] ≈ 16.0 rtol = 0.15
        @test errors[2] / errors[3] ≈ 16.0 rtol = 0.15
        errors = [integration_error(ExplicitEuler, rates, exact, u0, T, 40 * 2^k) for k in 0:2]
        @test errors[1] / errors[2] ≈ 2.0 rtol = 0.15
    end

    @testset "stages and times of the right-hand side calls" begin
        dt = 0.1
        euler = ExplicitEuler(; time_evol_kwargs=TimeEvolArgs(0.0, 1.0, dt))
        rk4 = RK4(; time_evol_kwargs=TimeEvolArgs(0.0, 1.0, dt))
        rhs = LinearRHS((t -> 1.0,))
        timestep(euler, (u0[1],), rhs, 0.3)
        @test rhs.calls == [(0.3, 1)]
        rhs = LinearRHS((t -> 1.0,))
        timestep(rk4, (u0[1],), rhs, 0.3)
        @test first.(rhs.calls) ≈ [0.3, 0.35, 0.35, 0.4]
        @test last.(rhs.calls) == 1:4
        # The default time is 0
        rhs = LinearRHS((t -> 1.0,))
        timestep(euler, (u0[1],), rhs)
        @test rhs.calls == [(0.0, 1)]
    end

    @testset "one Euler step is u + dt f(u); one RK4 step is the Taylor polynomial of degree 4" begin
        dt, λ = 0.2, -1.1
        rhs = LinearRHS((t -> λ,))
        z = λ * dt
        euler_step = timestep(ExplicitEuler(; time_evol_kwargs=TimeEvolArgs(0.0, dt, 1), cutoff=1e-14), (u0[1],), rhs)
        rk4_step = timestep(RK4(; time_evol_kwargs=TimeEvolArgs(0.0, dt, 1), cutoff=1e-14), (u0[1],), rhs)
        @test norm(euler_step[1] - (1 + z) * u0[1]) < 1e-12
        @test norm(rk4_step[1] - (1 + z + z^2 / 2 + z^3 / 6 + z^4 / 24) * u0[1]) < 1e-12
        # The input state is not modified, and the output keeps the sites
        @test siteinds(euler_step[1]) == sites
        @test norm(u0[1]) ≈ norm(random_mps(rng(1), sites; linkdims=3))
    end

    @testset "the truncation keywords of the stepper are applied to every sum" begin
        rhs = LinearRHS((t -> -0.5, t -> 0.25))
        capped = RK4(; time_evol_kwargs=TimeEvolArgs(0.0, 0.1, 1), cutoff=1e-14, maxdim=2)
        out = timestep(capped, u0, rhs)
        @test all(maxlinkdim(m) <= 2 for m in out)
        wide = RK4(; time_evol_kwargs=TimeEvolArgs(0.0, 0.1, 1), cutoff=1e-14, maxdim=32)
        @test maxlinkdim(timestep(wide, u0, rhs)[1]) == maxlinkdim(u0[1])
    end
end

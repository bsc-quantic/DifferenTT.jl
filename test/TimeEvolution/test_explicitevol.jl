using DifferenTT
using Test

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

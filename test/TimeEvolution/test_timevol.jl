using DifferenTT
using Test

@testset "TimeEvolArgs Configuration" begin
    @testset "Constructor with nsteps" begin
        t = TimeEvolArgs(0.0, 1.0, 10)
        @test t.ti == 0.0
        @test t.tf == 1.0
        @test t.dt == 0.1
        @test t.nsteps == 10
    end

    @testset "Constructor with dt" begin
        t = TimeEvolArgs(0.0, 1.0, 0.1)
        @test t.ti == 0.0
        @test t.tf == 1.0
        @test t.dt == 0.1
        @test t.nsteps == 10
    end

    @testset "No integrator scheme field" begin
        @test !hasfield(TimeEvolArgs, :scheme)
    end

    @testset "Steppers store TimeEvolArgs" begin
        t = TimeEvolArgs(0.0, 1.0, 10)
        for stepper in
            (ExplicitEuler(time_evol_kwargs=t, cutoff=1e-8, maxdim=10), RK4(time_evol_kwargs=t, cutoff=1e-8, maxdim=10))
            @test stepper isa AbstractTimeStepper
            @test stepper.time_evol_kwargs === t
            @test stepper.tt_kwargs == (cutoff=1e-8, maxdim=10)
        end
        @test_throws UndefKeywordError RK4(cutoff=1e-8)
    end
end

@testset "TimeEvolArgs rounding" begin
    # Given dt, the number of steps is round((tf - ti)/dt), and ti, tf, dt are kept as given
    for (ti, tf, dt, nsteps) in (
        (0.0, 1.0, 0.3, 3),   # 3.33 → 3
        (0.0, 1.0, 0.35, 3),  # 2.86 → 3
        (0.0, 1.0, 0.7, 1),   # 1.43 → 1
        (0.5, 2.0, 0.25, 6),  # exact
        (0.0, 1.0, 0.4, 2),   # 2.5 rounds to even
        (-1.0, 1.0, 0.5, 4),  # negative start
        (0.0, 1.0, 3.0, 0),   # a step larger than the interval: no step
    )
        t = TimeEvolArgs(ti, tf, dt)
        @test (t.ti, t.tf, t.dt, t.nsteps) == (ti, tf, dt, nsteps)
        # the evolution loops end within dt/2 of tf
        @test abs(t.ti + t.nsteps * t.dt - t.tf) <= t.dt / 2 + 1e-12
    end

    # Given nsteps, dt = (tf - ti)/nsteps exactly covers the interval
    for (ti, tf, nsteps) in ((0.0, 1.0, 4), (0.0, 1.0, 3), (2.0, 5.0, 7), (-1.0, 1.0, 10))
        t = TimeEvolArgs(ti, tf, nsteps)
        @test t.nsteps == nsteps
        @test t.dt == (tf - ti) / nsteps
        @test t.ti + t.nsteps * t.dt ≈ tf
    end

    # The type of the last argument selects the method: an Int is the number of steps, a Float64 the step
    @test TimeEvolArgs(0.0, 1.0, 10).dt == 0.1 && TimeEvolArgs(0.0, 1.0, 10).nsteps == 10
    @test TimeEvolArgs(0.0, 1.0, 10.0).dt == 10.0 && TimeEvolArgs(0.0, 1.0, 10.0).nsteps == 0
    # The times must be Float64, and nothing else is converted
    @test_throws MethodError TimeEvolArgs(0, 1, 10)
    @test_throws MethodError TimeEvolArgs(0.0, 1, 0.1)

    # The inner constructor stores any four values as given (it does not check their consistency)
    t = TimeEvolArgs(0.0, 1.0, 0.3, 7)
    @test (t.dt, t.nsteps) == (0.3, 7)

    # The default show is the constructor call, as in the docstring (qualified with the module unless it is imported)
    @test endswith(sprint(show, TimeEvolArgs(0.0, 1.0, 4)), "TimeEvolArgs(0.0, 1.0, 0.25, 4)")
    @test endswith(sprint(show, TimeEvolArgs(0.0, 1.0, 0.3)), "TimeEvolArgs(0.0, 1.0, 0.3, 3)")

    # A stepper without truncation keywords stores an empty NamedTuple
    for stepper_type in (ExplicitEuler, RK4)
        stepper = stepper_type(; time_evol_kwargs=TimeEvolArgs(0.0, 1.0, 10))
        @test stepper.tt_kwargs == NamedTuple()
        @test stepper.time_evol_kwargs.nsteps == 10
        @test stepper isa AbstractExplicitStepper
    end
end

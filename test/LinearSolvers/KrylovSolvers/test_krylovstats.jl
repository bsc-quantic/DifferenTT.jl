using Test
using DifferenTT

@testset "KrylovSweepStats and KrylovCondStats" begin
    # KrylovSweepStats
    stats = DifferenTT.KrylovSweepStats()
    DifferenTT.record_krylov_numiter!(stats, 0, 10)
    DifferenTT.record_krylov_numiter!(stats, 1, 12)
    DifferenTT.record_krylov_numiter!(stats, 1, 11) # return sweep
    DifferenTT.record_krylov_numiter!(stats, 0, 9)

    # It should correctly record
    numiters = DifferenTT.krylov_numiters(stats)
    @test length(numiters) == 2 # positions 1 and 2

    sums = DifferenTT.positions_sum(stats)
    @test sums[1] == 10 + 9
    @test sums[2] == 12 + 11

    # KrylovCondStats
    cstats = DifferenTT.KrylovCondStats()
    DifferenTT.record_krylov_cond!(cstats, 0, 1.5)
    DifferenTT.record_krylov_cond!(cstats, 1, 2.0)
    DifferenTT.record_krylov_cond!(cstats, 1, 2.5)
    DifferenTT.record_krylov_cond!(cstats, 0, 1.0)

    conds = DifferenTT.krylov_conds(cstats)
    @test length(conds) == 2

    c_means = DifferenTT.positions_mean_cond(cstats)
    @test c_means[1] ≈ (1.5 + 1.0) / 2
    @test c_means[2] ≈ (2.0 + 2.5) / 2
end

using ITensors
using ITensorMPS
using DifferenTT: positions_sum
include(joinpath(@__DIR__, "..", "..", "testutils.jl"))
using .TestUtils

@testset "positions_mean, positions_sum and empty positions" begin
    stats = DifferenTT.KrylovSweepStats()
    DifferenTT.record_krylov_numiter!(stats, 0, 10)
    DifferenTT.record_krylov_numiter!(stats, 2, 4) # position 2 is never visited
    DifferenTT.record_krylov_numiter!(stats, 2, 6)
    @test DifferenTT.positions_mean(stats) == [10.0, 0.0, 5.0]
    @test DifferenTT.positions_sum(stats) == [10.0, 0.0, 10.0]
    @test DifferenTT.krylov_numiters(stats) == [[[10]], [], [[4, 6]]]
    @test DifferenTT.positions_mean_cond(DifferenTT.KrylovCondStats()) == Float64[]
end

# The layout `position → sweep → visits` of KrylovSweepStats, as documented: positions 1:(N - nsite + 1), and two visits
# per position and sweep (the forward and the reverse half-sweep)
@testset "Recorded layout of real solves" begin
    A, sites, _, _ = qtt_laplacian((5,))
    N = length(sites)
    b = random_mps(rng(1), sites; linkdims=2)
    nsweeps = 3
    local_args(store) = LocalUpdaterArgs(; ishermitian=true, isposdef=true, numiter_store=store)
    @testset "$name" for (name, nsite, make) in (
        (
            "ALS, nsite = 2",
            2,
            st -> ALS(; sweep_kwargs=SweepingArgs(; nsweeps, nsite=2), updater_kwargs=local_args(st)),
        ),
        (
            "ALS, nsite = 1",
            1,
            st -> ALS(; sweep_kwargs=SweepingArgs(; nsweeps, nsite=1), updater_kwargs=local_args(st)),
        ),
        ("ResidualALS", 2, st -> ResidualALS(; sweep_kwargs=SweepingArgs(; nsweeps), updater_kwargs=local_args(st))),
        (
            "AMEn",
            1,
            st -> AMEn(; sweep_kwargs=SweepingArgs(; nsweeps), updater_kwargs=local_args(st), local_stop=false),
        ),
    )
        store = KrylovSweepStats()
        linearsolve(make(store), A, b, nothing)
        numiters = DifferenTT.krylov_numiters(store)
        @test length(numiters) == N - nsite + 1
        @test all(length(p) == nsweeps for p in numiters)
        @test all(length(s) == 2 for p in numiters for s in p)
        # A local problem that the current tensor already solves takes 0 iterations
        @test all(n >= 0 for p in numiters for s in p for n in s)
        @test sum(positions_sum(store)) > 0

        # A store reused by a second solve numbers its sweeps on
        linearsolve(make(store), A, b, nothing)
        @test all(length(p) == 2nsweeps for p in DifferenTT.krylov_numiters(store))
    end

    # The same layout for the condition numbers, aligned with the iteration counts
    numiter_store, cond_store = KrylovSweepStats(), KrylovCondStats()
    solver = ALS(;
        sweep_kwargs=SweepingArgs(; nsweeps=2),
        updater_kwargs=LocalUpdaterArgs(; ishermitian=true, isposdef=true, numiter_store, cond_store),
    )
    linearsolve(solver, A, b, nothing)
    conds = DifferenTT.krylov_conds(cond_store)
    @test length.(conds) == length.(DifferenTT.krylov_numiters(numiter_store))
    @test all(κ >= 1 for p in conds for s in p for κ in s)
    @test length(positions_mean_cond(cond_store)) == N - 1

    # A single position (N = nsite) never moves, so all its solves are recorded under sweep 1
    single = KrylovSweepStats()
    for _ in 1:3
        DifferenTT.record_krylov_numiter!(single, 0, 1)
    end
    @test DifferenTT.krylov_numiters(single) == [[[1, 1, 1]]]
end

@testset "Non-finite condition numbers are not recorded" begin
    A, sites, _, _ = qtt_laplacian((4,))
    problem = ITensorMPS.ReducedLinearProblem(A, random_mps(rng(1), sites))
    store = KrylovCondStats()
    DifferenTT._record_cond!(store, problem, Inf)
    @test isempty(DifferenTT.krylov_conds(store))
    DifferenTT._record_cond!(store, problem, 2.0)
    @test DifferenTT.krylov_conds(store) == [[[2.0]]]
    @test isnothing(DifferenTT._record_cond!(nothing, problem, 2.0))
end

@testset "KrylovSpectrumStats and NotSPDError" begin
    @test spectrum_table(KrylovSpectrumStats()) == Dict{String,Vector}()
    @test KrylovSpectrumStats().max_blockdim == 4096

    e = NotSPDError("Preconditioned operator P⁻¹A", 3, -0.5, 2.0)
    msg = sprint(showerror, e)
    @test occursin("NotSPDError", msg)
    @test occursin("Preconditioned operator P⁻¹A", msg)
    @test occursin("lpos = 3", msg)
    @test occursin("min Re λ / max|λ| = -0.25", msg)
end

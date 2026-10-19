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

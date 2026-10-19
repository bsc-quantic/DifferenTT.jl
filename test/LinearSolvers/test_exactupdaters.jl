using DifferenTT
using Test
using ITensors, ITensorMPS
using DifferenTT: QRUpdater, ALS

@testset "Exact Updaters" begin
    n = 4
    s = siteinds("S=1/2", n)
    A = random_mpo(s)
    b = random_mps(s)

    updater = QRUpdater()
    solver = ALS(; updater=updater, sweep_kwargs=DifferenTT.SweepingArgs(; nsweeps=1))

    x = linearsolve(solver, A, b, nothing)
    @test x isa MPS
    @test length(x) == n
end

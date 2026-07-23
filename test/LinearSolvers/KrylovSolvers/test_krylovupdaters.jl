using Test
using ITensors
using ITensorMPS
using KrylovKit
using DifferenTT

@testset "KrylovUpdater instantiation" begin
    updater = DifferenTT.KrylovUpdater()
    @test updater isa DifferenTT.AbstractUpdater
end

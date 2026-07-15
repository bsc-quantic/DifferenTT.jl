using DifferenTT
using Test
using ITensors: ITensors, siteinds
using ITensorMPS: MPS
using DifferenTT: ExpandedALS, NullExpansion, RandomSPSDexpansion, RandomLTexpansion, SweepingArgs, LocalUpdaterArgs, KrylovUpdater, QRUpdater

@testset "ExpandedALS functionality" begin
    @testset "ExpandedALS creation" begin
        sweep_kwargs_1 = SweepingArgs(nsite=1)
        updater_kwargs = LocalUpdaterArgs()
        
        # Test creation with correct nsite
        solver = ExpandedALS(
            sweep_kwargs_1, 
            updater_kwargs, 
            NullExpansion(), 
            KrylovUpdater()
        )
        @test solver isa DifferenTT.ExpandedALS
        
        # Test creation throws if nsite != 1
        sweep_kwargs_2 = SweepingArgs(nsite=2)
        @test_throws ArgumentError ExpandedALS(
            sweep_kwargs_2, 
            updater_kwargs, 
            NullExpansion(), 
            KrylovUpdater()
        )
    end

    @testset "Expansions" begin
        # Create dummy sites
        sites = siteinds(4, 3)
        
        # Test NullExpansion
        null_exp = NullExpansion()
        @test null_exp(sites) === nothing

        # Testing RandomSPSDexpansion and RandomLTexpansion requires 
        # random_mpo and lower_triang_mpo from DifferenTT which should be available
        spsd_exp = RandomSPSDexpansion()
        spsd_mps = spsd_exp(sites)
        @test spsd_mps isa MPS
        @test length(spsd_mps) == 3

        lt_exp = RandomLTexpansion()
        lt_mps = lt_exp(sites)
        @test lt_mps isa MPS
        @test length(lt_mps) == 3
    end
end

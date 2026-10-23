using Aqua
using DifferenTT
using Test

@testset "Aqua" begin
    Aqua.test_all(
        DifferenTT;
        persistent_tasks=false,       # slow (spawns a precompilation); candidate for an optional nightly job
    )
    # Aqua checks the exported names only; names(DifferenTT) also lists the `public` ones
    @test all(n -> isdefined(DifferenTT, n), names(DifferenTT))
    Aqua.test_undocumented_names(DifferenTT)
end

using Aqua
using DifferenTT
using Test

@testset "Aqua" begin
    Aqua.test_all(
        DifferenTT;
        piracies=(broken=true,),      # WP2 removes the type piracy and makes this strict
        persistent_tasks=false,       # slow (spawns a precompilation); candidate for an optional nightly job
    )
    # WP6: Aqua.test_undocumented_names(DifferenTT)
end

using Aqua
using DifferenTT
using Test

@testset "Aqua" begin
    Aqua.test_all(
        DifferenTT;
        persistent_tasks=false,       # slow (spawns a precompilation); candidate for an optional nightly job
    )
    # WP6: Aqua.test_undocumented_names(DifferenTT)
end

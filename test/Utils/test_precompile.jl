using Test
using DifferenTT
using Preferences: load_preference

# The switch writes the preference that PrecompileTools reads when the package is precompiled. Both cases write to
# temporary environments, so the test environment and the default environment of the user are left untouched.
@testset "set_precompile_workload" begin
    @testset "active environment" begin
        mktempdir() do dir
            active = Base.active_project()
            Base.set_active_project(joinpath(dir, "Project.toml"))
            try
                @test DifferenTT.set_precompile_workload(false) === false
                @test isfile(joinpath(dir, "LocalPreferences.toml"))
                @test load_preference(DifferenTT, "precompile_workload") === false
                @test DifferenTT.set_precompile_workload(true) === true
                @test load_preference(DifferenTT, "precompile_workload") === true
            finally
                Base.set_active_project(active)
            end
        end
    end

    @testset "default environment" begin
        mktempdir() do depot
            depots = copy(DEPOT_PATH)
            empty!(DEPOT_PATH)
            push!(DEPOT_PATH, depot)
            try
                DifferenTT.set_precompile_workload(false; default_environment=true)
                env = joinpath(depot, "environments", "v$(VERSION.major).$(VERSION.minor)")
                @test isfile(joinpath(env, "LocalPreferences.toml"))
                # The active environment sets nothing, so the setting is inherited through the load path
                @test load_preference(DifferenTT, "precompile_workload") === false
            finally
                append!(empty!(DEPOT_PATH), depots)
            end
        end
    end
end

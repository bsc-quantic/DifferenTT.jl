using Test
using DifferenTT

# Without Plots, calling a visualize_* function is a MethodError with a hint to load Plots. The hint depends on whether the
# extension is loaded, which is global state of the process, so it is checked in a fresh process that has not loaded Plots.
@testset "MethodError hint of visualize_*" begin
    project = dirname(Base.active_project())
    script = """
    using DifferenTT
    for f in (visualize_scalar, visualize_vector, visualize_vorticity)
        try
            f(1)
        catch err
            showerror(stdout, err)
            println()
        end
    end
    println("EXTENSION LOADED: ", !isnothing(Base.get_extension(DifferenTT, :DifferenTTPlotsExt)))
    """
    out = read(`$(Base.julia_cmd()) --startup-file=no --project=$project -e $script`, String)
    for name in ("visualize_scalar", "visualize_vector", "visualize_vorticity")
        @test occursin("`$name` is defined in the Plots extension of DifferenTT; load it with `using Plots`.", out)
    end
    @test occursin("EXTENSION LOADED: false", out)
end

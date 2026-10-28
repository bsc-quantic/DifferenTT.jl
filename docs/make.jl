using DifferenTT, Documenter, Literate
using Plots  # loads the plotting extension, so that the `visualize_*` methods appear in the API reference

# Headless plotting in the tutorials and in CI
get!(ENV, "GKSwstype", "100")

DocMeta.setdocmeta!(DifferenTT, :DocTestSetup, :(using DifferenTT, ITensors, ITensorMPS); recursive=true)

# Tutorials. `examples/*.jl` are Literate scripts owned by the examples; they are rendered into `docs/src/tutorials/`
# (ignored by git) and executed by Documenter during the build. A missing or empty folder gives an empty section.
examples_dir = joinpath(@__DIR__, "..", "examples")
tutorials_dir = joinpath(@__DIR__, "src", "tutorials")

# The tutorials are written without a `CurrentModule` block, so that they also run as plain scripts. Prepending the block
# makes their `[`name`](@ref)` links resolve to the exported and `public` names of DifferenTT, as in the manual pages.
with_current_module(content) = "```@meta\nCurrentModule = DifferenTT\n```\n\n" * content

function render(dir, outdir, prefix)
    pages = String[]
    isdir(dir) || return pages
    for f in sort(filter(endswith(".jl"), readdir(dir)))
        Literate.markdown(joinpath(dir, f), outdir; documenter=true, credit=false, postprocess=with_current_module)
        push!(pages, joinpath(prefix, replace(f, ".jl" => ".md")))
    end
    return pages
end

# The two boundary-condition tutorials define the anchors `bc_velocity_pressure` and `bc_vorticity_streamfunction`, which
# the docstrings of every boundary-condition function link to.
bc_pages = render(
    joinpath(examples_dir, "boundary_conditions"),
    joinpath(tutorials_dir, "boundary_conditions"),
    joinpath("tutorials", "boundary_conditions"),
)

tutorials = Any["Overview"=>"tutorials.md"]
append!(tutorials, render(examples_dir, tutorials_dir, "tutorials"))
push!(tutorials, "Boundary conditions" => bc_pages)

makedocs(;
    modules=[DifferenTT],
    sitename="DifferenTT.jl",
    authors="Pau Torrente Badia, Stefano Carignano, Artur García Sáez, Daniel Pérez García",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        edit_link="master",
        canonical="https://bsc-quantic.github.io/DifferenTT.jl",
        # The API pages list every docstring of an area, and the internals page all of the others
        size_threshold_warn=300 * 2^10,
        size_threshold=600 * 2^10,
        # The search index covers the tutorials too (680 KiB at 4b)
        search_size_threshold_warn=1024 * 2^10,
    ),
    checkdocs=:public,
    doctest=true,
    pages=[
        "Home" => "index.md",
        "Getting started" => "getting_started.md",
        "Manual" => [
            "Architectures and conventions" => "manual/architectures.md",
            "Grids, fields and boundary conditions" => "manual/fields_bcs.md",
            "Finite-difference operators" => "manual/operators.md",
            "Tensor-train utilities" => "manual/tt_utilities.md",
            "Linear solvers" => "manual/linear_solvers.md",
            "Preconditioners (FAI, sinc, …)" => "manual/preconditioners.md",
            "Navier–Stokes" => "manual/navier_stokes.md",
            "GPU (experimental)" => "manual/gpu.md",
        ],
        "Tutorials" => tutorials,
        "API reference" => [
            "Tensor trains" => "api/tensortrains.md",
            "PDE problem" => "api/pdeproblem.md",
            "Physics" => "api/physics.md",
            "Time evolution" => "api/timeevolution.md",
            "Linear solvers" => "api/linearsolvers.md",
            "Utilities" => "api/utils.md",
            "Internals" => "api/internals.md",
        ],
        "Reproducing the paper" => "paper.md",
    ],
)

deploydocs(; repo="github.com/bsc-quantic/DifferenTT.jl.git", devbranch="master", push_preview=true)

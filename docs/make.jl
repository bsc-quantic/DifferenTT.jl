using DifferenTT, Documenter, Literate
using Plots  # loads the plotting extension, so that the `visualize_*` methods appear in the API reference

# Cross-references written as [`name`](@ref) outside a `CurrentModule` block (such as `manual/preconditioners.md`) are
# looked up in `Main`. `using DifferenTT` brings in the exported names, and this brings in the `public` ones.
for name in names(DifferenTT)
    isdefined(Main, name) || Core.eval(Main, :(import DifferenTT: $name))
end

# Headless plotting in the tutorials and in CI
get!(ENV, "GKSwstype", "100")

DocMeta.setdocmeta!(DifferenTT, :DocTestSetup, :(using DifferenTT, ITensors, ITensorMPS); recursive=true)

# Tutorials. `examples/*.jl` are Literate scripts owned by the examples; they are rendered into `docs/src/tutorials/`
# (ignored by git) and executed by Documenter during the build. A missing or empty folder gives an empty section.
examples_dir = joinpath(@__DIR__, "..", "examples")
tutorials_dir = joinpath(@__DIR__, "src", "tutorials")

function render(dir, outdir, prefix)
    pages = String[]
    isdir(dir) || return pages
    for f in sort(filter(endswith(".jl"), readdir(dir)))
        Literate.markdown(joinpath(dir, f), outdir; documenter=true, credit=false)
        push!(pages, joinpath(prefix, replace(f, ".jl" => ".md")))
    end
    return pages
end

# TODO(4b): remove this function when both boundary-condition tutorials are merged. The docstrings of every
# boundary-condition function link to the anchors below, so each one must exist; a rendered tutorial defines its own
# anchor, and the placeholder is written only while the tutorial is missing.
function placeholder(name, title, anchor, outdir, prefix)
    file = name * ".md"
    isfile(joinpath(outdir, file)) && return joinpath(prefix, file)
    mkpath(outdir)
    write(
        joinpath(outdir, file),
        """
# [$title](@id $anchor)

This tutorial is not available in this build of the documentation.
""",
    )
    return joinpath(prefix, file)
end

bc_examples_dir = joinpath(examples_dir, "boundary_conditions")
bc_outdir = joinpath(tutorials_dir, "boundary_conditions")
bc_prefix = joinpath("tutorials", "boundary_conditions")
bc_pages = render(bc_examples_dir, bc_outdir, bc_prefix)
for (name, title, anchor) in (
    ("01_velocity_pressure", "Velocity–pressure formulation", "bc_velocity_pressure"),
    ("02_vorticity_streamfunction", "Vorticity–streamfunction formulation", "bc_vorticity_streamfunction"),
)
    page = joinpath(bc_prefix, name * ".md")
    page in bc_pages || push!(bc_pages, placeholder(name, title, anchor, bc_outdir, bc_prefix))
end
sort!(bc_pages)

tutorials = Any["Overview"=>"tutorials.md"]
append!(tutorials, render(examples_dir, tutorials_dir, "tutorials"))
push!(tutorials, "Boundary conditions" => bc_pages)

makedocs(;
    modules=[DifferenTT],
    sitename="DifferenTT.jl",
    authors="Pau Torrente, Stefano Carignano, Artur Garcia",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        edit_link="master",
        canonical="https://bsc-quantic.github.io/DifferenTT.jl",
        # The API pages list every docstring of an area, and the internals page all of the others
        size_threshold_warn=300 * 2^10,
        size_threshold=600 * 2^10,
    ),
    checkdocs=:exports,  # WP6: :public
    warnonly=[:missing_docs],  # WP6: remove
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

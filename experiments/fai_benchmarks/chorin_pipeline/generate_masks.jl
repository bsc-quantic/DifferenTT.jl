"""
    Cylinder masks of the Chorin pipeline

The Chorin benchmarks impose the obstacle through a mask that multiplies the velocity at every step, and the mask must
be the same in every task so that the right-hand sides are reproducible. We therefore precompute it once. The obstacle
is a cylinder of radius 0.1 centered at (0.5, 0.5) in the box [0, 2] × [0, 1], smoothed as ½[1 + tanh(700 ((x-0.5)² +
(y-0.5)² - 0.1²))], and we compress it into an MPS by cross interpolation (tolerance 1e-10, bond dimension at most 50)
for the grids 2⁸ × 2⁸, 2¹⁰ × 2¹⁰ and 2¹² × 2¹² (keys `mask_8_8`, `mask_10_10`, `mask_12_12`).

# Inputs and outputs

No input. Output: `masks.jld2` and a picture of each mask in `maskplots/`, both in this folder (the committed copies are
overwritten). The script prints the bond dimensions and the error of every mask.

In smoke mode (`DIFFERENTT_SMOKE=1`) it generates only the 2⁸ × 2⁸ mask, into `DIFFERENTT_SMOKE_DIR` (a fresh temporary
folder if it is unset), and the smoke worker reads it from there.

# How to run

From the root of the repository:

    julia --project=experiments experiments/fai_benchmarks/chorin_pipeline/generate_masks.jl

# Resources

There is no job script: we run it locally.

# Paper

None.
"""

using ITensors
using ITensorMPS
using TensorCrossInterpolation
using LinearAlgebra
using JLD2
using DifferenTT

include(joinpath(@__DIR__, "..", "..", "common", "smoke.jl"))
# Smoke mode writes under DIFFERENTT_SMOKE_DIR (a fresh temporary folder if unset) instead of this folder.
const OUT_ROOT = output_root(@__DIR__; fresh=true)

box_size = (2.0, 1.0)
system_sizes = SMOKE ? [(8, 8)] : [(8, 8), (10, 10), (12, 12)]
mask_path = joinpath(OUT_ROOT, "masks.jld2")

maskplots_subfolder = joinpath(OUT_ROOT, "maskplots")
mkpath(maskplots_subfolder)

function circle(x, y; bx=0.5, by=0.5, rad=0.1)
    value = (x - bx)^2 + (y - by)^2 - rad^2
    return 0.5 * (1 + tanh(700 * value))
end

mask_func(x) = circle(x...)

ITensors.disable_warn_order()

jldopen(mask_path, "w") do file
    for nodes in system_sizes
        @info "Generating mask for system size $nodes"

        sites = siteinds(2, sum(nodes))
        arch = QTTArchitecture(nodes; interleaved=false)
        grid = Grid(arch, box_size)

        initialpivot = optfirstpivot(Float64, mask_func, grid, arch)

        tci_mask, ranks_mask, error_mask = crossinterpolate2(
            Float64, mask_func, grid, arch; tolerance=1e-10, maxbonddim=50, initialpivots=[initialpivot]
        )

        @show ranks_mask
        @show error_mask

        object_mask = replace_siteinds!(MPS(tci_mask), sites)
        visualize_scalar(
            object_mask,
            grid,
            arch,
            joinpath(maskplots_subfolder, "mask_$(nodes[1])_$(nodes[2])_buffer.png");
            npoints=200,
            aspect_ratio=:equal,
            size=(1200, 600),
        )

        key = "mask_$(nodes[1])_$(nodes[2])"
        file[key] = object_mask
        @info "  Mask '$key' saved ($(length(object_mask)) sites)"
    end
end

@info "All masks saved to $mask_path"

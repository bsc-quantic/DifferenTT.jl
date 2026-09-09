"""
    Generate masks for chorin pipeline

    Creates and saves circle obstacle masks for system sizes (8,8), (10,10), (12,12)
    into a single JLD2 file for consistent, reproducible RHS construction.
"""

using ITensors
using ITensorMPS
using TensorCrossInterpolation
using LinearAlgebra
using JLD2
using DifferenTT

box_size = (2.0, 1.0)
system_sizes = [(8, 8), (10, 10), (12, 12)]
mask_path = "experiments/preconditioner_benchmarks/chorin_pipeline_local_precond/masks.jld2"

maskplots_subfolder = "experiments/preconditioner_benchmarks/chorin_pipeline_local_precond/maskplots"
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
            Float64, mask_func, grid, arch;
            tolerance=1e-10, maxdim=50, initialpivots=[initialpivot]
        )

        @show ranks_mask
        @show error_mask

        object_mask = replace_siteinds!(MPS(tci_mask), sites)
        visualize_scalar(object_mask, grid, arch, joinpath(maskplots_subfolder, "mask_$(nodes[1])_$(nodes[2])_buffer.png"); npoints=200, aspect_ratio=:equal, size=(1200,600))

        key = "mask_$(nodes[1])_$(nodes[2])"
        file[key] = object_mask
        @info "  Mask '$key' saved ($(length(object_mask)) sites)"
    end
end

@info "All masks saved to $mask_path"

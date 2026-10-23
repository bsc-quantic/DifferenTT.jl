# Safeguard against a stale cylinder mask of the Chorin benchmarks.
#
# The masks of `fai_benchmarks/chorin_pipeline/masks.jld2` store the node values of the cylinder indicator, sampled on one
# grid. The same mask on another grid (another box, another number of nodes, or a periodic axis, whose nodes are at
# y_j = j L/n instead of j L/(n-1)) is a different obstacle, and nothing in the file says so. The generator therefore
# stores the grid next to every mask, under the key `"<mask key>_grid"`, and every worker compares it with the grid it
# built before it uses the mask:
#
#     file["mask_8_8_grid"] = mask_grid_metadata((8, 8), (2.0, 1.0), (false, true))                 # generator
#     check_mask_grid(load_mask_grid(MASK_FILE, "mask_8_8"), nodes, box_size, periodic; file=MASK_FILE, key="mask_8_8")
#
# `box_size` is the physical box `L` that is passed to `Grid`, not the span `L - h` of a periodic axis.

using JLD2

"""
    mask_grid_metadata(nodes, box_size, periodic)

The description of the grid that a mask was sampled on, as a `NamedTuple` that JLD2 can store.
"""
function mask_grid_metadata(nodes, box_size, periodic)
    return (; nodes=Tuple(Int.(nodes)), box_size=Tuple(Float64.(box_size)), periodic=Tuple(Bool.(periodic)))
end

"""
    load_mask_grid(file, key)

The metadata stored next to the mask `key` of `file`, or `nothing` if the file has none (every file written before the
metadata existed).
"""
function load_mask_grid(file::AbstractString, key::AbstractString)
    return jldopen(file, "r") do f
        haskey(f, "$(key)_grid") ? f["$(key)_grid"] : nothing
    end
end

"""
    check_mask_grid(meta, nodes, box_size, periodic; file, key)

Throw an `ArgumentError` unless the metadata `meta` of a mask is that of the grid with `nodes`, `box_size` and `periodic`:
`nodes` and `periodic` must be equal and `box_size` must agree to a relative tolerance of 1e-12. `meta` is `nothing` for
a mask without metadata, which fails the same way. The message names the mask (`file`, `key`), both descriptions and
the fix.
"""
function check_mask_grid(meta, nodes, box_size, periodic; file, key)
    expected = mask_grid_metadata(nodes, box_size, periodic)
    ok =
        meta !== nothing &&
        all(k -> hasproperty(meta, k), (:nodes, :box_size, :periodic)) &&
        Tuple(meta.nodes) == expected.nodes &&
        Tuple(meta.periodic) == expected.periodic &&
        length(meta.box_size) == length(expected.box_size) &&
        all(isapprox.(Tuple(meta.box_size), expected.box_size; rtol=1e-12))
    ok && return nothing
    found = meta === nothing ? "no grid metadata" : "grid $(meta)"
    throw(
        ArgumentError(
            "The mask '$key' of $file does not belong to the grid of this run: it has $found, the run uses grid " *
            "$(expected). Regenerate the masks with `generate_masks.jl` (experiments/fai_benchmarks/chorin_pipeline/) " *
            "on the same grid.",
        ),
    )
end

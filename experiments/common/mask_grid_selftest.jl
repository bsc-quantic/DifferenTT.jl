"""
    Self-test of the mask safeguard (`common/mask_grid.jl`)

Writes a file with the metadata of one mask, reads it back and checks that `check_mask_grid` accepts the grid it was
written for and throws an `ArgumentError` for a mask without metadata, a mismatch of the periodic flags, of the number of
nodes and of the box, and for a box that differs by a rounding error only (accepted). It needs no DifferenTT. Run by
`experiments/smoke.jl`:

    julia --project=experiments experiments/common/mask_grid_selftest.jl
"""

using JLD2

include(joinpath(@__DIR__, "mask_grid.jl"))

path = joinpath(mktempdir(), "masks.jld2")
jldopen(path, "w") do f
    f["mask_8_8"] = [1.0]
    f["mask_8_8_grid"] = mask_grid_metadata((8, 8), (2.0, 1.0), (false, true))
    f["mask_10_10"] = [1.0]   # no metadata, like every file written before the safeguard
end

check(key, nodes, box, periodic) = check_mask_grid(load_mask_grid(path, key), nodes, box, periodic; file=path, key=key)

check("mask_8_8", (8, 8), (2.0, 1.0), (false, true))                       # the grid it was written for
check("mask_8_8", (8, 8), (2.0 * (1 + 1e-14), 1.0), (false, true))         # rounding error in the box: accepted

cases = [
    ("no metadata", "mask_10_10", (10, 10), (2.0, 1.0), (false, true)),
    ("periodic flag", "mask_8_8", (8, 8), (2.0, 1.0), (false, false)),
    ("nodes", "mask_8_8", (8, 10), (2.0, 1.0), (false, true)),
    ("box", "mask_8_8", (8, 8), (2.0, 1.0 - 1.0 / 8), (false, true)),
]
for (label, key, nodes, box, periodic) in cases
    err = try
        check(key, nodes, box, periodic)
        nothing
    catch e
        e
    end
    err isa ArgumentError || error("check_mask_grid did not throw an ArgumentError for $label (got $err)")
    occursin("Regenerate the masks", err.msg) || error("the message for $label does not give the fix: $(err.msg)")
    println("rejected ($label): ", err.msg)
end
println("mask_grid_selftest: ok")

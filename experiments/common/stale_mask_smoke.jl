"""
    Stale mask for the smoke test of the mask safeguard

Copies the smoke mask `masks.jld2` of `DIFFERENTT_SMOKE_DIR` to `masks_stale.jld2` and changes the metadata of every mask
to a grid with no periodic axis, as if the file had been sampled on the plain grid of the old workers. The smoke test
(`experiments/smoke.jl`) then runs a Chorin worker on that file (`CHORIN_MASK_FILE`) and requires that it stops with
an `ArgumentError` (`common/mask_grid.jl`).

    DIFFERENTT_SMOKE_DIR=<folder with masks.jld2> julia --project=experiments experiments/common/stale_mask_smoke.jl
"""

using JLD2
using ITensors   # JLD2 needs the types of the stored masks (MPS) loaded to read them
using ITensorMPS

include(joinpath(@__DIR__, "mask_grid.jl"))

dir = get(ENV, "DIFFERENTT_SMOKE_DIR", "")
isfile(joinpath(dir, "masks.jld2")) || error("set DIFFERENTT_SMOKE_DIR to the folder that the smoke generator wrote")

jldopen(joinpath(dir, "masks_stale.jld2"), "w") do out
    jldopen(joinpath(dir, "masks.jld2"), "r") do masks
        for key in keys(masks)
            endswith(key, "_grid") && continue
            out[key] = masks[key]
            meta = masks["$(key)_grid"]
            out["$(key)_grid"] = mask_grid_metadata(meta.nodes, meta.box_size, (false, false))
        end
    end
end
@info "Wrote masks_stale.jld2 (metadata: no periodic axis)" dir

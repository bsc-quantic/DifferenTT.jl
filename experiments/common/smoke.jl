# Smoke mode of the experiment scripts.
#
# `DIFFERENTT_SMOKE=1` makes a script run on a tiny problem (a configuration branch next to its constants) and write
# to a temporary folder, so that it finishes quickly and never touches the committed results. `experiments/smoke.jl`
# runs the worker, the collector and the plotter of each benchmark in turn and gives them one shared folder through
# `DIFFERENTT_SMOKE_DIR`.

const SMOKE = get(ENV, "DIFFERENTT_SMOKE", "0") == "1"

"""
    output_root(default; fresh=false)

Folder that holds the `results*/` and `plots/` folders of a script: `default` outside smoke mode. In smoke mode it is
`DIFFERENTT_SMOKE_DIR`. If that variable is unset, a worker (`fresh=true`) starts in a new temporary folder and logs
it, whereas a collector or a plotter throws, since it would otherwise read nothing or the committed results.
"""
function output_root(default::AbstractString; fresh::Bool=false)
    SMOKE || return default
    if haskey(ENV, "DIFFERENTT_SMOKE_DIR")
        return mkpath(ENV["DIFFERENTT_SMOKE_DIR"])
    elseif fresh
        root = mktempdir()
        @info "Smoke mode: writing to a temporary folder" root
        return root
    end
    return error("smoke mode: set DIFFERENTT_SMOKE_DIR to the folder that the smoke worker wrote")
end

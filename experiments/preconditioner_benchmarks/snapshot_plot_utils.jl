"""
Last-snapshot plots shared by the Chorin benchmark plotters.

The Chorin workers store the final fluid state of a run (the stats workers: of repeat 1 only) under
`"last_snapshot"` in their task files (`grid`, `u_fields`, `p_field`, `n_steps`). `plot_last_snapshots` draws, for every
configuration, the velocity, vorticity and pressure of that state into
`<plots_dir>/<size>/last_snapshots/`, like `experiments/inse/chorin_worker_8x8/plot_snapshots.jl`
does for its intermediate snapshots.
"""

using ITensors
using ITensorMPS
using DifferenTT
using JLD2
using Plots

const SNAPSHOT_FIG_SIZE = (1200, 600)

# task_<id>.jld2 (chorin_pipeline) or task_<id>_rep_<r>.jld2 (chorin_pipeline_stats, chorin_pipeline_globalcg_stats)
const SNAPSHOT_TASK_FILE_RE = r"^task_(\d+)(?:_rep_(\d+))?\.jld2$"

"""
    snapshot_task_files(tasks_dir) -> Vector{String}

One task file per task id. When a task has several repeats, the one with the lowest repeat index is
taken: all repeats solve the same problem, so their final states only differ by the solver
tolerance, and the stats workers store the snapshot only for repeat 1 (`SNAPSHOT_REPEAT`).
"""
function snapshot_task_files(tasks_dir::AbstractString)
    chosen = Dict{Int,Tuple{Int,String}}()
    for file in readdir(tasks_dir)
        m = match(SNAPSHOT_TASK_FILE_RE, file)
        isnothing(m) && continue
        task_id = parse(Int, m[1])
        rep = isnothing(m[2]) ? 0 : parse(Int, m[2])
        if !haskey(chosen, task_id) || rep < first(chosen[task_id])
            chosen[task_id] = (rep, file)
        end
    end
    return [joinpath(tasks_dir, last(chosen[id])) for id in sort(collect(keys(chosen)))]
end

function plot_last_snapshot(snapshot, out_dir::AbstractString, name::AbstractString)
    grid = snapshot["grid"]
    u_fields = snapshot["u_fields"]
    p_field = snapshot["p_field"]

    visualize_vector(
        u_fields,
        length(u_fields),
        grid,
        joinpath(out_dir, "$(name)_vector.svg");
        npoints=256,
        quiver_npoints=16,
        arrow_scale=0.05,
        aspect_ratio=:equal,
        size=SNAPSHOT_FIG_SIZE,
    )
    visualize_vorticity(
        u_fields,
        length(u_fields),
        grid,
        joinpath(out_dir, "$(name)_vorticity.svg");
        npoints=256,
        quiver_npoints=16,
        arrow_scale=0.0,
        aspect_ratio=:equal,
        size=SNAPSHOT_FIG_SIZE,
    )
    visualize_scalar(
        p_field,
        length(p_field),
        grid,
        joinpath(out_dir, "$(name)_press.svg");
        npoints=256,
        aspect_ratio=:equal,
        size=SNAPSHOT_FIG_SIZE,
    )
end

"""
    plot_last_snapshots(tasks_dir, plots_dir)

Plot the last snapshot of every task in `tasks_dir` into `<plots_dir>/<n1>x<n2>/last_snapshots/`,
named after the configuration key (`baseline`, `mu_<μ>_gdim_<g>`). Task files written before the
workers stored the snapshot are skipped with a warning.
"""
function plot_last_snapshots(tasks_dir::AbstractString, plots_dir::AbstractString)
    if !isdir(tasks_dir)
        @warn "No tasks folder at $tasks_dir, skipping the last-snapshot plots"
        return nothing
    end
    for path in snapshot_task_files(tasks_dir)
        result = try
            load(path, "result")
        catch e
            @warn "Could not load $path, skipping its last snapshot" exception=e
            continue
        end
        if !haskey(result, "last_snapshot")
            @warn "No last snapshot stored in $path (written before the workers saved it), skipping"
            continue
        end
        nodes = Tuple(result["nodes"])
        out_dir = joinpath(plots_dir, "$(nodes[1])x$(nodes[2])", "last_snapshots")
        mkpath(out_dir)
        name = result["key"]
        haskey(result, "repeat") && (name *= "_rep_$(lpad(result["repeat"], 3, '0'))")
        @info "Plotting last snapshot of $(result["key"]) ($nodes)" path
        plot_last_snapshot(result["last_snapshot"], out_dir, name)
    end
end

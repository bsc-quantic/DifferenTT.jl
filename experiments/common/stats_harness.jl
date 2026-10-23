"""
Repeat / statistics harness shared by the `*_stats` benchmark workers.

Unit of work: ONE repeat of ONE task, written to its own `task_<id>_rep_<r>.jld2` file by
`run_units`. Two layouts decide which units a SLURM job runs (`BENCH_LAYOUT`):

  * `block` (default for the Chorin and anisotropy workers): one job = one BLOCK = every task of
    one problem group (Chorin: a system size; anisotropy: an (N, F) case) at ONE repeat index, in a
    seeded random order. All methods of a block run on the same node, so the paired speedups
    (baseline / method at the same repeat) are within-node ratios and a node-level slowdown cancels
    in them; different repeats are different jobs, so the absolute times sample node-to-node
    variability instead of hiding it. Array index -> (group, repeat) via `decode_block`,
    repeat-major, so the jobs that start first cover every group at the low repeat indices.
  * `task`: one job = one task, all its repeats in sequence (`run_repeats`). Every repeat of a
    configuration then runs on the same node, so a node-level slowdown (it lasts the whole job)
    shifts that configuration as a whole and is invisible in its spread, and the "paired" speedups
    pair repeats that ran on different nodes. Kept for the tt_poisson workers and old data.
  * `unit`: one job = ONE unit (`run_unit`), for units so long that a block does not fit in the
    wall-clock limit (Chorin (10,10) over many steps). Array index -> (task, repeat) via
    `decode_unit`, repeat-major, so the first jobs cover every method at the lowest repeat. Every
    unit runs on its own node: the speedups pair runs on different nodes, so the node effect is NOT
    canceled but enters their spread (it does not bias them, since no method is tied to a node),
    and it cannot be separated from method noise. Everything runs in parallel: one submission.

The rule that makes the statistics meaningful: the problem is fixed, the solver's random inputs
vary. The driver seeds the global RNG with `solver_seed(global_rep)` immediately before every unit,
so a unit must not reseed the RNG with a repeat-independent value, and anything that must NOT vary
across repeats (the TT problems, built with `crossinterpolate2`) has to be built before the units
under its own seed. `solver_seed` deliberately does not depend on the task id, so the baseline and
every preconditioned task of the same group draw the same random inputs at the same repeat index;
that is what makes the per-repeat speedup ratios paired.

Configuration is through environment variables (positional `ARGS` are already used for
`nocond`):

    BENCH_LAYOUT          block | task | unit                       (default block)
    BENCH_N_REPEATS       task layout: repeats run in this process; block/unit layout: repeats
                          covered by the array (n_groups * N_REPEATS, resp. n_tasks * N_REPEATS
                          indices)                                  (default 20)
    BENCH_REPEAT_OFFSET   global index of the first repeat - 1      (default 0)
    BENCH_BASE_SEED       base of the per-repeat seeds              (default 20260917)
    BENCH_TIME_BUDGET_S   stop starting units past this many seconds since process start
                          (default 158400 = 44 h, for a 48 h SLURM limit)
    BENCH_SAFETY_FACTOR   a new unit starts only if it is expected to finish, estimated as
                          SAFETY_FACTOR * (longest unit so far)    (default 1.25)
    BENCH_RETRY_FAILED    1 = retry units that previously failed    (default 0)

A unit that throws (for example the library's condition-number estimator hitting a LAPACK error
for one particular random draw) is logged loudly, recorded in a `task_<id>_rep_<r>.failed` marker
and skipped; the job carries on with the next unit instead of losing all of them. It is not
retried on re-submission unless BENCH_RETRY_FAILED=1, because a seed-dependent failure would just
repeat. The worker exits non-zero at the end if anything failed, and the collector lists the markers.

Wall-clock guard in the block layout: the units of a block cost very different amounts (a baseline
takes seconds, an FAI build hours), so "longest unit so far" can underestimate the next one. The
guard is a safety net, not a scheduler: if it stops a block, every finished unit is on disk, and
resubmitting the same array fills only the missing units. Those refilled units then run in another
job, usually on another node, so their pairs are no longer within-node; the collector counts them
(`n_pairs_samejob`) and `BENCH_SAMEJOB_PAIRS_ONLY=1` drops them.
"""

using Random
using JLD2

const LAYOUT = Symbol(get(ENV, "BENCH_LAYOUT", "block"))
const N_REPEATS = parse(Int, get(ENV, "BENCH_N_REPEATS", "20"))
const REPEAT_OFFSET = parse(Int, get(ENV, "BENCH_REPEAT_OFFSET", "0"))
const BASE_SEED = parse(Int, get(ENV, "BENCH_BASE_SEED", "20260917"))
const TIME_BUDGET_S = parse(Float64, get(ENV, "BENCH_TIME_BUDGET_S", "158400"))
const SAFETY_FACTOR = parse(Float64, get(ENV, "BENCH_SAFETY_FACTOR", "1.25"))
const RETRY_FAILED = get(ENV, "BENCH_RETRY_FAILED", "0") in ("1", "true", "yes")
const JOB_T0 = time()   # set once, when the worker starts

LAYOUT in (:block, :task, :unit) || error("BENCH_LAYOUT must be `block`, `task` or `unit`, got `$LAYOUT`")

# Depends on the GLOBAL repeat index (never the local one) and not on the task id.
solver_seed(rep::Int) = Int(hash((BASE_SEED, rep)) & 0x7fffffff)

function repeat_path(tasks_dir::AbstractString, task_id::Int, global_rep::Int)
    joinpath(tasks_dir, "task_$(lpad(task_id, 3, '0'))_rep_$(lpad(global_rep, 3, '0')).jld2")
end

failed_path(path::AbstractString) = replace(path, r"\.jld2$" => ".failed")

# ---------------------------------------------------------------------------------------------
# Block layout
# ---------------------------------------------------------------------------------------------

"""
    task_groups(group_keys) -> Vector{Vector{Int}}

Task ids grouped by `group_keys[task_id]` (Chorin: `task.nodes`; anisotropy: `(task.N, task.F)`),
groups in order of first appearance, so the numbering follows `build_task_list()`.
"""
function task_groups(group_keys::AbstractVector)
    order = unique(group_keys)
    return [findall(==(k), group_keys) for k in order]
end

"""
    decode_block(array_id, n_groups) -> (group, global_rep)

Repeat-major: array ids `1:n_groups` are repeat `REPEAT_OFFSET + 1` of every group, the next
`n_groups` ids repeat `REPEAT_OFFSET + 2`, and so on, so the array has `n_groups * N_REPEATS`
indices.
"""
function decode_block(array_id::Int, n_groups::Int)
    (1 <= array_id <= n_groups * N_REPEATS) || error(
        "Invalid block array id $array_id: there are $n_groups groups x $N_REPEATS repeats = $(n_groups * N_REPEATS) blocks",
    )
    k, g = divrem(array_id - 1, n_groups)
    return g + 1, REPEAT_OFFSET + k + 1
end

"""
    block_order(task_ids, global_rep, group) -> Vector{Int}

The order in which a block runs its tasks: a shuffle that is deterministic in (BASE_SEED, group,
repeat) and differs between repeats, so no method is systematically the first unit of a job
(residual JIT cost) or the last (drift over the job, neighbours arriving on a shared node). It uses
its own RNG, never the global one, so it cannot change any `solver_seed` stream.
"""
function block_order(task_ids::AbstractVector{Int}, global_rep::Int, group::Int)
    shuffle(MersenneTwister(hash((BASE_SEED, "block_order", group, global_rep))), collect(task_ids))
end

"""
    block_plan(array_id, groups; group_ids=eachindex(groups)) -> (; group, global_rep, task_ids)

The block a block-layout array index runs: its group, its repeat index, and its member task ids
in execution order. `group_ids` restricts the array to some groups (the others run in another
array, e.g. in the unit layout); `group` stays the index into `groups`, so the block order and the
recorded `block_group` do not depend on which groups an array covers.
"""
function block_plan(array_id::Int, groups::Vector{Vector{Int}}; group_ids::AbstractVector{Int}=eachindex(groups))
    i, global_rep = decode_block(array_id, length(group_ids))
    g = group_ids[i]
    return (; group=g, global_rep, task_ids=block_order(groups[g], global_rep, g))
end

# ---------------------------------------------------------------------------------------------
# Unit layout
# ---------------------------------------------------------------------------------------------

"""
    decode_unit(array_id, task_ids) -> (task_id, global_rep)

Repeat-major: array ids `1:length(task_ids)` are repeat `REPEAT_OFFSET + 1` of every task, the
next ones repeat `REPEAT_OFFSET + 2`, and so on, so the array has `length(task_ids) * N_REPEATS`
indices.
"""
function decode_unit(array_id::Int, task_ids::AbstractVector{Int})
    n = length(task_ids)
    (1 <= array_id <= n * N_REPEATS) ||
        error("Invalid unit array id $array_id: there are $n tasks x $N_REPEATS repeats = $(n * N_REPEATS) units")
    k, i = divrem(array_id - 1, n)
    return task_ids[i+1], REPEAT_OFFSET + k + 1
end

# ---------------------------------------------------------------------------------------------
# Running units
# ---------------------------------------------------------------------------------------------

"""
    budget_allows(durations) -> Bool

Whether a further unit is expected to finish inside the wall-clock budget. The first unit of a
process is never blocked; the job has to be sized so that warmup plus one unit fits.
"""
function budget_allows(durations::Vector{Float64})
    isempty(durations) && return true
    return (time() - JOB_T0) + SAFETY_FACTOR * maximum(durations) <= TIME_BUDGET_S
end

function repeat_metadata(
    task_id::Int,
    global_rep::Int;
    first_in_process::Bool,
    layout::Symbol,
    block_group,
    order_in_block::Int,
    block_size::Int,
)
    return Dict{String,Any}(
        "task_id" => task_id,
        "repeat" => global_rep,
        "seed" => solver_seed(global_rep),
        "stopped_early" => false,
        "n_repeats_planned" => N_REPEATS,
        "repeat_offset" => REPEAT_OFFSET,
        "base_seed" => BASE_SEED,
        "slurm_array_job_id" => get(ENV, "SLURM_ARRAY_JOB_ID", ""),
        # The first unit a process runs may still carry residual JIT cost, see
        # BENCH_DISCARD_FIRST_REPEAT in the collectors. In the block layout this is a different,
        # randomly chosen method in every block.
        "first_in_process" => first_in_process,
        "layout" => String(layout),
        # Block layout only (0 / nothing in the task layout). The block itself is identified by
        # `slurm_job_id`, which every worker records: all units of a block share it.
        "block_group" => block_group,
        "order_in_block" => order_in_block,
        "block_size" => block_size,
    )
end

# Write-then-rename so a job killed mid-save never leaves a truncated file that the collector
# (which only reads `task_*_rep_*.jld2`) would try to read. The temp name keeps the `.jld2`
# extension because FileIO picks the format from it.
function atomic_save(path::AbstractString, result::AbstractDict)
    tmp = joinpath(dirname(path), ".tmp_" * basename(path))
    save(tmp, "result", result)
    mv(tmp, path; force=true)
    return path
end

function mark_stopped_early(path::AbstractString)
    result = load(path, "result")
    result["stopped_early"] = true
    atomic_save(path, result)
    return path
end

"""
    run_units(units, tasks_dir; layout, block_group=nothing)

Run each unit `(; task_id, global_rep, run_one)` in the given order: seed the global RNG with
`solver_seed(global_rep)`, call `run_one()`, and write its result dict immediately to
`repeat_path(tasks_dir, task_id, global_rep)`. Units whose file already exists are skipped (so
re-submitting a timed-out job only fills the gaps), and a unit that is unlikely to finish inside
`TIME_BUDGET_S` is not started.
"""
function run_units(units::AbstractVector, tasks_dir::AbstractString; layout::Symbol, block_group=nothing)
    mkpath(tasks_dir)
    @info "Repeat harness" layout n_units=length(units) block_group n_repeats=N_REPEATS repeat_offset=REPEAT_OFFSET base_seed=BASE_SEED time_budget_h=TIME_BUDGET_S /
                                                                                                                                                      3600 safety_factor=SAFETY_FACTOR
    @info "Unit order" units=[(u.task_id, u.global_rep) for u in units]
    flush(stderr)

    block_size = layout == :block ? length(units) : 0
    durations = Float64[]
    last_path = nothing
    n_skipped = 0
    n_started = 0
    n_failed = 0
    stopped_early = false

    for (k, unit) in enumerate(units)
        task_id, global_rep = unit.task_id, unit.global_rep
        path = repeat_path(tasks_dir, task_id, global_rep)
        if isfile(path)
            @info "Unit $k/$(length(units)) (task $task_id, repeat $global_rep) already present, skipping" path
            n_skipped += 1
            continue
        end
        if isfile(failed_path(path)) && !RETRY_FAILED
            @warn "Unit $k/$(length(units)) (task $task_id, repeat $global_rep) previously FAILED, not retrying (BENCH_RETRY_FAILED=1 to retry)" marker=failed_path(
                path
            )
            n_failed += 1
            continue
        end
        if !budget_allows(durations)
            @warn "WALL-CLOCK BUDGET EXHAUSTED, stopping before unit $k/$(length(units)) (task $task_id, repeat $global_rep)" completed=length(
                durations
            ) elapsed_h=(time() - JOB_T0) / 3600 longest_unit_h=maximum(durations) / 3600
            stopped_early = true
            break
        end

        @info "Unit $k/$(length(units)): task $task_id, repeat $global_rep"
        flush(stderr)
        t_rep = time()
        n_started += 1
        Random.seed!(solver_seed(global_rep))
        result = try
            unit.run_one()
        catch e
            e isa InterruptException && rethrow()
            msg = sprint(showerror, e, catch_backtrace())
            @error "UNIT $k (task $task_id, repeat $global_rep) FAILED, continuing with the next unit" seed=solver_seed(
                global_rep
            ) exception=(e, catch_backtrace())
            write(failed_path(path), "seed=$(solver_seed(global_rep))\n" * msg)
            n_failed += 1
            GC.gc(true)
            flush(stderr)
            continue
        end
        push!(durations, time() - t_rep)

        merge!(
            result,
            repeat_metadata(
                task_id,
                global_rep;
                first_in_process=n_started == 1,
                layout,
                block_group,
                order_in_block=layout == :block ? k : 0,
                block_size,
            ),
        )
        result["repeat_wall_time"] = durations[end]
        atomic_save(path, result)
        rm(failed_path(path); force=true)    # a successful retry clears an old marker
        last_path = path
        result = nothing

        # `run_one` keeps its large objects (grids, preconditioners) in locals, so they are
        # unreachable now. GC_BALLAST is a const global and is untouched by this.
        GC.gc(true)
        @info "Unit done" task_id global_rep seconds=durations[end] live_gb=Base.gc_live_bytes() / 2^30
        flush(stderr)
    end

    if stopped_early
        if isnothing(last_path)
            @warn "Stopped early without writing any unit in this process"
        else
            mark_stopped_early(last_path)
        end
    end
    @info "Repeat harness finished" completed=length(durations) skipped=n_skipped failed=n_failed stopped_early
    n_failed > 0 &&
        @error "$n_failed unit(s) FAILED or were skipped as previously failed: see the *.failed markers in $tasks_dir"
    flush(stderr)
    return (; completed=length(durations), skipped=n_skipped, failed=n_failed, stopped_early)
end

"""
    run_block(run_one, plan, tasks_dir)

Block layout: run `run_one(task_id)` for every task of `plan = block_plan(...)`, in `plan.task_ids`
order, all at repeat `plan.global_rep`.
"""
function run_block(run_one::Function, plan, tasks_dir::AbstractString)
    units = [(; task_id=t, global_rep=plan.global_rep, run_one=() -> run_one(t)) for t in plan.task_ids]
    return run_units(units, tasks_dir; layout=:block, block_group=plan.group)
end

"""
    run_repeats(run_one, task_id, tasks_dir)

Task layout: run `run_one(global_rep)` for `global_rep in REPEAT_OFFSET+1 : REPEAT_OFFSET+N_REPEATS`,
one file per repeat, all in this process (so all on one node; see the header).
"""
function run_repeats(run_one::Function, task_id::Int, tasks_dir::AbstractString)
    reps = REPEAT_OFFSET .+ (1:N_REPEATS)
    units = [(; task_id, global_rep=r, run_one=() -> run_one(r)) for r in reps]
    return run_units(units, tasks_dir; layout=:task)
end

"""
    run_unit(run_one, task_id, global_rep, tasks_dir)

Unit layout: run `run_one()` once, for `task_id` at repeat `global_rep` (from `decode_unit`).
"""
function run_unit(run_one::Function, task_id::Int, global_rep::Int, tasks_dir::AbstractString)
    return run_units([(; task_id, global_rep, run_one)], tasks_dir; layout=:unit)
end

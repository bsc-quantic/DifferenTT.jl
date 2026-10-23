"""
    Smoke test of the experiment scripts

An experiment script that no longer runs after a change of the library is only discovered when someone submits a job that
takes hours. We therefore run here every non-archived script of `experiments/` once, on a tiny problem (smoke mode,
`DIFFERENTT_SMOKE=1`), and report which ones fail. For each benchmark we run, in this order, the generator of its inputs (if
any), the worker on a few tasks (the baseline and one FAI task; `tt_poisson` and `tt_anisotropy` also run once with local condition
numbers),
the collector and the plotter; the paper's statistics benchmarks (`paper/*_stats`) run two blocks (two repeats) of their
worker, and `chorin_pipeline_stats` ends with a step that must fail, to check the safeguard against a stale mask (the
worker is given a mask whose metadata do not match its grid and must stop with the `ArgumentError`). Every step is a separate `julia --project=experiments` process, run one at a time, with a
timeout of 5 minutes. The steps of one benchmark share a temporary folder (`DIFFERENTT_SMOKE_DIR`), so that the collector and
the plotter read what the smoke worker wrote and never the committed results.

# Usage

From the root of the repository:

    julia --project=experiments experiments/smoke.jl [name ...]

With no argument it runs every benchmark; with names (as printed in the first column of the table) it runs only those. It prints one line per step as it
finishes, and a table with the result, the time and the last error line of every step at the end; the logs of the steps are in
the temporary folder that it prints. The exit status is nonzero if a step failed.

# Resources

Julia is started with `--heap-size-hint=2G` and one thread per step. We have not measured the total time.
"""

using Dates
using Printf

const EXPERIMENTS = @__DIR__
const TIMEOUT = 5 * 60  # seconds per step

struct Step
    script::String          # relative to experiments/
    args::Vector{String}
    env::Vector{Pair{String,String}}   # more environment variables; "{dir}" in a value is the folder of the benchmark
    expect_error::String    # "" for a step that must succeed; else the step must fail with this text in its log
end
function Step(
    script::String,
    args::Vector{String};
    env::Vector{Pair{String,String}}=Pair{String,String}[],
    expect_error::String="",
)
    return Step(script, args, env, expect_error)
end

struct Job
    name::String
    steps::Vector{Step}
end

# Steps of a benchmark of fai_benchmarks/: worker tasks, collector(s), plotter.
function benchmark(
    folder::String;
    masks::Bool=false,
    stale_mask_check::Bool=false,
    nocond_tasks::Vector{Int}=[1, 2],
    has_nocond::Bool=false,
    cond_tasks::Vector{Int}=Int[],
    collect_args::Vector{String}=String[],
    plot_args::Vector{String}=String[],
)
    dir = "fai_benchmarks/$folder"
    steps = Step[]
    masks && push!(steps, Step("$dir/generate_masks.jl", String[]))
    for t in nocond_tasks
        push!(steps, Step("$dir/worker.jl", [string(t), (has_nocond ? ["nocond"] : String[])...]))
    end
    for t in cond_tasks
        push!(steps, Step("$dir/worker.jl", [string(t)]))
    end
    if stale_mask_check
        # The worker must refuse a mask whose metadata do not match its grid (common/mask_grid.jl).
        push!(steps, Step("common/stale_mask_smoke.jl", String[]))
        push!(
            steps,
            Step(
                "$dir/worker.jl",
                ["1", (has_nocond ? ["nocond"] : String[])...];
                env=["CHORIN_MASK_FILE" => "{dir}/masks_stale.jld2"],
                expect_error="does not belong to the grid of this run",
            ),
        )
    end
    push!(steps, Step("$dir/collect.jl", [collect_args; (has_nocond ? ["nocond"] : String[])]))
    isempty(cond_tasks) || push!(steps, Step("$dir/collect.jl", collect_args))
    push!(steps, Step("$dir/plot_results.jl", plot_args))
    return Job(folder, steps)
end

const JOBS = [
    benchmark("tt_poisson"; has_nocond=true, cond_tasks=[2]),
    benchmark("tt_poisson_global"),
    benchmark("tt_anisotropy"; has_nocond=true, nocond_tasks=[1, 2, 3], cond_tasks=[2]),
    benchmark("tt_anisotropy_global"; nocond_tasks=[1, 2, 3]),
    benchmark("chorin_pipeline"; masks=true, stale_mask_check=true, has_nocond=true),
    benchmark("chorin_pipeline_globalcg"; masks=true, stale_mask_check=true),
    benchmark("liddriven_pipeline"; has_nocond=true),
    benchmark("liddriven_pipeline_globalcg"),
    Job(
        "liddriven_cavity",
        [
            Step("validation/liddriven_cavity/worker.jl", ["1"]),
            Step("validation/liddriven_cavity/plot_snapshots.jl", String[]),
            Step("validation/liddriven_cavity/ghia_comparison.jl", String[]),
        ],
    ),
    Job("tunnel2d", [Step("validation/tunnel2d/tunnel2d.jl", String[])]),
    Job("tunnel2dsteps", [Step("validation/tunnel2dsteps/tunnel2dpoisson.jl", String[])]),
    Job("anisotropy_floor", [Step("paper/anisotropy_floor/discrete_floor.jl", String[])]),
    Job("mask_grid_selftest", [Step("common/mask_grid_selftest.jl", String[])]),
    Job("mu_rules_check", [Step("common/mu_rules_check.jl", String[])]),
    Job(
        "linsolve_spd",
        [
            Step("fai_benchmarks/chorin_pipeline/generate_masks.jl", String[]),
            Step("paper/linsolve_spd/worker.jl", String[]),
            Step("paper/linsolve_spd/plot_paper.jl", String[]),
            Step("paper/linsolve_spd/plot_results.jl", String[]),
        ],
    ),
    # The paper's statistics runs: two blocks (two repeats), the collector and the paper plotter. The Chorin one reads the
    # mask of chorin_pipeline, so the generator runs first, and a last step checks that the worker refuses a mask whose
    # metadata do not match its grid.
    Job(
        "tt_anisotropy_stats",
        [
            Step("paper/tt_anisotropy_stats/worker.jl", ["1", "nocond"]),
            Step("paper/tt_anisotropy_stats/worker.jl", ["2", "nocond"]),
            Step("paper/tt_anisotropy_stats/collect.jl", ["nocond"]),
            Step("paper/tt_anisotropy_stats/plot_paper.jl", String[]),
        ],
    ),
    Job(
        "chorin_pipeline_stats",
        [
            Step("fai_benchmarks/chorin_pipeline/generate_masks.jl", String[]),
            Step("paper/chorin_pipeline_stats/worker.jl", ["1", "nocond"]),
            Step("paper/chorin_pipeline_stats/worker.jl", ["2", "nocond"]),
            Step("paper/chorin_pipeline_stats/collect.jl", ["nocond"]),
            Step("paper/chorin_pipeline_stats/plot_paper.jl", String[]),
            Step("common/stale_mask_smoke.jl", String[]),
            Step(
                "paper/chorin_pipeline_stats/worker.jl",
                ["1", "nocond"];
                env=["CHORIN_MASK_FILE" => "{dir}/masks_stale.jld2"],
                expect_error="does not belong to the grid of this run",
            ),
        ],
    ),
]

struct Outcome
    job::String
    step::String
    status::String
    seconds::Float64
    last_error::String
end

# Last line of the log that names the error, or its last non-empty line.
function last_error_line(log::String)
    lines = filter(!isempty, strip.(split(read(log, String), '\n')))
    isempty(lines) && return ""
    errors = filter(l -> startswith(l, "ERROR:"), lines)
    isempty(errors) && (errors = filter(l -> occursin("Error", l), lines))
    return first(isempty(errors) ? last(lines) : first(errors), 160)
end

function run_step(job::Job, step::Step, root::String, logdir::String, index::Int)
    script = joinpath(EXPERIMENTS, step.script)
    label = join([basename(dirname(step.script)), basename(step.script), step.args...], " ")
    log = joinpath(logdir, @sprintf("%02d_%s_%s.log", index, job.name, replace(basename(step.script), ".jl" => "")))
    dir = joinpath(root, job.name)
    env = ["DIFFERENTT_SMOKE" => "1", "DIFFERENTT_SMOKE_DIR" => dir]
    append!(env, [k => replace(v, "{dir}" => dir) for (k, v) in step.env])
    isempty(step.expect_error) || (label *= " (expect an error)")
    julia = joinpath(Sys.BINDIR, Base.julia_exename())
    cmd = `$julia --project=$EXPERIMENTS --heap-size-hint=2G -t 1 $script $(step.args)`
    t0 = time()
    status = "pass"
    open(log, "w") do io
        proc = run(pipeline(Cmd(addenv(cmd, env...); dir=dirname(EXPERIMENTS)); stdout=io, stderr=io); wait=false)
        while process_running(proc)
            if time() - t0 > TIMEOUT
                kill(proc)
                sleep(2)
                process_running(proc) && kill(proc, Base.SIGKILL)
                status = "timeout"
                break
            end
            sleep(1)
        end
        wait(proc)
        if isempty(step.expect_error)
            status == "pass" && !success(proc) && (status = "FAIL")
        elseif status == "pass"
            # a step that tests a safeguard passes only if it fails, and for the reason that it names
            failed_as_expected = !success(proc) && occursin(step.expect_error, read(log, String))
            failed_as_expected || (status = "FAIL")
        end
    end
    seconds = time() - t0
    # A collector that finds no task file still exits normally: a smoke run must not count it as a pass.
    if status == "pass" && occursin("Collected 0 tasks", read(log, String))
        status = "FAIL"
        write(log, read(log, String) * "\nERROR: the collector found no task file\n")
    end
    err = status == "pass" ? "" : last_error_line(log)
    @printf("%-9s %6.1f s  %-28s %s\n", status, seconds, job.name, label)
    flush(stdout)
    return Outcome(job.name, label, status, seconds, err)
end

function main(names::Vector{String})
    jobs = isempty(names) ? JOBS : filter(j -> j.name in names, JOBS)
    unknown = setdiff(names, [j.name for j in JOBS])
    isempty(unknown) ||
        error("unknown benchmark(s) $(join(unknown, ", ")); known: $(join([j.name for j in JOBS], ", "))")
    root = mktempdir(; cleanup=false)  # kept, so that the logs can be read after the run
    logdir = mkpath(joinpath(root, "logs"))
    println(
        "Smoke run of $(length(jobs)) benchmarks, $(Dates.format(now(), "yyyy-mm-dd HH:MM")); logs and outputs in $root"
    )
    outcomes = Outcome[]
    index = 0
    for job in jobs, step in job.steps
        index += 1
        push!(outcomes, run_step(job, step, root, logdir, index))
    end

    println("\n", rpad("benchmark", 28), rpad("step", 62), rpad("result", 9), rpad("time (s)", 10), "last error line")
    for o in outcomes
        println(
            rpad(o.job, 28), rpad(o.step, 62), rpad(o.status, 9), rpad(@sprintf("%.1f", o.seconds), 10), o.last_error
        )
    end
    nfail = count(o -> o.status != "pass", outcomes)
    println("\n$(length(outcomes) - nfail) of $(length(outcomes)) steps passed; logs in $(logdir)")
    return nfail == 0 ? 0 : 1
end

if abspath(PROGRAM_FILE) == @__FILE__
    exit(main(String.(ARGS)))
end

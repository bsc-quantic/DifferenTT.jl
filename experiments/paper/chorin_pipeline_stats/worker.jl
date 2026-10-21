"""
    Chorin pipeline benchmark, statistics variant (paper Table 2 and Figs. 12-13): worker

We solve the pressure Poisson system of every time step of a Chorin projection of the flow past a cylinder, the setup of
Sec. 5.2 of the paper: Re = 3000 in the box [0, 2] × [0, 1] (periodic in y), a cylinder of radius 0.1 at (0.5, 0.5)
imposed through a mask, an RK4 predictor and stencils of order 8, in the quantized TT format on 2⁸ × 2⁸ and 2¹⁰ × 2¹⁰
nodes. This is the benchmark of `fai_benchmarks/chorin_pipeline`, repeated: every task is run `BENCH_N_REPEATS` times
(`common/stats_harness.jl`), and the worker writes one result file per task and repeat, so that the plotter can draw the
spread of the times and the speedups paired by repeat. Every repeat rebuilds the grid from the initial condition, since
`chorin_step!` changes it in place, and the repeat sets the seed of the solver's random inputs and of the FAI build.

# Problem and parameters

Each task is one system size with either no preconditioner (the baseline) or the FAI with μ = 10⁷ and χ_G ∈ {2, …, 10}
(10 tasks per size, 20 in total, `TASKS_PER_SIZE` of `submit.sh`):

- 2⁸ × 2⁸ nodes, 500 steps of dt = 0.001, solution bond dimension at most 150;
- 2¹⁰ × 2¹⁰ nodes, 1000 steps of dt = 0.0005, solution bond dimension at most 200.

At each step we solve with `ResidualALS` (the "MALS" of the paper, local solves preconditioned at every site) to a relative
residual of 1e-3, with `cutoff = 0` and at most 100 sweeps and 2000 local iterations, starting from the previous
pressure. We build the FAI once per task, before the time loop, on the pressure operator with the two-site ALS solver (30 sweeps, `cutoff = 1e-12`, stopped at a
relative residual of 0.1 of Eq. (22)); the element-wise map D uses `cutoff = 1e-11` and an oversampling of 20. The mask is
that of `fai_benchmarks/chorin_pipeline/masks.jld2` (the worker deliberately shares it with that benchmark). The grid is
`Grid(arch, (2.0, 1.0); periodic=periodic_axes(bcs))`, so the period in y is 1, and the worker stops with an
`ArgumentError` if the metadata of the mask do not match that grid (`common/mask_grid.jl`).

The setups above are those of the paper repository. Three changes of the library since the data of the current figures
(tag `paper-ttfai-data`) make new runs not comparable with them: the FAI now solves the correct (not the transposed)
FSAI problem, the grid has the period of the box in y (it was the box plus one cell, so the masks had to be regenerated),
and the Frobenius floor of `ResidualALS` uses `cutoff` where it used `cutoff²` (no effect here, where `cutoff = 0`).
A known limitation of the library bears on the numbers: the Dirichlet closure with `stencil_order = 8` is only first-order
accurate (see "Known limitations" in `NEWS.md`).

# Inputs and outputs

With the default layout (`BENCH_LAYOUT=block`), `ARGS[1]` is a block index and the job runs every task of one system size
at one repeat index, in a seeded random order; with `BENCH_LAYOUT=unit` it is a (task, repeat) unit, as `submit.sh` does
for the 2¹⁰ × 2¹⁰ grid (`CHORIN_STATS_SIZES` selects the sizes of an array); with `BENCH_LAYOUT=task` it is a task id
and the job runs all the repeats of that task. `nocond` is `ARGS[2]`: without it, we estimate the local condition
numbers κ(A) and κ(P⁻¹A) of every site. The worker writes `tasks/task_NNN_rep_RRR.jld2` (key `"result"`) under
`results/` or `results_nocond/` of this folder, or under `CHORIN_STATS_RESULTS_DIR` if set; repeat 1 also stores the final
fluid state. The mask file is `CHORIN_MASK_FILE` if set, else the committed one. The other configuration variables
(`BENCH_N_REPEATS`, `BENCH_REPEAT_OFFSET`, `BENCH_BASE_SEED`, `BENCH_TIME_BUDGET_S`, `CHORIN_STATS_N_STEPS_OVERRIDE`)
are described in `common/stats_harness.jl` and `submit.sh`.

Smoke mode (`DIFFERENTT_SMOKE=1`) shrinks the run to the 2⁸ × 2⁸ grid, 2 steps, the baseline and FAI at χ_G = 2, without
local condition numbers, through the `SMOKE` branches next to the constants. It reads the mask that the smoke generator
`fai_benchmarks/chorin_pipeline/generate_masks.jl` wrote into `DIFFERENTT_SMOKE_DIR` and writes there too (a fresh
temporary folder if it is unset), so that the committed masks and data are never touched. `experiments/smoke.jl` runs
the generator, two blocks, the collector and the plotter.

# How to run

From the root of the repository, after regenerating the masks (see `paper/README.md`):

    DIFFERENTT_SMOKE=1 julia --project=experiments experiments/paper/chorin_pipeline_stats/worker.jl 1 nocond
    julia --project=experiments experiments/paper/chorin_pipeline_stats/worker.jl 1 nocond
    bash experiments/paper/chorin_pipeline_stats/submit.sh nocond

# Resources

The job script requests, for the default 10 repeats, an array of 10 block jobs for the 2⁸ × 2⁸ grid (2 days each) and
an array of 100 unit jobs for the 2¹⁰ × 2¹⁰ grid (1 day each), each on one node with 20 CPUs (`CPUS_PER_TASK`), without a
`--mem` directive; the worker sets Julia's heap-size hint to 80 % of the memory of the allocation, and `GC_BALLAST_GB`
controls the GC ballast (`common/gc_ballast.jl`; the job script sets 7). The comments of `submit.sh` give the time of
each unit as measured in the run of October 2026.

# Paper

Sec. 5.2: Table 2 (`tab:chorin_mals_params`) lists the parameters of the runs, which are the constants of this worker,
and Figs. 12-13 (`fig:chorin_mals_iterations`, `fig:chorin_mals_cumulative_speedup`) are drawn by `plot_paper.jl` from
the output of `collect.jl`.
"""

using ITensors
using ITensorMPS
using LinearAlgebra
using Statistics
using JLD2
using DifferenTT
using Random

const cpus_per_task = parse(Int, get(ENV, "SLURM_CPUS_PER_TASK", "1"))
LinearAlgebra.BLAS.set_num_threads(cpus_per_task)
@info "Set BLAS threads to $cpus_per_task"
flush(stderr)
@info "Runtime" blas_threads=LinearAlgebra.BLAS.get_num_threads() julia_threads=Threads.nthreads() host=gethostname()
flush(stderr)

# ---- GC memory ballast (see common/gc_ballast.jl) ----
include(joinpath(@__DIR__, "..", "..", "common", "gc_ballast.jl"))
flush(stderr)

include(joinpath(@__DIR__, "..", "..", "common", "stats_harness.jl"))
include(joinpath(@__DIR__, "..", "..", "common", "mask_grid.jl"))

include(joinpath(@__DIR__, "..", "..", "common", "smoke.jl"))
# Smoke mode writes under DIFFERENTT_SMOKE_DIR (a fresh temporary folder if unset) instead of this folder.
const OUT_ROOT = output_root(@__DIR__; fresh=true)

const use_nocond = length(ARGS) > 1 && ARGS[2] == "nocond"
const default_base_folder = if use_nocond
    joinpath(OUT_ROOT, "results_nocond")
else
    joinpath(OUT_ROOT, "results")
end
base_folder = get(ENV, "CHORIN_STATS_RESULTS_DIR", default_base_folder)
const TASKS_DIR = joinpath(base_folder, "tasks")
mkpath(TASKS_DIR)

# Deliberately the ORIGINAL directory's masks: masks.jld2 is a 770 KB binary that the gitignore of
# `*_stats/` would not re-include, and a divergent copy would silently make these runs
# non-comparable to the originals. Do not copy it here.
# Smoke mode reads the mask that the smoke generator wrote into DIFFERENTT_SMOKE_DIR, never the committed file;
# CHORIN_MASK_FILE overrides either.
const MASK_FILE = get(
    ENV,
    "CHORIN_MASK_FILE",
    if SMOKE
        joinpath(OUT_ROOT, "masks.jld2")
    else
        joinpath(@__DIR__, "..", "..", "fai_benchmarks", "chorin_pipeline", "masks.jld2")
    end,
)
# Note: we assume the user has the masks in MASK_FILE. If not, they must be copied over or generated.

const BOX_SIZE = (2.0, 1.0)
const RE_NUM = 3000
const STENCIL_ORDER = 8

const dt_map = Dict((8, 8) => 0.001, (10, 10) => 0.0005, (12, 12) => 0.00001)

# Smoke mode: the 8×8 grid, 2 steps, the baseline and FAI at χ_G = 2, and no local condition numbers (the dense local
# eigenproblems take about ten minutes even on this grid).
const n_steps_map = SMOKE ? Dict((8, 8) => 2) : Dict((8, 8) => 500, (10, 10) => 1000, (12, 12) => 20)

# Overridable for fast local verification only — the default reproduces n_steps_map[nodes] exactly.
function resolve_n_steps(nodes::Tuple{Int,Int})
    override = get(ENV, "CHORIN_STATS_N_STEPS_OVERRIDE", "")
    return isempty(override) ? n_steps_map[nodes] : parse(Int, override)
end

const system_sizes = SMOKE ? [(8, 8)] : [(8, 8), (10, 10)]
const cond_nodes = SMOKE ? Tuple{Int,Int}[] : [(8, 8), (10, 10)]
# const system_sizes = [(10, 10)]
# const cond_nodes  = [(10, 10)]

function should_compute_cond(nodes::Tuple{Int,Int}, use_nocond::Bool)::Bool
    return !use_nocond && nodes in cond_nodes
end

# const mu_values = [5e6, 1e7, 5e7, 1e8]
const mu_values = [1e7]

const g_maxdims = SMOKE ? (2:2) : (2:10)

const maxdims_map = Dict((8, 8) => 150, (10, 10) => 200, (12, 12) => 250)

# WARNING: MAKE SURE TO CHANGE THIS TO >0.0 IF RUNNING ALS -> OOM WILL HAPPEN
const cutoff_map = Dict((8, 8) => 0.0, (10, 10) => 0.0, (12, 12) => 0.0)

const nsweeps_map = Dict((8, 8) => 100, (10, 10) => 100, (12, 12) => 100)

const tol_map = Dict((8, 8) => 1e-3, (10, 10) => 1e-3, (12, 12) => 1e-3)

const fai_tol_map = Dict((8, 8) => 0.1, (10, 10) => 0.1, (12, 12) => 0.1)

ITensors.disable_warn_order()
ENV["JULIA_DEBUG"] = ""

struct TaskSpec
    nodes::Tuple{Int,Int}
    is_baseline::Bool
    mu::Float64
    gdim::Int
end

function build_task_list()
    tasks = TaskSpec[]
    for nodes in system_sizes
        push!(tasks, TaskSpec(nodes, true, 0.0, 0))
        for gdim in g_maxdims
            for μ in mu_values
                push!(tasks, TaskSpec(nodes, false, μ, gdim))
            end
        end
    end
    return tasks
end

function build_grid(nodes::NTuple{2,Int}, max_steps::Int; shared_sites::Union{Nothing,Vector{<:Index}}=nothing)
    # `siteinds` mints fresh, uniquely-tagged Index objects on every call. A throwaway warmup
    # grid built independently from the real grid would therefore carry *different* site
    # indices even with identical nodes/mask — any MPO built against one (e.g. a FAI
    # preconditioner built from the real grid's A) cannot be applied to the other's MPS/MPO,
    # and ITensorMPS raises "must share a set of site indices". Passing `shared_sites` lets a
    # warmup grid reuse the real grid's indices so cross-grid objects like `precond` stay valid.
    sites = isnothing(shared_sites) ? siteinds(2, sum(nodes)) : shared_sites
    arch = QTTArchitecture(nodes; interleaved=false)

    bcsvxx = BoundaryCond1d(DirichletBC(1.0), NeumannBC(0.0))
    bcsvxy = BoundaryCond1d(PeriodicBC(), PeriodicBC())

    bcsvyx = BoundaryCond1d(DirichletBC(0.0), NeumannBC(0.0))
    bcsvyy = BoundaryCond1d(PeriodicBC(), PeriodicBC())

    bcspx = BoundaryCond1d(NeumannBC(0.0), DirichletBC(1.0))
    bcspy = BoundaryCond1d(PeriodicBC(), PeriodicBC())

    bcsvx = BoundaryConditions(bcsvxx, bcsvxy)
    bcsvy = BoundaryConditions(bcsvyx, bcsvyy)
    bcsp = BoundaryConditions(bcspx, bcspy)

    # The y axis is periodic: the grid takes the flags from the boundary conditions, so the wrapped stencils have period L
    periodic = DifferenTT.periodic_axes(bcsvx)
    grid = Grid(arch, BOX_SIZE; periodic=periodic)

    # We will assume object_mask is generated properly somewhere or just create a dummy one for now.
    # The benchmark requires the true masks to reflect exact conditions.
    # For now, we will create an empty mask if the file doesn't exist, to prevent crashing.
    if isfile(MASK_FILE)
        mask_key = "mask_$(nodes[1])_$(nodes[2])"
        object_mask = load(MASK_FILE, mask_key)
        # The mask is only valid on the grid it was sampled on: stop if the file says otherwise (or says nothing)
        check_mask_grid(load_mask_grid(MASK_FILE, mask_key), nodes, BOX_SIZE, periodic; file=MASK_FILE, key=mask_key)
        siteinds(object_mask) != sites && replace_siteinds!(object_mask, sites)
    else
        object_mask = ones_mps(sites)
        @warn "mask not found; using ones_mps (all fluid)"
    end

    mps_vx = tt_mul(object_mask, ones_mps(sites))
    mps_vy = 0.0 * ones_mps(sites)

    v_field = VectorField((mps_vx, mps_vy), arch, (bcsvx, bcsvy))
    p_field = ScalarField(zeros_mps(sites), arch, bcsp)

    time_stepper = RK4(;
        time_evol_kwargs=TimeEvolArgs(0.0, dt_map[nodes]*max_steps, max_steps), cutoff=1e-11, maxdim=maxdims_map[nodes]
    )
    experiment = INSExperiment(
        grid,
        v_field,
        p_field,
        RE_NUM,
        object_mask;
        stencil_order=STENCIL_ORDER,
        maxdim=maxdims_map[nodes],
        cutoff=1e-11,
    )

    u_diff_ops = DifferentialOps(grid, experiment.u_fields; stencil_order=STENCIL_ORDER)
    u_boundary_null = BoundaryNull(experiment.u_fields; stencil_order=STENCIL_ORDER)
    p_diff_ops = DifferentialOpsScalar(grid, experiment.p_field; stencil_order=STENCIL_ORDER)
    Δ_cache = symmetrized_laplacian(experiment.p_field, experiment.grid; stencil_order=STENCIL_ORDER)
    A = -1.0 * Δ_cache

    # The pressure has a Dirichlet boundary, so A is not pinned and the RHS needs no mean removal
    regularize = DifferenTT.regularization_check(experiment)
    regularize && error("Benchmark operator A is not regularized; pressure BCs need a Dirichlet boundary")
    cache = ChorinCache(u_diff_ops, p_diff_ops, u_boundary_null, A, regularize)

    return (;
        nodes=nodes,
        A=A,
        experiment=experiment,
        time_stepper=time_stepper,
        cache=cache,
        object_mask=object_mask,
        sites=sites,
    )
end

function masked_velocity(grid)
    experiment = grid.experiment
    prev_u = last(experiment.u_fields)
    return DifferenTT.mask_correction(prev_u, grid.object_mask; cutoff=experiment.cutoff, maxdim=experiment.maxdim)
end

function build_rhs(grid, masked_u)
    dt = grid.time_stepper.time_evol_kwargs.dt
    return DifferenTT.poisson_rhs(grid.experiment, grid.cache, masked_u, dt)
end

# l2 norm of the discrete divergence of `u`, measured like the divergence in `poisson_rhs`: the rows
# next to Dirichlet velocity boundaries are removed, since the projection does not act on them.
function velocity_divergence_norm(grid, u)
    experiment = grid.experiment
    cutoff, maxdim = experiment.cutoff, experiment.maxdim
    ∇u = DifferenTT.divergence(u, grid.cache.u_diff_ops; cutoff=cutoff, maxdim=maxdim)
    correction = DifferenTT.divergence_dirichlet_correction(
        experiment.u_fields; stencil_order=experiment.stencil_order, cutoff=cutoff, maxdim=maxdim
    )
    ∇u = isnothing(correction) ? ∇u : apply(correction, ∇u; cutoff=cutoff, maxdim=maxdim)
    return norm(∇u)
end

# Final fluid state of a run, stored in the task file so the plotters can draw the last snapshot.
# Only the repeat SNAPSHOT_REPEAT stores it: every repeat solves the same problem, so the others
# would only add tens of MB per file. The collectors strip it before building their dataframes.
const SNAPSHOT_REPEAT = 1

function add_last_snapshot!(result_dict, grid, n_steps::Int, global_rep::Int)
    global_rep == SNAPSHOT_REPEAT && (result_dict["last_snapshot"] = last_snapshot(grid, n_steps))
    return result_dict
end

function last_snapshot(grid, n_steps::Int)
    experiment = grid.experiment
    return Dict{String,Any}(
        "grid" => experiment.grid,
        "u_fields" => experiment.u_fields,
        "p_field" => experiment.p_field,
        "n_steps" => n_steps,
    )
end

function chorin_step!(
    grid, sweep_kwargs, compute_cond::Bool, tol_val::Float64, step_index::Int, precond=nothing; global_rep::Int=0
)
    experiment = grid.experiment
    masked_u = masked_velocity(grid)
    rhs = build_rhs(grid, masked_u)

    numiter_store = KrylovSweepStats()
    cond_store = compute_cond ? KrylovCondStats() : nothing
    pc_cond_store = compute_cond ? KrylovCondStats() : nothing
    updater_kwargs = LocalUpdaterArgs(;
        ishermitian=true,
        isposdef=true,
        rtol=0.1 * tol_val,
        krylovdim=1,
        maxiter=2000,
        numiter_store=numiter_store,
        cond_store=cond_store,
        pc_cond_store=pc_cond_store,
    )

    # A and b are bound by linearsolve to the system grid.A * pdt = rhs of this step
    res_observer = ResidualObserver(; rtol=tol_val)
    step_sweep_kwargs = SweepingArgs(; NamedTuple(sweep_kwargs)..., (observer!)=res_observer)
    resals_solver = ResidualALS(;
        sweep_kwargs=step_sweep_kwargs, updater_kwargs=updater_kwargs, residual_tol=0.5 * tol_val
    )
    amen_solver = AMEn(;
        sweep_kwargs=step_sweep_kwargs, updater_kwargs=updater_kwargs, enrichment_maxdim=3, enrichment_cutoff=1e-10
    )
    als_solver = ALS(; sweep_kwargs=step_sweep_kwargs, updater_kwargs=updater_kwargs)

    solver = resals_solver

    dt = grid.time_stepper.time_evol_kwargs.dt
    previous_p = last(experiment.p_field)
    # previous_p = nothing
    if isnothing(previous_p)
        Random.seed!(hash((BASE_SEED, global_rep, grid.nodes, step_index)))
        pdt_guess = random_mps(grid.experiment.u_fields.usites)
    else
        pdt_guess = previous_p * dt
    end

    # Same RNG state before every timed solve of a given repeat (baseline and preconditioned alike),
    # so no random choice inside the solver can differ between methods. It depends on the repeat, so
    # the statistics still sample the solver's randomness.
    Random.seed!(hash((BASE_SEED, global_rep, grid.nodes, step_index)))

    # Collect the garbage from masked_velocity/build_rhs before timing, so it isn't charged to
    # the solve.
    GC.gc(true)
    if isnothing(precond)
        stats = @timed linearsolve(solver, grid.A, rhs, pdt_guess)
    else
        stats = @timed linearsolve(solver, grid.A, rhs, precond, pdt_guess)
    end
    pdt = stats.value
    t = stats.time

    # Compute relative residual using inner product to save memory
    term1 = ITensorMPS.inner(grid.A, pdt, grid.A, pdt)
    term2 = 2 * real(ITensorMPS.inner(rhs', grid.A, pdt))
    term3 = ITensorMPS.inner(rhs, rhs)
    rel_res = sqrt(max(0.0, term1 - term2 + term3)) / norm(rhs)

    @info "Residual $rel_res"
    flush(stderr)

    sol_maxdim = maxlinkdim(pdt)
    @info "Sol_maxdim = $sol_maxdim"
    flush(stderr)

    divfree_p = pdt * (1/dt)

    # Overwrite in place: p_field keeps only the latest physical p (not a time series).
    experiment.p_field[length(experiment.p_field)] = divfree_p
    div_free_u = DifferenTT.∇p_correction(masked_u, experiment, grid.cache, divfree_p, dt)

    # How much the pressure correction reduces the divergence (outside the timed solve)
    div_norm_before = velocity_divergence_norm(grid, masked_u)
    div_norm_after = velocity_divergence_norm(grid, div_free_u)
    div_ratio = div_norm_after / div_norm_before
    @info "Divergence norm ratio (after / before) $div_ratio"
    flush(stderr)

    rhs_func = NavierStokesPredictorRHS(experiment, grid.cache.u_diff_ops, grid.cache.u_boundary_null)
    timestepped_u = timestep(grid.time_stepper, div_free_u, rhs_func, 0.0)

    experiment.u_fields[length(experiment.u_fields)] = timestepped_u

    iters_per_site = DifferenTT.positions_sum(numiter_store)

    res_dict = (;
        t,
        compile_t=stats.compile_time,
        gc_t=stats.gctime,
        rel_res,
        sol_maxdim,
        iters_per_site,
        div_norm_before,
        div_norm_after,
        div_ratio,
    )

    if compute_cond
        local_conds = isnothing(precond) ? positions_mean_cond(cond_store) : positions_mean_cond(pc_cond_store)
        local_cond_mean = isempty(local_conds) ? NaN : mean(local_conds)
        res_dict = merge(res_dict, (; local_conds, local_cond_mean))
    end

    return res_dict
end

"""
    run_single(task, global_rep; warmup_time)

One repeat of a task. The caller (`run_repeats`) has already seeded the global RNG with
`solver_seed(global_rep)`; nothing here may reseed with a repeat-independent value, or every
repeat would build the same FAI preconditioner. The grid is rebuilt from the initial condition on
every call because `chorin_step!` mutates it in place.
"""
function run_single(task::TaskSpec, global_rep::Int; warmup_time::Float64=0.0)
    N_STEPS = resolve_n_steps(task.nodes)
    compute_cond = should_compute_cond(task.nodes, use_nocond)
    grid = build_grid(task.nodes, N_STEPS)
    maxdim_val = maxdims_map[task.nodes]
    nsweep_val = nsweeps_map[task.nodes]
    cutoff_val = cutoff_map[task.nodes]
    tol_val = tol_map[task.nodes]
    faibuild_tol = fai_tol_map[task.nodes]

    sweep_kwargs = SweepingArgs(; nsweeps=nsweep_val, maxdim=maxdim_val, cutoff=cutoff_val)

    if task.is_baseline
        @info "Running baseline for $(task.nodes)"
        flush(stderr)

        solve_times = Float64[]
        compile_times = Float64[]
        gc_times = Float64[]
        iters = Vector{Float64}[]
        residuals = Float64[]
        solution_maxdims = Int[]
        div_norms_before = Float64[]
        div_norms_after = Float64[]
        div_ratios = Float64[]
        iters_per_site = Float64[]
        conds_per_site = Float64[]
        avg_conds = Float64[]

        for i in 1:N_STEPS
            @info "Step $i / $N_STEPS"
            flush(stderr)

            step_res = chorin_step!(grid, sweep_kwargs, compute_cond, tol_val, i; global_rep)
            push!(solve_times, step_res.t)
            push!(compile_times, step_res.compile_t)
            push!(gc_times, step_res.gc_t)
            push!(residuals, step_res.rel_res)
            push!(solution_maxdims, step_res.sol_maxdim)
            push!(div_norms_before, step_res.div_norm_before)
            push!(div_norms_after, step_res.div_norm_after)
            push!(div_ratios, step_res.div_ratio)
            push!(iters, step_res.iters_per_site)

            if isempty(iters_per_site)
                iters_per_site = copy(step_res.iters_per_site)
            else
                iters_per_site .+= step_res.iters_per_site
            end

            if compute_cond
                # For baseline, local_conds contains the standard condition numbers
                if isempty(conds_per_site)
                    conds_per_site = step_res.local_conds
                else
                    conds_per_site .+= step_res.local_conds
                end
                push!(avg_conds, step_res.local_cond_mean)
            end

            GC.gc(true)
        end

        result_dict = Dict{String,Any}(
            "key" => "baseline",
            "nodes" => task.nodes,
            "compute_cond" => compute_cond,
            "total_solve_time" => sum(solve_times),
            "solve_times" => solve_times,
            "iters" => iters,
            "iters_per_site" => iters_per_site,
            "residuals" => residuals,
            "solution_maxdims" => solution_maxdims,
            "div_norms_before" => div_norms_before,
            "div_norms_after" => div_norms_after,
            "div_ratios" => div_ratios,
            "precond_build_time" => 0.0,
            "precond_build_compile_time" => 0.0,
            "precond_build_gc_time" => 0.0,
            "compile_times" => compile_times,
            "gc_times" => gc_times,
            "total_compile_time" => sum(compile_times),
            "total_gc_time" => sum(gc_times),
            "hostname" => gethostname(),
            "slurm_job_id" => get(ENV, "SLURM_JOB_ID", ""),
            "blas_threads" => LinearAlgebra.BLAS.get_num_threads(),
            "julia_threads" => Threads.nthreads(),
            "n_steps_run" => N_STEPS,
            "warmup_time" => warmup_time,
            "gc_ballast_bytes" => length(GC_BALLAST),
            "heap_size_hint" => Int(Base.JLOptions().heap_size_hint),
        )
        add_last_snapshot!(result_dict, grid, N_STEPS, global_rep)

        if compute_cond
            conds_per_site ./= N_STEPS
            result_dict["conds_per_site"] = conds_per_site
            result_dict["avg_cond"] = mean(avg_conds)
        end

        return result_dict
    else
        μ = task.mu
        gdim = task.gdim
        @info "Running (μ=$μ, G_maxdim=$gdim) for $(task.nodes)"
        flush(stderr)

        fai_sweep_kwargs = SweepingArgs(; nsweeps=30, maxdim=gdim, cutoff=1e-12)

        fai_updater_kwargs = LocalUpdaterArgs(; ishermitian=true, isposdef=true, rtol=1e-8, krylovdim=1, maxiter=3000)

        fai_resals_solver = ResidualALS(;
            sweep_kwargs=fai_sweep_kwargs, updater_kwargs=fai_updater_kwargs, residual_tol=1e-9
        )

        fai_als_solver = ALS(; sweep_kwargs=fai_sweep_kwargs, updater_kwargs=fai_updater_kwargs)

        fai_amen_solver = AMEn(; sweep_kwargs=fai_sweep_kwargs, updater_kwargs=fai_updater_kwargs, enrichment_maxdim=1)

        fai_builder_solver = fai_als_solver

        # The driver already seeded with solver_seed(global_rep); re-seed right before the build so
        # nothing consumed in between (grid construction) can change the preconditioner.
        Random.seed!(solver_seed(global_rep))
        GC.gc(true)
        build_stats = @timed begin
            precond = fai_preconditioner(
                grid.A;
                μ=μ,
                solver=fai_builder_solver,
                D_cutoff=1e-11,
                D_oversample=20,
                D_maxdim=gdim,
                totalmaxdim=gdim,
                preconditioning=false,
                residual_tol=faibuild_tol,
            )
        end
        t_build = build_stats.time

        solve_times = Float64[]
        compile_times = Float64[]
        gc_times = Float64[]
        iters = Vector{Float64}[]
        residuals = Float64[]
        solution_maxdims = Int[]
        div_norms_before = Float64[]
        div_norms_after = Float64[]
        div_ratios = Float64[]
        iters_per_site = Float64[]
        pc_conds_per_site = Float64[]
        avg_pc_conds = Float64[]

        for i in 1:N_STEPS
            @info "Step $i / $N_STEPS"
            flush(stderr)

            step_res = chorin_step!(grid, sweep_kwargs, compute_cond, tol_val, i, precond; global_rep)
            push!(solve_times, step_res.t)
            push!(compile_times, step_res.compile_t)
            push!(gc_times, step_res.gc_t)
            push!(residuals, step_res.rel_res)
            push!(solution_maxdims, step_res.sol_maxdim)
            push!(div_norms_before, step_res.div_norm_before)
            push!(div_norms_after, step_res.div_norm_after)
            push!(div_ratios, step_res.div_ratio)
            push!(iters, step_res.iters_per_site)

            if isempty(iters_per_site)
                iters_per_site = copy(step_res.iters_per_site)
            else
                iters_per_site .+= step_res.iters_per_site
            end

            if compute_cond
                if isempty(pc_conds_per_site)
                    pc_conds_per_site = step_res.local_conds
                else
                    pc_conds_per_site .+= step_res.local_conds
                end
                push!(avg_pc_conds, step_res.local_cond_mean)
            end

            GC.gc(true)
        end

        result_dict = Dict{String,Any}(
            "key" => "mu_$(μ)_gdim_$(gdim)",
            "nodes" => task.nodes,
            "compute_cond" => compute_cond,
            "mu" => μ,
            "G_maxdim" => gdim,
            "precond_build_time" => t_build,
            "precond_build_compile_time" => build_stats.compile_time,
            "precond_build_gc_time" => build_stats.gctime,
            "total_solve_time" => sum(solve_times),
            "solve_times" => solve_times,
            "iters" => iters,
            "iters_per_site" => iters_per_site,
            "residuals" => residuals,
            "solution_maxdims" => solution_maxdims,
            "div_norms_before" => div_norms_before,
            "div_norms_after" => div_norms_after,
            "div_ratios" => div_ratios,
            "compile_times" => compile_times,
            "gc_times" => gc_times,
            "total_compile_time" => sum(compile_times),
            "total_gc_time" => sum(gc_times),
            "hostname" => gethostname(),
            "slurm_job_id" => get(ENV, "SLURM_JOB_ID", ""),
            "blas_threads" => LinearAlgebra.BLAS.get_num_threads(),
            "julia_threads" => Threads.nthreads(),
            "n_steps_run" => N_STEPS,
            "warmup_time" => warmup_time,
            "gc_ballast_bytes" => length(GC_BALLAST),
            "heap_size_hint" => Int(Base.JLOptions().heap_size_hint),
        )
        add_last_snapshot!(result_dict, grid, N_STEPS, global_rep)

        if compute_cond
            pc_conds_per_site ./= N_STEPS
            result_dict["pc_conds_per_site"] = pc_conds_per_site
            result_dict["avg_pc_cond"] = mean(avg_pc_conds)
        end

        return result_dict
    end
end

"""
    run_task(task, task_id; warmup_time)

Run `BENCH_N_REPEATS` independent repeats of `task`, one result file per repeat. `run_single`
rebuilds everything, so the loop body does not depend on previous iterations.
"""
function run_task(task::TaskSpec, task_id::Int; warmup_time::Float64=0.0)
    return run_repeats(task_id, TASKS_DIR) do global_rep
        run_single(task, global_rep; warmup_time=warmup_time)
    end
end

function warmup()
    @info "Running JIT warmup..."
    flush(stderr)
    t = @elapsed begin
        dummy_grid = build_grid((8, 8), 1)
        dummy_updater_kwargs = LocalUpdaterArgs(ishermitian=true, isposdef=true, rtol=1e-2, maxiter=2, krylovdim=1)
        dummy_als_solver = ALS(
            sweep_kwargs=SweepingArgs(nsweeps=2, maxdim=10, cutoff=1e-8), updater_kwargs=dummy_updater_kwargs
        )
        # Pass residual_tol like the timed build does: it adds a ResidualObserver to the FAI solves,
        # a different code path that would otherwise be compiled inside the timed build.
        dummy_fai = fai_preconditioner(
            dummy_grid.A;
            μ=1.0,
            solver=dummy_als_solver,
            D_cutoff=1e-11,
            D_oversample=2,
            D_maxdim=2,
            totalmaxdim=2,
            preconditioning=false,
            residual_tol=fai_tol_map[(8, 8)],
        )

        # When running with condition numbers, compute time does not really count, so it is better to avoid
        # crashing at precompilation due to dummy preconditioners than to have more accurate timings due to precompilation
        chorin_step!(dummy_grid, SweepingArgs(nsweeps=2, maxdim=10, cutoff=1e-8), false, 1e-3, 0, nothing)
        chorin_step!(dummy_grid, SweepingArgs(nsweeps=2, maxdim=10, cutoff=1e-8), false, 1e-3, 0, dummy_fai)
    end
    @info "JIT warmup complete" seconds=t
    flush(stderr)
    return t
end

# Block layout: a block is every task of one system size at one repeat index (see ../stats_harness.jl).
task_group(task::TaskSpec) = task.nodes

# System sizes this array covers in the block and unit layouts, e.g. CHORIN_STATS_SIZES="10x10"
# (default: all of `system_sizes`). submit.sh runs (8,8) as blocks and (10,10) as units, in two
# arrays. Task ids and group indices always refer to the full task list, so they are the same
# whichever sizes an array covers.
function selected_group_ids(groups::Vector{Vector{Int}}, all_tasks::Vector{TaskSpec})
    spec = get(ENV, "CHORIN_STATS_SIZES", "")
    isempty(spec) && return collect(eachindex(groups))
    sizes = [Tuple(parse.(Int, split(s, "x"))) for s in split(spec, ",")]
    group_nodes = [all_tasks[first(g)].nodes for g in groups]
    for s in sizes
        s in group_nodes || error("CHORIN_STATS_SIZES: $s is not in system_sizes = $system_sizes")
    end
    return [i for i in eachindex(groups) if group_nodes[i] in sizes]
end

function main()
    array_id = length(ARGS) > 0 ? parse(Int, ARGS[1]) : 1
    all_tasks = build_task_list()
    groups = task_groups(task_group.(all_tasks))
    group_ids = selected_group_ids(groups, all_tasks)

    if LAYOUT == :task
        task_id = array_id
        (1 <= task_id <= length(all_tasks)) || error("Invalid task_id=$task_id")
        # The global JIT warmup runs once per job, before the repeat loop, not once per repeat.
        warmup_time = warmup()
        res = run_task(all_tasks[task_id], task_id; warmup_time=warmup_time)
    elseif LAYOUT == :unit
        task_id, global_rep = decode_unit(array_id, reduce(vcat, groups[group_ids]))
        @info "Unit" array_id task_id nodes=all_tasks[task_id].nodes global_rep
        flush(stderr)
        warmup_time = warmup()
        res = run_unit(task_id, global_rep, TASKS_DIR) do
            run_single(all_tasks[task_id], global_rep; warmup_time=warmup_time)
        end
    else
        plan = block_plan(array_id, groups; group_ids)
        @info "Block" array_id group=plan.group nodes=all_tasks[first(plan.task_ids)].nodes global_rep=plan.global_rep order=plan.task_ids
        flush(stderr)
        warmup_time = warmup()
        # run_single rebuilds the grid on every call (chorin_step! mutates it), so the units of a
        # block are independent of each other and of their order.
        res = run_block(plan, TASKS_DIR) do task_id
            run_single(all_tasks[task_id], plan.global_rep; warmup_time=warmup_time)
        end
    end
    res.failed > 0 && exit(1)   # surfaces as a FAILED array task in SLURM; finished units are kept
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end

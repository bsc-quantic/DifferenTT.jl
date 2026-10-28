"""
    TT anisotropy benchmark, statistics variant (paper Table 2 and Figs. 9-10): worker

We solve the anisotropic diffusion equation -∇⋅(K∇u) = 1 on (0, 1)ᴺ with homogeneous Dirichlet conditions, where K =
(1-ρ) I + ρ 𝟏𝟏ᵀ is the equicorrelated diffusion tensor, in the non-quantized TT format with 15 nodes per dimension (Sec.
5.1 of the paper; the problem is defined in `common/anisotropy_problem.jl`). We sweep the floor F = κ(L₀⁻¹ L) of the
preconditioners built on the separable Laplacian L₀ = -Δ instead of the coupling ρ, with ρ = (F-1)/(F+N-1), so that the
dimensions are compared at equal difficulty. This is the benchmark of `fai_benchmarks/tt_anisotropy`, repeated: every
task is run `BENCH_N_REPEATS` times (`common/stats_harness.jl`), and the worker writes one result file per task and
repeat, so that the plotter can draw the spread of the times and the speedups paired by repeat.

# Problem and parameters

Each task is one combination of dimension, floor and preconditioner:

- N ∈ {10, 20} and F ∈ {1, 2, 4, 10, 30, 100} (F = 1 is the Poisson equation);
- no preconditioner (baseline); Sinc on L₀ with M ∈ {3, 4, 5, 7, 10}; FAI on L with μ = ‖L‖_∞ (Eq. (32) of the paper)
  and χ_G ∈ {4, 6, 8, 10}; and, as a control, FAI on L₀ with μ = ‖L₀‖_∞ and χ_G = 10 (11 tasks per (N, F), 132 in
  total, `NTASKS` of `submit.sh`).

We solve with `ResidualALS` (the "MALS" of the paper, local solves preconditioned at every site) to a relative
residual of 1e-5, with `cutoff = 1e-13`, at most 3000 local iterations, 30 sweeps and a solution bond dimension of at
most 150 for N = 10, and 50 sweeps and 300 for N = 20. The right-hand side is f ≡ 1 (`TT_ANISOTROPY_RHS` selects
`ones`, `random` or `manufactured`). The FAI is built with the two-site ALS solver (10 sweeps,
`TT_ANISOTROPY_STATS_FAI_NSWEEPS`, stopped at a relative residual of 0.1 of Eq. (21)), after a warm-up that compiles
every code path outside the timed region. The problem is built once per job under the seed `task.N`, so every repeat
solves the same system; what varies per repeat is the initial guess and the randomized build of the FAI.

The setups above are those of the paper repository, with the μ of the paper (see "Penalty μ of the FAI build" below).
The figures of the paper come from a run of this worker at the tag `paper-ttfai-v0.1-data` (`paper/README.md`). Two
changes of the library since the data of the earlier drafts of the paper (tag `paper-ttfai-data`) make new runs not
comparable with those: the FAI now solves the correct (not the transposed) FSAI problem, and the Frobenius floor of
`ResidualALS` uses `cutoff` where it used `cutoff²`, so the bond dimensions and the solutions at `cutoff > 0` differ
(see `NEWS.md`).

# Inputs and outputs

With the default layout (`BENCH_LAYOUT=block`), `ARGS[1]` is a block index and the job runs every method of one (N, F)
case at one repeat index, in a seeded random order and on one problem, so that the speedups compare runs on the same
node; array indices run over the 12 cases first and the repeats next. With `BENCH_LAYOUT=task`, `ARGS[1]` is a task id
and the job runs all the repeats of that task. `nocond` is `ARGS[2]`: without it, we estimate the local condition
numbers κ(A) and κ(P⁻¹A) of every site. The worker writes `tasks/task_NNN_rep_RRR.jld2` (key `"result"`) under
`results_mugersh/` or `results_nocond_mugersh/` of this folder (`results/` or `results_nocond/` with
`FAI_MU_RULE=fixed`), or under `TT_ANISOTROPY_STATS_RESULTS_DIR` if set. The other
configuration variables (`BENCH_N_REPEATS`, `BENCH_REPEAT_OFFSET`, `BENCH_BASE_SEED`, `BENCH_TIME_BUDGET_S`) are
described in `common/stats_harness.jl` and `submit.sh`. FAI results also store the iterations of the local solves of the
build (`precond_build_total_iters`, `precond_build_iters_per_site`, summed over sweeps, and `precond_build_nsweeps`).

# Penalty μ of the FAI build

`FAI_MU_RULE` selects μ (`common/mu_rules.jl`). With `gershgorin`, the default and the rule of the paper, μ = ‖A‖_∞ of
the operator the FAI is built on, bounded at run time from the stencil coefficients of its 1D factors:
N (4 + ρ (N - 1)) / h² for the FAI on L (Eq. (32) of the paper), and 4N / h² for the FAI on L₀ (ρ = 0), times
`FAI_MU_SCALE` (default 1.0). The μ of the task list (`fai_configs`, `fai_L0_configs`) is then only a placeholder. A
rule-based run writes under `results_mugersh/` or `results_nocond_mugersh/` (`_mugersh_x<scale>` with a scale), with
the keys `fai_mu_gersh_gdim_χ` and `fai_L0_mu_gersh_gdim_χ`, and stores the numeric μ in `"mu"` and the rule in
`"mu_rule"` and `"mu_scale"`; `collect.jl` and `plot_paper.jl` read the same variables to find the folder. With `fixed`,
the FAI uses the μ = 10⁷ of the task list, as in the earlier drafts of the paper, and writes under `results/` or
`results_nocond/` with the keys `fai_mu_1.0e7_gdim_χ`.

Smoke mode (`DIFFERENTT_SMOKE=1`) shrinks the sweep to N = 5, F = 10 and 7 nodes per dimension, with the baseline,
Sinc (M = 3), FAI and FAI on L₀ at χ_G = 4, through the `SMOKE` branches next to the constants, and writes under
`DIFFERENTT_SMOKE_DIR` (a fresh temporary folder if it is unset), so that the committed data are never touched.
`experiments/smoke.jl` runs two blocks, the collector and the plotter.

# How to run

From the root of the repository:

    DIFFERENTT_SMOKE=1 julia --project=experiments experiments/paper/tt_anisotropy_stats/worker.jl 1 nocond
    julia --project=experiments experiments/paper/tt_anisotropy_stats/worker.jl 1 nocond
    bash experiments/paper/tt_anisotropy_stats/submit.sh nocond
    FAI_MU_RULE=fixed bash experiments/paper/tt_anisotropy_stats/submit.sh nocond   # μ = 10⁷, earlier drafts

# Resources

The job script requests an array of 12 × `BENCH_N_REPEATS` blocks (120 for the default 10 repeats), each on one node
with 20 CPUs, for up to 2 days (`--time=02-00:00:00`), without a `--mem` directive; the worker sets Julia's heap-size
hint to 80 % of the memory of the allocation, and `GC_BALLAST_GB` controls the GC ballast (`common/gc_ballast.jl`; the
job script sets 7). The comments of `submit.sh` give the time of each block as measured in the run of October 2026.

# Paper

Sec. 5.1: Table 2 (`tab:anisotropy_params`) lists the parameters of the runs, which are the constants of this worker,
and Figs. 9-10 (`fig:mals_iters`, `fig:mals_site_iters`) are drawn by `plot_paper.jl` from the output of `collect.jl`.
"""

using ITensors
using ITensorMPS
using LinearAlgebra
using JLD2
using TensorCrossInterpolation
using DifferenTT
using Random

include(joinpath(@__DIR__, "..", "..", "common", "anisotropy_problem.jl"))
include(joinpath(@__DIR__, "..", "..", "common", "mu_rules.jl"))

# ---- Runtime configuration ----
# Match BLAS threads to the SLURM allocation (falls back to Julia's default if not under SLURM).
if haskey(ENV, "SLURM_CPUS_PER_TASK")
    LinearAlgebra.BLAS.set_num_threads(parse(Int, ENV["SLURM_CPUS_PER_TASK"]))
end
@info "Runtime" blas_threads=LinearAlgebra.BLAS.get_num_threads() julia_threads=Threads.nthreads() host=gethostname()

# ---- GC memory ballast (see common/gc_ballast.jl) ----
include(joinpath(@__DIR__, "..", "..", "common", "gc_ballast.jl"))
flush(stderr)

include(joinpath(@__DIR__, "..", "..", "common", "stats_harness.jl"))

include(joinpath(@__DIR__, "..", "..", "common", "smoke.jl"))
# Smoke mode writes under DIFFERENTT_SMOKE_DIR (a fresh temporary folder if unset) instead of this folder.
const OUT_ROOT = output_root(@__DIR__; fresh=true)

const use_nocond = length(ARGS) > 1 && ARGS[2] == "nocond"
# Penalty μ of the FAI build: "fixed" (the μ of fai_configs) or a rule evaluated per problem (common/mu_rules.jl)
const MU_RULE, MU_SCALE = fai_mu_rule()
const default_base_folder = joinpath(
    OUT_ROOT, mu_results_folder(use_nocond ? "results_nocond" : "results", MU_RULE, MU_SCALE)
)
base_folder = get(ENV, "TT_ANISOTROPY_STATS_RESULTS_DIR", default_base_folder)
const TASKS_DIR = joinpath(base_folder, "tasks")
mkpath(TASKS_DIR)

# Timed solves per repeat. The original runs a cold and a warm solve inside one task to separate
# JIT cost from solve cost; here the outer repeat loop does that job, and keeping both would double
# the cost and make `solve_time` ambiguous. With a single solve the `*_cold` fields equal the
# reported values and are kept only so the collector and plot code stay schema-compatible. Repeat 1
# of every process can still carry residual JIT cost (see BENCH_DISCARD_FIRST_REPEAT in collect.jl).
const N_SOLVE_REPEATS = 1

# Number of sweeps used to build the FAI preconditioner. Overridable for fast local verification.
const FAI_BUILD_NSWEEPS = parse(Int, get(ENV, "TT_ANISOTROPY_STATS_FAI_NSWEEPS", SMOKE ? "3" : "10"))

# Right-hand side: :ones (mean exit time, default), :random or :manufactured. See build_rhs.
const RHS_KIND = Symbol(get(ENV, "TT_ANISOTROPY_RHS", "ones"))

# ---- Benchmark parameters ----
# Sweeping over the floor F = κ(L0⁻¹ L) of separable preconditioners, for a few dimensions N.
# The number of tasks is length(system_dims) * length(floor_values) *
# (1 + length(sinc_M_values) + length(fai_configs) + length(fai_L0_configs)); keep NTASKS in
# submit.sh in sync.
# Smoke mode: N = 5, F = 10, 7 nodes per dimension, and baseline, Sinc (M = 3), FAI and FAI(L0) at χ_G = 4 (one block).
const system_dims = SMOKE ? [5] : [10, 20]
const floor_values = SMOKE ? [10.0] : [1.0, 2.0, 4.0, 10.0, 30.0, 100.0]
const N_NODES_PER_DIM = SMOKE ? 7 : 15
const residual_tol = 1e-5

# Cap on the solution bond dimension. The f ≡ 1 solution rank grows with F: at N = 5 a relative
# residual of 1e-7 needs χ ≈ 70 (F = 10) and χ ≈ 90 (F = 100), see anisotropy_floor/.
const maxdims_map = Dict(5 => 80, 10 => 150, 15 => 150, 20 => 300)

const nsweeps_map = SMOKE ? Dict(5 => 5) : Dict(5 => 30, 10 => 30, 15 => 40, 20 => 50)

const pc_tol_map = Dict(5 => 1e-1, 10 => 1e-1, 15 => 1e-1, 20 => 1e-1)

# Preconditioners: Baseline, Sinc-Quadrature (M sweep, built on L0 = -Δ), FAI on L (mu and gdim
# sweep) and FAI on L0 (control: shows whether FAI's advantage comes from capturing the coupling).
# The mu of these configurations is used only with FAI_MU_RULE=fixed; the default rule replaces it (common/mu_rules.jl).
const sinc_M_values = SMOKE ? [3] : [3, 4, 5, 7, 10]
const fai_configs = SMOKE ? [(1e7, 4)] : [(1e7, 4), (1e7, 6), (1e7, 8), (1e7, 10)]
const fai_L0_configs = SMOKE ? [(1e7, 4)] : [(1e7, 10)]

ITensors.disable_warn_order()
ENV["JULIA_DEBUG"] = ""

struct TaskSpec
    N::Int
    F::Float64
    method::String
    param1::Float64 # M for Sinc, mu for FAI (a placeholder under a μ rule)
    param2::Int # not used for Sinc, gdim for FAI
end

function build_task_list()
    tasks = TaskSpec[]
    for N in system_dims
        for F in floor_values
            # Baseline
            push!(tasks, TaskSpec(N, F, "baseline", 0.0, 0))

            # Sinc Quadrature
            for M in sinc_M_values
                push!(tasks, TaskSpec(N, F, "sinc", Float64(M), 0))
            end

            # FAI on L
            for (μ, gdim) in fai_configs
                push!(tasks, TaskSpec(N, F, "fai", μ, gdim))
            end

            # FAI on L0 (control)
            for (μ, gdim) in fai_L0_configs
                push!(tasks, TaskSpec(N, F, "fai_L0", μ, gdim))
            end
        end
    end
    return tasks
end

function problem(N::Int, F::Real)
    build_problem(N, F; nodes_per_dim=N_NODES_PER_DIM, rhs_kind=RHS_KIND, sol_maxdim=get(maxdims_map, N, 10))
end

"""
    make_outer_solver(A, rhs; nsweeps, maxdim)

Build the outer ResidualALS solver and its stat-tracking stores. Used both by the JIT `warmup()`
and by every timed task, with identical argument *types* (same `observer!` type, same store
types), so no keyword-method compilation is deferred into the timed solve.
"""
function make_outer_solver(A::MPO, rhs::MPS; nsweeps::Int, maxdim::Int)
    residual_observer = ResidualObserver(A, rhs, residual_tol)
    sweep_kwargs = SweepingArgs(; nsweeps=nsweeps, maxdim=maxdim, cutoff=1e-13, (observer!)=residual_observer)

    numiter_store = KrylovSweepStats()
    cond_store = use_nocond ? nothing : KrylovCondStats()
    pc_cond_store = use_nocond ? nothing : KrylovCondStats()

    updater_kwargs = LocalUpdaterArgs(;
        ishermitian=true,
        isposdef=true,
        rtol=0.1*residual_tol,
        krylovdim=1,
        maxiter=3000,
        numiter_store=numiter_store,
        cond_store=cond_store,
        pc_cond_store=pc_cond_store,
    )

    solver = ResidualALS(; sweep_kwargs=sweep_kwargs, updater_kwargs=updater_kwargs, residual_tol=0.5*residual_tol)
    return solver, (; numiter_store, cond_store, pc_cond_store)
end

"""
    build_fai(A, μ, gdim; nsweeps, residual_tol_build, numiter_store)

Build the FAI factorized preconditioner. Used both by `warmup()` and by every timed FAI task,
with the same keyword set as the real build (including `residual_tol` and `numiter_store`), so the
`sweep_observer!` path inside the FAI build is compiled ahead of time. `numiter_store` records the
iterations of the local solves of the build.
"""
function build_fai(
    A::MPO,
    μ::Real,
    gdim::Int;
    nsweeps::Int,
    residual_tol_build::Float64,
    numiter_store::KrylovSweepStats=KrylovSweepStats(),
)
    fai_sweep_kwargs = SweepingArgs(; nsweeps=nsweeps, maxdim=gdim, cutoff=1e-12)

    fai_updater_kwargs = LocalUpdaterArgs(;
        ishermitian=true, isposdef=true, rtol=1e-5, krylovdim=1, maxiter=2000, numiter_store=numiter_store
    )

    fai_als_solver = ALS(; sweep_kwargs=fai_sweep_kwargs, updater_kwargs=fai_updater_kwargs)

    fai_amen_solver = AMEn(;
        sweep_kwargs=fai_sweep_kwargs, updater_kwargs=fai_updater_kwargs, enrichment_maxdim=2, enrichment_cutoff=1e-10
    )

    return fai_preconditioner(
        A;
        μ=μ,
        solver=fai_als_solver,
        D_cutoff=1e-11,
        D_oversample=20,
        D_maxdim=gdim,
        totalmaxdim=gdim,
        preconditioning=false,
        residual_tol=residual_tol_build,
    )
end

"""
Summarize a `KrylovSweepStats` store: per-site iteration totals, grand total, and sweep count.
"""
function krylov_iters_summary(stats_store::KrylovSweepStats)
    per_site = DifferenTT.positions_sum(stats_store)
    nsweeps = isempty(stats_store.numiters) ? 0 : maximum(length, stats_store.numiters)
    return (; iters_per_site=per_site, total_iters=sum(per_site; init=0.0), nsweeps)
end

"""
    repeated_solves(A, rhs, init_mps, N, solve_fn)

Run the outer solve `N_SOLVE_REPEATS` times, each with a freshly built solver/stores and a
`GC.gc(true)` right before timing. Returns a `Vector{NamedTuple}`, one entry per repeat, holding
the solution, timings (`time`, `compile_time`, `gc_time`), iteration/sweep counts, and the stores
(needed afterwards for condition-number extraction).
"""
function repeated_solves(A::MPO, rhs::MPS, init_mps::MPS, N::Int, solve_fn)
    runs = NamedTuple[]
    for rep in 1:N_SOLVE_REPEATS
        solver, stores = make_outer_solver(A, rhs; nsweeps=nsweeps_map[N], maxdim=maxdims_map[N])
        x0 = copy(init_mps)
        GC.gc(true)
        stats = @timed solve_fn(solver, x0)
        numiters = krylov_iters_summary(stores.numiter_store)
        push!(
            runs,
            (;
                out_mps=stats.value,
                time=stats.time,
                compile_time=stats.compile_time,
                gc_time=stats.gctime,
                stores=stores,
                numiters...,
            ),
        )
        @info "solve repeat $rep" time=stats.time compile=stats.compile_time gc=stats.gctime total_iters=numiters.total_iters nsweeps=numiters.nsweeps
    end
    return runs
end

"""
    run_single(task, prob, global_rep; warmup_time)

One repeat of a task on an already built problem. The caller (`run_repeats`) has seeded the global
RNG with `solver_seed(global_rep)`, and the very first RNG consumer here is the initial guess, so
it is identical across methods at the same repeat (which is what pairs the speedup ratios). The
FAI build then continues the same stream.
"""
function run_single(task::TaskSpec, prob, global_rep::Int; warmup_time::Float64=0.0)
    N = task.N
    F = task.F
    A, rhs = prob.A, prob.rhs

    sites = siteinds(rhs)
    init_mps = random_mps(sites)

    method = task.method
    key = ""
    extra = Dict{String,Any}()
    pc = nothing
    t_build = 0.0
    build_compile = 0.0
    build_gc = 0.0

    if method == "baseline"
        key = "baseline"
        @info "Running Baseline N=$N F=$F"

    elseif method == "sinc"
        M = Int(task.param1)
        key = "sinc_M_$(M)"
        @info "Running Sinc (M=$M) N=$N F=$F"

        GC.gc(true)
        b = @timed sincquadrature_preconditioner(prob.arch, sites, prob.A_list, M)
        pc = b.value
        t_build = b.time
        build_compile = b.compile_time
        build_gc = b.gctime
        extra["M"] = M
        extra["precond_target"] = "L0"

    elseif method == "fai" || method == "fai_L0"
        gdim = task.param2
        # fai_L0 approximates the inverse of the separable L0 = -Δ (ρ = 0) but preconditions A.
        target = method == "fai" ? A : prob.A0
        ρ_target = method == "fai" ? prob.ρ : 0.0
        μ = resolve_fai_mu(
            MU_RULE, MU_SCALE, task.param1, () -> anisotropy_mu_gershgorin(prob.grid, ρ_target; stencil_order=2)
        )
        key = "$(method)_$(mu_key(MU_RULE, MU_SCALE, μ))_gdim_$(gdim)"
        @info "Running $(method) (μ=$μ, G_maxdim=$gdim) N=$N F=$F" mu_rule=MU_RULE mu_scale=MU_SCALE mu_over_1e7=μ / 1e7

        build_store = KrylovSweepStats()
        GC.gc(true)
        b = @timed build_fai(
            target, μ, gdim; nsweeps=FAI_BUILD_NSWEEPS, residual_tol_build=pc_tol_map[N], numiter_store=build_store
        )
        pc = b.value
        t_build = b.time
        build_compile = b.compile_time
        build_gc = b.gctime
        build_iters = krylov_iters_summary(build_store)
        @info "FAI build" total_iters=build_iters.total_iters nsweeps=build_iters.nsweeps
        extra["mu"] = μ
        extra["mu_rule"] = MU_RULE
        extra["mu_scale"] = MU_SCALE
        extra["precond_build_total_iters"] = build_iters.total_iters
        extra["precond_build_iters_per_site"] = build_iters.iters_per_site
        extra["precond_build_nsweeps"] = build_iters.nsweeps
        extra["G_maxdim"] = gdim
        extra["precond_target"] = method == "fai" ? "L" : "L0"

    else
        error("Unknown task method: $method")
    end

    solve_fn = isnothing(pc) ? (s, x0) -> linearsolve(s, A, rhs, x0) : (s, x0) -> linearsolve(s, A, rhs, pc, x0)

    runs = repeated_solves(A, rhs, init_mps, N, solve_fn)
    cold = first(runs)
    warm = last(runs)

    out_mps = warm.out_mps

    # Compute relative residual using inner product to save memory
    term1 = ITensorMPS.inner(A, out_mps, A, out_mps)
    term2 = 2 * real(ITensorMPS.inner(rhs', A, out_mps))
    term3 = ITensorMPS.inner(rhs, rhs)
    rel_residual = sqrt(max(0.0, term1 - term2 + term3)) / norm(rhs)

    @info "Residual $rel_residual"
    @info "Max bond dimension = $(maxlinkdim(out_mps))"

    result_dict = Dict{String,Any}(
        "key" => key,
        "N" => N,
        "F" => F,
        "rho" => prob.ρ,
        "floor_analytic" => floor_from_rho(prob.ρ, N),
        "rhs_kind" => String(RHS_KIND),
        # The problem is built once per job; these let the collector check it is identical across
        # repeats (and across jobs, if the repeats were chunked).
        "rhs_norm" => norm(rhs),
        "rhs_maxlinkdim" => maxlinkdim(rhs),
        "method" => method,
        "precond_build_time" => t_build,
        "solve_time" => warm.time,
        "total_time" => t_build + warm.time,
        "relative_residual" => rel_residual,
        "iters_per_site" => warm.iters_per_site,
        "solution_maxdim" => maxlinkdim(out_mps),
        "solution_hit_maxdim" => maxlinkdim(out_mps) >= maxdims_map[N],

        # Diagnostics: cold (first) vs warm (last, reported above) timed solve, to expose any
        # remaining compilation/GC bias directly in the data.
        "solve_time_cold" => cold.time,
        "compile_time_cold" => cold.compile_time,
        "gc_time_cold" => cold.gc_time,
        "total_iters_cold" => cold.total_iters,
        "compile_time" => warm.compile_time,
        "gc_time" => warm.gc_time,
        "total_iters" => warm.total_iters,
        "nsweeps" => warm.nsweeps,
        "n_solve_repeats" => N_SOLVE_REPEATS,
        "precond_build_compile_time" => build_compile,
        "precond_build_gc_time" => build_gc,
        "hostname" => gethostname(),
        "slurm_job_id" => get(ENV, "SLURM_JOB_ID", ""),
        "blas_threads" => LinearAlgebra.BLAS.get_num_threads(),
        "julia_threads" => Threads.nthreads(),
        "warmup_time" => warmup_time,
        "gc_ballast_bytes" => length(GC_BALLAST),
        "heap_size_hint" => Int(Base.JLOptions().heap_size_hint),
    )

    if !isnothing(prob.sol)
        result_dict["relative_error"] = relative_distance(out_mps, prob.sol)
    end

    if !isnothing(pc)
        result_dict["precond_linkdims"] = collect(linkdims(pc.operator))
        result_dict["precond_maxlinkdim"] = maxlinkdim(pc.operator)
    end

    for (k, v) in extra
        result_dict[k] = v
    end

    result_dict["compute_cond"] = !use_nocond
    if !use_nocond
        cond_store = warm.stores.cond_store
        conds_per_site = positions_mean_cond(cond_store)
        avg_cond = length(conds_per_site) > 0 ? sum(conds_per_site) / length(conds_per_site) : NaN
        result_dict["conds_per_site"] = conds_per_site
        result_dict["avg_cond"] = avg_cond

        if !isnothing(pc)
            pc_cond_store = warm.stores.pc_cond_store
            pc_conds_per_site = positions_mean_cond(pc_cond_store)
            avg_pc_cond = length(pc_conds_per_site) > 0 ? sum(pc_conds_per_site) / length(pc_conds_per_site) : NaN
            result_dict["pc_conds_per_site"] = pc_conds_per_site
            result_dict["avg_pc_cond"] = avg_pc_cond
        end
    end

    return result_dict
end

"""
    build_task_problem(task)

The (N, F) problem under the original problem seed `task.N`, so it is the same system in every
job, whichever repeat or block builds it. Never mutated by a solve.
"""
function build_task_problem(task::TaskSpec)
    N, F = task.N, task.F
    Random.seed!(task.N)    # problem seed, unchanged from the original worker
    @info "Building problem for N=$N, F=$F (rhs=$RHS_KIND)..."
    prob = problem(N, F)
    @info "Problem" rho=prob.ρ A_maxlinkdim=maxlinkdim(prob.A)
    return prob
end

"""
    run_task(task, task_id; warmup_time)

Build the problem once (under the original problem seed `task.N`), then run `BENCH_N_REPEATS`
repeats on it, one result file per repeat. `prob` is never mutated by a solve.
"""
function run_task(task::TaskSpec, task_id::Int; warmup_time::Float64=0.0)
    prob = build_task_problem(task)
    return run_repeats(task_id, TASKS_DIR) do global_rep
        @info "Problem check" norm_rhs=norm(prob.rhs) rhs_maxlinkdim=maxlinkdim(prob.rhs)
        run_single(task, prob, global_rep; warmup_time=warmup_time)
    end
end

"""
    warmup(N)

JIT-warm every code path that a timed task will exercise, using the *same* solver/FAI-builder
factories as the timed tasks (`make_outer_solver`, `build_fai`) so that argument-type-specific
compilation happens here rather than inside a timed solve. `dummy_N = 4` is the smallest chain with
an interior 2-site bond, and `dummy_F = 10` compiles the rank-3 (ρ ≠ 0) operator path.

`sincquadrature_preconditioner` is additionally compiled for the task's dimension `N`: it is
specialized on `TTArchitecture{N}`, so the `dummy_N` call does not cover it. `M` is only a value,
so a single call with `M = 1` compiles it for every `M`.
"""
function warmup(N::Int)
    @info "Running JIT warmup..."
    t = @elapsed begin
        dummy_N = 4
        dprob = build_problem(dummy_N, 10.0; nodes_per_dim=N_NODES_PER_DIM, rhs_kind=RHS_KIND, sol_maxdim=10)
        dA, drhs = dprob.A, dprob.rhs
        dsites = siteinds(drhs)
        dinit = random_mps(dsites)

        dsinc = sincquadrature_preconditioner(dprob.arch, dsites, dprob.A_list, 1)
        dfai = build_fai(dA, 1e7, 2; nsweeps=2, residual_tol_build=pc_tol_map[5], numiter_store=KrylovSweepStats())

        solve_fns = (
            (s, x0) -> linearsolve(s, dA, drhs, x0),
            (s, x0) -> linearsolve(s, dA, drhs, dsinc, x0),
            (s, x0) -> linearsolve(s, dA, drhs, dfai, x0),
        )
        for solve_fn in solve_fns
            solver, stores = make_outer_solver(dA, drhs; nsweeps=2, maxdim=10)
            @timed solve_fn(solver, copy(dinit))
            DifferenTT.positions_sum(stores.numiter_store)
        end

        # Warm the residual-computation path used after every solve.
        ITensorMPS.inner(dA, dinit, dA, dinit)

        # Sinc builder at the task's N. The builder only needs the architecture, sites and 1D
        # operators, so the full problem is not built here.
        arch = TTArchitecture(ntuple(_ -> N_NODES_PER_DIM, N))
        grid = Grid(arch, ntuple(_ -> 1.0, N))
        sincquadrature_preconditioner(arch, siteinds(arch), [get_1d_A(i, grid, arch) for i in 1:N], 1)
    end
    @info "JIT warmup complete" seconds=t
    return t
end

# Block layout: a block is every method at one (N, F) case and one repeat index (see
# ../stats_harness.jl).
task_group(task::TaskSpec) = (task.N, task.F)

function main()
    array_id = length(ARGS) > 0 ? parse(Int, ARGS[1]) : 1
    all_tasks = build_task_list()

    if LAYOUT == :task
        task_id = array_id
        if task_id < 1 || task_id > length(all_tasks)
            error("Invalid task_id=$task_id (there are $(length(all_tasks)) tasks)")
        end
        task = all_tasks[task_id]
        # The global JIT warmup runs once per job, before the repeat loop, not once per repeat.
        warmup_time = warmup(task.N)
        res = run_task(task, task_id; warmup_time=warmup_time)
    else
        plan = block_plan(array_id, task_groups(task_group.(all_tasks)))
        first_task = all_tasks[first(plan.task_ids)]
        @info "Block" array_id group=plan.group N=first_task.N F=first_task.F global_rep=plan.global_rep order=plan.task_ids
        flush(stderr)
        warmup_time = warmup(first_task.N)   # every task of a block has the same N
        # Built once and shared by every method of the block: they all solve the identical system.
        prob = build_task_problem(first_task)
        res = run_block(plan, TASKS_DIR) do task_id
            @info "Problem check" norm_rhs=norm(prob.rhs) rhs_maxlinkdim=maxlinkdim(prob.rhs)
            run_single(all_tasks[task_id], prob, plan.global_rep; warmup_time=warmup_time)
        end
    end
    res.failed > 0 && exit(1)   # surfaces as a FAILED array task in SLURM; finished units are kept
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end

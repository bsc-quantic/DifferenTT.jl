"""
    TT Anisotropy Benchmark (Global CG): Parallel Worker

Global (GlobalCG) solve of -∇⋅(K∇u) = f with the equicorrelated diffusion tensor
K = (1-ρ) I + ρ 𝟏𝟏ᵀ, sweeping the floor F of separable preconditioners instead of N.
Problem definition in ../anisotropy_problem.jl.
"""

using ITensors
using ITensorMPS
using LinearAlgebra
using JLD2
using TensorCrossInterpolation
using DifferenTT
using Random

include(joinpath(@__DIR__, "..", "anisotropy_problem.jl"))

# ---- Runtime configuration ----
# Match BLAS threads to the SLURM allocation (falls back to Julia's default if not under SLURM).
if haskey(ENV, "SLURM_CPUS_PER_TASK")
    LinearAlgebra.BLAS.set_num_threads(parse(Int, ENV["SLURM_CPUS_PER_TASK"]))
end
@info "Runtime" blas_threads=LinearAlgebra.BLAS.get_num_threads() julia_threads=Threads.nthreads() host=gethostname()

# ---- GC memory ballast ----
# Julia 1.12 derives its GC thresholds from the heap size, so a small live heap (as in this
# benchmark) collects very often and forces full sweeps, which are expensive. The ballast is a
# large, never-written buffer: the GC counts it as heap (raising the thresholds) but its pages
# are never touched, so it costs no resident memory. See chorin_gc_report.md.
const GC_BALLAST_GB = parse(Float64, get(ENV, "GC_BALLAST_GB", "0"))
const GIB = 2^30
gc_max_memory() = ccall(:jl_gc_get_max_memory, UInt64, ())

function allocate_ballast(gb::Float64)
    gb > 0 || return UInt8[]
    max_gb = 0.4 * 0.8 * gc_max_memory() / GIB   # keep well below the GC's heap cap
    if gb > max_gb
        @warn "Clamping GC ballast" requested_gb=gb max_gb
        gb = max_gb
    end
    try
        return Vector{UInt8}(undef, round(Int, gb * GIB))
    catch e
        e isa OutOfMemoryError || rethrow()
        @warn "Could not reserve the GC ballast, running without it" gb
        return UInt8[]
    end
end

# Must stay a global: if it is ever freed, the effect disappears. Never write into it.
const GC_BALLAST = allocate_ballast(GC_BALLAST_GB)
@info "GC ballast" gb=length(GC_BALLAST) / GIB heap_cap_gb=gc_max_memory() / GIB
flush(stderr)

const default_base_folder = "experiments/preconditioner_benchmarks/tt_anisotropy_global/results"
base_folder = get(ENV, "TT_ANISOTROPY_GLOBAL_RESULTS_DIR", default_base_folder)
const TASKS_DIR = joinpath(base_folder, "tasks")
mkpath(TASKS_DIR)

# Timed solves per task. Run 1 is "cold", the last run is "warm" and is what `solve_time`
# reports. This file has no cond/nocond distinction, so every task always repeats.
const N_SOLVE_REPEATS = 2

# Number of sweeps used to build the FAI preconditioner. Overridable for fast local verification.
const FAI_BUILD_NSWEEPS = parse(Int, get(ENV, "TT_ANISOTROPY_GLOBAL_FAI_NSWEEPS", "30"))

# Right-hand side: :ones (mean exit time, default), :random or :manufactured. See build_rhs.
const RHS_KIND = Symbol(get(ENV, "TT_ANISOTROPY_RHS", "ones"))

# ---- Benchmark parameters ----
# Sweeping over the floor F = κ(L0⁻¹ L) of separable preconditioners, for a few dimensions N.
# The number of tasks is length(system_dims) * length(floor_values) *
# (1 + length(sinc_M_values) + length(fai_configs) + length(fai_L0_configs)); keep NTASKS in
# submit.sh in sync.
const system_dims = [5, 10]
const floor_values = [1.0, 2.0, 4.0, 10.0, 30.0, 100.0]
const N_NODES_PER_DIM = 15
const residual_tol = 1e-3

# Cap on the solution bond dimension. The f ≡ 1 solution rank grows with F: at N = 5 a relative
# residual of 1e-7 needs χ ≈ 70 (F = 10) and χ ≈ 90 (F = 100), see anisotropy_floor/.
const maxdims_map = Dict(
    5 => 80,
    10 => 100,
    15 => 150,
    20 => 200
)

const maxiter_map = Dict(
    5 => 1000,
    10 => 1000,
    15 => 1000,
    20 => 1000
)

const pc_tol_map = Dict(
    5 => 1e-3,
    10 => 1e-3,
    15 => 1e-3,
    20 => 1e-3
)

# Preconditioners: Baseline, Sinc-Quadrature (M sweep, built on L0 = -Δ), FAI on L (mu and gdim
# sweep) and FAI on L0 (control: shows whether FAI's advantage comes from capturing the coupling).
const sinc_M_values = [3, 4, 5, 7, 10]
const fai_configs = [(1e7, 4), (1e7, 6), (1e7, 8), (1e7, 10)]
const fai_L0_configs = [(1e7, 10)]

ITensors.disable_warn_order()
ENV["JULIA_DEBUG"] = ""

struct TaskSpec
    N::Int
    F::Float64
    method::String
    param1::Float64 # M for Sinc, mu for FAI
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

problem(N::Int, F::Real) = build_problem(N, F; nodes_per_dim=N_NODES_PER_DIM, rhs_kind=RHS_KIND, sol_maxdim=get(maxdims_map, N, 10))

"""
    make_global_solver(; rtol, maxiter, maxdim, cutoff, orth_maxdim, orth_cutoff)

Build the outer GlobalCG solver and its iteration-count `Ref`. Used both by the JIT `warmup()`
and by every timed task, with identical argument *types*, so no keyword-method compilation is
deferred into the timed solve.
"""
function make_global_solver(; rtol::Float64, maxiter::Int, maxdim::Int, cutoff::Float64, orth_maxdim::Int, orth_cutoff::Float64)
    numiter_store = Ref{Int}(0)
    updater_kwargs = GlobalUpdaterArgs(
        ishermitian = true,
        isposdef = true,
        rtol = rtol,
        maxiter = maxiter,
        maxdim = maxdim,
        cutoff = cutoff,
        orth_maxdim = orth_maxdim,
        orth_cutoff = orth_cutoff,
        numiter_store = numiter_store,
    )
    return GlobalCG(updater_kwargs=updater_kwargs), numiter_store
end

"""
    build_fai(A, μ, gdim; nsweeps, residual_tol_build)

Build the FAI factorized preconditioner. Used both by `warmup()` and by every timed FAI task,
with the same keyword set as the real build (including `residual_tol`), so the
`sweep_observer!` path inside `fai_dg` is compiled ahead of time.
"""
function build_fai(A::MPO, μ::Real, gdim::Int; nsweeps::Int, residual_tol_build::Float64)
    fai_sweep_kwargs = SweepingArgs(nsweeps=nsweeps, maxdim=gdim, cutoff=1e-12)

    fai_updater_kwargs = LocalUpdaterArgs(
        ishermitian = true,
        isposdef = true,
        rtol = 1e-5,
        krylovdim = 1,
        maxiter = 2000
    )

    fai_als_solver = ALS(
        sweep_kwargs = fai_sweep_kwargs,
        updater_kwargs = fai_updater_kwargs,
    )

    D_kwargs_local = (; cutoff=1e-11, oversample=10, maxdim=gdim)

    return fai_preconditioner(A, μ, fai_als_solver, D_kwargs_local; totalmaxdim=gdim, preconditioning=false, residual_tol=residual_tol_build)
end

"""Run the outer GlobalCG solve N_SOLVE_REPEATS times with a fresh solver each time. Returns a
Vector{NamedTuple} — one entry per repeat, holding the solution, timings, and iteration count."""
function repeated_solves(A::MPO, rhs::MPS, init_mps::MPS, N::Int, solve_fn)
    runs = NamedTuple[]
    for rep in 1:N_SOLVE_REPEATS
        solver, numiter_store = make_global_solver(; rtol=residual_tol, maxiter=maxiter_map[N],
                                                     maxdim=maxdims_map[N], cutoff=1e-13,
                                                     orth_maxdim=maxdims_map[N], orth_cutoff=1e-14)
        x0 = copy(init_mps)
        # Same RNG state before every timed solve (all methods and repeats), so no random choice
        # inside the solver can differ between them.
        Random.seed!(N)
        GC.gc(true)
        stats = @timed solve_fn(solver, x0)
        push!(runs, (; out_mps = stats.value, time = stats.time, compile_time = stats.compile_time,
                       gc_time = stats.gctime, global_iters = numiter_store[]))
        @info "solve repeat $rep" time=stats.time compile=stats.compile_time gc=stats.gctime global_iters=numiter_store[]
    end
    return runs
end

function run_task(task::TaskSpec; warmup_time::Float64=0.0)
    N = task.N
    F = task.F
    Random.seed!(task.N)
    @info "Building problem for N=$N, F=$F (rhs=$RHS_KIND)..."
    prob = problem(N, F)
    A, rhs = prob.A, prob.rhs
    @info "Problem" rho=prob.ρ A_maxlinkdim=maxlinkdim(A)

    sites = siteinds(rhs)
    # build_problem may draw random TCI pivots: reseed so the initial guess only depends on N.
    Random.seed!(task.N)
    init_mps = random_mps(sites)

    method = task.method
    key = ""
    extra = Dict{String, Any}()
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

        Random.seed!(task.N)
        GC.gc(true)
        b = @timed sincquadrature_preconditioner(prob.arch, sites, prob.A_list, M)
        pc = b.value
        t_build = b.time
        build_compile = b.compile_time
        build_gc = b.gctime
        extra["M"] = M
        extra["precond_target"] = "L0"

    elseif method == "fai" || method == "fai_L0"
        μ = task.param1
        gdim = task.param2
        # fai_L0 approximates the inverse of the separable L0 = -Δ but preconditions A.
        target = method == "fai" ? A : prob.A0
        key = "$(method)_mu_$(μ)_gdim_$(gdim)"
        @info "Running $(method) (μ=$μ, G_maxdim=$gdim) N=$N F=$F"

        # The FAI build draws a randomized sketch: seed it so every FAI task of this N builds from
        # the same RNG state.
        Random.seed!(task.N)
        GC.gc(true)
        b = @timed build_fai(target, μ, gdim; nsweeps=FAI_BUILD_NSWEEPS, residual_tol_build=pc_tol_map[N])
        pc = b.value
        t_build = b.time
        build_compile = b.compile_time
        build_gc = b.gctime
        extra["mu"] = μ
        extra["G_maxdim"] = gdim
        extra["precond_target"] = method == "fai" ? "L" : "L0"

    else
        error("Unknown task method: $method")
    end

    solve_fn = isnothing(pc) ?
        (s, x0) -> linearsolve(s, A, rhs, x0) :
        (s, x0) -> linearsolve(s, A, rhs, pc, x0)

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

    if cold.global_iters != warm.global_iters
        @warn "cold/warm global_iters disagree" cold=cold.global_iters warm=warm.global_iters
    end

    result_dict = Dict{String, Any}(
        "key" => key,
        "N" => N,
        "F" => F,
        "rho" => prob.ρ,
        "floor_analytic" => floor_from_rho(prob.ρ, N),
        "rhs_kind" => String(RHS_KIND),
        "method" => method,
        "precond_build_time" => t_build,
        "solve_time" => warm.time,
        "total_time" => t_build + warm.time,
        "relative_residual" => rel_residual,
        "global_iters" => warm.global_iters,
        "converged" => warm.global_iters < maxiter_map[N],
        "solution_maxdim" => maxlinkdim(out_mps),
        "solution_hit_maxdim" => maxlinkdim(out_mps) >= maxdims_map[N],

        "solve_time_cold" => cold.time,
        "compile_time_cold" => cold.compile_time,
        "gc_time_cold" => cold.gc_time,
        "global_iters_cold" => cold.global_iters,

        "compile_time" => warm.compile_time,
        "gc_time" => warm.gc_time,

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

    return result_dict
end

"""
    warmup(N)

JIT-warm every code path that a timed task will exercise, using the *same* solver/FAI-builder
factories as the timed tasks (`make_global_solver`, `build_fai`) so that argument-type-specific
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
        dfai  = build_fai(dA, 1e7, 2; nsweeps=2, residual_tol_build=pc_tol_map[5])

        solve_fns = (
            (s, x0) -> linearsolve(s, dA, drhs, x0),
            (s, x0) -> linearsolve(s, dA, drhs, dsinc, x0),
            (s, x0) -> linearsolve(s, dA, drhs, dfai, x0),
        )
        for solve_fn in solve_fns
            solver, _ = make_global_solver(; rtol=1e-2, maxiter=2, maxdim=10, cutoff=1e-8, orth_maxdim=10, orth_cutoff=1e-8)
            @timed solve_fn(solver, copy(dinit))
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

function main()
    task_id = length(ARGS) > 0 ? parse(Int, ARGS[1]) : 1
    all_tasks = build_task_list()

    if task_id < 1 || task_id > length(all_tasks)
        error("Invalid task_id=$task_id (there are $(length(all_tasks)) tasks)")
    end

    task = all_tasks[task_id]

    warmup_time = warmup(task.N)

    result = run_task(task; warmup_time=warmup_time)

    mkpath(TASKS_DIR)
    output_file = joinpath(TASKS_DIR, "task_$(lpad(task_id, 3, '0')).jld2")
    save(output_file, "result", result)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end

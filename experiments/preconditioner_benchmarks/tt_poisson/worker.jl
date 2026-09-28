"""
    TT Poisson Benchmark: Parallel Worker
"""

using ITensors
using ITensorMPS
using LinearAlgebra
using JLD2
using TensorCrossInterpolation
using DifferenTT
using Random

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

const use_nocond = length(ARGS) > 1 && ARGS[2] == "nocond"
const default_base_folder = use_nocond ? "experiments/preconditioner_benchmarks/tt_poisson/results_nocond" : "experiments/preconditioner_benchmarks/tt_poisson/results"
base_folder = get(ENV, "TT_POISSON_RESULTS_DIR", default_base_folder)
const TASKS_DIR = joinpath(base_folder, "tasks")
mkpath(TASKS_DIR)

# Timed solves per task. Run 1 is "cold" (JIT/heap warmed only by the global warmup at process
# start), the last run is "warm" and is what `solve_time` reports. This isolates any remaining
# compilation/GC bias between methods so it shows up explicitly in `*_cold` fields instead of
# silently inflating one method's `solve_time` over another's.
const N_SOLVE_REPEATS = use_nocond ? 2 : 1

# Number of sweeps used to build the FAI preconditioner. Overridable for fast local verification;
# the default (30) matches the value used in every existing FAI run.
const FAI_BUILD_NSWEEPS = parse(Int, get(ENV, "TT_POISSON_FAI_NSWEEPS", "30"))

# Sweeping over number of dimensions N
const system_dims = [5, 10, 15, 20, 25]
const N_NODES_PER_DIM = 15
const residual_tol = 1e-7

const maxdims_map = Dict(
    5 => 20,
    10 => 25,
    15 => 25,
    20 => 30,
    25 => 30,
    30 => 50,
    40 => 50,
    50 => 50
)

const nsweeps_map = Dict(
    5 => 50,
    10 => 50,
    15 => 50,
    20 => 60,
    25 => 60,
    30 => 80,
    40 => 100,
    50 => 120
)

const pc_tol_map = Dict(
    5 => 1e-3,
    10 => 1e-3,
    15 => 1e-3,
    20 => 1e-2,
    25 => 1e-1,
    30 => 2e-1,
    40 => 2e-1,
    50 => 2e-1
)

# We will test: Baseline, Sinc-Quadrature (M sweep), FAI (mu and gdim sweep)
# To simplify the job array, we just do a few configs per N.
const sinc_M_values = [3, 4, 5, 7, 10]
const fai_configs = [(1e7, 4), (1e7, 6), (1e7, 8), (1e7, 10)]

ITensors.disable_warn_order()
ENV["JULIA_DEBUG"] = ""

struct TaskSpec
    N::Int
    method::String
    param1::Float64 # M for Sinc, mu for FAI
    param2::Int # not used for Sinc, gdim for FAI
end

function build_task_list()
    tasks = TaskSpec[]
    for N in system_dims
        # Baseline
        push!(tasks, TaskSpec(N, "baseline", 0.0, 0))
        
        # Sinc Quadrature
        for M in sinc_M_values
            push!(tasks, TaskSpec(N, "sinc", Float64(M), 0))
        end
        
        # FAI
        for (μ, gdim) in fai_configs
            push!(tasks, TaskSpec(N, "fai", μ, gdim))
        end
    end
    return tasks
end

function get_1d_A(i::Int, grid, arch)
    noneye_dim = arch.var_dims[i]
    bc = BoundaryCond1d(DirichletBC(0.0), DirichletBC(0.0))
    # onedim_differential_operator with an arch::TTArchitecture{N} returns the matrix we need
    mat = DifferenTT.onedim_differential_operator(2, arch, i, bc, Array; stencil_order=2)
    # the exact Laplacian has a minus sign to be positive definite
    # also scale by 1 / dx^2
    mat = -1.0 * mat .* (1.0 / grid.dx[i]^2)
    return mat
end

function get_gaussian_sol(N::Int, R::Int)
    Random.seed!(42)
    centers = rand(R, N) .* 0.6 .+ 0.2
    alphas = rand(R, N) .* 4.0 .+ 1.0
    
    function sol(x)
        val = 0.0
        for k in 1:R
            term = 1.0
            for j in 1:N
                term *= exp(-alphas[k,j] * (x[j] - centers[k,j])^2)
            end
            val += term
        end
        # Apply Dirichlet boundary condition envelope
        env = 1.0
        for j in 1:N
            env *= sin(pi * x[j])
        end
        return val * env
    end
    return sol
end

function get_coulomb_sol(N::Int, ε::Float64=0.05)
    c = fill(0.5, N)
    function sol(x)
        r2 = sum((x[j] - c[j])^2 for j in 1:N)
        env = prod(sin(pi * x[j]) for j in 1:N)
        return (1.0 / sqrt(r2 + ε^2)) * env
    end
    return sol
end

function build_problem(N::Int)

    box_size = ntuple(i -> 1.0, N)
    nodes = ntuple(i -> N_NODES_PER_DIM, N)
    
    arch = TTArchitecture(nodes)
    grid = Grid(arch, box_size)
    
    # Boundary conditions
    bcs = ntuple(i -> BoundaryCond1d(DirichletBC(0.0), DirichletBC(0.0)), N)
    bc = BoundaryConditions(bcs...)
    
    sites = siteinds(arch)
    
    # sol_func = get_gaussian_sol(N, 30)
    sol_func = get_coulomb_sol(N, 0.2)
    
    maxdim_N = get(maxdims_map, N, 10)
    
    sol_initialpivot = optfirstpivot(Float64, sol_func, grid, arch)
    # Cap ttcross by maxdim_N to avoid OOM
    sol_tci, _ = crossinterpolate2(Float64, sol_func, grid, arch; initialpivots=[sol_initialpivot], maxbonddim=maxdim_N, tolerance=1e-12)
    sol_mps = MPS(sol_tci)
    replace_siteinds!(sol_mps, sites)
    
    field = ScalarField(sol_mps, arch, bc)
    Δ = symmetrized_laplacian(field, grid; stencil_order=2)
    @show maxlinkdim(Δ)
    A = -1.0 * Δ
    
    # Discrete RHS A * sol_mps so FD truncation error does not dominate the solve error.
    rhs_mps = apply(A, sol_mps; alg="naive")
    
    # Incorporate Dirichlet boundary corrections through library functions (even if they do not do anything)
    # Do not cap maxdim here so we don't truncate rhs_mps and mix truncation error with linsolve error
    Δ_correction_mpo = scalar_dirichlet_correction(field, grid, 2; stencil_order=2, cutoff=0.0)
    Δ_correction = isnothing(Δ_correction_mpo) ? nothing : apply(Δ_correction_mpo, ones_mps(field.usites); alg="naive")
    rhs_mps = isnothing(Δ_correction) ? rhs_mps : +(rhs_mps, Δ_correction; alg="directsum")
    
    # Symmetrize the RHS
    rhs_mps = symmetrize_rhs(field, rhs_mps)
    
    A_list = [get_1d_A(i, grid, arch) for i in 1:N]
    
    return A, rhs_mps, sol_mps, A_list, grid, arch
end

"""
    make_outer_solver(A, rhs; nsweeps, maxdim)

Build the outer ResidualALS solver and its stat-tracking stores. Used both by the JIT `warmup()`
and by every timed task, with identical argument *types* (same `observer!` type, same store
types), so no keyword-method compilation is deferred into the timed solve.
"""
function make_outer_solver(A::MPO, rhs::MPS; nsweeps::Int, maxdim::Int)
    residual_observer = ResidualObserver(A, rhs, residual_tol)
    sweep_kwargs = SweepingArgs(nsweeps=nsweeps, maxdim=maxdim, cutoff=1e-13, observer! = residual_observer)

    numiter_store = KrylovSweepStats()
    cond_store    = use_nocond ? nothing : KrylovCondStats()
    pc_cond_store = use_nocond ? nothing : KrylovCondStats()

    updater_kwargs = LocalUpdaterArgs(
        ishermitian = true,
        isposdef = true,
        rtol = 0.1*residual_tol,
        krylovdim = 1,
        maxiter = 3000,
        numiter_store = numiter_store,
        cond_store = cond_store,
        pc_cond_store = pc_cond_store
    )

    solver = ResidualALS(sweep_kwargs=sweep_kwargs, updater_kwargs=updater_kwargs, residual_tol=0.5*residual_tol)
    return solver, (; numiter_store, cond_store, pc_cond_store)
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
        compute_cond = false,
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

"""Summarize a `KrylovSweepStats` store: per-site iteration totals, grand total, and sweep count."""
function krylov_iters_summary(stats_store::KrylovSweepStats)
    per_site = DifferenTT.positions_sum(stats_store)
    nsweeps = isempty(stats_store.numiters) ? 0 : maximum(length, stats_store.numiters)
    return (; iters_per_site = per_site, total_iters = sum(per_site; init=0.0), nsweeps)
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
        # Same RNG state before every timed solve (all methods and repeats), so no random choice
        # inside the solver can differ between them.
        Random.seed!(N)
        GC.gc(true)
        stats = @timed solve_fn(solver, x0)
        numiters = krylov_iters_summary(stores.numiter_store)
        push!(runs, (;
            out_mps = stats.value,
            time = stats.time,
            compile_time = stats.compile_time,
            gc_time = stats.gctime,
            stores = stores,
            numiters...
        ))
        @info "solve repeat $rep" time=stats.time compile=stats.compile_time gc=stats.gctime total_iters=numiters.total_iters nsweeps=numiters.nsweeps
    end
    return runs
end

function run_task(task::TaskSpec; warmup_time::Float64=0.0)
    N = task.N
    Random.seed!(task.N)
    @info "Building problem for N=$N..."
    A, rhs, sol_mps, A_list, grid, arch = build_problem(N)

    sites = siteinds(rhs)
    # build_problem draws random TCI pivots: reseed so the initial guess only depends on N.
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
        @info "Running Baseline N=$N"

    elseif method == "sinc"
        M = Int(task.param1)
        key = "sinc_M_$(M)"
        @info "Running Sinc (M=$M) N=$N"

        Random.seed!(task.N)
        GC.gc(true)
        b = @timed sincquadrature_preconditioner(arch, sites, A_list, M)
        pc = b.value
        t_build = b.time
        build_compile = b.compile_time
        build_gc = b.gctime
        extra["M"] = M

    elseif method == "fai"
        μ = task.param1
        gdim = task.param2
        key = "fai_mu_$(μ)_gdim_$(gdim)"
        @info "Running FAI (μ=$μ, G_maxdim=$gdim) N=$N"

        # The FAI build draws a randomized sketch: seed it so every FAI task of this N builds from
        # the same RNG state.
        Random.seed!(task.N)
        GC.gc(true)
        b = @timed build_fai(A, μ, gdim; nsweeps=FAI_BUILD_NSWEEPS, residual_tol_build=pc_tol_map[N])
        pc = b.value
        t_build = b.time
        build_compile = b.compile_time
        build_gc = b.gctime
        extra["mu"] = μ
        extra["G_maxdim"] = gdim

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

    result_dict = Dict{String, Any}(
        "key" => key,
        "N" => N,
        "method" => method,
        "precond_build_time" => t_build,
        "solve_time" => warm.time,
        "total_time" => t_build + warm.time,
        "relative_residual" => rel_residual,
        "iters_per_site" => warm.iters_per_site,
        "solution_maxdim" => maxlinkdim(out_mps),

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
    warmup(N)

JIT-warm every code path that a timed task will exercise, using the *same* solver/FAI-builder
factories as the timed tasks (`make_outer_solver`, `build_fai`) so that argument-type-specific
compilation (e.g. `observer! = ResidualObserver`, `numiter_store::KrylovSweepStats`) happens here
rather than inside a timed solve. `dummy_N = 4` is the smallest chain with an interior 2-site
bond (both left/right environments are real, non-trivial tensors), which is what N >= 3 problems
actually hit.

`sincquadrature_preconditioner` is additionally compiled for the task's dimension `N`: it is
specialized on `TTArchitecture{N}`, so the `dummy_N` call does not cover it. `M` is only a value,
so a single call with `M = 1` compiles it for every `M`.
"""
function warmup(N::Int)
    @info "Running JIT warmup..."
    t = @elapsed begin
        dummy_N = 4
        dA, drhs, dsol, dA_list, dgrid, darch = build_problem(dummy_N)
        dsites = siteinds(drhs)
        dinit = random_mps(dsites)

        dsinc = sincquadrature_preconditioner(darch, dsites, dA_list, 1)
        dfai  = build_fai(dA, 1e7, 2; nsweeps=2, residual_tol_build=pc_tol_map[5])

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
        # operators, so the TCI solution of build_problem is not built here.
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
        error("Invalid task_id=$task_id")
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

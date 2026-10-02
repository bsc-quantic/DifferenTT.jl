"""
    Lid-Driven Cavity (ψ–ω) Pipeline Benchmark: Parallel Worker

FAI preconditioner benchmark on the stream function Poisson systems `-Δψ = ω` of the lid-driven cavity
(`INSVorticityExperiment`, walls as ghost nodes). The operator is the plain Dirichlet 5-point Laplacian, so it
is exactly symmetric positive definite. Mirrors `chorin_pipeline/worker.jl`: one timed Poisson solve per step,
followed by the explicit vorticity transport.
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
const default_base_folder = use_nocond ? "experiments/preconditioner_benchmarks/liddriven_pipeline/results_nocond" : "experiments/preconditioner_benchmarks/liddriven_pipeline/results"
base_folder = get(ENV, "LIDDRIVEN_RESULTS_DIR", default_base_folder)
const TASKS_DIR = joinpath(base_folder, "tasks")
mkpath(TASKS_DIR)

const BOX_SIZE = (1.0, 1.0)
const RE_NUM = 1000
const U_LID = 1.0
const STENCIL_ORDER = 2
const WALL_FORMULA = :jensen

# DirichletBC logs at every construction: build the wall conditions once
const WALL = BoundaryCond1d(DirichletBC(0.0), DirichletBC(0.0))
const WALL_BCS = BoundaryConditions(WALL, WALL)

const dt_map = Dict(
    (8, 8) => 0.001,
    (10, 10) => 0.0002,
    (12, 12) => 0.00001
)

const n_steps_map = Dict(
    (8, 8) => 200,
    (10, 10) => 100,
    (12, 12) => 20
)

# Overridable for fast verification only — the default reproduces n_steps_map[nodes] exactly.
function resolve_n_steps(nodes::Tuple{Int,Int})
    override = get(ENV, "LIDDRIVEN_N_STEPS_OVERRIDE", "")
    return isempty(override) ? n_steps_map[nodes] : parse(Int, override)
end

const system_sizes = [(8, 8), (10, 10)]
const cond_nodes  = [(8, 8), (10, 10)]
# const system_sizes = [(10, 10)]
# const cond_nodes  = [(10, 10)]

function should_compute_cond(nodes::Tuple{Int, Int}, use_nocond::Bool)::Bool
    return !use_nocond && nodes in cond_nodes
end

# const mu_values = [5e6, 1e7, 5e7, 1e8]
const mu_values = [1e7]

const g_maxdims = 2:10

const maxdims_map = Dict(
    (8, 8) => 200,
    (10, 10) => 350,
    (12, 12) => 400
)

# WARNING: MAKE SURE TO CHANGE THIS TO >0.0 IF RUNNING ALS -> OOM WILL HAPPEN
const cutoff_map = Dict(
    (8, 8) => 0.0,
    (10, 10) => 0.0,
    (12, 12) => 0.0
)

const nsweeps_map = Dict(
    (8, 8) => 100,
    (10, 10) => 100,
    (12, 12) => 100
)

const tol_map = Dict(
    (8, 8) => 1e-4,
    (10, 10) => 1e-3,
    (12, 12) => 1e-2
)

const fai_tol_map = Dict(
    (8, 8) => 0.1,
    (10, 10) => 0.1,
    (12, 12) => 0.2
)

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

function build_grid(nodes::NTuple{2, Int}, max_steps::Int; shared_sites::Union{Nothing,Vector{<:Index}}=nothing)
    # `siteinds` mints fresh, uniquely-tagged Index objects on every call. A throwaway warmup
    # grid built independently from the real grid would therefore carry *different* site
    # indices even with identical nodes — any MPO built against one (e.g. a FAI
    # preconditioner built from the real grid's A) cannot be applied to the other's MPS/MPO,
    # and ITensorMPS raises "must share a set of site indices". Passing `shared_sites` lets a
    # warmup grid reuse the real grid's indices so cross-grid objects like `precond` stay valid.
    sites = isnothing(shared_sites) ? siteinds(2, sum(nodes)) : shared_sites
    arch = QTTArchitecture(nodes; interleaved=false)
    # Walls are ghost nodes: 2^n interior nodes per axis with h = L / (2^n + 1)
    grid = interior_grid(arch, BOX_SIZE)

    # Fluid at rest, impulsively started lid
    omega_field = ScalarField(zeros_mps(sites), arch, WALL_BCS)
    psi_field = ScalarField(zeros_mps(sites), arch, WALL_BCS)

    time_stepper = RK4(time_evol_kwargs=TimeEvolArgs(0.0, dt_map[nodes]*max_steps, max_steps), cutoff=1e-11, maxdim=maxdims_map[nodes])
    experiment = INSVorticityExperiment(grid, omega_field, psi_field, RE_NUM, U_LID; wall_formula=WALL_FORMULA, stencil_order=STENCIL_ORDER, maxdim=maxdims_map[nodes], cutoff=1e-11)
    check_cfl(experiment, time_stepper; vmax=U_LID)

    # The stream function operator A = -Δ (Dirichlet, ghost walls) is SPD: no pinning, no RHS corrections
    cache = VorticityCache(experiment)
    A = cache.poisson_op

    # Priming step: at rest ω = 0, so ψ = 0 and the first Poisson system is trivial. Advance one untimed step
    # with ψ = 0 (no solve) so that every timed step has a nonzero right-hand side.
    omega_primed = DifferenTT.vorticity_transport(experiment, cache, last(experiment.omega_field), last(experiment.psi_field), time_stepper, 0.0)
    experiment.omega_field[length(experiment.omega_field)] = omega_primed

    return (;
        nodes=nodes,
        A=A,
        experiment=experiment,
        time_stepper=time_stepper,
        cache=cache,
        sites=sites
    )
end

function build_rhs(grid, omega)
    return DifferenTT.poisson_rhs(grid.experiment, grid.cache, omega)
end

function liddriven_step!(grid, sweep_kwargs, compute_cond::Bool, tol_val::Float64, step_index::Int, precond=nothing)
    experiment = grid.experiment
    omega = last(experiment.omega_field)
    rhs = build_rhs(grid, omega)

    numiter_store = KrylovSweepStats()
    cond_store = compute_cond ? KrylovCondStats() : nothing
    pc_cond_store = compute_cond ? KrylovCondStats() : nothing
    updater_kwargs = LocalUpdaterArgs(;
        ishermitian = true,
        isposdef = true,
        rtol = 0.1 * tol_val,
        krylovdim=1,
        maxiter=2000,
        numiter_store = numiter_store,
        cond_store = cond_store,
        pc_cond_store = pc_cond_store
    )

    # A and b are bound by linearsolve to the system grid.A * psi = rhs of this step
    res_observer = ResidualObserver(rtol=tol_val)
    step_sweep_kwargs = SweepingArgs(; NamedTuple(sweep_kwargs)..., observer! = res_observer)
    resals_solver = ResidualALS(sweep_kwargs=step_sweep_kwargs, updater_kwargs=updater_kwargs, residual_tol=0.5 * tol_val)
    amen_solver = AMEn(sweep_kwargs=step_sweep_kwargs, updater_kwargs=updater_kwargs, enrichment_maxdim = 3, enrichment_cutoff = 1e-10)
    als_solver = ALS(sweep_kwargs=step_sweep_kwargs, updater_kwargs=updater_kwargs)

    solver = resals_solver

    # ψ carries no dt scaling: the previous stream function is directly the warm start
    previous_psi = last(experiment.psi_field)
    if DifferenTT.iszero_tt(previous_psi)
        Random.seed!(hash((grid.nodes, step_index)))
        psi_guess = random_mps(experiment.psi_field.usites)
    else
        psi_guess = previous_psi
    end

    # Same RNG state before every timed solve (baseline and preconditioned alike), so no random
    # choice inside the solver can differ between methods.
    Random.seed!(hash((grid.nodes, step_index)))

    # Collect the garbage from the previous step/build_rhs before timing, so it isn't charged to
    # the solve.
    GC.gc(true)
    if isnothing(precond)
        stats = @timed linearsolve(solver, grid.A, rhs, psi_guess)
    else
        stats = @timed linearsolve(solver, grid.A, rhs, precond, psi_guess)
    end
    psi = stats.value
    t = stats.time

    # Compute relative residual using inner product to save memory
    term1 = ITensorMPS.inner(grid.A, psi, grid.A, psi)
    term2 = 2 * real(ITensorMPS.inner(rhs', grid.A, psi))
    term3 = ITensorMPS.inner(rhs, rhs)
    rel_res = sqrt(max(0.0, term1 - term2 + term3)) / norm(rhs)

    @info "Residual $rel_res"
    flush(stderr)

    sol_maxdim = maxlinkdim(psi)
    @info "Sol_maxdim = $sol_maxdim"
    flush(stderr)

    # Overwrite in place: psi_field keeps only the latest stream function (not a time series).
    experiment.psi_field[length(experiment.psi_field)] = psi
    omega_next = DifferenTT.vorticity_transport(experiment, grid.cache, omega, psi, grid.time_stepper, 0.0)
    experiment.omega_field[length(experiment.omega_field)] = omega_next

    omega_maxdim = maxlinkdim(omega_next)
    iters_per_site = DifferenTT.positions_sum(numiter_store)

    res_dict = (; t, compile_t = stats.compile_time, gc_t = stats.gctime, rel_res, sol_maxdim, omega_maxdim, iters_per_site)

    if compute_cond
        local_conds = isnothing(precond) ? positions_mean_cond(cond_store) : positions_mean_cond(pc_cond_store)
        local_cond_mean = isempty(local_conds) ? NaN : mean(local_conds)
        res_dict = merge(res_dict, (; local_conds, local_cond_mean))
    end

    return res_dict
end

function run_task(task::TaskSpec; warmup_time::Float64=0.0)
    Random.seed!(task.nodes[1] + task.nodes[2])
    N_STEPS = resolve_n_steps(task.nodes)
    compute_cond = should_compute_cond(task.nodes, use_nocond)
    grid = build_grid(task.nodes, N_STEPS)
    maxdim_val = maxdims_map[task.nodes]
    nsweep_val = nsweeps_map[task.nodes]
    cutoff_val = cutoff_map[task.nodes]
    tol_val = tol_map[task.nodes]
    faibuild_tol = fai_tol_map[task.nodes]

    sweep_kwargs = SweepingArgs(nsweeps=nsweep_val, maxdim=maxdim_val, cutoff=cutoff_val)

    if task.is_baseline
        @info "Running baseline for $(task.nodes)"
        flush(stderr)

        solve_times = Float64[]
        compile_times = Float64[]
        gc_times = Float64[]
        iters = Vector{Float64}[]
        residuals = Float64[]
        solution_maxdims = Int[]
        omega_maxdims = Int[]
        iters_per_site = Float64[]
        conds_per_site = Float64[]
        avg_conds = Float64[]

        for i in 1:N_STEPS
            @info "Step $i / $N_STEPS"
            flush(stderr)

            step_res = liddriven_step!(grid, sweep_kwargs, compute_cond, tol_val, i)
            push!(solve_times, step_res.t)
            push!(compile_times, step_res.compile_t)
            push!(gc_times, step_res.gc_t)
            push!(residuals, step_res.rel_res)
            push!(solution_maxdims, step_res.sol_maxdim)
            push!(omega_maxdims, step_res.omega_maxdim)
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

        result_dict = Dict{String, Any}(
            "key" => "baseline",
            "nodes" => task.nodes,
            "compute_cond" => compute_cond,
            "total_solve_time" => sum(solve_times),
            "solve_times" => solve_times,
            "iters" => iters,
            "iters_per_site" => iters_per_site,
            "residuals" => residuals,
            "solution_maxdims" => solution_maxdims,
            "omega_maxdims" => omega_maxdims,
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
            "warmup_time" => warmup_time,
            "gc_ballast_bytes" => length(GC_BALLAST),
            "heap_size_hint" => Int(Base.JLOptions().heap_size_hint),
        )

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

        fai_sweep_kwargs = SweepingArgs(nsweeps=100, maxdim=gdim, cutoff=1e-12)

        fai_updater_kwargs = LocalUpdaterArgs(
            ishermitian = true,
            isposdef = true,
            compute_cond = false,
            rtol = 1e-9,
            krylovdim = 1,
            maxiter = 3000
        )

        fai_resals_solver = ResidualALS(
            sweep_kwargs = fai_sweep_kwargs,
            updater_kwargs = fai_updater_kwargs,
            residual_tol = 1e-9
        )

        fai_als_solver = ALS(
            sweep_kwargs = fai_sweep_kwargs,
            updater_kwargs = fai_updater_kwargs,
        )

        fai_amen_solver =  AMEn(
            sweep_kwargs = fai_sweep_kwargs,
            updater_kwargs = fai_updater_kwargs,
            enrichment_maxdim = 1,
        )

        D_kwargs_local = (; cutoff=1e-11, oversample=10, maxdim=gdim)
        fai_builder_solver = fai_als_solver

        # The FAI build draws a randomized sketch: seed it so every task of this size builds from
        # the same RNG state.
        Random.seed!(task.nodes[1] + task.nodes[2])
        GC.gc(true)
        build_stats = @timed begin
            unscaledG, D = DifferenTT.fai_dg(grid.A, μ, fai_builder_solver, D_kwargs_local; preconditioning=false, residual_tol=faibuild_tol)
            Gl = apply(D, unscaledG; maxdim=gdim)
            precond = FactorizedPreconditioner(Gl)
        end
        t_build = build_stats.time

        solve_times = Float64[]
        compile_times = Float64[]
        gc_times = Float64[]
        iters = Vector{Float64}[]
        residuals = Float64[]
        solution_maxdims = Int[]
        omega_maxdims = Int[]
        iters_per_site = Float64[]
        pc_conds_per_site = Float64[]
        avg_pc_conds = Float64[]

        for i in 1:N_STEPS
            @info "Step $i / $N_STEPS"
            flush(stderr)

            step_res = liddriven_step!(grid, sweep_kwargs, compute_cond, tol_val, i, precond)
            push!(solve_times, step_res.t)
            push!(compile_times, step_res.compile_t)
            push!(gc_times, step_res.gc_t)
            push!(residuals, step_res.rel_res)
            push!(solution_maxdims, step_res.sol_maxdim)
            push!(omega_maxdims, step_res.omega_maxdim)
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

        result_dict = Dict{String, Any}(
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
            "omega_maxdims" => omega_maxdims,
            "compile_times" => compile_times,
            "gc_times" => gc_times,
            "total_compile_time" => sum(compile_times),
            "total_gc_time" => sum(gc_times),
            "hostname" => gethostname(),
            "slurm_job_id" => get(ENV, "SLURM_JOB_ID", ""),
            "blas_threads" => LinearAlgebra.BLAS.get_num_threads(),
            "julia_threads" => Threads.nthreads(),
            "warmup_time" => warmup_time,
            "gc_ballast_bytes" => length(GC_BALLAST),
            "heap_size_hint" => Int(Base.JLOptions().heap_size_hint),
        )

        if compute_cond
            pc_conds_per_site ./= N_STEPS
            result_dict["pc_conds_per_site"] = pc_conds_per_site
            result_dict["avg_pc_cond"] = mean(avg_pc_conds)
        end

        return result_dict
    end
end

function warmup()
    @info "Running JIT warmup..."
    flush(stderr)
    t = @elapsed begin
        dummy_grid = build_grid((8, 8), 1)
        dummy_updater_kwargs = LocalUpdaterArgs(ishermitian=true, isposdef=true, rtol=1e-2, maxiter=2, krylovdim=1)
        dummy_als_solver = ALS(
            sweep_kwargs = SweepingArgs(nsweeps=2, maxdim=10, cutoff=1e-8),
            updater_kwargs = dummy_updater_kwargs
        )
        # Pass residual_tol like the timed build does: it adds a ResidualObserver to the FAI solves,
        # a different code path that would otherwise be compiled inside the timed build.
        unscaledG, D = DifferenTT.fai_dg(dummy_grid.A, 1.0, dummy_als_solver, (; cutoff=1e-11, oversample=2, maxdim=2); preconditioning=false, residual_tol=fai_tol_map[(8, 8)])
        dummy_fai = FactorizedPreconditioner(apply(D, unscaledG; maxdim=2))

        # When running with condition numbers, compute time does not really count, so it is better to avoid
        # crashing at precompilation due to dummy preconditioners than to have more accurate timings due to precompilation
        liddriven_step!(dummy_grid, SweepingArgs(nsweeps=2, maxdim=10, cutoff=1e-8), false, 1e-3, 0, nothing)
        liddriven_step!(dummy_grid, SweepingArgs(nsweeps=2, maxdim=10, cutoff=1e-8), false, 1e-3, 0, dummy_fai)
    end
    @info "JIT warmup complete" seconds=t
    flush(stderr)
    return t
end

function main()
    task_id = length(ARGS) > 0 ? parse(Int, ARGS[1]) : 1
    all_tasks = build_task_list()
    (1 <= task_id <= length(all_tasks)) || error("Invalid task_id=$task_id")

    warmup_time = warmup()
    result = run_task(all_tasks[task_id]; warmup_time=warmup_time)

    mkpath(TASKS_DIR)
    save(joinpath(TASKS_DIR, "task_$(lpad(task_id, 3, '0')).jld2"), "result", result)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end

"""
    Chorin Pipeline Benchmark with GlobalCG: Parallel Worker
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

const default_base_folder = "experiments/fai_benchmarks/chorin_pipeline_globalcg/results"
base_folder = get(ENV, "CHORIN_GLOBALCG_RESULTS_DIR", default_base_folder)
const TASKS_DIR = joinpath(base_folder, "tasks")
mkpath(TASKS_DIR)

const MASK_FILE = "experiments/fai_benchmarks/chorin_pipeline_globalcg/masks.jld2"
# Note: we assume the user has the masks in MASK_FILE. If not, they must be copied over or generated.

const BOX_SIZE = (2.0, 1.0)
const RE_NUM = 3000
const STENCIL_ORDER = 8

const dt_map = Dict((8, 8) => 0.001, (10, 10) => 0.0002, (12, 12) => 0.00001)

const n_steps_map = Dict((8, 8) => 50, (10, 10) => 20, (12, 12) => 5)

# Overridable for fast local verification only — the default reproduces n_steps_map[nodes] exactly.
function resolve_n_steps(nodes::Tuple{Int,Int})
    override = get(ENV, "CHORIN_GLOBALCG_N_STEPS_OVERRIDE", "")
    return isempty(override) ? n_steps_map[nodes] : parse(Int, override)
end

# const system_sizes = [(8, 8), (10, 10), (12, 12)]
const system_sizes = [(8, 8), (10, 10)]

# const mu_values = [5e6, 1e7, 5e7, 1e8]
const mu_values = [1e7]

const g_maxdims = 2:10

const maxdims_map = Dict((8, 8) => 150, (10, 10) => 200, (12, 12) => 250)

const cutoff_map = Dict((8, 8) => 1e-13, (10, 10) => 1e-13, (12, 12) => 1e-13)

const maxiter_map = Dict((8, 8) => 3000, (10, 10) => 3000, (12, 12) => 5000)

const tol_map = Dict((8, 8) => 1e-2, (10, 10) => 1e-2, (12, 12) => 1e-2)

const fai_tol_map = Dict((8, 8) => 0.1, (10, 10) => 0.1, (12, 12) => 0.2)

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
    grid = Grid(arch, BOX_SIZE)

    if isfile(MASK_FILE)
        mask_key = "mask_$(nodes[1])_$(nodes[2])"
        object_mask = load(MASK_FILE, mask_key)
        siteinds(object_mask) != sites && replace_siteinds!(object_mask, sites)
    else
        object_mask = ones_mps(sites)
        @warn "mask not found; using ones_mps (all fluid)"
    end

    v0 = 1.0
    mps_vx = tt_mul(object_mask, v0 * ones_mps(sites))
    mps_vy = 0.0 * ones_mps(sites)

    bcsvxx = BoundaryCond1d(DirichletBC(v0), NeumannBC(0.0))
    bcsvxy = BoundaryCond1d(PeriodicBC(), PeriodicBC())

    bcsvyx = BoundaryCond1d(DirichletBC(0.0), NeumannBC(0.0))
    bcsvyy = BoundaryCond1d(PeriodicBC(), PeriodicBC())

    bcspx = BoundaryCond1d(NeumannBC(0.0), DirichletBC(1.0))
    bcspy = BoundaryCond1d(PeriodicBC(), PeriodicBC())

    bcsvx = BoundaryConditions(bcsvxx, bcsvxy)
    bcsvy = BoundaryConditions(bcsvyx, bcsvyy)
    bcsp = BoundaryConditions(bcspx, bcspy)

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
    Δ_cache = symmetrized_laplacian(
        experiment.p_field,
        experiment.grid;
        stencil_order=STENCIL_ORDER,
        maxdim=maxdims_map[nodes],
        cutoff=cutoff_map[nodes],
    )
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

function chorin_step!(grid, global_updater_kwargs, step_index::Int, precond=nothing)
    global_updater_kwargs.numiter_store[] = 0
    experiment = grid.experiment
    masked_u = masked_velocity(grid)
    rhs = build_rhs(grid, masked_u)

    solver = GlobalCG(; updater_kwargs=global_updater_kwargs)

    dt = grid.time_stepper.time_evol_kwargs.dt
    previous_p = last(experiment.p_field)
    if isnothing(previous_p)
        Random.seed!(hash((grid.nodes, step_index)))
        pdt_guess = random_mps(grid.experiment.u_fields.usites)
    else
        pdt_guess = previous_p * dt
    end

    # Same RNG state before every timed solve (baseline and preconditioned alike), so no random
    # choice inside the solver can differ between methods.
    Random.seed!(hash((grid.nodes, step_index)))

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

    global_iters = global_updater_kwargs.numiter_store[]

    res_dict = (;
        t,
        compile_t=stats.compile_time,
        gc_t=stats.gctime,
        rel_res,
        sol_maxdim,
        global_iters,
        div_norm_before,
        div_norm_after,
        div_ratio,
    )

    return res_dict
end

function run_task(task::TaskSpec; warmup_time::Float64=0.0)
    Random.seed!(task.nodes[1] + task.nodes[2])
    N_STEPS = resolve_n_steps(task.nodes)
    grid = build_grid(task.nodes, N_STEPS)
    maxdim_val = maxdims_map[task.nodes]
    maxiter_val = maxiter_map[task.nodes]
    rtol = tol_map[task.nodes]
    fai_rtol = fai_tol_map[task.nodes]
    cutoff_val = cutoff_map[task.nodes]

    global_updater_kwargs = GlobalUpdaterArgs(;
        ishermitian=true,
        isposdef=true,
        rtol=rtol,
        maxiter=maxiter_val,
        maxdim=maxdim_val,
        cutoff=cutoff_val,
        orth_maxdim=maxdim_val,
        orth_cutoff=0.1 * cutoff_val,
        numiter_store=Ref{Int}(0),
    )

    if task.is_baseline
        @info "Running baseline for $(task.nodes)"
        flush(stderr)

        solve_times = Float64[]
        compile_times = Float64[]
        gc_times = Float64[]
        global_iters = Int[]
        residuals = Float64[]
        solution_maxdims = Int[]
        div_norms_before = Float64[]
        div_norms_after = Float64[]
        div_ratios = Float64[]

        for i in 1:N_STEPS
            @info "Step $i / $N_STEPS"
            flush(stderr)

            step_res = chorin_step!(grid, global_updater_kwargs, i)
            push!(solve_times, step_res.t)
            push!(compile_times, step_res.compile_t)
            push!(gc_times, step_res.gc_t)
            push!(residuals, step_res.rel_res)
            push!(solution_maxdims, step_res.sol_maxdim)
            push!(div_norms_before, step_res.div_norm_before)
            push!(div_norms_after, step_res.div_norm_after)
            push!(div_ratios, step_res.div_ratio)
            push!(global_iters, step_res.global_iters)

            GC.gc(true)
        end

        result_dict = Dict{String,Any}(
            "key" => "baseline",
            "nodes" => task.nodes,
            "total_solve_time" => sum(solve_times),
            "solve_times" => solve_times,
            "global_iters" => global_iters,
            "total_global_iters" => sum(global_iters),
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
            "n_steps_run" => N_STEPS,
            "hostname" => gethostname(),
            "slurm_job_id" => get(ENV, "SLURM_JOB_ID", ""),
            "blas_threads" => LinearAlgebra.BLAS.get_num_threads(),
            "julia_threads" => Threads.nthreads(),
            "warmup_time" => warmup_time,
            "gc_ballast_bytes" => length(GC_BALLAST),
            "heap_size_hint" => Int(Base.JLOptions().heap_size_hint),
        )

        return result_dict
    else
        μ = task.mu
        gdim = task.gdim
        @info "Running (μ=$μ, G_maxdim=$gdim) for $(task.nodes)"
        flush(stderr)

        global_fai_solver = GlobalCG(;
            updater_kwargs=GlobalUpdaterArgs(;
                ishermitian=true,
                isposdef=true,
                rtol=1e-9,
                maxiter=3000,
                maxdim=gdim,
                cutoff=1e-12,
                orth_maxdim=gdim,
                orth_cutoff=1e-12,
            ),
        )

        fai_sweep_kwargs = SweepingArgs(; nsweeps=100, maxdim=gdim, cutoff=1e-12)

        fai_updater_kwargs = LocalUpdaterArgs(; ishermitian=true, isposdef=true, rtol=1e-5, krylovdim=1, maxiter=3000)

        resals_solver = ResidualALS(;
            sweep_kwargs=fai_sweep_kwargs, updater_kwargs=fai_updater_kwargs, residual_tol=1e-9
        )

        als_solver = ALS(; sweep_kwargs=fai_sweep_kwargs, updater_kwargs=fai_updater_kwargs)

        fai_builder_solver = als_solver

        D_kwargs_local = (; cutoff=1e-11, oversample=10, maxdim=gdim)

        # The FAI build draws a randomized sketch: seed it so every task of this size builds from
        # the same RNG state.
        Random.seed!(task.nodes[1] + task.nodes[2])
        GC.gc(true)
        build_stats = @timed begin
            unscaledG, D = DifferenTT.fai_dg(
                grid.A, μ, fai_builder_solver, D_kwargs_local; preconditioning=false, residual_tol=fai_rtol
            )
            Gl = apply(D, unscaledG; maxdim=gdim)
            precond = FactorizedPreconditioner(Gl)
        end
        t_build = build_stats.time

        solve_times = Float64[]
        compile_times = Float64[]
        gc_times = Float64[]
        global_iters = Int[]
        residuals = Float64[]
        solution_maxdims = Int[]
        div_norms_before = Float64[]
        div_norms_after = Float64[]
        div_ratios = Float64[]

        for i in 1:N_STEPS
            @info "Step $i / $N_STEPS"
            flush(stderr)

            step_res = chorin_step!(grid, global_updater_kwargs, i, precond)
            push!(solve_times, step_res.t)
            push!(compile_times, step_res.compile_t)
            push!(gc_times, step_res.gc_t)
            push!(residuals, step_res.rel_res)
            push!(solution_maxdims, step_res.sol_maxdim)
            push!(div_norms_before, step_res.div_norm_before)
            push!(div_norms_after, step_res.div_norm_after)
            push!(div_ratios, step_res.div_ratio)
            push!(global_iters, step_res.global_iters)

            GC.gc(true)
        end

        result_dict = Dict{String,Any}(
            "key" => "mu_$(μ)_gdim_$(gdim)",
            "nodes" => task.nodes,
            "mu" => μ,
            "G_maxdim" => gdim,
            "precond_build_time" => t_build,
            "precond_build_compile_time" => build_stats.compile_time,
            "precond_build_gc_time" => build_stats.gctime,
            "total_solve_time" => sum(solve_times),
            "solve_times" => solve_times,
            "global_iters" => global_iters,
            "total_global_iters" => sum(global_iters),
            "residuals" => residuals,
            "solution_maxdims" => solution_maxdims,
            "div_norms_before" => div_norms_before,
            "div_norms_after" => div_norms_after,
            "div_ratios" => div_ratios,
            "compile_times" => compile_times,
            "gc_times" => gc_times,
            "total_compile_time" => sum(compile_times),
            "total_gc_time" => sum(gc_times),
            "n_steps_run" => N_STEPS,
            "hostname" => gethostname(),
            "slurm_job_id" => get(ENV, "SLURM_JOB_ID", ""),
            "blas_threads" => LinearAlgebra.BLAS.get_num_threads(),
            "julia_threads" => Threads.nthreads(),
            "warmup_time" => warmup_time,
            "gc_ballast_bytes" => length(GC_BALLAST),
            "heap_size_hint" => Int(Base.JLOptions().heap_size_hint),
        )

        return result_dict
    end
end

function warmup()
    @info "Running JIT warmup..."
    flush(stderr)
    t = @elapsed begin
        dummy_grid = build_grid((8, 8), 1)
        dummy_updater_kwargs = GlobalUpdaterArgs(
            ishermitian=true, isposdef=true, rtol=1e-2, maxiter=2, maxdim=10, cutoff=1e-8, numiter_store=Ref{Int}(0)
        )
        dummy_solver = GlobalCG(updater_kwargs=dummy_updater_kwargs)

        dummy_als_updater_kwargs = LocalUpdaterArgs(ishermitian=true, isposdef=true, rtol=1e-2, maxiter=2, krylovdim=1)
        dummy_als_solver = ALS(
            sweep_kwargs=SweepingArgs(nsweeps=2, maxdim=10, cutoff=1e-8), updater_kwargs=dummy_als_updater_kwargs
        )
        unscaledG, D = DifferenTT.fai_dg(
            dummy_grid.A,
            1.0,
            dummy_als_solver,
            (; cutoff=1e-11, oversample=2, maxdim=2);
            preconditioning=false,
            residual_tol=fai_tol_map[(8, 8)],
        )
        dummy_fai = FactorizedPreconditioner(apply(D, unscaledG; maxdim=2))

        chorin_step!(dummy_grid, dummy_updater_kwargs, 0, nothing)
        chorin_step!(dummy_grid, dummy_updater_kwargs, 0, dummy_fai)
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

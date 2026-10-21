"""
    Chorin pipeline benchmark (local ResidualALS): worker

We simulate a uniform flow past a cylinder in a channel, the setup of Sec. 5.2 of the paper, in the QTT format. The box
is [0, 2] × [0, 1], discretized with 2^nx × 2^ny points, with the fields in a non-interleaved QTT. At the inlet (x = 0)
the velocity is (1, 0) and the pressure has a homogeneous Neumann condition; at the outlet (x = 2) the velocity has a
Neumann condition and the pressure a Dirichlet condition with the constant 1, which makes the pressure operator
non-singular; the y direction is periodic. A cylinder of radius 0.1 centered at (0.5, 0.5) is imposed through the
smoothed indicator ½[1 + tanh(700 ((x-0.5)² + (y-0.5)² - 0.1²))], which `generate_masks.jl` precomputes as an MPS by
cross interpolation and stores in `masks.jld2`. We evolve the flow at Re = 3000 with RK4, eighth-order stencils and the
Chorin projection, in which every time step solves the pressure Poisson equation ∇²p = ∇⋅u*/dt with the compact
Laplacian. This operator does not change in time, and therefore the FAI is built once per task and reused at every step.

# Problem and parameters

We use a 2⁸ × 2⁸ grid with 100 time steps of dt = 1e-3, and a 2¹⁰ × 2¹⁰ grid with 50 steps of dt = 2e-4.

Each task is one preconditioner: the unpreconditioned baseline, or the FAI with μ = 10⁷ and a bond dimension χ_G ∈ {2,
…, 10}, for each grid size (20 tasks in total). We solve every step with `ResidualALS` (two-site ALS with residual-based
truncation), preconditioning the local problem of every site as in Sec. 4 of the paper, with at most 100 sweeps and a
solution bond dimension capped at 150 (8 × 8) and 200 (10 × 10), and no cutoff on the truncation. The relative residual
of every step must reach 1e-3 (both sizes), and the FAI is built to a residual tolerance of 0.1.

# Inputs and outputs

The task number is `ARGS[1]` (default 1) and `nocond` is `ARGS[2]`. The worker writes `tasks/task_NNN.jld2` (key
`"result"`) under `results/` or `results_nocond/` of this folder: per step, the solve time, the local Krylov iterations,
the relative residual, the bond dimension of the pressure and the divergence norm before and after the pressure
correction; the build time of the preconditioner; and the final fluid state (`"last_snapshot"`), which the plotter
draws. With `nocond` we do not estimate local condition numbers (the timings we use); without it we estimate κ(A) and
κ(P⁻¹A) of the local problems of every site, on the sizes listed in `cond_nodes`. `CHORIN_RESULTS_DIR` overrides the
results folder. The cylinder mask is read from `masks.jld2` in this folder; without it, the worker warns and runs
without obstacle.

Smoke mode (`DIFFERENTT_SMOKE=1`) shrinks the sweep to a few tasks on a tiny problem, through the `SMOKE` branches next
to the constants, and writes under `DIFFERENTT_SMOKE_DIR` (a fresh temporary folder if it is unset), so that the
committed results are never touched. `experiments/smoke.jl` runs the worker, the collector and the plotter of each
benchmark in turn.

# How to run

From the root of the repository:

    DIFFERENTT_SMOKE=1 julia --project=experiments experiments/fai_benchmarks/chorin_pipeline/worker.jl
    julia --project=experiments experiments/fai_benchmarks/chorin_pipeline/worker.jl 3 nocond
    bash experiments/fai_benchmarks/chorin_pipeline/submit.sh nocond

# Resources

The job script requests an array of 20 tasks, each on one node with 20 CPUs on an exclusive node (`--exclusive`), for up
to 36 h (`--time=01-12:00:00`); the worker sets Julia's heap-size hint to 80 % of the memory of the allocation, and
`GC_BALLAST_GB` controls the GC ballast (`common/gc_ballast.jl`). We have not measured the run time or the memory of a
task here.

# Paper

Sec. 5.2 (Chorin projection) uses the `chorin_pipeline_stats` variant of this benchmark (see `paper/`).
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

const use_nocond = length(ARGS) > 1 && ARGS[2] == "nocond"
const default_base_folder = if use_nocond
    joinpath(@__DIR__, "results_nocond")
else
    joinpath(@__DIR__, "results")
end
base_folder = get(ENV, "CHORIN_RESULTS_DIR", default_base_folder)
const TASKS_DIR = joinpath(base_folder, "tasks")
mkpath(TASKS_DIR)

const MASK_FILE = joinpath(@__DIR__, "masks.jld2")
# Note: we assume the user has the masks in MASK_FILE. If not, they must be copied over or generated.

const BOX_SIZE = (2.0, 1.0)
const RE_NUM = 3000
const STENCIL_ORDER = 8

const dt_map = Dict((8, 8) => 0.001, (10, 10) => 0.0002, (12, 12) => 0.00001)

const n_steps_map = Dict((8, 8) => 100, (10, 10) => 50, (12, 12) => 20)

const system_sizes = [(8, 8), (10, 10)]
const cond_nodes = [(8, 8), (10, 10)]
# const system_sizes = [(10, 10)]
# const cond_nodes  = [(10, 10)]

function should_compute_cond(nodes::Tuple{Int,Int}, use_nocond::Bool)::Bool
    return !use_nocond && nodes in cond_nodes
end

# const mu_values = [5e6, 1e7, 5e7, 1e8]
const mu_values = [1e7]

const g_maxdims = 2:10

const maxdims_map = Dict((8, 8) => 150, (10, 10) => 200, (12, 12) => 250)

# WARNING: MAKE SURE TO CHANGE THIS TO >0.0 IF RUNNING ALS -> OOM WILL HAPPEN
const cutoff_map = Dict((8, 8) => 0.0, (10, 10) => 0.0, (12, 12) => 0.0)

const nsweeps_map = Dict((8, 8) => 100, (10, 10) => 100, (12, 12) => 100)

const tol_map = Dict((8, 8) => 1e-3, (10, 10) => 1e-3, (12, 12) => 1e-2)

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

    # We will assume object_mask is generated properly somewhere or just create a dummy one for now.
    # The benchmark requires the true masks to reflect exact conditions.
    # For now, we will create an empty mask if the file doesn't exist, to prevent crashing.
    if isfile(MASK_FILE)
        mask_key = "mask_$(nodes[1])_$(nodes[2])"
        object_mask = load(MASK_FILE, mask_key)
        siteinds(object_mask) != sites && replace_siteinds!(object_mask, sites)
    else
        object_mask = ones_mps(sites)
        @warn "mask not found; using ones_mps (all fluid)"
    end

    mps_vx = tt_mul(object_mask, ones_mps(sites))
    mps_vy = 0.0 * ones_mps(sites)

    bcsvxx = BoundaryCond1d(DirichletBC(1.0), NeumannBC(0.0))
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
# The collectors strip it before building their dataframes.
function last_snapshot(grid, n_steps::Int)
    experiment = grid.experiment
    return Dict{String,Any}(
        "grid" => experiment.grid,
        "u_fields" => experiment.u_fields,
        "p_field" => experiment.p_field,
        "n_steps" => n_steps,
    )
end

function chorin_step!(grid, sweep_kwargs, compute_cond::Bool, tol_val::Float64, step_index::Int, precond=nothing)
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

function run_task(task::TaskSpec; warmup_time::Float64=0.0)
    Random.seed!(task.nodes[1] + task.nodes[2])
    N_STEPS = n_steps_map[task.nodes]
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

            step_res = chorin_step!(grid, sweep_kwargs, compute_cond, tol_val, i)
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
            "warmup_time" => warmup_time,
            "gc_ballast_bytes" => length(GC_BALLAST),
            "heap_size_hint" => Int(Base.JLOptions().heap_size_hint),
            "last_snapshot" => last_snapshot(grid, N_STEPS),
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

        fai_sweep_kwargs = SweepingArgs(; nsweeps=100, maxdim=gdim, cutoff=1e-12)

        fai_updater_kwargs = LocalUpdaterArgs(; ishermitian=true, isposdef=true, rtol=1e-9, krylovdim=1, maxiter=3000)

        fai_resals_solver = ResidualALS(;
            sweep_kwargs=fai_sweep_kwargs, updater_kwargs=fai_updater_kwargs, residual_tol=1e-9
        )

        fai_als_solver = ALS(; sweep_kwargs=fai_sweep_kwargs, updater_kwargs=fai_updater_kwargs)

        fai_amen_solver = AMEn(; sweep_kwargs=fai_sweep_kwargs, updater_kwargs=fai_updater_kwargs, enrichment_maxdim=1)

        D_kwargs_local = (; cutoff=1e-11, oversample=10, maxdim=gdim)
        fai_builder_solver = fai_als_solver

        # The FAI build draws a randomized sketch: seed it so every task of this size builds from
        # the same RNG state.
        Random.seed!(task.nodes[1] + task.nodes[2])
        GC.gc(true)
        build_stats = @timed begin
            unscaledG, D = DifferenTT.fai_dg(
                grid.A, μ, fai_builder_solver, D_kwargs_local; preconditioning=false, residual_tol=faibuild_tol
            )
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
        div_norms_before = Float64[]
        div_norms_after = Float64[]
        div_ratios = Float64[]
        iters_per_site = Float64[]
        pc_conds_per_site = Float64[]
        avg_pc_conds = Float64[]

        for i in 1:N_STEPS
            @info "Step $i / $N_STEPS"
            flush(stderr)

            step_res = chorin_step!(grid, sweep_kwargs, compute_cond, tol_val, i, precond)
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
            "warmup_time" => warmup_time,
            "gc_ballast_bytes" => length(GC_BALLAST),
            "heap_size_hint" => Int(Base.JLOptions().heap_size_hint),
            "last_snapshot" => last_snapshot(grid, N_STEPS),
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
            sweep_kwargs=SweepingArgs(nsweeps=2, maxdim=10, cutoff=1e-8), updater_kwargs=dummy_updater_kwargs
        )
        # Pass residual_tol like the timed build does: it adds a ResidualObserver to the FAI solves,
        # a different code path that would otherwise be compiled inside the timed build.
        unscaledG, D = DifferenTT.fai_dg(
            dummy_grid.A,
            1.0,
            dummy_als_solver,
            (; cutoff=1e-11, oversample=2, maxdim=2);
            preconditioning=false,
            residual_tol=fai_tol_map[(8, 8)],
        )
        dummy_fai = FactorizedPreconditioner(apply(D, unscaledG; maxdim=2))

        # When running with condition numbers, compute time does not really count, so it is better to avoid
        # crashing at precompilation due to dummy preconditioners than to have more accurate timings due to precompilation
        chorin_step!(dummy_grid, SweepingArgs(nsweeps=2, maxdim=10, cutoff=1e-8), false, 1e-3, 0, nothing)
        chorin_step!(dummy_grid, SweepingArgs(nsweeps=2, maxdim=10, cutoff=1e-8), false, 1e-3, 0, dummy_fai)
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

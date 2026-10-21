"""
    Lid-driven cavity pipeline benchmark (global TT-CG): worker

We benchmark the FAI on the stream function Poisson systems -Δψ = ω that appear along the vorticity–streamfunction
evolution of the lid-driven cavity of Kiffner & Jaksch (arXiv:2303.03010) in the QTT format (`INSVorticityExperiment`,
walls as ghost nodes). The operator is the plain Dirichlet five-point Laplacian, which is exactly SPD and does not
change in time, and therefore the FAI is built once per task and reused at every step. The box is the unit square with a
lid of velocity 1 moving in +x, and the fluid starts at rest; at every step we solve for ψ once and advance the
vorticity explicitly with RK4 and the Jensen wall formula. `CAVITY_LIDS=2` selects the doubly-driven cavity of the same
paper (Fig. 5), whose bottom lid moves with velocity -1; its results go to the `*_doublelid` folders.

# Problem and parameters

The grid is 2⁸ × 2⁸ at Re = 1000, with 50 steps of dt = 1e-3. Each task is one preconditioner: the unpreconditioned
baseline, or the FAI with μ = 10⁷ and a bond dimension χ_G ∈ {2, …, 10} (10 tasks). We solve every step with the global
TT-CG solver (`GlobalCG`) instead of `ResidualALS`: the preconditioner acts on a CG solve of the whole stream function
system, rather than on the local problem of every site. The iterates have a bond dimension capped at 120, a cutoff of
1e-13 and at most 3000 iterations, and `GlobalCG` runs to a relative tolerance of 1e-3; the FAI is built to a residual
tolerance of 0.1.

# Inputs and outputs

The task number is `ARGS[1]` (default 1). The worker writes `tasks/task_NNN.jld2` (key `"result"`) under `results/` of
this folder (`results_doublelid/` for two lids), or under `LIDDRIVEN_GLOBALCG_RESULTS_DIR` if set;
`LIDDRIVEN_GLOBALCG_N_STEPS_OVERRIDE` overrides the number of steps.

Smoke mode (`DIFFERENTT_SMOKE=1`) shrinks the sweep to a few tasks on a tiny problem, through the `SMOKE` branches next
to the constants, and writes under `DIFFERENTT_SMOKE_DIR` (a fresh temporary folder if it is unset), so that the
committed results are never touched. `experiments/smoke.jl` runs the worker, the collector and the plotter of each
benchmark in turn.

# How to run

From the root of the repository:

    DIFFERENTT_SMOKE=1 julia --project=experiments experiments/fai_benchmarks/liddriven_pipeline_globalcg/worker.jl
    [CAVITY_LIDS=1|2] julia --project=experiments experiments/fai_benchmarks/liddriven_pipeline_globalcg/worker.jl 3
    [CAVITY_LIDS=1|2] bash experiments/fai_benchmarks/liddriven_pipeline_globalcg/submit.sh

# Resources

The job script requests an array of 10 tasks, each on one node with 20 CPUs on an exclusive node (`--exclusive`), for up
to 36 h (`--time=01-12:00:00`); the worker sets Julia's heap-size hint to 80 % of the memory of the allocation, and
`GC_BALLAST_GB` controls the GC ballast (`common/gc_ballast.jl`). We have not measured the run time or the memory of a
task here.

# Paper

None.
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

include(joinpath(@__DIR__, "..", "..", "common", "smoke.jl"))
# Smoke mode writes under DIFFERENTT_SMOKE_DIR (a fresh temporary folder if unset) instead of this folder.
const OUT_ROOT = output_root(@__DIR__; fresh=true)

# Number of moving lids: 1 is the lid-driven cavity, 2 the doubly-driven cavity (bottom lid moving in -x)
const CAVITY_LIDS = parse(Int, get(ENV, "CAVITY_LIDS", "1"))
CAVITY_LIDS in (1, 2) || error("CAVITY_LIDS must be 1 or 2, got $CAVITY_LIDS")
const RESULTS_SUFFIX = CAVITY_LIDS == 2 ? "_doublelid" : ""

const default_base_folder = joinpath(OUT_ROOT, "results") * RESULTS_SUFFIX
base_folder = get(ENV, "LIDDRIVEN_GLOBALCG_RESULTS_DIR", default_base_folder)
const TASKS_DIR = joinpath(base_folder, "tasks")
mkpath(TASKS_DIR)

const BOX_SIZE = (1.0, 1.0)
# Reynolds number per grid size: 1000 up to 2^10 nodes, 24000 on 2^11 (Kiffner & Jaksch, where 2^11 resolves the
# Kolmogorov scale Re^(-3/4) = 5.2e-4 with h = 4.9e-4)
const re_map = Dict((8, 8) => 1000, (10, 10) => 1000, (11, 11) => 24000, (12, 12) => 1000)
const U_LID = 1.0
const STENCIL_ORDER = 2
const WALL_FORMULA = :jensen

# Four no-slip walls with ψ = 0, the top one moving in +x and, for two lids, the bottom one moving in -x.
# DirichletBC logs at every construction: build the boundary conditions once
const FACES = if CAVITY_LIDS == 1
    cavity_faces(U_LID)
else
    ((NoSlipWall(), NoSlipWall()), (NoSlipWall(; velocity=(-U_LID)), NoSlipWall(; velocity=U_LID)))
end
const CAVITY_BCS = vorticity_boundary_conditions(FACES)

const dt_map = Dict(
    (8, 8) => 0.001,
    (10, 10) => 0.0002,
    (11, 11) => 0.0003, # RK4 limit at Re = 24000 is 6.9e-4 (advection)
    (12, 12) => 0.00001,
)

# Smoke mode: two steps, the baseline and one FAI task.
const n_steps_map = if SMOKE
    Dict((8, 8) => 2, (10, 10) => 2, (11, 11) => 2, (12, 12) => 2)
else
    Dict((8, 8) => 50, (10, 10) => 20, (11, 11) => 10, (12, 12) => 10)
end

# Overridable for fast local verification only — the default reproduces n_steps_map[nodes] exactly.
function resolve_n_steps(nodes::Tuple{Int,Int})
    override = get(ENV, "LIDDRIVEN_GLOBALCG_N_STEPS_OVERRIDE", "")
    return isempty(override) ? n_steps_map[nodes] : parse(Int, override)
end

# const system_sizes = [(8, 8), (10, 10), (12, 12)]
const system_sizes = [(8, 8)]
# 2^11 x 2^11 at Re = 24000 (the array has 1 + length(g_maxdims) = 10 tasks):
# const system_sizes = [(11, 11)]

# const mu_values = [5e6, 1e7, 5e7, 1e8]
const mu_values = [1e7]

const g_maxdims = SMOKE ? (2:2) : (2:10)

const maxdims_map = Dict((8, 8) => 120, (10, 10) => 150, (11, 11) => 200, (12, 12) => 250)

const cutoff_map = Dict((8, 8) => 1e-13, (10, 10) => 1e-13, (11, 11) => 1e-13, (12, 12) => 1e-13)

# Smoke mode caps the global iterations: it checks that the solver runs, not that it converges.
const maxiter_map = if SMOKE
    Dict((8, 8) => 30, (10, 10) => 30, (11, 11) => 30, (12, 12) => 30)
else
    Dict((8, 8) => 3000, (10, 10) => 3000, (11, 11) => 4000, (12, 12) => 5000)
end

const tol_map = Dict((8, 8) => 1e-3, (10, 10) => 1e-3, (11, 11) => 1e-3, (12, 12) => 1e-3)

const fai_tol_map = Dict((8, 8) => 0.1, (10, 10) => 0.1, (11, 11) => 0.1, (12, 12) => 0.2)

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
    # indices even with identical nodes — any MPO built against one (e.g. a FAI
    # preconditioner built from the real grid's A) cannot be applied to the other's MPS/MPO,
    # and ITensorMPS raises "must share a set of site indices". Passing `shared_sites` lets a
    # warmup grid reuse the real grid's indices so cross-grid objects like `precond` stay valid.
    sites = isnothing(shared_sites) ? siteinds(2, sum(nodes)) : shared_sites
    arch = QTTArchitecture(nodes; interleaved=false)
    # Walls are ghost nodes: 2^n interior nodes per axis with h = L / (2^n + 1)
    grid = vorticity_grid(arch, BOX_SIZE, FACES)

    # Fluid at rest, impulsively started lid(s)
    omega_field = ScalarField(zeros_mps(sites), arch, CAVITY_BCS)
    psi_field = ScalarField(zeros_mps(sites), arch, CAVITY_BCS)

    time_stepper = RK4(;
        time_evol_kwargs=TimeEvolArgs(0.0, dt_map[nodes]*max_steps, max_steps), cutoff=1e-11, maxdim=maxdims_map[nodes]
    )
    experiment = INSVorticityExperiment(
        grid,
        omega_field,
        psi_field,
        re_map[nodes],
        FACES;
        wall_formula=WALL_FORMULA,
        stencil_order=STENCIL_ORDER,
        maxdim=maxdims_map[nodes],
        cutoff=1e-11,
    )
    check_cfl(experiment, time_stepper; vmax=experiment.vmax)

    # The stream function operator A = -Δ (Dirichlet, ghost walls) is SPD: no pinning, no RHS corrections
    cache = VorticityCache(experiment)
    A = cache.poisson_op

    # Priming step: at rest ω = 0, so ψ = 0 and the first Poisson system is trivial. Advance one untimed step
    # with ψ = 0 (no solve) so that every timed step has a nonzero right-hand side.
    omega_primed = DifferenTT.vorticity_transport(
        experiment, cache, last(experiment.omega_field), last(experiment.psi_field), time_stepper, 0.0
    )
    experiment.omega_field[length(experiment.omega_field)] = omega_primed

    return (; nodes=nodes, A=A, experiment=experiment, time_stepper=time_stepper, cache=cache, sites=sites)
end

function build_rhs(grid, omega)
    return DifferenTT.poisson_rhs(grid.experiment, grid.cache, omega)
end

# Incompressibility of the velocity (u, v) = (∂ᵧψ, -∂ₓψ) that the vorticity transport uses. The ψ–ω scheme has
# no pressure correction: the divergence ∂ₓu + ∂ᵧv cancels exactly in exact arithmetic, so its norm relative to
# the norm of the two terms measures the TT truncation of the velocity. The sum is a direct sum, so that no
# truncation hides the cancellation.
function velocity_divergence_norms(grid, psi)
    experiment = grid.experiment
    cutoff, maxdim = experiment.cutoff, experiment.maxdim
    d_ops = grid.cache.diff_ops.d_ops
    u, v = DifferenTT.streamfunction_velocity(grid.cache, psi; cutoff=cutoff, maxdim=maxdim)
    ∂ₓu = apply(d_ops[1], u; cutoff=cutoff, maxdim=maxdim)
    ∂ᵧv = apply(d_ops[2], v; cutoff=cutoff, maxdim=maxdim)
    div_norm = norm(+(∂ₓu, ∂ᵧv; alg="directsum"))
    div_scale = sqrt(norm(∂ₓu)^2 + norm(∂ᵧv)^2)
    return div_norm, div_scale
end

function liddriven_step!(grid, global_updater_kwargs, step_index::Int, precond=nothing)
    global_updater_kwargs.numiter_store[] = 0
    experiment = grid.experiment
    omega = last(experiment.omega_field)
    rhs = build_rhs(grid, omega)

    solver = GlobalCG(; updater_kwargs=global_updater_kwargs)

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

    # Overwrite in place: psi_field keeps only the latest stream function (not a time series).
    experiment.psi_field[length(experiment.psi_field)] = psi
    omega_next = DifferenTT.vorticity_transport(experiment, grid.cache, omega, psi, grid.time_stepper, 0.0)
    experiment.omega_field[length(experiment.omega_field)] = omega_next

    omega_maxdim = maxlinkdim(omega_next)

    # Incompressibility of the velocity of this step (outside the timed solve)
    div_norm, div_scale = velocity_divergence_norms(grid, psi)
    div_ratio = div_norm / div_scale
    @info "Divergence norm ratio ‖∂ₓu + ∂ᵧv‖ / ‖(∂ₓu, ∂ᵧv)‖ $div_ratio"
    flush(stderr)
    global_iters = global_updater_kwargs.numiter_store[]

    res_dict = (;
        t,
        compile_t=stats.compile_time,
        gc_t=stats.gctime,
        rel_res,
        sol_maxdim,
        omega_maxdim,
        global_iters,
        div_norm,
        div_scale,
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
        omega_maxdims = Int[]
        div_norms = Float64[]
        div_scales = Float64[]
        div_ratios = Float64[]

        for i in 1:N_STEPS
            @info "Step $i / $N_STEPS"
            flush(stderr)

            step_res = liddriven_step!(grid, global_updater_kwargs, i)
            push!(solve_times, step_res.t)
            push!(compile_times, step_res.compile_t)
            push!(gc_times, step_res.gc_t)
            push!(residuals, step_res.rel_res)
            push!(solution_maxdims, step_res.sol_maxdim)
            push!(omega_maxdims, step_res.omega_maxdim)
            push!(div_norms, step_res.div_norm)
            push!(div_scales, step_res.div_scale)
            push!(div_ratios, step_res.div_ratio)
            push!(global_iters, step_res.global_iters)

            GC.gc(true)
        end

        result_dict = Dict{String,Any}(
            "key" => "baseline",
            "nodes" => task.nodes,
            "cavity_lids" => CAVITY_LIDS,
            "Re" => re_map[task.nodes],
            "total_solve_time" => sum(solve_times),
            "solve_times" => solve_times,
            "global_iters" => global_iters,
            "total_global_iters" => sum(global_iters),
            "residuals" => residuals,
            "solution_maxdims" => solution_maxdims,
            "omega_maxdims" => omega_maxdims,
            "div_norms" => div_norms,
            "div_scales" => div_scales,
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
        omega_maxdims = Int[]
        div_norms = Float64[]
        div_scales = Float64[]
        div_ratios = Float64[]

        for i in 1:N_STEPS
            @info "Step $i / $N_STEPS"
            flush(stderr)

            step_res = liddriven_step!(grid, global_updater_kwargs, i, precond)
            push!(solve_times, step_res.t)
            push!(compile_times, step_res.compile_t)
            push!(gc_times, step_res.gc_t)
            push!(residuals, step_res.rel_res)
            push!(solution_maxdims, step_res.sol_maxdim)
            push!(omega_maxdims, step_res.omega_maxdim)
            push!(div_norms, step_res.div_norm)
            push!(div_scales, step_res.div_scale)
            push!(div_ratios, step_res.div_ratio)
            push!(global_iters, step_res.global_iters)

            GC.gc(true)
        end

        result_dict = Dict{String,Any}(
            "key" => "mu_$(μ)_gdim_$(gdim)",
            "nodes" => task.nodes,
            "cavity_lids" => CAVITY_LIDS,
            "Re" => re_map[task.nodes],
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
            "omega_maxdims" => omega_maxdims,
            "div_norms" => div_norms,
            "div_scales" => div_scales,
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

        liddriven_step!(dummy_grid, dummy_updater_kwargs, 0, nothing)
        liddriven_step!(dummy_grid, dummy_updater_kwargs, 0, dummy_fai)
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

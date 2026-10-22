"""
    Lid-driven cavity validation (vorticity–streamfunction): worker

A solver for incompressible flows has to reproduce a flow whose answer is known, and the lid-driven cavity is the
standard one. We evolve the cavity of Kiffner & Jaksch (arXiv:2303.03010) in the QTT format with the ψ–ω scheme of
`INSVorticityExperiment`: the walls are ghost nodes, the lid moves in +x with velocity 1 and the fluid starts at rest.
At every step we solve the SPD stream function Poisson system -Δψ = ω once, and we advance ω explicitly with RK4 using
the resulting velocity and the Jensen wall vorticity. `ghia_comparison.jl` then compares the final state with Ghia, Ghia
& Shin (1982), and `plot_snapshots.jl` draws the fields. `CAVITY_LIDS=2` selects the doubly-driven cavity of the same
paper (Fig. 5), whose bottom lid moves with velocity -1; its results go to the `*_doublelid` folders.

# Problem and parameters

The box is the unit square at Re = 1000, with second-order stencils, up to time 40 on the 2⁷ × 2⁷ grid (8000 steps of dt
= 0.005) or on the 2⁸ × 2⁸ grid (20000 steps of dt = 0.002); the worker's task list holds one task, the baseline on 2⁷ ×
2⁷ (no preconditioner), with a solution bond dimension capped at 50 and a relative residual of 1e-3. The fields are
saved every 200 steps in `snapshots.jld2`, and a rerun resumes from the last saved step.

# Inputs and outputs

The task number is `ARGS[1]` (default 1) and `nocond` is `ARGS[2]`. The worker writes `tasks/task_NNN.jld2` (key
`"result"`) and `snapshots.jld2` (grid, ω, ψ and the Poisson residual at every saved step) under `results/` or
`results_nocond/` of this folder (with the suffix `_doublelid` for two lids). With `nocond` the local condition numbers
are not estimated. `LIDDRIVEN_CAVITY_N_STEPS_OVERRIDE` stops the run after the given number of steps.

Smoke mode (`DIFFERENTT_SMOKE=1`) shrinks the sweep to a few tasks on a tiny problem, through the `SMOKE` branches next
to the constants, and writes under `DIFFERENTT_SMOKE_DIR` (a fresh temporary folder if it is unset), so that the
committed results are never touched. `experiments/smoke.jl` runs the worker, the collector and the plotter of each
benchmark in turn.

# How to run

From the root of the repository:

    DIFFERENTT_SMOKE=1 julia --project=experiments experiments/validation/liddriven_cavity/worker.jl
    [CAVITY_LIDS=1|2] julia --project=experiments experiments/validation/liddriven_cavity/worker.jl 1 nocond
    [CAVITY_LIDS=1|2] bash experiments/validation/liddriven_cavity/submit.sh nocond

# Resources

The job script requests one task (`--array=1`) on one node with 16 CPUs for up to 36 h (`--time=01-12:00:00`), without a
`--mem` directive, and sets Julia's heap-size hint to 28 GB; after the worker it runs `plot_snapshots.jl` and, for one
lid, `ghia_comparison.jl`. We have not measured the run time or the memory of the task here.

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

include(joinpath(@__DIR__, "..", "..", "common", "smoke.jl"))
# Smoke mode writes under DIFFERENTT_SMOKE_DIR (a fresh temporary folder if unset) instead of this folder.
const OUT_ROOT = output_root(@__DIR__; fresh=true)

const cpus_per_task = parse(Int, get(ENV, "SLURM_CPUS_PER_TASK", "1"))
LinearAlgebra.BLAS.set_num_threads(cpus_per_task)
@info "Set BLAS threads to $cpus_per_task"

# Number of moving lids: 1 is the lid-driven cavity, 2 the doubly-driven cavity (bottom lid moving in -x)
const CAVITY_LIDS = parse(Int, get(ENV, "CAVITY_LIDS", "1"))
CAVITY_LIDS in (1, 2) || error("CAVITY_LIDS must be 1 or 2, got $CAVITY_LIDS")
const RESULTS_SUFFIX = CAVITY_LIDS == 2 ? "_doublelid" : ""

const use_nocond = length(ARGS) > 1 && ARGS[2] == "nocond"
base_folder = (
    if use_nocond
        joinpath(OUT_ROOT, "results_nocond")
    else
        joinpath(OUT_ROOT, "results")
    end
) * RESULTS_SUFFIX
const TASKS_DIR = joinpath(base_folder, "tasks")
mkpath(TASKS_DIR)

const BOX_SIZE = (1.0, 1.0)
const RE_NUM = 1000
const U_LID = 1.0
const STENCIL_ORDER = 2
const WALL_FORMULA = :jensen
const SAVE_EVERY = SMOKE ? 2 : 200

# Four no-slip walls with ψ = 0, the top one moving in +x and, for two lids, the bottom one moving in -x.
# DirichletBC logs at every construction: build the boundary conditions once
const FACES = if CAVITY_LIDS == 1
    cavity_faces(U_LID)
else
    ((NoSlipWall(), NoSlipWall()), (NoSlipWall(; velocity=(-U_LID)), NoSlipWall(; velocity=U_LID)))
end
const CAVITY_BCS = vorticity_boundary_conditions(FACES)

# Smoke mode: 4 steps of dt = 0.005, with a snapshot every 2 steps.
const tf_map = SMOKE ? Dict((7, 7) => 0.02, (8, 8) => 0.02) : Dict((7, 7) => 40.0, (8, 8) => 40.0)

# RK4: dt = 0.005 at 128² and 0.002 at 256², about half the CFL bound
const n_steps_map = SMOKE ? Dict((7, 7) => 4, (8, 8) => 4) : Dict((7, 7) => 8000, (8, 8) => 20000)

# Overridable for fast verification only: stops the run after the given number of steps, keeping the time step
# tf_map[nodes] / n_steps_map[nodes]. The default reproduces n_steps_map[nodes] exactly.
function resolve_n_steps(nodes::Tuple{Int,Int})
    override = get(ENV, "LIDDRIVEN_CAVITY_N_STEPS_OVERRIDE", "")
    return isempty(override) ? n_steps_map[nodes] : parse(Int, override)
end

const system_sizes = [(7, 7), (8, 8)]
const cond_nodes = [(7, 7), (8, 8)]

function should_compute_cond(nodes::Tuple{Int,Int}, use_nocond::Bool)::Bool
    return !use_nocond && nodes in cond_nodes
end

const mu_values = [5e6, 1e7, 5e7, 1e8]
const g_maxdims = 2:10

const maxdims_map = Dict((7, 7) => 50, (8, 8) => 70)

# WARNING: MAKE SURE TO CHANGE THIS TO >0.0 IF RUNNING ALS -> OOM WILL HAPPEN
# const cutoff_map = Dict(
#     (7, 7) => 1e-13,
#     (8, 8) => 1e-13
# )
const cutoff_map = Dict((7, 7) => 0.0, (8, 8) => 0.0)

const nsweeps_map = Dict((7, 7) => 30, (8, 8) => 100)

const tol_map = Dict((7, 7) => 1e-3, (8, 8) => 1e-3)

ITensors.disable_warn_order()
ENV["JULIA_DEBUG"] = ""

struct TaskSpec
    nodes::Tuple{Int,Int}
    is_baseline::Bool
    mu::Float64
    gdim::Int
end

function build_task_list()
    return [TaskSpec((7, 7), true, 0.0, 0)]
end

function build_grid(nodes::NTuple{2,Int}, max_steps::Int)
    sites = siteinds(2, sum(nodes))
    arch = QTTArchitecture(nodes; interleaved=false)
    # Walls are ghost nodes: 2^n interior nodes per axis with h = L / (2^n + 1)
    grid = vorticity_grid(arch, BOX_SIZE, FACES)

    # Fluid at rest, impulsively started lid(s)
    omega_field = ScalarField(zeros_mps(sites), arch, CAVITY_BCS)
    psi_field = ScalarField(zeros_mps(sites), arch, CAVITY_BCS)

    time_stepper = RK4(;
        time_evol_kwargs=TimeEvolArgs(0.0, tf_map[nodes], max_steps), cutoff=1e-12, maxdim=maxdims_map[nodes]
    )
    experiment = INSVorticityExperiment(
        grid,
        omega_field,
        psi_field,
        RE_NUM,
        FACES;
        wall_formula=WALL_FORMULA,
        stencil_order=STENCIL_ORDER,
        maxdim=maxdims_map[nodes],
        cutoff=1e-12,
    )
    check_cfl(experiment, time_stepper; vmax=experiment.vmax)

    cache = VorticityCache(experiment)

    # Priming step: at rest ω = 0, so ψ = 0 and the first Poisson system is trivial. Advance one untimed step
    # with ψ = 0 (no solve) so that every solved step has a nonzero right-hand side.
    omega_primed = DifferenTT.vorticity_transport(
        experiment, cache, last(experiment.omega_field), last(experiment.psi_field), time_stepper, 0.0
    )
    experiment.omega_field[length(experiment.omega_field)] = omega_primed

    return (; nodes=nodes, A=cache.poisson_op, experiment=experiment, time_stepper=time_stepper, cache=cache)
end

function build_rhs(grid, omega)
    return DifferenTT.poisson_rhs(grid.experiment, grid.cache, omega)
end

function liddriven_step!(
    grid,
    sweep_kwargs,
    compute_cond::Bool,
    tol_val::Float64,
    step_index::Int,
    precond=nothing;
    save_residual::Bool=false,
)
    experiment = grid.experiment
    omega = last(experiment.omega_field)
    rhs = build_rhs(grid, omega)

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

    # A and b are bound by linearsolve to the system grid.A * psi = rhs of this step
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

    # ψ carries no dt scaling: the previous stream function is directly the warm start
    previous_psi = last(experiment.psi_field)
    if DifferenTT.iszero_tt(previous_psi)
        Random.seed!(hash((grid.nodes, step_index)))
        psi_guess = random_mps(experiment.psi_field.usites)
    else
        psi_guess = previous_psi
    end

    if isnothing(precond)
        t = @elapsed psi = linearsolve(solver, grid.A, rhs, psi_guess)
    else
        t = @elapsed psi = linearsolve(solver, grid.A, rhs, precond, psi_guess)
    end

    # Compute relative residual using inner product to save memory
    term1 = ITensorMPS.inner(grid.A, psi, grid.A, psi)
    term2 = 2 * real(ITensorMPS.inner(rhs', grid.A, psi))
    term3 = ITensorMPS.inner(rhs, rhs)
    rel_res = sqrt(max(0.0, term1 - term2 + term3)) / norm(rhs)

    @info "Residual $rel_res"

    sol_maxdim = maxlinkdim(psi)
    @info "Sol_maxdim = $sol_maxdim"

    residual_mps = save_residual ? +(apply(grid.A, psi; alg="naive"), -rhs; alg="directsum") : nothing

    # Overwrite in place: psi_field keeps only the latest stream function (not a time series).
    experiment.psi_field[length(experiment.psi_field)] = psi
    omega_next = DifferenTT.vorticity_transport(experiment, grid.cache, omega, psi, grid.time_stepper, 0.0)
    experiment.omega_field[length(experiment.omega_field)] = omega_next

    omega_maxdim = maxlinkdim(omega_next)
    iters_per_site = DifferenTT.positions_sum(numiter_store)

    res_dict = (; t, rel_res, sol_maxdim, omega_maxdim, iters_per_site, residual_mps)

    if compute_cond
        local_conds = isnothing(precond) ? positions_mean_cond(cond_store) : positions_mean_cond(pc_cond_store)
        local_cond_mean = isempty(local_conds) ? NaN : mean(local_conds)
        res_dict = merge(res_dict, (; local_conds, local_cond_mean))
    end
    flush(stderr)
    return res_dict
end

function snapshot_steps(file, prefix::String)
    return [parse(Int, k[(length(prefix)+1):end]) for k in keys(file) if startswith(k, prefix)]
end

function load_kickstart!(grid, base_folder)
    snapshots_path = joinpath(base_folder, "snapshots.jld2")
    start_step = 1
    if isfile(snapshots_path)
        jldopen(snapshots_path, "r") do file
            steps = snapshot_steps(file, "omega_field_")
            last_step = isempty(steps) ? 0 : maximum(steps)
            if last_step > 0
                @info "Kickstarting simulation from step $last_step in snapshots.jld2"
                start_step = last_step + 1
                usites = grid.experiment.omega_field.usites

                loaded_omega = last(file["omega_field_$last_step"])
                siteinds(loaded_omega) != usites && replace_siteinds!(loaded_omega, usites)
                grid.experiment.omega_field[1] = loaded_omega

                loaded_psi = last(file["psi_field_$last_step"])
                siteinds(loaded_psi) != usites && replace_siteinds!(loaded_psi, usites)
                grid.experiment.psi_field[1] = loaded_psi
            end
        end
    end
    return start_step
end

function save_snapshot!(grid, base_folder, step::Int, residual_mps)
    @info "Saving fields for step $step"
    jldopen(joinpath(base_folder, "snapshots.jld2"), "a+") do file
        if !haskey(file, "grid")
            file["grid"] = grid.experiment.grid
        end
        file["omega_field_$step"] = grid.experiment.omega_field
        file["psi_field_$step"] = grid.experiment.psi_field
        file["poisson_residual_$step"] = residual_mps
    end
end

function run_task(task::TaskSpec)
    Random.seed!(task.nodes[1] + task.nodes[2])
    N_STEPS = resolve_n_steps(task.nodes)
    compute_cond = should_compute_cond(task.nodes, use_nocond)
    grid = build_grid(task.nodes, n_steps_map[task.nodes])
    maxdim_val = maxdims_map[task.nodes]
    nsweep_val = nsweeps_map[task.nodes]
    cutoff_val = cutoff_map[task.nodes]
    tol_val = tol_map[task.nodes]

    sweep_kwargs = SweepingArgs(; nsweeps=nsweep_val, maxdim=maxdim_val, cutoff=cutoff_val)

    if task.is_baseline
        @info "Running baseline for $(task.nodes)"

        start_step = load_kickstart!(grid, base_folder)

        solve_times = zeros(Float64, start_step - 1)
        iters = [Float64[] for _ in 1:(start_step-1)]
        residuals = zeros(Float64, start_step - 1)
        solution_maxdims = zeros(Int, start_step - 1)
        omega_maxdims = zeros(Int, start_step - 1)
        iters_per_site = Float64[]
        conds_per_site = Float64[]
        avg_conds = Float64[]

        for i in start_step:N_STEPS
            @info "Step $i / $N_STEPS"

            save_now = i % SAVE_EVERY == 0
            step_res = liddriven_step!(grid, sweep_kwargs, compute_cond, tol_val, i; save_residual=save_now)
            push!(solve_times, step_res.t)
            push!(residuals, step_res.rel_res)
            push!(solution_maxdims, step_res.sol_maxdim)
            push!(omega_maxdims, step_res.omega_maxdim)
            push!(iters, step_res.iters_per_site)

            save_now && save_snapshot!(grid, base_folder, i, step_res.residual_mps)

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
            "cavity_lids" => CAVITY_LIDS,
            "compute_cond" => compute_cond,
            "total_solve_time" => isempty(solve_times) ? 0.0 : sum(solve_times),
            "solve_times" => solve_times,
            "iters" => iters,
            "iters_per_site" => iters_per_site,
            "residuals" => residuals,
            "solution_maxdims" => solution_maxdims,
            "omega_maxdims" => omega_maxdims,
            "precond_build_time" => 0.0,
        )

        if compute_cond
            performed_steps = max(1, N_STEPS - start_step + 1)
            conds_per_site ./= performed_steps
            result_dict["conds_per_site"] = conds_per_site
            result_dict["avg_cond"] = isempty(avg_conds) ? NaN : mean(avg_conds)
        end

        return result_dict
    else
        μ = task.mu
        gdim = task.gdim
        @info "Running (μ=$μ, G_maxdim=$gdim) for $(task.nodes)"

        fai_sweep_kwargs = SweepingArgs(; nsweeps=30, maxdim=gdim, cutoff=1e-12)

        fai_updater_kwargs = LocalUpdaterArgs(; ishermitian=true, isposdef=true, rtol=1e-9, krylovdim=1, maxiter=3000)

        fai_resals_solver = ResidualALS(;
            sweep_kwargs=fai_sweep_kwargs, updater_kwargs=fai_updater_kwargs, residual_tol=1e-9
        )

        fai_als_solver = ALS(; sweep_kwargs=fai_sweep_kwargs, updater_kwargs=fai_updater_kwargs)

        fai_amen_solver = AMEn(; sweep_kwargs=fai_sweep_kwargs, updater_kwargs=fai_updater_kwargs, enrichment_maxdim=1)

        fai_builder_solver = fai_als_solver

        t_build = @elapsed begin
            precond = fai_preconditioner(
                grid.A;
                μ=μ,
                solver=fai_builder_solver,
                D_cutoff=1e-11,
                D_oversample=10,
                D_maxdim=gdim,
                totalmaxdim=gdim,
                preconditioning=false,
                residual_tol=1e-2,
            )
        end

        start_step = load_kickstart!(grid, base_folder)

        solve_times = zeros(Float64, start_step - 1)
        iters = [Float64[] for _ in 1:(start_step-1)]
        residuals = zeros(Float64, start_step - 1)
        solution_maxdims = zeros(Int, start_step - 1)
        omega_maxdims = zeros(Int, start_step - 1)
        iters_per_site = Float64[]
        pc_conds_per_site = Float64[]
        avg_pc_conds = Float64[]

        for i in start_step:N_STEPS
            @info "Step $i / $N_STEPS"

            save_now = i % SAVE_EVERY == 0
            step_res = liddriven_step!(grid, sweep_kwargs, compute_cond, tol_val, i, precond; save_residual=save_now)
            push!(solve_times, step_res.t)
            push!(residuals, step_res.rel_res)
            push!(solution_maxdims, step_res.sol_maxdim)
            push!(omega_maxdims, step_res.omega_maxdim)
            push!(iters, step_res.iters_per_site)

            save_now && save_snapshot!(grid, base_folder, i, step_res.residual_mps)

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
            "cavity_lids" => CAVITY_LIDS,
            "compute_cond" => compute_cond,
            "mu" => μ,
            "G_maxdim" => gdim,
            "precond_build_time" => t_build,
            "total_solve_time" => isempty(solve_times) ? 0.0 : sum(solve_times),
            "solve_times" => solve_times,
            "iters" => iters,
            "iters_per_site" => iters_per_site,
            "residuals" => residuals,
            "solution_maxdims" => solution_maxdims,
            "omega_maxdims" => omega_maxdims,
        )

        if compute_cond
            performed_steps = max(1, N_STEPS - start_step + 1)
            pc_conds_per_site ./= performed_steps
            result_dict["pc_conds_per_site"] = pc_conds_per_site
            result_dict["avg_pc_cond"] = isempty(avg_pc_conds) ? NaN : mean(avg_pc_conds)
        end

        return result_dict
    end
end

function warmup()
    @info "Running JIT warmup..."
    # A throwaway grid with the physical time step (the solves of the real run never see it)
    dummy_grid = build_grid((7, 7), n_steps_map[(7, 7)])
    dummy_updater_kwargs = LocalUpdaterArgs(; ishermitian=true, isposdef=true, rtol=1e-2, maxiter=2, krylovdim=1)
    dummy_als_solver = ALS(;
        sweep_kwargs=SweepingArgs(; nsweeps=2, maxdim=10, cutoff=1e-8), updater_kwargs=dummy_updater_kwargs
    )
    fai_preconditioner(
        dummy_grid.A;
        μ=1.0,
        solver=dummy_als_solver,
        D_cutoff=1e-11,
        D_oversample=2,
        D_maxdim=2,
        totalmaxdim=2,
        preconditioning=false,
    )

    # When running with condition numbers, compute time does not really count, so it is better to avoid
    # crashing at precompilation due to dummy preconditioners than to have more accurate timings due to precompilation
    liddriven_step!(dummy_grid, SweepingArgs(; nsweeps=2, maxdim=10, cutoff=1e-8), false, 1e-3, 0)
    @info "JIT warmup complete."
end

function main()
    task_id = length(ARGS) > 0 ? parse(Int, ARGS[1]) : 1
    all_tasks = build_task_list()
    (1 <= task_id <= length(all_tasks)) || error("Invalid task_id=$task_id")

    warmup()
    result = run_task(all_tasks[task_id])

    save(joinpath(TASKS_DIR, "task_$(lpad(task_id, 3, '0')).jld2"), "result", result)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end

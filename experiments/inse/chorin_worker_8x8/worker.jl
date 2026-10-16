"""
    Chorin Pipeline Benchmark: Parallel Worker
"""

using ITensors
using ITensorMPS
using LinearAlgebra
using Plots
using Statistics
using JLD2
using DifferenTT
using Random

const cpus_per_task = parse(Int, get(ENV, "SLURM_CPUS_PER_TASK", "1"))
LinearAlgebra.BLAS.set_num_threads(cpus_per_task)
@info "Set BLAS threads to $cpus_per_task"

const use_nocond = length(ARGS) > 1 && ARGS[2] == "nocond"
base_folder = use_nocond ? "experiments/inse/chorin_worker_8x8/results_nocond" : "experiments/inse/chorin_worker_8x8/results"
const TASKS_DIR = joinpath(base_folder, "tasks")
mkpath(TASKS_DIR)

const MASK_FILE = "experiments/inse/chorin_worker_8x8/masks.jld2"
# Note: we assume the user has the masks in MASK_FILE. If not, they must be copied over or generated.

const BOX_SIZE = (2.0, 1.0)
const RE_NUM = 3000
const STENCIL_ORDER = 8

const tf_map = Dict(
    (8, 7) => 5.0,
    (10, 10) => 1.0,
    (12, 12) => 0.1
)

const n_steps_map = Dict(
    (8, 7) => 2001,
    (10, 10) => 15,
    (12, 12) => 10
)

const system_sizes = [(8,7), (10, 10), (12, 12)]
const cond_nodes  = [(8,7), (10, 10), (12, 12)]

function should_compute_cond(nodes::Tuple{Int, Int}, use_nocond::Bool)::Bool
    return !use_nocond && nodes in cond_nodes
end

const mu_values = [5e6, 1e7, 5e7, 1e8]
const g_maxdims = 2:10

const maxdims_map = Dict(
    (8, 7) => 200,
    (10, 10) => 250,
    (12, 12) => 400
)

# WARNING: MAKE SURE TO CHANGE THIS TO >0.0 IF RUNNING ALS -> OOM WILL HAPPEN
const cutoff_map = Dict(
    (8,7) => 1e-13,
    (10, 10) => 1e-13,
    (12, 12) => 1e-13
)

const nsweeps_map = Dict(
    (8,7) => 30,
    (10, 10) => 100,
    (12, 12) => 100
)

const tol_map = Dict(
    (8,7) => 1e-4,
    (10, 10) => 1e-4,
    (12, 12) => 1e-4
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
    return [TaskSpec((8,7), true, 0.0, 0)]
end

function build_grid(nodes::NTuple{2, Int}, max_steps::Int)
    sites = siteinds(2, sum(nodes))
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
        object_mask = zeros_mps(sites)
        @warn "mask not found"
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
    
    time_stepper = RK4(time_evol_kwargs=TimeEvolArgs(0.0, tf_map[nodes], max_steps), cutoff=1e-12, maxdim=maxdims_map[nodes])
    experiment = INSExperiment(grid, v_field, p_field, RE_NUM, object_mask; stencil_order=STENCIL_ORDER, maxdim=maxdims_map[nodes], cutoff=1e-12)

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
        object_mask=object_mask
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

function chorin_step!(grid, sweep_kwargs, compute_cond::Bool, tol_val::Float64, step_index::Int, precond=nothing; save_residual::Bool=false)
    experiment = grid.experiment
    masked_u = masked_velocity(grid)
    rhs = build_rhs(grid, masked_u)
    
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

    res_observer = ResidualObserver(grid.A, rhs, tol_val)
    step_sweep_kwargs = SweepingArgs(; NamedTuple(sweep_kwargs)..., observer! = res_observer)
    resals_solver = ResidualALS(sweep_kwargs=step_sweep_kwargs, updater_kwargs=updater_kwargs, residual_tol=0.5 * tol_val)
    amen_solver = AMEn(sweep_kwargs=step_sweep_kwargs, updater_kwargs=updater_kwargs, enrichment_maxdim = 3, enrichment_cutoff = 1e-10)
    als_solver = ALS(sweep_kwargs=step_sweep_kwargs, updater_kwargs=updater_kwargs)

    solver = als_solver
    
    dt = grid.time_stepper.time_evol_kwargs.dt
    previous_p = last(experiment.p_field)
    # previous_p = nothing
    if isnothing(previous_p)
        Random.seed!(hash((grid.nodes, step_index)))
        pdt_guess = random_mps(grid.experiment.u_fields.usites)
    else
        pdt_guess = previous_p * dt
    end

    if isnothing(precond)
        t = @elapsed pdt = linearsolve(solver, grid.A, rhs, pdt_guess)
    else
        t = @elapsed pdt = linearsolve(solver, grid.A, rhs, precond, pdt_guess)
    end
    
    # Compute relative residual using inner product to save memory
    term1 = ITensorMPS.inner(grid.A, pdt, grid.A, pdt)
    term2 = 2 * real(ITensorMPS.inner(rhs', grid.A, pdt))
    term3 = ITensorMPS.inner(rhs, rhs)
    rel_res = sqrt(max(0.0, term1 - term2 + term3)) / norm(rhs)

    @info "Residual $rel_res"

    sol_maxdim = maxlinkdim(pdt)
    @info "Sol_maxdim = $sol_maxdim"

    residual_mps = save_residual ? +(apply(grid.A, pdt; alg="naive"), -rhs; alg="directsum") : nothing
    
    divfree_p = pdt * (1/dt)
    
    # Overwrite in place: p_field keeps only the latest physical p (not a time series).
    experiment.p_field[length(experiment.p_field)] = divfree_p
    div_free_u = DifferenTT.∇p_correction(masked_u, experiment, grid.cache, divfree_p, dt)
    rhs_func = NavierStokesPredictorRHS(experiment, grid.cache.u_diff_ops, grid.cache.u_boundary_null)
    timestepped_u = timestep(grid.time_stepper, div_free_u, rhs_func, 0.0)
    
    experiment.u_fields[length(experiment.u_fields)] = timestepped_u
    
    iters_per_site = DifferenTT.positions_sum(numiter_store)
    
    res_dict = (; t, rel_res, sol_maxdim, iters_per_site, residual_mps)
    
    if compute_cond
        local_conds = isnothing(precond) ? positions_mean_cond(cond_store) : positions_mean_cond(pc_cond_store)
        local_cond_mean = isempty(local_conds) ? NaN : mean(local_conds)
        res_dict = merge(res_dict, (; local_conds, local_cond_mean))
    end
    flush(stderr)
    return res_dict
end

function load_kickstart!(grid, base_folder)
    snapshots_path = joinpath(base_folder, "snapshots.jld2")
    start_step = 1
    if isfile(snapshots_path)
        jldopen(snapshots_path, "r") do file
            last_step = 0
            for k in keys(file)
                if startswith(k, "u_fields_")
                    step = parse(Int, k[10:end])
                    last_step = max(last_step, step)
                end
            end
            if last_step > 0
                @info "Kickstarting simulation from step $last_step in snapshots.jld2"
                start_step = last_step + 1
                loaded_u = last(file["u_fields_$last_step"])
                for m in loaded_u
                    siteinds(m) != grid.experiment.u_fields.usites && replace_siteinds!(m, grid.experiment.u_fields.usites)
                end
                grid.experiment.u_fields[1] = loaded_u
                
                loaded_p = last(file["p_field_$last_step"])
                siteinds(loaded_p) != grid.experiment.p_field.usites && replace_siteinds!(loaded_p, grid.experiment.p_field.usites)
                grid.experiment.p_field[1] = loaded_p
            end
        end
    end
    return start_step
end

function run_task(task::TaskSpec)
    Random.seed!(task.nodes[1] + task.nodes[2])
    N_STEPS = n_steps_map[task.nodes]
    compute_cond = should_compute_cond(task.nodes, use_nocond)
    grid = build_grid(task.nodes, N_STEPS)
    maxdim_val = maxdims_map[task.nodes]
    nsweep_val = nsweeps_map[task.nodes]
    cutoff_val = cutoff_map[task.nodes]
    tol_val = tol_map[task.nodes]
    
    sweep_kwargs = SweepingArgs(nsweeps=nsweep_val, maxdim=maxdim_val, cutoff=cutoff_val)

    if task.is_baseline
        @info "Running baseline for $(task.nodes)"
        
        # Warmup
        chorin_step!(grid, sweep_kwargs, compute_cond, tol_val, 0)
        
        start_step = load_kickstart!(grid, base_folder)
        
        solve_times = zeros(Float64, start_step - 1)
        iters = [Float64[] for _ in 1:(start_step - 1)]
        residuals = zeros(Float64, start_step - 1)
        solution_maxdims = zeros(Int, start_step - 1)
        iters_per_site = Float64[]
        conds_per_site = Float64[]
        avg_conds = Float64[]
        
        for i in start_step:N_STEPS
            @info "Step $i / $N_STEPS"

            step_res = chorin_step!(grid, sweep_kwargs, compute_cond, tol_val, i; save_residual=(i % 200 == 0))
            push!(solve_times, step_res.t)
            push!(residuals, step_res.rel_res)
            push!(solution_maxdims, step_res.sol_maxdim)
            push!(iters, step_res.iters_per_site)

            if i % 200 == 0
                @info "Saving fields for step $i"
                jldopen(joinpath(base_folder, "snapshots.jld2"), "a+") do file
                    if !haskey(file, "grid")
                        file["grid"] = grid.experiment.grid
                    end
                    file["u_fields_$i"] = grid.experiment.u_fields
                    file["p_field_$i"] = grid.experiment.p_field
                    file["poisson_residual_$i"] = step_res.residual_mps
                end
            end
            
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
            "total_solve_time" => isempty(solve_times) ? 0.0 : sum(solve_times),
            "solve_times" => solve_times,
            "iters" => iters,
            "iters_per_site" => iters_per_site,
            "residuals" => residuals,
            "solution_maxdims" => solution_maxdims,
            "precond_build_time" => 0.0
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
        
        fai_sweep_kwargs = SweepingArgs(nsweeps=30, maxdim=gdim, cutoff=1e-12)
        
        fai_updater_kwargs = LocalUpdaterArgs(
            ishermitian = true,
            isposdef = true,
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
        
        t_build = @elapsed begin
            unscaledG, D = DifferenTT.fai_dg(grid.A, μ, fai_builder_solver, D_kwargs_local; preconditioning=false, residual_tol=1e-2)
            Gl = apply(D, unscaledG; maxdim=gdim)
            precond = FactorizedPreconditioner(Gl)
        end
        
        # Warmup
        chorin_step!(grid, sweep_kwargs, compute_cond, tol_val, 0, precond)
        
        start_step = load_kickstart!(grid, base_folder)
        
        solve_times = zeros(Float64, start_step - 1)
        iters = [Float64[] for _ in 1:(start_step - 1)]
        residuals = zeros(Float64, start_step - 1)
        solution_maxdims = zeros(Int, start_step - 1)
        iters_per_site = Float64[]
        pc_conds_per_site = Float64[]
        avg_pc_conds = Float64[]
        
        for i in start_step:N_STEPS
            @info "Step $i / $N_STEPS"
            
            step_res = chorin_step!(grid, sweep_kwargs, compute_cond, tol_val, i, precond)
            push!(solve_times, step_res.t)
            push!(residuals, step_res.rel_res)
            push!(solution_maxdims, step_res.sol_maxdim)
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
            "total_solve_time" => isempty(solve_times) ? 0.0 : sum(solve_times),
            "solve_times" => solve_times,
            "iters" => iters,
            "iters_per_site" => iters_per_site,
            "residuals" => residuals,
            "solution_maxdims" => solution_maxdims
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

task_id = length(ARGS) > 0 ? parse(Int, ARGS[1]) : 1
all_tasks = build_task_list()

if task_id < 1 || task_id > length(all_tasks)
    error("Invalid task_id=$task_id")
end

task = all_tasks[task_id]

@info "Running JIT warmup..."
dummy_grid = build_grid((8,7), 1)
dummy_solver = ResidualALS(
    sweep_kwargs = SweepingArgs(nsweeps=2, maxdim=10, cutoff=1e-8),
    residual_tol=1e-2
)
dummy_updater_kwargs = LocalUpdaterArgs(ishermitian=true, isposdef=true, rtol=1e-2, maxiter=2, krylovdim=1)
dummy_als_solver = ALS(
    sweep_kwargs = SweepingArgs(nsweeps=2, maxdim=10, cutoff=1e-8),
    updater_kwargs = dummy_updater_kwargs
)
DifferenTT.fai_dg(dummy_grid.A, 1.0, dummy_als_solver, (; cutoff=1e-11, oversample=2, maxdim=2); preconditioning=false)

# When running with condition numbers, compute time does not really count, so it is better to avoid
# crashing at precompilation due to dummy preconditioners than to have more accurate timings due to precompilation
chorin_step!(dummy_grid, SweepingArgs(nsweeps=2, maxdim=10, cutoff=1e-8), false, 1e-3, 0)
@info "JIT warmup complete."

result = run_task(task)

output_file = joinpath(TASKS_DIR, "task_$(lpad(task_id, 3, '0')).jld2")
save(output_file, "result", result)
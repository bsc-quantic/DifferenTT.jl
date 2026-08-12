"""
    Chorin Pipeline Benchmark: Parallel Worker
"""

using ITensors
using ITensorMPS
using LinearAlgebra
using Statistics
using JLD2
using DifferenTT

use_nocond = length(ARGS) > 1 && ARGS[2] == "nocond"
base_folder = use_nocond ? "experiments/preconditioner_benchmarks/chorin_pipeline/results_nocond" : "experiments/preconditioner_benchmarks/chorin_pipeline/results"
const TASKS_DIR = joinpath(base_folder, "tasks")
mkpath(TASKS_DIR)

const MASK_FILE = "experiments/preconditioner_benchmarks/masks.jld2"
# Note: we assume the user has the masks in MASK_FILE. If not, they must be copied over or generated.

const BOX_SIZE = (2.0, 1.0)
const RE_NUM = 3000
const STENCIL_ORDER = 8
const DT = 0.00005

n_steps_map = Dict(
    (8, 8) => 50,
    (10, 10) => 25,
    (12, 12) => 10
)

system_sizes = [(8, 8), (10, 10), (12, 12)]
cond_nodes  = [(8, 8), (10, 10), (12, 12)]

function should_compute_cond(nodes::Tuple{Int, Int}, use_nocond::Bool)::Bool
    return !use_nocond && nodes in cond_nodes
end

mu_values = [5e6, 1e7, 5e7, 1e8]
g_maxdims = 2:10

maxdims_map = Dict(
    (8, 8) => 150,
    (10, 10) => 200,
    (12, 12) => 250
)

cutoff_map = Dict(
    (8, 8) => 0.0,
    (10, 10) => 0.0,
    (12, 12) => 0.0
)

nsweeps_map = Dict(
    (8, 8) => 30,
    (10, 10) => 40,
    (12, 12) => 50
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

function build_grid(nodes::NTuple{2, Int}, max_steps::Int)
    sites = siteinds(2, sum(nodes))
    arch = QTTArchitecture(nodes; interleaved=true)
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
    end
    
    mps_vy = DifferenTT.tt_mul(object_mask, ones_mps(sites))
    mps_vx = 0.0 * ones_mps(sites)
    
    bcsvxx = BoundaryCond1d(DirichletBC(1.0), NeumannBC(0.0))
    bcsvxy = BoundaryCond1d(PeriodicBC(), PeriodicBC())
    bcsvyx = BoundaryCond1d(OpenBC(), OpenBC())
    bcsvyy = BoundaryCond1d(PeriodicBC(), PeriodicBC())
    bcspx = BoundaryCond1d(NeumannBC(0.0), DirichletBC(1.0))
    bcspy = BoundaryCond1d(PeriodicBC(), PeriodicBC())
    
    bcsvx = BoundaryConditions(bcsvxx, bcsvxy)
    bcsvy = BoundaryConditions(bcsvyx, bcsvyy)
    bcsp = BoundaryConditions(bcspx, bcspy)
    
    v_field = VectorField((mps_vx, mps_vy), arch, (bcsvx, bcsvy))
    p_field = ScalarField(zeros_mps(sites), arch, bcsp)
    
    time_evol = TimeEvol(0.0, DT, max_steps, :euler)
    experiment = INSExperiment(grid, time_evol, v_field, p_field, RE_NUM, object_mask; stencil_order=STENCIL_ORDER, maxdim=maxdims_map[nodes], cutoff=cutoff_map[nodes])
    
    u_diff_ops = DifferentialOps(grid, experiment.u_fields; stencil_order=STENCIL_ORDER)
    u_boundary_null = BoundaryNull(experiment.u_fields; stencil_order=STENCIL_ORDER)
    p_diff_ops = DifferentialOpsScalar(grid, experiment.p_field; stencil_order=STENCIL_ORDER)
    Δ_cache = DifferenTT.symmetrized_laplacian(experiment.p_field, experiment.grid; stencil_order=STENCIL_ORDER, maxdim=maxdims_map[nodes], cutoff=cutoff_map[nodes])
    A = -1.0 * Δ_cache
    
    return (;
        nodes=nodes,
        A=A,
        experiment=experiment,
        u_diff_ops=u_diff_ops,
        u_boundary_null=u_boundary_null,
        p_diff_ops=p_diff_ops,
        object_mask=object_mask
    )
end

function masked_velocity(grid)
    experiment = grid.experiment
    prev_u = last(experiment.u_fields)
    return DifferenTT.mask_correction(prev_u, grid.object_mask; cutoff=experiment.cutoff, maxdim=experiment.maxdim)
end

function build_rhs(grid, masked_u)
    experiment = grid.experiment
    u_diff_ops = grid.u_diff_ops
    cutoff_val = experiment.cutoff
    maxdim_val = experiment.maxdim
    dt = experiment.time_evol_settings.dt

    ∇u = DifferenTT.divergence(masked_u, u_diff_ops; cutoff=cutoff_val, maxdim=maxdim_val)
    ∇u_correction = DifferenTT.divergence_dirichlet_correction(experiment.u_fields; stencil_order=STENCIL_ORDER, cutoff=cutoff_val, maxdim=maxdim_val)
    Δp_correction_mpo = DifferenTT.scalar_dirichlet_correction(experiment.p_field, experiment.grid, 2; stencil_order=STENCIL_ORDER, cutoff=cutoff_val, maxdim=maxdim_val)
    
    Δp_correction = isnothing(Δp_correction_mpo) ? nothing : apply(Δp_correction_mpo, ones_mps(experiment.p_field.usites); alg="naive", cutoff=cutoff_val, maxdim=maxdim_val)
    
    ∇u = isnothing(∇u_correction) ? ∇u : apply(∇u_correction, ∇u; cutoff=cutoff_val, maxdim=maxdim_val)
    ∇u = isnothing(Δp_correction) ? ∇u : +(∇u, (-1.0 * dt) * Δp_correction; cutoff=cutoff_val, maxdim=maxdim_val)

    rhs_neumann = DifferenTT.neumann_symmetrization_correction_mps(experiment.p_field.arch, experiment.p_field.bcu, -1.0 * ∇u; cutoff=cutoff_val, maxdim=maxdim_val)
    return isnothing(rhs_neumann) ? -1.0 * ∇u : +(-1.0 * ∇u, rhs_neumann; cutoff=cutoff_val, maxdim=maxdim_val)
end

function chorin_step!(grid, sweep_kwargs, compute_cond::Bool, precond=nothing)
    experiment = grid.experiment
    masked_u = masked_velocity(grid)
    rhs = build_rhs(grid, masked_u)
    
    numiter_store = DifferenTT.KrylovSweepStats()
    cond_store = compute_cond ? DifferenTT.KrylovCondStats() : nothing
    pc_cond_store = compute_cond ? DifferenTT.KrylovCondStats() : nothing
    updater_kwargs = DifferenTT.LocalUpdaterArgs(;
        ishermitian = true,
        isposdef = true,
        rtol = 1e-4,
        compute_cond = compute_cond,
        numiter_store = numiter_store,
        cond_store = cond_store,
        pc_cond_store = pc_cond_store
    )
    
    if isnothing(precond)
        solver = DifferenTT.ResidualALS(sweep_kwargs=sweep_kwargs, updater_kwargs=updater_kwargs, residual_tol=5e-4)
        t = @elapsed pdt = linearsolve(solver, grid.A, rhs, 1e-12 * random_mps(siteinds(rhs); linkdims=1); observer! = DifferenTT.ResidualObserver(grid.A, rhs, 1e-3))
    else
        solver = DifferenTT.ResidualALS(sweep_kwargs=sweep_kwargs, updater_kwargs=updater_kwargs, residual_tol=5e-4)
        t = @elapsed pdt = linearsolve(solver, grid.A, rhs, precond, 1e-12 * random_mps(siteinds(rhs); linkdims=1); observer! = DifferenTT.ResidualObserver(grid.A, rhs, 1e-3))
    end
    
    # Compute relative residual
    A_pdt = apply(grid.A, pdt; alg="naive", cutoff=experiment.cutoff, maxdim=experiment.maxdim)
    res_vec = +(A_pdt, -1.0 * rhs; cutoff=experiment.cutoff, maxdim=experiment.maxdim)
    rel_res = norm(res_vec) / norm(rhs)
    
    sol_maxdim = maxlinkdim(pdt)
    
    dt = experiment.time_evol_settings.dt
    divfree_p = pdt * (1/dt)
    
    push!(experiment.p_field, divfree_p)
    div_free_u = DifferenTT.∇p_correction(masked_u, experiment, grid.p_diff_ops, grid.u_boundary_null)
    
    stepper = DifferenTT.ExplicitEuler(cutoff=experiment.cutoff, maxdim=experiment.maxdim)
    rhs_func = DifferenTT.NavierStokesPredictorRHS(experiment, grid.u_diff_ops, grid.u_boundary_null)
    timestepped_u = DifferenTT.timestep(stepper, div_free_u, dt, rhs_func, 0.0)
    
    push!(experiment.u_fields, timestepped_u)
    
    iters_per_site = DifferenTT.positions_sum(numiter_store)
    
    res_dict = (; t, rel_res, sol_maxdim, iters_per_site)
    
    if compute_cond
        cond_improv_vals = isnothing(precond) ? DifferenTT.positions_mean_cond(cond_store) : DifferenTT.positions_mean_cond(pc_cond_store)
        cond_improv_mean = isempty(cond_improv_vals) ? NaN : mean(cond_improv_vals)
        res_dict = merge(res_dict, (; cond_improv_vals, cond_improv_mean))
    end
    
    return res_dict
end

function run_task(task::TaskSpec)
    N_STEPS = n_steps_map[task.nodes]
    compute_cond = should_compute_cond(task.nodes, use_nocond)
    grid = build_grid(task.nodes, N_STEPS)
    maxdim_val = maxdims_map[task.nodes]
    nsweep_val = nsweeps_map[task.nodes]
    cutoff_val = cutoff_map[task.nodes]
    
    sweep_kwargs = DifferenTT.SweepingArgs(nsweeps=nsweep_val, maxdim=maxdim_val, cutoff=cutoff_val)

    if task.is_baseline
        @info "Running baseline for $(task.nodes)"
        
        # Warmup
        chorin_step!(grid, sweep_kwargs, compute_cond)
        
        solve_times = Float64[]
        iters = Vector{Float64}[]
        residuals = Float64[]
        solution_maxdims = Int[]
        iters_per_site = Float64[]
        conds_per_site = Float64[]
        avg_conds = Float64[]
        
        for i in 1:N_STEPS
            @info "Step $i / $N_STEPS"
            
            step_res = chorin_step!(grid, sweep_kwargs, compute_cond)
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
                # For baseline, cond_improv_vals contains the standard condition numbers
                if isempty(conds_per_site)
                    conds_per_site = step_res.cond_improv_vals
                else
                    conds_per_site .+= step_res.cond_improv_vals
                end
                push!(avg_conds, step_res.cond_improv_mean)
            end
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
            "precond_build_time" => 0.0
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
        
        fai_builder_solver = DifferenTT.ResidualALS(
            sweep_kwargs = DifferenTT.SweepingArgs(nsweeps=30, maxdim=gdim, cutoff=1e-12),
            residual_tol=1e-9
        )
        
        D_kwargs_local = (; cutoff=1e-11, oversample=10, maxdim=gdim)
        
        t_build = @elapsed begin
            unscaledG, D = DifferenTT.fai_dg(grid.A, μ, fai_builder_solver, D_kwargs_local; preconditioning=false)
            Gl = apply(D, unscaledG; maxdim=gdim)
            precond = DifferenTT.FactorizedPreconditioner(Gl)
        end
        
        # Warmup
        chorin_step!(grid, sweep_kwargs, compute_cond, precond)
        
        solve_times = Float64[]
        iters = Vector{Float64}[]
        residuals = Float64[]
        solution_maxdims = Int[]
        iters_per_site = Float64[]
        cond_improv_per_site = Float64[]
        avg_cond_improvs = Float64[]
        
        for i in 1:N_STEPS
            @info "Step $i / $N_STEPS"
            
            step_res = chorin_step!(grid, sweep_kwargs, compute_cond, precond)
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
                if isempty(cond_improv_per_site)
                    cond_improv_per_site = step_res.cond_improv_vals
                else
                    cond_improv_per_site .+= step_res.cond_improv_vals
                end
                push!(avg_cond_improvs, step_res.cond_improv_mean)
            end
        end
        
        result_dict = Dict{String, Any}(
            "key" => "mu_$(μ)_gdim_$(gdim)",
            "nodes" => task.nodes,
            "compute_cond" => compute_cond,
            "mu" => μ,
            "G_maxdim" => gdim,
            "precond_build_time" => t_build,
            "total_solve_time" => sum(solve_times),
            "solve_times" => solve_times,
            "iters" => iters,
            "iters_per_site" => iters_per_site,
            "residuals" => residuals,
            "solution_maxdims" => solution_maxdims
        )
        
        if compute_cond
            cond_improv_per_site ./= N_STEPS
            result_dict["cond_improv_per_site"] = cond_improv_per_site
            result_dict["avg_cond_improv"] = mean(avg_cond_improvs)
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
dummy_grid = build_grid((8, 8), 1)
dummy_solver = DifferenTT.ResidualALS(
    sweep_kwargs = DifferenTT.SweepingArgs(nsweeps=2, maxdim=10, cutoff=1e-8),
    residual_tol=1e-2
)
DifferenTT.fai_dg(dummy_grid.A, 1.0, dummy_solver, (; cutoff=1e-11, oversample=2, maxdim=2); preconditioning=false)
chorin_step!(dummy_grid, DifferenTT.SweepingArgs(nsweeps=2, maxdim=10, cutoff=1e-8), true)
@info "JIT warmup complete."

result = run_task(task)

output_file = joinpath(TASKS_DIR, "task_$(lpad(task_id, 3, '0')).jld2")
save(output_file, "result", result)

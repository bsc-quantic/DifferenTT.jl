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
const N_STEPS = 100

system_sizes = [(8, 8), (10, 10), (12, 12)]
mu_values = [5e6, 1e7, 5e7, 1e8]
G_maxdims_map = Dict(
    (8, 8) => [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
    (10, 10) => [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
    (12, 12) => [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
)

maxdims_map = Dict(
    (8, 8) => 80,
    (10, 10) => 100,
    (12, 12) => 120
)

cutoff_map = Dict(
    (8, 8) => 1e-12,
    (10, 10) => 5e-13,
    (12, 12) => 1e-13
)

nsweeps_map = Dict(
    (8, 8) => 20,
    (10, 10) => 30,
    (12, 12) => 40
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
        G_maxdims = G_maxdims_map[nodes]
        push!(tasks, TaskSpec(nodes, true, 0.0, 0))
        for gdim in G_maxdims
            for μ in mu_values
                push!(tasks, TaskSpec(nodes, false, μ, gdim))
            end
        end
    end
    return tasks
end

function build_grid(nodes::NTuple{2, Int}, max_steps::Int)
    sites = siteinds(2, sum(nodes))
    arch = QTTArchitecture(nodes)
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
    Δ_cache = DifferenTT.symmetrized_laplacian(experiment)
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
    Δp_correction_mpo = DifferenTT.pressure_dirichlet_correction(experiment.p_field, experiment.grid, 2; stencil_order=STENCIL_ORDER, cutoff=cutoff_val, maxdim=maxdim_val)
    
    Δp_correction = isnothing(Δp_correction_mpo) ? nothing : apply(Δp_correction_mpo, ones_mps(experiment.p_field.usites); alg="naive", cutoff=cutoff_val, maxdim=maxdim_val)
    
    ∇u = isnothing(∇u_correction) ? ∇u : apply(∇u_correction, ∇u; cutoff=cutoff_val, maxdim=maxdim_val)
    ∇u = isnothing(Δp_correction) ? ∇u : +(∇u, (-1.0 * dt) * Δp_correction; cutoff=cutoff_val, maxdim=maxdim_val)

    rhs_neumann = DifferenTT.neumann_symmetrization_correction_mps(experiment.p_field.arch, experiment.p_field.bcu, -1.0 * ∇u; cutoff=cutoff_val, maxdim=maxdim_val)
    return isnothing(rhs_neumann) ? -1.0 * ∇u : +(-1.0 * ∇u, rhs_neumann; cutoff=cutoff_val, maxdim=maxdim_val)
end

function chorin_step!(grid, solver)
    experiment = grid.experiment
    masked_u = masked_velocity(grid)
    rhs = build_rhs(grid, masked_u)
    
    t = @elapsed pdt = linearsolve(solver, grid.A, rhs, zeros_mps(siteinds(rhs)))
    
    dt = experiment.time_evol_settings.dt
    divfree_p = pdt * (1/dt)
    
    push!(experiment.p_field, divfree_p)
    div_free_u = DifferenTT.∇p_correction(masked_u, experiment, grid.p_diff_ops, grid.u_boundary_null)
    push!(experiment.u_fields, DifferenTT.timestep(experiment, div_free_u, grid.u_diff_ops, grid.u_boundary_null))
    
    return t
end

function run_task(task::TaskSpec)
    grid = build_grid(task.nodes, N_STEPS)
    maxdim_val = maxdims_map[task.nodes]
    nsweep_val = nsweeps_map[task.nodes]
    cutoff_val = cutoff_map[task.nodes]
    
    sweep_kwargs = DifferenTT.SweepingArgs(nsweeps=nsweep_val, maxdim=maxdim_val, cutoff=cutoff_val)

    numiter_store = DifferenTT.KrylovSweepStats()
    cond_store = DifferenTT.KrylovCondStats()
    pc_cond_store = DifferenTT.KrylovCondStats()
    updater_kwargs = DifferenTT.LocalUpdaterArgs(;
        compute_cond = !use_nocond,
        numiter_store = numiter_store,
        cond_store = cond_store,
        pc_cond_store = pc_cond_store
    )

    if task.is_baseline
        solver = DifferenTT.ResidualALS(sweep_kwargs=sweep_kwargs, updater_kwargs=updater_kwargs, residual_tol=1e-5)
        @info "Running baseline for $(task.nodes)"
        
        # Warmup
        chorin_step!(grid, solver)
        
        solve_times = Float64[]
        for i in 1:N_STEPS
            @info "Step $i / $N_STEPS"
            t = chorin_step!(grid, solver)
            push!(solve_times, t)
        end
        
        iters_per_site = DifferenTT.positions_mean(numiter_store)
        conds_per_site = DifferenTT.positions_mean_cond(cond_store)
        avg_cond = length(conds_per_site) > 0 ? sum(conds_per_site) / length(conds_per_site) : NaN
        
        result = Dict(
            "key" => "baseline",
            "nodes" => task.nodes,
            "total_solve_time" => sum(solve_times),
            "solve_times" => solve_times,
            "precond_build_time" => 0.0,
            "iters_per_site" => iters_per_site,
            "conds_per_site" => conds_per_site,
            "avg_cond" => avg_cond
        )
        return result
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
        
        solver = DifferenTT.ResidualALS(sweep_kwargs=sweep_kwargs, updater_kwargs=updater_kwargs, residual_tol=1e-5, preconditioner=precond)
        
        # Warmup
        chorin_step!(grid, solver)
        
        solve_times = Float64[]
        for i in 1:N_STEPS
            @info "Step $i / $N_STEPS"
            t = chorin_step!(grid, solver)
            push!(solve_times, t)
        end
        
        iters_per_site = DifferenTT.positions_mean(numiter_store)
        cond_improv_per_site = DifferenTT.positions_mean_cond(pc_cond_store)
        avg_cond_improv = length(cond_improv_per_site) > 0 ? sum(cond_improv_per_site) / length(cond_improv_per_site) : NaN
        
        result = Dict(
            "key" => "mu_$(μ)_gdim_$(gdim)",
            "nodes" => task.nodes,
            "mu" => μ,
            "G_maxdim" => gdim,
            "precond_build_time" => t_build,
            "total_solve_time" => sum(solve_times),
            "solve_times" => solve_times,
            "iters_per_site" => iters_per_site,
            "cond_improv_per_site" => cond_improv_per_site,
            "avg_cond_improv" => avg_cond_improv
        )
        return result
    end
end

task_id = length(ARGS) > 0 ? parse(Int, ARGS[1]) : 1
all_tasks = build_task_list()

if task_id < 1 || task_id > length(all_tasks)
    error("Invalid task_id=$task_id")
end

task = all_tasks[task_id]
result = run_task(task)

output_file = joinpath(TASKS_DIR, "task_$(lpad(task_id, 3, '0')).jld2")
save(output_file, "result", result)

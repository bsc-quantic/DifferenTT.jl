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

base_folder = "experiments/preconditioner_benchmarks/chorin_pipeline_globalcg/results"
const TASKS_DIR = joinpath(base_folder, "tasks")
mkpath(TASKS_DIR)

const MASK_FILE = "experiments/preconditioner_benchmarks/chorin_pipeline_globalcg/masks.jld2"
# Note: we assume the user has the masks in MASK_FILE. If not, they must be copied over or generated.

const BOX_SIZE = (2.0, 1.0)
const RE_NUM = 3000
const STENCIL_ORDER = 8

const dt_map = Dict(
    (8, 8) => 0.001,
    (10, 10) => 0.0005,
    (12, 12) => 0.00001
)

const n_steps_map = Dict(
    (8, 8) => 30,
    (10, 10) => 10,
    (12, 12) => 5
)

const system_sizes = [(8, 8), (10, 10), (12, 12)]

const mu_values = [5e6, 1e7, 5e7, 1e8]
const g_maxdims = 2:10

const maxdims_map = Dict(
    (8, 8) => 150,
    (10, 10) => 200,
    (12, 12) => 250
)

const cutoff_map = Dict(
    (8, 8) => 1e-13,
    (10, 10) => 1e-13,
    (12, 12) => 1e-13
)

const maxiter_map = Dict(
    (8, 8) => 3000,
    (10, 10) => 5000,
    (12, 12) => 10000
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
    
    mps_vx = tt_mul(object_mask, ones_mps(sites))
    mps_vy = 0.0 * ones_mps(sites)
    
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
    
    time_evol = TimeEvol(0.0, dt_map[nodes]*max_steps, max_steps)
    experiment = INSExperiment(grid, time_evol, v_field, p_field, RE_NUM, object_mask; stencil_order=STENCIL_ORDER, maxdim=maxdims_map[nodes], cutoff=1e-11)
    
    u_diff_ops = DifferentialOps(grid, experiment.u_fields; stencil_order=STENCIL_ORDER)
    u_boundary_null = BoundaryNull(experiment.u_fields; stencil_order=STENCIL_ORDER)
    p_diff_ops = DifferentialOpsScalar(grid, experiment.p_field; stencil_order=STENCIL_ORDER)
    Δ_cache = symmetrized_laplacian(experiment.p_field, experiment.grid; stencil_order=STENCIL_ORDER, maxdim=maxdims_map[nodes], cutoff=cutoff_map[nodes])
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

    ∇u = divergence(masked_u, u_diff_ops; cutoff=cutoff_val, maxdim=maxdim_val)
    ∇u_correction = divergence_dirichlet_correction(experiment.u_fields; stencil_order=STENCIL_ORDER, cutoff=cutoff_val, maxdim=maxdim_val)
    Δp_correction_mpo = scalar_dirichlet_correction(experiment.p_field, experiment.grid, 2; stencil_order=STENCIL_ORDER, cutoff=cutoff_val, maxdim=maxdim_val)
    
    Δp_correction = isnothing(Δp_correction_mpo) ? nothing : apply(Δp_correction_mpo, ones_mps(experiment.p_field.usites); alg="naive", cutoff=cutoff_val, maxdim=maxdim_val)
    
    ∇u = isnothing(∇u_correction) ? ∇u : apply(∇u_correction, ∇u; cutoff=cutoff_val, maxdim=maxdim_val)
    ∇u = isnothing(Δp_correction) ? ∇u : +(∇u, (-1.0 * dt) * Δp_correction; cutoff=cutoff_val, maxdim=maxdim_val)

    rhs_neumann = DifferenTT.neumann_symmetrization_correction_mps(experiment.p_field.arch, experiment.p_field.bcu, -1.0 * ∇u; cutoff=cutoff_val, maxdim=maxdim_val)
    return isnothing(rhs_neumann) ? -1.0 * ∇u : +(-1.0 * ∇u, rhs_neumann; cutoff=cutoff_val, maxdim=maxdim_val)
end

function chorin_step!(grid, global_updater_kwargs, precond=nothing)
    global_updater_kwargs.numiter_store[] = 0
    experiment = grid.experiment
    masked_u = masked_velocity(grid)
    rhs = build_rhs(grid, masked_u)
    
    solver = GlobalCG(updater_kwargs=global_updater_kwargs)
    
    dt = experiment.time_evol_settings.dt
    previous_p = last(experiment.p_field)
    if isnothing(previous_p)
        Random.seed!(grid.nodes[1] + grid.nodes[2])
        pdt_guess = random_mps(siteinds(rhs))
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
    
    divfree_p = pdt * (1/dt)
    
    # Overwrite in place: p_field keeps only the latest physical p (not a time series).
    experiment.p_field[length(experiment.p_field)] = divfree_p
    div_free_u = DifferenTT.∇p_correction(masked_u, experiment, grid.p_diff_ops, grid.u_boundary_null)
    
    stepper = RK4(cutoff=experiment.cutoff, maxdim=experiment.maxdim)
    rhs_func = NavierStokesPredictorRHS(experiment, grid.u_diff_ops, grid.u_boundary_null)
    timestepped_u = timestep(stepper, div_free_u, dt, rhs_func, 0.0)
    
    experiment.u_fields[length(experiment.u_fields)] = timestepped_u
    
    global_iters = global_updater_kwargs.numiter_store[]
    
    res_dict = (; t, rel_res, sol_maxdim, global_iters)
    
    return res_dict
end

function run_task(task::TaskSpec)
    N_STEPS = n_steps_map[task.nodes]
    grid = build_grid(task.nodes, N_STEPS)
    maxdim_val = maxdims_map[task.nodes]
    maxiter_val = maxiter_map[task.nodes]
    cutoff_val = cutoff_map[task.nodes]
    
    global_updater_kwargs = GlobalUpdaterArgs(
        ishermitian = true,
        isposdef = true,
        rtol = 1e-3,
        maxiter = maxiter_val,
        maxdim = maxdim_val,
        cutoff = cutoff_val,
        orth_maxdim = maxdim_val,
        orth_cutoff = 0.1 * cutoff_val,
        numiter_store = Ref{Int}(0)
    )

    if task.is_baseline
        @info "Running baseline for $(task.nodes)"
        
        solve_times = Float64[]
        global_iters = Int[]
        residuals = Float64[]
        solution_maxdims = Int[]
        
        for i in 1:N_STEPS
            @info "Step $i / $N_STEPS"
            
            step_res = chorin_step!(grid, global_updater_kwargs)
            push!(solve_times, step_res.t)
            push!(residuals, step_res.rel_res)
            push!(solution_maxdims, step_res.sol_maxdim)
            push!(global_iters, step_res.global_iters)
            
            GC.gc(true)
        end
        
        result_dict = Dict{String, Any}(
            "key" => "baseline",
            "nodes" => task.nodes,
            "total_solve_time" => sum(solve_times),
            "solve_times" => solve_times,
            "global_iters" => global_iters,
            "total_global_iters" => sum(global_iters),
            "residuals" => residuals,
            "solution_maxdims" => solution_maxdims,
            "precond_build_time" => 0.0
        )
        
        return result_dict
    else
        μ = task.mu
        gdim = task.gdim
        @info "Running (μ=$μ, G_maxdim=$gdim) for $(task.nodes)"
        
        global_fai_solver = GlobalCG(
            updater_kwargs = GlobalUpdaterArgs(
                ishermitian = true,
                isposdef = true,
                rtol = 1e-9,
                maxiter = 3000,
                maxdim = gdim,
                cutoff = 1e-12,
                orth_maxdim = gdim,
                orth_cutoff = 1e-12
            )
        )

        fai_sweep_kwargs = SweepingArgs(nsweeps=100, maxdim=gdim, cutoff=1e-12)
        
        fai_updater_kwargs = LocalUpdaterArgs(
            ishermitian = true,
            isposdef = true,
            compute_cond = false,
            rtol = 1e-5,
            krylovdim = 1,
            maxiter = 3000
        )
        
        resals_solver = ResidualALS(
            sweep_kwargs = fai_sweep_kwargs,
            updater_kwargs = fai_updater_kwargs,
            residual_tol = 1e-9
        )

        als_solver = ALS(
            sweep_kwargs = fai_sweep_kwargs,
            updater_kwargs = fai_updater_kwargs,
        )

        fai_builder_solver = als_solver

        D_kwargs_local = (; cutoff=1e-11, oversample=10, maxdim=gdim)
        
        t_build = @elapsed begin
            unscaledG, D = DifferenTT.fai_dg(grid.A, μ, fai_builder_solver, D_kwargs_local; preconditioning=false, residual_tol=1e-2)
            Gl = apply(D, unscaledG; maxdim=gdim)
            precond = FactorizedPreconditioner(Gl)
        end
        
        solve_times = Float64[]
        global_iters = Int[]
        residuals = Float64[]
        solution_maxdims = Int[]
        
        for i in 1:N_STEPS
            @info "Step $i / $N_STEPS"
            
            step_res = chorin_step!(grid, global_updater_kwargs, precond)
            push!(solve_times, step_res.t)
            push!(residuals, step_res.rel_res)
            push!(solution_maxdims, step_res.sol_maxdim)
            push!(global_iters, step_res.global_iters)
            
            GC.gc(true)
        end
        
        result_dict = Dict{String, Any}(
            "key" => "mu_$(μ)_gdim_$(gdim)",
            "nodes" => task.nodes,
            "mu" => μ,
            "G_maxdim" => gdim,
            "precond_build_time" => t_build,
            "total_solve_time" => sum(solve_times),
            "solve_times" => solve_times,
            "global_iters" => global_iters,
            "total_global_iters" => sum(global_iters),
            "residuals" => residuals,
            "solution_maxdims" => solution_maxdims
        )
        
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
dummy_updater_kwargs = GlobalUpdaterArgs(
    ishermitian = true,
    isposdef = true,
    rtol = 1e-2,
    maxiter = 2,
    maxdim = 10,
    cutoff = 1e-8,
    numiter_store = Ref{Int}(0)
)
dummy_als_solver = ALS(
    sweep_kwargs = SweepingArgs(nsweeps=2, maxdim=10, cutoff=1e-8),
    updater_kwargs = LocalUpdaterArgs(ishermitian=true, isposdef=true, rtol=1e-2, maxiter=2, krylovdim=1)
)

# Baseline path: unpreconditioned GlobalCG inside chorin_step!
chorin_step!(dummy_grid, dummy_updater_kwargs)

# FAI path: FactorizedPreconditioner + chorin_step!
unscaledG, D = DifferenTT.fai_dg(
    dummy_grid.A, 1.0, dummy_als_solver, (; cutoff=1e-11, oversample=2, maxdim=2);
    preconditioning=false, residual_tol=1e-2
)
dummy_precond = FactorizedPreconditioner(apply(D, unscaledG; maxdim=2))
chorin_step!(dummy_grid, dummy_updater_kwargs, dummy_precond)
@info "JIT warmup complete."

result = run_task(task)

output_file = joinpath(TASKS_DIR, "task_$(lpad(task_id, 3, '0')).jld2")
save(output_file, "result", result)

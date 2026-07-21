"""
    TT Poisson Benchmark: Parallel Worker
"""

using ITensors
using ITensorMPS
using LinearAlgebra
using JLD2
using DifferenTT

use_nocond = length(ARGS) > 1 && ARGS[2] == "nocond"
base_folder = use_nocond ? "experiments/preconditioner_benchmarks/tt_poisson/results_nocond" : "experiments/preconditioner_benchmarks/tt_poisson/results"
const TASKS_DIR = joinpath(base_folder, "tasks")
mkpath(TASKS_DIR)

# Sweeping over number of dimensions N
system_dims = [3, 4, 5, 6, 7, 8, 9, 10]
const N_SITES_PER_DIM = 16

# We will test: Baseline, Sinc-Quadrature (M sweep), FAI (mu and gdim sweep)
# To simplify the job array, we just do a few configs per N.
sinc_M_values = [10, 20, 30]
fai_configs = [(1e5, 5), (1e5, 10), (1e6, 5), (1e6, 10)]

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
    # onedim_differential_operator(2, arch, i, bc)
    mat = DifferenTT.onedim_differential_operator(2, arch, i, bc, Array)
    # the exact Laplacian has a minus sign to be positive definite
    # also scale by 1 / dx^2
    mat = -1.0 * mat .* (1.0 / grid.dx[i]^2)
    return mat
end

function build_problem(N::Int)
    box_size = ntuple(i -> 1.0, N)
    nodes = ntuple(i -> N_SITES_PER_DIM, N)
    
    arch = TTArchitecture(nodes)
    grid = Grid(arch, box_size)
    
    # Boundary conditions
    bcs = ntuple(i -> BoundaryCond1d(DirichletBC(0.0), DirichletBC(0.0)), N)
    bc = BoundaryConditions(bcs...)
    
    sites = siteinds(2, N) # Wait, TTArchitecture has var_dims. ITensor siteinds should match var_dims!
    # For TTArchitecture, sites are `siteinds(arch.var_dims)`. Actually DifferenTT uses `siteinds(d, N)`? No, `Index(var_dims[i])`.
    # Let's generate it correctly using ScalarField.
    
    # We need a target function u(x) = prod sin(pi * x_i)
    # Then f(x) = N * pi^2 * prod sin(pi * x_i)
    
    function rhs_func(x)
        val = 1.0
        for i in 1:length(x)
            val *= sin(pi * x[i])
        end
        return length(x) * (pi^2) * val
    end
    
    function sol_func(x)
        val = 1.0
        for i in 1:length(x)
            val *= sin(pi * x[i])
        end
        return val
    end

    # Cross interpolate RHS
    rhs_initialpivot = DifferenTT.optfirstpivot(Float64, rhs_func, grid, arch)
    rhs_tci, _ = DifferenTT.crossinterpolate2(Float64, rhs_func, grid, arch; initialpivots=[rhs_initialpivot])
    rhs_mps = MPS(rhs_tci)
    
    sol_initialpivot = DifferenTT.optfirstpivot(Float64, sol_func, grid, arch)
    sol_tci, _ = DifferenTT.crossinterpolate2(Float64, sol_func, grid, arch; initialpivots=[sol_initialpivot])
    sol_mps = MPS(sol_tci)
    replace_siteinds!(sol_mps, siteinds(rhs_mps))
    
    field = ScalarField(rhs_mps, arch, bc)
    Δ = DifferenTT.symmetrized_laplacian(field, grid)
    A = -1.0 * Δ
    
    A_list = [get_1d_A(i, grid, arch) for i in 1:N]
    
    return A, rhs_mps, sol_mps, A_list, grid
end

function run_task(task::TaskSpec)
    N = task.N
    @info "Building problem for N=$N..."
    A, rhs, sol_mps, A_list, grid = build_problem(N)
    
    sites = siteinds(rhs)
    init_mps = zeros_mps(sites)
    
    sweep_kwargs = DifferenTT.SweepingArgs(nsweeps=30, maxdim=20, cutoff=1e-10)
    
    numiter_store = DifferenTT.KrylovSweepStats()
    cond_store = DifferenTT.KrylovCondStats()
    pc_cond_store = DifferenTT.KrylovCondStats()
    updater_kwargs = DifferenTT.LocalUpdaterArgs(;
        compute_cond = !use_nocond,
        numiter_store = numiter_store,
        cond_store = cond_store,
        pc_cond_store = pc_cond_store
    )
    
    if task.method == "baseline"
        @info "Running Baseline N=$N"
        solver = DifferenTT.ResidualALS(sweep_kwargs=sweep_kwargs, updater_kwargs=updater_kwargs, residual_tol=1e-6)
        
        t_build = 0.0
        t_solve = @elapsed out_mps = linearsolve(solver, A, rhs, init_mps)
        
        diff = +(out_mps, -1.0 * sol_mps; cutoff=1e-10)
        rel_error = norm(diff) / norm(sol_mps)
        
        iters_per_site = DifferenTT.positions_mean(numiter_store)
        conds_per_site = DifferenTT.positions_mean_cond(cond_store)
        avg_cond = length(conds_per_site) > 0 ? sum(conds_per_site) / length(conds_per_site) : NaN
        
        return Dict(
            "key" => "baseline",
            "N" => N,
            "method" => "baseline",
            "precond_build_time" => t_build,
            "solve_time" => t_solve,
            "total_time" => t_solve,
            "relative_error" => rel_error,
            "iters_per_site" => iters_per_site,
            "conds_per_site" => conds_per_site,
            "avg_cond" => avg_cond,
            "solution_maxdim" => maxlinkdim(out_mps)
        )
        
    elseif task.method == "sinc"
        M = Int(task.param1)
        @info "Running Sinc (M=$M) N=$N"
        
        t_build = @elapsed begin
            sinc_pc = DifferenTT.sincquadrature_preconditioner(grid.arch, sites, A_list, M; maxdim=20, cutoff=1e-12)
        end
        
        solver = DifferenTT.ResidualALS(sweep_kwargs=sweep_kwargs, updater_kwargs=updater_kwargs, residual_tol=1e-6, preconditioner=sinc_pc)
        t_solve = @elapsed out_mps = linearsolve(solver, A, rhs, init_mps)
        
        diff = +(out_mps, -1.0 * sol_mps; cutoff=1e-10)
        rel_error = norm(diff) / norm(sol_mps)
        
        iters_per_site = DifferenTT.positions_mean(numiter_store)
        cond_improv_per_site = DifferenTT.positions_mean_cond(pc_cond_store)
        avg_cond_improv = length(cond_improv_per_site) > 0 ? sum(cond_improv_per_site) / length(cond_improv_per_site) : NaN
        
        return Dict(
            "key" => "sinc_M_$(M)",
            "N" => N,
            "method" => "sinc",
            "M" => M,
            "precond_build_time" => t_build,
            "solve_time" => t_solve,
            "total_time" => t_build + t_solve,
            "relative_error" => rel_error,
            "iters_per_site" => iters_per_site,
            "cond_improv_per_site" => cond_improv_per_site,
            "avg_cond_improv" => avg_cond_improv,
            "solution_maxdim" => maxlinkdim(out_mps)
        )
        
    elseif task.method == "fai"
        μ = task.param1
        gdim = task.param2
        @info "Running FAI (μ=$μ, G_maxdim=$gdim) N=$N"
        
        fai_builder_solver = DifferenTT.ResidualALS(
            sweep_kwargs = DifferenTT.SweepingArgs(nsweeps=30, maxdim=gdim, cutoff=1e-12),
            residual_tol=1e-9
        )
        D_kwargs_local = (; cutoff=1e-11, oversample=10, maxdim=gdim)
        
        t_build = @elapsed begin
            unscaledG, D = DifferenTT.fai_dg(A, μ, fai_builder_solver, D_kwargs_local; preconditioning=false)
            Gl = apply(D, unscaledG; maxdim=gdim)
            fai_pc = DifferenTT.FactorizedPreconditioner(Gl)
        end
        
        solver = DifferenTT.ResidualALS(sweep_kwargs=sweep_kwargs, updater_kwargs=updater_kwargs, residual_tol=1e-6, preconditioner=fai_pc)
        t_solve = @elapsed out_mps = linearsolve(solver, A, rhs, init_mps)
        
        diff = +(out_mps, -1.0 * sol_mps; cutoff=1e-10)
        rel_error = norm(diff) / norm(sol_mps)
        
        iters_per_site = DifferenTT.positions_mean(numiter_store)
        cond_improv_per_site = DifferenTT.positions_mean_cond(pc_cond_store)
        avg_cond_improv = length(cond_improv_per_site) > 0 ? sum(cond_improv_per_site) / length(cond_improv_per_site) : NaN
        
        return Dict(
            "key" => "fai_mu_$(μ)_gdim_$(gdim)",
            "N" => N,
            "method" => "fai",
            "mu" => μ,
            "G_maxdim" => gdim,
            "precond_build_time" => t_build,
            "solve_time" => t_solve,
            "total_time" => t_build + t_solve,
            "relative_error" => rel_error,
            "iters_per_site" => iters_per_site,
            "cond_improv_per_site" => cond_improv_per_site,
            "avg_cond_improv" => avg_cond_improv,
            "solution_maxdim" => maxlinkdim(out_mps)
        )
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

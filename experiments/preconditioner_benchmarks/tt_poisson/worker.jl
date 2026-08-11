"""
    TT Poisson Benchmark: Parallel Worker
"""

using ITensors
using ITensorMPS
using LinearAlgebra
using JLD2
using TensorCrossInterpolation
using DifferenTT

use_nocond = length(ARGS) > 1 && ARGS[2] == "nocond"
base_folder = use_nocond ? "experiments/preconditioner_benchmarks/tt_poisson/results_nocond" : "experiments/preconditioner_benchmarks/tt_poisson/results"
const TASKS_DIR = joinpath(base_folder, "tasks")
mkpath(TASKS_DIR)

# Sweeping over number of dimensions N
system_dims = [5, 10, 15, 20, 25, 30, 40, 50]
const N_SITES_PER_DIM = 5

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
    # onedim_differential_operator with an arch::TTArchitecture{N} returns the matrix we need
    mat = DifferenTT.onedim_differential_operator(2, arch, i, bc, Array; stencil_order=2)
    # the exact Laplacian has a minus sign to be positive definite
    # also scale by 1 / dx^2
    mat = -1.0 * mat .* (1.0 / grid.dx[i]^2)
    return mat
end

# TODO Make sure this problem is well suited to be atacked with an MPS of bond dimension neither too large nor too small
# The solution is defined as:
#     u(x) = v(x) ⋅ g(x)²

# where:
#     • v(x) = ∏ᵢ₌₁ᴺ sin(π xᵢ)
#     • g(x) = ∑ᵢ₌₁ᴺ sin(π xᵢ)

function sol_func(x)
    N = length(x)
    val = 1.0
    for i in 1:N
        val *= sin(pi * x[i])
    end
    
    g = 0.0
    for i in 1:N
        g += sin(pi * x[i])
    end
    
    return val * (g^2)
end

# Using the product rule for the Laplacian operator Δ:
#     Δu = Δ(v ⋅ g²) = v ⋅ Δ(g²) + 2 (∇v ⋅ ∇(g²)) + g² ⋅ Δv

# where the components are derived as:
#     • v(x)     = ∏ᵢ₌₁ᴺ sin(π xᵢ)
#     • g(x)     = ∑ᵢ₌₁ᴺ sin(π xᵢ)
#     • Δv       = -N π² v
#     • Δ(g²)    = 2 π² ∑ᵢ₌₁ᴺ cos²(π xᵢ) - 2 π² g²
#     • ∇v ⋅ ∇(g²) = 2 π² g ∑ᵢ₌₁ᴺ [ cos²(π xᵢ) ∏ⱼneq i sin(π xⱼ) ]

# The function returns f(x) = -Δexact_solution(x).

function rhs_func(x)
    N = length(x)
    
    v = 1.0
    for i in 1:N
        v *= sin(pi * x[i])
    end
    
    g = 0.0
    for i in 1:N
        g += sin(pi * x[i])
    end
    
    delta_g2 = 0.0
    for i in 1:N
        delta_g2 += 2 * pi^2 * cos(pi * x[i])^2
    end
    delta_g2 -= 2 * pi^2 * g^2
    
    grad_v_dot_grad_g2 = 0.0
    for i in 1:N
        prod_j = 1.0
        for j in 1:N
            if j != i
                prod_j *= sin(pi * x[j])
            end
        end
        grad_v_dot_grad_g2 += 2 * g * pi^2 * cos(pi * x[i])^2 * prod_j
    end
    
    delta_v = -N * pi^2 * v
    
    delta_u = v * delta_g2 + 2 * grad_v_dot_grad_g2 + g^2 * delta_v
    
    return -delta_u
end

function build_problem(N::Int)
    box_size = ntuple(i -> 1.0, N)
    nodes = ntuple(i -> N_SITES_PER_DIM, N)
    
    arch = TTArchitecture(nodes)
    grid = Grid(arch, box_size)
    
    # Boundary conditions
    bcs = ntuple(i -> BoundaryCond1d(DirichletBC(0.0), DirichletBC(0.0)), N)
    bc = BoundaryConditions(bcs...)
    
    sites = siteinds(arch)
    
    # Cross interpolate RHS
    rhs_initialpivot = optfirstpivot(Float64, rhs_func, grid, arch)
    rhs_tci, _ = crossinterpolate2(Float64, rhs_func, grid, arch; initialpivots=[rhs_initialpivot])
    rhs_mps = MPS(rhs_tci)
    
    sol_initialpivot = optfirstpivot(Float64, sol_func, grid, arch)
    sol_tci, _ = crossinterpolate2(Float64, sol_func, grid, arch; initialpivots=[sol_initialpivot])
    sol_mps = MPS(sol_tci)
    replace_siteinds!(sol_mps, siteinds(rhs_mps))
    
    field = ScalarField(rhs_mps, arch, bc)
    Δ = symmetrized_laplacian(field, grid; stencil_order=2)
    A = -1.0 * Δ
    
    # Incorporate Dirichlet boundary corrections through library functions (even if they do not do anything)
    Δ_correction_mpo = DifferenTT.scalar_dirichlet_correction(field, grid, 2; stencil_order=2, cutoff=1e-12)
    Δ_correction = isnothing(Δ_correction_mpo) ? nothing : apply(Δ_correction_mpo, ones_mps(field.usites); alg="naive", cutoff=1e-12)
    rhs_mps = isnothing(Δ_correction) ? rhs_mps : +(rhs_mps, Δ_correction; cutoff=1e-12)
    
    # Symmetrize the RHS
    rhs_mps = DifferenTT.symmetrize_rhs(field, rhs_mps; cutoff=1e-12)
    
    # Redefine RHS as A * sol_mps to avoid finite difference truncation errors dominating
    rhs_mps = apply(A, sol_mps; alg="naive", cutoff=1e-12)
    
    A_list = [get_1d_A(i, grid, arch) for i in 1:N]
    
    return A, rhs_mps, sol_mps, A_list, grid, arch
end

function run_task(task::TaskSpec)
    N = task.N
    @info "Building problem for N=$N..."
    A, rhs, sol_mps, A_list, grid, arch = build_problem(N)
    
    sites = siteinds(rhs)
    init_mps = random_mps(sites)
    
    # Tight maxdim truncation may mess up the residual truncation...
    sweep_kwargs = SweepingArgs(nsweeps=50, maxdim=50, cutoff=0.0)
    
    numiter_store = KrylovSweepStats()
    cond_store = use_nocond ? nothing : KrylovCondStats()
    pc_cond_store = use_nocond ? nothing : KrylovCondStats()
    updater_kwargs = LocalUpdaterArgs(
        rtol = 1e-3,
        compute_cond = !use_nocond,
        numiter_store = numiter_store,
        cond_store = cond_store,
        pc_cond_store = pc_cond_store
    )

    residual_observer = ResidualObserver(A, rhs, 1e-2)
    
    if task.method == "baseline"
        @info "Running Baseline N=$N"
        solver = ResidualALS(sweep_kwargs=sweep_kwargs, updater_kwargs=updater_kwargs, residual_tol=5e-3)
        
        t_build = 0.0
        t_solve = @elapsed out_mps = linearsolve(solver, A, rhs, init_mps; observer! = residual_observer)
        
        diff = +(out_mps, -1.0 * sol_mps; cutoff=1e-10)
        rel_error = norm(diff) / norm(sol_mps)
        
        iters_per_site = DifferenTT.positions_sum(numiter_store)
        
        result_dict = Dict{String, Any}(
            "key" => "baseline",
            "N" => N,
            "method" => "baseline",
            "precond_build_time" => t_build,
            "solve_time" => t_solve,
            "total_time" => t_solve,
            "relative_error" => rel_error,
            "iters_per_site" => iters_per_site,
            "solution_maxdim" => maxlinkdim(out_mps)
        )
        
        if !use_nocond
            conds_per_site = positions_mean_cond(cond_store)
            avg_cond = length(conds_per_site) > 0 ? sum(conds_per_site) / length(conds_per_site) : NaN
            result_dict["conds_per_site"] = conds_per_site
            result_dict["avg_cond"] = avg_cond
        end
        
        return result_dict
        
    elseif task.method == "sinc"
        M = Int(task.param1)
        @info "Running Sinc (M=$M) N=$N"
        
        t_build = @elapsed begin
            sinc_pc = sincquadrature_preconditioner(arch, sites, A_list, M; maxdim=20, cutoff=1e-12)
        end
        
        solver = ResidualALS(sweep_kwargs=sweep_kwargs, updater_kwargs=updater_kwargs, residual_tol=1e-6)
        t_solve = @elapsed out_mps = linearsolve(solver, A, rhs, sinc_pc, init_mps; observer! = residual_observer)
        
        diff = +(out_mps, -1.0 * sol_mps; cutoff=1e-10)
        rel_error = norm(diff) / norm(sol_mps)
        
        iters_per_site = DifferenTT.positions_sum(numiter_store)
        
        result_dict = Dict{String, Any}(
            "key" => "sinc_M_$(M)",
            "N" => N,
            "method" => "sinc",
            "M" => M,
            "precond_build_time" => t_build,
            "solve_time" => t_solve,
            "total_time" => t_build + t_solve,
            "relative_error" => rel_error,
            "iters_per_site" => iters_per_site,
            "solution_maxdim" => maxlinkdim(out_mps)
        )
        
        if !use_nocond
            cond_improv_per_site = positions_mean_cond(pc_cond_store)
            avg_cond_improv = length(cond_improv_per_site) > 0 ? sum(cond_improv_per_site) / length(cond_improv_per_site) : NaN
            result_dict["cond_improv_per_site"] = cond_improv_per_site
            result_dict["avg_cond_improv"] = avg_cond_improv
        end
        
        return result_dict
        
    elseif task.method == "fai"
        μ = task.param1
        gdim = task.param2
        @info "Running FAI (μ=$μ, G_maxdim=$gdim) N=$N"
        
        fai_builder_solver = ResidualALS(
            sweep_kwargs = SweepingArgs(nsweeps=30, maxdim=gdim, cutoff=1e-12),
            residual_tol=1e-5
        )

        D_kwargs_local = (; cutoff=1e-11, oversample=10, maxdim=gdim)
        
        t_build = @elapsed begin
            fai_pc = fai_preconditioner(A, μ, fai_builder_solver, D_kwargs_local; preconditioning=false)
        end
        
        solver = ResidualALS(sweep_kwargs=sweep_kwargs, updater_kwargs=updater_kwargs, residual_tol=1e-6)
        t_solve = @elapsed out_mps = linearsolve(solver, A, rhs, fai_pc, init_mps; observer! = residual_observer)
        
        diff = +(out_mps, -1.0 * sol_mps; cutoff=1e-10)
        rel_error = norm(diff) / norm(sol_mps)
        
        iters_per_site = DifferenTT.positions_sum(numiter_store)
        
        result_dict = Dict{String, Any}(
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
            "solution_maxdim" => maxlinkdim(out_mps)
        )
        
        if !use_nocond
            cond_improv_per_site = positions_mean_cond(pc_cond_store)
            avg_cond_improv = length(cond_improv_per_site) > 0 ? sum(cond_improv_per_site) / length(cond_improv_per_site) : NaN
            result_dict["cond_improv_per_site"] = cond_improv_per_site
            result_dict["avg_cond_improv"] = avg_cond_improv
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
dummy_N = 2
dummy_A, dummy_rhs, dummy_sol, dummy_A_list, dummy_grid, dummy_arch = build_problem(dummy_N)
dummy_sites = siteinds(dummy_rhs)
dummy_init = random_mps(dummy_sites)
dummy_solver = ResidualALS(
    sweep_kwargs = SweepingArgs(nsweeps=2, maxdim=10, cutoff=1e-8),
    residual_tol=1e-2
)
sincquadrature_preconditioner(dummy_arch, dummy_sites, dummy_A_list, 2; maxdim=10, cutoff=1e-12)
fai_preconditioner(dummy_A, 1.0, dummy_solver, (; cutoff=1e-11, oversample=2, maxdim=2); preconditioning=false)
linearsolve(dummy_solver, dummy_A, dummy_rhs, dummy_init)
@info "JIT warmup complete."

result = run_task(task)

output_file = joinpath(TASKS_DIR, "task_$(lpad(task_id, 3, '0')).jld2")
save(output_file, "result", result)

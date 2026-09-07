"""
    TT Poisson Benchmark: Parallel Worker
"""

using ITensors
using ITensorMPS
using LinearAlgebra
using JLD2
using TensorCrossInterpolation
using DifferenTT
using Random

base_folder = "experiments/preconditioner_benchmarks/tt_poisson_global/results"
const TASKS_DIR = joinpath(base_folder, "tasks")
mkpath(TASKS_DIR)

# Sweeping over number of dimensions N
const system_dims = [5, 10, 15, 20, 25, 30, 40, 50]
const N_NODES_PER_DIM = 20

const maxdims_map = Dict(
    5 => 20,
    10 => 30,
    15 => 30,
    20 => 30,
    25 => 50,
    30 => 50,
    40 => 50,
    50 => 50
)

const nsweeps_map = Dict(
    5 => 50,
    10 => 50,
    15 => 50,
    20 => 60,
    25 => 60,
    30 => 80,
    40 => 100,
    50 => 120
)

const rtol_map = Dict(
    5 => 1e-5,
    10 => 1e-5,
    15 => 1e-5,
    20 => 1e-5,
    25 => 1e-5,
    30 => 1e-5,
    40 => 1e-5,
    50 => 1e-5
)

const maxiter_map = Dict(
    5 => 2000,
    10 => 2000,
    15 => 2000,
    20 => 3000,
    25 => 3000,
    30 => 3000,
    40 => 5000,
    50 => 5000
)

# We will test: Baseline, Sinc-Quadrature (M sweep), FAI (mu and gdim sweep)
# To simplify the job array, we just do a few configs per N.
const sinc_M_values = [1, 2, 3, 5, 7, 10]
const fai_configs = [(1e6, 4), (1e6, 6), (1e6, 8), (1e6, 10)]

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

function get_gaussian_sol(N::Int, R::Int)
    Random.seed!(42)
    centers = rand(R, N) .* 0.6 .+ 0.2
    alphas = rand(R, N) .* 4.0 .+ 1.0
    
    function sol(x)
        val = 0.0
        for k in 1:R
            term = 1.0
            for j in 1:N
                term *= exp(-alphas[k,j] * (x[j] - centers[k,j])^2)
            end
            val += term
        end
        # Apply Dirichlet boundary condition envelope
        env = 1.0
        for j in 1:N
            env *= sin(pi * x[j])
        end
        return val * env
    end
    return sol
end

function get_coulomb_sol(N::Int, ε::Float64=0.05)
    c = fill(0.5, N)
    function sol(x)
        r2 = sum((x[j] - c[j])^2 for j in 1:N)
        env = prod(sin(pi * x[j]) for j in 1:N)
        return (1.0 / sqrt(r2 + ε^2)) * env
    end
    return sol
end

function build_problem(N::Int)
    box_size = ntuple(i -> 1.0, N)
    nodes = ntuple(i -> N_NODES_PER_DIM, N)
    
    arch = TTArchitecture(nodes)
    grid = Grid(arch, box_size)
    
    # Boundary conditions
    bcs = ntuple(i -> BoundaryCond1d(DirichletBC(0.0), DirichletBC(0.0)), N)
    bc = BoundaryConditions(bcs...)
    
    sites = siteinds(arch)
    
    # sol_func = get_gaussian_sol(N, 30)
    sol_func = get_coulomb_sol(N, 0.02)
    
    maxdim_N = get(maxdims_map, N, 10)
    
    sol_initialpivot = optfirstpivot(Float64, sol_func, grid, arch)
    # Cap ttcross by maxdim_N to avoid OOM
    sol_tci, _ = crossinterpolate2(Float64, sol_func, grid, arch; initialpivots=[sol_initialpivot], maxbonddim=maxdim_N, tolerance=1e-12)
    sol_mps = MPS(sol_tci)
    replace_siteinds!(sol_mps, sites)
    
    field = ScalarField(sol_mps, arch, bc)
    Δ = symmetrized_laplacian(field, grid; stencil_order=2)
    @show maxlinkdim(Δ)

    A = -1.0 * Δ
    
    # Discrete RHS A * sol_mps so FD truncation error does not dominate the solve error.
    rhs_mps = apply(A, sol_mps; alg="naive")
    
    # Incorporate Dirichlet boundary corrections through library functions (even if they do not do anything)
    # Do not cap maxdim here so we don't truncate rhs_mps and mix truncation error with linsolve error
    Δ_correction_mpo = scalar_dirichlet_correction(field, grid, 2; stencil_order=2, cutoff=0.0)
    Δ_correction = isnothing(Δ_correction_mpo) ? nothing : apply(Δ_correction_mpo, ones_mps(field.usites); alg="naive")
    rhs_mps = isnothing(Δ_correction) ? rhs_mps : +(rhs_mps, Δ_correction; alg="directsum")
    
    # Symmetrize the RHS
    rhs_mps = symmetrize_rhs(field, rhs_mps)
    
    A_list = [get_1d_A(i, grid, arch) for i in 1:N]
    
    return A, rhs_mps, sol_mps, A_list, grid, arch
end


function run_task(task::TaskSpec)
    N = task.N
    @info "Building problem for N=$N..."
    A, rhs, sol_mps, A_list, grid, arch = build_problem(N)
    
    sites = siteinds(rhs)
    Random.seed!(task.N)
    init_mps = random_mps(sites)
    
    global_updater_kwargs = GlobalUpdaterArgs(
        ishermitian = true,
        isposdef = true,
        rtol = rtol_map[N],
        maxiter = maxiter_map[N],
        maxdim = maxdims_map[N],
        cutoff = 1e-13,
        orth_maxdim = maxdims_map[N],
        orth_cutoff =  1e-13,
        numiter_store = Ref{Int}(0)
    )

    global_solver = GlobalCG(updater_kwargs=global_updater_kwargs)

    solver = global_solver
    
    if task.method == "baseline"
        @info "Running Baseline N=$N"
        
        t_build = 0.0
        t_solve = @elapsed out_mps = linearsolve(solver, A, rhs, init_mps)
        
        term1 = ITensorMPS.inner(A, out_mps, A, out_mps)
        term2 = 2 * real(ITensorMPS.inner(rhs', A, out_mps))
        term3 = ITensorMPS.inner(rhs, rhs)
        rel_error = sqrt(max(0.0, term1 - term2 + term3)) / norm(rhs)
        
        @info "Residual $rel_error"
        @info "Max bond dimension = $(maxlinkdim(out_mps))"
        
        result_dict = Dict{String, Any}(
            "key" => "baseline",
            "N" => N,
            "method" => "baseline",
            "precond_build_time" => t_build,
            "solve_time" => t_solve,
            "total_time" => t_solve,
            "relative_error" => rel_error,
            "global_iters" => global_updater_kwargs.numiter_store[],
            "solution_maxdim" => maxlinkdim(out_mps)
        )
        
        return result_dict
        
    elseif task.method == "sinc"
        M = Int(task.param1)
        @info "Running Sinc (M=$M) N=$N"
        
        t_build = @elapsed begin
            sinc_pc = sincquadrature_preconditioner(arch, sites, A_list, M)
        end
        
        t_solve = @elapsed out_mps = linearsolve(solver, A, rhs, sinc_pc, init_mps)
        
        term1 = ITensorMPS.inner(A, out_mps, A, out_mps)
        term2 = 2 * real(ITensorMPS.inner(rhs', A, out_mps))
        term3 = ITensorMPS.inner(rhs, rhs)
        rel_error = sqrt(max(0.0, term1 - term2 + term3)) / norm(rhs)
        
        @info "Residual $rel_error"
        @info "Max bond dimension = $(maxlinkdim(out_mps))"
        
        result_dict = Dict{String, Any}(
            "key" => "sinc_M_$(M)",
            "N" => N,
            "method" => "sinc",
            "M" => M,
            "precond_build_time" => t_build,
            "solve_time" => t_solve,
            "total_time" => t_build + t_solve,
            "relative_error" => rel_error,
            "global_iters" => global_updater_kwargs.numiter_store[],
            "solution_maxdim" => maxlinkdim(out_mps)
        )
        
        return result_dict
        
    elseif task.method == "fai"
        μ = task.param1
        gdim = task.param2
        @info "Running FAI (μ=$μ, G_maxdim=$gdim) N=$N"
        
        fai_sweep_kwargs = SweepingArgs(nsweeps=50, maxdim=gdim, cutoff=1e-12)
        
        fai_updater_kwargs = LocalUpdaterArgs(
            ishermitian = true,
            isposdef = true,
            compute_cond = false,
            rtol = 1e-2,
            krylovdim = 1,
            maxiter = 1000
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

        fai_global_solver = GlobalCG(
            updater_kwargs = GlobalUpdaterArgs(
                ishermitian = true,
                isposdef = true,
                rtol = 5e-2,
                maxiter = 3000,
                maxdim = gdim,
                cutoff = 1e-12,
                orth_maxdim = gdim,
                orth_cutoff = 0.1 * 1e-12
            )
        )

        D_kwargs_local = (; cutoff=1e-11, oversample=10, maxdim=gdim)
        fai_builder_solver = fai_als_solver
        
        t_build = @elapsed begin
            fai_pc = fai_preconditioner(A, μ, fai_builder_solver, D_kwargs_local; totalmaxdim=gdim, preconditioning=false, residual_tol=5e-2)
        end
        
        t_solve = @elapsed out_mps = linearsolve(solver, A, rhs, fai_pc, init_mps)
        
        term1 = ITensorMPS.inner(A, out_mps, A, out_mps)
        term2 = 2 * real(ITensorMPS.inner(rhs', A, out_mps))
        term3 = ITensorMPS.inner(rhs, rhs)
        rel_error = sqrt(max(0.0, term1 - term2 + term3)) / norm(rhs)
        
        @info "Residual $rel_error"
        @info "Max bond dimension = $(maxlinkdim(out_mps))"

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
            "global_iters" => global_updater_kwargs.numiter_store[],
            "solution_maxdim" => maxlinkdim(out_mps)
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
dummy_N = 2
dummy_A, dummy_rhs, dummy_sol, dummy_A_list, dummy_grid, dummy_arch = build_problem(dummy_N)
dummy_sites = siteinds(dummy_rhs)
dummy_init = random_mps(dummy_sites)
dummy_updater_kwargs = GlobalUpdaterArgs(
    ishermitian = true,
    isposdef = true,
    rtol = 1e-2,
    maxiter = 2,
    maxdim = 10,
    cutoff = 1e-8,
    orth_maxdim = 10,
    orth_cutoff = 1e-8,
    numiter_store = Ref{Int}(0)
)
dummy_solver = GlobalCG(updater_kwargs = dummy_updater_kwargs)
sincquadrature_preconditioner(dummy_arch, dummy_sites, dummy_A_list, 2)
fai_preconditioner(dummy_A, 1.0, dummy_solver, (; cutoff=1e-11, oversample=2, maxdim=2); preconditioning=false)
linearsolve(dummy_solver, dummy_A, dummy_rhs, dummy_init)
@info "JIT warmup complete."

result = run_task(task)

output_file = joinpath(TASKS_DIR, "task_$(lpad(task_id, 3, '0')).jld2")
save(output_file, "result", result)

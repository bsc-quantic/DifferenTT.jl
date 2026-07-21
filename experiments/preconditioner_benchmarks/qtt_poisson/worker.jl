"""
    QTT Poisson Benchmark: Parallel Worker
"""

using ITensors
using ITensorMPS
using LinearAlgebra
using JLD2
using DifferenTT

use_nocond = length(ARGS) > 1 && ARGS[2] == "nocond"
base_folder = use_nocond ? "experiments/preconditioner_benchmarks/qtt_poisson/results_nocond" : "experiments/preconditioner_benchmarks/qtt_poisson/results"
tasks_folder = joinpath(base_folder, "tasks")
mkpath(tasks_folder)

# We extrapolate system sizes from FAI. QTT uses 1D. (16,), (20,), (24,) correspond to 2^16, 2^20, 2^24.
system_sizes = [(16,), (20,), (24,)]
mu_values = [1e4, 5e4, 1e5, 5e5, 1e6]

G_maxdims_map = Dict(
    (16,) => [2, 3, 4, 5, 6, 7, 8],
    (20,) => [2, 3, 4, 5, 6, 7, 8],
    (24,) => [2, 3, 4, 5, 6, 7, 8]
)

cutoff = 1e-13
box_size = (1.0,)

Glinsolve_kwargs_base = (;
    sweep_kwargs = DifferenTT.SweepingArgs(nsweeps=30, maxdim=20, cutoff=1e-12),
    residual_tol = 1e-9
)

D_kwargs = (;
    cutoff=1e-11,
    oversample=10,
)

ITensors.disable_warn_order()
ENV["JULIA_DEBUG"] = ""

struct TaskSpec
    nodes::Tuple{Int}
    is_baseline::Bool
    mu::Float64
    gdim::Int
end

function build_task_list()
    tasks = TaskSpec[]
    for nodes in system_sizes
        G_maxdims = G_maxdims_map[nodes]
        for gdim in G_maxdims
            for μ in mu_values
                push!(tasks, TaskSpec(nodes, false, μ, gdim))
            end
        end
    end
    return tasks
end

function setup_problem(nodes)
    arch = QTTArchitecture(nodes)
    grid = Grid(arch, box_size)
    
    bc = BoundaryConditions(BoundaryCond1d(DirichletBC(0.0), DirichletBC(0.0)))
    sites = siteinds(2, sum(nodes))
    # Dummy field to generate the Laplacian
    dummy_mps = zeros_mps(sites)
    
    field = ScalarField(dummy_mps, arch, bc)
    
    Δ = symmetrized_laplacian(field, grid)
    A = -1.0 * Δ
    
    # Exact Inverse Laplacian MPO
    exact_inv = DifferenTT.exact_inverse_laplacian(sites, arch, bc, Array)
    # The analytical laplacian in Kazeev is for step size h=1 (i.e. matrix index).
    # Discretized Laplacian is A = -D2 / h^2. So A^-1 = -h^2 D2^-1.
    # Actually, Kazeev returns the exact inverse of the tridiagonal matrix directly.
    # The symmetrized_laplacian also constructs the tridiagonal matrix, scaled by 1/h^2.
    # Let's scale exact_inv to match A^{-1}.
    # A = (1/h^2) * T, so A^{-1} = h^2 * T^{-1}.
    # Wait, Kazeev returns the inverse of T, where T is the tridiagonal matrix with 2 on diag and -1 on off-diag.
    # symmetrized_laplacian builds exactly this matrix scaled by 1/h^2.
    h = grid.dx[1]
    exact_inv = h^2 * exact_inv
    
    return A, exact_inv
end

function get_fai(A, μ, gdim, solver)
    D_kwargs_local = (; D_kwargs..., maxdim=gdim)
    # We call the FAI builder
    unscaledG, D = DifferenTT.fai_dg(A, μ, solver, D_kwargs_local; preconditioning=false)
    Gl = apply(D, unscaledG; maxdim=gdim)
    return Gl
end

function run_task(task::TaskSpec, A, exact_inv)
    μ = task.mu
    gdim = task.gdim
    
    @info "Running (μ=$μ, G_maxdim=$gdim) for $(task.nodes)"

    solver = DifferenTT.ResidualALS(; Glinsolve_kwargs_base...)
    
    t_build = @elapsed begin
        Gl = get_fai(A, μ, gdim, solver)
    end
    
    # We want to compare Gl^T Gl against exact_inv.
    # Gl^T Gl
    Gl_T = transpose(Gl)
    replace_siteinds!(Gl_T, siteinds(Gl)) # Wait, transpose might flip the siteinds primes.
    # Actually, FAI builds G_l such that G_l^T G_l approximates A^-1.
    # In DifferenTT, FAI Gl returns an MPO. 
    # MPOs have unprimed indices on one side, primed on the other. 
    # apply(dag(Gl), Gl) would give G_l^T G_l? Let's just compare by computing Frobenius norm of (exact_inv - Gl^T Gl).
    # Since G_l is real, dag(Gl) is transpose.
    # But wait, we can just apply both to a random vector and compare? No, exact norm is better.
    # Frobenius norm of an MPO diff = inner(diff, diff)
    
    # To do Gl^T Gl:
    # First we need to make sure the indices match
    Gl_dag = dag(Gl)
    Gl_T_Gl = apply(Gl_dag, Gl; maxdim=20, cutoff=1e-10)
    
    # ensure primes are matched with exact_inv
    replace_siteinds!(Gl_T_Gl, siteinds(exact_inv))
    
    # Frobenius norm
    diff = +(exact_inv, -1.0 * Gl_T_Gl; cutoff=1e-10, maxdim=20)
    norm_diff = sqrt(abs(inner(diff, diff)))
    norm_exact = sqrt(abs(inner(exact_inv, exact_inv)))
    rel_error = norm_diff / norm_exact
    
    result = Dict(
        "key" => "mu_$(μ)_gdim_$(gdim)",
        "nodes" => task.nodes,
        "mu" => μ,
        "G_maxdim" => gdim,
        "precond_build_time" => t_build,
        "relative_error" => rel_error,
        "norm_exact" => norm_exact,
        "norm_diff" => norm_diff
    )
    
    @info "  build=$(round(t_build; digits=2))s, rel_error=$(round(rel_error; sigdigits=3))"
    return result
end

task_id = parse(Int, ARGS[1])
all_tasks = build_task_list()

if task_id < 1 || task_id > length(all_tasks)
    error("Invalid task_id=$task_id. Must be in 1..$(length(all_tasks))")
end

task = all_tasks[task_id]
@info "Task $task_id / $(length(all_tasks)): nodes=$(task.nodes), μ=$(task.mu), G=$(task.gdim)"

A, exact_inv = setup_problem(task.nodes)
result = run_task(task, A, exact_inv)

output_file = joinpath(tasks_folder, "task_$(lpad(task_id, 3, '0')).jld2")
save(output_file, "result", result)
@info "Task $task_id saved to $output_file"

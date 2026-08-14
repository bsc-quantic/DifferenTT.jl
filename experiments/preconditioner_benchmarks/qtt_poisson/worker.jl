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
system_sizes = [(8,), (20,), (24,)]
mu_values = [1e4, 5e4, 1e5, 5e5, 1e6]

G_maxdims_map = Dict(
    (8,) => [2, 3, 4, 5, 6, 7, 8],
    (16,) => [2, 3, 4, 5, 6, 7, 8],
    (20,) => [2, 3, 4, 5, 6, 7, 8],
    (24,) => [2, 3, 4, 5, 6, 7, 8]
)

cutoff = 1e-13
box_size = (1.0,)

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
    sites = siteinds(arch)
    # Dummy field to generate the Laplacian
    dummy_mps = zeros_mps(sites)
    
    field = ScalarField(dummy_mps, arch, bc)
    
    Δ = symmetrized_laplacian(field, grid; stencil_order=2)
    A = -1.0 * Δ
    
    # Exact Inverse Laplacian MPO
    exact_inv_op = inverse_laplacian_preconditioner(sites, arch, grid, bc).operator
    return A, exact_inv_op
end


function run_task(task::TaskSpec, A, exact_inv)
    μ = task.mu
    gdim = task.gdim
    
    @info "Running (μ=$μ, G_maxdim=$gdim) for $(task.nodes)"

    solver = ResidualALS(
        sweep_kwargs = SweepingArgs(nsweeps=30, maxdim=gdim, cutoff=1e-12),
        residual_tol = 1e-9
    )
    
    t_build = @elapsed begin
        D_kwargs_local = (; cutoff=1e-11, oversample=10, maxdim=gdim)
        precond = fai_preconditioner(A, μ, solver, D_kwargs_local; totalmaxdim=gdim, preconditioning=false)
        Gl = precond.operator
    end
    
    # Gl^T Gl approximates A^{-1} = (-Δ)^{-1}; exact_inv is Δ^{-1}
    target_inv = -1.0 * exact_inv
    Gl_T = dag(transpose(Gl))
    Gl_T_Gl = apply(Gl_T, Gl; alg="naive")
    replace_siteinds!(Gl_T_Gl, only.(siteinds(exact_inv; plev=0)))
    
    # Frobenius norm
    diff = +(target_inv, -1.0 * Gl_T_Gl; cutoff=1e-10, maxdim=20)
    norm_diff = sqrt(abs(inner(diff, diff)))
    norm_exact = sqrt(abs(inner(target_inv, target_inv)))
    rel_error = norm_diff / norm_exact
    
    result = Dict(
        "key" => "mu_$(μ)_gdim_$(gdim)",
        "nodes" => task.nodes,
        "mu" => μ,
        "G_maxdim" => gdim,
        "precond_build_time" => t_build,
        "relative_error" => rel_error,
        "norm_exact" => norm_exact,
        "norm_diff" => norm_diff,
        "compute_cond" => false
    )
    
    @info "  build=$(round(t_build; digits=2))s, rel_error=$(round(rel_error; sigdigits=3))"
    return result
end

task_id = length(ARGS) > 0 ? parse(Int, ARGS[1]) : 1
all_tasks = build_task_list()

if task_id < 1 || task_id > length(all_tasks)
    error("Invalid task_id=$task_id. Must be in 1..$(length(all_tasks))")
end

task = all_tasks[task_id]
@info "Task $task_id / $(length(all_tasks)): nodes=$(task.nodes), μ=$(task.mu), G=$(task.gdim)"

@info "Running JIT warmup..."
dummy_A, dummy_exact = setup_problem((4,))
dummy_solver = ResidualALS(
    sweep_kwargs = SweepingArgs(nsweeps=1, maxdim=2, cutoff=1e-12),
    residual_tol=1e-9
)
dummy_D_kwargs = (; cutoff=1e-11, oversample=2, maxdim=2)
dummy_precond = fai_preconditioner(dummy_A, 1.0, dummy_solver, dummy_D_kwargs; totalmaxdim=2, preconditioning=false)
dummy_Gl = dummy_precond.operator
dummy_Gl_T = dag(transpose(dummy_Gl))
dummy_Gl_T_Gl = apply(dummy_Gl_T, dummy_Gl; alg="naive")
replace_siteinds!(dummy_Gl_T_Gl, only.(siteinds(dummy_exact; plev=0)))
+( -1.0 * dummy_exact, -1.0 * dummy_Gl_T_Gl; cutoff=1e-10, maxdim=2)
@info "JIT warmup complete."

A, exact_inv = setup_problem(task.nodes)
result = run_task(task, A, exact_inv)

output_file = joinpath(tasks_folder, "task_$(lpad(task_id, 3, '0')).jld2")
save(output_file, "result", result)
@info "Task $task_id saved to $output_file"

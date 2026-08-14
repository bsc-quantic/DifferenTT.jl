using ITensors
using ITensorMPS
using LinearAlgebra
using TensorCrossInterpolation
using DifferenTT

function exact_solution(x)
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

function rhs_f(x)
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

function get_1d_A(i::Int, grid, arch)
    bc = BoundaryCond1d(DirichletBC(0.0), DirichletBC(0.0))
    mat = DifferenTT.onedim_differential_operator(2, arch, i, bc, Array)
    mat = -1.0 * mat .* (1.0 / grid.dx[i]^2)
    return mat
end

N = 20
d = 10
println("Setting up $(N)D Poisson problem on TT Architecture with dimension $(d) per site.")

box_size = ntuple(i -> 1.0, N)
nodes = ntuple(i -> d, N)

arch = TTArchitecture(nodes)
grid = Grid(arch, box_size)

# Cross interpolate exact solution
sol_initialpivot = optfirstpivot(Float64, exact_solution, grid, arch)
sol_tci, sol_ranks, sol_error = crossinterpolate2(
    Float64, exact_solution, grid, arch; initialpivots=[sol_initialpivot]
)
@show sol_ranks
@show sol_error
exact_mps = MPS(sol_tci)

# Cross interpolate RHS
println("Cross interpolating RHS...")
rhs_initialpivot = optfirstpivot(Float64, rhs_f, grid, arch)
rhs_tci, rhs_ranks, rhs_error = crossinterpolate2(
    Float64, rhs_f, grid, arch; initialpivots=[rhs_initialpivot]
)
@show rhs_ranks
@show rhs_error
rhs_mps = MPS(rhs_tci)

# Align site indices
replace_siteinds!(exact_mps, siteinds(rhs_mps))
sites = siteinds(rhs_mps)

# Build 1D operators to form the Laplacian
bc = BoundaryConditions(ntuple(i -> BoundaryCond1d(DirichletBC(0.0), DirichletBC(0.0)), N)...)
field = ScalarField(rhs_mps, arch, bc)

println("Building Laplacian MPO...")
Δ = symmetrized_laplacian(field, grid)
A = -1.0 * Δ

println("Setting up solver...")
init_mps = zeros_mps(sites)

sweep_kwargs = SweepingArgs(nsweeps=30, maxdim=20, cutoff=0.0)
updater_kwargs = LocalUpdaterArgs(ishermitian=true, isposdef=true, compute_cond=false, verbosity=2)

solver1 = ResidualALS(
    sweep_kwargs=sweep_kwargs, 
    updater_kwargs=updater_kwargs, 
    residual_tol=1e-4
)

solver2 = ALS(
    sweep_kwargs=sweep_kwargs, 
    updater_kwargs=updater_kwargs, 
)

println("Solving linear system...")
println("norm(rhs_mps) = ", norm(rhs_mps))
println("norm(exact_mps) = ", norm(exact_mps))

println("\n--- Baseline (No preconditioner) ---")
time_baseline = @elapsed out_mps_base = linearsolve(solver1, A, rhs_mps, init_mps)
diff_base = +(out_mps_base, -1.0 * exact_mps; cutoff=1e-10)
rel_error_base = norm(diff_base) / norm(exact_mps)
println("Baseline relative error: ", rel_error_base, " | Time: ", time_baseline)
println("Solution maxlinkdim: ", maxlinkdim(out_mps_base))

println("\n--- Sinc-Quadrature Preconditioner ---")
M = 20
println("Building Sinc-Quadrature PC (M = $M)...")
A_list = [get_1d_A(i, grid, arch) for i in 1:N]
time_sinc_build = @elapsed sinc_pc = sincquadrature_preconditioner(arch, sites, A_list, M; maxdim=20, cutoff=1e-12)

solver_sinc = ResidualALS(
    sweep_kwargs=sweep_kwargs, 
    updater_kwargs=updater_kwargs, 
    residual_tol=1e-4
)
time_sinc_solve = @elapsed out_mps_sinc = linearsolve(solver_sinc, A, rhs_mps, sinc_pc, init_mps)
diff_sinc = +(out_mps_sinc, -1.0 * exact_mps; cutoff=1e-10)
rel_error_sinc = norm(diff_sinc) / norm(exact_mps)
println("Sinc-Quadrature relative error: ", rel_error_sinc, " | Build Time: ", time_sinc_build, " | Solve Time: ", time_sinc_solve)
println("Solution maxlinkdim: ", maxlinkdim(out_mps_sinc))

println("\n--- FAI Preconditioner ---")
μ = 1e5
gdim = 5
println("Building FAI PC (μ = $μ, gdim = $gdim)...")
D_kwargs_local = (; cutoff=1e-11, oversample=10, maxdim=gdim)
fai_builder_solver = ResidualALS(sweep_kwargs=SweepingArgs(nsweeps=30, maxdim=gdim, cutoff=1e-12), residual_tol=1e-9)

time_fai_build = @elapsed begin
    unscaledG, D = DifferenTT.fai_dg(A, μ, fai_builder_solver, D_kwargs_local; preconditioning=false)
    Gl = apply(D, unscaledG; maxdim=gdim)
    fai_pc = FactorizedPreconditioner(Gl)
end

solver_fai = ResidualALS(
    sweep_kwargs=sweep_kwargs, 
    updater_kwargs=updater_kwargs, 
    residual_tol=1e-4
)
time_fai_solve = @elapsed out_mps_fai = linearsolve(solver_fai, A, rhs_mps, fai_pc, init_mps)
diff_fai = +(out_mps_fai, -1.0 * exact_mps; cutoff=1e-10)
rel_error_fai = norm(diff_fai) / norm(exact_mps)
println("FAI relative error: ", rel_error_fai, " | Build Time: ", time_fai_build, " | Solve Time: ", time_fai_solve)
println("Solution maxlinkdim: ", maxlinkdim(out_mps_fai))

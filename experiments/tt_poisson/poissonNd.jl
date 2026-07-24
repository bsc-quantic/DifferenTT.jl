using ITensors
using ITensorMPS
using LinearAlgebra
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

N = 20
d = 10
println("Setting up $(N)D Poisson problem on TT Architecture with dimension $(d) per site.")

box_size = ntuple(i -> 1.0, N)
nodes = ntuple(i -> d, N)

arch = TTArchitecture(nodes)
grid = Grid(arch, box_size)

# Cross interpolate exact solution
sol_initialpivot = DifferenTT.optfirstpivot(Float64, exact_solution, grid, arch)
sol_tci, sol_ranks, sol_error = DifferenTT.crossinterpolate2(
    Float64, exact_solution, grid, arch; initialpivots=[sol_initialpivot]
)
@show sol_ranks
@show sol_error
exact_mps = MPS(sol_tci)

# Cross interpolate RHS
println("Cross interpolating RHS...")
rhs_initialpivot = DifferenTT.optfirstpivot(Float64, rhs_f, grid, arch)
rhs_tci, rhs_ranks, rhs_error = DifferenTT.crossinterpolate2(
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
Δ = DifferenTT.symmetrized_laplacian(field, grid)
A = -1.0 * Δ

println("Setting up solver...")
init_mps = DifferenTT.zeros_mps(sites)

sweep_kwargs = SweepingArgs(nsweeps=30, maxdim=20, cutoff=0.0)
updater_kwargs = LocalUpdaterArgs(compute_cond=false, verbosity=2)

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

out_mps = linearsolve(solver1, A, rhs_mps, nothing)

# Compare against exact solution
diff = +(out_mps, -1.0 * exact_mps; cutoff=1e-10)
rel_error = norm(diff) / norm(exact_mps)

println("Relative error (MPS norm): ", rel_error)
println("Solution maxlinkdim: ", maxlinkdim(out_mps))


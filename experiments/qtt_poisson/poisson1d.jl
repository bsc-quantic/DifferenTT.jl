using ITensors
using ITensorMPS
using DifferenTT

function target(x)
    x = x[1]
    return 2 * 0.3 + 6 * 0.1 * x - 12 * 0.6 * x ^ 2 + 20 * 0.2 * x ^ 3
end

function sol(x)
    x = x[1]
    return (0.3 * (x ^ 2)) + (0.1 * (x ^ 3)) - (0.6 * x ^ 4) + (0.2 * x ^ 5)
end

box_size = (1.0,)
nodes = (4,)

arch = QTTArchitecture(nodes)
grid = Grid(arch, box_size)

target_tci, target_error = crossinterpolate2(Float64, target, grid, arch)
target_mps = MPS(target_tci)
println("Target Error: ", target_error)

sol_initialpivot = optfirstpivot(Float64, sol, grid, arch)
sol_tci, sol_error = crossinterpolate2(Float64, sol, grid, arch; initialpivots=[sol_initialpivot])
sol_mps = MPS(sol_tci)
println("Solution Error: ", sol_error)

bc = BoundaryConditions(BoundaryCond1d(DirichletBC(0.0), DirichletBC(0.0)))
field = ScalarField(target_mps, arch, bc)

Δ = symmetrized_laplacian(field, grid)

solver = ResidualALS(
    sweep_kwargs = SweepingArgs(nsweeps=5, maxdim=50, cutoff=1e-10),
    residual_tol = 1e-5
)

init_mps = zeros_mps(siteinds(target_mps))

output_mps = linearsolve(solver, Δ, target_mps, init_mps)
println("Problem solved successfully!")

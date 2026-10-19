using ITensors
using ITensorMPS
using TensorCrossInterpolation
using Plots
using DifferenTT
using LinearAlgebra

function exact_solution(x::Real, y::Real; k::Real=20.0)
    T = promote_type(typeof(x), typeof(y), typeof(k))
    return sin(k * π * x^2) * sin(k * π * y^2)
end

ε = 1e-0
function rhs_f(x::Real, y::Real; ε::Real=1e-0, k::Real=20.0)
    T = promote_type(typeof(x), typeof(y), typeof(ε), typeof(k))
    cx = cos(k * π * x^2)
    cy = cos(k * π * y^2)
    sx = sin(k * π * x^2)
    sy = sin(k * π * y^2)
    u_xx = 2.0 * k * π * cx * sy - 4.0 * k^2 * π^2 * x^2 * sx * sy
    u_yy = 2.0 * k * π * sx * cy - 4.0 * k^2 * π^2 * y^2 * sx * sy
    return -ε * u_xx - u_yy
end

x_vals = range(0, 1; length=1000)
y_vals = range(0, 1; length=1000)

heatmap(x_vals, y_vals, rhs_f; aspect_ratio=:equal, xlabel="x", ylabel="y", color=:viridis, size=(2000, 1000))

box_size = (1.0, 1.0)
nodes = (6, 6)
arch = QTTArchitecture(nodes)
sites = siteinds(arch)
grid = Grid(arch, box_size)

interpolating_rhs_target(x) = rhs_f(x...; ε=ε)
interpolating_solution_target(x) = exact_solution(x...)

initialpivot = TensorCrossInterpolation.optfirstpivot(Float64, interpolating_rhs_target, grid, arch)
rhs_tci, rhs_ranks, rhs_error = crossinterpolate2(
    Float64, interpolating_rhs_target, grid, arch; initialpivots=[initialpivot]
)
@show rhs_ranks
@show rhs_error
rhs_mps = replace_siteinds!(MPS(rhs_tci), sites)

sol_initialpivot = TensorCrossInterpolation.optfirstpivot(Float64, interpolating_solution_target, grid, arch)
sol_tci, sol_ranks, sol_error = crossinterpolate2(
    Float64, interpolating_solution_target, grid, arch; initialpivots=[sol_initialpivot]
)
@show sol_ranks
@show sol_error
exact_mps = replace_siteinds!(MPS(sol_tci), sites)

fig_size = (1200, 1200)
visualize_scalar(
    rhs_mps, grid, arch, "experiments/qtt_poisson/rhs_2d.png"; npoints=200, aspect_ratio=:equal, size=fig_size
)
visualize_scalar(
    exact_mps, grid, arch, "experiments/qtt_poisson/exact_2d.png"; npoints=200, aspect_ratio=:equal, size=fig_size
)

# Build 2D Laplacian with homogeneous Dirichlet BCs on all sides.
bc_x = BoundaryCond1d(DirichletBC(0.0), DirichletBC(0.0))
bc_y = BoundaryCond1d(DirichletBC(0.0), DirichletBC(0.0))

bc = BoundaryConditions(bc_x, bc_y)

field = ScalarField(sites, arch, bc)
Δ = symmetrized_laplacian(field, grid; asymmetry=ε, stencil_order=8)

visualize_scalar(
    -1.0 * apply(Δ, exact_mps),
    grid,
    arch,
    "experiments/qtt_poisson/opexact_2d.png";
    npoints=200,
    aspect_ratio=:equal,
    size=fig_size,
)

# Dirichlet correction on the pressure field moved to the rhs of the linear system
rhs_correction_mpo = scalar_dirichlet_correction(field, grid, 2)
rhs_correction =
    (isnothing(rhs_correction_mpo)) ? nothing : apply(rhs_correction_mpo, ones_mps(sites, 1, Array); alg="naive")

# Apply the boundary nullifier
rhs_mps = isnothing(rhs_correction) ? rhs_mps : +(rhs_mps, rhs_correction)
rhs_mps = symmetrize_rhs(field, rhs_mps)

updater_kwargs = LocalUpdaterArgs(; ishermitian=true, isposdef=true, rtol=1e-5, krylovdim=1, maxiter=1000, verbosity=2)
sweep_kwargs = SweepingArgs(; nsweeps=100, maxdim=150, cutoff=1e-11)

residual_solver = ResidualALS(; sweep_kwargs=sweep_kwargs, updater_kwargs=updater_kwargs)

global_updater_kwargs = GlobalUpdaterArgs(;
    ishermitian=true,
    isposdef=true,
    krylovdim=1,
    maxiter=100,
    rtol=1e-8,
    verbosity=0,
    maxdim=typemax(Int),
    cutoff=1e-8,
    orth_maxdim=typemax(Int),
    orth_cutoff=1e-12,
)

ttcg_solver = GlobalCG(; updater_kwargs=global_updater_kwargs)

output_mps = linearsolve(ttcg_solver, -1.0 * Δ, rhs_mps, nothing)
visualize_scalar(
    output_mps, grid, arch, "experiments/qtt_poisson/numerical_2d.png"; npoints=200, aspect_ratio=:equal, size=fig_size
)

# Compare against exact solution directly in MPS space.
rel_mps_error = norm(output_mps - exact_mps) / norm(exact_mps)
println("Relative error (MPS norm): ", rel_mps_error)

# Also compare on a regular sampled grid for a pointwise-style metric.
ncompare = (65, 65)
_, numerical_data = samples(output_mps, grid, arch, ncompare)
_, exact_data = samples(exact_mps, grid, arch, ncompare)
abs_error = abs.(numerical_data .- exact_data)
rel_l2_sample_error = norm(abs_error) / norm(exact_data)
max_abs_sample_error = maximum(abs_error)

println("Sampled relative L2 error: ", rel_l2_sample_error)
println("Sampled max absolute error: ", max_abs_sample_error)

# Residual check: ||Δu_h - f|| / ||f||
residual = apply(Δ, output_mps) - rhs_mps
rel_residual = norm(residual) / norm(rhs_mps)
println("Relative residual norm: ", rel_residual)

using ITensors
using ITensorMPS
using TensorCrossInterpolation
using Adapt
using LinearAlgebra
using MKL
using TimerOutputs
using JLD2
using Revise
using DifferenTT

BLAS.set_num_threads(8)

destination_folder = "experiments/inse/tunnel2d/circle_data"
destination_file = "circle_data.jld2"

# logger = init_logger(destination_folder)
logger = nothing
# --- Different masks ---

function circle(x, y; bx=0.5, by=0.5, rad=0.05)
    value = (x - bx) ^ 2 + (y - by) ^ 2 - rad ^ 2
    value += abs(value)
    # return 1 - exp(-700 * value)
    return 0.5 * (1 + tanh(500 * value))
end


function squares(x, y; rad=0.1, xdist=0.5, ydist=0.3, bx=0.5, by=0.5)
    square1 = abs((x - bx - xdist/2) + (y - by)) + abs((x - bx - xdist/2) - (y - by)) - rad
    square2 = abs((x - bx + xdist/2) + (y - by)) + abs((x - bx + xdist/2) - (y - by)) - rad
    square3 = abs((x - bx) + (y - by - ydist/2)) + abs((x - bx) - (y - by - ydist/2)) - rad
    square4 = abs((x - bx) + (y - by + ydist/2)) + abs((x - bx) - (y - by + ydist/2)) - rad

    # square1 = abs(x + y - 1.0 + dist/2) + abs(x - y) - rad

    # square2 = abs(x + y - 0.5) + abs(y - x - 0.5) - rad
    # square3 = abs(x + y + 0.5) + abs(y - x + 0.5) - rad
    # square4 = abs(x + y - 0.5) + abs(y - x + 0.5) - rad
    value = square1 * square2 * square3 * square4
    value += abs(value)
    return 1 - exp(-3000 * value)
end

function square(x, y; rad=0.1)
    value = abs(x + y - 1.0) + abs(x - y) - rad
    value += abs(value)
    return 1 - exp(-100 * value)
end

function symmetric_wing(x, y; bx=0.6, by=0.5, rad=0.25, k=0.2)
    chord = 0.5 # Or another value for the length
    t = rad # thickness
    x0 = bx - chord/2

    csq = 0.2969 * t
    c1 = -0.1260 * t
    c2 = -0.3516 * t
    c3 = 0.2843 * t
    c4 = -0.1015 * t
    
    # Normalize position along the chord to [0,1]
    arg = abs(x - bx) <= chord/2 ? (x - x0) / chord : 0.0
    
    # Airfoil half-thickness (NACA 00xx)
    v = chord * 5 * (c1 * arg + c2 * arg ^ 2 + c3 * arg ^ 3 + c4 * arg ^ 4 + csq * arg ^ 0.5)
    
    # Elliptical-like boundary centered at `by`, with inclination `k`
    value = v ^ 2 - (y - by - k * (arg - 0.5)) ^ 2
    value += abs(value)
    return exp(-1000 * value)
end

using Plots
x_vals = range(0, 2, length=1000)
y_vals = range(0, 1, length=1000)

# Evaluate and plot the mask using a heatmap
heatmap(
    x_vals, y_vals, circle,
    aspect_ratio = :equal,
    xlabel = "x",
    ylabel = "y",
    color = :viridis, # or :grays for a black-and-white mask look
    size = (2000, 1000)
)

mask(x) = circle(x...)

# longevol_data = load("experiments/inse/tunnel2d/restricted_bm_evol_data/chorin.jld2")
# kickstart_mps_vx, kickstart_mps_vy = last(longevol_data["u_fields"])

box_size = (2.0, 1.0)
fig_size = (1200, 600)
clims=(-20, 20)
nodes = (10, 9)

arch = QTTArchitecture(nodes)
grid = Grid(arch, box_size)
sites = siteinds(arch)
# sites = siteinds(kickstart_mps_vx)


save_mask = true
mask_name = "circle_mask"
load_mask = false
mask_path = "experiments/inse/tunnel2d/"*mask_name*".jld2"

object_mask = nothing

if save_mask
    initialpivot = TensorCrossInterpolation.optfirstpivot(Float64, mask, grid, arch)

    tci_mask, ranks_mask, error_mask = crossinterpolate2(Float64, mask, grid, arch, maxdim=30)
    # tci_mask, ranks_mask, error_mask = crossinterpolate2(Float64, mask, grid, arch; tolerance=1e-8, maxdim=30, maxiter=200, initialpivots=[initialpivot])

    @show ranks_mask
    @show error_mask
    object_mask = replace_siteinds!(MPS(tci_mask), sites)
    visualize_scalar(object_mask, grid, arch, "experiments/inse/tunnel2d/"*mask_name*".png"; npoints =  200, size = fig_size)

    # save(mask_path, mask_name, object_mask)
end

if load_mask
    object_mask = load(mask_path)[mask_name]
    length(object_mask) != sum(nodes) && throw(ArgumentError("Given object_mask does not have the required length."))
    siteinds(object_mask) != sites && replace_siteinds!(object_mask, sites)

    visualize_scalar(object_mask, grid, arch, "experiments/inse/tunnel2d/"*mask_name*".png"; npoints =  200, size = fig_size)

end

isnothing(object_mask) && @warn "Mask was not initialized."

# time_evol_settings = TimeEvol(0.0, 2.0, 401)
# time_evol_settings = TimeEvol(0.0, 1.0, 4001)
time_evol_settings = TimeEvol(0.0, 0.001, 1)

Re = 3000

# ALWAYS SET V0 = 1, IF MORE TURBULENCE IS DESIRED, INCREASE RE
v0 = 1.0
mps_vx = tt_mul(object_mask, v0 * ones_mps(sites, 1))
mps_vy = 0.0 * ones_mps(sites, 1)

# bcsvxx = BoundaryCond1d(NeumannBC(0.0), DirichletBC(-1.0*v0))
# bcsvxy = BoundaryCond1d(PeriodicBC(), PeriodicBC())

# bcsvyx = BoundaryCond1d(OpenBC(), OpenBC())
# bcsvyy = BoundaryCond1d(PeriodicBC(), PeriodicBC())

# bcspx = BoundaryCond1d(DirichletBC(1.0), NeumannBC(0.0))
# bcspy = BoundaryCond1d(PeriodicBC(), PeriodicBC())

bcsvxx = BoundaryCond1d(DirichletBC(v0), NeumannBC(0.0))
bcsvxy = BoundaryCond1d(PeriodicBC(), PeriodicBC())

bcsvyx = BoundaryCond1d(OpenBC(), OpenBC())
bcsvyy = BoundaryCond1d(PeriodicBC(), PeriodicBC())

bcspx = BoundaryCond1d(NeumannBC(0.0), DirichletBC(1.0))
bcspy = BoundaryCond1d(PeriodicBC(), PeriodicBC())
# 
# bcsvxx = BoundaryCond1d(DirichletBC(v0), DirichletBC(v0))
# bcsvxy = BoundaryCond1d(DirichletBC(v0), DirichletBC(v0))

# bcsvyx = BoundaryCond1d(DirichletBC(0.0), DirichletBC(0.0))
# bcsvyy = BoundaryCond1d(NeumannBC(0.0), NeumannBC(0.0))

# bcspx = BoundaryCond1d(NeumannBC(0.0), DirichletBC(1.0))
# bcspy = BoundaryCond1d(DirichletBC(0.0), DirichletBC(0.0))

bcsvx = BoundaryConditions(bcsvxx, bcsvxy)
bcsvy = BoundaryConditions(bcsvyx, bcsvyy)
bcsp = BoundaryConditions(bcspx, bcspy)

v_field = VectorField((mps_vx, mps_vy), arch, (bcsvx, bcsvy))
# v_field = VectorField((kickstart_mps_vx, kickstart_mps_vy), arch, (bcsvx, bcsvy))


p_field = ScalarField(sites, arch, bcsp)

# visualize_vector(v_field, grid, "experiments/inse/tunnel2d/tunnel_field.png"; npoints=512, quiver_npoints=5, arrow_scale=0.1, aspect_ratio=:equal, size = fig_size)
# visualize_vorticity(v_field, grid, "experiments/inse/tunnel2d/tunnel_vorticity.png"; npoints=512, quiver_npoints=20, arrow_scale=0.1, aspect_ratio=:equal, size = fig_size)

updater_kwargs = LocalUpdaterArgs(;
    ishermitian=true,
    isposdef=true,
    compute_cond=true,
    rtol=1e-4,
    krylovdim=1,
    maxiter=500,
    verbosity=1
)
sweep_kwargs = SweepingArgs(;
    nsweeps=5,
    maxdim=150,
    cutoff=1e-11
)
solver = ALS(; sweep_kwargs=sweep_kwargs, updater_kwargs=updater_kwargs)

experiment = INSExperiment(grid, time_evol_settings, v_field, p_field, Re, object_mask;stencil_order=8, maxdim=150, cutoff=1e-11)

log_data(logger, destination_file, "experiment", experiment)

TimerOutputs.enable_debug_timings(DifferenTT)
reset_timer!(timeit_to)

save_step = 1
plot_step = 1 #How many save steps to skip from plot to plot
stepper = RK4(cutoff=experiment.cutoff, maxdim=experiment.maxdim)
chorin_evolve!(experiment, solver, stepper, save_step, logger, destination_file)
display(timeit_to)
TimerOutputs.disable_debug_timings(DifferenTT)

nplots = length(experiment.u_fields)
for k in 1:plot_step:min(nplots * plot_step, length(experiment.u_fields))
    evolution_step = 1 + (save_step * plot_step) * (k - 1)
    visualize_vector(experiment.u_fields, k, grid, destination_folder*"/vector_$(evolution_step).png"; npoints=300, quiver_npoints=20, arrow_scale=0.05, aspect_ratio=:equal, size = fig_size)
    visualize_vorticity(experiment.u_fields, k, grid, destination_folder*"/vorticity_$(evolution_step).png"; npoints=300, quiver_npoints=20, arrow_scale=0.0, aspect_ratio=:equal, size = fig_size)
    k > 1 && visualize_scalar(experiment.p_field, k-1, grid, destination_folder*"/press_$(evolution_step - 1).png"; npoints=300, aspect_ratio=:equal, size = fig_size)
end

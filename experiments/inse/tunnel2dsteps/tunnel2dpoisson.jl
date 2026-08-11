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

############################################################

# 1. Define a destination folder and file to store the experiment information and the subsequent fields

############################################################

destination_folder = "experiments/inse/tunnel2dsteps/"
destination_file = "chorin.jld2"

############################################################

# 2. Load a snapshot from another run or start from scratch
# !! Make sure these fields match the Architecture definition below !!

############################################################

load_timevolved_mps = false

if load_timevolved_mps
    longevol_data = load("experiments/inse/tunnel2dsteps/chorin.jld2")
    kickstart_mps_vx, kickstart_mps_vy = last(longevol_data["u_fields"])
end

############################################################

# 3. Define the box size, nodes count and the sites

############################################################
box_size = (2.0, 1.0)
fig_size = (1200, 600)
clims = (-20, 20)
nodes = (13, 12)
sites = load_timevolved_mps ? siteinds(kickstart_mps_vx) : siteinds(2, sum(abs, nodes))

############################################################

# 4. Define the Grid and Architecture objects 

############################################################

arch = QTTArchitecture(nodes)
grid = Grid(arch, box_size)

############################################################

# 5. Create the mask

object_mask = nothing # Id the following code blocks in this section are not run, simulation will proceed without a mask

# Set the following parameters:
# - Whether to compute a new mask
# - What name this mask will take (also used to retrieve it if a mask is loaded)
# - Whether to load a mask from the specified file

save_mask = true
mask_name = "circle_mask"
load_mask = false
mask_path = "experiments/inse/tunnel2dsteps/" * mask_name * ".jld2"

object_mask = nothing

# More object masks can be found in: "experiments/inse/tunnel2d/tunnel2d.jl

function circle(x, y; bx=0.5, by=0.5, rad=0.05)
    value = (x - bx)^2 + (y - by)^2 - rad^2
    value += abs(value)
    # return 1 - exp(-700 * value)
    return 0.5 * (1 + tanh(500 * value))
end

mask(x) = circle(x...)

if save_mask
    initialpivot = TensorCrossInterpolation.optfirstpivot(Float64, mask, grid, arch)

    # tci_mask, ranks_mask, error_mask = crossinterpolate2(Float64, mask, grid, arch; maxdim=10, maxiter=50, initialpivots=[initialpivot])
    tci_mask, ranks_mask, error_mask = crossinterpolate2(Float64, mask, grid, arch; tolerance=1e-14, maxdim=40)#, strictlynested=true)

    @show ranks_mask
    @show error_mask
    object_mask = replace_siteinds!(MPS(tci_mask), sites)

    visualize_scalar(object_mask, grid, arch, destination_folder * "/mask_buffer.png"; npoints=200, aspect_ratio=:equal, size=fig_size)
    # save(mask_path, mask_name, object_mask)
end

if load_mask
    object_mask = load(mask_path)[mask_name]

    length(object_mask) != sum(nodes) && throw(ArgumentError("Given object_mask does not have the required length."))
    siteinds(object_mask) != sites && replace_siteinds!(object_mask, sites)
end

isnothing(object_mask) && @warn "Mask was not initialized."

############################################################

# 5. Initialize the fields

############################################################

v0 = 1.0
mps_vx = isnothing(object_mask) ? ones_mps(sites, 1) : tt_mul(object_mask, ones_mps(sites, 1))
mps_vy = 0.0 * ones_mps(sites, 1)

bcsvxx = BoundaryCond1d(DirichletBC(v0), NeumannBC(0.0))
bcsvxy = BoundaryCond1d(PeriodicBC(), PeriodicBC())

bcsvyx = BoundaryCond1d(OpenBC(), OpenBC())
bcsvyy = BoundaryCond1d(PeriodicBC(), PeriodicBC())

bcspx = BoundaryCond1d(NeumannBC(0.0), DirichletBC(1.0))
# bcspx = BoundaryCond1d(OpenBC(), OpenBC())
bcspy = BoundaryCond1d(PeriodicBC(), PeriodicBC())

# bcspy = BoundaryCond1d(OpenBC(), OpenBC())
# bcspy = BoundaryCond1d(OpenBC(), OpenBC())

bcsvx = BoundaryConditions(bcsvxx, bcsvxy)
bcsvy = BoundaryConditions(bcsvyx, bcsvyy)
bcsp = BoundaryConditions(bcspx, bcspy)

v_field = load_timevolved_mps ? VectorField((kickstart_mps_vx, kickstart_mps_vy), arch, (bcsvx, bcsvy)) : VectorField((mps_vx, mps_vy), arch, (bcsvx, bcsvy))
p_field = ScalarField(sites, arch, bcsp)

############################################################

# 6. Set up the time evolution settings, the Reynolds number and the experiment object  

############################################################
time_evol_settings = TimeEvol(0.0, 0.00005, 4, :euler)
Re = 3000
experiment = INSExperiment(grid, time_evol_settings, v_field, p_field, Re, object_mask; stencil_order=8, maxdim=150, cutoff=1e-11)

ITensors.disable_warn_order()
ENV["JULIA_DEBUG"] = ""

#################################################

# If this is uncommented, many dense matrices will start to be built to debug operator construction. 
# ONLY ACTIVATE WITH SMALL NSITE < 12 OR SO (otherwise the simulation will run out of memory and crash)

# ENV["JULIA_DEBUG"]="DifferenTT"

#################################################

############################################################

# 7. Build all the operators  

############################################################
p_diff_ops = DifferentialOpsScalar(experiment.grid, experiment.p_field; stencil_order=experiment.stencil_order)
u_diff_ops = DifferentialOps(experiment.grid, experiment.u_fields; stencil_order=experiment.stencil_order)
u_boundary_null = BoundaryNull(experiment.u_fields; stencil_order=experiment.stencil_order)
symmetrized_Δ_cache = DifferenTT.symmetrized_laplacian(experiment.p_field, experiment.grid; stencil_order=experiment.stencil_order, maxdim=experiment.maxdim, cutoff=experiment.cutoff)

############################################################

# 8. Build the preconditioner  

############################################################
prec_updater_kwargs = LocalUpdaterArgs(;
    ishermitian=true,
    isposdef=true,
    compute_cond=false,
    rtol=1e-8,
    krylovdim=1,
    maxiter=1000,
    verbosity=1
)
prec_sweep_kwargs = SweepingArgs(;
    nsweeps=20,
    maxdim=8,
    cutoff=1e-11,
    nsite=2,
    outputlevel=0,
)
prec_solver = ALS(; sweep_kwargs=prec_sweep_kwargs, updater_kwargs=prec_updater_kwargs)

g_updater_kwargs = LocalUpdaterArgs(;
    ishermitian=true,
    isposdef=true,
    compute_cond=false,
    rtol=1e-9,
    krylovdim=1,
    maxiter=2000,
    verbosity=1
)
g_sweep_kwargs = SweepingArgs(;
    nsweeps=20,
    maxdim=5,
    cutoff=1e-12,
    nsite=2,
    outputlevel=1
)
g_solver = ALS(; sweep_kwargs=g_sweep_kwargs, updater_kwargs=g_updater_kwargs)

D_kwargs = (;
    maxdim=5,
    cutoff=1e-11,
    oversample=10,
)

input = -1.0 * truncate(symmetrized_Δ_cache; maxdim=3)
# Here the 3 different available constructors are set up, and tests on their eigenvalues are 
linsolve_precond = linsolve_preconditioner(-1.0 * symmetrized_Δ_cache, prec_solver)
newtoniter_precond = newtoniter_preconditioner(-1.0 * symmetrized_Δ_cache, 10; maxdim=6)
rank1_precond = rank1_preconditioner(-1.0 * symmetrized_Δ_cache)
fai_precond = fai_preconditioner(-1.0 * symmetrized_Δ_cache, 1.0e7, g_solver, D_kwargs; totalmaxdim=5)
randmpo = random_mpo(sites) + random_mpo(sites)
spdrandmpo = apply(transpose(randmpo), randmpo; alg="naive")
random_precond = DifferenTT.Preconditioner(spdrandmpo)
init = random_mps(sites)

dmrg(-1.0 * symmetrized_Δ_cache, init; nsweeps=5)

dmrg(linsolve_precond.operator, init; nsweeps=5)
dmrg(newtoniter_precond.operator, init; nsweeps=5)
dmrg(rank1_precond.operator, init; nsweeps=5)
dmrg(apply(transpose(fai_precond.operator), fai_precond.operator), init; nsweeps=5)

dmrg(apply(linsolve_precond.operator, -1.0 * symmetrized_Δ_cache), init; nsweeps=5)
dmrg(apply(rank1_precond.operator, -1.0 * symmetrized_Δ_cache), init; nsweeps=5)
dmrg(apply(transpose(fai_precond.operator), apply(fai_precond.operator, -1.0 * symmetrized_Δ_cache)), init; nsweeps=5)

vals_list = Dict()

for i in 3:15
    observer = LinsolveObserver()
    loop_prec_updater_kwargs = LocalUpdaterArgs(;
        ishermitian=true,
        isposdef=true,
        compute_cond=false,
        rtol=1e-3,
        krylovdim=1,
        maxiter=200,
        verbosity=1
    )
    loop_prec_sweep_kwargs = SweepingArgs(;
        nsweeps=100,
        maxdim=i,
        cutoff=1e-11,
        nsite=2,
        outputlevel=0,
        observer!=observer
    )
    loop_prec_solver = ALS(; sweep_kwargs=loop_prec_sweep_kwargs, updater_kwargs=loop_prec_updater_kwargs)

    input = -1.0 * truncate(symmetrized_Δ_cache; maxdim=3)
    # Here the 3 different available constructors are set up, and tests on their eigenvalues are 
    linsolve_precond = linsolve_preconditioner(-1.0 * symmetrized_Δ_cache, loop_prec_solver; preconditioning=true)

    op = DifferenTT.contract_mpo(linsolve_precond.operator)
    vals = eigvals(op)
    vals_list[i] = vals
end


#---------------------------------------------------

# 9. Poisson solver 

#---------------------------------------------------

# Struct initialization to track how many iterations does the cg take ate each site, check the struct docstirng for more info.
stats = KrylovSweepStats()

local_updater_kwargs = LocalUpdaterArgs(;
    ishermitian=false,
    isposdef=false,
    compute_cond=false,
    rtol=1e-4,
    krylovdim=1,
    maxiter=500,
    verbosity=1,
    numiter_store=stats
)
sweep_kwargs = SweepingArgs(;
    nsweeps=5,
    maxdim=150,
    cutoff=1e-11
)
als_solver = ALS(; sweep_kwargs=sweep_kwargs, updater_kwargs=local_updater_kwargs)

global_updater_kwargs = GlobalUpdaterArgs(;
    ishermitian = true,
    isposdef = true,
    krylovdim = 1,
    maxiter = 50,
    rtol = 1e-6,
    verbosity = 1,
    maxdim = 70,
    cutoff = 1e-8,
    orth_maxdim = 150,
    orth_cutoff = 1e-12,
)

ttcg_solver = GlobalCG(; updater_kwargs = global_updater_kwargs)

dt = experiment.time_evol_settings.dt
cutoff = experiment.cutoff
maxdim = experiment.maxdim

masked_u = DifferenTT.mask_correction(last(experiment.u_fields), object_mask; cutoff=cutoff, maxdim=maxdim)

nondivfree_advection_terms = advection(masked_u, u_diff_ops; cutoff=cutoff, maxdim=maxdim)
visualize_vector(-1.0 .* nondivfree_advection_terms, experiment.grid, experiment.u_fields.arch, "experiments/inse/tunnel2dsteps/test_nondivfree_advection.png"; arrow_scale=0.0, npoints=100, size=(1200, 600), clims=(-20, 20))

# Obtain the divergence of the field
∇u = DifferenTT.divergence(masked_u, u_diff_ops; cutoff=cutoff, maxdim=maxdim)
visualize_scalar(∇u, experiment.grid, experiment.u_fields.arch, "experiments/inse/tunnel2dsteps/test_∇u.png"; arrow_scale=0.0, npoints=200, size=(1200, 600), clims=(-20, 20))

# Boundary nullifiers for the divergence
∇u_correction = DifferenTT.divergence_dirichlet_correction(experiment.u_fields; stencil_order=experiment.stencil_order, cutoff=cutoff, maxdim=maxdim)

# Dirichlet correction on the pressure field moved to the rhs of the linear system
Δp_correction_mpo = DifferenTT.scalar_dirichlet_correction(experiment.p_field, experiment.grid, 2; stencil_order=experiment.stencil_order, cutoff=cutoff, maxdim=maxdim)
Δp_correction = (isnothing(Δp_correction_mpo)) ? nothing : apply(Δp_correction_mpo, ones_mps(experiment.p_field.usites, 1, backend_type(Δp_correction_mpo)); alg="naive", cutoff=cutoff, maxdim=maxdim)

# Apply the boundary nullifier
∇u = isnothing(∇u_correction) ? ∇u : apply(∇u_correction, ∇u; cutoff=cutoff, maxdim=maxdim)

# Add the pressure dirichlet contribution coming from the lhs, scaled with dt
∇u = isnothing(Δp_correction) ? ∇u : +(∇u, (-1.0 * dt) * Δp_correction; cutoff=cutoff, maxdim=maxdim)
visualize_scalar(∇u, experiment.grid, experiment.u_fields.arch, "experiments/inse/tunnel2dsteps/test_∇u_corrected.png"; arrow_scale=0.0, npoints=200, size=(1200, 600), clims=(-20, 20))

previous_p = nothing
epsilon = 1e-10

# Correct the rhs of the equation taking into account that the rows affected by Neumann boundary condition get scaled by 0.5 at both sides.
# The correction already has the - sign such that only need to sum it, but since we are solving -∇²p = -dt⁻¹∇⋅u we compute it with -1.0 * ∇u
rhs_neumann_symmetrization = DifferenTT.neumann_symmetrization_correction_mps(experiment.p_field.arch, experiment.p_field.bcu, -1.0 * ∇u; cutoff=cutoff, maxdim=maxdim)
rhs = isnothing(rhs_neumann_symmetrization) ? -1.0 * ∇u : +(-1.0 * ∇u, rhs_neumann_symmetrization; cutoff=cutoff, maxdim=maxdim)
visualize_scalar(rhs, experiment.grid, experiment.u_fields.arch, "experiments/inse/tunnel2dsteps/test_rhs.png"; arrow_scale=0.0, npoints=200, size=(1200, 600), clims=(-20, 20))

# ---------------------------

# Choose between the preconditioners built before or nothing
precond = nothing
@time begin
    pdt = isnothing(precond) ? DifferenTT.linearsolve(als_solver, -1.0 * symmetrized_Δ_cache, rhs, previous_p) : DifferenTT.linearsolve(als_solver, -1.0 * symmetrized_Δ_cache, rhs, precond, previous_p)
end

divfree_p = pdt * (1 / dt)

visualize_scalar(divfree_p, experiment.grid, experiment.u_fields.arch, "experiments/inse/tunnel2dsteps/test_divfreep.png"; arrow_scale=0.0, npoints=200, size=(1200, 600))#, clims=(-0.1, 0.1))
push!(experiment.p_field, divfree_p)

# l2 error of the linear solver = ||Ax - b||² = <psi|A†A|psi> - 2Re<psi|A†|b> + <b|b>
term1 = inner(symmetrized_Δ_cache, pdt, symmetrized_Δ_cache, pdt)
term2 = -2 * real(inner(rhs', symmetrized_Δ_cache, pdt)) # - sign comes from A = -1.0 * symmetrized_Δ_cache
term3 = inner(rhs, rhs)

res_l2norm = sqrt(max(0.0, term1 - term2 + term3))
@show res_l2norm / norm(rhs) # 0.03, but has gone up in some tests upt to 0.76 with the same resources as the coarse case...

#show the iteration stats from the solve
@show DifferenTT.positions_sum(stats)

#---------------------------------------------------

# 10. Project onto the divergence free manifold

#---------------------------------------------------

div_free_u_mpss = ntuple(i -> begin
        dp = apply(p_diff_ops.d_ops[i], divfree_p; cutoff=cutoff, maxdim=maxdim)
        # We correct dp/dxi per dimension, not mixing in corrections from other dimensions.
        dirichlet_correction_mpo = DifferenTT.scalar_dirichlet_correction_dim(experiment.p_field, experiment.grid, 1, i; stencil_order=experiment.stencil_order)#, cutoff=cutoff, maxdim=maxdim)

        # We assume that the pressure field is contstant at the Dirichlet boundaries -> Identity MPS (one could encode an arbitrary field)
        dirichlet_correction = (isnothing(dirichlet_correction_mpo)) ? nothing : apply(dirichlet_correction_mpo, DifferenTT.ones_mps(experiment.p_field.usites, 1, backend_type(dirichlet_correction_mpo)); cutoff=cutoff, maxdim=maxdim)

        full_dp = isnothing(dirichlet_correction) ? dp : +(dp, dirichlet_correction; cutoff=cutoff, maxdim=maxdim) # Correct according to -dt * ∇p

        @debug begin
            println("left ∇p correction for component $i")
            display(round.(contract_mps(full_dp))[1:8])
            println("right ∇p correction for component $i")
            display(round.(contract_mps(full_dp))[end-7:end])
        end

        output = +(masked_u[i], (-1.0 * dt) * full_dp; cutoff=cutoff, maxdim=maxdim)
        visualize_scalar((-1.0 * dt) * full_dp, experiment.grid, experiment.u_fields.arch, "experiments/inse/tunnel2dsteps/test_dp$(i).png"; npoints=200, size=(1200, 600))
        return output
    end, 2)

visualize_vector(div_free_u_mpss, experiment.grid, experiment.u_fields.arch, "experiments/inse/tunnel2dsteps/test_div_free_u_mpss.png"; arrow_scale=0.0, npoints=300, size=(1200, 600))

#---------------------------------------------------

# 11. Evaluate ∇p behaviour with the diffusin term, which is linear   

#---------------------------------------------------
∇p = ntuple(i -> begin
        dp = apply(p_diff_ops.d_ops[i], divfree_p)#; cutoff=cutoff, maxdim=maxdim)
        # We correct dp/dxi per dimension, not mixing in corrections from other dimensions.
        dirichlet_correction_mpo = DifferenTT.scalar_dirichlet_correction_dim(experiment.p_field, experiment.grid, 1, i; stencil_order=experiment.stencil_order, cutoff=cutoff, maxdim=maxdim)

        # We assume that the pressure field is contstant at the Dirichlet boundaries -> Identity MPS (one could encode an arbitrary field)
        dirichlet_correction = (isnothing(dirichlet_correction_mpo)) ? nothing : apply(dirichlet_correction_mpo, DifferenTT.ones_mps(experiment.p_field.usites, 1, backend_type(dirichlet_correction_mpo)); cutoff=cutoff, maxdim=maxdim)

        full_dp = isnothing(dirichlet_correction) ? dp : +(dp, dirichlet_correction)#; cutoff=cutoff, maxdim=maxdim) # Correct according to -dt * ∇p
        return full_dp
    end, 2)

∇p_diffusion = diffusion(-1.0 * dt .* ∇p, u_diff_ops; cutoff=cutoff, maxdim=maxdim)
visualize_vector((1 / Re) .* ∇p_diffusion, experiment.grid, experiment.u_fields.arch, "experiments/inse/tunnel2dsteps/test_∇p_diffusion.png"; arrow_scale=0.0, npoints=200, size=(1200, 600))
# BIG BIG ISSUE Here

#---------------------------------------------------

# 12. Timestep  

#---------------------------------------------------

divfree_advection_terms = advection(div_free_u_mpss, u_diff_ops; cutoff=cutoff, maxdim=maxdim)
visualize_vector(-1.0 .* divfree_advection_terms, experiment.grid, experiment.u_fields.arch, "experiments/inse/tunnel2dsteps/test_divfree_advection.png"; arrow_scale=0.0, npoints=200, size=(1200, 600), clims=(-5, 5))
divfree_diffusion_terms = diffusion(div_free_u_mpss, u_diff_ops; cutoff=cutoff, maxdim=maxdim)
visualize_vector((1 / Re) .* divfree_diffusion_terms, experiment.grid, experiment.u_fields.arch, "experiments/inse/tunnel2dsteps/test_divfree_diffusion.png"; arrow_scale=0.0, npoints=200, size=(1200, 600), clims=(-5, 5))

NSrhs = DifferenTT.NavierStokesPredictorRHS(experiment, u_diff_ops, u_boundary_null)
stepper = ExplicitEuler(cutoff=experiment.cutoff, maxdim=experiment.maxdim)

timestepped_u_mpss = timestep(stepper, div_free_u_mpss, experiment.time_evol_settings.dt, NSrhs, 1.0 * experiment.time_evol_settings.dt)


#---------------------------------------------------

# 13. Visualize and store the fields  

#---------------------------------------------------

visualize_vector(timestepped_u_mpss, grid, arch, destination_folder * "/vector_buffer.png"; npoints=200, quiver_npoints=20, arrow_scale=0.1, aspect_ratio=:equal, size=fig_size)
visualize_vorticity(timestepped_u_mpss, grid, arch, destination_folder * "/vorticity_buffer.png"; npoints=200, quiver_npoints=20, arrow_scale=0.0, aspect_ratio=:equal, size=fig_size)
visualize_scalar(divfree_p, grid, arch, destination_folder * "/press_buffer.png"; npoints=200, aspect_ratio=:equal, size=fig_size)


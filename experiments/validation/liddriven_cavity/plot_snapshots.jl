using ITensors
using ITensorMPS
using DifferenTT
using JLD2
using Plots

# Usage: [CAVITY_LIDS=1|2] julia --project=. experiments/validation/liddriven_cavity/plot_snapshots.jl [nocond]
# Number of moving lids of the worker run (see worker.jl): 2 reads the doubly-driven cavity results
const CAVITY_LIDS = parse(Int, get(ENV, "CAVITY_LIDS", "1"))
CAVITY_LIDS in (1, 2) || error("CAVITY_LIDS must be 1 or 2, got $CAVITY_LIDS")
const RESULTS_SUFFIX = CAVITY_LIDS == 2 ? "_doublelid" : ""
const use_nocond = length(ARGS) > 0 && ARGS[1] == "nocond"
base_folder =
    (use_nocond ? "experiments/validation/liddriven_cavity/results_nocond" : "experiments/validation/liddriven_cavity/results") *
    RESULTS_SUFFIX
snapshots_file = joinpath(base_folder, "snapshots.jld2")

if !isfile(snapshots_file)
    error("Snapshots file not found at $snapshots_file")
end

file = jldopen(snapshots_file, "r")
grid = file["grid"]

fig_size = (900, 800)
plot_dir = base_folder

# Axes are grid coordinates: the walls are ghost nodes, so physical coordinates are shifted by +dx
diff_ops = nothing

for i in 200:200:25000
    if haskey(file, "omega_field_$i")
        @info "Generating plots for step $i"
        omega_field = file["omega_field_$i"]
        psi_field = file["psi_field_$i"]
        psi = last(psi_field)

        # The velocity of the stream function: u = ∂ᵧψ, v = -∂ₓψ
        global diff_ops = isnothing(diff_ops) ? DifferentialOpsScalar(grid, psi_field; stencil_order=2) : diff_ops
        u = apply(diff_ops.d_ops[2], psi)
        v = -1.0 * apply(diff_ops.d_ops[1], psi)

        # The vorticity is O(U/h) at the lid: clip the colour range to show the interior
        visualize_scalar(
            omega_field,
            length(omega_field),
            grid,
            joinpath(plot_dir, "vorticity_$(i).svg");
            npoints=256,
            aspect_ratio=:equal,
            size=fig_size,
            clims=(-5.0, 5.0),
        )
        visualize_scalar(
            psi_field,
            length(psi_field),
            grid,
            joinpath(plot_dir, "streamfunction_$(i).svg");
            npoints=256,
            aspect_ratio=:equal,
            size=fig_size,
        )
        visualize_vector(
            (u, v),
            grid,
            psi_field.arch,
            joinpath(plot_dir, "vector_$(i).svg");
            npoints=256,
            quiver_npoints=16,
            arrow_scale=0.05,
            aspect_ratio=:equal,
            size=fig_size,
        )

        if haskey(file, "poisson_residual_$i") && !isnothing(file["poisson_residual_$i"])
            poisson_residual = file["poisson_residual_$i"]
            visualize_scalar(
                poisson_residual,
                grid,
                psi_field.arch,
                joinpath(plot_dir, "poisson_residual_$(i).svg");
                npoints=256,
                aspect_ratio=:equal,
                size=fig_size,
            )
        end
    end
end

close(file)
@info "Plotting complete."

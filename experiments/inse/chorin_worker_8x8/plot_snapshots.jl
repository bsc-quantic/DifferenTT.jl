using ITensors
using ITensorMPS
using DifferenTT
using JLD2
using Plots

base_folder = "experiments/inse/chorin_worker_8x8/results_nocond"
snapshots_file = joinpath(base_folder, "snapshots.jld2")

if !isfile(snapshots_file)
    error("Snapshots file not found at $snapshots_file")
end

file = jldopen(snapshots_file, "r")
grid = file["grid"]

fig_size = (1200, 600)
plot_dir = base_folder

for i in 200:200:3000
    if haskey(file, "u_fields_$i")
        @info "Generating plots for step $i"
        u_fields = file["u_fields_$i"]
        p_field = file["p_field_$i"]

        visualize_vector(u_fields, length(u_fields), grid, joinpath(plot_dir, "vector_$(i).svg"); npoints=256, quiver_npoints=16, arrow_scale=0.05, aspect_ratio=:equal, size=fig_size)
        visualize_vorticity(u_fields, length(u_fields), grid, joinpath(plot_dir, "vorticity_$(i).svg"); npoints=256, quiver_npoints=16, arrow_scale=0.0, aspect_ratio=:equal, size=fig_size)
        visualize_scalar(p_field, length(p_field), grid, joinpath(plot_dir, "press_$(i).svg"); npoints=256, aspect_ratio=:equal, size=fig_size)

        if haskey(file, "poisson_residual_$i")
            poisson_residual = file["poisson_residual_$i"]
            visualize_scalar(poisson_residual, grid, p_field.arch, joinpath(plot_dir, "poisson_residual_$(i).svg"); npoints=256, aspect_ratio=:equal, size=fig_size)
        end
    else
        @warn "Snapshots for step $i not found in the file."
    end
end

close(file)
@info "Plotting complete."

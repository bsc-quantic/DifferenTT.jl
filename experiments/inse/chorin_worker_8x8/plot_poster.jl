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

# Step to render on the poster. Pick a snapshot where the flow looks developed.
step = 1600

# Sampling resolution for the background magnitude field and for the arrows.
npoints = 256
quiver_npoints = 18
arrow_scale = 0.06

plot_dir = base_folder
out_path_svg = joinpath(plot_dir, "poster_vector_$(step).svg")
out_path_png = joinpath(plot_dir, "poster_vector_$(step).png")

file = jldopen(snapshots_file, "r")
grid = file["grid"]

if !haskey(file, "u_fields_$step")
    close(file)
    error("Snapshot for step $step not found in $snapshots_file")
end

u_fields = file["u_fields_$step"]
close(file)

n_components = length(u_fields.components)
last_idx = length(u_fields)
mpss = Tuple(u_fields.components[i].u[last_idx] for i in 1:n_components)
arch = u_fields.arch

# White -> deep navy sequential palette, matched to the BSC/UPC poster banner blue.
# Stops are skewed so most of the field stays white/light (background is mostly
# uniform flow) and only the higher-magnitude regions pick up the deep navy accent.
poster_cmap = cgrad(
    [:white, colorant"#dce8f5", colorant"#aec2dd", colorant"#4a6fa1", colorant"#1b2f5e", colorant"#0f1c3d"],
    [0.0, 0.35, 0.55, 0.75, 0.9, 1.0],
)
arrow_color = colorant"#1b2f5e"

h_ranges, h_u = samples(mpss[1], grid, arch, npoints)
_, h_v = samples(mpss[2], grid, arch, npoints)
mag = sqrt.(h_u .^ 2 .+ h_v .^ 2)

plt = heatmap(
    h_ranges[1],
    h_ranges[2],
    mag';
    color=poster_cmap,
    colorbar=false,
    aspect_ratio=:equal,
    axis=false,
    ticks=false,
    grid=false,
    framestyle=:none,
    legend=false,
    title="",
    margin=0Plots.mm,
    size=(1200, 600),
)

q_ranges, q_u = samples(mpss[1], grid, arch, quiver_npoints)
_, q_v = samples(mpss[2], grid, arch, quiver_npoints)

X = [x for x in q_ranges[1], y in q_ranges[2]]
Y = [y for x in q_ranges[1], y in q_ranges[2]]

quiver!(plt, X[:], Y[:]; quiver=(arrow_scale .* q_u[:], arrow_scale .* q_v[:]), color=arrow_color, linewidth=1.5)

mkpath(plot_dir)
savefig(plt, out_path_svg)
savefig(plt, out_path_png)

@info "Poster plot saved" out_path_svg out_path_png

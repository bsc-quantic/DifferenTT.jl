"""
    Lid-Driven Cavity: comparison with Ghia, Ghia & Shin (1982)

Reads the last snapshot of the worker and compares the centreline velocities and the primary vortex with the
Re = 1000 benchmark of Ghia, Ghia & Shin, J. Comput. Phys. 48, 387-411 (1982), Tables I, II and V.

The walls are ghost nodes: interior node k (1-based) sits at the physical coordinate k·h, and the walls at 0 and
(K + 1)·h = L. The velocity is rebuilt on the nodes and the walls from ψ with the same central differences as the
solver, then interpolated linearly on the centrelines x = 0.5 and y = 0.5.

Only for the single-lid cavity: Ghia et al. have no doubly-driven reference data.

Usage: julia --project=. experiments/inse/liddriven_cavity/ghia_comparison.jl [nocond]
"""

using ITensors
using ITensorMPS
using DifferenTT
using JLD2
using Plots
using Printf
using Statistics

get(ENV, "CAVITY_LIDS", "1") == "1" || error("ghia_comparison.jl only applies to the single-lid cavity (CAVITY_LIDS=1)")

const use_nocond = length(ARGS) > 0 && ARGS[1] == "nocond"
base_folder = use_nocond ? "experiments/inse/liddriven_cavity/results_nocond" : "experiments/inse/liddriven_cavity/results"
snapshots_file = joinpath(base_folder, "snapshots.jld2")
isfile(snapshots_file) || error("Snapshots file not found at $snapshots_file")

const U_LID = 1.0

# Ghia, Ghia & Shin (1982), Re = 1000 columns, checked against Tables I, II and V of the paper. Their uniform
# 129 × 129 grid includes the walls (h = 1/128), so the tabulated points are nodes of their grid, not of ours.
# Their tabulated vorticities (Tables IV and V, not used here) have the opposite sign to ω = ∂ₓv - ∂ᵧu.
# Table I: u along the vertical line through the geometric centre (x = 0.5)
const GHIA_Y = [1.0000, 0.9766, 0.9688, 0.9609, 0.9531, 0.8516, 0.7344, 0.6172, 0.5000, 0.4531, 0.2813, 0.1719, 0.1016, 0.0703, 0.0625, 0.0547, 0.0000]
const GHIA_U = [1.00000, 0.65928, 0.57492, 0.51117, 0.46604, 0.33304, 0.18719, 0.05702, -0.06080, -0.10648, -0.27805, -0.38289, -0.29730, -0.22220, -0.20196, -0.18109, 0.00000]
# Table II: v along the horizontal line through the geometric centre (y = 0.5)
const GHIA_X = [1.0000, 0.9688, 0.9609, 0.9531, 0.9453, 0.9063, 0.8594, 0.8047, 0.5000, 0.2344, 0.2266, 0.1563, 0.0938, 0.0781, 0.0703, 0.0625, 0.0000]
const GHIA_V = [0.00000, -0.21388, -0.27669, -0.33714, -0.39188, -0.51550, -0.42665, -0.31966, 0.02526, 0.32235, 0.33075, 0.37095, 0.32627, 0.30353, 0.29012, 0.27485, 0.00000]
# Table V: primary vortex
const GHIA_PSI_MIN = -0.117929
const GHIA_PSI_MIN_LOC = (0.5313, 0.5625)

function interp1(xs::AbstractVector, ys::AbstractVector, x::Real)
    k = clamp(searchsortedlast(xs, x), 1, length(xs) - 1)
    w = (x - xs[k]) / (xs[k+1] - xs[k])
    return (1 - w) * ys[k] + w * ys[k+1]
end

file = jldopen(snapshots_file, "r")
steps = [parse(Int, k[length("psi_field_")+1:end]) for k in keys(file) if startswith(k, "psi_field_")]
isempty(steps) && error("No psi_field snapshots in $snapshots_file")
last_step = maximum(steps)
grid = file["grid"]
psi_field = file["psi_field_$last_step"]
omega_field = file["omega_field_$last_step"]
close(file)

psi = last(psi_field)
arch = psi_field.arch
K1, K2 = variable_dims(arch)
hx, hy = grid.dx
@info "Comparing step $last_step" nodes = (K1, K2) h = grid.dx

# ψ on the interior nodes, padded with the walls (ψ = 0)
Ψ = [DifferenTT.sample(psi, ((i - 1) * hx, (j - 1) * hy), grid, arch) for i in 1:K1, j in 1:K2]
Ψp = zeros(K1 + 2, K2 + 2)
Ψp[2:end-1, 2:end-1] .= Ψ
xs = hx .* (0:K1+1)
ys = hy .* (0:K2+1)

# Velocity on nodes and walls: central differences inside, no-slip walls and the moving lid on the boundary
U = zeros(K1 + 2, K2 + 2)
V = zeros(K1 + 2, K2 + 2)
U[2:end-1, 2:end-1] .= (Ψp[2:end-1, 3:end] .- Ψp[2:end-1, 1:end-2]) ./ (2hy)
V[2:end-1, 2:end-1] .= -(Ψp[3:end, 2:end-1] .- Ψp[1:end-2, 2:end-1]) ./ (2hx)
U[2:end-1, end] .= U_LID

# Centrelines, interpolated in the normal direction at 0.5, then at Ghia's points
u_centre = [interp1(xs, U[:, j], 0.5) for j in 1:K2+2]
v_centre = [interp1(ys, V[i, :], 0.5) for i in 1:K1+2]
u_at_ghia = [interp1(ys, u_centre, y) for y in GHIA_Y]
v_at_ghia = [interp1(xs, v_centre, x) for x in GHIA_X]

# The first and last Ghia points are wall values (boundary conditions): leave them out of the error
interior = 2:length(GHIA_Y)-1
u_rms = sqrt(mean((u_at_ghia[interior] .- GHIA_U[interior]) .^ 2))
v_rms = sqrt(mean((v_at_ghia[interior] .- GHIA_V[interior]) .^ 2))
centreline_rms = sqrt(mean(vcat(u_at_ghia[interior] .- GHIA_U[interior], v_at_ghia[interior] .- GHIA_V[interior]) .^ 2))

psi_min, idx = findmin(Ψ)
psi_min_loc = (idx[1] * hx, idx[2] * hy)

chi_psi = maxlinkdim(psi)
chi_omega = maxlinkdim(last(omega_field))

report = """
Lid-driven cavity vs Ghia et al. (1982), Re = 1000, step $last_step, $(K1)×$(K2) interior nodes
  centreline RMS error: $(@sprintf("%.4f", centreline_rms))  (u: $(@sprintf("%.4f", u_rms)), v: $(@sprintf("%.4f", v_rms)))
  ψ_min = $(@sprintf("%.5f", psi_min)) at $(@sprintf("(%.4f, %.4f)", psi_min_loc...))  (Ghia: $(GHIA_PSI_MIN) at $(GHIA_PSI_MIN_LOC))
  χ(ψ) = $chi_psi, χ(ω) = $chi_omega
"""
print(report)
write(joinpath(base_folder, "ghia_comparison.txt"), report)

ENV["GKSwstype"] = "100"
p_u = plot(u_centre, ys; label="ψ–ω TT", xlabel="u(0.5, y)", ylabel="y", title="Vertical centreline")
scatter!(p_u, GHIA_U, GHIA_Y; label="Ghia et al. (1982)")
p_v = plot(xs, v_centre; label="ψ–ω TT", xlabel="x", ylabel="v(x, 0.5)", title="Horizontal centreline")
scatter!(p_v, GHIA_X, GHIA_V; label="Ghia et al. (1982)")
savefig(plot(p_u, p_v; layout=(1, 2), size=(1200, 500)), joinpath(base_folder, "ghia_centrelines.svg"))

jldsave(joinpath(base_folder, "ghia_comparison.jld2");
    step=last_step, centreline_rms, u_rms, v_rms, psi_min, psi_min_loc, chi_psi, chi_omega,
    xs, ys, u_centre, v_centre, u_at_ghia, v_at_ghia)

@info "Ghia comparison complete."

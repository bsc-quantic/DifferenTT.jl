"""
    Paper-Ready Plotter Script: SPD Breakdown of Non-Factorized TT Approximate Inverses (linsolve_spd)

    Targets: `results/linsolve_spd.jld2` written by this directory's worker.jl (one entry per case, stored as
             `<section>/nodes_<nx>x<ny>/<entry>`). This copy is byte-identical to the DifferenTT run
             (experiments/preconditioner_benchmarks/linsolve_spd, SLURM job 412823, 2026-10-08).
    Outputs: `plots_paper/`.

    S = (P + Pᵀ)/2 is the symmetric part of the preconditioner P. It alone sets the sign of xᵀPx, so a negative
    eigenvalue of S proves that P is not SPD. Figure families (each can be switched off below):
      * nneg:   number, fraction or percentage of negative eigenvalues of S against the maximum bond dimension χ,
                one curve per grid. This is the motivation figure at the start of Sec. 3. On a log axis, cases
                with no negative eigenvalue are drawn hollow on a "0" level under the decades.
      * eigmin: |λ_min(S)| / λ_max(S), filled where λ_min(S) < 0; the dashed lines are 1/κ(A), the value of the
                exact inverse.
      * weyl:   ‖S − A⁻¹‖₂ / λ_min(A⁻¹). Below 1, S is SPD by Weyl's inequality. Above 1 nothing is guaranteed.
      * large:  first Chorin step on the benchmark grids, with the local SPD checks of the solver:
                |min Re λ| / max |λ| of the first local P⁻¹A found indefinite (filled, the solve stops there), or
                hollow on the "passed" level when every visited local P⁻¹A was positive. The FAI control is a star.
    The dense families are drawn as
      * `pairs/<family>_<method>.pdf`: legend-free panels at their printed size, one per construction, placed side
        by side as LaTeX subfigures, plus the shared `pairs/legend_strip_<family>.pdf` (as in the other paper
        plotters);
      * `<family>_combined.pdf`: every construction in one panel with its own legend (line style = construction).
    The large family gives `pairs/large_spd_check.pdf` (+ `pairs/legend_strip_large.pdf`) and `large_spd_check.pdf`.
    `plots_paper/figure_notes.txt` records the data, the selection and the numbers the text and captions quote.

Usage: julia --project=. results/linsolve_spd/plot_paper.jl   (from the paper repository root)
"""

using JLD2
using Plots
using Printf
using LaTeXStrings

# =========================================================================
# Paper Plot Styling & Parameters (Tweakable for publication)
# =========================================================================

# Export formats: e.g. [:png, :pdf], [:svg, :pdf], or [:png]
OUTPUT_FORMATS = [:pdf]
DPI = 300
FONT_FAMILY = "Computer Modern"

# Font sizing (standalone figures)
TITLE_FONT_SIZE = 14
GUIDE_FONT_SIZE = 12
TICK_FONT_SIZE = 10
LEGEND_FONT_SIZE = 9

# Dimensions (width, height) in px (standalone figures)
SIZE_STANDARD = (850, 480)
MARGIN_MM     = 6Plots.mm

# Line & Marker styling (standalone figures)
LINE_WIDTH  = 2.2
MARKER_SIZE = 5
MARKER_STROKE_WIDTH = 1.5   # outline of the hollow markers and width of the crosses

# -------------------------------------------------------------------------
# Data selection
# -------------------------------------------------------------------------
RESULTS_FILE = joinpath(@__DIR__, "results", "linsolve_spd.jld2")
PLOTS_DIR    = joinpath(@__DIR__, "plots_paper")

# Constructions of the dense families, in panel order: worker section, output name, legend label and line style
# in the combined figures. A future dense section (e.g. the TT-FAI or the truncated exact L_A⁻¹ of review A1/A10)
# is one more line here.
METHODS = [
    (section="chorin_linsolve_dense", name="linsolve", label=L"TT inversion ($AX + XA = 2I$)", ls=:solid),
    (section="chorin_truncinv_dense", name="truncinv", label=L"Truncated exact $A^{-1}$",      ls=:dash),
]
LARGE_SECTION = "chorin_linsolve_large"

SELECTED_NODES    = nothing   # e.g. [(5, 5), (6, 6)]; nothing = every grid in the data
CHI_RANGE         = nothing   # e.g. 1:10; nothing = every χ in the data
INCLUDE_FULL_RANK = false     # the truncated A⁻¹ also stores its lossless bond dimension (30, 43, 59)

# -------------------------------------------------------------------------
# What is drawn
# -------------------------------------------------------------------------
NNEG_METRIC      = :fraction  # :fraction, :percent or :count of negative eigenvalues of S
YSCALE_NNEG      = :log10     # :log10 or :identity. The fractions span 2e-4 (linsolve) to 0.5 (truncated A⁻¹)
SHARED_YLIMS     = true       # same y range in every panel of a family, so the panels compare directly
ZERO_LEVEL_GAP   = 0.6        # log axes: decades between the lowest decade tick and the "0" level
ZERO_TICK_LABEL  = "0"
CONNECT_ZEROS    = true       # draw the curve down to the "0" level (false: break it there)
ZERO_LINE        = true       # dotted line separating the "0" level from the decades
MARK_SINGULAR    = true       # cases with no negative eigenvalue but λ_min(S)/λ_max(S) at rounding level get a cross
SINGULAR_TOL     = 1e-12      # (the χ = 1 truncated A⁻¹ is positive semidefinite and numerically singular)
SINGULAR_MARKER  = :xcross
SHOW_WEYL_THRESHOLD = false   # nneg panels: dashed vertical line per grid at the χ from which ‖S − A⁻¹‖₂ < λ_min(A⁻¹)
SHOW_INV_COND    = true       # eigmin panels: dashed 1/κ(A) per grid

# Grid labels: :grid (16 × 16), :nodes ((4, 4)), :dofs (n = 256), :grid_dofs (16 × 16 (n = 256))
SIZE_LABEL_STYLE = :grid
SIZE_COLOR_SCHEME     = :tab10     # palette (or gradient if SIZE_COLOR_SEQUENTIAL), one color per grid
SIZE_COLOR_SEQUENTIAL = false      # true: sample SIZE_COLOR_SCHEME as a gradient, e.g. :viridis or :Blues
SIZE_GRADIENT_RANGE   = (0.2, 0.9)
SIZE_MARKERS     = [:circle, :square, :diamond, :utriangle, :dtriangle, :pentagon]
FAI_MARKER       = :star5
COLOR_REFERENCE  = :black          # Weyl threshold, legend samples of the line styles
COLOR_CONVENTION = :gray20         # legend samples of the filled / hollow / cross conventions
XTICK_STEP       = nothing         # nothing: 1 up to 8 bond dimensions, 2 up to 16, 5 above
LARGE_LINES      = false           # large family: join the χ of a grid with a line (the values do not form a trend)
LARGE_X_DODGE    = 0.15            # large family: horizontal offset between grids, so coinciding markers stay visible

# -------------------------------------------------------------------------
# Plot families (true / false)
# -------------------------------------------------------------------------
PLOT_NNEG     = true
PLOT_EIGMIN   = false
PLOT_WEYL     = false
PLOT_LARGE    = false
PLOT_PAIRS    = false   # pairs/ panels + legend strips
PLOT_COMBINED = true    # standalone figures with legends
# Standalone figures drawn at their printed size (fonts in pt, as the pairs), to include with width=\textwidth in
# a figure of COMBINED_PRINT_FRACTION of \textwidth. false: SIZE_STANDARD and the standalone font sizes above.
COMBINED_PRINT_SIZED      = true
COMBINED_PRINT_FRACTION   = 0.75
COMBINED_PRINT_ASPECT     = 0.62
COMBINED_LEGEND_FONT_SIZE = 7

# Per-figure axis limits, keyed by the output name without extension: either the file name ("nneg_linsolve"),
# applied in every folder, or the path relative to PLOTS_DIR ("pairs/nneg_linsolve"). Value: (min, max).
XLIMS = Dict{String, Tuple{Float64, Float64}}(
    # "nneg_combined" => (0.5, 10.5),
)
YLIMS = Dict{String, Tuple{Float64, Float64}}()

# Legend Position Toggles (e.g. :topleft, :topright, :bottomleft, :bottomright, :outertopright, :none)
LEGEND_POS_NNEG   = :topright
LEGEND_POS_EIGMIN = :bottomright
LEGEND_POS_WEYL   = :topright
LEGEND_POS_LARGE  = :outertopright

# Axis labels
XLABEL_CHI = L"Maximum bond dimension $\chi$"
YLABEL_NNEG = Dict(:fraction => L"Fraction of negative eigenvalues of $S$",
                   :percent  => L"Negative eigenvalues of $S$ (\%)",
                   :count    => L"Negative eigenvalues of $S$")
YLABEL_EIGMIN = L"|\lambda_{\min}(S)| \,/\, \lambda_{\max}(S)"
YLABEL_WEYL   = L"\|S - A^{-1}\|_2 \,/\, \lambda_{\min}(A^{-1})"
YLABEL_LARGE  = L"Local $P^{-1}A$: $|\min\,\mathrm{Re}\,\lambda| \,/\, \max |\lambda|$"
LARGE_PASS_TICK_LABEL = "passed"

# -------------------------------------------------------------------------
# Paper figure pairs (plots_paper/pairs/), as in ../chorin_pipeline_stats/plot_paper.jl: legend-free panels drawn at
# their printed size, placed side by side as LaTeX subfigures (PAIR_PANEL_FRACTION of \textwidth, included with
# width=\textwidth), plus one legend strip per family that goes under the pair at full \textwidth. Sizes in pt.
# -------------------------------------------------------------------------
PAIR_TEXTWIDTH_PT   = 390      # \textwidth of the paper (elsarticle preprint, 12pt)
PAIR_PANEL_FRACTION = 0.49     # subfigure width as a fraction of \textwidth
PAIR_ASPECT         = 0.8      # panel height / width
PAIR_SHOW_TITLES    = false    # construction name above each panel (normally the LaTeX subcaption says it)
PAIR_YLABEL_NNEG = Dict(:fraction => "Negative eigenvalue fraction", :percent => "Negative eigenvalues (%)",
                        :count => "Negative eigenvalues")
PAIR_YLABEL_EIGMIN = YLABEL_EIGMIN
PAIR_YLABEL_WEYL   = YLABEL_WEYL
PAIR_YLABEL_LARGE  = L"$|\min\,\mathrm{Re}\,\lambda| \,/\, \max |\lambda|$"
PAIR_XLABEL_CHI    = L"Maximum bond dimension $\chi$"
# GR draws text about 1.5 times its nominal pt size when 1 px = 1 pt, so the pair figures are drawn
# PAIR_CANVAS_SCALE times larger than printed and LaTeX scales them back down: font sizes below are the printed
# sizes, and line widths / marker sizes are given in printed units too.
PAIR_CANVAS_SCALE       = 1.5
PAIR_TITLE_FONT_SIZE    = 8
PAIR_GUIDE_FONT_SIZE    = 8
PAIR_TICK_FONT_SIZE     = 7
PAIR_LEGEND_FONT_SIZE   = 8
PAIR_LINE_WIDTH         = 1.0
PAIR_MARKER_SIZE        = 2.5
PAIR_MARKER_STROKE_WIDTH = 2.5  # GR draws marker outlines much thinner than lines of the same width on small canvases
PAIR_MARGIN             = 1Plots.mm
PAIR_LEGEND_COLUMNS     = 3    # entries per legend row (-1 = all in a single row)
PAIR_LEGEND_ROW_HEIGHT  = 15   # pt
PAIR_LEGEND_LINE_LENGTH = 0.2  # sample line length, as a fraction of the column width

# =========================================================================

gr()
default(fmt=:png, dpi=DPI, size=SIZE_STANDARD, fontfamily=FONT_FAMILY, margin=MARGIN_MM,
        tickfontsize=TICK_FONT_SIZE, guidefontsize=GUIDE_FONT_SIZE,
        titlefontsize=TITLE_FONT_SIZE, legendfontsize=LEGEND_FONT_SIZE)

ENV["GKSwstype"] = "100"
mkpath(PLOTS_DIR)

const PAIR_PANEL_W = PAIR_CANVAS_SCALE * PAIR_PANEL_FRACTION * PAIR_TEXTWIDTH_PT
const PAIR_PANEL_KW = (; legend=false, size=round.(Int, (PAIR_PANEL_W, PAIR_ASPECT * PAIR_PANEL_W)),
                       margin=PAIR_CANVAS_SCALE * PAIR_MARGIN, framestyle=:box,
                       guidefontsize=PAIR_GUIDE_FONT_SIZE, tickfontsize=PAIR_TICK_FONT_SIZE,
                       titlefontsize=PAIR_TITLE_FONT_SIZE)
const COMBINED_W = PAIR_CANVAS_SCALE * COMBINED_PRINT_FRACTION * PAIR_TEXTWIDTH_PT
const STANDARD_KW = COMBINED_PRINT_SIZED ?
    (; legend=:none, size=round.(Int, (COMBINED_W, COMBINED_PRINT_ASPECT * COMBINED_W)), framestyle=:box,
       margin=PAIR_CANVAS_SCALE * PAIR_MARGIN, guidefontsize=PAIR_GUIDE_FONT_SIZE, tickfontsize=PAIR_TICK_FONT_SIZE,
       legendfontsize=COMBINED_LEGEND_FONT_SIZE) :
    (; legend=:none, size=SIZE_STANDARD, framestyle=:box)
# Line width, marker size and marker stroke of the standalone figures
const STD_LW, STD_MS, STD_MSW = COMBINED_PRINT_SIZED ?
    PAIR_CANVAS_SCALE .* (PAIR_LINE_WIDTH, PAIR_MARKER_SIZE, PAIR_MARKER_STROKE_WIDTH) :
    (LINE_WIDTH, MARKER_SIZE, MARKER_STROKE_WIDTH)

"""Save in all configured publication formats, applying any XLIMS / YLIMS override."""
function save_paper_plot(p, base_path_no_ext)
    mkpath(dirname(base_path_no_ext))
    rel, name = relpath(base_path_no_ext, PLOTS_DIR), basename(base_path_no_ext)
    lims_for(d) = get(d, rel, get(d, name, nothing))
    (xl = lims_for(XLIMS)) === nothing || xlims!(p, xl...)
    (yl = lims_for(YLIMS)) === nothing || ylims!(p, yl...)
    for fmt in OUTPUT_FORMATS
        savefig(p, "$(base_path_no_ext).$(fmt)")
    end
    @info "Saved: $(base_path_no_ext).($(join(OUTPUT_FORMATS, ", ")))"
end

islog(scale) = !(scale in (:identity, :linear, :none))

# =========================================================================
# Load data
# =========================================================================

isfile(RESULTS_FILE) || error("Could not find $RESULTS_FILE")
@info "Loading $RESULTS_FILE"

parse_nodes(case) = (m = match(r"^nodes_(\d+)x(\d+)$", case); m === nothing ? nothing : (parse(Int, m[1]), parse(Int, m[2])))
keep_nodes(nodes) = nodes !== nothing && (SELECTED_NODES === nothing || nodes in SELECTED_NODES)
keep_chi(χ) = CHI_RANGE === nothing || χ in CHI_RANGE

"""Dense section: grid => (op, entries sorted by χ), after the χ selection."""
function load_dense(f, section)
    out = Dict{NTuple{2, Int}, Any}()
    haskey(f, section) || (@warn "Section $section not in $RESULTS_FILE"; return out)
    for case in keys(f[section])
        nodes = parse_nodes(case)
        keep_nodes(nodes) || continue
        names = keys(f[section][case])
        op = "operator" in names ? f["$section/$case/operator"] : nothing
        entries = [f["$section/$case/$k"] for k in names if startswith(k, "chi_")]
        filter!(e -> keep_chi(e["chi"]) && (INCLUDE_FULL_RANK || e["chi"] != get(e, "full_chi", -1)), entries)
        isempty(entries) || (out[nodes] = (op=op, entries=sort(entries; by=e -> e["chi"])))
    end
    return out
end

"""Large section: grid => (linsolve entries sorted by χ, FAI entries)."""
function load_large(f, section)
    out = Dict{NTuple{2, Int}, Any}()
    haskey(f, section) || return out
    for case in keys(f[section])
        nodes = parse_nodes(case)
        keep_nodes(nodes) || continue
        names = keys(f[section][case])
        ls = sort([f["$section/$case/$k"] for k in names if startswith(k, "chi_")]; by=e -> e["chi"])
        fai = [f["$section/$case/$k"] for k in names if startswith(k, "fai_")]
        out[nodes] = (linsolve=ls, fai=fai)
    end
    return out
end

dense, large, config = jldopen(RESULTS_FILE, "r") do f
    Dict(m.name => load_dense(f, m.section) for m in METHODS),
    load_large(f, LARGE_SECTION),   # loaded even if not drawn: it fixes the grid colors and feeds the notes
    haskey(f, "meta/config") ? f["meta/config"] : Dict{String, Any}()
end

# One color and marker per grid, shared by every figure (dense and large grids together)
const ALL_NODES = sort(unique(vcat([collect(keys(d)) for d in values(dense)]..., collect(keys(large)))); by=sum)
function make_colors(n)
    n == 0 && return []
    if SIZE_COLOR_SEQUENTIAL
        g = cgrad(SIZE_COLOR_SCHEME)
        zs = n == 1 ? [SIZE_GRADIENT_RANGE[2]] : range(SIZE_GRADIENT_RANGE[1], SIZE_GRADIENT_RANGE[2], length=n)
        return [g[z] for z in zs]
    end
    pal = palette(SIZE_COLOR_SCHEME)
    return [pal[mod1(i, length(pal))] for i in 1:n]
end
const SIZE_COLOR  = Dict(zip(ALL_NODES, make_colors(length(ALL_NODES))))
const SIZE_MARKER = Dict(nodes => SIZE_MARKERS[mod1(i, length(SIZE_MARKERS))] for (i, nodes) in enumerate(ALL_NODES))

function size_label(nodes)
    nx, ny = 2^nodes[1], 2^nodes[2]
    SIZE_LABEL_STYLE === :nodes     && return "($(nodes[1]), $(nodes[2]))"
    SIZE_LABEL_STYLE === :dofs      && return latexstring("\$n = $(nx * ny)\$")
    SIZE_LABEL_STYLE === :grid_dofs && return latexstring("\$$nx \\times $ny\$ (\$n = $(nx * ny)\$)")
    return latexstring("\$$nx \\times $ny\$")
end

dense_nodes(name) = sort(collect(keys(dense[name])); by=sum)
active_methods() = [m for m in METHODS if !isempty(dense[m.name])]

# =========================================================================
# Point values of each family
#   kind: :neg (filled marker), :pos (hollow), :zero (hollow, on the "0" level of a log axis),
#         :singular (cross on the "0" level), :pass (hollow on the "passed" level)
# =========================================================================

is_singular(e) = e["nneg_S"] == 0 && abs(e["eigmin_S"]) / e["eigmax_S"] < SINGULAR_TOL
nneg_scale(e) = NNEG_METRIC === :count ? 1.0 : (NNEG_METRIC === :percent ? 100.0 : 1.0) / e["n"]

function family_points(family, e)
    neg = e["nneg_S"] > 0
    if family === :nneg
        return e["nneg_S"] * nneg_scale(e), neg ? :neg : (MARK_SINGULAR && is_singular(e) ? :singular : :zero)
    elseif family === :eigmin
        is_singular(e) && return 0.0, (MARK_SINGULAR ? :singular : :zero)
        return abs(e["eigmin_S"]) / e["eigmax_S"], neg ? :neg : :pos
    elseif family === :weyl
        return e["sym_error_2norm"] / e["eigmin_Ainv"], neg ? :neg : :pos
    end
    error("Unknown family $family")
end

on_zero_level(k) = k in (:zero, :singular, :pass)

"""Smallest χ of the selection from which Weyl's inequality guarantees S ≻ 0 for every larger χ (nothing if never)."""
function weyl_threshold(entries)
    ok = [e["sym_error_2norm"] < e["eigmin_Ainv"] for e in entries]
    i = findlast(!, ok)
    i === nothing && return entries[1]["chi"]
    return i == length(entries) ? nothing : entries[i + 1]["chi"]
end

# =========================================================================
# Axes
# =========================================================================

"""Integer bond-dimension ticks."""
function chi_ticks(χs)
    isempty(χs) && return :auto
    lo, hi = extrema(χs)
    step = XTICK_STEP !== nothing ? XTICK_STEP : (hi - lo < 8 ? 1 : hi - lo < 16 ? 2 : 5)
    ticks = collect(lo:step:hi)
    return (ticks, string.(ticks))
end

"""
Log y axis with decade ticks (L"10^{k}") and, when some points sit on it, a labelled level under the decades
(the "0" or "passed" level). Returns (; ylims, yticks, zero_y, sep_y).
"""
function log_axis(values, needs_zero; zero_label=ZERO_TICK_LABEL)
    v = filter(x -> isfinite(x) && x > 0, values)
    isempty(v) && (v = [1.0])
    lo_dec, hi_dec = floor(Int, log10(minimum(v))), ceil(Int, log10(maximum(v)) - 1e-9)
    hi_dec = max(hi_dec, lo_dec + 1)
    step = max(1, ceil(Int, (hi_dec - lo_dec) / 6))
    decs = collect(hi_dec:-step:lo_dec)[end:-1:1]
    ticks = Float64[10.0^d for d in decs]
    labels = Any[latexstring("10^{$d}") for d in decs]
    zero_y = 10.0^(decs[1] - ZERO_LEVEL_GAP)
    bottom = needs_zero ? zero_y / 10^0.3 : 10.0^(log10(minimum(v)) - 0.3)
    top = max(10.0^hi_dec, 10.0^(log10(maximum(v)) + 0.15))
    if needs_zero
        pushfirst!(ticks, zero_y)
        pushfirst!(labels, zero_label)
    end
    return (; ylims=(bottom, top), yticks=(ticks, labels), zero_y,
             sep_y=needs_zero ? 10.0^(decs[1] - ZERO_LEVEL_GAP / 2) : nothing)
end

"""(ylims, yticks, zero_y, sep_y) of a family drawn over `entries_by_panel` (vectors of entries)."""
function family_axis(family, yscale, all_entries)
    pts = [family_points(family, e) for e in all_entries]
    if !islog(yscale)
        vals = [k === :neg || k === :pos ? y : 0.0 for (y, k) in pts]
        top = isempty(vals) ? 1.0 : maximum(vals)
        return (; ylims=(-0.04 * top, 1.06 * top), yticks=:auto, zero_y=0.0, sep_y=nothing)
    end
    return log_axis([y for (y, k) in pts if !on_zero_level(k)], any(on_zero_level(k) for (_, k) in pts))
end

# =========================================================================
# Drawing
# =========================================================================

"""Curve of one grid: filled markers on :neg, hollow on :pos / :zero / :pass, crosses on :singular."""
function draw_curve!(p, x, y, kinds; color, marker, ls, lw, ms, msw, zero_y, line=true)
    yplot = [on_zero_level(k) ? zero_y : v for (v, k) in zip(y, kinds)]
    yline = CONNECT_ZEROS ? yplot : [on_zero_level(k) ? NaN : v for (v, k) in zip(yplot, kinds)]
    line && plot!(p, x, yline; color, ls, lw, label="")
    sel(f) = findall(f, kinds)
    for (idx, mk, fill) in ((sel(==(:neg)), marker, color),
                            (sel(k -> k in (:pos, :zero, :pass)), marker, :white),
                            (sel(==(:singular)), SINGULAR_MARKER, color))
        isempty(idx) && continue
        scatter!(p, x[idx], yplot[idx]; marker=mk, markercolor=fill, markerstrokecolor=color, markersize=ms,
                 markerstrokewidth=msw, label="")
    end
    return p
end

"""Dotted separator between the labelled bottom level and the decades of a log axis."""
separator!(p, ax, lw) = (ZERO_LINE && ax.sep_y !== nothing && hline!(p, [ax.sep_y]; color=:gray70, ls=:dot, lw=lw / 2, label=""); p)

"""One dense panel: every grid of `methods` (one construction for the pairs, all of them for the combined figure)."""
function dense_panel(family, methods, yscale, ax; pair::Bool, ylabel, title="")
    lw = pair ? PAIR_CANVAS_SCALE * PAIR_LINE_WIDTH : STD_LW
    ms = pair ? PAIR_CANVAS_SCALE * PAIR_MARKER_SIZE : STD_MS
    msw = pair ? PAIR_CANVAS_SCALE * PAIR_MARKER_STROKE_WIDTH : STD_MSW
    χs = sort(unique(e["chi"] for m in methods for nodes in dense_nodes(m.name) for e in dense[m.name][nodes].entries))
    p = plot(; xlabel=pair ? PAIR_XLABEL_CHI : XLABEL_CHI, ylabel, yscale=islog(yscale) ? :log10 : :identity,
             xticks=chi_ticks(χs), xlims=(minimum(χs) - 0.5, maximum(χs) + 0.5), ylims=ax.ylims, yticks=ax.yticks,
             title, (pair ? PAIR_PANEL_KW : STANDARD_KW)...)
    islog(yscale) && separator!(p, ax, lw)
    family === :weyl && hline!(p, [1.0]; color=COLOR_REFERENCE, ls=:dash, lw, label="")
    for m in methods, nodes in dense_nodes(m.name)
        d = dense[m.name][nodes]
        color = SIZE_COLOR[nodes]
        pts = [family_points(family, e) for e in d.entries]
        draw_curve!(p, [e["chi"] for e in d.entries], first.(pts), last.(pts);
                    color, marker=SIZE_MARKER[nodes], ls=length(methods) > 1 ? m.ls : :solid, lw, ms, msw, zero_y=ax.zero_y)
        if family === :eigmin && SHOW_INV_COND && d.op !== nothing
            hline!(p, [1 / d.op["cond_A"]]; color, ls=:dashdot, lw=lw / 2, label="")
        end
        if family === :nneg && SHOW_WEYL_THRESHOLD
            χw = weyl_threshold(d.entries)
            χw === nothing || vline!(p, [χw]; color, ls=:dash, lw=lw / 2, label="")
        end
    end
    return p
end

# Legend entries: (label, color, ls, marker, hollow, line) drawn either in the plot legend or in a strip
legend_entry(label, color; ls=:solid, marker=:none, hollow=false, line=true) = (; label, color, ls, marker, hollow, line)

function size_entries(nodes_list; line=true)
    [legend_entry(size_label(n), SIZE_COLOR[n]; marker=SIZE_MARKER[n], line) for n in nodes_list]
end

"""Conventions present in a family: filled / hollow / cross / reference lines."""
function convention_entries(family, entries)
    kinds = Set(last(family_points(family, e)) for e in entries)
    out = []
    if family === :nneg
        :zero in kinds && push!(out, legend_entry("no negative eigenvalue", COLOR_CONVENTION; marker=:circle, hollow=true, line=false))
        :singular in kinds && push!(out, legend_entry("PSD, numerically singular", COLOR_CONVENTION; marker=SINGULAR_MARKER, line=false))
        SHOW_WEYL_THRESHOLD && push!(out, legend_entry("Weyl guarantee from here", COLOR_CONVENTION; ls=:dash))
    else
        push!(out, legend_entry(L"$\lambda_{\min}(S) < 0$", COLOR_CONVENTION; marker=:circle, line=false))
        push!(out, legend_entry(L"$\lambda_{\min}(S) > 0$", COLOR_CONVENTION; marker=:circle, hollow=true, line=false))
        family === :eigmin && :singular in kinds &&
            push!(out, legend_entry("numerically singular", COLOR_CONVENTION; marker=SINGULAR_MARKER, line=false))
        family === :eigmin && SHOW_INV_COND && push!(out, legend_entry(L"$1/\kappa(A)$ (exact inverse)", COLOR_CONVENTION; ls=:dashdot))
        family === :weyl && push!(out, legend_entry("Weyl threshold", COLOR_REFERENCE; ls=:dash))
    end
    return out
end

"""Invisible series so that the plot legend shows `entries`."""
function legend_series!(p, entries; lw, ms, msw=STD_MSW)
    for e in entries
        plot!(p, [NaN], [NaN]; label=e.label, color=e.color, ls=e.ls, lw=e.line ? lw : 0,
              linealpha=e.line ? 1 : 0, marker=e.marker, markersize=ms,
              markercolor=e.hollow ? :white : e.color, markerstrokecolor=e.color, markerstrokewidth=msw)
    end
    return p
end

"""Legend-only figure: the entries filled row by row into `columns` equal columns, no axes. Drawn by hand
(sample line + marker + text per cell) because GR's multi-column legend overlaps rows at these sizes."""
function legend_strip(entries; columns, width)
    ncol = columns == -1 ? length(entries) : min(columns, length(entries))
    nrow = cld(length(entries), ncol)
    lw, ms = PAIR_CANVAS_SCALE * PAIR_LINE_WIDTH, PAIR_CANVAS_SCALE * PAIR_MARKER_SIZE
    p = plot(; size=round.(Int, PAIR_CANVAS_SCALE .* (width, nrow * PAIR_LEGEND_ROW_HEIGHT)), framestyle=:none,
             grid=false, showaxis=false, ticks=false, legend=false, xlims=(0, ncol), ylims=(0, nrow), margin=0Plots.mm)
    for (i, e) in enumerate(entries)
        x0, y = mod1(i, ncol) - 1 + 0.03, nrow - cld(i, ncol) + 0.5
        x1 = x0 + PAIR_LEGEND_LINE_LENGTH
        e.line && plot!(p, [x0, x1], [y, y]; color=e.color, ls=e.ls, lw)
        # GR thins marker outlines on a canvas this short, so hollow markers are a white marker over a filled one
        # and crosses are two line segments
        xm, msz = (x0 + x1) / 2, e.line ? ms : 1.4 * ms
        if e.marker === SINGULAR_MARKER
            dx, dy = 0.6 * msz * ncol / p[:size][1], 0.6 * msz * nrow / p[:size][2]
            plot!(p, [xm - dx, xm + dx], [y - dy, y + dy]; color=e.color, lw)
            plot!(p, [xm - dx, xm + dx], [y + dy, y - dy]; color=e.color, lw)
        elseif e.marker !== :none
            scatter!(p, [xm], [y]; marker=e.marker, markersize=msz, color=e.color, markerstrokewidth=0)
            e.hollow && scatter!(p, [xm], [y]; marker=e.marker, markersize=0.7 * msz, color=:white, markerstrokewidth=0)
        end
        annotate!(p, x1 + 0.04, y, text(e.label, PAIR_LEGEND_FONT_SIZE, :left, :black, FONT_FAMILY))
    end
    return p
end
save_strip(name, entries) = save_paper_plot(legend_strip(entries; columns=PAIR_LEGEND_COLUMNS, width=PAIR_TEXTWIDTH_PT),
                                            joinpath(PLOTS_DIR, "pairs", name))

# =========================================================================
# Dense families
# =========================================================================

const FAMILIES = [
    (family=:nneg,   enabled=PLOT_NNEG,   yscale=YSCALE_NNEG, ylabel=YLABEL_NNEG[NNEG_METRIC], pair_ylabel=PAIR_YLABEL_NNEG[NNEG_METRIC], legend=LEGEND_POS_NNEG),
    (family=:eigmin, enabled=PLOT_EIGMIN, yscale=:log10,      ylabel=YLABEL_EIGMIN,             pair_ylabel=PAIR_YLABEL_EIGMIN,             legend=LEGEND_POS_EIGMIN),
    (family=:weyl,   enabled=PLOT_WEYL,   yscale=:log10,      ylabel=YLABEL_WEYL,               pair_ylabel=PAIR_YLABEL_WEYL,               legend=LEGEND_POS_WEYL),
]

methods = active_methods()
isempty(methods) && error("No dense data left after the selection")
method_entries(m) = [e for nodes in dense_nodes(m.name) for e in dense[m.name][nodes].entries]
dense_grids = sort(unique(vcat([dense_nodes(m.name) for m in methods]...)); by=sum)

for fam in FAMILIES
    fam.enabled || continue
    all_entries = vcat([method_entries(m) for m in methods]...)
    shared_ax = family_axis(fam.family, fam.yscale, all_entries)
    conventions = convention_entries(fam.family, all_entries)

    if PLOT_PAIRS
        for m in methods
            ax = SHARED_YLIMS ? shared_ax : family_axis(fam.family, fam.yscale, method_entries(m))
            p = dense_panel(fam.family, [m], fam.yscale, ax; pair=true, ylabel=fam.pair_ylabel,
                            title=PAIR_SHOW_TITLES ? m.label : "")
            save_paper_plot(p, joinpath(PLOTS_DIR, "pairs", "$(fam.family)_$(m.name)"))
        end
        save_strip("legend_strip_$(fam.family)", vcat(size_entries(dense_grids), conventions))
    end

    if PLOT_COMBINED
        p = dense_panel(fam.family, methods, fam.yscale, shared_ax; pair=false, ylabel=fam.ylabel)
        style_entries = length(methods) > 1 ? [legend_entry(m.label, COLOR_REFERENCE; ls=m.ls) for m in methods] : []
        legend_series!(p, vcat(size_entries(dense_grids), style_entries, conventions); lw=STD_LW, ms=STD_MS)
        plot!(p; legend=fam.legend)
        save_paper_plot(p, joinpath(PLOTS_DIR, "$(fam.family)_combined"))
    end
end

# =========================================================================
# Large grids: first Chorin step with local SPD checks
# =========================================================================

large_grids = sort(collect(keys(large)); by=sum)
large_value(s) = s["not_spd"] ? (abs(s["not_spd_eigmin_rel"]), :neg) : (0.0, :pass)

if PLOT_LARGE && !isempty(large_grids)
    vals = [abs(e["first_step"]["not_spd_eigmin_rel"]) for n in large_grids for e in large[n].linsolve if e["first_step"]["not_spd"]]
    ax = log_axis(vals, true; zero_label=LARGE_PASS_TICK_LABEL)
    χs = sort(unique(vcat([[e["chi"] for e in large[n].linsolve] for n in large_grids]...,
                          [[e["G_maxdim"] for e in large[n].fai] for n in large_grids]...)))
    for pair in (true, false)
        (pair ? PLOT_PAIRS : PLOT_COMBINED) || continue
        lw = pair ? PAIR_CANVAS_SCALE * PAIR_LINE_WIDTH : STD_LW
        ms = pair ? PAIR_CANVAS_SCALE * PAIR_MARKER_SIZE : STD_MS
        msw = pair ? PAIR_CANVAS_SCALE * PAIR_MARKER_STROKE_WIDTH : STD_MSW
        p = plot(; xlabel=pair ? PAIR_XLABEL_CHI : XLABEL_CHI, ylabel=pair ? PAIR_YLABEL_LARGE : YLABEL_LARGE,
                 yscale=:log10, xticks=chi_ticks(χs), xlims=(minimum(χs) - 0.7, maximum(χs) + 0.7),
                 ylims=ax.ylims, yticks=ax.yticks, (pair ? PAIR_PANEL_KW : STANDARD_KW)...)
        separator!(p, ax, lw)
        for (j, nodes) in enumerate(large_grids)
            color = SIZE_COLOR[nodes]
            dx = LARGE_X_DODGE * (j - (length(large_grids) + 1) / 2)
            for e in large[nodes].linsolve
                e["first_step"]["failed"] && @warn "$(nodes) χ = $(e["chi"]): solver failure ($(first(e["first_step"]["error"], 80))), drawn as passed"
            end
            pts = [large_value(e["first_step"]) for e in large[nodes].linsolve]
            draw_curve!(p, [e["chi"] + dx for e in large[nodes].linsolve], first.(pts), last.(pts);
                        color, marker=SIZE_MARKER[nodes], ls=:solid, lw, ms, msw, zero_y=ax.zero_y, line=LARGE_LINES)
            for e in large[nodes].fai
                y, k = large_value(e["first_step"])
                scatter!(p, [e["G_maxdim"] + dx], [k === :pass ? ax.zero_y : y]; marker=FAI_MARKER, markercolor=color,
                         markerstrokecolor=color, markersize=1.4 * ms, markerstrokewidth=msw, label="")
            end
        end
        entries = vcat(size_entries(large_grids; line=LARGE_LINES),
                       [legend_entry(L"local $P^{-1}A$ indefinite", COLOR_CONVENTION; marker=:circle, line=false),
                        legend_entry(L"all local $P^{-1}A$ positive", COLOR_CONVENTION; marker=:circle, hollow=true, line=false)],
                       any(!isempty(large[n].fai) for n in large_grids) ?
                           [legend_entry("FAI control", COLOR_CONVENTION; marker=FAI_MARKER, line=false)] : [])
        if pair
            save_paper_plot(p, joinpath(PLOTS_DIR, "pairs", "large_spd_check"))
            save_strip("legend_strip_large", entries)
        else
            legend_series!(p, entries; lw, ms, msw)
            plot!(p; legend=LEGEND_POS_LARGE)
            save_paper_plot(p, joinpath(PLOTS_DIR, "large_spd_check"))
        end
    end
end

# =========================================================================
# Figure notes: the numbers the text and captions quote
# =========================================================================

fmt(x) = isfinite(x) ? @sprintf("%.2e", x) : "–"
chi_list(v) = isempty(v) ? "none" : join(v, ", ")
plain_size(nodes) = "$(2^nodes[1])×$(2^nodes[2]) (nodes $(nodes))"

open(joinpath(PLOTS_DIR, "figure_notes.txt"), "w") do io
    println(io, "Figures drawn by linsolve_spd/plot_paper.jl")
    println(io, "Data: ", RESULTS_FILE)
    isempty(config) || println(io, "Worker config: build sweeps = $(get(config, "precond_nsweeps", "?")), linsolve residual target = ",
                                   "$(get(config, "ls_residual_tol", "?")), local SPD tolerance = $(get(config, "spd_tol", "?")), ",
                                   "SLURM job $(get(config, "slurm_job_id", "?"))")
    println(io, "Selection: grids = $(SELECTED_NODES === nothing ? "all" : SELECTED_NODES), χ = $(CHI_RANGE === nothing ? "all" : CHI_RANGE), ",
                "full-rank truncated inverse included: $(INCLUDE_FULL_RANK)")
    println(io, "S = (P + Pᵀ)/2. An eigenvalue counts as negative below -n·eps·max|λ| (worker count_negative).")
    println(io, "nneg metric: $(NNEG_METRIC), y scale $(YSCALE_NNEG), shared y limits: $(SHARED_YLIMS)")
    println(io, "Grid sizes: 2^nx × 2^ny points (QTT nodes (nx, ny)); operator = -(symmetrised 8th-order Laplacian) of the Chorin pressure solve")
    for m in methods
        println(io, "\n[$(m.name)] $(m.section)")
        for nodes in dense_nodes(m.name)
            d = dense[m.name][nodes]
            ind = [e for e in d.entries if e["nneg_S"] > 0]
            imax = isempty(ind) ? nothing : argmax(e -> e["nneg_S"], ind)
            fmax = isempty(ind) ? nothing : argmax(e -> e["nneg_S"] / e["n"], ind)
            sing = [e["chi"] for e in d.entries if is_singular(e)]
            χw = weyl_threshold(d.entries)
            println(io, "  $(plain_size(nodes)), n = $(d.entries[1]["n"])", d.op === nothing ? "" : ", κ(A) = $(fmt(d.op["cond_A"]))",
                        ", λ_min(A⁻¹) = $(fmt(d.entries[1]["eigmin_Ainv"]))")
            println(io, "    indefinite at χ = $(chi_list([e["chi"] for e in ind]))")
            imax === nothing || println(io, "    most negative eigenvalues: $(imax["nneg_S"]) at χ = $(imax["chi"]); ",
                                            "largest fraction $(fmt(fmax["nneg_S"] / fmax["n"])) at χ = $(fmax["chi"])")
            if !isempty(ind) && d.op !== nothing
                rel = [e["sym_error_2norm"] / d.op["eigmax_Ainv"] for e in ind]
                println(io, "    relative spectral error ‖S − A⁻¹‖₂/‖A⁻¹‖₂ of the indefinite cases: $(fmt(minimum(rel)))–$(fmt(maximum(rel)))")
                println(io, "    asymmetry ‖P − Pᵀ‖/‖P‖ of the indefinite cases: $(fmt(minimum(e -> e["asym_P"], ind)))–$(fmt(maximum(e -> e["asym_P"], ind)))")
            end
            isempty(sing) || println(io, "    PSD but numerically singular (|λ_min/λ_max| < $(SINGULAR_TOL)) at χ = $(chi_list(sing))")
            println(io, "    Weyl guarantee (‖S − A⁻¹‖₂ < λ_min(A⁻¹) for every larger χ drawn): ",
                        χw === nothing ? "not reached in the χ range drawn" : "χ ≥ $χw")
        end
    end
    if !isempty(large_grids)
        println(io, "\n[large] $(LARGE_SECTION): first Chorin step, local SPD checks (stop at the first local P⁻¹A with min Re λ < -spd_tol·max|λ|)")
        for nodes in large_grids
            println(io, "  $(plain_size(nodes))")
            for e in large[nodes].linsolve
                s = e["first_step"]
                outcome = s["not_spd"] ? @sprintf("NOT SPD in sweep %d at position %d, min Re λ/max|λ| = %.2e", s["nsweeps_done"],
                                                  s["not_spd_position"], s["not_spd_eigmin_rel"]) :
                          @sprintf("passed, %d sweeps, rel. residual %.2e", s["nsweeps_done"], get(s, "rel_res", NaN))
                println(io, "    linsolve χ = $(e["chi"]): $outcome")
            end
            for e in large[nodes].fai
                s = e["first_step"]
                println(io, "    FAI χ = $(e["G_maxdim"]): ", s["not_spd"] ? "NOT SPD" :
                            @sprintf("passed, %d sweeps, rel. residual %.2e", s["nsweeps_done"], get(s, "rel_res", NaN)))
            end
        end
    end
end
@info "Figure notes written to $(joinpath(PLOTS_DIR, "figure_notes.txt"))"
@info "All paper plots successfully generated in $PLOTS_DIR"

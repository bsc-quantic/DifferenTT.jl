"""
    Chorin pipeline benchmark, statistics variant (paper Table 2 and Figs. 12-13): paper plotter

We draw here the figures of Sec. 5.2 of the paper from the table of repeats that `collect.jl` wrote. Every statistic is
recomputed from the per-repeat rows, so the centre and the band that we draw are chosen in the parameter block below
(mean ± sd, median and IQR, geometric mean ×/÷ gsd, ...), and the speedups are paired by repeat index with the baseline of
the same grid (repeat r uses the same seed in every configuration; `PAIRED_SPEEDUPS = false` gives the ratios of all
pairs).

# Figures

  - speedup against χ_max, heatmaps (μ × χ_max), Pareto front, per-χ_max plots (iterations and speedup per step,
    cumulative time, iterations per site), combined overlays, per-repeat distributions and the scaling summary. The
    per-step curves of the single-χ plots can show every repeat behind the central line;
  - cumulative-time plots (amortization of the preconditioner build), per-step overlays of the residual and of the
    divergence ratio;
  - `plots_paper/pairs/`: legend-free panels per grid (iterations per step and speedups on log axes, Pareto front),
    drawn at their printed size and placed two per row in the paper, plus the shared `legend_strip` (`PLOT_PAIRS`).
    These are the figures of the paper;
  - `plots_paper/stats_used.txt` records which statistics we drew, for the figure captions.

The families other than the pairs are switched off by default (`PLOT_*` flags below). The dataset carries no condition
numbers, so the κ profiles are not drawn.

# Inputs and outputs

Reads `results_nocond/collected_results.jld2` (`"df"`: one row per configuration and repeat) of this folder, written by
`collect.jl nocond`, and writes PDF figures to `plots_paper/` (and subfolders per grid and χ_max) of this folder. In
smoke mode (`DIFFERENTT_SMOKE=1`) it reads and writes the folder `DIFFERENTT_SMOKE_DIR` that the smoke worker wrote, and
draws the single smoke grid; `experiments/smoke.jl` sets it.

# How to run

From the root of the repository, after `collect.jl nocond`:

    julia --project=experiments experiments/paper/chorin_pipeline_stats/plot_paper.jl

# Resources

There is no job script: we run it locally on the collected data.

# Paper

Figs. 12-13 (`fig:chorin_mals_iterations`, `fig:chorin_mals_cumulative_speedup`) of Sec. 5.2; the parameters of Table 2
(`tab:chorin_mals_params`) are the constants of the worker.
"""

using JLD2
using DataFrames
using Plots
using Statistics
using Printf
using LaTeXStrings

# =========================================================================
# Paper Plot Styling & Parameters (Tweakable for publication)
# =========================================================================

# Export formats: e.g. [:png, :pdf], [:svg, :pdf], or [:png]
OUTPUT_FORMATS = [:pdf]
DPI = 300
FONT_FAMILY = "Computer Modern"

# Font sizing
TITLE_FONT_SIZE = 14
GUIDE_FONT_SIZE = 12
TICK_FONT_SIZE = 10
LEGEND_FONT_SIZE = 9
ANNOTATION_FONT_SIZE = 7

# Dimensions (width, height) in px
SIZE_STANDARD = (850, 480)
SIZE_HEATMAP = (750, 480)
SIZE_PARETO = (850, 480)
SIZE_SCALING = (800, 450)
SIZE_COMBINED_SITE = (950, 520)
SIZE_DIST = (900, 480)
MARGIN_MM = 6Plots.mm

# Line & Marker styling
LINE_WIDTH = 2.2
LINE_WIDTH_THICK = 3.0
LINE_WIDTH_THIN = 1.5
MARKER_SIZE = 5
STEP_MARKERS = false   # markers on per-step curves (crowded for long runs)
SITE_MARKERS = true    # markers on per-site curves

# -------------------------------------------------------------------------
# Data selection
# -------------------------------------------------------------------------
include(joinpath(@__DIR__, "..", "..", "common", "smoke.jl"))
# Smoke mode reads (and writes) the folder DIFFERENTT_SMOKE_DIR instead of this folder.
const OUT_ROOT = output_root(@__DIR__)
RESULTS_FILE = joinpath(OUT_ROOT, "results_nocond", "collected_results.jld2")
PLOTS_DIR = joinpath(OUT_ROOT, "plots_paper")

SELECTED_NODES = nothing   # e.g. [(8, 8)]; nothing = every grid in the data
SELECTED_CHI = nothing   # e.g. [2, 4, 6, 8, 10]; nothing = every χ_max in the data
COMBINED_CHI = nothing   # χ_max subset drawn in the combined_* overlays; nothing = all selected
EXCLUDED_KEYS = String[]  # e.g. ["mu_1.0e7_gdim_9"]
EXCLUDED_REPEATS = Int[]     # repeat indices dropped from every configuration
DISCARD_FIRST_IN_PROCESS = false  # drop the first repeat each worker process ran (may carry JIT cost)
STEP_RANGE = nothing   # e.g. 1:100 to crop the per-step plots; nothing = every step
BEST_MU_BY = :solve    # :solve or :total, the speedup that picks the best μ for each χ_max

# -------------------------------------------------------------------------
# Statistics
#   centre: :mean, :median, :gm (geometric mean)
#   spread: :sd, :sem, :ci95, :iqr, :minmax, :gsd (×/÷ geometric sd), :none
# -------------------------------------------------------------------------
TIME_CENTER = :mean;
TIME_SPREAD = :sd    # times, iterations, cumulative times, Pareto
SPEEDUP_CENTER = :gm;
SPEEDUP_SPREAD = :gsd   # aggregate speedups (vs χ, heatmaps, scaling)
PROFILE_CENTER = :median;
PROFILE_SPREAD = :iqr   # per-step / per-site curves (incl. per-step speedup)
PAIRED_SPEEDUPS = true      # false: ratios over every (baseline repeat, configuration repeat) pair

SPREAD_STYLE = :ribbon    # curves vs χ / system size: :ribbon, :errorbar, :none
PROFILE_STYLE = :ribbon    # per-step / per-site curves: :ribbon or :none
BAND_ALPHA = 0.18
LOG_BAND_FLOOR = 0.5        # on log axes a band that would cross zero is cut at this fraction of the centre

SHOW_REPEAT_TRACES = true   # thin line per repeat behind the curves of the single-χ plots
TRACE_ALPHA = 0.25
TRACE_LW = 0.6
SHOW_BANDS_IN_COMBINED = true   # bands on the combined_* overlays (best μ per χ)
SHOW_BANDS_IN_ALL_SETUPS = false  # bands on the *_all_setups overlays
FILL_BETWEEN_ITERS = true   # shade between baseline and FAI in step_iterations

ANNOTATE_HEATMAPS = true
ANNOTATE_HEATMAP_SPREAD = true    # add the spread under the value in each heatmap cell
HEATMAP_DIGITS = 2

PARETO_ERRORBARS = true      # TIME spread on both axes
PARETO_BASELINE_BAND = true

DIST_JITTER = 0.35        # horizontal spread of the per-repeat points
DIST_POINT_SIZE = 4
DIST_POINT_ALPHA = 0.6
DIST_SUMMARY_OFFSET = 0.3         # the centre ± spread marker sits this far right of the points
DIST_XROTATION = 45
DIST_BOTTOM_MARGIN = 12Plots.mm   # room for the rotated configuration labels

PARETO_PAD = 0.03                 # axes start this fraction of the range below 0 so markers at 0 stay whole

# Log axes spanning fewer than LOG_TICKS_NICE_MAX_DECADES decades get plain 1-2-5 style ticks
# (0.5, 1, 2, ...) instead of fractional powers of ten (10^0.25, 10^0.5, ...)
LOG_TICKS_NICE = true
LOG_TICKS_NICE_MAX_DECADES = 2.0

# -------------------------------------------------------------------------
# Plot families (true / false)
# -------------------------------------------------------------------------
PLOT_SPEEDUP_VS_CHI = false   # speedup_vs_chi_*, speedup_solve_vs_chi_*, speedup_total_vs_chi_*
PLOT_HEATMAPS = false   # heatmap_solve_*, heatmap_total_* (μ × χ_max)
PLOT_PARETO = false   # pareto_*
PLOT_PER_CHI = false   # <size>/chi_<χ>/step_iterations, step_speedup, step_cumulative_time, site_iterations
PLOT_COMBINED = false   # <size>/combined_*
PLOT_DISTRIBUTIONS = false   # dist_* (every repeat of every configuration)
PLOT_SCALING = false   # scaling_summary_evolution

# Axis scales: :identity or :log10
YSCALE_SPEEDUP_VS_CHI = :identity
YSCALE_STEP_ITERS = :identity
YSCALE_COMBINED_STEP_ITERS = :log10
YSCALE_STEP_SPEEDUP = :identity
YSCALE_CUM_TIME = :identity
YSCALE_SITE_ITERS = :identity
YSCALE_RESIDUALS = :log10
YSCALE_DIV_RATIO = :identity
YSCALE_DIST_TIME = :identity
YSCALE_DIST_SPEEDUP = :identity
YSCALE_SCALING = :identity

# Per-figure axis limits, keyed by the output name without extension: either the file name, applied to
# that figure in every folder ("combined_step_speedup", "step_speedup", "pareto_8x8"), or the path
# relative to PLOTS_DIR for a single figure ("8x8/chi_4/step_speedup"). Value: (min, max).
XLIMS = Dict{String,Tuple{Float64,Float64}}(
# "combined_step_iterations" => (1, 100),
)
YLIMS = Dict{String,Tuple{Float64,Float64}}(
# "combined_step_speedup" => (0.5, 3.0),
# "step_speedup"          => (0.5, 3.0),
)

# Legend Position Toggles (e.g. :topleft, :topright, :bottomleft, :bottomright, :outertopright, :none)
LEGEND_POS_PARETO = :outertopright
LEGEND_POS_STEP_ITERS = :topright
LEGEND_POS_STEP_SPEEDUP = :topright
LEGEND_POS_CUM_TIME = :topleft
LEGEND_POS_COMBINED_STEP_ITERS = :topright
LEGEND_POS_COMBINED_STEP_SPEEDUP = :topright
LEGEND_POS_COMBINED_CUM_TIME = :topleft
LEGEND_POS_SITE_ITERS = :topright
LEGEND_POS_COMBINED_SITE_ITERS = :topright
LEGEND_POS_SITE_ITERS_ALL = :outertopright
LEGEND_POS_RESIDUALS_ALL = :outertopright
LEGEND_POS_DIV_RATIO_ALL = :outertopright
LEGEND_POS_SCALING = :topleft
LEGEND_POS_SPEEDUP_VS_CHI = :topright
LEGEND_POS_DIST = :none

# Color Palettes
COLOR_BASELINE = :black   # baseline in combined / all-setups plots
COLOR_BASELINE_PER_CHI = :red     # baseline in the single-χ plots
COLOR_BREAK_EVEN = :red
COLOR_BEST = :blue    # FAI in the single-χ plots
COLOR_SOLVE = :blue
COLOR_TOTAL = :green
COLOR_STEP_SPEEDUP = :purple
CHI_COLOR_SCHEME = :tab10   # palette (or gradient if CHI_COLOR_SEQUENTIAL) giving one color per χ_max
CHI_COLOR_SEQUENTIAL = false    # true: sample CHI_COLOR_SCHEME as a gradient, e.g. :viridis or :Blues
CHI_GRADIENT_RANGE = (0.15, 0.9)
ALL_SETUPS_PALETTE = :tab20   # used by *_all_setups when several μ share a χ_max
HEATMAP_CMAP = :viridis

# -------------------------------------------------------------------------
# Paper figure pairs (plots_paper/pairs/), as in ../tt_anisotropy_stats/plot_paper.jl: legend-free panels
# drawn at their printed size, one per grid, placed two per row as LaTeX subfigures (PAIR_PANEL_FRACTION of
# \textwidth, included with width=\textwidth), plus one shared legend strip (legend_strip.pdf) that goes
# under every pair at full \textwidth. Sizes in pt. Each panel shows the best μ of every χ_max.
# -------------------------------------------------------------------------
PLOT_PAIRS = true
# Pair panels drawn (the paper uses the first two): :step_iterations, :step_cumulative_total_speedup,
# :step_speedup, :step_cumulative_speedup, :pareto
PAIR_PANELS = [:step_iterations, :step_cumulative_total_speedup]
PAIR_TEXTWIDTH_PT = 390      # \textwidth of the paper (elsarticle preprint, 12pt)
PAIR_PANEL_FRACTION = 0.49     # subfigure width as a fraction of \textwidth
PAIR_ASPECT = 0.8      # panel height / width
PAIR_SHOW_BANDS = false    # spread bands on the per-step pair panels (crowded with every χ_max)
PAIR_PARETO_REGION = true     # shade the net speedup region of the Pareto panels
# Grids whose iterations panel gets a dashed vertical line per χ_max at the step where FAI (incl. build) overtakes
# the baseline (mean cumulative time). Left out at 10x10, where every χ_max overtakes within steps 7-27 of 1000.
PAIR_AMORTIZATION_GRIDS = String[]
PAIR_LEGEND_PARETO_REGION = true  # list the net speedup region in the Pareto legend strip
# Upper y limit of the per-step speedup panels, per grid (e.g. "8x8" => 2.5); medians above it are drawn as
# arrows on the top edge and their maximum is annotated. Grids not listed keep the automatic limits.
PAIR_SPEEDUP_YMAX = Dict{String,Float64}("8x8" => 2.5)
PAIR_SPEEDUP_YMIN = 0.7
# Lower y limit of the (linear) cumulative total speedup panels, per grid; grids not listed are not capped
PAIR_CUM_TOTAL_SPEEDUP_YMIN = Dict{String,Float64}("10x10" => 0.9)
# Explicit y ticks of the cumulative total speedup panels, per grid; grids not listed use the default ticks
PAIR_CUM_TOTAL_SPEEDUP_YTICKS = Dict{String,Vector{Float64}}("10x10" => [1.0, 1.2, 1.4])
PAIR_ANNOTATION_FONT_SIZE = 7
PAIR_XLABEL_STEP = "Time Step"
PAIR_YLABEL_ITERS = "Local CG Iterations"
PAIR_YLABEL_SPEEDUP = "Solve Speedup"
PAIR_YLABEL_CUM_SPEEDUP = "Cumulative Solve Speedup"
PAIR_YLABEL_CUM_TOTAL_SPEEDUP = "Cumulative Total Speedup"
PAIR_XLABEL_PARETO = "Build Time (s)"
PAIR_YLABEL_PARETO = "Solve Time (s)"
# GR draws text about 1.5 times its nominal pt size when 1 px = 1 pt (Plots' gr_point_mult), so the pair
# figures are drawn PAIR_CANVAS_SCALE times larger than printed and LaTeX scales them back down: font sizes
# below are then the printed sizes, and line widths / marker sizes are given in printed units too.
PAIR_CANVAS_SCALE = 1.5
PAIR_GUIDE_FONT_SIZE = 8
PAIR_TICK_FONT_SIZE = 7
PAIR_LEGEND_FONT_SIZE = 8
PAIR_LINE_WIDTH = 1.0
PAIR_LINE_WIDTH_THICK = 1.6
# Thinner (lw, lw_thick) on the per-step iterations panel of noisy grids, where step-to-step jitter clutters
PAIR_ITERS_LINE_WIDTH = Dict{String,Tuple{Float64,Float64}}("8x8" => (0.5, 0.8), "10x10" => (0.5, 0.8))
PAIR_MARKER_SIZE = 2.5
PAIR_MARGIN = 1Plots.mm
PAIR_LEGEND_COLUMNS = 4    # entries per legend row (-1 = all in a single row)
PAIR_LEGEND_ROW_HEIGHT = 15   # pt
PAIR_LEGEND_LINE_LENGTH = 0.2  # sample line length, as a fraction of the column width

# =========================================================================

gr()
default(;
    fmt=:png,
    dpi=DPI,
    size=SIZE_STANDARD,
    fontfamily=FONT_FAMILY,
    margin=MARGIN_MM,
    tickfontsize=TICK_FONT_SIZE,
    guidefontsize=GUIDE_FONT_SIZE,
    titlefontsize=TITLE_FONT_SIZE,
    legendfontsize=LEGEND_FONT_SIZE,
)

ENV["GKSwstype"] = "100"
mkpath(PLOTS_DIR)

const PAIR_PANEL_W = PAIR_CANVAS_SCALE * PAIR_PANEL_FRACTION * PAIR_TEXTWIDTH_PT
const PAIR_PANEL_KW = (;
    legend=false,
    size=round.(Int, (PAIR_PANEL_W, PAIR_ASPECT * PAIR_PANEL_W)),
    margin=PAIR_CANVAS_SCALE * PAIR_MARGIN,
    framestyle=:box,
    guidefontsize=PAIR_GUIDE_FONT_SIZE,
    tickfontsize=PAIR_TICK_FONT_SIZE,
)

"""
Plain-number ticks on a log y axis that spans only a few decades (single-panel plots).
"""
function nice_log_yticks!(p)
    length(p.subplots) == 1 || return p
    sp = p[1]
    islog(sp[:yaxis][:scale]) || return p
    lo, hi = Plots.ylims(sp)
    (lo > 0 && hi > lo && log10(hi / lo) < LOG_TICKS_NICE_MAX_DECADES) || return p
    ticks = Float64[]
    for mantissas in ((1, 2, 5), (1, 1.5, 2, 3, 4, 5, 6, 8), 1:0.1:9.9)
        ticks = [
            m * 10.0^e for e in floor(Int, log10(lo)):ceil(Int, log10(hi)) for m in mantissas if lo <= m * 10.0^e <= hi
        ]
        length(ticks) >= 3 && break
    end
    length(ticks) >= 2 && plot!(p; yticks=(ticks, [@sprintf("%g", t) for t in ticks]))
    return p
end

# Integer ticks on the MPS site axis (otherwise Plots picks decimals such as 2.5, 5.0, ...). Applied to every
# plot whose x label is SITE_AXIS_LABEL; dense chains get a tick every 2, 5 or 10 sites.
const SITE_AXIS_LABEL = "MPS Site Index"
function integer_site_ticks!(p)
    string(p.subplots[1][:xaxis][:guide]) == SITE_AXIS_LABEL || return p
    xs = Float64[]
    for s in p.series_list
        (s[:seriestype] in (:hline, :vline) || s[:x] === nothing) && continue
        append!(xs, filter(isfinite, Float64.(s[:x])))
    end
    isempty(xs) && return p
    lo, hi = ceil(Int, minimum(xs)), floor(Int, maximum(xs))
    n = hi - lo + 1
    step = if n <= 12
        1
    elseif n <= 24
        2
    elseif n <= 60
        5
    else
        10
    end
    ticks = step <= 2 ? collect(lo:step:hi) : unique([lo; (step*cld(lo, step)):step:hi])
    plot!(p; xticks=(ticks, string.(ticks)))
    return p
end

# Helper function to save in all configured publication formats (applying any XLIMS / YLIMS override)
function save_paper_plot(p, base_path_no_ext)
    mkpath(dirname(base_path_no_ext))
    rel, name = relpath(base_path_no_ext, PLOTS_DIR), basename(base_path_no_ext)
    lims_for(d) = get(d, rel, get(d, name, nothing))
    (xl = lims_for(XLIMS)) === nothing || xlims!(p, xl...)
    (yl = lims_for(YLIMS)) === nothing || ylims!(p, yl...)
    LOG_TICKS_NICE && nice_log_yticks!(p)
    integer_site_ticks!(p)
    for fmt in OUTPUT_FORMATS
        savefig(p, "$(base_path_no_ext).$(fmt)")
    end
    @info "Saved: $(base_path_no_ext).($(join(OUTPUT_FORMATS, ", ")))"
end

# =========================================================================
# Statistics helpers
# =========================================================================

const CENTER_NAMES = Dict(:mean => "mean", :median => "median", :gm => "geometric mean")
const SPREAD_NAMES = Dict(
    :sd => "± 1 standard deviation",
    :sem => "± 1 standard error of the mean",
    :ci95 => "95% confidence interval of the mean (normal approximation)",
    :iqr => "interquartile range (25th-75th percentile)",
    :minmax => "min-max range",
    :gsd => "×/÷ 1 geometric standard deviation",
    :none => "no band",
)
describe_stat(center, spread) = "centre = $(CENTER_NAMES[center]), band = $(SPREAD_NAMES[spread])"

"""
(centre, lower, upper) of the finite samples in `x`; NaNs when nothing is left.
"""
function center_band(x; center=:mean, spread=:sd)
    v = Float64[Float64(y) for y in x if !ismissing(y) && isfinite(y)]
    (center === :gm || spread === :gsd) && filter!(>(0), v)
    isempty(v) && return (NaN, NaN, NaN)
    n = length(v)
    c = if center === :mean
        mean(v)
    elseif center === :median
        median(v)
    elseif center === :gm
        exp(mean(log.(v)))
    else
        error("Unknown centre statistic: $center")
    end
    sd = n > 1 ? std(v) : 0.0
    lo, hi = if spread === :sd
        (c - sd, c + sd)
    elseif spread === :sem
        (c - sd / sqrt(n), c + sd / sqrt(n))
    elseif spread === :ci95
        (c - 1.959964 * sd / sqrt(n), c + 1.959964 * sd / sqrt(n))
    elseif spread === :iqr
        (quantile(v, 0.25), quantile(v, 0.75))
    elseif spread === :minmax
        extrema(v)
    elseif spread === :gsd
        (g=n > 1 ? exp(std(log.(v))) : 1.0; (c / g, c * g))
    elseif spread === :none
        (c, c)
    else
        error("Unknown spread statistic: $spread")
    end
    return (c, lo, hi)
end

# Vector of (c, lo, hi) tuples -> (c, lo, hi) vectors
unz(v) = (Float64[t[1] for t in v], Float64[t[2] for t in v], Float64[t[3] for t in v])

"""
Row-wise statistics of a (steps or sites) × repeats matrix.
"""
function rowwise(M; center, spread)
    st = [center_band(view(M, i, :); center, spread) for i in axes(M, 1)]
    return (c=Float64[s[1] for s in st], lo=Float64[s[2] for s in st], hi=Float64[s[3] for s in st])
end

"""
Per-repeat vectors of `col` as the columns of a matrix (NaN-padded to the longest repeat).
"""
function repeat_matrix(sub, col)
    hasproperty(sub, col) || return zeros(0, 0)
    vs = [Float64.(v) for v in sub[!, col] if !ismissing(v) && !isnothing(v)]
    isempty(vs) && return zeros(0, 0)
    M = fill(NaN, maximum(length, vs), length(vs))
    for (j, v) in enumerate(vs)
        M[1:length(v), j] = v
    end
    return M
end

"""
Cumulative solve time per step plus the preconditioner build time, one column per repeat.
"""
function cumtime_matrix(sub)
    M = repeat_matrix(sub, :solve_times)
    isempty(M) && return M
    return cumsum(M; dims=1) .+ reshape(Float64.(sub.precond_build_time), 1, :)
end

"""
Rows of `base` and `cfg` with a common repeat index, in the same order.
"""
function pair_rows(base, cfg)
    common = sort(intersect(base.repeat, cfg.repeat))
    return base[[findfirst(==(r), base.repeat) for r in common], :],
    cfg[[findfirst(==(r), cfg.repeat) for r in common], :]
end

"""
Speedup samples base[num_col] / cfg[den_col] (paired by repeat, or over all pairs).
"""
function ratio_samples(base, cfg, num_col, den_col)
    if PAIRED_SPEEDUPS
        b, c = pair_rows(base, cfg)
        return Float64.(b[!, num_col]) ./ Float64.(c[!, den_col])
    end
    return vec([Float64(x) / Float64(y) for x in base[!, num_col], y in cfg[!, den_col]])
end

"""
Per-step ratio base[num_col] ./ cfg[den_col], one column per (paired) repeat.
"""
function ratio_matrix(base, cfg, num_col, den_col)
    pairs = if PAIRED_SPEEDUPS
        b, c = pair_rows(base, cfg)
        collect(zip(b[!, num_col], c[!, den_col]))
    else
        [(x, y) for x in base[!, num_col] for y in cfg[!, den_col]]
    end
    isempty(pairs) && return zeros(0, 0)
    L = minimum(min(length(x), length(y)) for (x, y) in pairs)
    M = fill(NaN, L, length(pairs))
    for (j, (x, y)) in enumerate(pairs)
        M[:, j] = Float64.(x[1:L]) ./ Float64.(y[1:L])
    end
    return M
end

"""
Running speedup cumsum(base[col]) ./ cumsum(cfg[col]) at every step, one column per (paired) repeat.
With `include_build`, each cfg repeat's precond_build_time is added to its running time.
"""
function cumulative_ratio_matrix(base, cfg, col; include_build=false)
    build(c) = include_build ? Float64.(c.precond_build_time) : zeros(nrow(c))
    pairs = if PAIRED_SPEEDUPS
        b, c = pair_rows(base, cfg)
        collect(zip(b[!, col], c[!, col], build(c)))
    else
        [(x, y, t) for x in base[!, col] for (y, t) in zip(cfg[!, col], build(cfg))]
    end
    isempty(pairs) && return zeros(0, 0)
    L = minimum(min(length(x), length(y)) for (x, y, _) in pairs)
    M = fill(NaN, L, length(pairs))
    for (j, (x, y, t)) in enumerate(pairs)
        M[:, j] = cumsum(Float64.(x[1:L])) ./ (cumsum(Float64.(y[1:L])) .+ t)
    end
    return M
end

"""
(step indices, rows) of a per-step matrix after applying STEP_RANGE.
"""
function steps_view(M)
    idx = STEP_RANGE === nothing ? collect(axes(M, 1)) : [i for i in STEP_RANGE if i in axes(M, 1)]
    return idx, M[idx, :]
end
show_step(s) = s !== nothing && (STEP_RANGE === nothing || s in STEP_RANGE)

# =========================================================================
# Plot helpers
# =========================================================================

islog(scale) = !(scale in (:identity, :linear, :none))

"""
Central line with its spread drawn as a ribbon, as error bars, or not at all.
"""
function stat_series!(p, x, c, lo, hi; style=SPREAD_STYLE, logy=false, kw...)
    c, lo, hi = Float64.(c), Float64.(lo), Float64.(hi)
    logy && (lo = [l > 0 ? l : cc * LOG_BAND_FLOOR for (l, cc) in zip(lo, c)])
    dlo = replace(c .- lo, NaN => 0.0)
    dhi = replace(hi .- c, NaN => 0.0)
    if style === :ribbon
        plot!(p, x, c; ribbon=(dlo, dhi), fillalpha=BAND_ALPHA, kw...)
    elseif style === :errorbar
        plot!(p, x, c; yerror=(dlo, dhi), merge((; markerstrokecolor=get(kw, :color, :black)), kw)...)
    else
        plot!(p, x, c; kw...)
    end
    return p
end

"""
One thin line per repeat (column of M).
"""
function traces!(p, x, M; color)
    for j in axes(M, 2)
        plot!(p, x, M[:, j]; color, alpha=TRACE_ALPHA, lw=TRACE_LW, label="")
    end
    return p
end

function fmt_num(x)
    isfinite(x) || return ""
    return Printf.format(Printf.Format("%.$(HEATMAP_DIGITS)f"), x)
end

function spread_text(c, lo, hi, spread)
    (!ANNOTATE_HEATMAP_SPREAD || spread === :none || !isfinite(lo)) && return fmt_num(c)
    spread === :gsd && return "$(fmt_num(c))\n×/÷$(fmt_num(hi / c))"
    return "$(fmt_num(c))\n[$(fmt_num(lo)), $(fmt_num(hi))]"
end

"""
Heatmap of the centres Z (rows = y labels), cells annotated with value and spread.
"""
function stat_heatmap(xlabels, ylabels, Z, Zlo, Zhi; xlabel, ylabel, spread)
    nx, ny = length(xlabels), length(ylabels)
    valid = filter(isfinite, Z)
    clim = isempty(valid) ? :auto : (minimum(valid), maximum(valid) + (maximum(valid) == minimum(valid) ? 1e-9 : 0.0))
    p = heatmap(
        1:nx,
        1:ny,
        Z;
        xticks=(1:nx, xlabels),
        yticks=(1:ny, ylabels),
        xlabel,
        ylabel,
        color=HEATMAP_CMAP,
        clim,
        size=SIZE_HEATMAP,
        right_margin=8Plots.mm,
        framestyle=:box,
    )
    if ANNOTATE_HEATMAPS && !isempty(valid)
        zmin, zmax = extrema(valid)
        for i in 1:nx, j in 1:ny
            isfinite(Z[j, i]) || continue
            t = zmax > zmin ? (Z[j, i] - zmin) / (zmax - zmin) : 0.0
            annotate!(
                p,
                i,
                j,
                text(
                    spread_text(Z[j, i], Zlo[j, i], Zhi[j, i], spread), ANNOTATION_FONT_SIZE, t < 0.6 ? :white : :black
                ),
            )
        end
    end
    return p
end

"""
Every repeat as a jittered point, plus the centre ± spread marker next to it.
"""
function dist_plot(labels, samples, colors; ylabel, center, spread, yscale=:identity, breakeven=false)
    n = length(samples)
    p = plot(;
        ylabel=ylabel,
        xticks=(1:n, labels),
        xrotation=DIST_XROTATION,
        xlims=(0.4, n + 0.4 + DIST_SUMMARY_OFFSET),
        legend=LEGEND_POS_DIST,
        size=SIZE_DIST,
        yscale=yscale,
        bottom_margin=DIST_BOTTOM_MARGIN,
        framestyle=:box,
    )
    breakeven && hline!(p, [1.0]; color=COLOR_BREAK_EVEN, lw=LINE_WIDTH, ls=:dash, label="Break-even (1.0x)")
    for (i, s) in enumerate(samples)
        v = filter(isfinite, Float64.(s))
        islog(yscale) && filter!(>(0), v)
        isempty(v) && continue
        jit = DIST_JITTER .* (mod.((1:length(v)) .* 0.6180339887, 1.0) .- 0.5)
        scatter!(
            p,
            i .+ jit,
            v;
            color=colors[i],
            alpha=DIST_POINT_ALPHA,
            markersize=DIST_POINT_SIZE,
            markerstrokewidth=0,
            label="",
        )
        c, lo, hi = center_band(v; center, spread)
        islog(yscale) && lo <= 0 && (lo = c * LOG_BAND_FLOOR)
        scatter!(
            p,
            [i + DIST_SUMMARY_OFFSET],
            [c];
            yerror=([c - lo], [hi - c]),
            color=:white,
            markerstrokecolor=:black,
            marker=:diamond,
            markersize=DIST_POINT_SIZE + 2,
            label="",
        )
    end
    return p
end

function fmt_mu(μ)
    (isfinite(μ) && μ > 0) || return string(μ)
    e = floor(Int, log10(μ))
    m = μ / 10.0^e
    return isapprox(m, 1; atol=1e-8) ? "10^{$e}" : "$(round(m; digits=2)) \\times 10^{$e}"
end
chi_label(g) = latexstring("FAI \$\\chi_{max}=$(g)\$")

function make_colors(n)
    n == 0 && return []
    if CHI_COLOR_SEQUENTIAL
        g = cgrad(CHI_COLOR_SCHEME)
        zs = n == 1 ? [CHI_GRADIENT_RANGE[2]] : range(CHI_GRADIENT_RANGE[1], CHI_GRADIENT_RANGE[2]; length=n)
        return [g[z] for z in zs]
    end
    pal = palette(CHI_COLOR_SCHEME)
    return [pal[mod1(i, length(pal))] for i in 1:n]
end

# =========================================================================
# Load data
# =========================================================================

isfile(RESULTS_FILE) || error("Could not find $RESULTS_FILE")
@info "Loading per-repeat data from $RESULTS_FILE"
df = copy(load(RESULTS_FILE, "df"))

if DISCARD_FIRST_IN_PROCESS && hasproperty(df, :first_in_process)
    df = df[.!df.first_in_process, :]
end
isempty(EXCLUDED_REPEATS) || (df = df[.!in(EXCLUDED_REPEATS).(df.repeat), :])
isempty(EXCLUDED_KEYS) || (df = df[.!in(EXCLUDED_KEYS).(df.key), :])

keep_chi(g) = SELECTED_CHI === nothing || g in SELECTED_CHI
getval(x, default) = ismissing(x) || isnothing(x) ? default : x

const ALL_CHIS = sort(unique(Int[g for g in skipmissing(df.G_maxdim) if keep_chi(g)]))
const CHI_COLOR = Dict(zip(ALL_CHIS, make_colors(length(ALL_CHIS))))

"""
One entry per configuration of a grid: (key, mu, chi, sub) where sub holds its repeats.
"""
function config_entries(nodes)
    dn = df[df.nodes .== Ref(nodes), :]
    out = []
    for k in unique(dn.key)
        sub = sort(dn[dn.key .== k, :], :repeat)
        r = first(eachrow(sub))
        push!(out, (key=String(k), mu=Float64(getval(r.mu, NaN)), chi=Int(getval(r.G_maxdim, 0)), sub=sub))
    end
    return out
end

# =========================================================================
# Per system size
# =========================================================================

function plot_system_size(nodes)
    sz = "$(nodes[1])x$(nodes[2])"
    entries = config_entries(nodes)
    bi = findfirst(e -> e.key == "baseline", entries)
    if bi === nothing
        @warn "Skipping $sz: missing baseline data"
        return nothing
    end
    base = entries[bi]
    fai = sort([e for e in entries if e.key != "baseline" && keep_chi(e.chi)]; by=e -> (e.chi, e.mu))
    if isempty(fai)
        @warn "No preconditioned runs found for $sz"
        return nothing
    end
    @info "Generating paper plots for $sz ($(length(fai)) FAI setups, baseline n=$(nrow(base.sub)))"

    chis = [e.chi for e in fai]
    mus = [e.mu for e in fai]
    unique_chis = sort(unique(chis))
    unique_mus = sort(unique(mus))
    multi_mu = length(unique_mus) > 1
    setup_label(e) = multi_mu ? latexstring("FAI \$\\chi_{max}=$(e.chi), \\mu=$(fmt_mu(e.mu))\$") : chi_label(e.chi)
    short_label(e) =
        multi_mu ? latexstring("\$\\chi=$(e.chi), \\mu=$(fmt_mu(e.mu))\$") : latexstring("\$\\chi=$(e.chi)\$")
    all_pal = palette(ALL_SETUPS_PALETTE)
    setup_color(e, j) = multi_mu ? all_pal[mod1(j, length(all_pal))] : CHI_COLOR[e.chi]

    # Scalar statistics per FAI setup
    sp_solve_samples = [ratio_samples(base.sub, e.sub, :total_solve_time, :total_solve_time) for e in fai]
    sp_total_samples = [ratio_samples(base.sub, e.sub, :total_solve_time, :total_time) for e in fai]
    sp_solve = [center_band(s; center=SPEEDUP_CENTER, spread=SPEEDUP_SPREAD) for s in sp_solve_samples]
    sp_total = [center_band(s; center=SPEEDUP_CENTER, spread=SPEEDUP_SPREAD) for s in sp_total_samples]
    build = [center_band(e.sub.precond_build_time; center=TIME_CENTER, spread=TIME_SPREAD) for e in fai]
    solve = [center_band(e.sub.total_solve_time; center=TIME_CENTER, spread=TIME_SPREAD) for e in fai]

    # Best μ for each χ_max
    rank = [t[1] for t in (BEST_MU_BY === :total ? sp_total : sp_solve)]
    best = Dict{Int,Int}()
    for g in unique_chis
        cand = [i for i in eachindex(fai) if chis[i] == g && isfinite(rank[i])]
        isempty(cand) || (best[g] = cand[argmax(rank[cand])])
    end
    best_chis = [g for g in unique_chis if haskey(best, g)]
    combined_chis = [g for g in best_chis if COMBINED_CHI === nothing || g in COMBINED_CHI]

    # ---------------------------------------------------------
    # 1. Speedup vs χ_max (best μ per χ_max)
    # ---------------------------------------------------------
    if PLOT_SPEEDUP_VS_CHI && !isempty(best_chis)
        bs = unz([sp_solve[best[g]] for g in best_chis])
        bt = unz([sp_total[best[g]] for g in best_chis])
        function sp_chi_plot(ylabel, series)
            p = plot(;
                xlabel=L"Bond Dimension $\chi_{max}$",
                ylabel=ylabel,
                xticks=(best_chis, string.(best_chis)),
                legend=LEGEND_POS_SPEEDUP_VS_CHI,
                yscale=YSCALE_SPEEDUP_VS_CHI,
                size=SIZE_STANDARD,
                framestyle=:box,
            )
            for (lbl, s, col, mk) in series
                stat_series!(
                    p,
                    best_chis,
                    s...;
                    logy=islog(YSCALE_SPEEDUP_VS_CHI),
                    label=lbl,
                    color=col,
                    lw=LINE_WIDTH_THICK,
                    marker=mk,
                    markersize=MARKER_SIZE + 1,
                )
            end
            hline!(p, [1.0]; label="Break-even (1.0x)", color=COLOR_BREAK_EVEN, lw=LINE_WIDTH, ls=:dash)
            return p
        end
        save_paper_plot(
            sp_chi_plot(
                "Speedup Factor",
                [("Solve Speedup", bs, COLOR_SOLVE, :circle), ("Total Speedup", bt, COLOR_TOTAL, :square)],
            ),
            joinpath(PLOTS_DIR, "speedup_vs_chi_$sz"),
        )
        save_paper_plot(
            sp_chi_plot("Solve Speedup Factor", [("Solve Speedup", bs, COLOR_SOLVE, :circle)]),
            joinpath(PLOTS_DIR, "speedup_solve_vs_chi_$sz"),
        )
        save_paper_plot(
            sp_chi_plot("Total Speedup Factor", [("Total Speedup", bt, COLOR_TOTAL, :square)]),
            joinpath(PLOTS_DIR, "speedup_total_vs_chi_$sz"),
        )
    end

    # ---------------------------------------------------------
    # 2. Heatmaps (μ × χ_max) and Pareto front
    # ---------------------------------------------------------
    if PLOT_HEATMAPS
        function to_grid(v, k)
            Z = fill(NaN, length(unique_chis), length(unique_mus))
            for i in eachindex(fai)
                Z[findfirst(==(chis[i]), unique_chis), findfirst(==(mus[i]), unique_mus)] = v[i][k]
            end
            return Z
        end
        xl = [latexstring("\$" * fmt_mu(μ) * "\$") for μ in unique_mus]
        yl = string.(unique_chis)
        for (name, v) in (("heatmap_solve", sp_solve), ("heatmap_total", sp_total))
            hm = stat_heatmap(
                xl,
                yl,
                to_grid(v, 1),
                to_grid(v, 2),
                to_grid(v, 3);
                xlabel=L"Regularization parameter $\mu$",
                ylabel=L"Bond Dimension $\chi_{max}$",
                spread=SPEEDUP_SPREAD,
            )
            save_paper_plot(hm, joinpath(PLOTS_DIR, "$(name)_$sz"))
        end
    end

    if PLOT_PARETO
        bx, sx = unz(build), unz(solve)
        bc, blo, bhi = center_band(base.sub.total_solve_time; center=TIME_CENTER, spread=TIME_SPREAD)
        pf = plot(;
            xlabel="Preconditioner Build Time (s)",
            ylabel="Aggregate Linear Solve Time (s)",
            legend=LEGEND_POS_PARETO,
            size=SIZE_PARETO,
            framestyle=:box,
        )
        plot!(
            pf, [0, bc], [bc, 0]; fillrange=0, fillalpha=0.1, fillcolor=:green, linealpha=0, label="Net Speedup Region"
        )
        if PARETO_BASELINE_BAND && isfinite(blo) && bhi > blo
            hspan!(pf, [blo, bhi]; color=COLOR_BREAK_EVEN, alpha=BAND_ALPHA, linealpha=0, label="")
        end
        hline!(pf, [bc]; label="Baseline Total Solve", color=COLOR_BREAK_EVEN, lw=LINE_WIDTH, ls=:dash)
        for g in unique_chis
            idx = findall(==(g), chis)
            err = if PARETO_ERRORBARS
                (;
                    xerror=(bx[1][idx] .- bx[2][idx], bx[3][idx] .- bx[1][idx]),
                    yerror=(sx[1][idx] .- sx[2][idx], sx[3][idx] .- sx[1][idx]),
                )
            else
                (;)
            end
            scatter!(
                pf,
                bx[1][idx],
                sx[1][idx];
                err...,
                label=chi_label(g),
                color=CHI_COLOR[g],
                markerstrokecolor=CHI_COLOR[g],
                marker=:circle,
                markersize=MARKER_SIZE,
            )
        end
        xmax = maximum(filter(isfinite, PARETO_ERRORBARS ? bx[3] : bx[1]); init=0.0)
        ymax = maximum(filter(isfinite, PARETO_ERRORBARS ? sx[3] : sx[1]); init=0.0)
        xlims!(pf, -PARETO_PAD * max(xmax, 1e-12), max(xmax, 1e-12) * (1.0 + PARETO_PAD))
        ylims!(pf, -PARETO_PAD * max(ymax, bhi, bc), max(ymax, bhi, bc) * (1.0 + PARETO_PAD))
        save_paper_plot(pf, joinpath(PLOTS_DIR, "pareto_$sz"))
    end

    # ---------------------------------------------------------
    # 3. Per-step and per-site profiles (single χ_max) and combined overlays
    # ---------------------------------------------------------
    stat_kw = (; center=PROFILE_CENTER, spread=PROFILE_SPREAD)
    time_kw = (; center=TIME_CENTER, spread=TIME_SPREAD)
    step_mk(m) = STEP_MARKERS ? m : :none
    site_mk(m) = SITE_MARKERS ? m : :none

    xb, Mb_iters = steps_view(repeat_matrix(base.sub, :iters_per_step))
    rb_iters = rowwise(Mb_iters; stat_kw...)
    Cb_full = cumtime_matrix(base.sub)
    cum_b_full = rowwise(Cb_full; time_kw...)
    xcb, Cb = steps_view(Cb_full)
    rcb = rowwise(Cb; time_kw...)
    Mb_site = repeat_matrix(base.sub, :iters_per_site)
    rb_site = rowwise(Mb_site; stat_kw...)
    have_sites = !isempty(Mb_site)
    sites_b = 1:size(Mb_site, 1)

    function amort_step(e)
        cv = rowwise(cumtime_matrix(e.sub); time_kw...)
        L = min(length(cv.c), length(cum_b_full.c))
        return findfirst(i -> cv.c[i] <= cum_b_full.c[i], 1:L)
    end

    combined_band = SHOW_BANDS_IN_COMBINED ? PROFILE_STYLE : :none
    if PLOT_COMBINED
        p_iters_c = plot(;
            xlabel="Chorin Time Step",
            ylabel="Total Krylov Iterations",
            yscale=YSCALE_COMBINED_STEP_ITERS,
            legend=LEGEND_POS_COMBINED_STEP_ITERS,
            framestyle=:box,
        )
        p_sp_c = plot(;
            xlabel="Chorin Time Step",
            ylabel="Speedup Ratio",
            yscale=YSCALE_STEP_SPEEDUP,
            legend=LEGEND_POS_COMBINED_STEP_SPEEDUP,
            framestyle=:box,
        )
        p_cum_c = plot(;
            xlabel="Chorin Time Step",
            ylabel="Cumulative Time (s)",
            yscale=YSCALE_CUM_TIME,
            legend=LEGEND_POS_COMBINED_CUM_TIME,
            framestyle=:box,
        )
        p_site_c = plot(;
            xlabel="MPS Site Index",
            ylabel="Total Krylov Iterations",
            yscale=YSCALE_SITE_ITERS,
            legend=LEGEND_POS_COMBINED_SITE_ITERS,
            framestyle=:box,
        )

        stat_series!(
            p_iters_c,
            xb,
            rb_iters...;
            style=combined_band,
            logy=islog(YSCALE_COMBINED_STEP_ITERS),
            label="Baseline",
            color=COLOR_BASELINE,
            lw=LINE_WIDTH_THICK,
            marker=step_mk(:circle),
            markersize=MARKER_SIZE,
        )
        hline!(p_sp_c, [1.0]; color=COLOR_BASELINE, lw=LINE_WIDTH_THICK, ls=:dash, label="Break-even (1.0x)")
        stat_series!(
            p_cum_c,
            xcb,
            rcb...;
            style=combined_band,
            logy=islog(YSCALE_CUM_TIME),
            label="Baseline",
            color=COLOR_BASELINE,
            lw=LINE_WIDTH_THICK,
        )
        have_sites && stat_series!(
            p_site_c,
            sites_b,
            rb_site...;
            style=combined_band,
            logy=islog(YSCALE_SITE_ITERS),
            label="Baseline",
            color=COLOR_BASELINE,
            lw=LINE_WIDTH_THICK,
            marker=site_mk(:circle),
            markersize=MARKER_SIZE,
        )
    end

    any_amortized = false
    for g in best_chis
        bidx = best[g]
        e = fai[bidx]
        col = CHI_COLOR[g]
        amort = amort_step(e)
        xv, Mv_iters = steps_view(repeat_matrix(e.sub, :iters_per_step))
        rv_iters = rowwise(Mv_iters; stat_kw...)
        xs, Msp = steps_view(ratio_matrix(base.sub, e.sub, :solve_times, :solve_times))
        rsp = rowwise(Msp; stat_kw...)
        xcv, Cv = steps_view(cumtime_matrix(e.sub))
        rcv = rowwise(Cv; time_kw...)

        if PLOT_PER_CHI
            out_dir = joinpath(PLOTS_DIR, sz, "chi_$g")
            mkpath(out_dir)
            fai_lbl = multi_mu ? latexstring("FAI \$\\mu=$(fmt_mu(e.mu))\$") : "FAI"
            fai_lbl_build = multi_mu ? latexstring("FAI \$\\mu=$(fmt_mu(e.mu))\$ (incl. build)") : "FAI (incl. build)"

            # -- Krylov iterations per step
            p = plot(;
                xlabel="Chorin Time Step",
                ylabel="Total Krylov Iterations",
                yscale=YSCALE_STEP_ITERS,
                legend=LEGEND_POS_STEP_ITERS,
                framestyle=:box,
            )
            if SHOW_REPEAT_TRACES
                traces!(p, xb, Mb_iters; color=COLOR_BASELINE_PER_CHI)
                traces!(p, xv, Mv_iters; color=COLOR_BEST)
            end
            L = min(length(xb), length(xv))
            if FILL_BETWEEN_ITERS && L > 0
                plot!(
                    p,
                    xb[1:L],
                    rb_iters.c[1:L];
                    fillrange=rv_iters.c[1:L],
                    fillalpha=0.12,
                    fillcolor=COLOR_BEST,
                    label="",
                    linealpha=0,
                )
            end
            stat_series!(
                p,
                xb,
                rb_iters...;
                style=PROFILE_STYLE,
                logy=islog(YSCALE_STEP_ITERS),
                label="Baseline",
                color=COLOR_BASELINE_PER_CHI,
                lw=LINE_WIDTH,
                marker=step_mk(:circle),
                markersize=MARKER_SIZE,
            )
            stat_series!(
                p,
                xv,
                rv_iters...;
                style=PROFILE_STYLE,
                logy=islog(YSCALE_STEP_ITERS),
                label=fai_lbl,
                color=COLOR_BEST,
                lw=LINE_WIDTH,
                marker=step_mk(:square),
                markersize=MARKER_SIZE,
            )
            show_step(amort) && vline!(p, [amort]; color=COLOR_BEST, lw=LINE_WIDTH, ls=:dash, label="Amortization step")
            save_paper_plot(p, joinpath(out_dir, "step_iterations"))

            # -- per-step solve speedup (paired per repeat)
            p = plot(;
                xlabel="Chorin Time Step",
                ylabel="Speedup Ratio",
                yscale=YSCALE_STEP_SPEEDUP,
                legend=LEGEND_POS_STEP_SPEEDUP,
                framestyle=:box,
            )
            SHOW_REPEAT_TRACES && traces!(p, xs, Msp; color=COLOR_STEP_SPEEDUP)
            stat_series!(
                p,
                xs,
                rsp...;
                style=PROFILE_STYLE,
                logy=islog(YSCALE_STEP_SPEEDUP),
                label=fai_lbl,
                color=COLOR_STEP_SPEEDUP,
                lw=LINE_WIDTH,
                marker=step_mk(:diamond),
                markersize=MARKER_SIZE,
            )
            hline!(p, [1.0]; color=COLOR_BREAK_EVEN, lw=LINE_WIDTH, ls=:dash, label="Break-even (1.0x)")
            save_paper_plot(p, joinpath(out_dir, "step_speedup"))

            # -- cumulative time (solve + build): where FAI pays off
            p = plot(;
                xlabel="Chorin Time Step",
                ylabel="Cumulative Time (s)",
                yscale=YSCALE_CUM_TIME,
                legend=LEGEND_POS_CUM_TIME,
                framestyle=:box,
            )
            if SHOW_REPEAT_TRACES
                traces!(p, xcb, Cb; color=COLOR_BASELINE_PER_CHI)
                traces!(p, xcv, Cv; color=COLOR_BEST)
            end
            stat_series!(
                p,
                xcb,
                rcb...;
                style=PROFILE_STYLE,
                logy=islog(YSCALE_CUM_TIME),
                label="Baseline",
                color=COLOR_BASELINE_PER_CHI,
                lw=LINE_WIDTH,
            )
            stat_series!(
                p,
                xcv,
                rcv...;
                style=PROFILE_STYLE,
                logy=islog(YSCALE_CUM_TIME),
                label=fai_lbl_build,
                color=COLOR_BEST,
                lw=LINE_WIDTH,
            )
            show_step(amort) && vline!(p, [amort]; color=COLOR_BEST, lw=LINE_WIDTH, ls=:dash, label="Amortization step")
            save_paper_plot(p, joinpath(out_dir, "step_cumulative_time"))

            # -- site-by-site iterations, every μ for this χ_max
            if have_sites
                p = plot(;
                    xlabel="MPS Site Index",
                    ylabel="Total Krylov Iterations",
                    yscale=YSCALE_SITE_ITERS,
                    legend=LEGEND_POS_SITE_ITERS,
                    framestyle=:box,
                )
                stat_series!(
                    p,
                    sites_b,
                    rb_site...;
                    style=PROFILE_STYLE,
                    logy=islog(YSCALE_SITE_ITERS),
                    label="Baseline",
                    color=COLOR_BASELINE,
                    lw=LINE_WIDTH_THICK,
                    marker=site_mk(:circle),
                    markersize=MARKER_SIZE,
                )
                sub_colors = palette(:tab10)
                g_idx = sort(findall(==(g), chis); by=i -> mus[i])
                for (j, i) in enumerate(g_idx)
                    M = repeat_matrix(fai[i].sub, :iters_per_site)
                    isempty(M) && continue
                    is_best = i == bidx
                    lbl = if length(g_idx) == 1
                        "FAI"
                    else
                        latexstring("FAI \$\\mu=$(fmt_mu(mus[i]))\$" * (is_best ? " (best)" : ""))
                    end
                    stat_series!(
                        p,
                        1:size(M, 1),
                        rowwise(M; stat_kw...)...;
                        style=PROFILE_STYLE,
                        logy=islog(YSCALE_SITE_ITERS),
                        label=lbl,
                        color=length(g_idx) == 1 ? COLOR_BEST : sub_colors[mod1(j, length(sub_colors))],
                        lw=is_best ? LINE_WIDTH_THICK : LINE_WIDTH,
                        ls=is_best ? :solid : :dash,
                        marker=site_mk(is_best ? :square : :diamond),
                        markersize=MARKER_SIZE,
                    )
                end
                save_paper_plot(p, joinpath(out_dir, "site_iterations"))
            end
        end

        if PLOT_COMBINED && g in combined_chis
            lbl = setup_label(e)
            stat_series!(
                p_iters_c,
                xv,
                rv_iters...;
                style=combined_band,
                logy=islog(YSCALE_COMBINED_STEP_ITERS),
                label=lbl,
                color=col,
                lw=LINE_WIDTH,
                marker=step_mk(:square),
                markersize=MARKER_SIZE,
            )
            if show_step(amort)
                vline!(p_iters_c, [amort]; color=col, lw=LINE_WIDTH, ls=:dash, label="")
                vline!(p_cum_c, [amort]; color=col, lw=LINE_WIDTH_THIN, ls=:dash, label="")
                any_amortized = true
            end
            stat_series!(
                p_sp_c,
                xs,
                rsp...;
                style=combined_band,
                logy=islog(YSCALE_STEP_SPEEDUP),
                label=lbl,
                color=col,
                lw=LINE_WIDTH,
                marker=step_mk(:diamond),
                markersize=MARKER_SIZE,
            )
            stat_series!(
                p_cum_c,
                xcv,
                rcv...;
                style=combined_band,
                logy=islog(YSCALE_CUM_TIME),
                label=lbl,
                color=col,
                lw=LINE_WIDTH,
            )
            M = repeat_matrix(e.sub, :iters_per_site)
            if have_sites && !isempty(M)
                stat_series!(
                    p_site_c,
                    1:size(M, 1),
                    rowwise(M; stat_kw...)...;
                    style=combined_band,
                    logy=islog(YSCALE_SITE_ITERS),
                    label=lbl,
                    color=col,
                    lw=LINE_WIDTH,
                    marker=site_mk(:square),
                    markersize=MARKER_SIZE,
                )
            end
        end
    end

    if PLOT_COMBINED
        out_dir_c = joinpath(PLOTS_DIR, sz)
        mkpath(out_dir_c)
        if any_amortized
            for pp in (p_iters_c, p_cum_c)
                plot!(pp, [NaN], [NaN]; color=:black, lw=LINE_WIDTH, ls=:dash, label="Amortization step")
            end
        end
        save_paper_plot(p_iters_c, joinpath(out_dir_c, "combined_step_iterations"))
        save_paper_plot(p_sp_c, joinpath(out_dir_c, "combined_step_speedup"))
        save_paper_plot(p_cum_c, joinpath(out_dir_c, "combined_step_cumulative_time"))
        have_sites && save_paper_plot(p_site_c, joinpath(out_dir_c, "combined_site_iterations"))

        # -- every setup: site iterations (centres), residuals and divergence ratios per step
        all_band = SHOW_BANDS_IN_ALL_SETUPS ? PROFILE_STYLE : :none
        function all_setups_plot(col, ylabel, yscale, legend; per_step=true)
            p = plot(;
                xlabel=per_step ? "Chorin Time Step" : "MPS Site Index",
                ylabel=ylabel,
                yscale=yscale,
                legend=legend,
                size=SIZE_COMBINED_SITE,
                framestyle=:box,
            )
            view_(M) = per_step ? steps_view(M) : (collect(axes(M, 1)), M)
            x, M = view_(repeat_matrix(base.sub, col))
            isempty(M) && return nothing
            stat_series!(
                p,
                x,
                rowwise(M; stat_kw...)...;
                style=all_band,
                logy=islog(yscale),
                label="Baseline",
                color=COLOR_BASELINE,
                lw=LINE_WIDTH_THICK,
                marker=per_step ? :none : site_mk(:circle),
                markersize=MARKER_SIZE,
            )
            for (j, e) in enumerate(fai)
                x, M = view_(repeat_matrix(e.sub, col))
                isempty(M) && continue
                stat_series!(
                    p,
                    x,
                    rowwise(M; stat_kw...)...;
                    style=all_band,
                    logy=islog(yscale),
                    label=short_label(e),
                    color=setup_color(e, j),
                    lw=LINE_WIDTH_THIN,
                )
            end
            return p
        end
        if have_sites
            p = all_setups_plot(
                :iters_per_site, "Total Krylov Iterations", YSCALE_SITE_ITERS, LEGEND_POS_SITE_ITERS_ALL; per_step=false
            )
            isnothing(p) || save_paper_plot(p, joinpath(out_dir_c, "combined_site_iterations_all_setups"))
        end
        p = all_setups_plot(:residuals, "Relative Residual", YSCALE_RESIDUALS, LEGEND_POS_RESIDUALS_ALL)
        isnothing(p) || save_paper_plot(p, joinpath(out_dir_c, "combined_step_residuals_all_setups"))
        p = all_setups_plot(
            :div_ratios, "Divergence Norm Ratio (after / before)", YSCALE_DIV_RATIO, LEGEND_POS_DIV_RATIO_ALL
        )
        isnothing(p) || save_paper_plot(p, joinpath(out_dir_c, "combined_step_div_ratio_all_setups"))
    end

    # ---------------------------------------------------------
    # 4. Paper figure pairs: legend-free panels of this grid, the other half of each pair is the other grid
    # ---------------------------------------------------------
    if PLOT_PAIRS && !isempty(best_chis)
        pair_dir = joinpath(PLOTS_DIR, "pairs")
        band = PAIR_SHOW_BANDS ? PROFILE_STYLE : :none
        lw, lw_thick, ms = PAIR_CANVAS_SCALE .* (PAIR_LINE_WIDTH, PAIR_LINE_WIDTH_THICK, PAIR_MARKER_SIZE)
        be_kw = (; color=COLOR_BASELINE, lw, ls=:dash)   # break-even / baseline reference, as in legend_strip

        if :step_iterations in PAIR_PANELS
            # -- Krylov iterations per step, baseline and every χ_max
            it_lw, it_lw_thick =
                PAIR_CANVAS_SCALE .* get(PAIR_ITERS_LINE_WIDTH, sz, (PAIR_LINE_WIDTH, PAIR_LINE_WIDTH_THICK))
            p = plot(; xlabel=PAIR_XLABEL_STEP, ylabel=PAIR_YLABEL_ITERS, yscale=:log10, PAIR_PANEL_KW...)
            stat_series!(p, xb, rb_iters...; style=band, logy=true, color=COLOR_BASELINE, lw=it_lw_thick)
            for g in best_chis
                x, M = steps_view(repeat_matrix(fai[best[g]].sub, :iters_per_step))
                isempty(M) ||
                    stat_series!(p, x, rowwise(M; stat_kw...)...; style=band, logy=true, color=CHI_COLOR[g], lw=it_lw)
            end
            if sz in PAIR_AMORTIZATION_GRIDS
                for g in best_chis
                    amort = amort_step(fai[best[g]])
                    show_step(amort) && vline!(p, [amort]; color=CHI_COLOR[g], lw, ls=:dash)
                end
            end
            save_paper_plot(p, joinpath(pair_dir, "step_iterations_$sz"))
        end

        if :step_speedup in PAIR_PANELS
            # -- per-step solve speedup (paired per repeat), every χ_max
            p = plot(; xlabel=PAIR_XLABEL_STEP, ylabel=PAIR_YLABEL_SPEEDUP, yscale=:log10, PAIR_PANEL_KW...)
            hline!(p, [1.0]; be_kw...)
            ymax = get(PAIR_SPEEDUP_YMAX, sz, nothing)
            clipped = Float64[]
            for g in best_chis
                x, M = steps_view(ratio_matrix(base.sub, fai[best[g]].sub, :solve_times, :solve_times))
                isempty(M) && continue
                r = rowwise(M; stat_kw...)
                stat_series!(p, x, r...; style=band, logy=true, color=CHI_COLOR[g], lw)
                ymax === nothing && continue
                # Off-scale medians: arrow at the top edge, in the colour of the configuration
                out = findall(>(ymax), r.c)
                isempty(out) && continue
                append!(clipped, r.c[out])
                scatter!(
                    p,
                    x[out],
                    fill(ymax / 1.04, length(out));
                    marker=:utriangle,
                    color=CHI_COLOR[g],
                    markerstrokecolor=CHI_COLOR[g],
                    markersize=ms,
                )
            end
            if ymax !== nothing
                ylims!(p, PAIR_SPEEDUP_YMIN, ymax)
                isempty(clipped) || annotate!(
                    p,
                    (0.06, 0.93),
                    text(
                        @sprintf("off-scale, up to %.1f×", maximum(clipped)),
                        PAIR_ANNOTATION_FONT_SIZE,
                        :left,
                        :black,
                        FONT_FAMILY,
                    ),
                )
            end
            save_paper_plot(p, joinpath(pair_dir, "step_speedup_$sz"))
        end

        if :step_cumulative_speedup in PAIR_PANELS
            # -- cumulative solve speedup: total baseline solve time so far / total FAI solve time so far (excl. build)
            p = plot(; xlabel=PAIR_XLABEL_STEP, ylabel=PAIR_YLABEL_CUM_SPEEDUP, yscale=:log10, PAIR_PANEL_KW...)
            hline!(p, [1.0]; be_kw...)
            for g in best_chis
                x, M = steps_view(cumulative_ratio_matrix(base.sub, fai[best[g]].sub, :solve_times))
                isempty(M) ||
                    stat_series!(p, x, rowwise(M; stat_kw...)...; style=band, logy=true, color=CHI_COLOR[g], lw)
            end
            save_paper_plot(p, joinpath(pair_dir, "step_cumulative_speedup_$sz"))
        end

        if :step_cumulative_total_speedup in PAIR_PANELS
            # -- cumulative total speedup: as above, with the FAI build time added to its running solve time
            p = plot(; xlabel=PAIR_XLABEL_STEP, ylabel=PAIR_YLABEL_CUM_TOTAL_SPEEDUP, PAIR_PANEL_KW...)
            hline!(p, [1.0]; be_kw...)
            ytop = 1.0
            for g in best_chis
                x, M = steps_view(cumulative_ratio_matrix(base.sub, fai[best[g]].sub, :solve_times; include_build=true))
                isempty(M) && continue
                r = rowwise(M; stat_kw...)
                stat_series!(p, x, r...; style=band, logy=false, color=CHI_COLOR[g], lw)
                ytop = max(ytop, maximum(filter(isfinite, r.c); init=1.0))
            end
            ymin = get(PAIR_CUM_TOTAL_SPEEDUP_YMIN, sz, nothing)
            ymin === nothing || ylims!(p, ymin, ytop + 0.05 * (ytop - ymin))
            yt = get(PAIR_CUM_TOTAL_SPEEDUP_YTICKS, sz, nothing)
            yt === nothing || plot!(p; yticks=(yt, [@sprintf("%.1f", t) for t in yt]))
            save_paper_plot(p, joinpath(pair_dir, "step_cumulative_total_speedup_$sz"))
        end

        if :pareto in PAIR_PANELS
            # -- Pareto front: build time vs aggregate solve time, baseline solve time as the break-even line
            bc, blo, bhi = center_band(base.sub.total_solve_time; time_kw...)
            p = plot(; xlabel=PAIR_XLABEL_PARETO, ylabel=PAIR_YLABEL_PARETO, PAIR_PANEL_KW...)
            PAIR_PARETO_REGION && plot!(p, [0, bc], [bc, 0]; fillrange=0, fillalpha=0.1, fillcolor=:green, linealpha=0)
            PARETO_BASELINE_BAND &&
                isfinite(blo) &&
                bhi > blo &&
                hspan!(p, [blo, bhi]; color=COLOR_BASELINE, alpha=BAND_ALPHA / 2, linealpha=0)
            hline!(p, [bc]; be_kw...)
            bmax, smax = 0.0, bhi
            for g in best_chis
                e = fai[best[g]]
                b = center_band(e.sub.precond_build_time; time_kw...)
                s = center_band(e.sub.total_solve_time; time_kw...)
                err = if PARETO_ERRORBARS
                    (; xerror=([b[1] - b[2]], [b[3] - b[1]]), yerror=([s[1] - s[2]], [s[3] - s[1]]))
                else
                    (;)
                end
                scatter!(
                    p,
                    [b[1]],
                    [s[1]];
                    err...,
                    color=CHI_COLOR[g],
                    markerstrokecolor=CHI_COLOR[g],
                    marker=:circle,
                    markersize=ms,
                    linewidth=lw,
                )
                bmax = max(bmax, PARETO_ERRORBARS ? b[3] : b[1])
                smax = max(smax, PARETO_ERRORBARS ? s[3] : s[1], bc)
            end
            xlims!(p, -PARETO_PAD * max(bmax, 1e-12), max(bmax, 1e-12) * (1.0 + PARETO_PAD))
            ylims!(p, -PARETO_PAD * smax, smax * (1.0 + PARETO_PAD))
            save_paper_plot(p, joinpath(pair_dir, "pareto_$sz"))
        end
    end

    # ---------------------------------------------------------
    # 5. Distributions: every repeat of every configuration
    # ---------------------------------------------------------
    if PLOT_DISTRIBUTIONS
        all_e = vcat([base], fai)
        labels = ["Baseline"; [short_label(e) for e in fai]]
        colors = [COLOR_BASELINE; [setup_color(e, j) for (j, e) in enumerate(fai)]]
        for (col, ylabel, name) in (
            (:total_solve_time, "Aggregate Solve Time (s)", "dist_solve_time"),
            (:total_time, "Total Time (s)", "dist_total_time"),
            (:total_iters, "Total Krylov Iterations", "dist_iters"),
        )
            hasproperty(base.sub, col) || continue
            p = dist_plot(labels, [e.sub[!, col] for e in all_e], colors; ylabel, yscale=YSCALE_DIST_TIME, time_kw...)
            save_paper_plot(p, joinpath(PLOTS_DIR, "$(name)_$sz"))
        end
        for (samples, ylabel, name) in (
            (sp_solve_samples, "Solve Speedup Factor", "dist_speedup_solve"),
            (sp_total_samples, "Total Speedup Factor", "dist_speedup_total"),
        )
            p = dist_plot(
                labels[2:end],
                samples,
                colors[2:end];
                ylabel,
                yscale=YSCALE_DIST_SPEEDUP,
                breakeven=true,
                center=SPEEDUP_CENTER,
                spread=SPEEDUP_SPREAD,
            )
            save_paper_plot(p, joinpath(PLOTS_DIR, "$(name)_$sz"))
        end
    end

    ok_s = findall(t -> isfinite(t[1]), sp_solve)
    ok_t = findall(t -> isfinite(t[1]), sp_total)
    (isempty(ok_s) || isempty(ok_t)) && return nothing
    return (
        label=sz,
        solve=sp_solve[ok_s[argmax([sp_solve[i][1] for i in ok_s])]],
        total=sp_total[ok_t[argmax([sp_total[i][1] for i in ok_t])]],
        n_min=minimum(nrow(e.sub) for e in vcat([base], fai)),
        n_max=maximum(nrow(e.sub) for e in vcat([base], fai)),
    )
end

system_sizes = sort(unique(df.nodes))
if SELECTED_NODES !== nothing && !isempty(SELECTED_NODES)
    system_sizes = filter(n -> n in SELECTED_NODES, system_sizes)
end
scaling = filter(!isnothing, [plot_system_size(nodes) for nodes in system_sizes])

# ---------------------------------------------------------
# 6. Scaling summary: best configuration per size
# ---------------------------------------------------------
if PLOT_SCALING && length(scaling) > 1
    xs = 1:length(scaling)
    p_scale = plot(;
        xlabel="System Size",
        ylabel="Speedup Factor",
        xticks=(xs, [s.label for s in scaling]),
        legend=LEGEND_POS_SCALING,
        size=SIZE_SCALING,
        yscale=YSCALE_SCALING,
        framestyle=:box,
    )
    stat_series!(
        p_scale,
        xs,
        unz([s.solve for s in scaling])...;
        logy=islog(YSCALE_SCALING),
        label="Max Agg. Solve Speedup",
        color=COLOR_SOLVE,
        lw=LINE_WIDTH_THICK,
        marker=:circle,
        markersize=MARKER_SIZE + 2,
    )
    stat_series!(
        p_scale,
        xs,
        unz([s.total for s in scaling])...;
        logy=islog(YSCALE_SCALING),
        label="Max Agg. Total Speedup",
        color=COLOR_TOTAL,
        lw=LINE_WIDTH_THICK,
        marker=:square,
        markersize=MARKER_SIZE + 2,
    )
    hline!(p_scale, [1.0]; label="Break-even (1.0x)", color=COLOR_BREAK_EVEN, lw=LINE_WIDTH, ls=:dash)
    save_paper_plot(p_scale, joinpath(PLOTS_DIR, "scaling_summary_evolution"))
end

# ---------------------------------------------------------
# 7. Shared legend strip of the paper figure pairs
# ---------------------------------------------------------

"""
Legend-only figure: the entries filled row by row into `columns` equal columns, no axes. Drawn by hand
(sample line + marker + text per cell) because GR's multi-column legend overlaps rows at these sizes.
"""
function legend_strip(entries; columns, width)
    ncol = columns == -1 ? length(entries) : columns
    nrow = cld(length(entries), ncol)
    p = plot(;
        size=round.(Int, PAIR_CANVAS_SCALE .* (width, nrow * PAIR_LEGEND_ROW_HEIGHT)),
        framestyle=:none,
        grid=false,
        showaxis=false,
        ticks=false,
        legend=false,
        xlims=(0, ncol),
        ylims=(0, nrow),
        margin=0Plots.mm,
    )
    for (i, (label, color, ls, marker, lw)) in enumerate(entries)
        x0, y = mod1(i, ncol) - 1 + 0.03, nrow - cld(i, ncol) + 0.5
        x1 = x0 + PAIR_LEGEND_LINE_LENGTH
        if marker === :patch   # filled area sample (e.g. the net speedup region)
            plot!(
                p, Shape([x0, x1, x1, x0], [y - 0.25, y - 0.25, y + 0.25, y + 0.25]); color, fillalpha=0.1, linealpha=0
            )
        else
            plot!(p, [x0, x1], [y, y]; color, ls, lw)
            marker === :none || scatter!(
                p,
                [(x0 + x1) / 2],
                [y];
                color,
                marker,
                markersize=PAIR_CANVAS_SCALE * PAIR_MARKER_SIZE,
                markerstrokecolor=color,
            )
        end
        annotate!(p, x1 + 0.04, y, text(label, PAIR_LEGEND_FONT_SIZE, :left, :black, FONT_FAMILY))
    end
    return p
end

# One strip per figure pair, each listing only the series drawn in those panels
if PLOT_PAIRS && !isempty(ALL_CHIS)
    lw, lw_thick = PAIR_CANVAS_SCALE * PAIR_LINE_WIDTH, PAIR_CANVAS_SCALE * PAIR_LINE_WIDTH_THICK
    chi_entries(marker) = [(chi_label(g), CHI_COLOR[g], :solid, marker, lw) for g in ALL_CHIS]
    strips = Dict(
        "legend_strip_iterations" => vcat(
            Any[("Baseline", COLOR_BASELINE, :solid, :none, lw_thick)],
            chi_entries(:none),
            !isempty(PAIR_AMORTIZATION_GRIDS) ? Any[("Amortization step", :gray40, :dash, :none, lw)] : Any[],
        ),
        "legend_strip_speedup" => vcat(Any[("Break-even", COLOR_BASELINE, :dash, :none, lw)], chi_entries(:none)),
        "legend_strip_pareto" => vcat(
            Any[("Baseline", COLOR_BASELINE, :dash, :none, lw)],
            chi_entries(:circle),
            if PAIR_PARETO_REGION && PAIR_LEGEND_PARETO_REGION
                Any[("Net speedup region", :green, :solid, :patch, lw)]
            else
                Any[]
            end,
        ),
    )
    # Strips of the pair panels that are drawn
    strip_panels = Dict(
        "legend_strip_iterations" => [:step_iterations],
        "legend_strip_speedup" => [:step_speedup, :step_cumulative_speedup, :step_cumulative_total_speedup],
        "legend_strip_pareto" => [:pareto],
    )
    for (name, entries) in strips
        any(in(PAIR_PANELS), strip_panels[name]) || continue
        save_paper_plot(
            legend_strip(entries; columns=PAIR_LEGEND_COLUMNS, width=PAIR_TEXTWIDTH_PT),
            joinpath(PLOTS_DIR, "pairs", name),
        )
    end
end

# ---------------------------------------------------------
# 8. Record the statistics drawn (for the figure captions)
# ---------------------------------------------------------
open(joinpath(PLOTS_DIR, "stats_used.txt"), "w") do io
    println(io, "Statistics drawn by chorin_pipeline_stats/plot_paper.jl")
    println(io, "Data: ", RESULTS_FILE)
    for s in scaling
        println(io, "  $(s.label): $(s.n_min == s.n_max ? s.n_min : "$(s.n_min)-$(s.n_max)") repeats per configuration")
    end
    println(
        io, "Times, iterations, cumulative times, Pareto, time distributions: ", describe_stat(TIME_CENTER, TIME_SPREAD)
    )
    println(
        io,
        "Aggregate speedups (vs χ, heatmaps, scaling, speedup distributions): ",
        describe_stat(SPEEDUP_CENTER, SPEEDUP_SPREAD),
    )
    println(
        io, "Per-step / per-site profiles (incl. per-step speedup): ", describe_stat(PROFILE_CENTER, PROFILE_SPREAD)
    )
    println(
        io,
        "Speedups: ",
        if PAIRED_SPEEDUPS
            "paired by repeat index with the baseline of the same grid"
        else
            "all (baseline repeat, configuration repeat) pairs"
        end,
    )
    println(
        io,
        "Solve speedup = baseline total_solve_time / FAI total_solve_time; ",
        "total speedup = baseline total_solve_time / (FAI total_solve_time + precond_build_time)",
    )
    println(
        io,
        "Cumulative solve speedup (pairs/step_cumulative_speedup_*): per repeat, cumsum(baseline solve_times) / cumsum(FAI solve_times), ",
        "build time excluded; then ",
        describe_stat(PROFILE_CENTER, PROFILE_SPREAD),
        " over repeats",
    )
    println(
        io,
        "Cumulative total speedup (pairs/step_cumulative_total_speedup_*): per repeat, cumsum(baseline solve_times) / ",
        "(cumsum(FAI solve_times) + FAI precond_build_time); same statistics",
    )
    println(io, "Best μ per χ_max chosen by the $(BEST_MU_BY) speedup centre")
    println(
        io,
        "Discard first repeat per process: $(DISCARD_FIRST_IN_PROCESS); excluded repeats: $(EXCLUDED_REPEATS); excluded keys: $(EXCLUDED_KEYS)",
    )
end

@info "All paper plots successfully generated in $PLOTS_DIR"

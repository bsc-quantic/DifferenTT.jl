"""
    TT anisotropy benchmark, statistics variant (paper Figs. 9-10): paper plotter

We draw here the figures of Sec. 5.1 of the paper from the table of repeats that `collect.jl` wrote. A case is a pair
(dimension N, floor F); every configuration is compared with the baseline of the same case. We recompute every
statistic from the per-repeat rows, so the center and the band that we draw are chosen in the parameter block below, and
the speedups are paired by repeat index with the baseline.

# Figures

  - per N, against the floor F: solve and total speedup, solve, total and build time, and iterations
    (`*_vs_F_N_<N>`), and a configuration × F heatmap of the speedup (`heatmap_speedup_*_N_<N>`);
  - all N in one figure (`*_vs_F_all_N`), with a line style per N;
  - per case: speedup against χ_max with Sinc reference lines, Pareto front, site-by-site iterations, and the
    distributions over the repeats;
  - `plots_paper_mugersh/pairs/`: legend-free panels drawn at their printed size, to be placed two per row in the paper,
    plus the shared `legend_strip` that goes under each pair (`PLOT_PAIRS`). These are the figures of the paper;
  - `plots_paper_mugersh/stats_used.txt` records which statistics we drew, for the figure captions.

The families other than the pairs are switched off by default (`PLOT_*` flags below).

# Inputs and outputs

Reads `results_nocond_mugersh/collected_results.jld2` (`"df"`: one row per N, F, configuration and repeat) of this
folder, written by `collect.jl nocond` with the default μ rule, `gershgorin` (see `worker.jl`), and writes PDF figures
to `plots_paper_mugersh/` (and `N_<N>/F_<F>/` for the per-case ones) of this folder. The μ of a rule-based run differs
between cases but is one configuration, labelled μ = ‖A‖_∞. With `FAI_MU_RULE=fixed` it reads `results_nocond/` and
writes `plots_paper/` instead, labelling each μ by its value. In smoke mode (`DIFFERENTT_SMOKE=1`) it reads and writes
the folder `DIFFERENTT_SMOKE_DIR` that the smoke worker wrote, and draws the single smoke case; `experiments/smoke.jl`
sets it.

# How to run

From the root of the repository, after `collect.jl nocond`:

    julia --project=experiments experiments/paper/tt_anisotropy_stats/plot_paper.jl

# Resources

There is no job script: we run it locally on the collected data.

# Paper

Figs. 9-10 (`fig:mals_iters`, `fig:mals_site_iters`) of Sec. 5.1, from the pairs of total local iterations against F and
of iterations per site. The solve-time pair (`solve_time_vs_F` in `PAIR_VS_F`) is drawn too, but the paper does not show
it.
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
SIZE_VS_F = (850, 480)
SIZE_ALL_N = (950, 520)
SIZE_HEATMAP = (850, 520)
SIZE_PARETO = (850, 480)
SIZE_DIST = (900, 480)
MARGIN_MM = 6Plots.mm

# Line & Marker styling
LINE_WIDTH = 2.2
LINE_WIDTH_THICK = 3.0
LINE_WIDTH_THIN = 1.5
MARKER_SIZE = 5

# Series labels (LaTeX allowed)
LABEL_FAI = "FAI"
LABEL_FAI_L0 = "FAI\$(L_0)\$"
LABEL_SINC = "Sinc"
XLABEL_F = L"Floor $F$ (bound on $\kappa(L_0^{-1} L)$)"

# -------------------------------------------------------------------------
# Data selection
# -------------------------------------------------------------------------
include(joinpath(@__DIR__, "..", "..", "common", "smoke.jl"))
include(joinpath(@__DIR__, "..", "..", "common", "mu_rules.jl"))
# Smoke mode reads (and writes) the folder DIFFERENTT_SMOKE_DIR instead of this folder.
const OUT_ROOT = output_root(@__DIR__)
# FAI_MU_RULE / FAI_MU_SCALE select the results of a μ rule (common/mu_rules.jl); the default is the rule of the paper
const MU_RULE, MU_SCALE = fai_mu_rule()
RESULTS_FILE = joinpath(OUT_ROOT, mu_results_folder("results_nocond", MU_RULE, MU_SCALE), "collected_results.jld2")
PLOTS_DIR = joinpath(OUT_ROOT, mu_results_folder("plots_paper", MU_RULE, MU_SCALE))

SELECTED_N = nothing   # e.g. [20]; nothing = every dimension in the data
SELECTED_F = nothing   # e.g. [1, 4, 10, 100]; nothing = every floor
EXCLUDED_CASES = Tuple{Int,Float64}[]   # e.g. [(20, 100.0)]
SELECTED_METHODS = ["sinc", "fai", "fai_L0"]   # preconditioner families drawn (baseline always kept)
SELECTED_CHI = nothing   # FAI / FAI(L0) χ_max kept, e.g. [4, 8]; nothing = all
SELECTED_M = [3]       # Sinc M kept, e.g. [5, 7, 10]; nothing = all
EXCLUDED_KEYS = String[]  # e.g. ["sinc_M_3"]
EXCLUDED_REPEATS = Int[]     # repeat indices dropped from every configuration
DISCARD_FIRST_IN_PROCESS = false  # drop the first repeat each worker process ran (may carry JIT cost)

# How Sinc enters the curves vs F and the per-case plots:
#   :best -> one curve, the best M at each F (by the speedup being plotted), annotated with that M
#   :all  -> one curve per M;  :none -> Sinc omitted
SINC_MODE = :all
ANNOTATE_BEST_M = true
# Which Sinc M counts as "best" at each F when SINC_MODE = :best:
#   :plotted  -> the best value of the quantity each figure shows (highest speedup, lowest time or
#                iteration count), as in plot_results.jl, so different figures may pick different M
#   :sp_solve / :sp_total -> the highest solve / total speedup, one M per F shared by every figure
SINC_BEST_BY = :plotted
SINC_HLINE_BANDS = true   # spread band around the Sinc reference lines of the speedup-vs-χ plots
# Where the best-M labels of the vs-F curves go:
#   :top   -> one row of labels in a strip above the curves, centered on each F (never covers the data)
#   :point -> next to each Sinc marker, drawn above every curve
BEST_M_LABEL_POS = :top
BEST_M_STRIP_FRACTION = 0.12            # height of the :top strip, as a fraction of the extended y range
BEST_M_STRIP_TEXT = :value          # :value -> "10" (the narrow log-spaced low F ticks fit); :full -> "M = 10"
COLOR_BEST_M_LABEL = :darkorange3    # a shade darker than the Sinc curve, so the text reads on white

# -------------------------------------------------------------------------
# Statistics
#   center: :mean, :median, :gm (geometric mean)
#   spread: :sd, :sem, :ci95, :iqr, :minmax, :gsd (×/÷ geometric sd), :none
# -------------------------------------------------------------------------
TIME_CENTER = :median;
TIME_SPREAD = :iqr   # times, iterations, Pareto, time distributions
SPEEDUP_CENTER = :gm;
SPEEDUP_SPREAD = :gsd   # speedups (paired by repeat)
PROFILE_CENTER = :median;
PROFILE_SPREAD = :iqr   # per-site iteration profiles
PAIRED_SPEEDUPS = true      # false: ratios over every (baseline repeat, configuration repeat) pair

SPREAD_STYLE = :ribbon    # curves vs F / χ_max: :ribbon, :errorbar, :none
PROFILE_STYLE = :ribbon    # per-site curves: :ribbon or :none
SHOW_BANDS_ALL_N = false    # bands in the *_all_N figures
BAND_ALPHA = 0.18
LOG_BAND_FLOOR = 0.5        # on log axes a band that would cross zero is cut at this fraction of the center

ANNOTATE_HEATMAPS = true
ANNOTATE_HEATMAP_SPREAD = false   # add the spread under the value in each heatmap cell
HEATMAP_DIGITS = 2
HEATMAP_LOG_COLOR = true    # color by log10(speedup) (values still printed as speedups)
HEATMAP_DIVERGING = true    # color scale centered on a speedup of 1
HEATMAP_CMAP = :RdBu

PARETO_ERRORBARS = true       # TIME spread on both axes
PARETO_BASELINE_BAND = true

DIST_JITTER = 0.35
DIST_POINT_SIZE = 4
DIST_POINT_ALPHA = 0.6
DIST_SUMMARY_OFFSET = 0.3
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
PLOT_VS_F = false   # speedup / time / iterations vs F, one figure per N
PLOT_VS_F_ALL_N = false   # speedups vs F, every N in one figure
PLOT_HEATMAPS = false   # heatmap_speedup_{solve,total}_N_<N>
PLOT_CASE_SPEEDUP_CHI = false   # N_<N>/F_<F>/speedup_{solve,total}_vs_chi
PLOT_CASE_PARETO = false   # N_<N>/F_<F>/pareto
PLOT_CASE_SITE_ITERS = false   # N_<N>/F_<F>/site_iterations
PLOT_DISTRIBUTIONS = false   # N_<N>/F_<F>/dist_*

# Axis scales: :identity or :log10
XSCALE_F = :log10
YSCALE_SPEEDUP_VS_F = :log10
YSCALE_TIME_VS_F = :log10
YSCALE_ITERS_VS_F = :log10
YSCALE_SPEEDUP_VS_CHI = :log10
YSCALE_SITE_ITERS = :log10
YSCALE_DIST_TIME = :log10
YSCALE_DIST_SPEEDUP = :log10
N_LINESTYLES = [:solid, :dash, :dot, :dashdot]   # one per N in the *_all_N figures

# Per-figure axis limits, keyed by the output name without extension: either the file name, applied to
# that figure in every folder ("pareto", "speedup_solve_vs_F_N_20"), or the path relative to PLOTS_DIR
# for a single figure ("N_20/F_4/pareto"). Value: (min, max).
XLIMS = Dict{String,Tuple{Float64,Float64}}(
# "N_10/F_4/pareto" => (0, 120),
)
YLIMS = Dict{String,Tuple{Float64,Float64}}(
# "speedup_solve_vs_F_N_20" => (0.1, 20),
)

# Legend Position Toggles (e.g. :topleft, :topright, :bottomleft, :bottomright, :outertopright, :none)
LEGEND_POS_SPEEDUP_VS_F = :outertopright
LEGEND_POS_TIME_VS_F = :outertopright
LEGEND_POS_ITERS_VS_F = :topleft
LEGEND_POS_ALL_N = :outertopright
LEGEND_POS_SPEEDUP_CHI = :outertopright
LEGEND_POS_PARETO = :outertopright
LEGEND_POS_SITE_ITERS = :outertopright
LEGEND_POS_DIST = :none

# Color Palettes
COLOR_BASELINE = :black
COLOR_BREAK_EVEN = :black
COLOR_FAI_CURVE = :blue          # the FAI-vs-χ curve of the per-case plots
COLOR_SINC_BEST = :darkorange    # Sinc curve / line when SINC_MODE = :best
FAI_COLOR_SCHEME = :Blues;
FAI_GRADIENT_RANGE = (0.45, 1.0)   # one shade per FAI χ_max
L0_COLOR_SCHEME = :Purples;
L0_GRADIENT_RANGE = (0.6, 0.9)    # one shade per FAI(L0) χ_max
SINC_COLOR_SCHEME = :Oranges;
SINC_GRADIENT_RANGE = (0.4, 1.0)    # one shade per Sinc M

# -------------------------------------------------------------------------
# Paper figure pairs (plots_paper/pairs/): legend-free panels drawn at their printed size, placed two per
# row as LaTeX subfigures (PAIR_PANEL_FRACTION of \textwidth, included with width=\textwidth), plus one
# shared legend strip (legend_strip.pdf) that goes under every pair at full \textwidth. Sizes in pt.
# -------------------------------------------------------------------------
PLOT_PAIRS = true
PAIR_TEXTWIDTH_PT = 390      # \textwidth of the paper (elsarticle preprint, 12pt)
PAIR_PANEL_FRACTION = 0.49     # subfigure width as a fraction of \textwidth
PAIR_ASPECT = 0.8      # panel height / width
PAIR_N = nothing  # dimensions of the vs-F pairs, e.g. [10, 20]; nothing = every dimension
PAIR_VS_F = [   # one vs-F pair per entry: (field, ylabel, yscale, name)
    (:iters, "Total Local CG Iterations", YSCALE_ITERS_VS_F, "iters_vs_F"),
    (:solve, "Solve Time (s)", YSCALE_TIME_VS_F, "solve_time_vs_F"),
]
PAIR_SITE_CASES = SMOKE ? [(5, 10.0)] : [(10, 1.0), (10, 100.0), (20, 1.0), (20, 100.0)]   # (N, F) panels of the site-iteration pairs
# Short axis labels: the long ones do not fit a panel this small (the captions explain F)
PAIR_XLABEL_F = L"Floor $F$"
PAIR_SITE_YLABEL = "Local CG Iterations"
# GR draws text about 1.5 times its nominal pt size when 1 px = 1 pt (Plots' gr_point_mult), so the pair
# figures are drawn PAIR_CANVAS_SCALE times larger than printed and LaTeX scales them back down: font sizes
# below are then the printed sizes, and line widths / marker sizes are given in printed units too.
PAIR_CANVAS_SCALE = 1.5
PAIR_GUIDE_FONT_SIZE = 8
PAIR_TICK_FONT_SIZE = 7
PAIR_LEGEND_FONT_SIZE = 8
PAIR_ANNOTATION_FONT_SIZE = 7
PAIR_LINE_WIDTH = 1.2
PAIR_LINE_WIDTH_THICK = 1.6
PAIR_MARKER_SIZE = 2.5
PAIR_MARGIN = 1Plots.mm
PAIR_ANNOTATE_BEST_M = true
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
describe_stat(center, spread) = "center = $(CENTER_NAMES[center]), band = $(SPREAD_NAMES[spread])"

const NAN3 = (NaN, NaN, NaN)

"""
(center, lower, upper) of the finite samples in `x`; NaNs when nothing is left.
"""
function center_band(x; center=:mean, spread=:sd)
    v = Float64[Float64(y) for y in x if !ismissing(y) && isfinite(y)]
    (center === :gm || spread === :gsd) && filter!(>(0), v)
    isempty(v) && return NAN3
    n = length(v)
    c = if center === :mean
        mean(v)
    elseif center === :median
        median(v)
    elseif center === :gm
        exp(mean(log.(v)))
    else
        error("Unknown center statistic: $center")
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
Row-wise statistics of a sites × repeats matrix.
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

# =========================================================================
# Plot helpers
# =========================================================================

islog(scale) = !(scale in (:identity, :linear, :none))
clamp_lo(c, lo, logy) = logy && !(lo > 0) ? c * LOG_BAND_FLOOR : lo

"""
Central line with its spread drawn as a ribbon, as error bars, or not at all.
"""
function stat_series!(p, x, c, lo, hi; style=SPREAD_STYLE, logy=false, kw...)
    c, lo, hi = Float64.(c), Float64.(lo), Float64.(hi)
    lo = [clamp_lo(cc, l, logy) for (cc, l) in zip(c, lo)]
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
Heatmap of the centers Z (rows = y labels, bottom to top), cells annotated with value (and spread).
"""
function stat_heatmap(xlabels, ylabels, Z, Zlo, Zhi; xlabel, ylabel, spread, colorbar_title="")
    nx, ny = length(xlabels), length(ylabels)
    Zp = HEATMAP_LOG_COLOR ? [z > 0 ? log10(z) : NaN for z in Z] : Z
    valid = filter(isfinite, Zp)
    mid = HEATMAP_LOG_COLOR ? 0.0 : 1.0
    clim = if isempty(valid)
        :auto
    elseif HEATMAP_DIVERGING
        m = max(maximum(abs.(valid .- mid)), 1e-9)
        (mid - m, mid + m)
    else
        (minimum(valid), maximum(valid) + (maximum(valid) == minimum(valid) ? 1e-9 : 0.0))
    end
    p = heatmap(
        1:nx,
        1:ny,
        Zp;
        xticks=(1:nx, xlabels),
        yticks=(1:ny, ylabels),
        xlabel,
        ylabel,
        color=HEATMAP_CMAP,
        clim,
        colorbar_title,
        size=SIZE_HEATMAP,
        right_margin=8Plots.mm,
        framestyle=:box,
    )
    if ANNOTATE_HEATMAPS && clim !== :auto
        for i in 1:nx, j in 1:ny
            isfinite(Zp[j, i]) || continue
            t = (Zp[j, i] - clim[1]) / (clim[2] - clim[1])
            dark = HEATMAP_DIVERGING ? abs(t - 0.5) > 0.3 : t < 0.6
            annotate!(
                p,
                i,
                j,
                text(spread_text(Z[j, i], Zlo[j, i], Zhi[j, i], spread), ANNOTATION_FONT_SIZE, dark ? :white : :black),
            )
        end
    end
    return p
end

"""
Every repeat as a jittered point, plus the center ± spread marker next to it.
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
        lo = clamp_lo(c, lo, islog(yscale))
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

fmtF(F) = isinteger(F) ? string(Int(F)) : string(F)
function fmt_mu(μ)
    (isfinite(μ) && μ > 0) || return string(μ)
    e = floor(Int, log10(μ))
    m = μ / 10.0^e
    return isapprox(m, 1; atol=1e-8) ? "10^{$e}" : "$(round(m; digits=2)) \\times 10^{$e}"
end

function gradient_colors(scheme, n, zrange)
    n == 0 && return []
    g = cgrad(scheme)
    zs = n == 1 ? [zrange[2]] : range(zrange[1], zrange[2]; length=n)
    return [g[z] for z in zs]
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

getval(x, default) = ismissing(x) || isnothing(x) ? default : x

function keep_row(r)
    (SELECTED_N === nothing || r.N in SELECTED_N) || return false
    (SELECTED_F === nothing || r.F in SELECTED_F) || return false
    (Int(r.N), Float64(r.F)) in EXCLUDED_CASES && return false
    r.method == "baseline" && return true
    r.method in SELECTED_METHODS || return false
    r.method == "sinc" && return SELECTED_M === nothing || getval(r.M, 0) in SELECTED_M
    return SELECTED_CHI === nothing || getval(r.G_maxdim, 0) in SELECTED_CHI
end
df = df[[keep_row(r) for r in eachrow(df)], :]

# Series catalogue: one entry per configuration key, shared by every case. Under a μ rule the numeric μ changes with the
# case, so a key is identified by its rule (`mu_id`, `mu_tex`) and `mu` is NaN.
const INFO = Dict{String,NamedTuple}()
for r in eachrow(unique(df, :key))
    rule = hasproperty(df, :mu_rule) ? String(getval(r.mu_rule, "fixed")) : "fixed"
    scale = hasproperty(df, :mu_scale) ? Float64(getval(r.mu_scale, 1.0)) : 1.0
    μ = rule == "fixed" ? Float64(getval(r.mu, NaN)) : NaN
    INFO[r.key] = (
        method=String(r.method),
        chi=Int(getval(r.G_maxdim, 0)),
        mu=μ,
        mu_id=rule == "fixed" ? string(μ) : "rule:" * mu_rule_tag(rule, scale),
        mu_tex=rule == "fixed" ? fmt_mu(μ) : mu_rule_latex(rule, scale),
        M=Int(getval(r.M, 0)),
    )
end
const SINC_KEYS = sort([k for (k, v) in INFO if v.method == "sinc"]; by=k -> INFO[k].M)
const FAI_KEYS = sort([k for (k, v) in INFO if v.method == "fai"]; by=k -> (INFO[k].chi, INFO[k].mu, INFO[k].mu_id))
const L0_KEYS = sort([k for (k, v) in INFO if v.method == "fai_L0"]; by=k -> (INFO[k].chi, INFO[k].mu, INFO[k].mu_id))
const MULTI_MU = length(unique([INFO[k].mu_id for k in vcat(FAI_KEYS, L0_KEYS)])) > 1

const SERIES_COLOR = Dict{String,Any}("baseline" => COLOR_BASELINE)
for (keys_, scheme, zr) in (
    (FAI_KEYS, FAI_COLOR_SCHEME, FAI_GRADIENT_RANGE),
    (L0_KEYS, L0_COLOR_SCHEME, L0_GRADIENT_RANGE),
    (SINC_KEYS, SINC_COLOR_SCHEME, SINC_GRADIENT_RANGE),
)
    for (k, c) in zip(keys_, gradient_colors(scheme, length(keys_), zr))
        SERIES_COLOR[k] = c
    end
end
length(SINC_KEYS) == 1 && (SERIES_COLOR[only(SINC_KEYS)] = COLOR_SINC_BEST)   # a single Sinc M keeps the plain orange
series_marker(k) = (
    m=INFO[k].method;
    if m == "sinc"
        :diamond
    elseif m == "fai_L0"
        :utriangle
    else
        :circle
    end
)
series_ls(k) = (
    m=INFO[k].method;
    if m == "sinc"
        :dashdot
    elseif m == "fai_L0"
        :dot
    else
        :solid
    end
)

function series_label(k)
    k == "baseline" && return "Baseline"
    v = INFO[k]
    v.method == "sinc" && return latexstring("$LABEL_SINC \$M=$(v.M)\$")
    name = v.method == "fai_L0" ? LABEL_FAI_L0 : LABEL_FAI
    return if MULTI_MU
        latexstring("$name \$\\chi_{max}=$(v.chi), \\mu=$(v.mu_tex)\$")
    else
        latexstring("$name \$\\chi_{max}=$(v.chi)\$")
    end
end
INFO["baseline"] = (method="baseline", chi=0, mu=NaN, mu_id="", mu_tex="", M=0)

# Per-case statistics: STATS[(N, F)][key] -> NamedTuple of (c, lo, hi) tuples and speedup samples
const SP_KW = (; center=SPEEDUP_CENTER, spread=SPEEDUP_SPREAD)
const TIME_KW = (; center=TIME_CENTER, spread=TIME_SPREAD)
const STATS = Dict{Tuple{Int,Float64},Dict{String,Any}}()
for gk in groupby(df, [:N, :F])
    case = (Int(first(gk.N)), Float64(first(gk.F)))
    bsub = sort(gk[gk.key .== "baseline", :], :repeat)
    if nrow(bsub) == 0
        @warn "Skipping N=$(case[1]), F=$(fmtF(case[2])): missing baseline"
        continue
    end
    S = Dict{String,Any}()
    for k in unique(gk.key)
        sub = sort(gk[gk.key .== k, :], :repeat)
        s_solve = k == "baseline" ? ones(nrow(sub)) : ratio_samples(bsub, sub, :solve_time, :solve_time)
        s_total = k == "baseline" ? ones(nrow(sub)) : ratio_samples(bsub, sub, :solve_time, :total_time)
        S[k] = (
            sub=sub,
            n=nrow(sub),
            sp_solve_samples=s_solve,
            sp_total_samples=s_total,
            sp_solve=center_band(s_solve; SP_KW...),
            sp_total=center_band(s_total; SP_KW...),
            solve=center_band(sub.solve_time; TIME_KW...),
            total=center_band(sub.total_time; TIME_KW...),
            build=center_band(sub.precond_build_time; TIME_KW...),
            iters=center_band(sub.total_iters; TIME_KW...),
        )
    end
    STATS[case] = S
end
isempty(STATS) && error("No case with a baseline left after the selection")

const ALL_N = sort(unique(first.(collect(keys(STATS)))))
floors_of(N) = sort([F for (n, F) in keys(STATS) if n == N])
stat_of(N, F, k, field) = (S=get(STATS, (N, F), nothing); S === nothing || !haskey(S, k) ? NAN3 : getfield(S[k], field))

"""
Field that ranks the Sinc setups of a figure showing `field` (see SINC_BEST_BY).
"""
sinc_rank(field) = SINC_BEST_BY === :plotted ? field : SINC_BEST_BY

"""
Best Sinc key in S by the center of `field`: largest for speedups, smallest for times / iterations.
"""
function best_sinc(S, field)
    ks = [k for k in SINC_KEYS if haskey(S, k) && isfinite(getfield(S[k], field)[1])]
    isempty(ks) && return nothing
    vals = [getfield(S[k], field)[1] for k in ks]
    return ks[field in (:sp_solve, :sp_total) ? argmax(vals) : argmin(vals)]
end

# =========================================================================
# 1. Per N: curves vs the floor F
# =========================================================================

"""
Add baseline / Sinc / FAI / FAI(L0) curves of `field` against F for dimension N.
"""
function add_vs_F!(
    p,
    N,
    field;
    logy,
    style=SPREAD_STYLE,
    ls_override=nothing,
    with_labels=true,
    include_baseline=false,
    annotate_m=ANNOTATE_BEST_M,
    lw=LINE_WIDTH,
    lw_thick=LINE_WIDTH_THICK,
    ms=MARKER_SIZE,
    ann_size=ANNOTATION_FONT_SIZE,
    y_include=Float64[],
)
    Fs = floors_of(N)
    lbl(s) = with_labels ? s : ""
    lsx(ls) = ls_override === nothing ? ls : ls_override
    curve!(vals, label, color, ls, marker; lw=lw) = stat_series!(
        p,
        Fs,
        unz(vals)...;
        style,
        logy,
        label=lbl(label),
        color,
        ls=lsx(ls),
        lw,
        marker,
        markersize=ms,
        markerstrokecolor=color,
    )

    include_baseline &&
        curve!([stat_of(N, F, "baseline", field) for F in Fs], "Baseline", COLOR_BASELINE, :solid, :circle; lw=lw_thick)
    picks, vals = nothing, nothing
    if SINC_MODE === :best && !isempty(SINC_KEYS)
        picks = [best_sinc(get(STATS, (N, F), Dict()), sinc_rank(field)) for F in Fs]
        vals = [k === nothing ? NAN3 : stat_of(N, F, k, field) for (F, k) in zip(Fs, picks)]
        curve!(vals, latexstring("$LABEL_SINC (best \$M\$)"), COLOR_SINC_BEST, :dashdot, :diamond)
    elseif SINC_MODE === :all
        for k in SINC_KEYS
            curve!([stat_of(N, F, k, field) for F in Fs], series_label(k), SERIES_COLOR[k], :dashdot, :diamond)
        end
    end
    for k in vcat(FAI_KEYS, L0_KEYS)
        curve!([stat_of(N, F, k, field) for F in Fs], series_label(k), SERIES_COLOR[k], series_ls(k), series_marker(k))
    end
    # labeled last, so the text sits above every curve
    annotate_m && picks !== nothing && label_best_m!(p, Fs, picks, vals; logy, ann_size, y_include)
    return p
end

"""
Label each best-Sinc point with its M (see BEST_M_LABEL_POS); fixes the axis limits in :top mode,
so anything drawn afterwards must already be covered by them or listed in `y_include`.
"""
function label_best_m!(p, Fs, picks, vals; logy, ann_size, y_include=Float64[])
    shown = [(F, k, v[1]) for (F, k, v) in zip(Fs, picks, vals) if k !== nothing && isfinite(v[1])]
    isempty(shown) && return p
    lbl(k) = latexstring("\$M=$(INFO[k].M)\$")
    if BEST_M_LABEL_POS === :point
        for (F, k, c) in shown
            annotate!(p, F, c, text(lbl(k), ann_size, COLOR_BEST_M_LABEL, :bottom, FONT_FAMILY))
        end
        return p
    end
    # :top -> grow the y range by a strip and write the labels in it; widen x so the outer labels stay inside the box
    fw(x) = logy ? log10(x) : x
    bw(x) = logy ? 10^x : x
    ylo, yhi = fw.(extrema(vcat(collect(ylims(p)), y_include)))
    top = yhi + BEST_M_STRIP_FRACTION / (1 - BEST_M_STRIP_FRACTION) * (yhi - ylo)
    ylims!(p, (bw(ylo), bw(top)))
    xlog = islog(XSCALE_F)
    gx(x) = xlog ? log10(x) : x
    bx(x) = xlog ? 10^x : x
    pad = 0.06 * (gx(maximum(Fs)) - gx(minimum(Fs)))
    xlims!(p, (bx(gx(minimum(Fs)) - pad), bx(gx(maximum(Fs)) + pad)))
    hline!(p, [bw(yhi)]; color=:gray70, lw=0.5, ls=:dot, label="")
    for (F, k, _) in shown
        t = BEST_M_STRIP_TEXT === :full ? lbl(k) : latexstring("\$$(INFO[k].M)\$")
        annotate!(p, F, bw((yhi + top) / 2), text(t, ann_size, COLOR_BEST_M_LABEL, :center, FONT_FAMILY))
    end
    return p
end

if PLOT_VS_F
    for N in ALL_N
        Fs = floors_of(N)
        @info "Generating vs-F plots for N=$N"
        specs = [
            # (field, ylabel, yscale, legend, baseline?, break-even?, name)
            (
                :sp_solve,
                "Solve Speedup Factor",
                YSCALE_SPEEDUP_VS_F,
                LEGEND_POS_SPEEDUP_VS_F,
                false,
                true,
                "speedup_solve_vs_F",
            ),
            (
                :sp_total,
                "Total Speedup Factor",
                YSCALE_SPEEDUP_VS_F,
                LEGEND_POS_SPEEDUP_VS_F,
                false,
                true,
                "speedup_total_vs_F",
            ),
            (:solve, "Solve Time (s)", YSCALE_TIME_VS_F, LEGEND_POS_TIME_VS_F, true, false, "solve_time_vs_F"),
            (:total, "Total Time (s)", YSCALE_TIME_VS_F, LEGEND_POS_TIME_VS_F, true, false, "total_time_vs_F"),
            (
                :build,
                "Preconditioner Build Time (s)",
                YSCALE_TIME_VS_F,
                LEGEND_POS_TIME_VS_F,
                false,
                false,
                "build_time_vs_F",
            ),
            (
                :iters,
                "Total Local Krylov Iterations",
                YSCALE_ITERS_VS_F,
                LEGEND_POS_ITERS_VS_F,
                true,
                false,
                "iters_vs_F",
            ),
        ]
        for (field, ylabel, yscale, legend, with_base, breakeven, name) in specs
            p = plot(;
                xlabel=XLABEL_F,
                ylabel=ylabel,
                xscale=XSCALE_F,
                yscale=yscale,
                xticks=(Fs, fmtF.(Fs)),
                legend=legend,
                size=SIZE_VS_F,
                framestyle=:box,
            )
            add_vs_F!(
                p, N, field; logy=islog(yscale), include_baseline=with_base, y_include=breakeven ? [1.0] : Float64[]
            )
            breakeven && hline!(p, [1.0]; label="Break-even (1.0x)", color=COLOR_BREAK_EVEN, lw=LINE_WIDTH, ls=:dash)
            save_paper_plot(p, joinpath(PLOTS_DIR, "$(name)_N_$N"))
        end
    end
end

if PLOT_VS_F_ALL_N && length(ALL_N) > 1
    all_F = sort(unique(vcat([floors_of(N) for N in ALL_N]...)))
    for (field, ylabel, name) in (
        (:sp_solve, "Solve Speedup Factor", "speedup_solve_vs_F_all_N"),
        (:sp_total, "Total Speedup Factor", "speedup_total_vs_F_all_N"),
    )
        p = plot(;
            xlabel=XLABEL_F,
            ylabel=ylabel,
            xscale=XSCALE_F,
            yscale=YSCALE_SPEEDUP_VS_F,
            xticks=(all_F, fmtF.(all_F)),
            legend=LEGEND_POS_ALL_N,
            size=SIZE_ALL_N,
            framestyle=:box,
        )
        for (i, N) in enumerate(ALL_N)
            add_vs_F!(
                p,
                N,
                field;
                logy=islog(YSCALE_SPEEDUP_VS_F),
                style=SHOW_BANDS_ALL_N ? SPREAD_STYLE : :none,
                ls_override=N_LINESTYLES[mod1(i, length(N_LINESTYLES))],
                with_labels=(i == 1),
                annotate_m=false,
            )
        end
        for (i, N) in enumerate(ALL_N)
            plot!(
                p,
                [NaN],
                [NaN];
                color=:gray40,
                lw=LINE_WIDTH,
                ls=N_LINESTYLES[mod1(i, length(N_LINESTYLES))],
                label="N = $N",
            )
        end
        hline!(p, [1.0]; label="Break-even (1.0x)", color=COLOR_BREAK_EVEN, lw=LINE_WIDTH_THIN, ls=:dot)
        save_paper_plot(p, joinpath(PLOTS_DIR, name))
    end
end

# =========================================================================
# 2. Per N: configuration × F speedup heatmaps
# =========================================================================

if PLOT_HEATMAPS
    rows = reverse(vcat(SINC_KEYS, FAI_KEYS, L0_KEYS))   # heatmap rows run bottom to top
    for N in ALL_N
        Fs = floors_of(N)
        present = [k for k in rows if any(F -> haskey(STATS[(N, F)], k), Fs)]
        isempty(present) && continue
        for (field, name) in ((:sp_solve, "heatmap_speedup_solve"), (:sp_total, "heatmap_speedup_total"))
            grid(i) = [stat_of(N, F, k, field)[i] for k in present, F in Fs]
            hm = stat_heatmap(
                [fmtF(F) for F in Fs],
                [series_label(k) for k in present],
                grid(1),
                grid(2),
                grid(3);
                xlabel=XLABEL_F,
                ylabel="",
                spread=SPEEDUP_SPREAD,
                colorbar_title=HEATMAP_LOG_COLOR ? "log10 speedup" : "speedup",
            )
            save_paper_plot(hm, joinpath(PLOTS_DIR, "$(name)_N_$N"))
        end
    end
end

# =========================================================================
# 3. Per case (N, F)
# =========================================================================

"""
Sinc keys of case S drawn in a figure showing `field` (see SINC_MODE).
"""
function sinc_shown(S, field)
    if SINC_MODE === :all
        [k for k in SINC_KEYS if haskey(S, k)]
    elseif SINC_MODE === :best
        filter(!isnothing, [best_sinc(S, sinc_rank(field))])
    else
        String[]
    end
end
sinc_color(k) = SINC_MODE === :best ? COLOR_SINC_BEST : SERIES_COLOR[k]
sinc_label(k) = SINC_MODE === :best ? latexstring("$LABEL_SINC \$M=$(INFO[k].M)\$ (best)") : series_label(k)

"""
Site-by-site local iterations of case (N, F); `plot_kw` overrides the axis attributes (legend, size, ...).
"""
function site_iters_plot(
    N, F; ylabel="Local Krylov Iterations", lw=LINE_WIDTH, lw_thick=LINE_WIDTH_THICK, ms=MARKER_SIZE, plot_kw...
)
    S = STATS[(N, F)]
    Mb = repeat_matrix(S["baseline"].sub, :iters_per_site)
    isempty(Mb) && return nothing
    logy = islog(YSCALE_SITE_ITERS)
    axis_kw = merge((; legend=LEGEND_POS_SITE_ITERS, size=SIZE_STANDARD, framestyle=:box), values(plot_kw))
    p = plot(; xlabel=SITE_AXIS_LABEL, ylabel, yscale=YSCALE_SITE_ITERS, axis_kw...)
    stat_series!(
        p,
        1:size(Mb, 1),
        rowwise(Mb; center=PROFILE_CENTER, spread=PROFILE_SPREAD)...;
        style=PROFILE_STYLE,
        logy,
        label="Baseline",
        color=COLOR_BASELINE,
        lw=lw_thick,
        marker=:circle,
        markersize=ms,
    )
    for k in vcat(sinc_shown(S, :iters), [k for k in FAI_KEYS if haskey(S, k)], [k for k in L0_KEYS if haskey(S, k)])
        M = repeat_matrix(S[k].sub, :iters_per_site)
        isempty(M) && continue
        is_sinc = INFO[k].method == "sinc"
        stat_series!(
            p,
            1:size(M, 1),
            rowwise(M; center=PROFILE_CENTER, spread=PROFILE_SPREAD)...;
            style=PROFILE_STYLE,
            logy,
            label=is_sinc ? sinc_label(k) : series_label(k),
            color=is_sinc ? sinc_color(k) : SERIES_COLOR[k],
            lw,
            ls=series_ls(k),
            marker=series_marker(k),
            markersize=ms,
        )
    end
    return p
end

function plot_case(N, F)
    S = STATS[(N, F)]
    out_dir = joinpath(PLOTS_DIR, "N_$N", "F_$(fmtF(F))")
    fai_present = [k for k in FAI_KEYS if haskey(S, k)]
    l0_present = [k for k in L0_KEYS if haskey(S, k)]
    sinc_present = [k for k in SINC_KEYS if haskey(S, k)]

    # -- speedup vs χ_max with the Sinc references
    if PLOT_CASE_SPEEDUP_CHI && !isempty(fai_present)
        chis = sort(unique([INFO[k].chi for k in vcat(fai_present, l0_present)]))
        logy = islog(YSCALE_SPEEDUP_VS_CHI)
        for (field, ylabel, name) in (
            (:sp_solve, "Solve Speedup Factor", "speedup_solve_vs_chi"),
            (:sp_total, "Total Speedup Factor", "speedup_total_vs_chi"),
        )
            p = plot(;
                xlabel=L"Bond Dimension $\chi_{max}$",
                ylabel=ylabel,
                xticks=(chis, string.(chis)),
                yscale=YSCALE_SPEEDUP_VS_CHI,
                legend=LEGEND_POS_SPEEDUP_CHI,
                size=SIZE_STANDARD,
                framestyle=:box,
            )
            for k in sinc_shown(S, field)
                c, lo, hi = getfield(S[k], field)
                if SINC_HLINE_BANDS && isfinite(lo) && hi > lo
                    hspan!(p, [clamp_lo(c, lo, logy), hi]; color=sinc_color(k), alpha=BAND_ALPHA, linealpha=0, label="")
                end
                hline!(p, [c]; label=sinc_label(k), color=sinc_color(k), lw=LINE_WIDTH, ls=:dashdot)
            end
            fai_mus = sort(unique([(INFO[k].mu, INFO[k].mu_id) for k in fai_present]))   # by numeric μ, rules last
            for (j, (_, μ)) in enumerate(fai_mus)
                ks = sort([k for k in fai_present if INFO[k].mu_id == μ]; by=k -> INFO[k].chi)
                μ_tex = INFO[first(ks)].mu_tex
                col = length(fai_mus) == 1 ? COLOR_FAI_CURVE : palette(:tab10)[mod1(j, 10)]
                stat_series!(
                    p,
                    [INFO[k].chi for k in ks],
                    unz([getfield(S[k], field) for k in ks])...;
                    logy,
                    label=length(fai_mus) == 1 ? LABEL_FAI : latexstring("$LABEL_FAI \$\\mu=$(μ_tex)\$"),
                    color=col,
                    lw=LINE_WIDTH,
                    marker=:circle,
                    markersize=MARKER_SIZE,
                    markerstrokecolor=col,
                )
            end
            for k in l0_present
                c, lo, hi = getfield(S[k], field)
                scatter!(
                    p,
                    [INFO[k].chi],
                    [c];
                    yerror=SPREAD_STYLE === :none ? nothing : ([c - clamp_lo(c, lo, logy)], [hi - c]),
                    label=series_label(k),
                    color=SERIES_COLOR[k],
                    markerstrokecolor=SERIES_COLOR[k],
                    marker=:utriangle,
                    markersize=MARKER_SIZE + 2,
                )
            end
            hline!(p, [1.0]; label="Break-even (1.0x)", color=COLOR_BREAK_EVEN, lw=LINE_WIDTH, ls=:dash)
            save_paper_plot(p, joinpath(out_dir, name))
        end
    end

    # -- Pareto front: build vs solve time
    if PLOT_CASE_PARETO
        bc, blo, bhi = S["baseline"].solve
        pf = plot(;
            xlabel="Preconditioner Build Time (s)",
            ylabel="Linear Solve Time (s)",
            legend=LEGEND_POS_PARETO,
            size=SIZE_PARETO,
            framestyle=:box,
        )
        plot!(
            pf, [0, bc], [bc, 0]; fillrange=0, fillalpha=0.1, fillcolor=:green, linealpha=0, label="Net Speedup Region"
        )
        if PARETO_BASELINE_BAND && isfinite(blo) && bhi > blo
            hspan!(pf, [blo, bhi]; color=COLOR_BASELINE, alpha=BAND_ALPHA * 0.6, linealpha=0, label="")
        end
        hline!(pf, [bc]; label="Baseline Solve Time", color=COLOR_BREAK_EVEN, lw=LINE_WIDTH, ls=:dash)
        xmax, ymax = 0.0, max(bhi, bc)
        for k in vcat(fai_present, l0_present, sinc_present)
            b, s = S[k].build, S[k].solve
            isfinite(b[1]) && isfinite(s[1]) || continue
            err = if PARETO_ERRORBARS
                (; xerror=([b[1] - b[2]], [b[3] - b[1]]), yerror=([s[1] - s[2]], [s[3] - s[1]]))
            else
                (;)
            end
            scatter!(
                pf,
                [b[1]],
                [s[1]];
                err...,
                label=series_label(k),
                color=SERIES_COLOR[k],
                markerstrokecolor=SERIES_COLOR[k],
                marker=series_marker(k),
                markersize=INFO[k].method == "fai" ? MARKER_SIZE : MARKER_SIZE + 2,
            )
            xmax = max(xmax, PARETO_ERRORBARS ? b[3] : b[1])
            ymax = max(ymax, PARETO_ERRORBARS ? s[3] : s[1])
        end
        xlims!(pf, -PARETO_PAD * max(xmax, 1e-12), max(xmax, 1e-12) * (1.0 + PARETO_PAD))
        ylims!(pf, -PARETO_PAD * ymax, ymax * (1.0 + PARETO_PAD))
        save_paper_plot(pf, joinpath(out_dir, "pareto"))
    end

    # -- site-by-site local iterations
    if PLOT_CASE_SITE_ITERS
        p = site_iters_plot(N, F)
        p === nothing || save_paper_plot(p, joinpath(out_dir, "site_iterations"))
    end

    # -- distributions: every repeat of every configuration
    if PLOT_DISTRIBUTIONS
        ks = vcat(["baseline"], sinc_present, fai_present, l0_present)
        labels = [series_label(k) for k in ks]
        colors = [SERIES_COLOR[k] for k in ks]
        for (col, ylabel, name) in (
            (:solve_time, "Solve Time (s)", "dist_solve_time"),
            (:total_time, "Total Time (s)", "dist_total_time"),
            (:total_iters, "Total Local Krylov Iterations", "dist_iters"),
        )
            p = dist_plot(labels, [S[k].sub[!, col] for k in ks], colors; ylabel, yscale=YSCALE_DIST_TIME, TIME_KW...)
            save_paper_plot(p, joinpath(out_dir, name))
        end
        for (field, ylabel, name) in (
            (:sp_solve_samples, "Solve Speedup Factor", "dist_speedup_solve"),
            (:sp_total_samples, "Total Speedup Factor", "dist_speedup_total"),
        )
            p = dist_plot(
                labels[2:end],
                [getfield(S[k], field) for k in ks[2:end]],
                colors[2:end];
                ylabel,
                yscale=YSCALE_DIST_SPEEDUP,
                breakeven=true,
                SP_KW...,
            )
            save_paper_plot(p, joinpath(out_dir, name))
        end
    end
end

if PLOT_CASE_SPEEDUP_CHI || PLOT_CASE_PARETO || PLOT_CASE_SITE_ITERS || PLOT_DISTRIBUTIONS
    for N in ALL_N, F in floors_of(N)
        @info "Generating per-case plots for N=$N, F=$(fmtF(F))"
        plot_case(N, F)
    end
end

# =========================================================================
# 4. Paper figure pairs: legend-free panels at their printed size + one shared legend strip
# =========================================================================

"""
(label, color, linestyle, marker, linewidth) of every series drawn in the pair panels, in plot order.
"""
function pair_legend_entries()
    entries = Any[("Baseline", COLOR_BASELINE, :solid, :circle, PAIR_CANVAS_SCALE * PAIR_LINE_WIDTH_THICK)]
    if SINC_MODE === :best && !isempty(SINC_KEYS)
        push!(
            entries,
            (
                latexstring("$LABEL_SINC (best \$M\$)"),
                COLOR_SINC_BEST,
                :dashdot,
                :diamond,
                PAIR_CANVAS_SCALE * PAIR_LINE_WIDTH,
            ),
        )
    elseif SINC_MODE === :all
        append!(
            entries,
            [
                (series_label(k), SERIES_COLOR[k], :dashdot, :diamond, PAIR_CANVAS_SCALE * PAIR_LINE_WIDTH) for
                k in SINC_KEYS
            ],
        )
    end
    append!(
        entries,
        [
            (series_label(k), SERIES_COLOR[k], series_ls(k), series_marker(k), PAIR_CANVAS_SCALE * PAIR_LINE_WIDTH) for
            k in vcat(FAI_KEYS, L0_KEYS)
        ],
    )
    return entries
end

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
        plot!(p, [x0, x1], [y, y]; color, ls, lw)
        scatter!(
            p,
            [(x0 + x1) / 2],
            [y];
            color,
            marker,
            markersize=PAIR_CANVAS_SCALE * PAIR_MARKER_SIZE,
            markerstrokecolor=color,
        )
        annotate!(p, x1 + 0.04, y, text(label, PAIR_LEGEND_FONT_SIZE, :left, :black, FONT_FAMILY))
    end
    return p
end

if PLOT_PAIRS
    pair_dir = joinpath(PLOTS_DIR, "pairs")
    w = PAIR_CANVAS_SCALE * PAIR_PANEL_FRACTION * PAIR_TEXTWIDTH_PT
    panel_kw = (;
        legend=false,
        size=round.(Int, (w, PAIR_ASPECT * w)),
        margin=PAIR_CANVAS_SCALE * PAIR_MARGIN,
        framestyle=:box,
        guidefontsize=PAIR_GUIDE_FONT_SIZE,
        tickfontsize=PAIR_TICK_FONT_SIZE,
    )
    style_kw = map(
        x -> PAIR_CANVAS_SCALE * x, (; lw=PAIR_LINE_WIDTH, lw_thick=PAIR_LINE_WIDTH_THICK, ms=PAIR_MARKER_SIZE)
    )
    @info "Generating paper figure pairs in $pair_dir"
    for N in (PAIR_N === nothing ? ALL_N : intersect(PAIR_N, ALL_N)), (field, ylabel, yscale, name) in PAIR_VS_F
        Fs = floors_of(N)
        p = plot(; xlabel=PAIR_XLABEL_F, ylabel, xscale=XSCALE_F, yscale, xticks=(Fs, fmtF.(Fs)), panel_kw...)
        add_vs_F!(
            p,
            N,
            field;
            logy=islog(yscale),
            include_baseline=true,
            annotate_m=PAIR_ANNOTATE_BEST_M,
            ann_size=PAIR_ANNOTATION_FONT_SIZE,
            style_kw...,
        )
        save_paper_plot(p, joinpath(pair_dir, "$(name)_N_$N"))
    end
    for (N, F) in PAIR_SITE_CASES
        if !haskey(STATS, (N, F))
            @warn "No data for the site-iteration pair panel N=$N, F=$(fmtF(F))"
            continue
        end
        p = site_iters_plot(N, F; ylabel=PAIR_SITE_YLABEL, panel_kw..., style_kw...)
        p === nothing || save_paper_plot(p, joinpath(pair_dir, "site_iterations_N_$(N)_F_$(fmtF(F))"))
    end
    save_paper_plot(
        legend_strip(pair_legend_entries(); columns=PAIR_LEGEND_COLUMNS, width=PAIR_TEXTWIDTH_PT),
        joinpath(pair_dir, "legend_strip"),
    )
end

# =========================================================================
# 5. Record the statistics drawn (for the figure captions)
# =========================================================================
open(joinpath(PLOTS_DIR, "stats_used.txt"), "w") do io
    println(io, "Statistics drawn by tt_anisotropy_stats/plot_paper.jl")
    println(io, "Data: ", RESULTS_FILE)
    ns = [S[k].n for S in values(STATS) for k in keys(S)]
    println(io, "Repeats per configuration: $(minimum(ns))-$(maximum(ns)) (median $(median(ns)))")
    for S in values(STATS), k in keys(S)
        n_rep = maximum(ns)
        S[k].n < n_rep &&
            println(io, "  fewer repeats: N=$(first(S[k].sub.N)), F=$(fmtF(first(S[k].sub.F))), $k: n=$(S[k].n)")
    end
    println(io, "Times, iterations, Pareto, time distributions: ", describe_stat(TIME_CENTER, TIME_SPREAD))
    println(
        io, "Speedups (vs F, vs χ, heatmaps, speedup distributions): ", describe_stat(SPEEDUP_CENTER, SPEEDUP_SPREAD)
    )
    println(io, "Per-site iteration profiles: ", describe_stat(PROFILE_CENTER, PROFILE_SPREAD))
    println(
        io,
        "Speedups: ",
        if PAIRED_SPEEDUPS
            "paired by repeat index with the baseline of the same (N, F)"
        else
            "all (baseline repeat, configuration repeat) pairs"
        end,
    )
    println(
        io,
        "Solve speedup = baseline solve_time / solve_time; total speedup = baseline solve_time / total_time (incl. build)",
    )
    println(
        io,
        "Sinc mode: $SINC_MODE; best M per F chosen by: ",
        SINC_BEST_BY === :plotted ? "the center of the quantity each figure plots" : "the $(SINC_BEST_BY) center",
    )
    println(
        io,
        "Discard first repeat per process: $(DISCARD_FIRST_IN_PROCESS); excluded repeats: $(EXCLUDED_REPEATS); ",
        "excluded keys: $(EXCLUDED_KEYS); excluded cases: $(EXCLUDED_CASES)",
    )
end

@info "All paper plots successfully generated in $PLOTS_DIR"

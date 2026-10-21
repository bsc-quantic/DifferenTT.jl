"""
Plotting helpers shared by the `*_stats` plot scripts (needs `Plots`, `Random`, `Printf`, and
`stats_utils.jl` already included). Everything is built from `scatter`/`plot` primitives: no
StatsPlots.

Conventions (see stats_utils.jl):
  * central line = MEDIAN for anything derived from a ratio, MEAN otherwise; band = the matching
    spread (IQR for the median, +-1 sd for the mean);
  * on a LOG axis a symmetric +-sd ribbon would cross zero, so log axes use median + IQR (always
    positive) or a geometric band [gm/gsd, gm*gsd];
  * marker error bars on Pareto plots are the SEM ("where the mean is"), never the SD.
"""

using Plots
using Random
using Printf

"""
    band!(p, x, y, lo, hi; alpha=0.15, logy=false, kwargs...)

Line `y(x)` with an asymmetric ribbon from `lo` to `hi`. Non-finite bounds collapse onto `y`
(zero-width) so a NaN never breaks the fill polygon. With `logy=true` the lower edge is kept
strictly positive.
"""
function band!(p, x, y, lo, hi; alpha=0.15, logy::Bool=false, kwargs...)
    ys = Float64.(collect(y))
    lo2 = Float64[(isfinite(l) && isfinite(v)) ? min(l, v) : v for (l, v) in zip(lo, ys)]
    hi2 = Float64[(isfinite(h) && isfinite(v)) ? max(h, v) : v for (h, v) in zip(hi, ys)]
    if logy
        pos = filter(v -> isfinite(v) && v > 0, ys)
        floor_ = isempty(pos) ? 1e-12 : 0.5 * minimum(pos)
        lo2 = [isfinite(v) ? max(v, floor_) : v for v in lo2]
    end
    return plot!(p, collect(x), ys; ribbon=(ys .- lo2, hi2 .- ys), fillalpha=alpha, kwargs...)
end

"Individual repeat traces (columns of `M`) drawn faintly behind the central line."
function traces!(p, x, M::AbstractMatrix; color=:gray, alpha=0.25, lw=1.0)
    for j in 1:size(M, 2)
        plot!(p, collect(x), M[:, j]; color=color, alpha=alpha, lw=lw, label="")
    end
    return p
end

"""
    dist_plot(labels, samples; title, ylabel, yscale=:identity, colors=nothing)

One column per configuration: every repeat as a jittered point, a diamond at the median and a
vertical line from q25 to q75. The figure to look at first when a result looks too good: it makes a
bimodal or single-outlier distribution obvious in a way that mean +- std cannot.
"""
function dist_plot(labels::Vector{String}, samples::Vector{<:AbstractVector}; title::AbstractString="",
                   ylabel::AbstractString="", yscale::Symbol=:identity, colors=nothing)
    n = length(labels)
    p = plot(; title=title, ylabel=ylabel, yscale=yscale, legend=false, framestyle=:box,
             xticks=(1:n, labels), xrotation=45, xlims=(0.4, n + 0.6), bottom_margin=15Plots.mm,
             size=(max(900, 60 * n + 300), 520))
    rng = MersenneTwister(1234)     # local RNG: jitter only, deterministic figures
    pal = palette(:tab10)
    for (i, raw) in enumerate(samples)
        xs = _finite_floats(raw)
        yscale == :log10 && (xs = filter(>(0), xs))
        isempty(xs) && continue
        c = colors === nothing ? pal[mod1(i, length(pal))] : colors[mod1(i, length(colors))]
        jitter = 0.36 .* (rand(rng, length(xs)) .- 0.5)
        scatter!(p, i .+ jitter, xs; color=c, alpha=0.55, markersize=3.5, markerstrokewidth=0, label="")
        q25, q75 = quantile(xs, 0.25), quantile(xs, 0.75)
        plot!(p, [i, i], [q25, q75]; color=:black, lw=3, label="")
        scatter!(p, [i], [median(xs)]; marker=:diamond, markersize=7, color=:white,
                 markerstrokecolor=:black, markerstrokewidth=1.5, label="")
    end
    return p
end

"A colour limit that never errors on an all-NaN or constant matrix."
function safe_clim(Z::AbstractMatrix)
    v = filter(isfinite, Z)
    isempty(v) && return (0.0, 1.0)
    lo, hi = minimum(v), maximum(v)
    return lo == hi ? (lo - 0.5 * max(abs(lo), 1e-9), hi + 0.5 * max(abs(hi), 1e-9)) : (lo, hi)
end

"""
    annotated_heatmap(xlabels, ylabels, Z, A; kwargs...)

Heatmap of `Z` (rows = `ylabels`, columns = `xlabels`) with the string `A[i, j]` written in each
cell (empty strings are skipped).
"""
function annotated_heatmap(xlabels, ylabels, Z::AbstractMatrix, A::AbstractMatrix{<:AbstractString}; kwargs...)
    lo, hi = safe_clim(Z)
    hm = heatmap(xlabels, ylabels, Z; color=:viridis, clim=(lo, hi), right_margin=5Plots.mm, kwargs...)
    for i in axes(Z, 1), j in axes(Z, 2)
        (isempty(A[i, j]) || !isfinite(Z[i, j])) && continue
        frac = (Z[i, j] - lo) / (hi - lo)
        # On the categorical axes a cell spans [j-1, j] x [i-1, i], so its centre is (j-0.5, i-0.5).
        annotate!(hm, j - 0.5, i - 0.5, text(A[i, j], 7, frac > 0.55 ? :black : :white))
    end
    return hm
end

"Annotation strings `gm\\n×/÷gsd` for a ratio heatmap."
ratio_annotations(gm::AbstractMatrix, gsd::AbstractMatrix) =
    [isfinite(g) ? @sprintf("%.2f\n×/÷%.2f", g, s) : "" for (g, s) in zip(gm, gsd)]

"Annotation strings `mean\\n±sd` (or any pair)."
pm_annotations(m::AbstractMatrix, s::AbstractMatrix) =
    [isfinite(a) ? @sprintf("%.3g\n±%.2g", a, b) : "" for (a, b) in zip(m, s)]

"Annotation strings for a CV heatmap, in percent."
cv_annotations(cv::AbstractMatrix) = [isfinite(c) ? @sprintf("%.1f%%", 100 * c) : "" for c in cv]

"Geometric band `(lo, hi) = (gm/gsd, gm*gsd)` of vectors."
geo_band(gm, gsd) = (gm ./ gsd, gm .* gsd)

"""
    step_profile_bands(series; title, ylabel, yaxis=:log10) -> Union{Plot, Nothing}

Per-step median + IQR band of several configurations, on a log axis by default (residuals,
divergence ratios, ...). `series` holds one `(; label, M, color, lw)` per configuration, with `M` the
`L x n_repeats` matrix from `repeat_matrix`. Configurations without data (results written before
the metric existed) are skipped; returns `nothing` when none has any.
"""
function step_profile_bands(series; title, ylabel, yaxis=:log10)
    series = [s for s in series if !isempty(s.M)]
    isempty(series) && return nothing
    p = plot(title=title, xlabel="Chorin Time Step", ylabel=ylabel, yaxis=yaxis,
        legend=:outertopright, size=(1050, 600), legendfont=font(8, "Computer Modern"))
    for s in series
        r = rowwise_stats(s.M)
        band!(p, 1:length(r.median), r.median, r.q25, r.q75;
            label=s.label, color=s.color, lw=s.lw, logy=(yaxis == :log10), alpha=0.10)
    end
    return p
end

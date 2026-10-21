"""
Plotting shared by the `tt_anisotropy_stats` and `tt_anisotropy_global_stats` plot scripts.

Restricted to iteration counts, solve time and total time (preconditioner build + solve), and the
paired speedups built from those times. A configuration is a (case, key) pair with
`case = (N, F)`; the statistics come from the collector's `df_stats` (one row per configuration)
and the individual repeats from its long frame `df`.

Every vs-F figure groups the methods into series as `anisotropy_plot_utils.jl` does:
`Baseline`, `Sinc (best M)`, `FAI χ=g` (on L) and `FAI(L0) χ=g` (on the separable L0, control).
"Best M" is picked per F from the plotted statistic, so that series is biased favourably by
selection on noisy estimates; the chosen M is written next to each point.

Conventions (see stats_plot_utils.jl): times and iteration counts are on log axes, so they are
shown as median + IQR band (never mean ± sd, which can cross zero); speedups are per-repeat paired
ratios shown as geometric mean with the band [gm/gsd, gm*gsd].
"""

using JLD2
using DataFrames
using Plots
using Statistics
using Printf
using LaTeXStrings

include(joinpath(@__DIR__, "stats_utils.jl"))
include(joinpath(@__DIR__, "stats_plot_utils.jl"))

gr()
default(fmt=:png, dpi=300, size=(900, 500), fontfamily="Computer Modern", margin=5Plots.mm)
ENV["GKSwstype"] = "100"

const F_XLABEL = L"Floor $F$ = bound on $\kappa(L_0^{-1}L)$"

fmtF(F) = isinteger(F) ? string(Int(F)) : string(F)
case_label(c) = "N=$(c[1]), F=$(fmtF(c[2]))"

method_of(r) = string(getf(r, :method, ""))

"""
    stats_series(stN, score) -> Vector{(label, Dict F => stats row, style)}

Series of stats rows over F for one N. Where a group has several rows at the same F (sinc over M,
FAI over μ), the row with the smallest finite `score(row)` is kept.
"""
function stats_series(stN::AbstractDataFrame, score::Function)
    function best(rows)
        out = Dict{Float64, Any}()
        for r in rows
            v = score(r)
            (v isa Real && isfinite(v)) || continue
            F = Float64(r.case[2])
            (!haskey(out, F) || v < score(out[F])) && (out[F] = r)
        end
        return out
    end
    rows(m) = [r for r in eachrow(stN) if method_of(r) == m]
    series = Tuple{String, Dict{Float64, Any}, NamedTuple}[]

    b = best([r for r in eachrow(stN) if r.key == "baseline"])
    isempty(b) || push!(series, ("Baseline", b, (color=:black, marker=:circle, ls=:solid)))
    s = best(rows("sinc"))
    isempty(s) || push!(series, ("Sinc (best M)", s, (color=:darkorange, marker=:diamond, ls=:dashdot)))

    fai = rows("fai")
    gdims = sort(unique(Int(getf(r, :G_maxdim, 0)) for r in fai))
    blues = length(gdims) > 1 ? palette(:Blues, length(gdims) + 2)[3:end] : [:royalblue]
    for (i, g) in enumerate(gdims)
        push!(series, ("FAI χ=$g", best([r for r in fai if getf(r, :G_maxdim, 0) == g]), (color=blues[i], marker=:circle, ls=:solid)))
    end
    fai_L0 = rows("fai_L0")
    for g in sort(unique(Int(getf(r, :G_maxdim, 0)) for r in fai_L0))
        push!(series, ("FAI(L0) χ=$g", best([r for r in fai_L0 if getf(r, :G_maxdim, 0) == g]), (color=:purple, marker=:utriangle, ls=:dot)))
    end
    return series
end

"Red crosses on the points of configurations where some repeats hit maxiter (GlobalCG only)."
function mark_unconverged!(p, Fs, ys, rows; labelled::Ref{Bool})
    fr = [getf(r, :converged_mean, 1.0) for r in rows]
    bad = findall(i -> isfinite(ys[i]) && fr[i] isa Real && fr[i] < 1, eachindex(ys))
    isempty(bad) && return
    scatter!(p, Fs[bad], ys[bad]; marker=:xcross, color=:red, markersize=8, markerstrokewidth=2,
             label=labelled[] ? "" : "some repeats hit maxiter")
    labelled[] = true
end

"""
    plot_metric_vs_F(stN, N, metric; title, ylabel, file)

`metric` (a scalar column of the stats frame) vs F for every series: median line, IQR band, log
axes. Sinc picks the M with the smallest median per F.
"""
function plot_metric_vs_F(stN::AbstractDataFrame, N::Int, metric::Symbol; title::String, ylabel, file::String)
    p = plot(title=title, xlabel=F_XLABEL, ylabel=ylabel, xscale=:log10, yscale=:log10,
             legend=:outertopright, framestyle=:box)
    flagged = Ref(false)
    for (label, s, style) in stats_series(stN, r -> sstat(r, metric, :median))
        Fs = sort(collect(keys(s)))
        rows = [s[F] for F in Fs]
        mid = [sstat(r, metric, :median) for r in rows]
        ok = isfinite.(mid) .& (mid .> 0)
        any(ok) || continue
        Fs, rows, mid = Fs[ok], rows[ok], mid[ok]
        band!(p, Fs, mid, [sstat(r, metric, :q25) for r in rows], [sstat(r, metric, :q75) for r in rows];
              logy=true, label=label, lw=2, markersize=4, color=style.color, marker=style.marker, linestyle=style.ls)
        if label == "Sinc (best M)"
            annotate!(p, [(F, y, text("M=$(Int(getf(r, :M, 0)))", 6, style.color, :bottom)) for (F, y, r) in zip(Fs, mid, rows)])
        end
        mark_unconverged!(p, Fs, mid, rows; labelled=flagged)
    end
    savefig(p, file)
    return p
end

"""
    plot_speedup_vs_F(stN, N, ratio; file)

Paired per-repeat speedup (`:speedup_solve` or `:speedup_total`, baseline at the same (N, F) at the
same repeat index) vs F: geometric mean, band [gm/gsd, gm*gsd]. Sinc picks the M with the largest
geometric mean per F.
"""
function plot_speedup_vs_F(stN::AbstractDataFrame, N::Int, ratio::Symbol; file::String)
    name = ratio == :speedup_solve ? "Solve" : "Total"
    p = plot(title="$name speedup vs baseline (N=$N; paired, geo. mean ×/÷ geo. sd)", xlabel=F_XLABEL,
             ylabel="Speedup", xscale=:log10, yscale=:log10, legend=:outertopright, framestyle=:box)
    flagged = Ref(false)
    for (label, s, style) in stats_series(stN, r -> -sstat(r, ratio, :gm))
        label == "Baseline" && continue
        Fs = sort(collect(keys(s)))
        rows = [s[F] for F in Fs]
        gm = [sstat(r, ratio, :gm) for r in rows]
        ok = isfinite.(gm) .& (gm .> 0)
        any(ok) || continue
        Fs, rows, gm = Fs[ok], rows[ok], gm[ok]
        lo, hi = geo_band(gm, [sstat(r, ratio, :gsd) for r in rows])
        band!(p, Fs, gm, lo, hi; logy=true, label=label, lw=2, markersize=4, color=style.color,
              marker=style.marker, linestyle=style.ls)
        if label == "Sinc (best M)"
            annotate!(p, [(F, y, text("M=$(Int(getf(r, :M, 0)))", 6, style.color, :bottom)) for (F, y, r) in zip(Fs, gm, rows)])
        end
        mark_unconverged!(p, Fs, gm, rows; labelled=flagged)
    end
    hline!(p, [1.0], label="Break-even", color=:black, ls=:dash, lw=1.5)
    savefig(p, file)
    return p
end

"""
    plot_dists(df, st, case, dist_dir; metrics)

Per case, one `dist_plot` per metric: every configuration on the x axis, every repeat as a
jittered point, median diamond and IQR bar. `metrics` is a list of `(column, ylabel, file stem)`.
A `†` after a label marks configurations where some repeats hit maxiter.
"""
function plot_dists(df::AbstractDataFrame, st::AbstractDataFrame, case, dist_dir::String; metrics)
    rows = sort!([r for r in eachrow(st) if r.case == case], by=config_order)
    isempty(rows) && return
    labels = [config_label(r) * (getf(r, :converged_mean, 1.0) < 1 ? " †" : "") for r in rows]
    nrep = maximum(r.n_repeats for r in rows)
    N, F = case
    for (col, ylabel, stem) in metrics
        hasproperty(df, col) || continue
        samples = [_finite_floats(sel(df, :case, case, r.key)[!, col]) for r in rows]
        savefig(dist_plot(labels, samples; title="$ylabel per repeat ($(case_label(case)), n ≤ $nrep)",
                          ylabel=ylabel, yscale=:log10),
                joinpath(dist_dir, "dist_$(stem)_N_$(N)_F_$(fmtF(F)).png"))
    end
end

"Markdown section listing the configurations where some repeats hit maxiter (GlobalCG)."
function unconverged_markdown(st::AbstractDataFrame)
    hasproperty(st, :converged_mean) || return ""
    bad = sort!([r for r in eachrow(st) if getf(r, :converged_mean, 1.0) < 1], by=r -> (r.case, config_order(r)))
    isempty(bad) && return "## Convergence\n\nEvery repeat of every configuration converged before maxiter.\n"
    io = IOBuffer()
    println(io, "## Convergence\n")
    println(io, "These configurations hit maxiter in some or all repeats: their solve time is the time to maxiter, not to convergence. Their own speedups are not meaningful; where the listed configuration is the Baseline, every speedup at that (N, F) is a lower bound.\n")
    println(io, "| case | configuration | fraction converged | global iterations median |")
    println(io, "|---|---|---|---|")
    for r in bad
        println(io, @sprintf("| %s | %s | %.2f | %g |", case_label(r.case), config_label(r), sstat(r, :converged, :mean), sstat(r, :global_iters, :median)))
    end
    println(io)
    return String(take!(io))
end

"""
    plot_anisotropy_stats(; bench, results_folder, plots_dir, iters_col, iters_label, dataset_note)

Every figure for one benchmark, per N with F on the x axis:

  * `solve_time_vs_F_N_<N>.png`, `total_time_vs_F_N_<N>.png`, `iters_vs_F_N_<N>.png`
    (median, IQR band)
  * `speedup_solve_vs_F_N_<N>.png`, `speedup_total_vs_F_N_<N>.png` (paired geometric mean,
    geometric sd band)
  * `dist/dist_{solve_time,total_time,iters}_N_<N>_F_<F>.png`: every repeat, per configuration
  * `variability_summary.md`
"""
function plot_anisotropy_stats(; bench::String, results_folder::String, plots_dir::String,
                               iters_col::Symbol, iters_label::String, dataset_note::String="")
    results_file = joinpath(results_folder, "collected_results.jld2")
    @info "Loading data from $results_file"
    data = load(results_file)
    df, st = data["df"], data["df_stats"]
    dist_dir = joinpath(plots_dir, "dist")
    mkpath(dist_dir)

    st_N = [Int(c[1]) for c in st.case]
    for N in sort(unique(st_N))
        @info "Generating plots for N = $N"
        stN = st[st_N .== N, :]
        any(==("baseline"), stN.key) || @warn "No baseline for N=$N: speedups will be missing"

        plot_metric_vs_F(stN, N, :solve_time; title="Solve time (N=$N; median, IQR)", ylabel="Solve time [s]",
                         file=joinpath(plots_dir, "solve_time_vs_F_N_$(N).png"))
        plot_metric_vs_F(stN, N, :total_time; title="Total time = build + solve (N=$N; median, IQR)", ylabel="Total time [s]",
                         file=joinpath(plots_dir, "total_time_vs_F_N_$(N).png"))
        plot_metric_vs_F(stN, N, iters_col; title="$iters_label (N=$N; median, IQR)", ylabel="Iterations",
                         file=joinpath(plots_dir, "iters_vs_F_N_$(N).png"))
        plot_speedup_vs_F(stN, N, :speedup_solve; file=joinpath(plots_dir, "speedup_solve_vs_F_N_$(N).png"))
        plot_speedup_vs_F(stN, N, :speedup_total; file=joinpath(plots_dir, "speedup_total_vs_F_N_$(N).png"))

        for case in sort(unique(stN.case))
            plot_dists(df, st, case, dist_dir; metrics=[(:solve_time, "Solve time [s]", "solve_time"),
                                                        (:total_time, "Total time [s]", "total_time"),
                                                        (iters_col, iters_label, "iters")])
        end
    end

    write_variability_summary(joinpath(plots_dir, "variability_summary.md");
        bench=bench, df=df, df_stats=st, size_col=:case, size_label=case_label,
        solve_col=:solve_time, iters_col=iters_col, dataset_note=dataset_note,
        extra_markdown=unconverged_markdown(st),
        df_stats_all=get(data, "df_stats_all", nothing), n_discarded=nrow(get(data, "df_dropped", DataFrame())))
    @info "Plots saved to $plots_dir"
end

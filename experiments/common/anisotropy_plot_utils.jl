"""
Shared plotting helpers for the rotated anisotropy benchmarks (`tt_anisotropy`,
`tt_anisotropy_global`). Every plot has the floor F on the x-axis (log scale), and methods are
grouped into series:

- `Baseline`
- `Sinc (best M)`: for each F, the sinc run with the best value of the plotted metric
- `FAI χ=g`: FAI on L with G_maxdim = g (best μ for each F)
- `FAI(L0) χ=g`: FAI on the separable L0 = -Δ, used to precondition L (control)

"Best" always means the smallest value of the underlying metric (time, iterations, κ).
"""

using JLD2
using DataFrames
using Plots
using Statistics
using Printf
using LaTeXStrings

gr()
default(; fmt=:png, dpi=300, size=(900, 500), fontfamily="Computer Modern", margin=5Plots.mm)
ENV["GKSwstype"] = "100"

"""
Numeric value of `row[col]`, or NaN if the column is absent, missing or not a finite number.
"""
function numval(row, col::Symbol)::Float64
    hasproperty(row, col) || return NaN
    v = getproperty(row, col)
    (v === nothing || ismissing(v) || !(v isa Real)) && return NaN
    return Float64(v)
end

"""
For each F, the row of `rows` with the smallest finite `col`.
"""
function best_per_F(rows, col::Symbol)
    out = Dict{Float64,Any}()
    for r in rows
        v = numval(r, col)
        isfinite(v) || continue
        F = Float64(r.F)
        if !haskey(out, F) || v < numval(out[F], col)
            out[F] = r
        end
    end
    return out
end

method_rows(dfN, method::String) = [r for r in eachrow(dfN) if r.method == method]

"""
    method_series(dfN, col)

Series (label, Dict F => row, plot style) for every method group, choosing the best run per F
according to `col`.
"""
function method_series(dfN::AbstractDataFrame, col::Symbol)
    series = Tuple{String,Dict{Float64,Any},NamedTuple}[]

    base = best_per_F(method_rows(dfN, "baseline"), col)
    isempty(base) || push!(series, ("Baseline", base, (color=:black, marker=:circle, ls=:solid)))

    sinc = best_per_F(method_rows(dfN, "sinc"), col)
    isempty(sinc) || push!(series, ("Sinc (best M)", sinc, (color=:darkorange, marker=:diamond, ls=:dashdot)))

    fai = method_rows(dfN, "fai")
    gdims = sort(unique(Int(numval(r, :G_maxdim)) for r in fai))
    blues = length(gdims) > 1 ? palette(:Blues, length(gdims) + 2)[3:end] : [:royalblue]
    for (i, g) in enumerate(gdims)
        s = best_per_F([r for r in fai if numval(r, :G_maxdim) == g], col)
        push!(series, ("FAI χ=$g", s, (color=blues[i], marker=:circle, ls=:solid)))
    end

    fai_L0 = method_rows(dfN, "fai_L0")
    for g in sort(unique(Int(numval(r, :G_maxdim)) for r in fai_L0))
        s = best_per_F([r for r in fai_L0 if numval(r, :G_maxdim) == g], col)
        push!(series, ("FAI(L0) χ=$g", s, (color=:purple, marker=:utriangle, ls=:dot)))
    end
    return series
end

"""
Load the full-format discrete floor table (anisotropy_floor/discrete_floor.jl), if it exists.
"""
function load_discrete_floor()
    file = joinpath(@__DIR__, "..", "preconditioner_benchmarks", "anisotropy_floor", "discrete_floor.jld2")
    isfile(file) || return Dict{Tuple{Int,Float64},Float64}()
    rows = load(file, "rows")
    return Dict((r["N"], Float64(r["F"])) => r["floor_discrete"] for r in rows)
end

"""
Add the analytic floor (y = F) and, if available, the discrete floor of dimension N.
"""
function add_floor_refs!(p, Fs::Vector{Float64}, N::Int, discrete_floor)
    plot!(p, Fs, Fs; label="Analytic floor F", color=:gray, ls=:dash, lw=1.5)
    dF = [get(discrete_floor, (N, F), NaN) for F in Fs]
    if any(isfinite, dF)
        plot!(p, Fs, dF; label="Discrete floor (n=15)", color=:gray, ls=:dot, lw=2)
    end
    return p
end

"""
Plot `yfun(row)` (default: `row[col]`) vs F for every method series.
"""
function plot_metric_vs_F(
    dfN::AbstractDataFrame,
    col::Symbol;
    title::String,
    ylabel,
    file::String,
    yscale=:log10,
    yfun=r -> numval(r, col),
    skip_baseline::Bool=false,
    refs=nothing,
)
    p = plot(;
        title=title,
        xlabel="Floor F = κ(L0⁻¹L) bound",
        ylabel=ylabel,
        xscale=:log10,
        yscale=yscale,
        legend=:outertopright,
        framestyle=:box,
    )
    for (label, s, style) in method_series(dfN, col)
        skip_baseline && label == "Baseline" && continue
        Fs = sort(collect(keys(s)))
        ys = [yfun(s[F]) for F in Fs]
        ok = isfinite.(ys) .& (yscale == :log10 ? ys .> 0 : trues(length(ys)))
        any(ok) || continue
        plot!(p, Fs[ok], ys[ok]; label=label, lw=2, markersize=4, style.color, style.marker, linestyle=style.ls)
    end
    isnothing(refs) || refs(p)
    savefig(p, file)
    return p
end

"""
Speedup vs F relative to the baseline at the same F, for `time_col` ∈ (:solve_time, :total_time).
"""
function plot_speedup_vs_F(dfN::AbstractDataFrame, N::Int, time_col::Symbol, plots_dir::String)
    base = best_per_F(method_rows(dfN, "baseline"), :solve_time)
    isempty(base) && return @warn "No baseline data for N=$N, skipping speedup plot"
    name = time_col == :solve_time ? "solve" : "total"
    yfun = r -> haskey(base, Float64(r.F)) ? numval(base[Float64(r.F)], :solve_time) / numval(r, time_col) : NaN
    plot_metric_vs_F(
        dfN,
        time_col;
        title="$(titlecase(name)) speedup vs baseline (N=$N)",
        ylabel="Speedup",
        file=joinpath(plots_dir, "speedup_$(name)_vs_F_N_$(N).png"),
        yfun=yfun,
        skip_baseline=true,
        refs=p -> hline!(p, [1.0]; label="Break-even", color=:black, ls=:dash, lw=1.5),
    )
end

"""
Sinc plateau: `col` vs M for each F (sinc κ/iterations saturate at the floor as M grows).
"""
function plot_sinc_plateau(dfN::AbstractDataFrame, N::Int, col::Symbol, ylabel, file::String)
    sinc = method_rows(dfN, "sinc")
    isempty(sinc) && return nothing
    Fs = sort(unique(Float64(r.F) for r in sinc))
    p = plot(;
        title="Sinc plateau (N=$N)",
        xlabel="Sinc quadrature parameter M",
        ylabel=ylabel,
        yscale=:log10,
        legend=:outertopright,
        framestyle=:box,
    )
    colors = palette(:viridis, max(length(Fs), 2))
    for (i, F) in enumerate(Fs)
        rows = sort([r for r in sinc if Float64(r.F) == F]; by=r -> numval(r, :M))
        Ms = [numval(r, :M) for r in rows]
        ys = [numval(r, col) for r in rows]
        ok = isfinite.(ys) .& (ys .> 0)
        any(ok) || continue
        plot!(p, Ms[ok], ys[ok]; label="F = $(F)", color=colors[i], marker=:diamond, lw=2)
    end
    savefig(p, file)
end

"""
FAI cost vs F: preconditioner build time and bond dimension of the solution.
"""
function plot_cost_vs_F(dfN::AbstractDataFrame, N::Int, plots_dir::String)
    p_build = plot_metric_vs_F(
        dfN,
        :precond_build_time;
        title="Preconditioner build time (N=$N)",
        ylabel="Build time [s]",
        skip_baseline=true,
        file=joinpath(plots_dir, "build_time_vs_F_N_$(N).png"),
    )
    p_rank = plot_metric_vs_F(
        dfN,
        :solution_maxdim;
        title="Solution bond dimension (N=$N)",
        ylabel="max bond dimension",
        yscale=:identity,
        file=joinpath(plots_dir, "solution_maxdim_vs_F_N_$(N).png"),
    )
    p = plot(p_build, p_rank; layout=(2, 1), size=(900, 900))
    savefig(p, joinpath(plots_dir, "fai_cost_vs_F_N_$(N).png"))
end

"""
    crossover_summary(df, plots_dir; iters_col)

Per N and per time metric: best sinc vs best FAI (on L) at every F, the F = 1 winner (H1) and
the smallest F from which the best FAI beats the best sinc (F*, H3). Writes
`crossover_summary.md` and `crossover_ratio.png` (best sinc time / best FAI time vs F, per N).
"""
function crossover_summary(df::AbstractDataFrame, plots_dir::String; iters_col::Symbol)
    Ns = sort(unique(df.N))
    lines = String[
        "# Crossover summary: FAI (on L) vs sinc (on L0)\n",
        "Ratio = best sinc time / best FAI time (> 1: FAI faster). F* = smallest F from which FAI stays faster.\n",
    ]
    p_ratio = plot(;
        title="Best sinc time / best FAI time",
        xlabel="Floor F = κ(L0⁻¹L) bound",
        ylabel="Ratio",
        xscale=:log10,
        yscale=:log10,
        legend=:outertopright,
        framestyle=:box,
    )
    for N in Ns
        dfN = df[df.N .== N, :]
        for (time_col, ls) in ((:solve_time, :dash), (:total_time, :solid))
            sinc = best_per_F(method_rows(dfN, "sinc"), time_col)
            fai = best_per_F(method_rows(dfN, "fai"), time_col)
            fai_L0 = best_per_F(method_rows(dfN, "fai_L0"), time_col)
            base = best_per_F(method_rows(dfN, "baseline"), time_col)
            Fs = sort(collect(intersect(keys(sinc), keys(fai))))
            isempty(Fs) && continue

            push!(lines, "\n## N = $N, $(time_col)\n")
            push!(
                lines,
                "| F | baseline [s] | best sinc [s] (M) | best FAI [s] (key) | FAI(L0) [s] | ratio | $(iters_col) sinc / FAI | winner |",
            )
            push!(lines, "|---|---|---|---|---|---|---|---|")
            ratios = Float64[]
            for F in Fs
                s, f = sinc[F], fai[F]
                ratio = numval(s, time_col) / numval(f, time_col)
                push!(ratios, ratio)
                b = haskey(base, F) ? @sprintf("%.2f", numval(base[F], time_col)) : "–"
                l0 = haskey(fai_L0, F) ? @sprintf("%.2f", numval(fai_L0[F], time_col)) : "–"
                push!(
                    lines,
                    @sprintf(
                        "| %g | %s | %.2f (%d) | %.2f (%s) | %s | %.2f | %g / %g | %s |",
                        F,
                        b,
                        numval(s, time_col),
                        Int(numval(s, :M)),
                        numval(f, time_col),
                        f.key,
                        l0,
                        ratio,
                        numval(s, iters_col),
                        numval(f, iters_col),
                        ratio > 1 ? "FAI" : "sinc"
                    )
                )
            end
            idx = findfirst(i -> all(ratios[i:end] .> 1), eachindex(ratios))
            fstar = isnothing(idx) ? "not reached" : string(Fs[idx])
            push!(lines, "\nF = $(Fs[1]) winner: **$(ratios[1] > 1 ? "FAI" : "sinc")**. F* = **$fstar**.")
            plot!(
                p_ratio,
                Fs,
                ratios;
                label="N=$N, $(time_col == :solve_time ? "solve" : "total")",
                marker=:circle,
                lw=2,
                linestyle=ls,
            )
        end
    end
    hline!(p_ratio, [1.0]; label="Break-even", color=:black, ls=:dot, lw=1.5)
    savefig(p_ratio, joinpath(plots_dir, "crossover_ratio.png"))
    write(joinpath(plots_dir, "crossover_summary.md"), join(lines, "\n") * "\n")
    println(join(lines, "\n"))
end

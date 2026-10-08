"""
    Linsolve preconditioner SPD benchmark: Plotting Script

Reads the single results file of worker.jl and generates, in `plots/`:
    1. `<section>_eigmin.png` (dense sections): λ_min/λ_max of the symmetric part S of the preconditioner and of S·A
       vs χ, one line per system size. Log scale on |value|: filled markers are negative values, hollow markers
       positive ones. The dashed lines are 1/κ(A), the value of S = A⁻¹ (for which SA = I has ratio 1).
    2. `<section>_nneg.png` (dense sections): fraction of negative eigenvalues of S vs χ.
    3. `<section>_weyl.png` (dense sections): ‖S - A⁻¹‖₂ vs χ. Below λ_min(A⁻¹) (dashed), S is SPD by Weyl's inequality.
    4. `chorin_linsolve_vs_truncinv.png`: fraction of negative eigenvalues of both preconditioners, per size.
    5. `<section>_spd_check.png` (large sections): outcome of the local SPD checks of the first solve vs χ. Filled
       markers: the solve stopped at a local P⁻¹A that is not SPD (the preconditioner is not SPD); hollow markers: the
       solve ran through. The y-axis is the number of local blocks checked (before the stop, if any).
plus `summary.md` with the numbers.

Usage: julia --project=. experiments/preconditioner_benchmarks/linsolve_spd/plot_results.jl [results_file]
"""

using JLD2
using Plots
using Printf
using LaTeXStrings

gr()
default(fmt=:png, dpi=300, size=(900, 500), fontfamily="Computer Modern", margin=5Plots.mm)
ENV["GKSwstype"] = "100"

const BASE_FOLDER = "experiments/preconditioner_benchmarks/linsolve_spd"
const RESULTS_FILE = length(ARGS) > 0 ? ARGS[1] : joinpath(BASE_FOLDER, "results", "linsolve_spd.jld2")
const PLOTS_DIR = joinpath(dirname(RESULTS_FILE), "..", "plots")
mkpath(PLOTS_DIR)

const DENSE_SECTIONS = ["chorin_linsolve_dense", "chorin_truncinv_dense"]
const LARGE_SECTIONS = ["chorin_linsolve_large"]

const SECTION_TITLES = Dict(
    "chorin_linsolve_dense" => "Chorin pressure Poisson: linsolve preconditioner",
    "chorin_truncinv_dense" => L"Chorin pressure Poisson: truncated exact $A^{-1}$",
    "chorin_linsolve_large" => "Chorin pressure Poisson: linsolve preconditioner",
)

"""Nested `section => case => entry => Dict` view of the results file."""
function load_results(file::AbstractString)
    results = Dict{String, Dict{String, Dict{String, Any}}}()
    jldopen(file, "r") do f
        for section in keys(f)
            section == "meta" && continue
            results[section] = Dict(case => Dict(entry => f["$section/$case/$entry"] for entry in keys(f[section][case]))
                                    for case in keys(f[section]))
        end
    end
    return results
end

"""Readable label of a case key: `nodes_4x4` → `(4, 4)`."""
function case_label(case::AbstractString)
    m = match(r"^nodes_(\d+)x(\d+)$", case)
    isnothing(m) || return "($(m[1]), $(m[2]))"
    return case
end

"""Cases sorted by system size (number of rows)."""
sorted_cases(section_results) = sort(collect(keys(section_results)); by = c -> begin
    m = match(r"(\d+)\D+(\d+)$", c)
    isnothing(m) ? (0, 0) : (parse(Int, m[1]), parse(Int, m[2]))
end)

"""`χ`-sorted entries of a case (the entries named `chi_*`)."""
function chi_entries(case_results)
    entries = [v for (k, v) in case_results if startswith(k, "chi_")]
    return sort(entries; by = e -> e["chi"])
end

"""
Line of |y| on a log axis, with filled markers where y < 0 and hollow markers where y > 0. Zeros are dropped,
since they have no place on a log axis.
"""
function signed_log_series!(p, x, y; color, label, linestyle=:solid, marker=:circle)
    keep = y .!= 0 .& isfinite.(y)
    x, y = x[keep], y[keep]
    isempty(x) && return p
    plot!(p, x, abs.(y); color, label, linestyle, marker, markersize=5, markerstrokecolor=color,
          markercolor=[v < 0 ? color : :white for v in y])
    return p
end

"""Invisible series that explain the filled/hollow marker convention in the legend."""
function sign_legend!(p)
    scatter!(p, [NaN], [NaN]; marker=:circle, markercolor=:black, markerstrokecolor=:black, label="negative value")
    scatter!(p, [NaN], [NaN]; marker=:circle, markercolor=:white, markerstrokecolor=:black, label="positive value")
    return p
end

series_color(i::Int) = palette(:default)[mod1(i, 16)]

# ---- Dense sections ----

function plot_dense_eigmin(section, res)
    p1 = plot(; title=L"Symmetric part $S$ of the preconditioner", xlabel=L"\chi", ylabel=L"|\lambda_{\min}(S)| / \lambda_{\max}(S)", yscale=:log10, legend=:outerright)
    p2 = plot(; title=L"Preconditioned operator $SA$", xlabel=L"\chi", ylabel=L"|\lambda_{\min}(SA)| / \lambda_{\max}(SA)", yscale=:log10, legend=false)
    for (i, case) in enumerate(sorted_cases(res))
        entries = chi_entries(res[case])
        isempty(entries) && continue
        χs = [e["chi"] for e in entries]
        color = series_color(i)
        signed_log_series!(p1, χs, [e["eigmin_S"] / e["eigmax_S"] for e in entries]; color, label=case_label(case))
        signed_log_series!(p2, χs, [e["eigmin_SA"] / e["eigmax_SA"] for e in entries]; color, label=case_label(case))
        if haskey(res[case], "operator")
            hline!(p1, [1 / res[case]["operator"]["cond_A"]]; color, linestyle=:dash, label="")
        end
    end
    sign_legend!(p1)
    fig = plot(p1, p2; layout=(1, 2), size=(1500, 550), plot_title=SECTION_TITLES[section])
    savefig(fig, joinpath(PLOTS_DIR, "$(section)_eigmin.png"))
end

function plot_dense_nneg(section, res)
    p = plot(; title=SECTION_TITLES[section], xlabel=L"\chi", ylabel=L"Fraction of negative eigenvalues of $S$", legend=:topright)
    for (i, case) in enumerate(sorted_cases(res))
        entries = chi_entries(res[case])
        isempty(entries) && continue
        plot!(p, [e["chi"] for e in entries], [e["nneg_S"] / e["n"] for e in entries]; color=series_color(i), marker=:circle, label=case_label(case))
    end
    savefig(p, joinpath(PLOTS_DIR, "$(section)_nneg.png"))
end

function plot_dense_weyl(section, res)
    p = plot(; title=SECTION_TITLES[section], xlabel=L"\chi", ylabel=L"\|S - A^{-1}\|_2", yscale=:log10, legend=:outerright)
    for (i, case) in enumerate(sorted_cases(res))
        entries = filter(e -> e["sym_error_2norm"] > 0, chi_entries(res[case]))
        isempty(entries) && continue
        color = series_color(i)
        plot!(p, [e["chi"] for e in entries], [e["sym_error_2norm"] for e in entries]; color, marker=:circle, label=case_label(case))
        hline!(p, [first(entries)["eigmin_Ainv"]]; color, linestyle=:dash, label=L"\lambda_{\min}(A^{-1})")
    end
    savefig(p, joinpath(PLOTS_DIR, "$(section)_weyl.png"))
end

function plot_linsolve_vs_truncinv(results)
    ls, tr = get(results, "chorin_linsolve_dense", nothing), get(results, "chorin_truncinv_dense", nothing)
    (isnothing(ls) || isnothing(tr)) && return
    p = plot(; title="Chorin pressure Poisson: negative eigenvalues of S",
             xlabel=L"\chi", ylabel=L"Fraction of negative eigenvalues of $S$", legend=:outerright)
    for (i, case) in enumerate(sorted_cases(ls))
        color = series_color(i)
        for (res, style, name) in ((ls, :solid, "linsolve"), (tr, :dash, L"truncated $A^{-1}$"))
            haskey(res, case) || continue
            entries = chi_entries(res[case])
            plot!(p, [e["chi"] for e in entries], [e["nneg_S"] / e["n"] for e in entries]; color, linestyle=style, marker=:circle,
                  label="$(case_label(case)), $name")
        end
    end
    savefig(p, joinpath(PLOTS_DIR, "chorin_linsolve_vs_truncinv.png"))
end

# ---- Large sections ----

"""Solve entry of a large case: the first Chorin step."""
solve_entry(e) = get(e, "first_step", nothing)

function local_entry_label(key, e)
    startswith(key, "chi_") && return "linsolve χ = $(e["chi"])"
    haskey(e, "G_maxdim") && return "FAI gdim = $(e["G_maxdim"])"
    return key
end

function plot_large_spd_check(section, res)
    p = plot(; title=SECTION_TITLES[section], xlabel=L"\chi", ylabel="Local blocks checked", yscale=:log10, legend=:outerright)
    for (i, case) in enumerate(sorted_cases(res))
        entries = filter(e -> !isnothing(solve_entry(e)), chi_entries(res[case]))
        isempty(entries) && continue
        color = series_color(i)
        solves = solve_entry.(entries)
        plot!(p, [e["chi"] for e in entries], [max(s["nblocks_checked"], 1) for s in solves]; color, label=case_label(case),
              marker=:circle, markersize=6, markerstrokecolor=color, markercolor=[s["not_spd"] ? color : :white for s in solves])
    end
    scatter!(p, [NaN], [NaN]; marker=:circle, markercolor=:black, markerstrokecolor=:black, label="stopped: not SPD")
    scatter!(p, [NaN], [NaN]; marker=:circle, markercolor=:white, markerstrokecolor=:black, label="ran through")
    savefig(p, joinpath(PLOTS_DIR, "$(section)_spd_check.png"))
end

# ---- Summary ----

fmt(x::Real) = isfinite(x) ? @sprintf("%.3e", x) : "–"

function write_summary(results)
    io = IOBuffer()
    println(io, "# Linsolve preconditioner SPD benchmark: summary\n")
    println(io, "S is the symmetric part of the preconditioner. A negative λ_min(S) (or a positive #neg) proves the preconditioner is not SPD.\n")
    for section in DENSE_SECTIONS
        haskey(results, section) || continue
        println(io, "## $section\n")
        for case in sorted_cases(results[section])
            op = get(results[section][case], "operator", nothing)
            header = isnothing(op) ? "" : " (n = $(op["n"]), κ(A) = $(fmt(op["cond_A"])), λ_min(A⁻¹) = $(fmt(op["eigmin_Ainv"])))"
            println(io, "### $(case_label(case))$header\n")
            println(io, "| χ | λ_min(S) | λ_max(S) | #neg(S) | λ_min(SA) | κ(SA) | ‖S−A⁻¹‖₂ | ‖AP+PA−2I‖/‖2I‖ | asym(P) |")
            println(io, "|---|---|---|---|---|---|---|---|---|")
            for e in chi_entries(results[section][case])
                println(io, "| $(e["chi"]) | $(fmt(e["eigmin_S"])) | $(fmt(e["eigmax_S"])) | $(e["nneg_S"]) | $(fmt(e["eigmin_SA"])) | $(fmt(e["cond_SA"])) | ",
                        "$(fmt(e["sym_error_2norm"])) | $(fmt(e["sylvester_residual"])) | $(fmt(e["asym_P"])) |")
            end
            println(io)
        end
    end
    for section in LARGE_SECTIONS
        haskey(results, section) || continue
        println(io, "## $section\n")
        for case in sorted_cases(results[section])
            println(io, "### $(case_label(case))\n")
            println(io, "| preconditioner | ‖AX+XA−2I‖/‖2I‖ | SPD check | where | min Re λ(P⁻¹A) / max\\|λ\\| | blocks checked | sweeps | rel. residual | other failure |")
            println(io, "|---|---|---|---|---|---|---|---|---|")
            for key in sort(collect(keys(results[section][case])); by = k -> (startswith(k, "fai") ? 1 : 0, k))
                e = results[section][case][key]
                solve = solve_entry(e)
                isnothing(solve) && continue
                outcome = solve["not_spd"] ? "**not SPD**" : (solve["failed"] ? "–" : "passed")
                where_ = solve["not_spd"] ? "$(solve["not_spd_label"]), position $(solve["not_spd_position"])" : ""
                println(io, "| $(local_entry_label(key, e)) | $(fmt(get(e, "sylvester_residual", NaN))) | $outcome | $where_ | ",
                        "$(fmt(get(solve, "not_spd_eigmin_rel", NaN))) | $(solve["nblocks_checked"]) | $(solve["nsweeps_done"]) | ",
                        "$(fmt(get(solve, "rel_res", NaN))) | $(solve["failed"] ? first(solve["error"], 80) : "") |")
            end
            println(io)
        end
    end
    file = joinpath(PLOTS_DIR, "summary.md")
    write(file, String(take!(io)))
    @info "Summary written to $file"
end

# ---- Main ----

@info "Loading $RESULTS_FILE"
results = load_results(RESULTS_FILE)

for section in DENSE_SECTIONS
    haskey(results, section) || continue
    @info "Plotting $section"
    plot_dense_eigmin(section, results[section])
    plot_dense_nneg(section, results[section])
    plot_dense_weyl(section, results[section])
end

plot_linsolve_vs_truncinv(results)

for section in LARGE_SECTIONS
    haskey(results, section) || continue
    @info "Plotting $section"
    plot_large_spd_check(section, results[section])
end

write_summary(results)
@info "Plots saved to $PLOTS_DIR"

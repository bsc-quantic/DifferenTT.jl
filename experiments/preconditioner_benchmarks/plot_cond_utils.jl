"""
Shared helpers for preconditioner-benchmark plots.

`pc_conds_per_site` and `avg_pc_cond` store the absolute local
condition number κ(P^{-1}A) recorded in `pc_cond_store`, not the ratio
κ(A)/κ(P^{-1}A).
"""

function nonempty_numeric_vec(x)::Bool
    (x === nothing || ismissing(x)) && return false
    try
        return !isempty(x) && any(v -> v isa Number && isfinite(Float64(v)) && v > 0, x)
    catch
        return false
    end
end

function row_vec(row, col::Symbol)::Any
    hasproperty(row, col) || return Float64[]
    v = getproperty(row, col)
    (ismissing(v) || v === nothing) && return Float64[]
    return v
end

function clean_kappa(xs)::Vector{Float64}
    return [c isa Number && isfinite(Float64(c)) && c > 0 ? Float64(c) : NaN for c in xs]
end

function kappa_pc(d::AbstractDict)::Union{Nothing,Vector{Float64}}
    v = get(d, "pc_conds_per_site", nothing)
    nonempty_numeric_vec(v) || return nothing
    return clean_kappa(v)
end

function cond_improvement_factor(cond_v::AbstractDict; baseline_avg::Float64=NaN)::Float64
    κPC = get(cond_v, "avg_pc_cond", NaN)
    κA = get(cond_v, "avg_cond", NaN)
    κPC_ok = κPC isa Number && isfinite(Float64(κPC)) && Float64(κPC) > 0
    κPC_ok || return NaN
    κA_ok = κA isa Number && isfinite(Float64(κA)) && Float64(κA) > 0
    κA_use = κA_ok ? Float64(κA) : baseline_avg
    (isfinite(Float64(κA_use)) && Float64(κA_use) > 0) || return NaN
    return Float64(κA_use) / Float64(κPC)
end

function mean_finite(xs)::Float64
    f = filter(isfinite, xs)
    isempty(f) && return NaN
    return mean(f)
end

function avg_site_speedup(base_per_site, v_per_site)::Float64
    (nonempty_numeric_vec(base_per_site) && nonempty_numeric_vec(v_per_site)) || return NaN
    n = min(length(base_per_site), length(v_per_site))
    ratios = Float64[]
    for i in 1:n
        b, s = base_per_site[i], v_per_site[i]
        if b isa Number &&
            s isa Number &&
            isfinite(Float64(b)) &&
            isfinite(Float64(s)) &&
            Float64(b) > 0 &&
            Float64(s) > 0
            push!(ratios, Float64(b) / Float64(s))
        end
    end
    isempty(ratios) && return NaN
    return mean(ratios)
end

"""
    site_speedup_over_time(base_matrix, v_matrix) -> Vector{Float64}

For time-evolution data where iteration counts are recorded per step *and* per site
(`base_matrix`/`v_matrix` are `Vector{Vector{Float64}}`, outer index = time step, inner
index = MPS site), returns, for each site, the average across time steps of the per-step
speedup ratio `base_iters[t][site] / v_iters[site]`. This averages ratios over time first
(one value per step) rather than summing iteration counts over time and then taking a
single ratio, so a handful of high-iteration outlier steps cannot dominate the result.
"""
function site_speedup_over_time(base_matrix, v_matrix)::Vector{Float64}
    (base_matrix === nothing || v_matrix === nothing || isempty(base_matrix) || isempty(v_matrix)) && return Float64[]
    n_steps = min(length(base_matrix), length(v_matrix))
    n_steps == 0 && return Float64[]

    n_sites = minimum(length(base_matrix[t]) for t in 1:n_steps)
    n_sites = min(n_sites, minimum(length(v_matrix[t]) for t in 1:n_steps))
    n_sites == 0 && return Float64[]

    result = fill(NaN, n_sites)
    for s in 1:n_sites
        ratios = Float64[]
        for t in 1:n_steps
            b, v = base_matrix[t][s], v_matrix[t][s]
            if b isa Number &&
                v isa Number &&
                isfinite(Float64(b)) &&
                isfinite(Float64(v)) &&
                Float64(b) > 0 &&
                Float64(v) > 0
                push!(ratios, Float64(b) / Float64(v))
            end
        end
        isempty(ratios) || (result[s] = mean(ratios))
    end
    return result
end

"""
    site_speedup_ratio(base_per_site, v_per_site) -> Vector{Float64}

Elementwise per-site speedup `base_per_site[i] / v_per_site[i]`, for data with no time
dimension (a single solve per configuration). Invalid entries (non-finite or non-positive)
become `NaN`, and the result keeps every site's position so it can be plotted against the
MPS site index like `iters_per_site`.
"""
function site_speedup_ratio(base_per_site, v_per_site)::Vector{Float64}
    (nonempty_numeric_vec(base_per_site) && nonempty_numeric_vec(v_per_site)) || return Float64[]
    n = min(length(base_per_site), length(v_per_site))
    result = fill(NaN, n)
    for i in 1:n
        b, s = base_per_site[i], v_per_site[i]
        if b isa Number &&
            s isa Number &&
            isfinite(Float64(b)) &&
            isfinite(Float64(s)) &&
            Float64(b) > 0 &&
            Float64(s) > 0
            result[i] = Float64(b) / Float64(s)
        end
    end
    return result
end

function match_cond(grid_data_cond::AbstractDict, v::AbstractDict)
    k = get(v, "key", "")
    if k != "" && haskey(grid_data_cond, k)
        return grid_data_cond[k]
    end
    method = get(v, "method", nothing)
    for (ck, cv) in grid_data_cond
        ck == "baseline" && continue
        cv_method = get(cv, "method", nothing)
        if method !== nothing && cv_method !== nothing
            cv_method == method || continue
            if method == "sinc" && get(cv, "M", 0) == get(v, "M", 0)
                return cv
            elseif method != "sinc" &&
                get(cv, "mu", NaN) == get(v, "mu", NaN) &&
                get(cv, "G_maxdim", 0) == get(v, "G_maxdim", 0)
                return cv
            end
        elseif get(cv, "mu", NaN) == get(v, "mu", NaN) && get(cv, "G_maxdim", 0) == get(v, "G_maxdim", 0)
            return cv
        end
    end
    return nothing
end

"""
    step_profile_plot(series; title, ylabel, yaxis=:log10) -> Union{Plot, Nothing}

Per-step curves of several configurations, on a log axis by default (residuals, divergence ratios, ...).
`series` holds one `(; label, values, color, lw[, ls])` per configuration, drawn in order.
Non-finite or non-positive entries become gaps, and configurations without data (results written
before the metric existed) are skipped; returns `nothing` when none has any.
"""
function step_profile_plot(series; title, ylabel, yaxis=:log10)
    series = [s for s in series if nonempty_numeric_vec(s.values)]
    isempty(series) && return nothing
    p = plot(;
        title=title,
        xlabel="Chorin Time Step",
        ylabel=ylabel,
        yaxis=yaxis,
        legend=:outertopright,
        size=(1050, 600),
        legendfont=font(8, "Computer Modern"),
    )
    for s in series
        plot!(
            p, 1:length(s.values), clean_kappa(s.values); label=s.label, color=s.color, lw=s.lw, ls=get(s, :ls, :solid)
        )
    end
    return p
end

# Folder suffix of the lid-driven cavity results to plot. The lid count comes from the first command-line argument
# ("1"/"single" or "2"/"double") or from the CAVITY_LIDS environment variable; without either, it is detected as
# the only suffix whose results exist. `has_results(suffix)` tells whether the results of that suffix exist.
# Single-lid results may carry the "_singlelid" suffix or none (the workers' default folders).
function cavity_results_suffix(has_results::Function)
    candidates = Dict(
        "1" => ["_singlelid", ""], "single" => ["_singlelid", ""], "2" => ["_doublelid"], "double" => ["_doublelid"]
    )
    spec = lowercase(length(ARGS) > 0 ? ARGS[1] : get(ENV, "CAVITY_LIDS", ""))
    if !isempty(spec)
        haskey(candidates, spec) || error("Lid count must be 1/single or 2/double, got \"$spec\"")
        found = filter(has_results, candidates[spec])
        return isempty(found) ? first(candidates[spec]) : first(found)  # nothing found: the loader reports the missing file
    end
    found = filter(has_results, ["_singlelid", "", "_doublelid"])
    length(found) == 1 && return only(found)
    error(
        if isempty(found)
            "No lid-driven cavity results found: pass 1/single or 2/double to pick the folder"
        else
            "Results found for several suffixes $(found): pass 1/single or 2/double to pick one"
        end,
    )
end

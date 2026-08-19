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

function kappa_pc(d::AbstractDict)::Union{Nothing, Vector{Float64}}
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
            elseif method != "sinc" && get(cv, "mu", NaN) == get(v, "mu", NaN) && get(cv, "G_maxdim", 0) == get(v, "G_maxdim", 0)
                return cv
            end
        elseif get(cv, "mu", NaN) == get(v, "mu", NaN) && get(cv, "G_maxdim", 0) == get(v, "G_maxdim", 0)
            return cv
        end
    end
    return nothing
end

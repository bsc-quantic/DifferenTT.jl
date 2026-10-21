"""
Shared analysis helpers for the `*_stats` benchmark collectors and plot scripts.

Statistics only (`Statistics`, no StatsBase/StatsPlots, which are not in Project.toml).

What the repeats measure, and what they do not: each task already logs many numbers (200 Chorin
steps, a per-site iteration profile), but those are not repeated samples. Consecutive Chorin steps
are a physically evolving, correlated sequence. Only the repeats, which re-run the identical
configuration under a different random stream, answer "how different would the answer be if I ran
this again?". They mix two sources of variation that this file keeps separable:

  * algorithmic: iteration counts, residuals, bond dimensions, condition numbers (randomized FAI
    build, random initial guess, and any RNG use inside the solver: the Chorin ResidualALS
    baseline turned out to vary with the seed too, so a baseline is not automatically a
    machine-noise floor; the summary decides per configuration from the data);
  * machine: wall-clock times at fixed iteration count.

Ratios (speedups) are computed PAIRED, by repeat index, and summarised geometrically. In the
block layout (see stats_harness.jl) a repeat index of one system size is one block, i.e. one job on
one node, so a pair is a within-node ratio; `n_pairs_samejob` counts the pairs for which that
holds (a block refilled after a stop or failure is split over jobs), and `samejob_only` restricts
the ratios to them. `block_effects` estimates the node (job) effect that cancels in those ratios.
The layout is decided per system size: one dataset can hold block sizes and unit-layout sizes (one
job per unit, so no pair is within-node; `samejob_only` leaves those sizes alone).
"""

using Statistics
using DataFrames
using JLD2
using Printf
using Random

# ---------------------------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------------------------

const REPEAT_FILE_RE = r"^task_\d+_rep_\d+\.jld2$"

"""
    collect_repeat_files(tasks_folder; postprocess=identity) -> DataFrame

Long frame, one row per (task, repeat): every `task_*_rep_*.jld2` in `tasks_folder`. `postprocess`
receives each result `Dict{String,Any}` and returns the (possibly extended) dict to store.
"""
function collect_repeat_files(tasks_folder::AbstractString; postprocess=identity)
    df = DataFrame()
    isdir(tasks_folder) || error("No tasks folder at $tasks_folder")
    for file in sort(filter(f -> occursin(REPEAT_FILE_RE, f), readdir(tasks_folder)))
        try
            result = postprocess(load(joinpath(tasks_folder, file), "result"))
            push!(df, Dict{Symbol, Any}(Symbol(k) => v for (k, v) in result), cols=:union)
        catch e
            println("Error loading $file: $e")
        end
    end
    return df
end

"""
    split_first_repeats(df; discard) -> (kept, dropped)

`first_in_process` marks the first unit a worker process ran, the only one that can still carry
residual JIT cost. In the block layout that is one randomly chosen method per block, so discarding
drops one pair per block, not one method systematically. With `discard=false` (the default in the collectors) nothing is dropped; with
`true` those rows are moved to `dropped`, and the report must say so and show before/after.
"""
function split_first_repeats(df::DataFrame; discard::Bool)
    (discard && hasproperty(df, :first_in_process)) || return df, df[falses(nrow(df)), :]
    mask = coalesce.(df.first_in_process, false) .== true
    return df[.!mask, :], df[mask, :]
end

# ---------------------------------------------------------------------------------------------
# Scalar / vector / ratio statistics
# ---------------------------------------------------------------------------------------------

_finite_floats(v) = Float64[Float64(e) for e in v if e isa Real && isfinite(e)]

"""
    scalar_stats(v) -> NamedTuple

`n, mean, std, sem, median, q25, q75, min, max, cv` over the finite entries of `v` (missing and
non-finite values are skipped). `std` is the sample standard deviation (0 for n = 1).
"""
function scalar_stats(v::AbstractVector)
    x = _finite_floats(v)
    n = length(x)
    n == 0 && return (; n=0, mean=NaN, std=NaN, sem=NaN, median=NaN, q25=NaN, q75=NaN,
                        min=NaN, max=NaN, cv=NaN)
    m = mean(x)
    s = n > 1 ? std(x) : 0.0
    return (; n, mean=m, std=s, sem = n > 1 ? s / sqrt(n) : 0.0,
              median=median(x), q25=quantile(x, 0.25), q75=quantile(x, 0.75),
              min=minimum(x), max=maximum(x), cv = m == 0 ? NaN : s / abs(m))
end

"""
    rowwise_stats(M) -> NamedTuple

Statistics of each row of `M` (rows = positions such as steps or sites, columns = repeats or
pairs) over its finite entries: `mean, std, median, q25, q75, gm, gsd, n` as vectors. `gm`/`gsd`
are geometric and only defined where every finite entry is positive (NaN otherwise).
"""
function rowwise_stats(M::AbstractMatrix)
    L = size(M, 1)
    z() = fill(NaN, L)
    out = (; mean=z(), std=z(), median=z(), q25=z(), q75=z(), gm=z(), gsd=z(), n=zeros(Int, L))
    for i in 1:L
        x = _finite_floats(view(M, i, :))
        out.n[i] = length(x)
        isempty(x) && continue
        out.mean[i] = mean(x)
        out.std[i] = length(x) > 1 ? std(x) : 0.0
        out.median[i] = median(x)
        out.q25[i] = quantile(x, 0.25)
        out.q75[i] = quantile(x, 0.75)
        if all(>(0), x)
            lx = log.(x)
            out.gm[i] = exp(mean(lx))
            out.gsd[i] = length(x) > 1 ? exp(std(lx)) : 1.0
        end
    end
    return out
end

"""
    vector_stats(vs; label="") -> NamedTuple

Elementwise statistics of a set of per-repeat vectors (per-step or per-site metrics): stacks
them to `L x n_repeats` and reduces along the repeat dimension. Ragged lengths (a truncated job,
an early-stopped run) are truncated to the shortest with a warning, never zero-padded, which
would bias the tail of every curve downward.
"""
function vector_stats(vs::AbstractVector; label::AbstractString="")
    vs = [Float64.(collect(v)) for v in vs if v isa AbstractVector && !isempty(v)]
    isempty(vs) && return (; n=0, mean=Float64[], std=Float64[], median=Float64[],
                             q25=Float64[], q75=Float64[])
    L = minimum(length, vs)
    any(length(v) != L for v in vs) &&
        @warn "Ragged vector metric across repeats; truncating to the shortest" metric=label L lengths=length.(vs)
    M = reduce(hcat, [v[1:L] for v in vs])
    rs = rowwise_stats(M)
    return (; n=length(vs), mean=rs.mean, std=rs.std, median=rs.median, q25=rs.q25, q75=rs.q75)
end

"""
    ratio_stats(s) -> NamedTuple

Summary of a set of positive ratios (per-repeat speedups). Geometric: `gm = exp(mean(log s))`,
`gsd = exp(std(log s))`, band `[gm/gsd, gm*gsd]`. A ratio's sampling distribution is skewed, so an
arithmetic mean +- std can put the lower bar below zero, which is meaningless on a speedup axis and
impossible on a log scale. `median/q25/q75` are the distribution-free fallback; `mean/cv` are the
arithmetic ones (kept for the CV heatmaps).
"""
function ratio_stats(s::AbstractVector)
    x = [Float64(e) for e in s if e isa Real && isfinite(e) && e > 0]
    n = length(x)
    n == 0 && return (; n=0, gm=NaN, gsd=NaN, median=NaN, q25=NaN, q75=NaN, mean=NaN, cv=NaN)
    lx = log.(x)
    return (; n, gm=exp(mean(lx)), gsd = n > 1 ? exp(std(lx)) : 1.0,
              median=median(x), q25=quantile(x, 0.25), q75=quantile(x, 0.75),
              mean=mean(x), cv = n > 1 ? std(x) / mean(x) : 0.0)
end

# ---------------------------------------------------------------------------------------------
# Pairing by repeat index
# ---------------------------------------------------------------------------------------------

"Dict repeat index => value of `col` for the rows of `sub` (missing values skipped)."
function by_repeat(sub::AbstractDataFrame, col::Symbol)
    d = Dict{Int, Any}()
    hasproperty(sub, col) || return d
    for r in eachrow(sub)
        v = r[col]
        ismissing(v) && continue
        d[Int(r.repeat)] = v
    end
    return d
end

_common_repeats(a::AbstractDict, b::AbstractDict) = sort!(collect(intersect(keys(a), keys(b))))

"""
    same_job_repeats(a, b) -> Set{Int}

Repeat indices at which the rows of `a` and `b` ran in the same SLURM job (`slurm_job_id`, unique
per array element). In the block layout that means on the same node, so their ratio is
within-node; in the per-task layout it is always empty.
"""
function same_job_repeats(a::AbstractDataFrame, b::AbstractDataFrame)
    ja, jb = by_repeat(a, :slurm_job_id), by_repeat(b, :slurm_job_id)
    return Set{Int}(r for r in _common_repeats(ja, jb) if !isempty(string(ja[r])) && string(ja[r]) == string(jb[r]))
end

"`d` restricted to the repeat indices in `keep` (`nothing` keeps everything)."
restrict_repeats(d::AbstractDict, ::Nothing) = d
restrict_repeats(d::AbstractDict, keep::AbstractSet) = filter(p -> first(p) in keep, d)

"""
    paired_ratios(num, den) -> Vector{Float64}

`num[r] / den[r]` for the repeat indices present on both sides (a repeat missing on one side drops
that pair). Only finite positive ratios are kept.
"""
function paired_ratios(num::AbstractDict, den::AbstractDict)
    out = Float64[]
    for r in _common_repeats(num, den)
        a, b = num[r], den[r]
        (a isa Real && b isa Real && isfinite(a) && isfinite(b) && b != 0) || continue
        q = Float64(a) / Float64(b)
        q > 0 && push!(out, q)
    end
    return out
end

"""
    paired_vector_ratio(sub_num, sub_den, col_num[, col_den]) -> Matrix{Float64}

`L x n_pairs` matrix of `sub_num[r][col_num] ./ sub_den[r][col_den]` for the common repeats
(truncated to the shortest vector); non-finite or non-positive entries become NaN. `col_den`
defaults to `col_num`.
"""
function paired_vector_ratio(sub_num::AbstractDataFrame, sub_den::AbstractDataFrame, col_num::Symbol, col_den::Symbol=col_num)
    a, b = by_repeat(sub_num, col_num), by_repeat(sub_den, col_den)
    reps = [r for r in _common_repeats(a, b) if a[r] isa AbstractVector && b[r] isa AbstractVector &&
                                                !isempty(a[r]) && !isempty(b[r])]
    isempty(reps) && return zeros(Float64, 0, 0)
    L = minimum(min(length(a[r]), length(b[r])) for r in reps)
    M = fill(NaN, L, length(reps))
    for (j, r) in enumerate(reps), i in 1:L
        x, y = a[r][i], b[r][i]
        (x isa Real && y isa Real && isfinite(x) && isfinite(y) && x > 0 && y > 0) && (M[i, j] = Float64(x) / Float64(y))
    end
    return M
end

"`L x n_repeats` matrix of a per-repeat vector column (truncated to the shortest), columns by repeat."
function repeat_matrix(sub::AbstractDataFrame, col::Symbol)
    d = by_repeat(sub, col)
    reps = [r for r in sort!(collect(keys(d))) if d[r] isa AbstractVector && !isempty(d[r])]
    isempty(reps) && return zeros(Float64, 0, 0)
    L = minimum(length(d[r]) for r in reps)
    return reduce(hcat, [Float64.(d[r][1:L]) for r in reps])
end

"Per-repeat cond improvement `kappa(A) / kappa(P^-1 A)`, falling back to the baseline's kappa(A)."
function cond_improvement_by_repeat(g::AbstractDataFrame, b::AbstractDataFrame)
    pc, ac, bc = by_repeat(g, :avg_pc_cond), by_repeat(g, :avg_cond), by_repeat(b, :avg_cond)
    ok(x) = x isa Real && isfinite(x) && x > 0
    out = Float64[]
    for r in sort!(collect(keys(pc)))
        κPC = pc[r]
        ok(κPC) || continue
        κA = get(ac, r, NaN)
        ok(κA) || (κA = get(bc, r, NaN))
        ok(κA) && push!(out, Float64(κA) / Float64(κPC))
    end
    return out
end

# ---------------------------------------------------------------------------------------------
# Row selection and accessors
# ---------------------------------------------------------------------------------------------

"Value of `col` in `row`, or `default` if the column is absent or the entry missing."
function getf(row, col::Symbol, default=NaN)
    hasproperty(row, col) || return default
    v = getproperty(row, col)
    return (ismissing(v) || v === nothing) ? default : v
end

"Statistic `stat` (`:mean`, `:std`, `:sem`, `:median`, `:q25`, `:q75`, `:cv`, ...) of scalar `metric` in a stats row."
sstat(row, metric::Symbol, stat::Symbol) = getf(row, Symbol(metric, :_, stat))

"Vector statistic `stat` (`:mean`, `:std`, `:median`, `:q25`, `:q75`) of a per-step / per-site `metric` in a stats row; empty if absent."
function vstat(row, metric::Symbol, stat::Symbol)
    v = getf(row, Symbol(metric, :_, stat), Float64[])
    return v isa AbstractVector ? Float64.(v) : Float64[]
end

"Rows of the long frame `df` for one system size and configuration key."
sel(df::AbstractDataFrame, size_col::Symbol, sz, key) =
    nrow(df) == 0 ? df : df[Bool[r[size_col] == sz && r.key == key for r in eachrow(df)], :]

"The stats row for one system size and configuration key, or `nothing`."
function stats_row(df_stats::AbstractDataFrame, size_col::Symbol, sz, key)
    for r in eachrow(df_stats)
        (r[size_col] == sz && r.key == key) && return r
    end
    return nothing
end

"Human-readable configuration label of a stats row."
function config_label(row)
    key = String(row.key)
    key == "baseline" && return "Baseline"
    method = string(getf(row, :method, ""))
    if method == "sinc" || startswith(key, "sinc")
        return "Sinc M=$(Int(getf(row, :M, 0)))"
    end
    # FAI built on the separable L0 instead of the operator itself (anisotropy benchmarks)
    prefix = method == "fai_L0" ? "FAI(L0)" : "FAI"
    return "$prefix χ=$(Int(getf(row, :G_maxdim, 0))), μ=$(getf(row, :mu, NaN))"
end

"Sort key giving baseline, then sinc by M, then FAI by (χ, μ), then FAI(L0) by (χ, μ)."
function config_order(row)
    key = String(row.key)
    key == "baseline" && return (0, 0.0, 0.0)
    method = string(getf(row, :method, ""))
    (method == "sinc" || startswith(key, "sinc")) && return (1, Float64(getf(row, :M, 0)), 0.0)
    return (method == "fai_L0" ? 3 : 2, Float64(getf(row, :G_maxdim, 0)), Float64(getf(row, :mu, 0.0)))
end

# ---------------------------------------------------------------------------------------------
# Stats frame
# ---------------------------------------------------------------------------------------------

function _add_ratio!(row::Dict{Symbol, Any}, name::Symbol, samples::AbstractVector)
    st = ratio_stats(samples)
    for (f, v) in pairs(st)
        row[Symbol(name, :_, f)] = v
    end
end

"""
    build_stats_frame(df; size_col, solve_col, total_col, iters_col, scalar_cols, vector_cols,
                      samejob_only=false)

One row per (system size, key). For every scalar metric present in `df`, columns
`<metric>_{mean,std,sem,median,q25,q75,min,max,n,cv}`; for every vector metric,
`<metric>_{mean,std,median,q25,q75,n}` holding vectors; and, for non-baseline rows, per-repeat
PAIRED ratios against the baseline of the same size:

  * `speedup_solve`  = baseline `solve_col` / own `solve_col`
  * `speedup_total`  = baseline `solve_col` / own `total_col`   (includes the preconditioner build)
  * `iters_reduction` = baseline `iters_col` / own `iters_col`
  * `cond_improvement` = kappa(A) / kappa(P^-1 A)               (condition-number runs only)

each summarised as `<ratio>_{gm,gsd,median,q25,q75,mean,cv,n}` (see `ratio_stats`), plus
`n_pairs_samejob`: how many of the baseline pairs ran in the same job (block layout: same node).
With `samejob_only=true` the timing ratios use only those pairs (condition numbers do not depend
on the node and keep every pair).
"""
function build_stats_frame(df::AbstractDataFrame; size_col::Symbol, solve_col::Symbol,
                           total_col::Symbol=:total_time, iters_col::Union{Nothing, Symbol}=nothing,
                           scalar_cols::Vector{Symbol}=Symbol[], vector_cols::Vector{Symbol}=Symbol[],
                           samejob_only::Bool=false)
    out = DataFrame()
    nrow(df) == 0 && return out

    baselines = Dict{Any, SubDataFrame}()
    for g in groupby(df, [size_col, :key])
        g.key[1] == "baseline" && (baselines[g[1, size_col]] = g)
    end

    for g in groupby(df, [size_col, :key])
        sz, key = g[1, size_col], g.key[1]
        row = Dict{Symbol, Any}(size_col => sz, :key => key)
        for c in (:method, :mu, :G_maxdim, :M)
            hasproperty(g, c) && !ismissing(g[1, c]) && (row[c] = g[1, c])
        end
        row[:n_repeats] = length(unique(g.repeat))
        row[:repeats] = sort(collect(g.repeat))
        hosts = hasproperty(g, :hostname) ? unique(skipmissing(g.hostname)) : String[]
        row[:hostnames] = String.(hosts)
        row[:n_hosts] = length(hosts)
        row[:any_stopped_early] = hasproperty(g, :stopped_early) ? any(coalesce.(g.stopped_early, false)) : false

        for c in scalar_cols
            hasproperty(g, c) || continue
            for (f, v) in pairs(scalar_stats(g[!, c]))
                row[Symbol(c, :_, f)] = v
            end
        end
        for c in vector_cols
            hasproperty(g, c) || continue
            vs = vector_stats(g[!, c]; label="$(c) [$(sz), $(key)]")
            for f in (:mean, :std, :median, :q25, :q75, :n)
                row[Symbol(c, :_, f)] = getfield(vs, f)
            end
        end

        if key != "baseline" && haskey(baselines, sz)
            b = baselines[sz]
            same = same_job_repeats(b, g)
            row[:n_pairs_samejob] = length(same)
            # Only blocks can have within-node pairs; for the unit layout this would drop every pair
            keep = (samejob_only && is_block_layout(b) && is_block_layout(g)) ? same : nothing
            br(sub, col) = restrict_repeats(by_repeat(sub, col), keep)
            _add_ratio!(row, :speedup_solve, paired_ratios(br(b, solve_col), br(g, solve_col)))
            _add_ratio!(row, :speedup_total, paired_ratios(br(b, solve_col), br(g, total_col)))
            if iters_col !== nothing
                _add_ratio!(row, :iters_reduction, paired_ratios(br(b, iters_col), br(g, iters_col)))
            end
            if hasproperty(g, :avg_pc_cond)
                _add_ratio!(row, :cond_improvement, cond_improvement_by_repeat(g, b))
            end
        end
        push!(out, row, cols=:union)
    end
    return out
end

# ---------------------------------------------------------------------------------------------
# Sanity checks and console summary
# ---------------------------------------------------------------------------------------------

"""
    check_repeat_sanity(df; size_col, solve_col)

Prints, and fails loudly on the ones that indicate a seeding bug:
  * distinct repeats per task, and tasks with fewer than the rest (warning: partial data is normal
    while a chunked run is in progress);
  * seeds colliding across different repeat indices (ERROR: `solver_seed` is wrong);
  * groups of repeats whose `solve_col` values are bit-identical (loud warning: the seed is not
    reaching the solver);
  * a TT problem that is not identical across repeats (`rhs_norm`/`rhs_maxlinkdim`);
  * block layout: blocks with missing methods, and blocks split over several jobs (refilled after a
    stop or failure), whose pairs are then not all within-node; mixed layouts in one dataset.
"""
function check_repeat_sanity(df::AbstractDataFrame; size_col::Symbol, solve_col::Symbol)
    nrow(df) == 0 && (@warn "No rows collected"; return)
    println("\n=== Repeat sanity checks ===")

    per_task = combine(groupby(df, :task_id), :repeat => (r -> length(unique(r))) => :n_repeats)
    nmax = maximum(per_task.n_repeats)
    println("Distinct repeats per task: min=$(minimum(per_task.n_repeats)) max=$nmax over $(nrow(per_task)) tasks")
    short = per_task[per_task.n_repeats .< nmax, :]
    if nrow(short) > 0
        @warn "Tasks with fewer repeats than the others (job cut by the wall-clock guard, or still running?)" tasks=Tuple.(eachrow(short))
    end

    dups = combine(groupby(df, [:task_id, :repeat]), nrow => :n)
    any(dups.n .> 1) && @warn "Duplicate (task, repeat) rows" n=count(dups.n .> 1)

    nseeds, nreps = length(unique(df.seed)), length(unique(df.repeat))
    println("Distinct seeds: $nseeds, distinct repeat indices: $nreps")
    if nseeds < nreps
        error("SEEDING BUG: only $nseeds distinct seeds for $nreps distinct repeat indices; `solver_seed` collides across repeats.")
    elseif nseeds > nreps
        @warn "More distinct seeds than repeat indices: BENCH_BASE_SEED differed between the collected jobs, so repeat indices from different jobs are not paired"
    end

    for g in groupby(df, [size_col, :key])
        length(g.repeat) >= 2 || continue
        v = [x for x in g[!, solve_col] if x isa Real]
        if length(v) >= 2 && all(==(v[1]), v)
            @warn "!!! All repeats of $(g[1, size_col]) / $(g.key[1]) have BIT-IDENTICAL $solve_col: the per-repeat seed is not reaching the solver, or the result was duplicated !!!" value=v[1]
        end
    end

    if hasproperty(df, :hostname)
        lay = [(_host_layout(String[ismissing(h) ? "" : String(h) for h in g.hostname]), length(g.repeat)) for g in groupby(df, [size_col, :key])]
        ns, nd, nm = count(x -> x[1] == :single, lay), count(x -> x[1] == :distinct, lay), count(x -> x[1] == :mixed, lay)
        println("Host layout per configuration: $ns on a single host, $nd with every repeat on a different host, $nm mixed (of $(length(lay)))")
        nsingle = count(x -> x[1] == :single && x[2] >= 3, lay)
        nsingle > 0 && @warn "$nsingle configuration(s) ran all their (>= 3) repeats on a single host: a node-level slowdown lasts a whole job, so it does not average out over these repeats and is not visible in their spread, and their paired speedups compare runs on different nodes. Use BENCH_LAYOUT=block (see submit.sh)."
    end

    if hasproperty(df, :layout)
        # The layout is chosen per system size (e.g. (8,8) as blocks, (10,10) as units), so only a
        # mix WITHIN one size is a problem.
        for gs in groupby(df, size_col)
            lays = unique(row_layouts(gs))
            println("Layout of $(gs[1, size_col]): $(join(lays, ", "))")
            length(lays) > 1 && @warn "Mixed layouts for $(gs[1, size_col]) ($(join(lays, ", "))): per-task repeats of a configuration share a node, block repeats do not, unit repeats never pair within a node, so their spreads and pairs mean different things. Keep them in separate results folders."
        end
        dfb = df[row_layouts(df) .== "block", :]
        if nrow(dfb) > 0 && hasproperty(df, :slurm_job_id)
            nkeys = Dict(g[1, size_col] => length(unique(g.key)) for g in groupby(dfb, size_col))
            nblocks = nincomplete = nsplit = 0
            for g in groupby(dfb, [size_col, :repeat])
                nblocks += 1
                length(unique(g.key)) < nkeys[g[1, size_col]] && (nincomplete += 1)
                length(unique(string.(skipmissing(g.slurm_job_id)))) > 1 && (nsplit += 1)
            end
            println("Blocks (system size x repeat): $nblocks, $nincomplete with missing methods, $nsplit split over several jobs")
            nincomplete > 0 && @warn "$nincomplete block(s) miss some methods (still running, stopped by the guard, or failed units): their pairs with the missing methods are dropped"
            nsplit > 0 && @warn "$nsplit block(s) ran in several jobs (refilled after a stop or failure): some of their pairs are across nodes; see n_pairs_samejob, or set BENCH_SAMEJOB_PAIRS_ONLY=1"
        end
    end

    for c in (:rhs_norm, :rhs_maxlinkdim)
        hasproperty(df, c) || continue
        for g in groupby(df, size_col)
            vals = unique(skipmissing(g[!, c]))
            length(vals) > 1 && @warn "The problem is not identical across repeats: $c takes $(length(vals)) values for $(g[1, size_col]) (the per-repeat seed reached build_problem, or BLAS nondeterminism)" values=vals
        end
    end
    println("=== end sanity checks ===\n")
end

"Console table: size, key, n, solve mean +- std, CV, mean iterations."
function print_stats_summary(df_stats::AbstractDataFrame; size_col::Symbol, solve_col::Symbol,
                             iters_col::Union{Nothing, Symbol}=nothing)
    nrow(df_stats) == 0 && return
    w = max(10, maximum(length ∘ string, df_stats[!, size_col]) + 2)
    println(rpad("size", w), rpad("configuration", 34), rpad("n", 4), rpad("hosts", 7), rpad("$solve_col mean ± std [s]", 38), rpad("CV", 8), "iters mean")
    # Sizes (Int N, node tuples, (N, F) tuples) sort natively; as strings, 10 would come before 5.
    order = sortperm([(r[size_col], config_order(r)) for r in eachrow(df_stats)])
    for i in order
        r = df_stats[i, :]
        it = iters_col === nothing ? NaN : sstat(r, iters_col, :mean)
        println(rpad(string(r[size_col]), w), rpad(config_label(r), 34), rpad(string(r.n_repeats), 4), rpad(string(getf(r, :n_hosts, "?")), 7),
                rpad(@sprintf("%.4g ± %.2g", sstat(r, solve_col, :mean), sstat(r, solve_col, :std)), 38),
                rpad(@sprintf("%.1f%%", 100 * sstat(r, solve_col, :cv)), 8), @sprintf("%.6g", it))
    end
end

# ---------------------------------------------------------------------------------------------
# Collector driver
# ---------------------------------------------------------------------------------------------

"""
    collect_dataset(base_folder; postprocess, size_col, solve_col, total_col, iters_col,
                    scalar_cols, vector_cols, discard_first=false, samejob_only=false)

Reads `<base_folder>/tasks/task_*_rep_*.jld2`, runs the sanity checks, builds the stats frame and
saves `<base_folder>/collected_results.jld2` with

  * `"df"`         long frame, one row per (task, repeat), the rows the statistics use;
  * `"df_stats"`   one row per (system size, key), see `build_stats_frame`;
  * `"df_dropped"` first-in-process repeats removed by `discard_first` (empty by default);
  * `"df_stats_all"` (only if something was dropped) the statistics before discarding.

NOTE for readers of the file: `"df"` has SEVERAL rows per key. The original directories'
`plot_results.jl` expects one row per key and will produce nonsense from it; use the plot script
in the `*_stats` directory.
"""
function collect_dataset(base_folder::AbstractString; postprocess, size_col::Symbol, solve_col::Symbol,
                         total_col::Symbol=:total_time, iters_col::Union{Nothing, Symbol}=nothing,
                         scalar_cols::Vector{Symbol}, vector_cols::Vector{Symbol}, discard_first::Bool=false,
                         samejob_only::Bool=false)
    df_all = collect_repeat_files(joinpath(base_folder, "tasks"); postprocess)
    nrow(df_all) == 0 && error("No task_*_rep_*.jld2 files found in $(joinpath(base_folder, "tasks"))")
    failed = sort(filter(f -> occursin(r"^task_\d+_rep_\d+\.failed$", f), readdir(joinpath(base_folder, "tasks"))))
    isempty(failed) || @warn "$(length(failed)) repeat(s) FAILED in the workers and are missing from the data (see the *.failed markers for the error); if failures depend on the seed, the surviving repeats are not a random sample of seeds" markers=failed
    check_repeat_sanity(df_all; size_col, solve_col)

    df, dropped = split_first_repeats(df_all; discard=discard_first)
    nrow(dropped) > 0 && @warn "BENCH_DISCARD_FIRST_REPEAT: dropped $(nrow(dropped)) first-in-process repeat(s); the report must say so and show before/after"

    samejob_only && println("BENCH_SAMEJOB_PAIRS_ONLY: speedups use only pairs that ran in the same job (block layout: same node)")
    kw = (; size_col, solve_col, total_col, iters_col, scalar_cols, vector_cols, samejob_only)
    df_stats = build_stats_frame(df; kw...)
    out = Dict{String, Any}("df" => df, "df_stats" => df_stats, "df_dropped" => dropped)
    nrow(dropped) > 0 && (out["df_stats_all"] = build_stats_frame(df_all; kw...))

    output_file = joinpath(base_folder, "collected_results.jld2")
    save(output_file, out)
    println("Collected $(nrow(df)) repeat rows ($(nrow(df_stats)) configurations, $(length(unique(df.task_id))) tasks) into $output_file")
    print_stats_summary(df_stats; size_col, solve_col, iters_col)
    return out
end

# ---------------------------------------------------------------------------------------------
# Host (node) effects
# ---------------------------------------------------------------------------------------------

# Per-repeat machine-side metric of one configuration: time per iteration when every repeat has a
# positive iteration count (the count varies with the seed, the cost of an iteration mostly does
# not), else raw time. Returns (values, unit).
function _machine_metric(g::AbstractDataFrame, solve_col::Symbol, iters_col::Union{Nothing, Symbol})
    t = Float64[x isa Real ? Float64(x) : NaN for x in g[!, solve_col]]
    if iters_col !== nothing && hasproperty(g, iters_col)
        it = Float64[x isa Real ? Float64(x) : NaN for x in g[!, iters_col]]
        all(i -> isfinite(i) && i > 0, it) && return t ./ it, "s/iteration"
    end
    return t, "s"
end

"`:single` (one host), `:distinct` (every repeat on its own host) or `:mixed` (several hosts, some with several repeats)."
function _host_layout(hosts::AbstractVector)
    u = unique(hosts)
    length(u) <= 1 && return :single
    return maximum(count(==(h), hosts) for h in u) == 1 ? :distinct : :mixed
end

"""
    host_effect(g; solve_col, iters_col=nothing, nperm=2000) -> NamedTuple

Node-level effect for ONE configuration (a group of repeats with a `hostname` each). A slowdown that
belongs to a node lasts the whole job, so it does not average out over the repeats that ran on that
node and is invisible in the across-repeat spread unless the repeats span several hosts. What can
be concluded depends on `layout`:

  * `:single`   every repeat ran on one host: the spread EXCLUDES any node effect;
  * `:distinct` every repeat ran on its own host: node effects are part of the spread, but cannot
                be separated from repeat-to-repeat variation;
  * `:mixed`    several hosts, some with several repeats: the host effect is estimated and tested.

For `:mixed`: `spread_cv` = sd of the host means / overall mean, `ratio` = slowest / fastest host
mean, and `p` = permutation p-value of the one-way ANOVA F statistic (repeats shuffled across
hosts `nperm` times; distribution-free, needs no extra package). The metric is time per iteration
when available, so seed-driven iteration-count variation is not mistaken for a node effect.
"""
function host_effect(g::AbstractDataFrame; solve_col::Symbol, iters_col::Union{Nothing, Symbol}=nothing, nperm::Int=2000)
    none = (; layout=:none, n_hosts=0, n=0, per_host=(0, 0), unit="s", spread_cv=NaN, ratio=NaN, slowest="", fastest="", F=NaN, p=NaN)
    hasproperty(g, :hostname) || return none
    y, unit = _machine_metric(g, solve_col, iters_col)
    hosts = String[ismissing(h) ? "" : String(h) for h in g.hostname]
    ok = isfinite.(y) .& .!isempty.(hosts)
    y, hosts = y[ok], hosts[ok]
    isempty(y) && return none
    u = unique(hosts)
    counts = [count(==(h), hosts) for h in u]
    layout = _host_layout(hosts)
    base = (; layout, n_hosts=length(u), n=length(y), per_host=(minimum(counts), maximum(counts)), unit)
    layout == :mixed || return merge(base, (; spread_cv=NaN, ratio=NaN, slowest="", fastest="", F=NaN, p=NaN))

    groups = [findall(==(h), hosts) for h in u]
    function fstat(yy)
        ybar = mean(yy)
        ssb = sum(length(ix) * (mean(yy[ix]) - ybar)^2 for ix in groups)
        ssw = sum(sum((yy[ix] .- mean(yy[ix])) .^ 2) for ix in groups)
        msb, msw = ssb / (length(groups) - 1), ssw / (length(yy) - length(groups))
        return msw > 0 ? msb / msw : (msb > 0 ? Inf : NaN)
    end
    F_obs = fstat(y)
    rng = MersenneTwister(20260918)      # local RNG: deterministic p-values, global stream untouched
    exceed = count(_ -> fstat(y[randperm(rng, length(y))]) >= F_obs * (1 - 1e-9), 1:nperm)
    p = isnan(F_obs) ? 1.0 : (1 + exceed) / (nperm + 1)
    means = [mean(y[ix]) for ix in groups]
    slow, fast = argmax(means), argmin(means)
    return merge(base, (; spread_cv = std(means) / abs(mean(y)), ratio = means[slow] / means[fast],
                          slowest = u[slow], fastest = u[fast], F = F_obs, p))
end

"`host_effect` for every (system size, key) group of the long frame `df`; empty if `hostname` was not recorded."
function host_effects(df::AbstractDataFrame; size_col::Symbol, solve_col::Symbol,
                      iters_col::Union{Nothing, Symbol}=nothing, nperm::Int=2000)
    out = NamedTuple[]
    hasproperty(df, :hostname) || return out
    for g in groupby(df, [size_col, :key])
        push!(out, merge((; sz=g[1, size_col], key=String(g.key[1])), host_effect(g; solve_col, iters_col, nperm)))
    end
    return out
end

"Markdown section with the per-configuration host-effect table (only configurations that can be tested)."
function host_effect_markdown(he::AbstractVector, df_stats::AbstractDataFrame; size_col::Symbol, size_label::Function)
    mixed = [e for e in he if e.layout == :mixed]
    isempty(mixed) && return ""
    io = IOBuffer()
    println(io, "## Host effects\n")
    println(io, "Repeats are grouped by the node they ran on. A slowdown that belongs to a node lasts a whole job, so it does not average out over the repeats on that node; it only becomes visible when a configuration's repeats span several hosts. The metric is $(mixed[1].unit) (the iteration count varies with the seed, the cost of an iteration mostly does not). *p* is a permutation test (repeats shuffled across hosts, $(2000) times) for \"host means differ more than chance\"; with $(length(mixed)) configurations tested, about $(round(0.05 * length(mixed), digits=1)) are expected below 0.05 by chance, so weigh the size of the spread, not only *p*. Configurations that ran every repeat on a different host cannot be tested (the host is aliased with the repeat), but node effects are part of their spread by construction.\n")
    println(io, "| size | configuration | hosts | repeats/host | host spread (sd of host means / mean) | slowest / fastest host | p |")
    println(io, "|---|---|---|---|---|---|---|")
    sorted = sort(mixed, by=e -> (e.sz, let r = stats_row(df_stats, size_col, e.sz, e.key); r === nothing ? (9, 0.0, 0.0) : config_order(r) end))
    for e in sorted
        r = stats_row(df_stats, size_col, e.sz, e.key)
        label = r === nothing ? e.key : config_label(r)
        flag = isfinite(e.p) && e.p < 0.05
        pstr = e.p < 0.001 ? "<0.001" : @sprintf("%.3f", e.p)
        rph = e.per_host[1] == e.per_host[2] ? string(e.per_host[1]) : "$(e.per_host[1])–$(e.per_host[2])"
        row = "| $(size_label(e.sz)) | $label | $(e.n_hosts) | $rph | $(@sprintf("%.1f%%", 100 * e.spread_cv)) | $(@sprintf("%.2f×", e.ratio)) ($(e.slowest) vs $(e.fastest)) | $pstr |"
        println(io, flag ? replace(row, "| $label |" => "| **$label** |") : row)
    end
    println(io)
    return String(take!(io))
end

# ---------------------------------------------------------------------------------------------
# Block (node) effects, block layout
# ---------------------------------------------------------------------------------------------

"Whether every row of `df` was produced by the block layout."
is_block_layout(df::AbstractDataFrame) =
    nrow(df) > 0 && hasproperty(df, :layout) && all(l -> !ismissing(l) && l == "block", df.layout)

"Layout of every row (`task` where unrecorded: old per-task data)."
row_layouts(df::AbstractDataFrame) =
    hasproperty(df, :layout) ? String[ismissing(l) ? "task" : String(l) for l in df.layout] : fill("task", nrow(df))

"System sizes whose rows all come from the block layout."
block_sizes(df::AbstractDataFrame, size_col::Symbol) =
    [gs[1, size_col] for gs in groupby(df, size_col) if is_block_layout(gs)]

# Additive two-way fit of a complete blocks x methods matrix: returns the block effects and the
# residual variance, and the F statistic for "the blocks differ".
function _twoway(M::AbstractMatrix)
    nb, m = size(M)
    grand = mean(M)
    rowm = vec(mean(M; dims=2))
    colm = vec(mean(M; dims=1))
    E = M .- rowm .- colm' .+ grand
    s2_res = sum(abs2, E) / ((nb - 1) * (m - 1))
    be = rowm .- grand
    F = s2_res > 0 ? m * var(be) / s2_res : NaN
    return (; be, s2_res, F)
end

"""
    block_effects(df; size_col, solve_col, iters_col=nothing, nperm=2000) -> Vector{NamedTuple}

Block layout only: sizes from another layout are skipped. For every block-layout system size, the complete blocks (repeat indices at
which every method of that size finished) are decomposed additively,

    log y[block, method] = mu + block effect + method effect + residual,

where y is time per iteration for a method whose iteration counts are all positive, else time (see
`_machine_metric`). The block effect is a slowdown shared by every method of one job: the
node/allocation effect, which cancels in the paired speedups and enters the absolute times. The
residual is what the paired speedups still carry. Reported per size:

  * `sd_block`: sd of the block effects, corrected for the residual noise they contain
    (`sqrt(max(0, var(block effects) - s2_res / n_methods))`);
  * `sd_resid`: residual sd; both on the log scale, so `exp(sd) - 1` is a relative spread;
  * the slowest and fastest block (`exp(effect) - 1`, its repeat index and host(s));
  * `p`: permutation p-value of the F statistic, each method's column shuffled across blocks
    independently (local RNG, so deterministic and the global stream is untouched).
"""
function block_effects(df::AbstractDataFrame; size_col::Symbol, solve_col::Symbol,
                       iters_col::Union{Nothing, Symbol}=nothing, nperm::Int=2000)
    out = NamedTuple[]
    for gs in groupby(df, size_col)
        is_block_layout(gs) || continue
        keys_ = sort(unique(String.(gs.key)))
        Y = Dict{Tuple{Int, String}, Float64}()
        units = String[]
        for g in groupby(gs, :key)
            y, unit = _machine_metric(g, solve_col, iters_col)
            push!(units, unit)
            for (i, r) in enumerate(g.repeat)
                (isfinite(y[i]) && y[i] > 0) && (Y[(Int(r), String(g.key[1]))] = log(y[i]))
            end
        end
        complete = [r for r in sort(unique(Int.(gs.repeat))) if all(k -> haskey(Y, (r, k)), keys_)]
        nb, m = length(complete), length(keys_)
        (nb >= 3 && m >= 2) || continue
        M = [Y[(r, k)] for r in complete, k in keys_]
        fit = _twoway(M)
        s2_block = max(0.0, var(fit.be) - fit.s2_res / m)

        rng = MersenneTwister(20260919)
        Mp = copy(M)
        exceed = 0
        for _ in 1:nperm
            for j in 1:m
                Mp[:, j] = M[randperm(rng, nb), j]
            end
            _twoway(Mp).F >= fit.F * (1 - 1e-9) && (exceed += 1)
        end
        p = isfinite(fit.F) ? (1 + exceed) / (nperm + 1) : NaN

        hosts_of(r) = hasproperty(gs, :hostname) ?
            join(sort(unique(String[string(h) for h in gs.hostname[gs.repeat .== r] if !ismissing(h)])), "+") : ""
        slow, fast = argmax(fit.be), argmin(fit.be)
        push!(out, (; sz=gs[1, size_col], n_blocks=nb, n_methods=m,
                      unit = all(==("s/iteration"), units) ? "time per iteration" : "time per iteration (time where no iteration count)",
                      sd_block=sqrt(s2_block), sd_resid=sqrt(fit.s2_res), F=fit.F, p,
                      slowest=(repeat=complete[slow], effect=exp(fit.be[slow]) - 1, host=hostname_or_empty(hosts_of(complete[slow]))),
                      fastest=(repeat=complete[fast], effect=exp(fit.be[fast]) - 1, host=hostname_or_empty(hosts_of(complete[fast]))),
                      effects=[(repeat=complete[i], effect=exp(fit.be[i]) - 1, host=hosts_of(complete[i])) for i in 1:nb]))
    end
    return out
end

hostname_or_empty(h::AbstractString) = isempty(h) ? "?" : h

"Markdown section with the block-effect table (empty unless the data come from the block layout)."
function block_effects_markdown(be::AbstractVector; size_label::Function)
    isempty(be) && return ""
    io = IOBuffer()
    println(io, "## Node effects (blocks)\n")
    println(io, "Each block ran every method of one system size at one repeat index in one job, on one node. Decomposing log($(be[1].unit)) additively into block + method + residual separates a slowdown shared by all methods of a job (the node/allocation effect: it cancels in the paired speedups, and it is part of the spread of the absolute times) from method-specific noise (the residual: what the speedup bands carry). Only complete blocks are used. *p* is a permutation test for \"blocks differ more than chance\" (each method's values shuffled across blocks, 2000 times).\n")
    println(io, "| size | blocks | methods | node effect (sd) | residual (sd) | slowest block | fastest block | p |")
    println(io, "|---|---|---|---|---|---|---|---|")
    for e in sort(be, by=e -> e.sz)
        pstr = !isfinite(e.p) ? "n/a" : (e.p < 0.001 ? "<0.001" : @sprintf("%.3f", e.p))
        println(io, @sprintf("| %s | %d | %d | %.1f%% | %.1f%% | rep %d: %+.1f%% (%s) | rep %d: %+.1f%% (%s) | %s |",
                             size_label(e.sz), e.n_blocks, e.n_methods, 100 * (exp(e.sd_block) - 1), 100 * (exp(e.sd_resid) - 1),
                             e.slowest.repeat, 100 * e.slowest.effect, e.slowest.host, e.fastest.repeat, 100 * e.fastest.effect, e.fastest.host, pstr))
    end
    println(io)
    return String(take!(io))
end

# ---------------------------------------------------------------------------------------------
# variability_summary.md
# ---------------------------------------------------------------------------------------------

function _pearson(x::AbstractVector, y::AbstractVector)
    (length(x) >= 3 && std(x) > 0 && std(y) > 0) || return NaN
    return cor(x, y)
end

_fmt_pm(m, s; digits=3) = isfinite(m) ? @sprintf("%.*g ± %.2g", digits, m, s) : "n/a"
_fmt_gm(gm, gsd, q25, q75) = isfinite(gm) ? @sprintf("%.2f (×/÷%.2f; IQR %.2f–%.2f)", gm, gsd, q25, q75) : "n/a"

"""
    cond_summary_markdown(df_stats_cond; size_col, size_label) -> String

Markdown section with the spread of the condition-number metrics (algorithmic variability) from a
condition-number run's stats frame.
"""
function cond_summary_markdown(df_stats_cond::AbstractDataFrame; size_col::Symbol, size_label::Function)
    (nrow(df_stats_cond) == 0 || !hasproperty(df_stats_cond, :avg_pc_cond_mean)) && return ""
    io = IOBuffer()
    println(io, "## Condition numbers (condition-number run)\n")
    println(io, "| size | configuration | n | κ(P⁻¹A) mean ± sd | CV | κ(A)/κ(P⁻¹A) paired (gm, IQR) |")
    println(io, "|---|---|---|---|---|---|")
    rows = sort(collect(eachrow(df_stats_cond)), by=r -> (string(r[size_col]), config_order(r)))
    for r in rows
        r.key == "baseline" && continue
        println(io, "| $(size_label(r[size_col])) | $(config_label(r)) | $(r.n_repeats) | $(_fmt_pm(sstat(r, :avg_pc_cond, :mean), sstat(r, :avg_pc_cond, :std))) | $(@sprintf("%.1f%%", 100 * sstat(r, :avg_pc_cond, :cv))) | $(_fmt_gm(sstat(r, :cond_improvement, :gm), sstat(r, :cond_improvement, :gsd), sstat(r, :cond_improvement, :q25), sstat(r, :cond_improvement, :q75))) |")
    end
    println(io)
    return String(take!(io))
end

"""
    write_variability_summary(path; bench, df, df_stats, size_col, size_label, solve_col, iters_col,
                              dataset_note, df_stats_all=nothing, n_discarded=0)

Writes the markdown that makes the numbers defensible: a prose paragraph (how many repeats,
whether any job hit the wall-clock guard, how many distinct hostnames the repeats ran on and how
they are spread over configurations, a per-host effect test, and whether timing spread tracks
iteration spread or is machine-only) followed by one table per system size. `df` is the long
frame the stats were computed from.
"""
function write_variability_summary(path::AbstractString; bench::AbstractString, df::AbstractDataFrame,
                                   df_stats::AbstractDataFrame, size_col::Symbol, size_label::Function,
                                   solve_col::Symbol, iters_col::Union{Nothing, Symbol},
                                   dataset_note::AbstractString="", extra_markdown::AbstractString="",
                                   df_stats_all::Union{Nothing, AbstractDataFrame}=nothing, n_discarded::Int=0)
    io = IOBuffer()
    println(io, "# Variability summary: $bench\n")
    isempty(dataset_note) || println(io, dataset_note, "\n")

    nrep = df_stats.n_repeats
    hosts = hasproperty(df, :hostname) ? unique(skipmissing(df.hostname)) : String[]
    n_stopped = hasproperty(df, :stopped_early) ? length(unique(df.task_id[coalesce.(df.stopped_early, false) .== true])) : 0
    n_tasks = length(unique(df.task_id))

    # Time vs iteration spread, per configuration.
    varying_corr, varying_cv_t, identical_cv_t = Float64[], Float64[], Float64[]
    n_varying = n_identical = 0
    if iters_col !== nothing && hasproperty(df, iters_col)
        for g in groupby(df, [size_col, :key])
            t = _finite_floats(g[!, solve_col]); it = _finite_floats(g[!, iters_col])
            (length(t) >= 5 && length(it) == length(t)) || continue   # a correlation over fewer points means nothing
            cv_t = mean(t) == 0 ? NaN : std(t) / abs(mean(t))
            if std(it) > 0
                n_varying += 1
                push!(varying_cv_t, cv_t)
                c = _pearson(t, it); isfinite(c) && push!(varying_corr, c)
            else
                n_identical += 1
                push!(identical_cv_t, cv_t)
            end
        end
    end
    all_cv = filter(isfinite, [sstat(r, solve_col, :cv) for r in eachrow(df_stats)])

    println(io, "## Summary\n")
    rep_txt = minimum(nrep) == maximum(nrep) ? "$(minimum(nrep)) per configuration" : "$(minimum(nrep))–$(maximum(nrep)) per configuration (median $(round(Int, median(nrep))))"
    println(io, "* **Repeats:** $rep_txt, $(nrow(df)) result rows from $n_tasks tasks. Per-repeat seeds are `solver_seed(repeat)`; the problem is held fixed, so the spread below is solver randomness plus machine noise.")
    println(io, "* **Wall-clock guard:** " * (n_stopped == 0 ? "no job stopped early." : "**$n_stopped of $n_tasks tasks stopped early** (`stopped_early`); they have fewer repeats than requested, and their paired statistics use the repeats common to both sides."))
    he = host_effects(df; size_col, solve_col, iters_col)
    host_txt = "* **Hosts:** the repeats ran on $(length(hosts)) distinct hostname(s)."
    if !isempty(he)
        n_single, n_distinct, n_mixed = (count(e -> e.layout == l, he) for l in (:single, :distinct, :mixed))
        host_txt *= " Per configuration: $n_single ran every repeat on a single host, $n_distinct ran every repeat on a different host, $n_mixed ran on several hosts with more than one repeat on some."
        n_single > 0 && (host_txt *= " A node-level slowdown lasts a whole job and does not average out over repeats on the same node, so for the single-host configurations it is NOT visible in the spreads below: it would shift their means and paired speedups as a whole, and their \"paired\" speedups compare a baseline and a configuration that ran on different nodes. The block layout (`BENCH_LAYOUT=block`, see the header of submit.sh) fixes both.")
    end
    println(io, host_txt)
    block = is_block_layout(df)
    bsizes = block_sizes(df, size_col)
    usizes = [gs[1, size_col] for gs in groupby(df, size_col) if all(==("unit"), row_layouts(gs))]
    bfx = block_effects(df; size_col, solve_col, iters_col)
    if !isempty(bsizes)
        brows = [r for r in eachrow(df_stats) if r.key != "baseline" && r[size_col] in bsizes]
        npairs = [Int(getf(r, :speedup_solve_n, 0)) for r in brows]
        nsame = [Int(getf(r, :n_pairs_samejob, 0)) for r in brows]
        println(io, "* **Layout: blocks" * (block ? "" : " for $(join(size_label.(bsizes), ", "))") * ".** Every job ran all methods of one system size at one repeat index, on one node, so each paired speedup is a within-node ratio: $(sum(nsame; init=0)) of $(sum(npairs; init=0)) speedup pairs ran in the same job as their baseline" *
                    (sum(nsame; init=0) < sum(npairs; init=0) ? " (the rest come from blocks refilled in a later job, after a stop or a failure)." : "."))
    end
    if !isempty(usizes)
        println(io, "* **Layout: single units for $(join(size_label.(usizes), ", ")).** Every (method, repeat) ran in its own job, on its own node, so the paired speedups of these sizes compare runs on different nodes: the node effect does not bias them but widens their spread, and it cannot be separated from method noise (no block effects for these sizes).")
    end
    if !isempty(bfx)
        sb = [100 * (exp(e.sd_block) - 1) for e in bfx]
        sr = [100 * (exp(e.sd_resid) - 1) for e in bfx]
        worst = bfx[argmax(sb)]
        pct_range(v) = round(minimum(v), digits=1) == round(maximum(v), digits=1) ? @sprintf("%.1f%%", minimum(v)) : @sprintf("%.1f%%–%.1f%%", minimum(v), maximum(v))
        println(io, @sprintf("* **Node effects:** a slowdown shared by every method of a job has an sd of %s across blocks (largest at %s: slowest block %+.1f%% on %s); the method-specific residual is %s. The first cancels in the paired speedups and is included in the spread of the absolute times. Details below.",
                             pct_range(sb), size_label(worst.sz), 100 * worst.slowest.effect, worst.slowest.host, pct_range(sr)))
    end
    mixed = [e for e in he if e.layout == :mixed]
    if !isempty(mixed)
        nflag = count(e -> isfinite(e.p) && e.p < 0.05, mixed)
        sp = [e.spread_cv for e in mixed if isfinite(e.spread_cv)]
        worst = mixed[argmax([isfinite(e.spread_cv) ? e.spread_cv : -Inf for e in mixed])]
        wr = stats_row(df_stats, size_col, worst.sz, worst.key)
        println(io, @sprintf("* **Host effect** (%s, configurations with several repeats per host): host means differ beyond chance (permutation p < 0.05) in %d of %d configurations (about %.1f expected by chance); the spread of host means has median %.1f%% and maximum %.1f%% (%s: %s vs %s). Details below.",
                             mixed[1].unit, nflag, length(mixed), 0.05 * length(mixed), 100 * median(sp), 100 * maximum(sp),
                             wr === nothing ? worst.key : config_label(wr), worst.slowest, worst.fastest))
    end
    if !isempty(all_cv)
        println(io, @sprintf("* **Solve-time CV** (%s): median %.1f%%, max %.1f%% over configurations.", solve_col, 100 * median(all_cv), 100 * maximum(all_cv)))
    end
    if n_varying + n_identical == 0 && iters_col !== nothing
        println(io, "* **Timing vs. iterations:** not assessed (fewer than 5 repeats per configuration).")
    end
    if n_varying + n_identical > 0
        s = "* **Timing vs. iterations:** "
        if n_varying > 0
            corr_txt = isempty(varying_corr) ? "the per-repeat correlation is undefined" :
                       @sprintf("the median Pearson correlation between solve time and iteration count across repeats is %.2f", median(varying_corr))
            verdict = isempty(varying_corr) ? "" : (median(varying_corr) >= 0.7 ? ", so the timing spread largely tracks the iteration spread (algorithmic variability)" :
                                                    ", so the timing spread is only partly explained by iteration spread (machine noise contributes)")
            s *= "$n_varying of $(n_varying + n_identical) configurations show iteration counts that vary across repeats (random initial guess and/or randomized preconditioner); $corr_txt$verdict. "
        end
        if n_identical > 0
            s *= @sprintf("%d configuration(s) have identical iteration counts in every repeat, so their time spread is machine variability only (median CV %.1f%%); that is the noise floor the other spreads should be judged against.", n_identical, 100 * median(filter(isfinite, identical_cv_t)))
        end
        println(io, s)
    end
    if n_discarded > 0
        println(io, "* **First repeats discarded:** $n_discarded first-in-process repeat(s) were dropped before computing the statistics (`BENCH_DISCARD_FIRST_REPEAT`). Before/after is shown at the end.")
    end
    println(io)

    sizes = sort(unique(df_stats[!, size_col]))
    for sz in sizes
        println(io, "## $(size_label(sz))\n")
        println(io, "| configuration | n | solve time mean ± sd [s] | CV | iterations mean ± sd | paired solve speedup | paired total speedup |")
        println(io, "|---|---|---|---|---|---|---|")
        rows = [r for r in eachrow(df_stats) if r[size_col] == sz]
        sort!(rows, by=config_order)
        for r in rows
            it = iters_col === nothing ? "n/a" : _fmt_pm(sstat(r, iters_col, :mean), sstat(r, iters_col, :std); digits=5)
            sp_s = r.key == "baseline" ? "1 (reference)" : _fmt_gm(sstat(r, :speedup_solve, :gm), sstat(r, :speedup_solve, :gsd), sstat(r, :speedup_solve, :q25), sstat(r, :speedup_solve, :q75))
            sp_t = r.key == "baseline" ? "1 (reference)" : _fmt_gm(sstat(r, :speedup_total, :gm), sstat(r, :speedup_total, :gsd), sstat(r, :speedup_total, :q25), sstat(r, :speedup_total, :q75))
            println(io, "| $(config_label(r)) | $(r.n_repeats) | $(_fmt_pm(sstat(r, solve_col, :mean), sstat(r, solve_col, :std))) | $(@sprintf("%.1f%%", 100 * sstat(r, solve_col, :cv))) | $it | $sp_s | $sp_t |")
        end
        println(io)
    end

    pair_txt = block ? "same block: same job, same node, same random initial guess" :
               !isempty(bsizes) ? "same repeat index, same random initial guess; for the block-layout sizes also the same job and node" :
               "same repeat index, same random initial guess"
    println(io, "Speedups are per-repeat ratios (baseline / configuration at the $pair_txt), summarised by their geometric mean; `×/÷gsd` is the geometric standard deviation and IQR the interquartile range. Do not recompute them as a ratio of the means above.\n")
    println(io, "The best configuration in a sweep is picked from noisy estimates, so a \"max speedup\" over configurations is biased upward by selection; compare it with the per-configuration bands.\n")
    bfx_md = block_effects_markdown(bfx; size_label)
    isempty(bfx_md) || println(io, bfx_md)
    host_md = host_effect_markdown(he, df_stats; size_col, size_label)
    isempty(host_md) || println(io, host_md)
    isempty(extra_markdown) || println(io, extra_markdown)

    if df_stats_all !== nothing && n_discarded > 0
        println(io, "## Effect of discarding first repeats\n")
        println(io, "| size | configuration | n before → after | solve time mean before [s] | mean after [s] | CV before → after |")
        println(io, "|---|---|---|---|---|---|")
        for r in eachrow(df_stats)
            ra = stats_row(df_stats_all, size_col, r[size_col], r.key)
            ra === nothing && continue
            println(io, "| $(size_label(r[size_col])) | $(config_label(r)) | $(ra.n_repeats) → $(r.n_repeats) | $(@sprintf("%.4g", sstat(ra, solve_col, :mean))) | $(@sprintf("%.4g", sstat(r, solve_col, :mean))) | $(@sprintf("%.1f%% → %.1f%%", 100 * sstat(ra, solve_col, :cv), 100 * sstat(r, solve_col, :cv))) |")
        end
        println(io)
    end

    mkpath(dirname(path))
    write(path, String(take!(io)))
    @info "Wrote variability summary" path
    return path
end

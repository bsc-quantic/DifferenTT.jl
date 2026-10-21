"""
    Linsolve preconditioner SPD benchmark: single worker

Shows whether the TT preconditioner of `linsolve_preconditioner` (the TT solution X of AX + XA = 2I, truncated to
bond dimension χ) is symmetric positive definite, for the pressure Poisson operator of `chorin_pipeline` (2D QTT).

Sections (all of them run by default; select some with `LINSOLVE_SPD_SECTIONS=a,b`):

- `chorin_linsolve_dense`: small systems. linsolve preconditioner for each χ, contracted to a
  dense matrix and diagonalized exactly.
- `chorin_truncinv_dense`: same systems. The exact A⁻¹, converted to an MPO and SVD
  truncated to bond dimension χ. Shows that the indefiniteness comes from the TT truncation itself, not from the
  solver that builds the linsolve preconditioner.
- `chorin_linsolve_large`: the systems of the benchmark, too large for dense matrices.
  The first solve of the benchmark (the first Chorin step) runs with `spd_tol` set, whose
  local SPD checks stop the solve with a `NotSPDError` as soon as a local P⁻¹A has an eigenvalue with real part
  below -SPD_TOL · max|λ|. The preconditioner is then marked as not SPD. The local blocks are congruences of the
  global operators (Sylvester's law of inertia), so a negative local eigenvalue proves the global preconditioner is
  indefinite. The converse does not hold: a solve that runs through only shows P⁻¹A is positive on the visited bases.

In the dense analyses, the symmetric part S of the dense preconditioner P is diagonalized (it alone sets the sign of
xᵀPx), and the spectrum of S·A is found as the spectrum of the symmetric LᵀSL, with A = LLᵀ.

All results go to a single JLD2 file, one entry per case written as soon as the case finishes. Entries already in the
file are skipped, so a timed-out job can be resubmitted and continues where it stopped.

Usage: julia --project=experiments experiments/paper/linsolve_spd/worker.jl
"""

using ITensors
using ITensorMPS
using LinearAlgebra
using Statistics
using JLD2
using DifferenTT
using Random
using Dates
using Printf

const cpus_per_task = parse(Int, get(ENV, "SLURM_CPUS_PER_TASK", string(LinearAlgebra.BLAS.get_num_threads())))
LinearAlgebra.BLAS.set_num_threads(cpus_per_task)

ITensors.disable_warn_order()
ENV["JULIA_DEBUG"] = ""

const T_START = time()

"""
Timestamped progress line on stdout, flushed so it shows up in the SLURM log right away.
"""
function progress(msg::AbstractString)
    elapsed = round(Int, time() - T_START)
    @printf(
        "[%s | +%02d:%02d:%02d] %s\n",
        Dates.format(now(), "HH:MM:SS"),
        elapsed ÷ 3600,
        (elapsed ÷ 60) % 60,
        elapsed % 60,
        msg
    )
    flush(stdout)
    flush(stderr)
end

progress(
    "Runtime: blas_threads=$(LinearAlgebra.BLAS.get_num_threads()) julia_threads=$(Threads.nthreads()) host=$(gethostname())",
)

# ---- Configuration ----
const BASE_FOLDER = @__DIR__
const DEFAULT_RESULTS_FILE = joinpath(BASE_FOLDER, "results", "linsolve_spd.jld2")
const RESULTS_FILE = get(ENV, "LINSOLVE_SPD_RESULTS", DEFAULT_RESULTS_FILE)

const ALL_SECTIONS = ["chorin_linsolve_dense", "chorin_truncinv_dense", "chorin_linsolve_large"]
const SECTIONS = split(get(ENV, "LINSOLVE_SPD_SECTIONS", join(ALL_SECTIONS, ",")), ",")
all(in(ALL_SECTIONS), SECTIONS) || error("Unknown section in $SECTIONS, expected a subset of $ALL_SECTIONS")

# Bond dimensions of the preconditioners. The truncated inverse also adds its full (lossless) bond dimension.
const DENSE_MAXDIMS = collect(1:15)
const TRUNC_MAXDIMS = collect(1:15)
# The exact A⁻¹ MPO drops only the rounding noise of the dense inverse (ITensor cutoffs are on the squared singular
# values, so this is a relative Frobenius error of about 1e-14)
const EXACT_INV_CUTOFF = 1e-28

# Sweeps of every preconditioner build (linsolve and FAI)
const PRECOND_NSWEEPS = 10

# linsolve_preconditioner build. The residual target is below what a small χ can reach, so the build runs all its
# sweeps and returns the best preconditioner of each χ (the indefiniteness cannot be blamed on a loose tolerance).
const LS_RESIDUAL_TOL = 1e-6

# Relative tolerance of the local SPD checks of the large solves: a local P⁻¹A is not SPD when an eigenvalue has real
# part below -SPD_TOL · max|λ|. Far above the rounding level of the dense eigensolvers (≈ n·eps ≈ 1e-12 for n ≈ 8000),
# and far below the λ_min/λ_max ≈ -0.5 of the indefinite small cases.
const SPD_TOL = 1e-8

# -- Chorin (pressure Poisson operator of chorin_pipeline) --
const BOX_SIZE = (2.0, 1.0)
const RE_NUM = 3000
const STENCIL_ORDER = 8
const MASK_FILE = joinpath(@__DIR__, "..", "..", "fai_benchmarks", "chorin_pipeline", "masks.jld2")

const CHORIN_DENSE_NODES = [(4, 4), (5, 5), (6, 6), (7, 7)]
const CHORIN_LARGE_NODES = [(8, 8), (10, 10)]
const CHORIN_LARGE_MAXDIMS = [8, 10, 12, 14, 16, 18, 20]
# FAI is SPD by construction (Glᵀ Gl): a control for the local spectra of the first Chorin step. Empty to skip.
const CHORIN_FAI_CONTROL = [(1e7, 4)]
const CHORIN_FAI_TOL = 0.1

# Same values as chorin_pipeline/worker.jl
const dt_map = Dict((8, 8) => 0.001, (10, 10) => 0.0002)
const chorin_maxdims_map = Dict((8, 8) => 150, (10, 10) => 200)
const CHORIN_TOL = 1e-3
# The first Chorin step converges in a few sweeps with an SPD preconditioner. The cap only matters if the linsolve
# preconditioner keeps it from converging. The SPD checks diagonalize every local block (dense, nonsymmetric), which is
# affordable here because the first-step solution has χ ≈ 40-50 (blocks up to ≈ 8000 rows).
const CHORIN_SPECTRAL_NSWEEPS = 10

# ---- Results file ----
mkpath(dirname(RESULTS_FILE))

"""
Whether `key` is already in the results file (the case is then skipped).
"""
function has_result(key::AbstractString)
    isfile(RESULTS_FILE) || return false
    return jldopen(f -> haskey(f, key), RESULTS_FILE, "r")
end

"""
Write one case to the results file right away, so that it survives a timeout.
"""
function save_result(key::AbstractString, result::AbstractDict)
    jldopen(RESULTS_FILE, "a+") do f
        f[key] = result
    end
    progress("Saved $key")
end

"""
Run `f()` and save its result under `key`, unless `key` is already saved.
"""
function run_case(f, key::AbstractString)
    if has_result(key)
        progress("Skipping $key (already in $(RESULTS_FILE))")
        return nothing
    end
    progress("Starting $key")
    GC.gc(true)
    result = f()
    result["hostname"] = gethostname()
    result["slurm_job_id"] = get(ENV, "SLURM_JOB_ID", "")
    result["blas_threads"] = LinearAlgebra.BLAS.get_num_threads()
    save_result(key, result)
end

nodes_key(nodes::NTuple{2,Int}) = "nodes_$(nodes[1])x$(nodes[2])"
chi_key(χ::Int) = "chi_$(lpad(χ, 3, '0'))"

# ---- Dense helpers ----

operator_sites(A::MPO) = only.(siteinds(A; plev=0))

"""
Dense matrix of an MPO: rows on the primed (output) site indices, columns on the unprimed (input) ones.
"""
function dense_matrix(mpo::MPO, sites::Vector{<:Index})
    n = prod(dim.(sites))
    return reshape(Array(prod(mpo), prime.(sites)..., sites...), n, n)
end

"""
Inverse of `dense_matrix`: an MPO from a dense matrix, by successive SVDs (lossless for `cutoff = 0`).
"""
function dense_to_mpo(M::AbstractMatrix, sites::Vector{<:Index}; cutoff::Float64=0.0)
    T = ITensor(reshape(M, (dim.(sites)..., dim.(sites)...)), prime.(sites)..., sites...)
    return MPO(T, sites; cutoff=cutoff)
end

symmetric_part(M::AbstractMatrix) = Symmetric((M + transpose(M)) / 2)
relative_asymmetry(M::AbstractMatrix) = norm(M - transpose(M)) / norm(M)
# Negative eigenvalues beyond the rounding level of a dense eigensolver
count_negative(λ) = count(<(-length(λ) * eps(Float64) * maximum(abs, λ)), λ)
spectral_cond(λ) = maximum(abs, λ) / minimum(abs, λ)

"""
Spectral data of the dense operator A, shared by the dense analyses of its preconditioners.
"""
function dense_operator_info(A::MPO)
    sites = operator_sites(A)
    Ad = dense_matrix(A, sites)
    λA = eigvals(symmetric_part(Ad))
    first(λA) > 0 || error("Operator A is not SPD (λ_min = $(first(λA)))")
    return (; sites, Ad, chol=cholesky(symmetric_part(Ad)), λA, Ainv=inv(Ad))
end

"""
    dense_analysis(P, op) -> Dict

Spectra of the symmetric part S of the dense preconditioner `P` and of S·A (as the symmetric LᵀSL, A = LLᵀ), plus
how far `P` is from A⁻¹. By Weyl's inequality λ_min(S) ≥ λ_min(A⁻¹) - ‖S - A⁻¹‖₂, so S can only lose positive
definiteness when the approximation error exceeds λ_min(A⁻¹) = 1/λ_max(A).

Works in place on two n×n buffers (S and a scratch W): at 7x7 nodes each dense matrix is 2 GiB.
"""
function dense_analysis(P::AbstractMatrix, op)
    n = size(P, 1)
    S = similar(P)
    S .= (P .+ transpose(P)) ./ 2
    W = similar(P)
    # P - Pᵀ = 2(P - S)
    W .= P .- S
    asym_P = 2 * norm(W) / norm(P)
    W .= S
    λS = eigvals!(Symmetric(W))
    # LᵀSL
    L = op.chol.L
    W .= S
    rmul!(W, L)
    lmul!(transpose(L), W)
    λSA = eigvals!(Symmetric(W))
    W .= S .- (op.Ainv .+ transpose(op.Ainv)) ./ 2
    λerr = eigvals!(Symmetric(W))
    # AP + PA - 2I
    mul!(W, op.Ad, P)
    mul!(W, P, op.Ad, true, true)
    W[diagind(W)] .-= 2
    sylvester_residual = norm(W) / (2 * sqrt(n))
    W .= P .- op.Ainv
    inverse_rel_error = norm(W) / norm(op.Ainv)
    return Dict{String,Any}(
        "n" => n,
        "asym_P" => asym_P,
        "eigs_S" => λS,
        "eigmin_S" => first(λS),
        "eigmax_S" => last(λS),
        "nneg_S" => count_negative(λS),
        "eigs_SA" => λSA,
        "eigmin_SA" => first(λSA),
        "eigmax_SA" => last(λSA),
        "nneg_SA" => count_negative(λSA),
        "cond_SA" => spectral_cond(λSA),
        # ‖AP + PA - 2I‖_F / ‖2I‖_F: the residual of the system that linsolve_preconditioner solves
        "sylvester_residual" => sylvester_residual,
        "inverse_rel_error_fro" => inverse_rel_error,
        "sym_error_2norm" => maximum(abs, λerr),
        "eigmin_Ainv" => 1 / last(op.λA),
    )
end

function operator_summary(op)
    return Dict{String,Any}(
        "n" => length(op.λA),
        "eigs_A" => op.λA,
        "eigmin_A" => first(op.λA),
        "eigmax_A" => last(op.λA),
        "cond_A" => last(op.λA) / first(op.λA),
        "asym_A" => relative_asymmetry(op.Ad),
        "eigmin_Ainv" => 1 / last(op.λA),
        "eigmax_Ainv" => 1 / first(op.λA),
    )
end

# ---- TT helpers ----

"""
linsolve_preconditioner of `A` at bond dimension `χ`, with its build time.
"""
function build_linsolve(A::MPO, χ::Int; nsweeps::Int, residual_tol::Float64, rtol::Float64=1e-9, maxiter::Int=3000)
    updater_kwargs = LocalUpdaterArgs(; ishermitian=true, isposdef=true, rtol=rtol, krylovdim=1, maxiter=maxiter)
    solver = ALS(; sweep_kwargs=SweepingArgs(; nsweeps=nsweeps, maxdim=χ, cutoff=1e-12), updater_kwargs=updater_kwargs)
    # linsolve_preconditioner draws a random initial guess
    Random.seed!(χ)
    stats = @timed linsolve_preconditioner(A, solver; residual_tol=residual_tol)
    return stats.value, stats.time
end

"""
‖AX + XA - 2I‖_F / ‖2I‖_F in TT format, without dense matrices.
"""
function sylvester_residual_tt(A::MPO, X::MPO)
    sites = operator_sites(A)
    twoI = 2.0 * identity_mpo(sites)
    AX = apply(A, X; alg="naive", truncate=false)
    XA = apply(X, A; alg="naive", truncate=false)
    R = +(+(AX, XA; alg="directsum"), -1.0 * twoI; alg="directsum")
    return norm(R) / norm(twoI)
end

function mpo_info(P::MPO)
    return Dict{String,Any}("linkdims" => collect(linkdims(P)), "maxlinkdim" => maxlinkdim(P))
end

# ---- Problems ----

const bcsp = BoundaryConditions(
    BoundaryCond1d(NeumannBC(0.0), DirichletBC(1.0)), BoundaryCond1d(PeriodicBC(), PeriodicBC())
)

"""
Pressure Poisson operator A = -Δ of chorin_pipeline (it does not depend on the mask or on the velocity).
"""
function chorin_operator(nodes::NTuple{2,Int})
    sites = siteinds(2, sum(nodes))
    arch = QTTArchitecture(nodes; interleaved=false)
    grid = Grid(arch, BOX_SIZE)
    p_field = ScalarField(zeros_mps(sites), arch, bcsp)
    return -1.0 * symmetrized_laplacian(p_field, grid; stencil_order=STENCIL_ORDER)
end

"""
Full Chorin setup of chorin_pipeline/worker.jl (copied from its `build_grid`), to build the first-step RHS.
"""
function build_chorin_grid(nodes::NTuple{2,Int})
    sites = siteinds(2, sum(nodes))
    arch = QTTArchitecture(nodes; interleaved=false)
    grid = Grid(arch, BOX_SIZE)

    mask_key = "mask_$(nodes[1])_$(nodes[2])"
    if isfile(MASK_FILE) && jldopen(f -> haskey(f, mask_key), MASK_FILE, "r")
        object_mask = load(MASK_FILE, mask_key)
        siteinds(object_mask) != sites && replace_siteinds!(object_mask, sites)
    else
        object_mask = ones_mps(sites)
        @warn "mask not found; using ones_mps (all fluid)"
    end

    mps_vx = tt_mul(object_mask, ones_mps(sites))
    mps_vy = 0.0 * ones_mps(sites)

    bcsvx = BoundaryConditions(
        BoundaryCond1d(DirichletBC(1.0), NeumannBC(0.0)), BoundaryCond1d(PeriodicBC(), PeriodicBC())
    )
    bcsvy = BoundaryConditions(
        BoundaryCond1d(DirichletBC(0.0), NeumannBC(0.0)), BoundaryCond1d(PeriodicBC(), PeriodicBC())
    )

    v_field = VectorField((mps_vx, mps_vy), arch, (bcsvx, bcsvy))
    p_field = ScalarField(zeros_mps(sites), arch, bcsp)

    maxdim = chorin_maxdims_map[nodes]
    experiment = INSExperiment(
        grid, v_field, p_field, RE_NUM, object_mask; stencil_order=STENCIL_ORDER, maxdim=maxdim, cutoff=1e-11
    )

    u_diff_ops = DifferentialOps(grid, experiment.u_fields; stencil_order=STENCIL_ORDER)
    u_boundary_null = BoundaryNull(experiment.u_fields; stencil_order=STENCIL_ORDER)
    p_diff_ops = DifferentialOpsScalar(grid, experiment.p_field; stencil_order=STENCIL_ORDER)
    A = -1.0 * symmetrized_laplacian(experiment.p_field, experiment.grid; stencil_order=STENCIL_ORDER)

    regularize = DifferenTT.regularization_check(experiment)
    regularize && error("Benchmark operator A is not regularized; pressure BCs need a Dirichlet boundary")
    cache = ChorinCache(u_diff_ops, p_diff_ops, u_boundary_null, A, regularize)

    return (; nodes, A, experiment, cache, object_mask, sites, dt=dt_map[nodes])
end

"""
RHS and initial guess of the first Chorin step, as in `chorin_step!` of chorin_pipeline/worker.jl.
"""
function chorin_first_step_system(cgrid)
    experiment = cgrid.experiment
    prev_u = last(experiment.u_fields)
    masked_u = DifferenTT.mask_correction(prev_u, cgrid.object_mask; cutoff=experiment.cutoff, maxdim=experiment.maxdim)
    rhs = DifferenTT.poisson_rhs(experiment, cgrid.cache, masked_u, cgrid.dt)
    x0 = last(experiment.p_field) * cgrid.dt
    return rhs, x0
end

# ---- Preconditioned solve with local SPD checks ----

"""
    spectral_solve(A, rhs, x0, precond; maxdim, nsweeps, tol, cutoff) -> Dict

ResidualALS solve of `A x = rhs` preconditioned by `precond`, as in the benchmarks. The local
SPD checks throw a `NotSPDError` (`spd_tol = SPD_TOL`) at the first local P⁻¹A that is clearly not SPD, which stops the
solve and marks the preconditioner as not SPD. Other solver failures are recorded instead of aborting the benchmark.
"""
function spectral_solve(A::MPO, rhs::MPS, x0::MPS, precond; maxdim::Int, nsweeps::Int, tol::Float64, cutoff::Float64)
    numiter_store = KrylovSweepStats()
    cond_store = KrylovCondStats()
    pc_cond_store = KrylovCondStats()
    updater_kwargs = LocalUpdaterArgs(;
        ishermitian=true,
        isposdef=true,
        spd_tol=SPD_TOL,
        rtol=0.1 * tol,
        krylovdim=1,
        maxiter=2000,
        numiter_store=numiter_store,
        cond_store=cond_store,
        pc_cond_store=pc_cond_store,
    )
    sweep_kwargs = SweepingArgs(;
        nsweeps=nsweeps, maxdim=maxdim, cutoff=cutoff, (observer!)=ResidualObserver(; rtol=tol)
    )
    solver = ResidualALS(; sweep_kwargs=sweep_kwargs, updater_kwargs=updater_kwargs, residual_tol=0.5 * tol)

    Random.seed!(0)
    not_spd = nothing
    error_msg = ""
    x = nothing
    t = @elapsed try
        x = linearsolve(solver, A, rhs, precond, copy(x0))
    catch e
        e isa InterruptException && rethrow()
        if e isa NotSPDError
            not_spd = e
        else
            error_msg = sprint(showerror, e)
            progress("Solver failed: $(first(error_msg, 300))")
        end
    end

    pc_conds = DifferenTT.krylov_conds(pc_cond_store)
    result = Dict{String,Any}(
        "solve_time" => t,
        "spd_tol" => SPD_TOL,
        "not_spd" => !isnothing(not_spd),
        "failed" => !isempty(error_msg),
        "error" => error_msg,
        # The stores hold the blocks checked before the solve stopped (the failing block is not recorded)
        "nsweeps_done" => pc_cond_store.current_sweep,
        "nblocks_checked" => sum(pos -> sum(length, pos; init=0), pc_conds; init=0),
        "conds" => DifferenTT.krylov_conds(cond_store),
        "pc_conds" => pc_conds,
        "iters_per_site" => DifferenTT.positions_sum(numiter_store),
        "maxdim" => maxdim,
        "nsweeps_cap" => nsweeps,
    )
    if !isnothing(not_spd)
        result["not_spd_label"] = not_spd.label
        result["not_spd_position"] = not_spd.lpos + 1
        result["not_spd_eigmin"] = not_spd.λmin
        result["not_spd_eigmin_rel"] = not_spd.λmin / not_spd.λmax_abs
        progress(
            @sprintf(
                "  NOT SPD: %s at position %d in sweep %d (%d blocks checked before), min Re λ / max|λ| = %.3e",
                not_spd.label,
                not_spd.lpos + 1,
                result["nsweeps_done"],
                result["nblocks_checked"],
                result["not_spd_eigmin_rel"]
            )
        )
    elseif !isnothing(x)
        term1 = ITensorMPS.inner(A, x, A, x)
        term2 = 2 * real(ITensorMPS.inner(rhs', A, x))
        term3 = ITensorMPS.inner(rhs, rhs)
        result["rel_res"] = sqrt(max(0.0, term1 - term2 + term3)) / norm(rhs)
        result["solution_maxdim"] = maxlinkdim(x)
        progress(
            @sprintf(
                "  No local P⁻¹A found indefinite: %d blocks in %d sweeps, rel. residual %.3e",
                result["nblocks_checked"],
                result["nsweeps_done"],
                result["rel_res"]
            )
        )
    end
    return result
end

function log_dense(prefix::AbstractString, r::AbstractDict)
    progress(
        @sprintf(
            "  %s: λ_min(S) = %.3e, λ_max(S) = %.3e, #neg(S) = %d, λ_min(SA) = %.3e, #neg(SA) = %d, asym = %.1e, ‖S-A⁻¹‖₂ = %.3e vs λ_min(A⁻¹) = %.3e",
            prefix,
            r["eigmin_S"],
            r["eigmax_S"],
            r["nneg_S"],
            r["eigmin_SA"],
            r["nneg_SA"],
            r["asym_P"],
            r["sym_error_2norm"],
            r["eigmin_Ainv"]
        )
    )
end

# ---- Sections ----

"""
linsolve preconditioner at each χ, analyzed with dense matrices.
"""
function linsolve_dense_section(section::AbstractString, cases)
    for (case_key, build_A) in cases
        progress("[$section] $case_key: building operator")
        A = build_A()
        op = dense_operator_info(A)
        progress(
            @sprintf(
                "[%s] %s: %d rows, κ(A) = %.3e, asym(A) = %.1e",
                section,
                case_key,
                length(op.λA),
                last(op.λA) / first(op.λA),
                relative_asymmetry(op.Ad)
            )
        )
        run_case(() -> operator_summary(op), "$section/$case_key/operator")

        for χ in DENSE_MAXDIMS
            run_case("$section/$case_key/$(chi_key(χ))") do
                P, t_build = build_linsolve(A, χ; nsweeps=PRECOND_NSWEEPS, residual_tol=LS_RESIDUAL_TOL)
                progress(@sprintf("  χ = %d built in %.1f s (maxlinkdim %d)", χ, t_build, maxlinkdim(P.operator)))
                Pd = dense_matrix(P.operator, op.sites)
                result = dense_analysis(Pd, op)
                log_dense("χ = $χ", result)
                merge!(result, mpo_info(P.operator))
                result["chi"] = χ
                result["build_time"] = t_build
                result
            end
        end
    end
end

"""
Exact A⁻¹ as an MPO, SVD truncated to each χ and analyzed with dense matrices.
"""
function truncinv_dense_section(section::AbstractString, cases)
    for (case_key, build_A) in cases
        progress("[$section] $case_key: building operator and exact inverse")
        A = build_A()
        op = dense_operator_info(A)
        run_case(() -> operator_summary(op), "$section/$case_key/operator")

        exact = dense_to_mpo(op.Ainv, op.sites; cutoff=EXACT_INV_CUTOFF)
        full_χ = maxlinkdim(exact)
        progress("[$section] $case_key: exact inverse MPO, linkdims $(linkdims(exact))")

        for χ in unique([filter(<(full_χ), TRUNC_MAXDIMS); full_χ])
            run_case("$section/$case_key/$(chi_key(χ))") do
                P = truncate(exact; maxdim=χ, cutoff=0.0)
                result = dense_analysis(dense_matrix(P, op.sites), op)
                log_dense("χ = $χ", result)
                merge!(result, mpo_info(P))
                result["chi"] = χ
                result["full_chi"] = full_χ
                result
            end
        end
    end
end

chorin_dense_cases() = [nodes_key(nodes) => (() -> chorin_operator(nodes)) for nodes in CHORIN_DENSE_NODES]
"""
Bond dimensions of a large preconditioner and its Sylvester residual in TT format.
"""
function large_precond_info(A::MPO, P::MPO)
    result = mpo_info(P)
    result["sylvester_residual"] = sylvester_residual_tt(A, P)
    progress(@sprintf("  ‖AX+XA-2I‖/‖2I‖ = %.3e", result["sylvester_residual"]))
    return result
end

function chorin_linsolve_large_section(section::AbstractString)
    for nodes in CHORIN_LARGE_NODES
        case_key = nodes_key(nodes)
        progress("[$section] $case_key: building Chorin setup")
        cgrid = build_chorin_grid(nodes)
        rhs, x0 = chorin_first_step_system(cgrid)
        solve_kwargs = (; maxdim=chorin_maxdims_map[nodes], nsweeps=CHORIN_SPECTRAL_NSWEEPS, tol=CHORIN_TOL, cutoff=0.0)

        for χ in CHORIN_LARGE_MAXDIMS
            run_case("$section/$case_key/$(chi_key(χ))") do
                P, t_build = build_linsolve(cgrid.A, χ; nsweeps=PRECOND_NSWEEPS, residual_tol=LS_RESIDUAL_TOL)
                progress(@sprintf("  χ = %d built in %.1f s", χ, t_build))
                result = large_precond_info(cgrid.A, P.operator)
                progress("  first Chorin step with local SPD checks")
                result["first_step"] = spectral_solve(cgrid.A, rhs, x0, P; solve_kwargs...)
                result["chi"] = χ
                result["build_time"] = t_build
                result
            end
        end

        for (μ, gdim) in CHORIN_FAI_CONTROL
            run_case("$section/$case_key/fai_gdim_$(lpad(gdim, 3, '0'))") do
                fai_updater_kwargs = LocalUpdaterArgs(;
                    ishermitian=true, isposdef=true, rtol=1e-9, krylovdim=1, maxiter=3000
                )
                fai_solver = ALS(;
                    sweep_kwargs=SweepingArgs(; nsweeps=PRECOND_NSWEEPS, maxdim=gdim, cutoff=1e-12),
                    updater_kwargs=fai_updater_kwargs,
                )
                Random.seed!(nodes[1] + nodes[2])
                t_build = @elapsed begin
                    unscaledG, D = DifferenTT.fai_dg(
                        cgrid.A,
                        μ,
                        fai_solver,
                        (; cutoff=1e-11, oversample=10, maxdim=gdim);
                        residual_tol=CHORIN_FAI_TOL,
                    )
                    P = FactorizedPreconditioner(apply(D, unscaledG; maxdim=gdim))
                end
                progress(@sprintf("  FAI gdim = %d built in %.1f s", gdim, t_build))
                progress("  first Chorin step with local SPD checks")
                Dict{String,Any}(
                    "first_step" => spectral_solve(cgrid.A, rhs, x0, P; solve_kwargs...),
                    "mu" => μ,
                    "G_maxdim" => gdim,
                    "build_time" => t_build,
                )
            end
        end
    end
end

function main()
    progress("Results file: $RESULTS_FILE")
    progress("Sections: $(join(SECTIONS, ", "))")
    run_case("meta/config") do
        Dict{String,Any}(
            "dense_maxdims" => DENSE_MAXDIMS,
            "trunc_maxdims" => TRUNC_MAXDIMS,
            "exact_inv_cutoff" => EXACT_INV_CUTOFF,
            "precond_nsweeps" => PRECOND_NSWEEPS,
            "ls_residual_tol" => LS_RESIDUAL_TOL,
            "spd_tol" => SPD_TOL,
            "chorin_dense_nodes" => CHORIN_DENSE_NODES,
            "chorin_large_nodes" => CHORIN_LARGE_NODES,
            "chorin_large_maxdims" => CHORIN_LARGE_MAXDIMS,
            "chorin_fai_control" => CHORIN_FAI_CONTROL,
            "chorin_spectral_nsweeps" => CHORIN_SPECTRAL_NSWEEPS,
        )
    end

    for section in SECTIONS
        progress("===== Section $section =====")
        t = @elapsed begin
            if section == "chorin_linsolve_dense"
                linsolve_dense_section(section, chorin_dense_cases())
            elseif section == "chorin_truncinv_dense"
                truncinv_dense_section(section, chorin_dense_cases())
            elseif section == "chorin_linsolve_large"
                chorin_linsolve_large_section(section)
            end
        end
        progress(@sprintf("===== Section %s done in %.1f min =====", section, t / 60))
    end
    progress("All sections done")
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end

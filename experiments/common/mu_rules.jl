"""
    Rules for the penalty μ of the FAI build (shared by the paper workers, collectors and plotters)

The FAI solves Eq. (22) of the paper with the operator P_L (I ⊗ A) P_L + μ (I - P_L), whose condition number is
max(μ, λmax(A)) / min(μ, λmin(A)). μ only needs to sit at the top of the spectrum of A: a μ far above λmax(A) multiplies
the condition number of the build by μ / λmax(A). The paper uses μ = 10⁷, which is close to λmax(A) only on the finest
Chorin grid. The rule `gershgorin` sets μ = ‖A‖_∞, the Gershgorin upper bound of λmax(A), which we bound from the 1D
factors of A only, with ‖ΣX‖_∞ ≤ Σ‖X‖_∞, ‖X ⊗ I‖_∞ = ‖X‖_∞ and ‖X ⊗ Y‖_∞ = ‖X‖_∞ ‖Y‖_∞:

  - anisotropy, A = -(Σ_ℓ d2_ℓ/h_ℓ² + 2ρ Σ_{ℓ<m} (d1_ℓ/h_ℓ)(d1_m/h_m)):
    μ = Σ_ℓ ‖d2_ℓ‖_∞/h_ℓ² + 2|ρ| Σ_{ℓ<m} ‖d1_ℓ‖_∞ ‖d1_m‖_∞/(h_ℓ h_m) (`anisotropy_mu_gershgorin`);
  - Chorin, A = -W Δ with the Neumann weights W ∈ {1, 1/2}, so ‖W‖_∞ ≤ 1: μ = Σ_ℓ ‖d2_ℓ‖_∞/h_ℓ²
    (`laplacian_mu_gershgorin`).

The 1D operators are those of `onedim_differential_operator`, with the boundary closures of the field, so the bound
holds for the assembled operator. `experiments/common/mu_rules_check.jl` checks λmax(A) ≤ μ ≤ 1.5 λmax(A) on small cases.

# Environment variables

  - `FAI_MU_RULE`: `fixed` (default, the μ of the worker, 10⁷) or `gershgorin`.
  - `FAI_MU_SCALE`: factor applied to the value of the rule (default 1.0); only allowed with a rule other than `fixed`.

A rule-based run writes under a separate results folder (`results_nocond_mugersh`, see `mu_results_folder`), and its
result keys name the rule instead of the numeric μ (`mu_key`), so every (N, F) or grid shares one key per χ_G. The
numeric μ goes to the field `"mu"` and the rule to `"mu_rule"` (and `"mu_scale"`).

This file only needs DifferenTT (and ITensors, for QTT operators) when the bounds are evaluated, so the collectors and
plotters can include it for the naming helpers alone.
"""

using LinearAlgebra
using Printf

const MU_RULES = ("fixed", "gershgorin")

"""
    fai_mu_rule() -> (rule::String, scale::Float64)

Read `FAI_MU_RULE` and `FAI_MU_SCALE` from the environment, with the defaults that reproduce the paper (`"fixed"`, 1.0).
"""
function fai_mu_rule()
    rule = lowercase(strip(get(ENV, "FAI_MU_RULE", "fixed")))
    rule in MU_RULES || throw(ArgumentError("FAI_MU_RULE=$rule: expected one of $(join(MU_RULES, ", "))"))
    scale = parse(Float64, get(ENV, "FAI_MU_SCALE", "1.0"))
    (isfinite(scale) && scale > 0) || throw(ArgumentError("FAI_MU_SCALE=$scale must be positive"))
    rule == "fixed" &&
        scale != 1 &&
        throw(ArgumentError("FAI_MU_SCALE only applies to a rule, not to FAI_MU_RULE=fixed"))
    return rule, scale
end

"""
    mu_rule_tag(rule, scale) -> String

Short name of a rule for keys and folders: `""` for `fixed`, `"gersh"` for `gershgorin`, with `"_x<scale>"` appended
when the scale is not 1 (the scale printed with `%g`, as `submit.sh` does).
"""
function mu_rule_tag(rule::AbstractString, scale::Real)
    rule == "fixed" && return ""
    tag = rule == "gershgorin" ? "gersh" : rule
    return scale == 1 ? tag : "$(tag)_x$(@sprintf("%g", scale))"
end

"""
    mu_key(rule, scale, μ) -> String

The μ part of a result key: `"mu_<μ>"` for `fixed` (unchanged from the original workers), else `"mu_<tag>"`, so that the
key does not depend on the numeric μ of each problem.
"""
mu_key(rule::AbstractString, scale::Real, μ::Real) = rule == "fixed" ? "mu_$(μ)" : "mu_$(mu_rule_tag(rule, scale))"

"""
    mu_results_folder(name, rule, scale) -> String

The results folder of a run: `name` (e.g. `"results_nocond"`) for `fixed`, else `"<name>_mu<tag>"`, e.g.
`"results_nocond_mugersh"`, so that rule-based runs never mix with fixed-μ data.
"""
function mu_results_folder(name::AbstractString, rule::AbstractString, scale::Real)
    rule == "fixed" ? name : "$(name)_mu$(mu_rule_tag(rule, scale))"
end

"""
    mu_rule_latex(rule, scale) -> String

LaTeX (without the dollars) of the value of a rule, e.g. `\\|A\\|_\\infty` or `0.5\\|A\\|_\\infty`.
"""
function mu_rule_latex(rule::AbstractString, scale::Real)
    base = rule == "gershgorin" ? "\\|A\\|_\\infty" : "\\mathrm{$rule}"
    return scale == 1 ? base : "$(round(Float64(scale); sigdigits=3))\\,$base"
end

# ---------------------------------------------------------------------------------------------------------------------
# Bounds
# ---------------------------------------------------------------------------------------------------------------------

"""
    opnorm_inf(X) -> Float64

‖X‖_∞, the largest absolute row sum.
"""
opnorm_inf(X::AbstractMatrix) = maximum(sum(abs, X; dims=2))

"""
    qtt_onedim_matrix(mpo) -> Matrix{Float64}

Dense matrix of a 1D QTT operator (site 1 the most significant bit, rows on the primed indices). Only for small axes:
it has 4ⁿ entries.
"""
function qtt_onedim_matrix(mpo)
    s = DifferenTT.ITensorMPS.siteinds(first, mpo; plev=0)
    n = length(s)
    # The contraction has 2n indices: bound the order warning just past it, without changing the caller's setting
    previous = DifferenTT.ITensors.set_warn_order(2n + 1)
    T = try
        DifferenTT.ITensors.contract(collect(mpo))
    finally
        DifferenTT.ITensors.set_warn_order(previous)
    end
    # Reversed, so that site 1 is the slowest index of the column-major reshape
    arr = Array(T, reverse(DifferenTT.ITensors.prime.(s))..., reverse(s)...)
    return reshape(arr, 2^n, 2^n)
end

"""
    onedim_matrix(order, arch, dim, bc; stencil_order) -> AbstractMatrix

The unscaled 1D operator of `onedim_differential_operator` as a matrix, in TT (directly) or QTT (contracted).
"""
function onedim_matrix(order::Int, arch, dim::Int, bc; stencil_order::Int)
    op = DifferenTT.onedim_differential_operator(order, arch, dim, bc, Array; stencil_order=stencil_order)
    return op isa AbstractMatrix ? op : qtt_onedim_matrix(op)
end

"""
    anisotropy_mu_gershgorin(arch, grid, bcs, ρ; stencil_order=2) -> Float64

Upper bound of ‖A‖_∞ for A = -equicorrelated_laplacian(field, grid; correlation=ρ) on a field with the boundary
conditions `bcs` (one `BoundaryCond1d` per axis): Σ_ℓ ‖d2_ℓ‖_∞/h_ℓ² + 2|ρ| Σ_{ℓ<m} ‖d1_ℓ‖_∞ ‖d1_m‖_∞/(h_ℓ h_m).
"""
function anisotropy_mu_gershgorin(arch, grid, bcs, ρ::Real; stencil_order::Int=2)
    N = length(grid.dx)
    a2 = [opnorm_inf(onedim_matrix(2, arch, i, bcs[i]; stencil_order)) / grid.dx[i]^2 for i in 1:N]
    μ = sum(a2)
    iszero(ρ) && return μ
    a1 = [opnorm_inf(onedim_matrix(1, arch, i, bcs[i]; stencil_order)) / grid.dx[i] for i in 1:N]
    # Σ_{ℓ<m} a_ℓ a_m = ((Σ a)² - Σ a²) / 2
    return μ + 2 * abs(ρ) * (sum(a1)^2 - sum(abs2, a1)) / 2
end

"""
    laplacian_mu_gershgorin(field, grid; stencil_order=8) -> Float64

Upper bound Σ_ℓ ‖d2_ℓ‖_∞/h_ℓ² of ‖A‖_∞ for A = -symmetrized_laplacian(field, grid) (or -kronecker_laplacian), with
the boundary closures of `field` (Neumann mirrored, periodic, Dirichlet). The Neumann weights W ≤ 1 only lower the row
sums.
"""
function laplacian_mu_gershgorin(field, grid; stencil_order::Int=8)
    N = length(grid.dx)
    return sum(opnorm_inf(onedim_matrix(2, field.arch, i, field.bcu[i]; stencil_order)) / grid.dx[i]^2 for i in 1:N)
end

"""
    resolve_fai_mu(rule, scale, μ_fixed, bound) -> Float64

The μ of a FAI build: `μ_fixed` for `fixed`, else `scale * bound()` (`bound` is only called for a rule).
"""
function resolve_fai_mu(rule::AbstractString, scale::Real, μ_fixed::Real, bound)
    rule == "fixed" ? Float64(μ_fixed) : scale * Float64(bound())
end

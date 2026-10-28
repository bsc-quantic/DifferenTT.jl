"""
    Rules for the penalty μ of the FAI build (shared by the paper workers, collectors and plotters)

The FAI solves Eq. (21) of the paper with the operator P_L (I ⊗ A) P_L + μ (I - P_L), whose condition number is
max(μ, λmax(A)) / min(μ, λmin(A)). μ only needs to sit at the top of the spectrum of A: a μ far above λmax(A) multiplies
the condition number of the build by μ / λmax(A). The paper uses the rule `gershgorin` (Eqs. (32) and (37)), the
default here, which sets μ = ‖A‖_∞, the Gershgorin upper bound of λmax(A). The rule `fixed` keeps the μ of the task list
of each worker, the 10⁷ of the earlier drafts of the paper (the data of the tag `paper-ttfai-data`), which is close to
λmax(A) only on the finest Chorin grid. We bound ‖A‖_∞ from the stencil coefficients c_k of `derivative_coefficients`
alone, without building any operator:

  - in 1D, every row of the operator D (h = 1) is the central stencil, with the coefficients of the ghost nodes dropped
    (Dirichlet, open sides), wrapped (periodic) or moved onto an interior node of the same row (the Neumann mirror). By
    the triangle inequality no row sum exceeds that of an interior row, so ‖D‖_∞ ≤ s = Σ_k |c_k|, whatever the number of
    nodes and the format (TT or QTT);
  - in N dimensions, ‖ΣX‖_∞ ≤ Σ‖X‖_∞, ‖X ⊗ I‖_∞ = ‖X‖_∞ and ‖X ⊗ Y‖_∞ = ‖X‖_∞ ‖Y‖_∞ give

      - anisotropy, A = -(Σ_ℓ d2_ℓ/h_ℓ² + 2ρ Σ_{ℓ<m} (d1_ℓ/h_ℓ)(d1_m/h_m)):
        μ = s₂ Σ_ℓ h_ℓ⁻² + 2|ρ| s₁² Σ_{ℓ<m} (h_ℓ h_m)⁻¹ (`anisotropy_mu_gershgorin`), i.e. N (4 + ρ (N - 1)) / h² for
        the second-order stencils (s₂ = 4, s₁ = 1) on a uniform grid;
      - Chorin, A = -W Δ with the Neumann weights W ∈ {1, 1/2}, so ‖W‖_∞ ≤ 1: μ = s₂ Σ_ℓ h_ℓ⁻²
        (`laplacian_mu_gershgorin`), with s₂ = 205/72 + 2 (8/5 + 1/5 + 8/315 + 1/560) ≈ 6.5016 for order 8.

In both benchmarks an interior row attains the bound, so it equals ‖A‖_∞; we do not need λmax exactly, only a μ at the
top of the spectrum. `experiments/common/mu_rules_check.jl` checks ‖A‖_∞ ≤ μ and λmax(A) ≤ μ ≤ 1.5 λmax(A) on the
assembled operators of small cases.

# Environment variables

  - `FAI_MU_RULE`: `gershgorin` (default, μ = ‖A‖_∞, as in the paper) or `fixed` (the μ of the task list of the
    worker, 10⁷, as in the earlier drafts).
  - `FAI_MU_SCALE`: factor applied to the value of the rule (default 1.0); only allowed with a rule other than `fixed`.

A rule-based run writes under a separate results folder (`results_nocond_mugersh`, see `mu_results_folder`), and its
result keys name the rule instead of the numeric μ (`mu_key`), so every (N, F) or grid shares one key per χ_G. The
numeric μ goes to the field `"mu"` and the rule to `"mu_rule"` (and `"mu_scale"`).

This file only needs DifferenTT (for `derivative_coefficients`) when the bounds are evaluated, so the collectors and
plotters can include it for the naming helpers alone.
"""

using Printf

const MU_RULES = ("fixed", "gershgorin")

"""
    fai_mu_rule() -> (rule::String, scale::Float64)

Read `FAI_MU_RULE` and `FAI_MU_SCALE` from the environment, with the defaults that reproduce the paper (`"gershgorin"`,
1.0).
"""
function fai_mu_rule()
    rule = lowercase(strip(get(ENV, "FAI_MU_RULE", "gershgorin")))
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
    stencil_abs_sum(derivative_order; stencil_order) -> Float64

Σ_k |c_k| of the central stencil of `derivative_coefficients`: the row sum of an interior row of the 1D operator for a
unit spacing, and an upper bound of its ‖·‖_∞ with any of the closures of the library.
"""
function stencil_abs_sum(derivative_order::Int; stencil_order::Int)
    return sum(abs, values(DifferenTT.derivative_coefficients(derivative_order; stencil_order=stencil_order)))
end

"""
    anisotropy_mu_gershgorin(grid, ρ; stencil_order=2) -> Float64

Upper bound of ‖A‖_∞ for A = -equicorrelated_laplacian(field, grid; correlation=ρ, stencil_order):
s₂ Σ_ℓ h_ℓ⁻² + 2|ρ| s₁² Σ_{ℓ<m} (h_ℓ h_m)⁻¹, with s_k = `stencil_abs_sum(k)` and h_ℓ = `grid.dx[ℓ]`.
"""
function anisotropy_mu_gershgorin(grid, ρ::Real; stencil_order::Int=2)
    s2 = stencil_abs_sum(2; stencil_order)
    μ = s2 * sum(inv.(grid.dx .^ 2))
    iszero(ρ) && return μ
    a1 = stencil_abs_sum(1; stencil_order) .* inv.(grid.dx)
    # Σ_{ℓ<m} a_ℓ a_m = ((Σ a)² - Σ a²) / 2
    return μ + 2 * abs(ρ) * (sum(a1)^2 - sum(abs2, a1)) / 2
end

"""
    laplacian_mu_gershgorin(grid; stencil_order=8) -> Float64

Upper bound s₂ Σ_ℓ h_ℓ⁻² of ‖A‖_∞ for A = -symmetrized_laplacian(field, grid; stencil_order) (or
-kronecker_laplacian), for any boundary conditions of `field`; the Neumann weights W ≤ 1 only lower the row sums.
"""
laplacian_mu_gershgorin(grid; stencil_order::Int=8) = stencil_abs_sum(2; stencil_order) * sum(inv.(grid.dx .^ 2))

"""
    resolve_fai_mu(rule, scale, μ_fixed, bound) -> Float64

The μ of a FAI build: `μ_fixed` for `fixed`, else `scale * bound()` (`bound` is only called for a rule).
"""
function resolve_fai_mu(rule::AbstractString, scale::Real, μ_fixed::Real, bound)
    rule == "fixed" ? Float64(μ_fixed) : scale * Float64(bound())
end

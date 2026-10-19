# Adapted from KrylovKit.jl (https://github.com/Jutho/KrylovKit.jl): the unpreconditioned iteration below is that of
# KrylovKit's CG linsolve, and the preconditioner is added on top of it. KrylovKit.jl carries the following notice:
#
# The KrylovKit.jl package is licensed under the MIT "Expat" License:
#
# > Copyright (c) 2017: Jutho.
# >
# > Permission is hereby granted, free of charge, to any person obtaining a copy
# > of this software and associated documentation files (the "Software"), to deal
# > in the Software without restriction, including without limitation the rights
# > to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# > copies of the Software, and to permit persons to whom the Software is
# > furnished to do so, subject to the following conditions:
# >
# > The above copyright notice and this permission notice shall be included in all
# > copies or substantial portions of the Software.
# >
# > THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# > IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# > FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# > AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# > LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# > OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# > SOFTWARE.
#
# See also THIRD_PARTY_NOTICES.md.

"""
    preconditioned_linsolve(operator, b, x₀, M⁻¹, alg::KrylovKit.CG, a₀::Real=0, a₁::Real=1; alg_rrule=alg)

Solve ``(a_0 + a_1 A) x = b`` by preconditioned CG, with the preconditioner applied as `KrylovKit.apply(M⁻¹, r)`.
The unpreconditioned iteration is that of KrylovKit.jl; see the header of this file. Returns `(x, info)` as
`KrylovKit.linsolve` does.
"""
function preconditioned_linsolve(operator, b, x₀, M⁻¹, alg::CG, a₀::Real=0, a₁::Real=1; alg_rrule=alg)
    # Initial function operation and division defines number type
    y₀ = KrylovKit.apply(operator, x₀)
    T = typeof(KrylovKit.inner(b, y₀) / norm(b) * one(a₀) * one(a₁))
    α₀ = convert(T, a₀)
    α₁ = convert(T, a₁)
    # Continue computing r = b - a₀ * x₀ - a₁ * operator(x₀)
    r = scale(b, one(T))
    r = iszero(α₀) ? r : add!!(r, x₀, -α₀)
    r = add!!(r, y₀, -α₁)
    x = scale!!(zerovector(r), x₀, 1)
    normr = norm(r)
    S = typeof(normr)

    # Algorithm parameters
    maxiter = alg.maxiter
    tol::S = alg.tol
    numops = 1 # operator has been applied once to determine r
    numiter = 0

    # Check for early return
    if normr < tol
        if alg.verbosity >= STARTSTOP_LEVEL
            @info """CG linsolve converged without any iterations:
            * norm of residual = $(normres2string(normr))
            * number of operations = 1"""
        end
        return (x, ConvergenceInfo(1, r, normr, numiter, numops))
    elseif alg.verbosity >= STARTSTOP_LEVEL
        @info "CG linsolve starts with norm of residual = $(normres2string(normr))"
    end

    z = KrylovKit.apply(M⁻¹, r)
    ρ = KrylovKit.inner(r, z)
    p = scale!!(zerovector(z), z, 1)
    q = KrylovKit.apply(operator, p, α₀, α₁)
    α = ρ / KrylovKit.inner(p, q)
    x = add!!(x, p, +α)
    r = add!!(r, q, -α)
    normr = norm(r)
    ρold = ρ
    z = KrylovKit.apply(M⁻¹, r)
    ρ = KrylovKit.inner(r, z)
    β = ρ / ρold
    numops += 1
    numiter += 1

    if normr < tol
        if alg.verbosity >= STARTSTOP_LEVEL
            @info """CG linsolve converged at iteration $numiter:
            * norm of residual = $(normres2string(normr))
            * number of operations = $numops"""
        end
        return (x, ConvergenceInfo(1, r, normr, numiter, numops))
    end
    if alg.verbosity >= EACHITERATION_LEVEL
        @info "CG linsolve in iteration $numiter: normres = $(normres2string(normr))"
    end

    # Check for early return
    normr < tol && return (x, ConvergenceInfo(1, r, normr, numiter, numops))

    while true
        p = add!!(p, z, 1, β)
        q = KrylovKit.apply(operator, p, α₀, α₁)
        α = ρ / KrylovKit.inner(p, q)
        x = add!!(x, p, α)
        r = add!!(r, q, -α)
        z = KrylovKit.apply(M⁻¹, r)
        normr = norm(r)

        if normr < tol # recompute to account for buildup of floating point errors
            r = scale!!(r, b, 1)
            r = add!!(r, KrylovKit.apply(operator, x, α₀, α₁), -1)
            normr = norm(r)
            z = KrylovKit.apply(M⁻¹, r)
            ρ = KrylovKit.inner(r, z)
            β = zero(β) # restart CG
        else
            ρold = ρ
            ρ = KrylovKit.inner(r, z)
            β = ρ / ρold
        end
        numops += 1
        numiter += 1
        if normr < tol
            if alg.verbosity >= STARTSTOP_LEVEL
                @info """CG linsolve converged at iteration $numiter:
                * norm of residual = $(normres2string(normr))
                * number of operations = $numops"""
            end
            return (x, ConvergenceInfo(1, r, normr, numiter, numops))
        end
        if numiter >= maxiter
            if alg.verbosity >= WARN_LEVEL
                @warn """CG linsolve stopped without converging after $numiter iterations:
                * norm of residual = $(normres2string(normr))
                * number of operations = $numops"""
            end
            return (x, ConvergenceInfo(0, r, normr, numiter, numops))
        end
        if alg.verbosity >= EACHITERATION_LEVEL
            @info "CG linsolve in iteration $numiter: normres = $(normres2string(normr))"
        end
    end
    return nothing
end

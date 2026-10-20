"""
    KrylovKit.linsolve(operator::MPO, b::MPS, x₀::MPS, alg::TTCG, a₀::Real=0, a₁::Real=1; alg_rrule=alg) -> (x, info)
    KrylovKit.linsolve(operator::MPO, b::MPS, x₀::MPS, M⁻¹::AbstractPreconditioner, alg::TTCG, a₀::Real=0, a₁::Real=1;
                       alg_rrule=alg) -> (x, info)

Solve ``(a_0 + a_1 A)\\mathbf{x} = \\mathbf{b}`` by CG on tensor trains, from `x₀`, optionally preconditioned by the SPD
`M⁻¹`; the shifted operator must be SPD. See [`TTCG`](@ref) for the algorithm and its truncation parameters. With
`M⁻¹`, we evaluate ``\\rho = \\langle \\mathbf{r}, M^{-1} \\mathbf{r} \\rangle`` with the untruncated sandwich
`inner(r, M⁻¹, r)`, while the preconditioned residual
``\\mathbf{z} = M^{-1}\\mathbf{r}`` is truncated. `alg_rrule` is accepted for compatibility with KrylovKit and
ignored.

Returns the solution and a `KrylovKit.ConvergenceInfo`, whose `normres` is the norm of the last residual and
`numiter` the number of iterations.
"""
function KrylovKit.linsolve(operator, b, x₀, alg::TTCG, a₀::Real=0, a₁::Real=1; alg_rrule=alg)
    kwargs_tt = (; maxdim=alg.maxdim, cutoff=alg.cutoff)
    kwargs_orth = (; maxdim=alg.orth_maxdim, cutoff=alg.orth_cutoff)

    # Initial function operation and division defines number type
    y₀ = apply(operator, x₀; kwargs_tt...)
    normb = norm(b)
    T = typeof(ITensorMPS.inner(b', operator, x₀) / normb * one(a₀) * one(a₁))
    α₀ = convert(T, a₀)
    α₁ = convert(T, a₁)

    # Continue computing r = b - a₀ * x₀ - a₁ * operator(x₀)
    r = copy(b)
    if !iszero(α₀)
        r = +(r, -α₀ * x₀; kwargs_tt...)
    end
    r = +(r, -α₁ * y₀; kwargs_tt...)

    x = copy(x₀)
    normr = norm(r)
    S = typeof(normr)

    # Algorithm parameters
    maxiter = alg.maxiter
    tol::S = alg.tol * normb
    numops = 1 # operator has been applied once to determine r
    numiter = 0

    # Check for early return
    if normr <= tol
        if alg.verbosity >= STARTSTOP_LEVEL
            @info """TTCG linsolve converged without any iterations:
            * norm of residual = $(normres2string(normr))
            * number of operations = 1"""
        end
        return (x, ConvergenceInfo(1, r, normr, numiter, numops))
    elseif alg.verbosity >= STARTSTOP_LEVEL
        @info "TTCG linsolve starts with norm of residual = $(normres2string(normr))"
    end

    ρ = ITensorMPS.inner(r, r)
    p = copy(r)

    q_op = apply(operator, p; kwargs_tt...)
    q = iszero(α₀) ? (α₁ * q_op) : +(α₀ * p, α₁ * q_op; kwargs_tt...)

    # With TTs it is much more efficient to perform the full sandwich 
    denom = if iszero(α₀)
        α₁ * ITensorMPS.inner(p', operator, p)
    else
        α₀ * ITensorMPS.inner(p, p) + α₁ * ITensorMPS.inner(p', operator, p)
    end
    α = ρ / denom

    # TODO(release): evaluate the cost of recomputing the residual at each step in order to mitigate the
    # truncation errors
    # killing the A-conjugate property
    x = +(x, α * p; kwargs_tt...)
    r = +(r, -α * q; kwargs_tt...)
    normr = norm(r)
    ρold = ρ
    ρ = ITensorMPS.inner(r, r)
    β = ρ / ρold
    numops += 1
    numiter += 1

    if normr <= tol
        if alg.verbosity >= STARTSTOP_LEVEL
            @info """TTCG linsolve converged at iteration $numiter:
            * norm of residual = $(normres2string(normr))
            * number of operations = $numops"""
        end
        return (x, ConvergenceInfo(1, r, normr, numiter, numops))
    end
    if alg.verbosity >= EACHITERATION_LEVEL
        @info "TTCG linsolve in iteration $numiter: normres = $(normres2string(normr))"
    end

    # Check for early return
    normr <= tol && return (x, ConvergenceInfo(1, r, normr, numiter, numops))

    while true
        p = +(r, β * p; kwargs_orth...)

        q_op = apply(operator, p; kwargs_tt...)
        q = iszero(α₀) ? (α₁ * q_op) : +(α₀ * p, α₁ * q_op; kwargs_tt...)

        # With TTs it is much more efficient to perform the full sandwich 
        denom = if iszero(α₀)
            α₁ * ITensorMPS.inner(p', operator, p)
        else
            α₀ * ITensorMPS.inner(p, p) + α₁ * ITensorMPS.inner(p', operator, p)
        end
        α = ρ / denom
        x = +(x, α * p; kwargs_tt...)
        r = +(r, -α * q; kwargs_tt...)
        normr = norm(r)

        if normr <= tol # recompute to account for buildup of floating point errors
            r = copy(b)
            x_op = apply(operator, x; kwargs_tt...)
            q_x = iszero(α₀) ? (α₁ * x_op) : +(α₀ * x, α₁ * x_op; kwargs_tt...)
            r = +(r, -1 * q_x; kwargs_tt...)
            normr = norm(r)
            ρ = ITensorMPS.inner(r, r)
            β = zero(β) # restart CG
        else
            ρold = ρ
            ρ = ITensorMPS.inner(r, r)
            β = ρ / ρold
        end
        numops += 1
        numiter += 1
        if normr <= tol
            if alg.verbosity >= STARTSTOP_LEVEL
                @info """TTCG linsolve converged at iteration $numiter:
                * norm of residual = $(normres2string(normr))
                * number of operations = $numops"""
            end
            return (x, ConvergenceInfo(1, r, normr, numiter, numops))
        end
        if numiter >= maxiter
            if alg.verbosity >= WARN_LEVEL
                @warn """TTCG linsolve stopped without converging after $numiter iterations:
                * norm of residual = $(normres2string(normr))
                * number of operations = $numops"""
            end
            return (x, ConvergenceInfo(0, r, normr, numiter, numops))
        end
        if alg.verbosity >= EACHITERATION_LEVEL
            @info "TTCG linsolve in iteration $numiter: normres = $(normres2string(normr))"
        end
    end
    return nothing
end

function KrylovKit.linsolve(operator, b, x₀, M⁻¹, alg::TTCG, a₀::Real=0, a₁::Real=1; alg_rrule=alg)
    kwargs_tt = (; maxdim=alg.maxdim, cutoff=alg.cutoff)
    kwargs_orth = (; maxdim=alg.orth_maxdim, cutoff=alg.orth_cutoff)

    # Initial function operation and division defines number type
    y₀ = apply(operator, x₀; kwargs_tt...)
    normb = norm(b)
    T = typeof(ITensorMPS.inner(b', operator, x₀) / normb * one(a₀) * one(a₁))
    α₀ = convert(T, a₀)
    α₁ = convert(T, a₁)

    # Continue computing r = b - a₀ * x₀ - a₁ * operator(x₀)
    r = copy(b)
    if !iszero(α₀)
        r = +(r, -α₀ * x₀; kwargs_tt...)
    end
    r = +(r, -α₁ * y₀; kwargs_tt...)

    x = copy(x₀)
    normr = norm(r)
    S = typeof(normr)

    # Algorithm parameters
    maxiter = alg.maxiter
    tol::S = alg.tol * normb
    numops = 1 # operator has been applied once to determine r
    numiter = 0

    # Check for early return
    if normr <= tol
        if alg.verbosity >= STARTSTOP_LEVEL
            @info """TTCG linsolve converged without any iterations:
            * norm of residual = $(normres2string(normr))
            * number of operations = 1"""
        end
        return (x, ConvergenceInfo(1, r, normr, numiter, numops))
    elseif alg.verbosity >= STARTSTOP_LEVEL
        @info "TTCG linsolve starts with norm of residual = $(normres2string(normr))"
    end

    z = apply(M⁻¹, r; kwargs_tt...)
    ρ = ITensorMPS.inner(r, M⁻¹, r)
    p = copy(z)

    q_op = apply(operator, p; kwargs_tt...)
    q = iszero(α₀) ? (α₁ * q_op) : +(α₀ * p, α₁ * q_op; kwargs_tt...)

    # With TTs it is much more efficient to perform the full sandwich 
    denom = if iszero(α₀)
        α₁ * ITensorMPS.inner(p', operator, p)
    else
        α₀ * ITensorMPS.inner(p, p) + α₁ * ITensorMPS.inner(p', operator, p)
    end
    α = ρ / denom

    x = +(x, α * p; kwargs_tt...)
    r = +(r, -α * q; kwargs_tt...)
    normr = norm(r)
    ρold = ρ
    z = apply(M⁻¹, r; kwargs_tt...)
    ρ = ITensorMPS.inner(r, M⁻¹, r)
    β = ρ / ρold
    numops += 1
    numiter += 1

    if normr <= tol
        if alg.verbosity >= STARTSTOP_LEVEL
            @info """TTCG linsolve converged at iteration $numiter:
            * norm of residual = $(normres2string(normr))
            * number of operations = $numops"""
        end
        return (x, ConvergenceInfo(1, r, normr, numiter, numops))
    end
    if alg.verbosity >= EACHITERATION_LEVEL
        @info "TTCG linsolve in iteration $numiter: normres = $(normres2string(normr))"
    end

    # Check for early return
    normr <= tol && return (x, ConvergenceInfo(1, r, normr, numiter, numops))

    while true
        p = +(z, β * p; kwargs_orth...)

        q_op = apply(operator, p; kwargs_tt...)
        q = iszero(α₀) ? (α₁ * q_op) : +(α₀ * p, α₁ * q_op; kwargs_tt...)

        # With TTs it is much more efficient to perform the full sandwich 
        denom = if iszero(α₀)
            α₁ * ITensorMPS.inner(p', operator, p)
        else
            α₀ * ITensorMPS.inner(p, p) + α₁ * ITensorMPS.inner(p', operator, p)
        end
        α = ρ / denom
        x = +(x, α * p; kwargs_tt...)
        r = +(r, -α * q; kwargs_tt...)
        z = apply(M⁻¹, r; kwargs_tt...)
        normr = norm(r)

        if normr <= tol # recompute to account for buildup of floating point errors
            r = copy(b)
            x_op = apply(operator, x; kwargs_tt...)
            q_x = iszero(α₀) ? (α₁ * x_op) : +(α₀ * x, α₁ * x_op; kwargs_tt...)
            r = +(r, -1 * q_x; kwargs_tt...)
            normr = norm(r)
            z = apply(M⁻¹, r; kwargs_tt...)
            ρ = ITensorMPS.inner(r, M⁻¹, r)
            β = zero(β) # restart CG
        else
            ρold = ρ
            ρ = ITensorMPS.inner(r, M⁻¹, r)
            β = ρ / ρold
        end
        numops += 1
        numiter += 1
        if normr <= tol
            if alg.verbosity >= STARTSTOP_LEVEL
                @info """TTCG linsolve converged at iteration $numiter:
                * norm of residual = $(normres2string(normr))
                * number of operations = $numops"""
            end
            return (x, ConvergenceInfo(1, r, normr, numiter, numops))
        end
        if numiter >= maxiter
            if alg.verbosity >= WARN_LEVEL
                @warn """TTCG linsolve stopped without converging after $numiter iterations:
                * norm of residual = $(normres2string(normr))
                * number of operations = $numops"""
            end
            return (x, ConvergenceInfo(0, r, normr, numiter, numops))
        end
        if alg.verbosity >= EACHITERATION_LEVEL
            @info "TTCG linsolve in iteration $numiter: normres = $(normres2string(normr))"
        end
    end
    return nothing
end

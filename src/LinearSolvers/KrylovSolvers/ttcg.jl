using KrylovKit: ConvergenceInfo, normres2string, STARTSTOP_LEVEL, EACHITERATION_LEVEL, WARN_LEVEL
using ITensors, ITensorMPS

function KrylovKit.linsolve(operator, b, x₀, alg::TTCG, a₀::Real=0, a₁::Real=1; alg_rrule=alg)
    kwargs_tt = (; maxdim=alg.maxdim, cutoff=alg.cutoff)
    kwargs_orth = (; maxdim=alg.orth_maxdim, cutoff=alg.orth_cutoff)

    # Initial function operation and division defines number type
    y₀ = apply(operator, x₀; kwargs_tt...)
    T = typeof(inner(b, operator, x₀) / norm(b) * one(a₀) * one(a₁))
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
    tol::S = alg.tol
    numops = 1 # operator has been applied once to determine r
    numiter = 0

    # Check for early return
    if normr < tol
        if alg.verbosity >= STARTSTOP_LEVEL
            @info """TTCG linsolve converged without any iterations:
            * norm of residual = $(normres2string(normr))
            * number of operations = 1"""
        end
        return (x, ConvergenceInfo(1, r, normr, numiter, numops))
    elseif alg.verbosity >= STARTSTOP_LEVEL
        @info "TTCG linsolve starts with norm of residual = $(normres2string(normr))"
    end

    ρ = inner(r, r)
    p = copy(r)
    
    q_op = apply(operator, p; kwargs_tt...)
    q = iszero(α₀) ? (α₁ * q_op) : +(α₀ * p, α₁ * q_op; kwargs_tt...)

    # With TTs it is much more efficient to perform the full sandwich 
    denom = iszero(α₀) ? α₁ * inner(p, operator, p) : α₁ * inner(p, p) + α₁ * inner(p, operator, p)
    α = ρ / denom

    x = +(x, α * p; kwargs_tt...)
    r = +(r, -α * q; kwargs_tt...)
    normr = norm(r)
    ρold = ρ
    ρ = inner(r, r)
    β = ρ / ρold
    numops += 1
    numiter += 1

    if normr < tol
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
    normr < tol && return (x, ConvergenceInfo(1, r, normr, numiter, numops))

    while true
        p = +(r, β * p; kwargs_orth...)
        
        q_op = apply(operator, p; kwargs_tt...)
        q = iszero(α₀) ? (α₁ * q_op) : +(α₀ * p, α₁ * q_op; kwargs_tt...)
        
        # With TTs it is much more efficient to perform the full sandwich 
        denom = iszero(α₀) ? α₁ * inner(p, operator, p) : α₁ * inner(p, p) + α₁ * inner(p, operator, p)
        α = ρ / denom
        x = +(x, α * p; kwargs_tt...)
        r = +(r, -α * q; kwargs_tt...)
        normr = norm(r)

        if normr < tol # recompute to account for buildup of floating point errors
            r = copy(b)
            x_op = apply(operator, x; kwargs_tt...)
            q_x = iszero(α₀) ? (α₁ * x_op) : +(α₀ * x, α₁ * x_op; kwargs_tt...)
            r = +(r, -1 * q_x; kwargs_tt...)
            normr = norm(r)
            ρ = inner(r, r)
            β = zero(β) # restart CG
        else
            ρold = ρ
            ρ = inner(r, r)
            β = ρ / ρold
        end
        @debug @show ρ
        numops += 1
        numiter += 1
        if normr < tol
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
    return
end

function KrylovKit.linsolve(operator, b, x₀, M⁻¹, alg::TTCG, a₀::Real=0, a₁::Real=1; alg_rrule=alg)
    kwargs_tt = (; maxdim=alg.maxdim, cutoff=alg.cutoff)
    kwargs_orth = (; maxdim=alg.orth_maxdim, cutoff=alg.orth_cutoff)

    # Initial function operation and division defines number type
    y₀ = apply(operator, x₀; kwargs_tt...)
    T = typeof(inner(b, operator, x₀) / norm(b) * one(a₀) * one(a₁))
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
    tol::S = alg.tol
    numops = 1 # operator has been applied once to determine r
    numiter = 0

    # Check for early return
    if normr < tol
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
    ρ = inner(r, M⁻¹, r)
    p = copy(z)
    
    q_op = apply(operator, p; kwargs_tt...)
    q = iszero(α₀) ? (α₁ * q_op) : +(α₀ * p, α₁ * q_op; kwargs_tt...)

    # With TTs it is much more efficient to perform the full sandwich 
    denom = iszero(α₀) ? α₁ * inner(p, operator, p) : α₁ * inner(p, p) + α₁ * inner(p, operator, p)
    α = ρ / denom

    x = +(x, α * p; kwargs_tt...)
    r = +(r, -α * q; kwargs_tt...)
    normr = norm(r)
    ρold = ρ
    z = apply(M⁻¹, r; kwargs_tt...)
    ρ = inner(r, M⁻¹, r)
    β = ρ / ρold
    numops += 1
    numiter += 1

    if normr < tol
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
    normr < tol && return (x, ConvergenceInfo(1, r, normr, numiter, numops))

    while true
        p = +(z, β * p; kwargs_orth...)
        
        q_op = apply(operator, p; kwargs_tt...)
        q = iszero(α₀) ? (α₁ * q_op) : +(α₀ * p, α₁ * q_op; kwargs_tt...)
        
        # With TTs it is much more efficient to perform the full sandwich 
        denom = iszero(α₀) ? α₁ * inner(p, operator, p) : α₁ * inner(p, p) + α₁ * inner(p, operator, p)
        α = ρ / denom
        x = +(x, α * p; kwargs_tt...)
        r = +(r, -α * q; kwargs_tt...)
        z = apply(M⁻¹, r; kwargs_tt...)
        normr = norm(r)

        if normr < tol # recompute to account for buildup of floating point errors
            r = copy(b)
            x_op = apply(operator, x; kwargs_tt...)
            q_x = iszero(α₀) ? (α₁ * x_op) : +(α₀ * x, α₁ * x_op; kwargs_tt...)
            r = +(r, -1 * q_x; kwargs_tt...)
            normr = norm(r)
            z = apply(M⁻¹, r; kwargs_tt...)
            ρ = inner(r, M⁻¹, r)
            β = zero(β) # restart CG
        else
            ρold = ρ
            ρ = inner(r, M⁻¹, r)
            β = ρ / ρold
        end
        @debug @show ρ
        numops += 1
        numiter += 1
        if normr < tol
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
    return
end

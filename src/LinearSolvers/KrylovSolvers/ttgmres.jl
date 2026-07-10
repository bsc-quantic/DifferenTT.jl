using KrylovKit: ConvergenceInfo, normres2string, STARTSTOP_LEVEL, EACHITERATION_LEVEL, WARN_LEVEL, SILENT_LEVEL
using KrylovKit: Givens, givens
using LinearAlgebra: UpperTriangular, ldiv!, lmul!, rmul!
using ITensors, ITensorMPS

function KrylovKit.linsolve(operator, b, x₀, alg::TTGMRES, a₀::Number = 0, a₁::Number = 1; alg_rrule = alg)
    kwargs_tt = (; maxdim=alg.maxdim, cutoff=alg.cutoff)
    kwargs_orth = (; maxdim=alg.orth_maxdim, cutoff=alg.orth_cutoff)

    # Initial function operation and division defines number type
    y₀ = apply(operator, x₀; kwargs_tt...)
    T = typeof(inner(b, operator, x₀) / norm(b) * one(a₀) * one(a₁))
    α₀ = convert(T, a₀)::T
    α₁ = convert(T, a₁)::T
    
    # r = b - a₀ * x₀ - a₁ * operator(x₀)
    r = copy(b)
    if !iszero(α₀)
        r = +(r, -α₀ * x₀; kwargs_tt...)
    end
    r = +(r, -α₁ * y₀; kwargs_tt...)
    
    normr = norm(r)
    β = normr
    
    x = copy(x₀)
    S = typeof(β)

    # Algorithm parameters
    maxiter = alg.maxiter
    krylovdim = alg.krylovdim
    tol::S = alg.tol

    # Check for early return
    if normr < tol
        if alg.verbosity >= STARTSTOP_LEVEL
            @info """TTGMRES linsolve converged without any iterations:
            * norm of residual = $(normres2string(normr))
            * number of operations = 1"""
        end
        return (x, ConvergenceInfo(1, r, normr, 0, 1))
    elseif alg.verbosity >= STARTSTOP_LEVEL
        @info "TTGMRES linsolve starts with norm of residual = $(normres2string(normr))"
    end

    # Initialize data structures
    y = Vector{T}(undef, krylovdim + 1)
    gs = Vector{Givens{T}}(undef, krylovdim)
    H = fill(zero(T), (krylovdim + 1, krylovdim))
    R = fill(zero(T), (krylovdim, krylovdim))
    numiter = 0
    numops = 1
    
    function wrapped_op(v)
        v_out_op = apply(operator, v; kwargs_tt...)
        v_out = iszero(α₀) ? (α₁ * v_out_op) : +(α₀ * v, α₁ * v_out_op; kwargs_tt...)
        return v_out
    end

    V = Vector{typeof(r)}(undef, krylovdim + 1)
    
    while true # restart loop
        numiter += 1
        
        V[1] = (1 / β) * r
        y .= zero(T)
        y[1] = β
        
        k = 0
        β_inner = β
        
        while (β_inner > tol && k < krylovdim)
            k += 1
            if alg.verbosity >= EACHITERATION_LEVEL
                @info "TTGMRES linsolve in iteration $numiter; step $k: normres = $(normres2string(β_inner))"
            end
            
            # Arnoldi expansion
            w = wrapped_op(V[k])
            numops += 1
            
            # Modified Gram-Schmidt
            for i in 1:k
                H_ik_op = inner(V[i], operator, V[k])
                H[i, k] = iszero(α₀) ? α₁ * H_ik_op : (i == k ? α₀ + α₁ * H_ik_op : α₁ * H_ik_op)
                w = +(w, -H[i, k] * V[i]; kwargs_orth...)
            end
            
            H[k+1, k] = norm(w)
            V[k+1] = (1 / H[k+1, k]) * w
            
            # Copy H to R
            for i in 1:k
                R[i, k] = H[i, k]
            end
            
            # Apply previous Givens rotations
            Rk = view(R, 1:k, k)
            for i in 1:(k - 1)
                lmul!(gs[i], Rk)
            end
            
            # Compute new Givens rotation
            gs[k], R[k, k] = givens(R[k, k], H[k+1, k], k, k + 1)
            
            # Apply Givens rotation to right hand side
            y_view = view(y, k:k+1)
            lmul!(gs[k], y_view)
            
            β_inner = convert(S, abs(y[k + 1]))
        end

        # Solve upper triangular system
        y_k = y[1:k]
        ldiv!(UpperTriangular(view(R, 1:k, 1:k)), y_k)

        # Update x
        for i in 1:k
            x = +(x, y_k[i] * V[i]; kwargs_tt...)
        end
        
        # Recompute residual explicitly -> TODO Implement this without operator re-evaluation, as in regular gmres.jl
        r = copy(b)
        x_op = apply(operator, x; kwargs_tt...)
        q_x = iszero(α₀) ? (α₁ * x_op) : +(α₀ * x, α₁ * x_op; kwargs_tt...)
        r = +(r, -1 * q_x; kwargs_tt...)
        normr = norm(r)
        
        numops += 1
        β = normr
        
        if normr < tol
            if alg.verbosity >= STARTSTOP_LEVEL
                @info """TTGMRES linsolve converged at iteration $numiter, step $k:
                * norm of residual = $(normres2string(normr))
                * number of operations = $numops"""
            end
            return (x, ConvergenceInfo(1, r, normr, numiter, numops))
        end
        
        if numiter >= maxiter
            if alg.verbosity >= WARN_LEVEL
                @warn """TTGMRES linsolve stopped without converging after $numiter iterations:
                * norm of residual = $(normres2string(β))
                * number of operations = $numops"""
            end
            return (x, ConvergenceInfo(0, r, β, numiter, numops))
        end
        if alg.verbosity >= EACHITERATION_LEVEL
            @info "TTGMRES linsolve in iteration $numiter; step $k: normres = $(normres2string(β))"
        end
    end
    return
end

function KrylovKit.linsolve(operator, b, x₀, M⁻¹, alg::TTGMRES, a₀::Number = 0, a₁::Number = 1; alg_rrule = alg)
    kwargs_tt = (; maxdim=alg.maxdim, cutoff=alg.cutoff)
    kwargs_orth = (; maxdim=alg.orth_maxdim, cutoff=alg.orth_cutoff)

    # Initial function operation and division defines number type
    y₀ = apply(operator, x₀; kwargs_tt...)
    T = typeof(inner(b, operator, x₀) / norm(b) * one(a₀) * one(a₁))
    α₀ = convert(T, a₀)::T
    α₁ = convert(T, a₁)::T
    
    # r = b - a₀ * x₀ - a₁ * operator(x₀)
    r = copy(b)
    if !iszero(α₀)
        r = +(r, -α₀ * x₀; kwargs_tt...)
    end
    r = +(r, -α₁ * y₀; kwargs_tt...)
    
    normr = norm(r)
    z = apply(M⁻¹, r; kwargs_tt...)
    β = norm(z)
    
    x = copy(x₀)
    S = typeof(β)

    # Algorithm parameters
    maxiter = alg.maxiter
    krylovdim = alg.krylovdim
    tol::S = alg.tol

    # Check for early return
    if normr < tol
        if alg.verbosity >= STARTSTOP_LEVEL
            @info """TTGMRES linsolve converged without any iterations:
            * norm of residual = $(normres2string(normr))
            * number of operations = 1"""
        end
        return (x, ConvergenceInfo(1, r, normr, 0, 1))
    elseif alg.verbosity >= STARTSTOP_LEVEL
        @info "TTGMRES linsolve starts with norm of residual = $(normres2string(normr))"
    end

    # Initialize data structures
    y = Vector{T}(undef, krylovdim + 1)
    gs = Vector{Givens{T}}(undef, krylovdim)
    H = fill(zero(T), (krylovdim + 1, krylovdim))
    R = fill(zero(T), (krylovdim, krylovdim))
    numiter = 0
    numops = 1
    
    function wrapped_op(v)
        v_out_op = apply(operator, v; kwargs_tt...)
        v_out = iszero(α₀) ? (α₁ * v_out_op) : +(α₀ * v, α₁ * v_out_op; kwargs_tt...)
        w = apply(M⁻¹, v_out; kwargs_tt...)
        return w, v_out
    end

    V = Vector{typeof(z)}(undef, krylovdim + 1)
    
    while true # restart loop
        numiter += 1
        
        V[1] = (1 / β) * z
        y .= zero(T)
        y[1] = β
        
        k = 0
        β_inner = β
        
        while (β_inner > tol && k < krylovdim)
            k += 1
            if alg.verbosity >= EACHITERATION_LEVEL
                @info "TTGMRES linsolve in iteration $numiter; step $k: normres = $(normres2string(β_inner))"
            end
            
            # Arnoldi expansion
            w, v_out = wrapped_op(V[k])
            numops += 1
            
            # Modified Gram-Schmidt
            for i in 1:k
                H[i, k] = inner(V[i], M⁻¹, v_out)
                w = +(w, -H[i, k] * V[i]; kwargs_orth...)
            end
            
            H[k+1, k] = norm(w)
            V[k+1] = (1 / H[k+1, k]) * w
            
            # Copy H to R
            for i in 1:k
                R[i, k] = H[i, k]
            end
            
            # Apply previous Givens rotations
            Rk = view(R, 1:k, k)
            for i in 1:(k - 1)
                lmul!(gs[i], Rk)
            end
            
            # Compute new Givens rotation
            gs[k], R[k, k] = givens(R[k, k], H[k+1, k], k, k + 1)
            
            # Apply Givens rotation to right hand side
            y_view = view(y, k:k+1)
            lmul!(gs[k], y_view)
            
            β_inner = convert(S, abs(y[k + 1]))
        end

        # Solve upper triangular system
        y_k = y[1:k]
        ldiv!(UpperTriangular(view(R, 1:k, 1:k)), y_k)

        # Update x
        for i in 1:k
            x = +(x, y_k[i] * V[i]; kwargs_tt...)
        end
        
        # Recompute residual explicitly -> TODO Implement this without operator re-evaluation, as in regular gmres.jl
        r = copy(b)
        x_op = apply(operator, x; kwargs_tt...)
        q_x = iszero(α₀) ? (α₁ * x_op) : +(α₀ * x, α₁ * x_op; kwargs_tt...)
        r = +(r, -1 * q_x; kwargs_tt...)
        normr = norm(r)
        
        z = apply(M⁻¹, r; kwargs_tt...)
        numops += 1
        β = norm(z)
        
        if normr < tol
            if alg.verbosity >= STARTSTOP_LEVEL
                @info """TTGMRES linsolve converged at iteration $numiter, step $k:
                * norm of residual = $(normres2string(normr))
                * number of operations = $numops"""
            end
            return (x, ConvergenceInfo(1, r, normr, numiter, numops))
        end
        
        if numiter >= maxiter
            if alg.verbosity >= WARN_LEVEL
                @warn """TTGMRES linsolve stopped without converging after $numiter iterations:
                * norm of residual = $(normres2string(β))
                * number of operations = $numops"""
            end
            return (x, ConvergenceInfo(0, r, β, numiter, numops))
        end
        if alg.verbosity >= EACHITERATION_LEVEL
            @info "TTGMRES linsolve in iteration $numiter; step $k: normres = $(normres2string(β))"
        end
    end
    return
end

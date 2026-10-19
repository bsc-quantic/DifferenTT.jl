using KrylovKit: scale, scale!!, add!!, zerovector, ConvergenceInfo, normres2string
using KrylovKit: STARTSTOP_LEVEL, EACHITERATION_LEVEL, WARN_LEVEL, SILENT_LEVEL
using KrylovKit: Givens

# Regular non-preconditioned implentation is taken directly from KrylovKit.jl

"""
    preconditioned_linsolve(operator, b, x₀, M⁻¹, alg::KrylovKit.GMRES, a₀::Number=0, a₁::Number=1; alg_rrule=alg)

Solve ``(a_0 + a_1 A) x = b`` by left-preconditioned restarted GMRES, with the preconditioner applied as
`KrylovKit.apply(M⁻¹, ·)`. The unpreconditioned iteration is that of KrylovKit.jl; see the header of this file.
Returns `(x, info)` as `KrylovKit.linsolve` does.
"""
function preconditioned_linsolve(operator, b, x₀, M⁻¹, alg::GMRES, a₀::Number=0, a₁::Number=1; alg_rrule=alg)
    # Initial function operation and division defines number type
    y₀ = KrylovKit.apply(operator, x₀)
    T = typeof(KrylovKit.inner(b, y₀) / norm(b) * one(a₀) * one(a₁))
    α₀ = convert(T, a₀)::T
    α₁ = convert(T, a₁)::T
    # Continue computing r = b - a₀ * x₀ - a₁ * operator(x₀)
    r = scale(b, one(T))
    r = iszero(α₀) ? r : add!!(r, x₀, -α₀)
    r = add!!(r, y₀, -α₁)

    normr = norm(r)
    z = KrylovKit.apply(M⁻¹, r)
    β = norm(z)

    x = scale!!(zerovector(r), x₀, 1)

    S = typeof(β)

    # Algorithm parameters
    maxiter = alg.maxiter
    krylovdim = alg.krylovdim
    tol::S = alg.tol

    # Check for early return
    if normr < tol
        if alg.verbosity >= STARTSTOP_LEVEL
            @info """GMRES linsolve converged without any iterations:
            * norm of residual = $(normres2string(normr))
            * number of operations = 1"""
        end
        return (x, ConvergenceInfo(1, r, normr, 0, 1))
    elseif alg.verbosity >= STARTSTOP_LEVEL
        @info "GMRES linsolve starts with norm of residual = $(normres2string(normr))"
    end

    # Initialize data structures
    y = Vector{T}(undef, krylovdim + 1)
    gs = Vector{Givens{T}}(undef, krylovdim)
    R = fill(zero(T), (krylovdim, krylovdim))
    numiter = 0
    numops = 1 # operator has been applied once to determine T and r

    function wrapped_op(v)
        v_out = KrylovKit.apply(operator, v, α₀, α₁)
        return KrylovKit.apply(M⁻¹, v_out)
    end

    iter = ArnoldiIterator(wrapped_op, z, alg.orth)
    fact = initialize(iter; verbosity=SILENT_LEVEL)
    sizehint!(fact, alg.krylovdim)
    numops += 1 # start applies operator once

    while true # restart loop
        numiter += 1
        y[1] = β
        k = 1
        H = rayleighquotient(fact)
        R[1, 1] = H[1, 1]
        gs[1], R[1, 1] = givens(R[1, 1], normres(fact), 1, 2)
        y[2] = zero(T)
        lmul!(gs[1], y)
        β = convert(S, abs(y[2]))

        while (β > tol && length(fact) < krylovdim) # inner arnoldi loop
            if alg.verbosity >= EACHITERATION_LEVEL
                @info "GMRES linsolve in iteration $numiter; step $k: normres = $(normres2string(β))"
            end
            fact = expand!(iter, fact; verbosity=SILENT_LEVEL)
            numops += 1 # expand! applies the operator once
            k = length(fact)
            H = rayleighquotient(fact)

            # copy Arnoldi Hessenberg matrix into R
            @inbounds begin
                for i in 1:(k-1)
                    R[i, k] = H[i, k]
                end
                R[k, k] = H[k, k]
            end

            # Apply Givens rotations
            Rk = view(R, :, k)
            @inbounds for i in 1:(k-1)
                lmul!(gs[i], Rk)
            end
            gs[k], R[k, k] = givens(R[k, k], normres(fact), k, k + 1)

            # Apply Givens rotations to right hand side
            y[k+1] = zero(T)
            lmul!(gs[k], y)

            # New error
            β = convert(S, abs(y[k+1]))
        end

        # Solve upper triangular system
        y2 = copy(y)
        KrylovKit.ldiv!(UpperTriangular(R), y, 1:k)

        # Update x
        V = basis(fact)
        @inbounds for i in 1:k
            x = add!!(x, V[i], y[i])
        end

        if β > tol
            # Recompute residual without reevaluating operator
            w = residual(fact)
            push!(V, scale!!(w, 1 / normres(fact)))
            for i in 1:k
                rmul!(V, gs[i]')
            end
            z = scale!!(z, V[k+1], y[k+1])
        else
            # Recompute residual and its norm explicitly, to ensure that no
            # numerical errors have accumulated
            r = scale!!(r, b, 1)
            r = add!!(r, KrylovKit.apply(operator, x, α₀, α₁), -1)
            normr = norm(r)
            z = KrylovKit.apply(M⁻¹, r)
            numops += 1
            β = norm(z)
            if normr < tol
                if alg.verbosity >= STARTSTOP_LEVEL
                    @info """GMRES linsolve converged at iteration $numiter, step $k:
                    * norm of residual = $(normres2string(normr))
                    * number of operations = $numops"""
                end
                return (x, ConvergenceInfo(1, r, normr, numiter, numops))
            end
        end
        if numiter >= maxiter
            if alg.verbosity >= WARN_LEVEL
                @warn """GMRES linsolve stopped without converging after $numiter iterations:
                * norm of residual = $(normres2string(β))
                * number of operations = $numops"""
            end
            return (x, ConvergenceInfo(0, r, β, numiter, numops))
        end
        if alg.verbosity >= EACHITERATION_LEVEL
            @info "GMRES linsolve in iteration $numiter; step $k: normres = $(normres2string(β))"
        end

        # Restart Arnoldi factorization with new r
        iter = ArnoldiIterator(wrapped_op, z, alg.orth)
        fact = initialize!(iter, fact; verbosity=SILENT_LEVEL)
    end
    return nothing
end

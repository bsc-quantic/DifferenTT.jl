"""
    first_deriv_coeffs(stencil_order::Int=8) -> Dict{Int,Float64}

Return the coefficients ``c_k``, ``-4 \\le k \\le 4``, of the central first-derivative stencil of order ``q`` =
`stencil_order` (2 or 8) for a unit spacing, ``u'(x_i) \\approx \\sum_k c_k u_{i+k}``; the coefficients beyond
``q/2`` are zero. See [`derivative_coefficients`](@ref).
"""
function first_deriv_coeffs(stencil_order::Int=8)
    coeffs = Dict{Int,Float64}()
    if stencil_order == 2
        coeffs[0] = 0.0
        coeffs[1] = 1.0 / 2.0
        coeffs[2] = 0.0
        coeffs[3] = 0.0
        coeffs[4] = 0.0
        for i in range(1, 4)
            coeffs[-i] = -coeffs[i]
        end
    elseif stencil_order == 8
        coeffs[0] = 0.0
        coeffs[1] = 4.0 / 5.0
        coeffs[2] = -1.0 / 5.0
        coeffs[3] = 4.0 / 105.0
        coeffs[4] = -1.0 / 280.0
        for i in range(1, 4)
            coeffs[-i] = -coeffs[i]
        end
    else
        throw(ArgumentError("Only second and eighth order finite difference schemes are implemented"))
    end
    return coeffs
end

"""
    second_deriv_coeffs(stencil_order::Int=8) -> Dict{Int,Float64}

Return the coefficients ``c_k``, ``-4 \\le k \\le 4``, of the central second-derivative stencil of order ``q`` =
`stencil_order` (2 or 8) for a unit spacing; the coefficients beyond ``q/2`` are zero. See
[`derivative_coefficients`](@ref).
"""
function second_deriv_coeffs(stencil_order::Int=8)
    coeffs = Dict{Int,Float64}()
    if stencil_order == 2
        coeffs[0] = -2.0
        coeffs[1] = 1.0
        coeffs[2] = 0.0
        coeffs[3] = 0.0
        coeffs[4] = 0.0
        for i in range(1, 4)
            coeffs[-i] = coeffs[i]
        end
    elseif stencil_order == 8
        coeffs[0] = -205.0 / 72.0
        coeffs[1] = 8.0 / 5.0
        coeffs[2] = -1.0 / 5.0
        coeffs[3] = 8.0 / 315.0
        coeffs[4] = -1.0 / 560.0
        for i in range(1, 4)
            coeffs[-i] = coeffs[i]
        end
    else
        throw(ArgumentError("Only second and eighth order finite difference schemes are implemented"))
    end
    return coeffs
end

"""
    fourth_deriv_coeffs(stencil_order::Int=8) -> Dict{Int,Float64}

Return the coefficients ``c_k``, ``-4 \\le k \\le 4``, of a central fourth-derivative stencil for a unit spacing:
the five-point, second-order stencil for `stencil_order = 2`, and the nine-point stencil for `stencil_order = 8`, which
is only sixth-order accurate (the widest stencil within ``|k| \\le 4``). See [`derivative_coefficients`](@ref).
"""
function fourth_deriv_coeffs(stencil_order::Int=8)
    coeffs = Dict{Int,Float64}()
    if stencil_order == 2
        coeffs[0] = 6.0
        coeffs[1] = -4.0
        coeffs[2] = 1.0
        coeffs[3] = 0.0
        coeffs[4] = 0.0
        for i in range(1, 4)
            coeffs[-i] = coeffs[i]
        end
    elseif stencil_order == 8
        coeffs[0] = 91.0 / 8
        coeffs[1] = -122.0 / 15
        coeffs[2] = 169.0 / 60
        coeffs[3] = -2.0 / 5
        coeffs[4] = 7.0 / 240
        for i in range(1, 4)
            coeffs[-i] = coeffs[i]
        end
    else
        throw(ArgumentError("Only second and eighth order finite difference schemes are implemented"))
    end
    return coeffs
end

"""
    derivative_coefficients(derivative_order::Int; stencil_order::Int=8) -> Dict{Int,Float64}

Return the coefficients ``c_k``, ``-4 \\le k \\le 4``, of the central finite-difference stencil of the derivative
of order ``m`` = `derivative_order` for a unit spacing,

```math
\\frac{d^m u}{dx^m}(x_i) \\approx \\frac{1}{h^m} \\sum_{k=-4}^{4} c_k\\, u_{i+k},
```

where ``h`` is the grid spacing. Every operator of the library is built from these coefficients, as a band Toeplitz
matrix (see [`onedim_differential_operator`](@ref)). The stencils have ``q + 1`` points for the first and second
derivatives, with ``q`` = `stencil_order`, so the coefficients beyond ``|k| = q/2`` are zero, and they are accurate to
order ``q``. The exception is the fourth derivative with ``q = 8``, whose nine-point stencil is accurate to order 6.
The first-derivative coefficients are antisymmetric, ``c_{-k} = -c_k``, and the others symmetric.

# Arguments

  - `derivative_order::Int`: 1, 2 or 4.

# Keywords

  - `stencil_order::Int=8`: ``q``, 2 or 8.

# Throws

  - `ArgumentError`: for any other derivative or stencil order.

# Examples

```jldoctest
julia> c = DifferenTT.derivative_coefficients(2; stencil_order=2);
       [c[k] for k in -1:1]
3-element Vector{Float64}:
  1.0
 -2.0
  1.0

julia> c = DifferenTT.derivative_coefficients(1);
       [c[k] for k in 0:4]
5-element Vector{Float64}:
  0.0
  0.8
 -0.2
  0.0380952380952381
 -0.0035714285714285713
```
"""
function derivative_coefficients(derivative_order::Int; stencil_order::Int=8)
    if derivative_order == 1
        return first_deriv_coeffs(stencil_order)
    elseif derivative_order == 2
        return second_deriv_coeffs(stencil_order)
    elseif derivative_order == 4
        return fourth_deriv_coeffs(stencil_order)
    else
        throw(ArgumentError("Given derivative order = $derivative_order not implemented"))
    end
end

"""
    wall_vorticity_coefficients(formula::Symbol) -> NTuple{3,Float64}

Return the coefficients ``(c_1, c_2, c_U)`` of a wall vorticity formula for no-slip walls placed at ghost nodes,

```math
\\omega_\\text{wall} = \\frac{c_1 \\psi_1 + c_2 \\psi_2}{h^2} + \\frac{c_U U}{h},
```

where ``\\psi_1`` and ``\\psi_2`` are the stream function at the first and second nodes away from the wall, with
``\\psi_\\text{wall} = 0``, and ``U`` is the tangential velocity of a moving wall, with ``c_U`` taken for a top lid
moving in ``+x``.

In the vorticity–streamfunction formulation, the vorticity on a no-slip wall is not prescribed, so it is recovered
from the stream function: expanding ``\\psi`` in a Taylor series at the wall, with ``\\psi = 0`` and
``\\partial \\psi / \\partial n`` given by the wall velocity, and using ``\\omega = -\\Delta \\psi`` there, gives
Thom's first-order formula, ``(c_1, c_2, c_U) = (-2, 0, -2)``, from the first node, and Jensen's second-order formula,
``(-4, 1/2, -3)``, from the first two.

# Arguments

  - `formula::Symbol`: `:jensen` for Jensen's formula, used by Kiffner & Jaksch (2023, Eq. (7)) and identical to Eqs.
    (3.4)–(3.5) of Ghia, Ghia & Shin (1982), or `:thom` for Thom's formula.

# Throws

  - `ArgumentError`: for any other `formula`.

# Examples

```jldoctest
julia> DifferenTT.wall_vorticity_coefficients(:jensen)
(-4.0, 0.5, -3.0)
```

See also the boundary-condition tutorial [Vorticity–streamfunction formulation](@ref bc_vorticity_streamfunction).
"""
function wall_vorticity_coefficients(formula::Symbol)
    if formula == :jensen
        return (-4.0, 0.5, -3.0)
    elseif formula == :thom
        return (-2.0, 0.0, -2.0)
    else
        throw(ArgumentError("Unknown wall vorticity formula $formula; use :jensen or :thom"))
    end
end

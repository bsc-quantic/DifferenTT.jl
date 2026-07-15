"""
    first_deriv_coeffs(stencil_order::Int=8)

Coefficients for 1st derivative central difference stencil.

# Arguments
- `stencil_order::Int`: The order of accuracy (2 or 8).
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
        throw(ArgumentError("Only second and eighth dinite difference schemes are implemented"))
    end
    return coeffs
end

"""
    second_deriv_coeffs(stencil_order::Int=8)

Coefficients for 2nd derivative central difference stencil.

# Arguments
- `stencil_order::Int`: The order of accuracy (2 or 8).
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
        throw(ArgumentError("Only second and eighth dinite difference schemes are implemented"))
    end
    return coeffs
end

"""
    fourth_deriv_coeffs(stencil_order::Int=8)

Coefficients for 4th derivative central difference stencil.

# Arguments
- `stencil_order::Int`: The order of accuracy (2 or 8).
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
        throw(ArgumentError("Only second and eighth dinite difference schemes are implemented"))
    end
    return coeffs
end

"""
    derivative_coefficients(derivative_order::Int; stencil_order::Int=8)

Return a dictionary of finite difference coefficients for a specified derivative order and stencil accuracy.

# Arguments
- `derivative_order::Int`: The order of the derivative (1, 2, or 4).
- `stencil_order::Int`: The order of accuracy (2 or 8).
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
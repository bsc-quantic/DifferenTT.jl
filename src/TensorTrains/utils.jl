"""
    backend_type(array::AbstractArray) -> Type
    backend_type(itensor::ITensor) -> Type
    backend_type(tensortrain::AbstractMPS) -> Type

Return the array type, without its parameters (e.g. `Array` or `CuArray`), that stores `array`, the data of `itensor`,
or the data of the first tensor of `tensortrain`. Wrappers such as `Adjoint`, `Symmetric`, `Diagonal` or a view are
looked through, so `backend_type(A')` is `backend_type(A)`. The constructors of the package take this type as their
`array_type` argument, so that a new tensor train is built on the same backend as the one it is combined with. For an
MPS or MPO we assume, without checking, that all tensors live on the same backend.

# Examples

```jldoctest
julia> backend_type(zeros(2, 2))
Array
```
"""
function backend_type(array::AbstractArray)
    # Look through wrappers (Adjoint, Symmetric, Diagonal, SubArray, ...) to the array that stores the data
    data = parent(array)
    return data === array ? typeof(array).name.wrapper : backend_type(data)
end

function backend_type(itensor::ITensor)
    return typeof(array(itensor)).name.wrapper
end

function backend_type(tensortrain::T) where {T<:AbstractMPS}
    return typeof(array(first(tensortrain))).name.wrapper
end

"""
    ndim_index(onedim_index::Vector{Int}, axis::Int, arch::QTTArchitecture) -> Vector{Int}
    ndim_index(onedim_index::Vector{Int}, axis::Int, arch::TTArchitecture) -> Vector{Int}

Spread the site values `onedim_index` of variable `axis`, given from its first site to its last (most significant
bit first for a QTT variable), over the sites of `arch` that carry it, and set every other site to 1. For a reversed
QTT variable (negative `var_nsites`) the values are reversed first, so that they follow the bit order of `arch`.
Throws a `DimensionMismatch` if `length(onedim_index)` is not the number of sites of `axis`.
"""
function ndim_index(onedim_index::Vector{Int}, axis::Int, arch::QTTArchitecture)
    active_vars = count(==(axis), arch.site_map)
    length(onedim_index) == active_vars || throw(
        DimensionMismatch(
            "Input index length $(length(onedim_index)) does not match sites for axis $axis ($active_vars)"
        ),
    )

    total_sites = length(arch.site_map)

    working_index = arch.var_nsites[axis] < 0 ? reverse(onedim_index) : onedim_index

    full_index = ones(Int, total_sites)

    bit_counter = 1

    for i in 1:total_sites
        if arch.site_map[i] == axis
            full_index[i] = working_index[bit_counter]
            bit_counter += 1
        end
    end

    return full_index
end

function ndim_index(onedim_index::Vector{Int}, axis::Int, arch::TTArchitecture)
    active_vars = count(==(axis), arch.site_map)
    length(onedim_index) == active_vars || throw(
        DimensionMismatch(
            "Input index length $(length(onedim_index)) does not match sites for axis $axis ($active_vars)"
        ),
    )

    total_sites = length(arch.site_map)

    full_index = ones(Int, total_sites)

    bit_counter = 1

    for i in 1:total_sites
        if arch.site_map[i] == axis
            full_index[i] = onedim_index[bit_counter]
            bit_counter += 1
        end
    end

    return full_index
end

"""
    decimal_to_base(number::Int, base::Int, ndigits::Int) -> Vector{Int}

Return the `ndigits` digits of the non-negative integer `number` in base `base`, the most significant first, as in the
site order of a QTT variable. A `number` that needs more than `ndigits` digits returns all of them.
"""
function decimal_to_base(number::Int, base::Int, ndigits::Int)
    return reverse(digits(number; base=base, pad=ndigits))
end

"""
    base_to_decimal(digits::AbstractVector{Int}, base::Int) -> Int

Return the integer whose digits in base `base`, the most significant first, are `digits`; the inverse of
[`decimal_to_base`](@ref).
"""
function base_to_decimal(digits::AbstractVector{Int}, base::Int)
    return sum(digits[i] * base^(length(digits) - i) for i in 1:length(digits))
end

"""
    decimal_to_binary(number::Int, nbits::Int) -> Vector{Int}

Return the `nbits` bits of `number`, the most significant first; `decimal_to_base(number, 2, nbits)`.
"""
decimal_to_binary(number::Int, nbits::Int) = decimal_to_base(number, 2, nbits)

"""
    binary_to_decimal(bits::AbstractVector{Int}) -> Int

Return the integer whose bits, the most significant first, are `bits`; `base_to_decimal(bits, 2)`.
"""
binary_to_decimal(bits::AbstractVector{Int}) = base_to_decimal(bits, 2)

"""
    identity_array(n::Int) -> Matrix{Float64}

Return the dense ``n \\times n`` identity matrix.
"""
identity_array(n::Int) = Matrix{Float64}(I, n, n)

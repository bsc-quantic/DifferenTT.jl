"""
    backend_type(array::AbstractArray)

Return the backend array type (wrapper) of an `AbstractArray` (e.g., `Array`, `CuArray`).

# Arguments
- `array::AbstractArray`: The array to inspect.
"""
function backend_type(array::AbstractArray)
    return typeof(array).name.wrapper
end

"""
    backend_type(itensor::ITensor)

Return the backend array type of the storage of an `ITensor`.

# Arguments
- `itensor::ITensor`: The ITensor to inspect.
"""
function backend_type(itensor::ITensor)
    return typeof(array(itensor)).name.wrapper
end

"""
    backend_type(tensortrain::T) where T<:AbstractMPS

Return the backend array type of the first tensor in an MPS/MPO. It is assumed that all the tensors in the MPS/MPO
live in the same memory.

# Arguments
- `tensortrain::T`: The MPS or MPO to inspect.
"""
function backend_type(tensortrain::T) where T<:AbstractMPS
    return typeof(array(first(tensortrain))).name.wrapper
end

"""
    ndim_index(onedim_index::Vector{Int}, axis::Int, arch::QTTArchitecture) -> Vector{Int}

Expands an index (e.g., bit string) for a specific `axis` into the full N-dimensional 
representation defined by `arch`. Sites not belonging to `axis` are filled with 1.
"""
function ndim_index(onedim_index::Vector{Int}, axis::Int, arch::QTTArchitecture)
    active_vars = count(==(axis), arch.site_map)
    @assert length(onedim_index) == active_vars "Input index length $(length(onedim_index)) does not match sites for axis $axis ($active_vars)"
    
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

"""
    ndim_index(onedim_index::Vector{Int}, axis::Int, arch::TTArchitecture) -> Vector{Int}

Expands a 1D index for a specific `axis` into the full N-dimensional 
representation defined by `arch`. Sites not belonging to `axis` are filled with 1.
"""
function ndim_index(onedim_index::Vector{Int}, axis::Int, arch::TTArchitecture)
    active_vars = count(==(axis), arch.site_map)
    @assert length(onedim_index) == active_vars "Input index length $(length(onedim_index)) does not match sites for axis $axis ($active_vars)"
    
    total_sites = length(arch.site_map)
    
    full_index = ones(Int, total_sites)
    
    bit_counter = 1decimal_
    
    for i in 1:total_sites
        if arch.site_map[i] == axis
            full_index[i] = onedim_index[bit_counter]
            bit_counter += 1
        end
    end
    
    return full_index
end

"""
    decimal_to_base(number::Int, base::Int, ndigits::Int)

Convert an integer `number` to a base `base` representation as a vector of integers with length `ndigits`, 
ordered from most significant to least significant (big-endian).

# Arguments
- `number::Int`: The decimal number to convert.
- `base::Int`: The base of the output representation.
- `ndigits::Int`: The number of digits in the output representation.
"""
function decimal_to_base(number::Int, base::Int, ndigits::Int)
    return reverse(digits(number, base=base, pad=ndigits))
end

"""
    base_to_decimal(digits::AbstractVector{Int}, base::Int)

Convert a base `base` representation `digits` (ordered from most significant to least significant) 
to the corresponding decimal integer.
"""
function base_to_decimal(digits::AbstractVector{Int}, base::Int)
    return sum(digits[i] * base^(length(digits) - i) for i in 1:length(digits))
end

"""
    decimal_to_binary(number::Int, nbits::Int)

Convert an integer `number` to a binary representation as a vector of integers (0s and 1s) with length `nbits`.
This is equivalent to `decimal_to_base(number, 2, nbits)`.

# Arguments
- `number::Int`: The decimal number to convert.
- `nbits::Int`: The number of bits in the output representation.
"""
decimal_to_binary(number::Int, nbits::Int) = decimal_to_base(number, 2, nbits)


"""
    binary_to_decimal(bits::AbstractVector{Int})

Convert a binary representation `bits` (vector of 0s and 1s) to the corresponding decimal integer.
This is equivalent to `base_to_decimal(bits, 2)`.
"""
binary_to_decimal(bits::AbstractVector{Int}) = base_to_decimal(bits, 2)

"""
    identity_array(n::Int)

Return an `n x n` dense identity matrix.
"""
identity_array(n::Int) = Matrix{Float64}(I, n, n)

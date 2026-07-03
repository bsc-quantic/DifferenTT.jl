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
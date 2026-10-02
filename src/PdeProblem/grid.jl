"""
    struct Grid{N}

Represents a physical spatial grid.

# Fields
- `box_size::NTuple{N, Float64}`: The physical size of the domain in each dimension.
- `dx::NTuple{N, Float64}`: The grid spacing in each dimension.
- `points::NTuple{N, Vector{Float64}}`: The coordinate points for each dimension.
"""
struct Grid{N}
    box_size::NTuple{N, Float64}
    dx::NTuple{N, Float64}
    points::NTuple{N, Vector{Float64}}
end

"""
    Grid(arch::QTTArchitecture{N}, box_size::NTuple{N, Real}) where N

Construct a `Grid` from a `QTTArchitecture` and a given `box_size`.
"""
function Grid(arch::QTTArchitecture{N}, box_size::NTuple{N, Real}) where N
    box_size_f = Float64.(box_size)
    dims = variable_dims(arch)
    
    # Calculate grid spacing assuming points are placed from 0 to box_size inclusive
    dx = box_size_f ./ (dims .- 1)
    points = ntuple(i -> collect(range(0.0, box_size_f[i], length=dims[i])), N)
    
    return Grid{N}(box_size_f, dx, points)
end

"""
    Grid(arch::TTArchitecture{N}, box_size::NTuple{N, Real}) where N

Construct a `Grid` from a `TTArchitecture` and a given `box_size`.
"""
function Grid(arch::TTArchitecture{N}, box_size::NTuple{N, Real}) where N
    box_size_f = Float64.(box_size)
    dims = variable_dims(arch)
    
    # Calculate grid spacing assuming points are placed from 0 to box_size inclusive
    dx = box_size_f ./ (dims .- 1)
    points = ntuple(i -> collect(range(0.0, box_size_f[i], length=dims[i])), N)
    
    return Grid{N}(box_size_f, dx, points)
end

# Convenience constructors for 1D
Grid(arch::QTTArchitecture{1}, box_size::Real) = Grid(arch, (box_size,))
Grid(arch::TTArchitecture{1}, box_size::Real) = Grid(arch, (box_size,))

"""
    interior_grid(arch::Union{QTTArchitecture{N}, TTArchitecture{N}}, domain_size::NTuple{N, Real}) where N

Construct the `Grid` of the interior nodes of the domain `[0, L₁] × ... × [0, Lₙ]` when its walls are ghost nodes
that are not stored: with `n` nodes per axis, the spacing is `h = L / (n + 1)` and the stored nodes sit at the
physical positions `h, 2h, ..., n h`.

The returned grid spans the interior box of size `L (n - 1) / (n + 1)`, so that its spacing is exactly `h` and
sampling, cross interpolation and plotting work unchanged. Grid coordinates are physical coordinates minus `h`.
"""
function interior_grid(arch::Union{QTTArchitecture{N}, TTArchitecture{N}}, domain_size::NTuple{N, Real}) where N
    dims = variable_dims(arch)
    interior_size = ntuple(i -> Float64(domain_size[i]) * (dims[i] - 1) / (dims[i] + 1), N)
    return Grid(arch, interior_size)
end

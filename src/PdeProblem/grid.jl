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
    return interior_grid(arch, domain_size, ntuple(_ -> 2, N))
end

"""
    interior_grid(arch::Union{QTTArchitecture{N}, TTArchitecture{N}}, domain_size::NTuple{N, Real}, ghost_sides::NTuple{N, Int}) where N

General version of [`interior_grid`](@ref) where axis `i` has `ghost_sides[i] ∈ 0:2` sides whose boundary is a
ghost node that is not stored. With `n` nodes, the spacing is `h = L / (n - 1 + g)`:
- `g = 2`: both boundaries are ghost nodes (walls), the stored nodes sit at `h, ..., n h`.
- `g = 1`: one boundary is a ghost node and the other one is stored (e.g. inflow/outflow), or the axis is
  periodic (the node at `L` is the image of the node at `0`).
- `g = 0`: both boundaries are stored nodes.

Grid coordinate 0 is the physical coordinate `h` when the left side of the axis is a ghost node, and physical 0
otherwise.
"""
function interior_grid(arch::Union{QTTArchitecture{N}, TTArchitecture{N}}, domain_size::NTuple{N, Real}, ghost_sides::NTuple{N, Int}) where N
    all(g -> g in 0:2, ghost_sides) || throw(ArgumentError("ghost_sides must be in 0:2 on every axis, got $ghost_sides"))
    dims = variable_dims(arch)
    h = ntuple(i -> Float64(domain_size[i]) / (dims[i] - 1 + ghost_sides[i]), N)
    return Grid(arch, ntuple(i -> h[i] * (dims[i] - 1), N))
end

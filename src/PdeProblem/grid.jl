"""
    struct Grid{N}

Uniform Cartesian grid of the stored nodes of a field in ``N`` dimensions.

Every axis ``i`` of the box ``[0, L_1] \\times \\dots \\times [0, L_N]`` holds ``n_i`` nodes, one at each end, so the
spacing is ``h_i = L_i/(n_i - 1)`` and node ``j`` (1-based) sits at ``(j - 1) h_i``. This is the convention of the
velocity–pressure formulation, where the boundary nodes are stored. When the boundaries are ghost nodes that are not
stored, as in the vorticity–streamfunction formulation, the grid of the stored nodes is built by
[`interior_grid`](@ref) or [`vorticity_grid`](@ref), which return a `Grid` of the same type with its coordinates
shifted (see there).

# Fields

  - `box_size::NTuple{N,Float64}`: the side lengths ``L_i`` spanned by the stored nodes.
  - `dx::NTuple{N,Float64}`: the spacings ``h_i``.
  - `points::NTuple{N,Vector{Float64}}`: the node coordinates of each axis, from 0 to ``L_i``.

See also [`interior_grid`](@ref), [`vorticity_grid`](@ref).
"""
struct Grid{N}
    box_size::NTuple{N,Float64}
    dx::NTuple{N,Float64}
    points::NTuple{N,Vector{Float64}}
end

"""
    Grid(arch::QTTArchitecture{N}, box_size::NTuple{N,Real}) where N
    Grid(arch::TTArchitecture{N}, box_size::NTuple{N,Real}) where N
    Grid(arch::QTTArchitecture{1}, box_size::Real)
    Grid(arch::TTArchitecture{1}, box_size::Real)

Build the [`Grid`](@ref) of the box ``[0, L_1] \\times \\dots \\times [0, L_N]``, with `box_size`
``= (L_1, \\dots, L_N)``, and the number of nodes of each axis taken from `arch` (`variable_dims(arch)`: ``2^{N_i}`` in
QTT, the variable dimension in TT). The nodes include both ends of each axis, so ``h_i = L_i/(n_i - 1)``. The
one-dimensional methods accept a scalar `box_size`.

# Examples

```jldoctest
julia> grid = Grid(QTTArchitecture((3,)), 1.0);

julia> grid.dx, grid.points[1][end]
((0.14285714285714285,), 1.0)

julia> Grid(TTArchitecture((5, 3)), (2.0, 1.0)).dx
(0.5, 0.5)
```
"""
function Grid(arch::QTTArchitecture{N}, box_size::NTuple{N,Real}) where {N}
    box_size_f = Float64.(box_size)
    dims = variable_dims(arch)

    # Calculate grid spacing assuming points are placed from 0 to box_size inclusive
    dx = box_size_f ./ (dims .- 1)
    points = ntuple(i -> collect(range(0.0, box_size_f[i]; length=dims[i])), N)

    return Grid{N}(box_size_f, dx, points)
end

function Grid(arch::TTArchitecture{N}, box_size::NTuple{N,Real}) where {N}
    box_size_f = Float64.(box_size)
    dims = variable_dims(arch)

    # Calculate grid spacing assuming points are placed from 0 to box_size inclusive
    dx = box_size_f ./ (dims .- 1)
    points = ntuple(i -> collect(range(0.0, box_size_f[i]; length=dims[i])), N)

    return Grid{N}(box_size_f, dx, points)
end

# Convenience constructors for 1D
Grid(arch::QTTArchitecture{1}, box_size::Real) = Grid(arch, (box_size,))
Grid(arch::TTArchitecture{1}, box_size::Real) = Grid(arch, (box_size,))

"""
    interior_grid(arch::Union{QTTArchitecture{N},TTArchitecture{N}}, domain_size::NTuple{N,Real}) where N
    interior_grid(arch::Union{QTTArchitecture{N},TTArchitecture{N}}, domain_size::NTuple{N,Real},
                  ghost_sides::NTuple{N,Int}) where N

Build the [`Grid`](@ref) of the stored nodes of the domain ``[0, L_1] \\times \\dots \\times [0, L_N]`` when some of
its boundaries are ghost nodes that are not stored.

In the vorticity–streamfunction formulation, walls and inflow faces fix their values one spacing outside the stored
nodes, so the spacing is no longer ``L/(n - 1)``. Axis ``i`` has ``g_i`` = `ghost_sides[i]` ``\\in \\{0, 1, 2\\}``
ghost sides (2 for the two-argument method), and its ``n_i`` stored nodes are spaced by

```math
h_i = \\frac{L_i}{n_i - 1 + g_i},
```

where ``g = 2`` means that both boundaries are ghost nodes (walls), and the stored nodes sit at ``h, 2h, \\dots, n h``;
``g = 1`` means that one boundary is a ghost node and the other one a stored node (an inflow and an outflow), or that
the axis is periodic, in which case the node at ``L`` is the image of the node at 0; and ``g = 0`` means that both
boundaries are stored nodes, as in [`Grid`](@ref).

The returned grid spans the box of side ``h_i (n_i - 1)`` with spacing exactly ``h_i``, so that sampling, cross
interpolation and plotting work unchanged. Its coordinates are therefore shifted: grid coordinate 0 is the physical
coordinate ``h_i`` when the left side of the axis is a ghost node, and the physical 0 otherwise.

# Throws

  - `ArgumentError`: if a `ghost_sides` entry is not in `0:2`.

# Examples

```jldoctest
julia> grid = interior_grid(QTTArchitecture((3,)), (1.0,));

julia> grid.dx, grid.box_size
((0.11111111111111109,), (0.7777777777777777,))

julia> interior_grid(TTArchitecture((4,)), (1.0,), (1,)).dx
(0.25,)
```

See also [`vorticity_grid`](@ref), and the boundary-condition tutorial
[Vorticity–streamfunction formulation](@ref bc_vorticity_streamfunction).
"""
function interior_grid(arch::Union{QTTArchitecture{N},TTArchitecture{N}}, domain_size::NTuple{N,Real}) where {N}
    return interior_grid(arch, domain_size, ntuple(_ -> 2, N))
end

function interior_grid(
    arch::Union{QTTArchitecture{N},TTArchitecture{N}}, domain_size::NTuple{N,Real}, ghost_sides::NTuple{N,Int}
) where {N}
    all(g -> g in 0:2, ghost_sides) ||
        throw(ArgumentError("ghost_sides must be in 0:2 on every axis, got $ghost_sides"))
    dims = variable_dims(arch)
    h = ntuple(i -> Float64(domain_size[i]) / (dims[i] - 1 + ghost_sides[i]), N)
    return Grid(arch, ntuple(i -> h[i] * (dims[i] - 1), N))
end

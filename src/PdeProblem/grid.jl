"""
    struct Grid{N}

Uniform Cartesian grid of the stored nodes of a field in ``N`` dimensions.

Every axis ``i`` of the box ``[0, L_1] \\times \\dots \\times [0, L_N]`` holds ``n_i`` nodes, one at each end, so the
spacing is ``h_i = L_i/(n_i - 1)`` and node ``j`` (1-based) sits at ``(j - 1) h_i``. This is the convention of the
velocity–pressure formulation, where the boundary nodes are stored. When the boundaries are ghost nodes that are not
stored, as in the vorticity–streamfunction formulation, the grid of the stored nodes is built by
[`interior_grid`](@ref) or [`vorticity_grid`](@ref), which return a `Grid` of the same type with its coordinates
shifted (see there).

A periodic axis needs a different spacing: the node at ``L`` is the image of the node at 0 and is not stored, so the
``n`` stored nodes are spaced by ``h = L/n`` and the period of the wrapped stencils is ``L``. The grid records which
axes are periodic in `periodic`, which is data about the grid alone (plain `Bool`s, not boundary-condition types);
[`periodic_axes`](@ref) derives it from the boundary conditions of a field, and the experiments check that the two
agree.

# Fields

  - `box_size::NTuple{N,Float64}`: the side lengths spanned by the stored nodes, ``h_i (n_i - 1)``; this is ``L_i``
    except on periodic axes, where it is ``L_i - h_i``.
  - `dx::NTuple{N,Float64}`: the spacings ``h_i``.
  - `points::NTuple{N,Vector{Float64}}`: the node coordinates of each axis, from 0 to `box_size[i]`.
  - `periodic::NTuple{N,Bool}`: whether each axis is periodic. The three-argument constructor
    `Grid{N}(box_size, dx, points)` sets every axis to non-periodic.

See also [`interior_grid`](@ref), [`vorticity_grid`](@ref).
"""
struct Grid{N}
    box_size::NTuple{N,Float64}
    dx::NTuple{N,Float64}
    points::NTuple{N,Vector{Float64}}
    periodic::NTuple{N,Bool}
end

Grid{N}(box_size, dx, points) where {N} = Grid{N}(box_size, dx, points, ntuple(_ -> false, N))

"""
    _grid_axes(dims, box_size::NTuple{N,Float64}, periodic::NTuple{N,Bool}) where N

Return the spans, spacings and node coordinates of the axes of a [`Grid`](@ref): ``h_i = L_i/(n_i - 1)`` with nodes
at both ends, or ``h_i = L_i/n_i`` with the node at ``L_i`` left out on a periodic axis, whose span is then
``L_i - h_i``.
"""
function _grid_axes(dims, box_size::NTuple{N,Float64}, periodic::NTuple{N,Bool}) where {N}
    span = ntuple(i -> periodic[i] ? box_size[i] - box_size[i] / dims[i] : box_size[i], N)
    dx = ntuple(i -> periodic[i] ? box_size[i] / dims[i] : box_size[i] / (dims[i] - 1), N)
    points = ntuple(i -> collect(range(0.0, span[i]; length=dims[i])), N)
    return span, dx, points
end

"""
    Grid(arch::QTTArchitecture{N}, box_size::NTuple{N,Real}; periodic::NTuple{N,Bool}=ntuple(_ -> false, N)) where N
    Grid(arch::TTArchitecture{N}, box_size::NTuple{N,Real}; periodic::NTuple{N,Bool}=ntuple(_ -> false, N)) where N
    Grid(arch::QTTArchitecture{1}, box_size::Real; periodic::Bool=false)
    Grid(arch::TTArchitecture{1}, box_size::Real; periodic::Bool=false)

Build the [`Grid`](@ref) of the box ``[0, L_1] \\times \\dots \\times [0, L_N]``, with `box_size`
``= (L_1, \\dots, L_N)``, and the number of nodes of each axis taken from `arch` (`variable_dims(arch)`: ``2^{N_i}`` in
QTT, the variable dimension in TT). The nodes include both ends of each non-periodic axis, so ``h_i = L_i/(n_i - 1)``.
On an axis flagged in `periodic`, the node at ``L_i`` is the image of the node at 0, so ``h_i = L_i/n_i`` and the
wrapped stencils of a [`PeriodicBC`](@ref) have period ``L_i``; without the flag, their period is ``L_i + h_i``. Pass
`periodic = periodic_axes(bcs)` to take the flags from the boundary conditions of a field. The one-dimensional methods
accept a scalar `box_size` and a scalar `periodic`.

# Examples

```jldoctest
julia> grid = Grid(QTTArchitecture((3,)), 1.0);

julia> grid.dx, grid.points[1][end]
((0.14285714285714285,), 1.0)

julia> Grid(TTArchitecture((5, 3)), (2.0, 1.0)).dx
(0.5, 0.5)

julia> grid = Grid(TTArchitecture((4, 4)), (1.0, 1.0); periodic=(false, true));

julia> grid.dx, grid.points[2][end]
((0.3333333333333333, 0.25), 0.75)
```
"""
function Grid(
    arch::QTTArchitecture{N}, box_size::NTuple{N,Real}; periodic::NTuple{N,Bool}=ntuple(_ -> false, N)
) where {N}
    box_size_f = Float64.(box_size)
    dims = variable_dims(arch)

    # Points from 0 to box_size inclusive, or up to box_size - dx on a periodic axis
    span, dx, points = _grid_axes(dims, box_size_f, periodic)

    return Grid{N}(span, dx, points, periodic)
end

function Grid(
    arch::TTArchitecture{N}, box_size::NTuple{N,Real}; periodic::NTuple{N,Bool}=ntuple(_ -> false, N)
) where {N}
    box_size_f = Float64.(box_size)
    dims = variable_dims(arch)

    # Points from 0 to box_size inclusive, or up to box_size - dx on a periodic axis
    span, dx, points = _grid_axes(dims, box_size_f, periodic)

    return Grid{N}(span, dx, points, periodic)
end

# Convenience constructors for 1D
Grid(arch::QTTArchitecture{1}, box_size::Real; periodic::Bool=false) = Grid(arch, (box_size,); periodic=(periodic,))
Grid(arch::TTArchitecture{1}, box_size::Real; periodic::Bool=false) = Grid(arch, (box_size,); periodic=(periodic,))

"""
    interior_grid(arch::Union{QTTArchitecture{N},TTArchitecture{N}}, domain_size::NTuple{N,Real}) where N
    interior_grid(arch::Union{QTTArchitecture{N},TTArchitecture{N}}, domain_size::NTuple{N,Real},
                  ghost_sides::NTuple{N,Int}; periodic::NTuple{N,Bool}=ntuple(_ -> false, N)) where N

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

The `periodic` flags are stored in the grid (see [`Grid`](@ref)); a periodic axis must have one ghost side, which
gives the same spacing ``L_i/n_i`` as `Grid(...; periodic)`.

# Throws

  - `ArgumentError`: if a `ghost_sides` entry is not in `0:2`, or a periodic axis does not have one ghost side.

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
    arch::Union{QTTArchitecture{N},TTArchitecture{N}},
    domain_size::NTuple{N,Real},
    ghost_sides::NTuple{N,Int};
    periodic::NTuple{N,Bool}=ntuple(_ -> false, N),
) where {N}
    all(g -> g in 0:2, ghost_sides) ||
        throw(ArgumentError("ghost_sides must be in 0:2 on every axis, got $ghost_sides"))
    all(i -> !periodic[i] || ghost_sides[i] == 1, 1:N) ||
        throw(ArgumentError("a periodic axis must have one ghost side, got ghost_sides = $ghost_sides"))
    dims = variable_dims(arch)
    h = ntuple(i -> Float64(domain_size[i]) / (dims[i] - 1 + ghost_sides[i]), N)
    grid = Grid(arch, ntuple(i -> h[i] * (dims[i] - 1), N))
    return Grid{N}(grid.box_size, grid.dx, grid.points, periodic)
end

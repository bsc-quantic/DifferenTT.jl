```@meta
CurrentModule = DifferenTT
DocTestSetup = :(using DifferenTT, ITensors, ITensorMPS)
```

# Grids, fields and boundary conditions

A finite-difference operator needs the values of the field at nodes that are not stored, the ghost nodes beyond the
boundary. The guiding idea of DifferenTT is that every closure is the interior central stencil, completed by values at
those ghost nodes, and that a boundary condition is the rule that gives these values. When the rule involves the
unknowns, substituting it changes the operator; when it involves only data, it adds a known vector to the right-hand
side. This page describes the objects that carry this information, at the level of which type to pass and what it
imposes. The derivation of each closure is in the two boundary-condition tutorials
([Velocity–pressure formulation](@ref bc_velocity_pressure) and
[Vorticity–streamfunction formulation](@ref bc_vorticity_streamfunction)), and the exact rules are in the docstrings
linked below.

## Boundary conditions

The condition of one side of an axis is a [`BoundaryCond`](@ref), and the pair of sides of an axis is a
[`BoundaryCond1d`](@ref). The conditions of all axes make up [`BoundaryConditions`](@ref).

| Type | What it imposes |
|---|---|
| [`PeriodicBC`](@ref) | The stencil wraps around the axis. It must be set on both sides. |
| [`OpenBC`](@ref) | The stencil entries outside the grid are dropped; nothing else acts on this side. |
| [`DirichletBC`](@ref)`(value)` | The stencil is truncated, and the value enters the right-hand side through a lift. |
| [`NeumannBC`](@ref)`(0)` | The ghost nodes mirror the interior, ``u_{-k} = u_k``. Only the homogeneous condition is supported. |

```jldoctest
julia> bc_x = BoundaryCond1d(DirichletBC(1.0), NeumannBC(0.0));

julia> bc_y = BoundaryCond1d(PeriodicBC(), PeriodicBC());

julia> bcs = BoundaryConditions(bc_x, bc_y);

julia> periodic_axes(bcs)
(false, true)
```

Two details matter in practice. First, a Dirichlet value sits on the first ghost node, one spacing beyond the last
stored node, and the lift fills every ghost node with it. This is exact for the stencil order ``q = 2``, but with
``q = 8`` and a solution with a nonzero slope at the boundary the closure is only first-order accurate. This is a known
limitation of v0.1 ([Navier–Stokes](@ref), issue [#51](https://github.com/bsc-quantic/DifferenTT.jl/issues/51)). Second,
the mirrored Neumann closure breaks the symmetry of the Laplacian, and [`neumann_weight_mpo`](@ref) restores it for the
linear systems; [`symmetrized_laplacian`](@ref) applies it for us.

## Grids

The default [`Grid`](@ref) places a node at each end of every axis, so ``n`` nodes span a box of length ``L`` with
spacing ``h = L/(n-1)``. This is the convention of the velocity–pressure formulation, where the boundary nodes are
stored.

```jldoctest
julia> grid = Grid(QTTArchitecture((3,)), 1.0);

julia> grid.dx, grid.points[1][end]
((0.14285714285714285,), 1.0)
```

A periodic axis needs a different spacing: the node at ``L`` is the image of the node at 0, so the stored nodes are spaced
by ``h = L/n`` and the wrapped stencils have period ``L``. The grid records this in its field `periodic`, which we set with
the keyword of the constructor, `Grid(arch, box_size; periodic = periodic_axes(bcs))`. **Without the flag, a
periodic axis keeps its node at ``L`` and the period of the stencils is ``L + h``**, a discretization error of order
``h`` in the geometry. The default is unchanged from earlier versions, and the flag is opt-in.
[`INSExperiment`](@ref) throws an `ArgumentError` if a grid axis is periodic and the boundary conditions are not (the
reverse is accepted).

When the boundaries of the vorticity–streamfunction formulation are ghost nodes that are not stored, the spacing is
``h_i = L_i/(n_i - 1 + g_i)`` with ``g_i`` ghost sides on axis ``i``. [`interior_grid`](@ref) builds the grid of the
stored nodes for a given number of ghost sides, and [`vorticity_grid`](@ref) derives it from the boundary faces of an
experiment. The returned `Grid` has the same type as the default one, with its coordinates shifted, so that sampling and
plotting work unchanged.

## Fields

A [`ScalarField`](@ref) is a time series of MPS on fixed site indices, together with the architecture and the boundary
conditions of each axis. It grows with `push!` as an evolution stores its checkpoints; `field[k]` is the ``k``-th state
and `last(field)` the latest one. A [`VectorField`](@ref) holds the components of a vector quantity, each one a
`ScalarField` with its own boundary conditions, which lets the velocity components differ at the walls.

```julia
arch = QTTArchitecture((5, 4))
p = ScalarField(arch, bcs)                       # empty field, to be filled by an evolution
u = ScalarField(ones_mps(p.usites, 1), arch, bcs) # one stored state
```

Fields are built on the array backend `array_type` (default `Array`); [GPU (experimental)](@ref) lists what is tested.

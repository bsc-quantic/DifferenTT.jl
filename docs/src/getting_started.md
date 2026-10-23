```@meta
CurrentModule = DifferenTT
```

# Getting started

In this page we build and solve a two-dimensional Poisson problem in the QTT format, one step at a time. The guiding
idea is that every object of the library is either an MPS, for a field, or an MPO, for an operator, defined on a list
of site indices that all of them share. We first describe the layout of the sites, then the grid and the boundary
conditions that turn the layout into an operator, and finally the solver. The remaining pages of the manual describe each
step in detail.

!!! note "Precompilation"
    The first `using DifferenTT` after installing or updating the package precompiles it, and runs on small problems
    the common paths of the library (QTT operators, solvers, preconditioners and Navier–Stokes steps), so that their
    first calls in every later session are several times faster. This takes about two minutes once. Since it is
    repeated after every change to the source, we recommend to turn it off while developing the package, with
    `DifferenTT.set_precompile_workload(false)` and a restart of Julia; see [`set_precompile_workload`](@ref).

```@example gs
using DifferenTT, ITensors, ITensorMPS, TensorCrossInterpolation, LinearAlgebra
nothing # hide
```

## The layout

A quantized tensor train writes the index of a coordinate with ``2^n`` values in binary, ``i = \sum_{p=1}^{n} i_p
2^{n-p}``, and gives each bit its own site of dimension ``d = 2``. A ``2^5 \times 2^5`` grid therefore takes ``N = 10``
sites, and the smoothness of a field along each coordinate appears as a low bond dimension ``\chi`` between the bits.
The [`QTTArchitecture`](@ref) records which variable each site carries:

```@example gs
arch = QTTArchitecture((5, 5))
(total_sites(arch), variable_dims(arch), Tuple(arch.site_map))
```

Here the sites are blocked, the first five carrying the first variable. Passing `interleaved=true` alternates the
variables along the chain instead. Site 1 is always the most significant index, a convention that
[Architectures and conventions](@ref) states in full.

## The grid, the boundary conditions and the operator

The [`Grid`](@ref) places the nodes in the box ``[0, 1]^2``, with a node at each end of every axis. The boundary
conditions of each axis are a [`BoundaryCond1d`](@ref) of its two sides, and a [`ScalarField`](@ref) bundles them with
the architecture and the site indices:

```@example gs
grid = Grid(arch, (1.0, 1.0))
bc = BoundaryCond1d(DirichletBC(0.0), DirichletBC(0.0))
field = ScalarField(arch, BoundaryConditions(bc, bc))
grid.dx
```

The operator is built from the field and the grid. We use [`symmetrized_laplacian`](@ref), which is the finite-difference
Laplacian with its boundary closures, and change its sign so that the operator ``A = -\Delta`` is symmetric positive
definite (SPD), as CG requires. Stencil order 2 is the three-point stencil; the default, 8, is more accurate on smooth
fields but wider.

```@example gs
A = -1.0 * symmetrized_laplacian(field, grid; stencil_order=2)
maxlinkdim(A)
```

## The right-hand side

A function of the coordinates enters the TT format by cross interpolation, which evaluates it only at the entries that it
selects. The function must not vanish at the first pivot, the node at the origin, so we use a Gaussian centered in the
box. The result is an MPS on its own site indices, which we replace by those of the field so that `A` and `b` match:

```@example gs
f(x) = exp(-20 * ((x[1] - 0.5)^2 + (x[2] - 0.5)^2))
tci, = crossinterpolate2(Float64, f, grid, arch; tolerance=1e-8)
b = replace_siteinds!(MPS(tci), field.usites)
maxlinkdim(b)
```

## The solver

[`ResidualALS`](@ref) is the alternating solver that we recommend for SPD systems. It sweeps along the chain, solves a local
problem on two sites by CG, and truncates the SVD that splits the two-site tensor according to the residual, so that
the bond dimension is the smallest one that meets the requested accuracy. The three argument structures set the sweeps
([`SweepingArgs`](@ref)), the local solver ([`LocalUpdaterArgs`](@ref)), and, through the
[`ResidualObserver`](@ref), the relative residual at which the sweeps stop. KrylovKit selects CG for the local problems
only if both `ishermitian` and `isposdef` are set.

```@example gs
solver = ResidualALS(;
    sweep_kwargs=SweepingArgs(; nsweeps=10, maxdim=20, cutoff=0.0, (observer!)=ResidualObserver(; rtol=1e-8)),
    updater_kwargs=LocalUpdaterArgs(; ishermitian=true, isposdef=true, rtol=1e-10),
)
x = linearsolve(solver, A, b, nothing)
norm(apply(A, x) - b) / norm(b)
```

The last argument of [`linearsolve`](@ref) is the initial guess, and `nothing` starts from zero. A preconditioner goes
before it; [Preconditioners](@ref) explains how to build the FAI preconditioner and when it pays off.

## Reading the solution

An MPS is evaluated at a point of the box with [`DifferenTT.sample`](@ref), and the dense vector of a small system is
recovered by [`contract_mps`](@ref), with site 1 as the most significant index. The name `sample` also exists in
ITensorMPS, which is why we qualify it:

```@example gs
DifferenTT.sample(x, (0.5, 0.5), grid, arch)
```

```@example gs
size(contract_mps(x)), maxlinkdim(x)
```

Note that this solution is discrete: with Dirichlet conditions the boundary value sits one spacing beyond the last stored
node, as [Grids, fields and boundary conditions](@ref) explains, so it should be compared with a continuous solution only
with that in mind.

## Development

When we edit the source of DifferenTT, the package is precompiled again at the next `using DifferenTT`, and the
precompilation workload of the note above makes each of these rebuilds take about two minutes instead of a few seconds.
In order to keep the edit–test cycle short, we recommend to turn the workload off on a development machine, for every
environment at once:

```julia
using DifferenTT
DifferenTT.set_precompile_workload(false; default_environment=true)
```

The setting is a preference of the package, stored in the `LocalPreferences.toml` of the default environment `@v#.#`, which
every environment whose load path includes it inherits, and it takes effect after a restart of Julia. Without
`default_environment=true` it is written to the active environment only, and the active environment takes precedence over
the default one. The documentation workflow keeps the workload on: it builds the package once per run, and the workload
saves most of the compilation time of the tutorials. We turn the workload on again with
`DifferenTT.set_precompile_workload(true)`.

## Next steps

  - The [Tutorials](@ref tutorials_overview) repeat this construction with error studies and figures.
  - The manual pages [Linear solvers](@ref manual_linear_solvers) and [Preconditioners](@ref) compare the solvers of the library.
  - [Navier–Stokes](@ref) shows how the same fields and operators build the incompressible flow solvers.

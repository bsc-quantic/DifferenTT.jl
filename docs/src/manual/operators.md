```@meta
CurrentModule = DifferenTT
```

# Finite-difference operators

To solve a PDE in the TT format, we need the derivatives of a field as operators that act on its MPS and keep its low-rank
structure. A one-dimensional derivative on a grid is a banded Toeplitz matrix, which has an exact representation as an
MPO of small bond dimension in the QTT format (Kazeev & Khoromskij, 2012), and the operator in ``N`` dimensions is the
Kronecker sum of the one-dimensional ones, each acting on the sites of its own axis. This page explains which function
builds what, and where the boundary conditions enter. The signatures are in the [API reference](@ref api_pdeproblem).

## Stencils and closures

The derivatives use central stencils of order ``q = 2`` (three points) or ``q = 8`` (nine points), whose coefficients
[`derivative_coefficients`](@ref) returns. In the interior, the operator is the stencil scaled by ``1/h^m`` for a derivative
of order ``m``. Next to a boundary, the stencil reaches ``q/2`` ghost nodes, and the boundary condition decides what
happens to them, as [Grids, fields and boundary conditions](@ref) describes: periodic sides wrap, Dirichlet and open sides
truncate the stencil, and Neumann sides mirror it. The one-dimensional closed operator of an axis is built by
[`onedim_differential_operator`](@ref), and [`ndim_operator_mpo`](@ref) places it on the sites of its axis in the
``N``-dimensional chain.

## The operators of a field

[`DifferentialOpsScalar`](@ref) builds, for a scalar field and its grid, the first and second derivatives along every axis
with the boundary conditions of that field, and [`DifferentialOps`](@ref) does the same for each component of a vector
field. The default stencil order is 8. The functions [`gradient`](@ref), [`divergence`](@ref), [`advection`](@ref) and
[`diffusion`](@ref) apply them to MPS.

The Laplacian has dedicated builders, because the linear systems of a flow solver are built from it:

  - [`kronecker_laplacian`](@ref) is the Kronecker sum of the second derivatives, ``\Delta = \sum_i \partial_i^2``. Its
    bond dimension between axes is 2 (Kazeev & Khoromskij, 2012). The Neumann rows are not weighted, so it is the
    consistent operator for explicit terms, but it is not symmetric when a side is Neumann.
  - [`symmetrized_laplacian`](@ref) multiplies it by the Neumann weight, which gives a symmetric operator. We use it
    wherever CG solves the system, and weight the right-hand side the same way with [`symmetrize_rhs`](@ref).
  - [`equicorrelated_laplacian`](@ref) is the operator ``\mathcal{L}(\rho)`` of Sec. 5.1 of the paper, whose mixed
    derivatives couple every pair of axes at bond dimension 3, and which serves as the benchmark of the preconditioners.

## Corrections of the right-hand side

With Dirichlet sides, the lift of the ghost values is part of the right-hand side of the linear system. The functions
[`scalar_dirichlet_correction`](@ref) and [`divergence_dirichlet_correction`](@ref) build it for the pressure equation,
and [`face_dirichlet_lift`](@ref) for a single face. The explicit time steppers do not impose the value of a Dirichlet
side: [`BoundaryNullScalar`](@ref) and [`BoundaryNull`](@ref) zero the right-hand side on the rows next to it, so that
those rows keep the values of the initial field.

!!! note
    The derivation of every closure, with the dense matrices of a small grid, is in the boundary-condition tutorials
    ([Velocity–pressure formulation](@ref bc_velocity_pressure) and
    [Vorticity–streamfunction formulation](@ref bc_vorticity_streamfunction)).

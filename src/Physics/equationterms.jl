# None of these functions applies the Dirichlet boundary nullifiers (`BoundaryNull`): the boundary conditions enter
# only through the closures of the operators (periodic, Neumann, or truncated at Dirichlet and open sides)

"""
    advection(fields::NTuple{N,MPS}, diff_ops::DifferentialOps{N}; cutoff::Float64=1e-15,
              maxdim::Int=typemax(Int64)) where N -> NTuple{N,MPS}

Return the advection term ``(\\mathbf{u} \\cdot \\nabla)\\mathbf{u}`` of the velocity ``\\mathbf{u}`` = `fields`, in
non-conservative form: component ``i`` is ``\\sum_j u_j\\, \\partial_j u_i``, where ``\\partial_j u_i`` uses the first
derivative of `diff_ops` built with the boundary conditions of component ``i``. The products ``u_j\\, \\partial_j u_i``
are element-wise products of MPS computed by [`tt_mul`](@ref), and each sum is truncated with `cutoff` and `maxdim`.

See also [`diffusion`](@ref), [`NavierStokesPredictorRHS`](@ref).
"""
function advection(
    fields::NTuple{N,MPS}, diff_ops::DifferentialOps{N}; cutoff::Float64=1e-15, maxdim::Int=typemax(Int64)
) where {N}
    # TODO(#60): the products below are RSI products (tt_mul of two MPS), whose sketches come from the global
    # Random.default_rng(); advection, NavierStokesPredictorRHS and the experiments take no rng, so a run is only
    # reproducible by seeding the global generator (see examples/06_chorin_cylinder.jl). Thread an `rng` through.
    advection_components = ntuple(
        i -> begin
            terms = [
                tt_mul(
                    fields[j],
                    apply(diff_ops.ops[i].d_ops[j], fields[i]; cutoff=cutoff, maxdim=maxdim);
                    cutoff=cutoff,
                    maxdim=maxdim,
                ) for j in 1:N
            ]
            return sum(terms; cutoff=cutoff, maxdim=maxdim)
        end,
        N,
    )

    return advection_components
end

"""
    diffusion(fields::NTuple{N,MPS}, diff_ops::DifferentialOps{N}; cutoff::Float64=1e-15,
              maxdim::Int=typemax(Int64)) where N -> NTuple{N,MPS}

Return the diffusion term ``\\nabla^2 \\mathbf{u}`` of the vector field ``\\mathbf{u}`` = `fields`: component ``i``
is ``\\sum_j \\partial_j^2 u_i``, with the second derivatives of `diff_ops` built with the boundary conditions of
component ``i``. These are the unweighted closures, as in [`kronecker_laplacian`](@ref), since the term enters an
explicit time step. The sums are truncated with `cutoff` and `maxdim`.
"""
function diffusion(
    fields::NTuple{N,MPS}, diff_ops::DifferentialOps{N}; cutoff::Float64=1e-15, maxdim::Int=typemax(Int64)
) where {N}
    diffusion_components = ntuple(
        i -> begin
            terms = [apply(diff_ops.ops[i].d2_ops[j], fields[i]; cutoff=cutoff, maxdim=maxdim) for j in 1:N]
            return sum(terms; cutoff=cutoff, maxdim=maxdim)
        end, N
    )

    return diffusion_components
end

"""
    divergence(fields::NTuple{N,MPS}, diff_ops::DifferentialOps{N}; cutoff::Float64=1e-15,
               maxdim::Int=typemax(Int64)) where N -> MPS

Return the divergence ``\\nabla \\cdot \\mathbf{u} = \\sum_i \\partial_i u_i`` of the vector field ``\\mathbf{u}`` =
`fields`, where ``\\partial_i u_i`` uses the first derivative of `diff_ops` built with the boundary conditions of
component ``i``. Its rows next to the Dirichlet faces are truncated; the Chorin projection removes them with
[`divergence_dirichlet_correction`](@ref).
"""
function divergence(
    fields::NTuple{N,MPS}, diff_ops::DifferentialOps{N}; cutoff::Float64=1e-15, maxdim::Int=typemax(Int64)
) where {N}
    terms = [apply(diff_ops.ops[i].d_ops[i], fields[i]; cutoff=cutoff, maxdim=maxdim) for i in 1:N]
    return sum(terms; cutoff=cutoff, maxdim=maxdim)
end

"""
    gradient(field::MPS, diff_ops::DifferentialOpsScalar{N}; cutoff::Float64=1e-15,
             maxdim::Int=typemax(Int64)) where N -> NTuple{N,MPS}

Return the gradient ``(\\partial_1 p, \\dots, \\partial_N p)`` of the scalar field ``p`` = `field`, with the first
derivatives of `diff_ops`. No Dirichlet lift is added (see [`scalar_dirichlet_correction`](@ref)).
"""
function gradient(
    field::MPS, diff_ops::DifferentialOpsScalar{N}; cutoff::Float64=1e-15, maxdim::Int=typemax(Int64)
) where {N}
    return ntuple(i -> apply(diff_ops.d_ops[i], field; cutoff=cutoff, maxdim=maxdim), N)
end

"""
    symmetrized_laplacian(field::ScalarField{N}, grid::Grid{N}; asymmetry::Float64=1.0, stencil_order::Int=8,
                          maxdim::Int=typemax(Int), cutoff::Float64=1e-15) where N -> MPO

Build the Laplacian of `field` with its boundary closures, weighted on the Neumann rows so that it is symmetric.

The pressure and stream-function systems are solved with CG, so their operator must be SPD (after the sign change
``A = -W\\Delta``). The mirrored Neumann closure breaks the symmetry of the Laplacian, since it counts the boundary row
twice. We therefore build the Kronecker sum ``\\Delta`` of [`kronecker_laplacian`](@ref) and, if `field` has a
`NeumannBC` side, multiply the whole operator by the Neumann weight ``W`` of [`neumann_weight_mpo`](@ref), which gives
the symmetric finite-volume half-cell operator ``W\\Delta``. The right-hand side must be weighted in the same way, with
[`symmetrize_rhs`](@ref). Without Neumann sides, ``W = \\mathbb{I}`` and the result is `kronecker_laplacian`. `cutoff`
and `maxdim` truncate the operators and the product with ``W``; the default `cutoff` keeps the operator exact.

# Keywords

  - `asymmetry::Float64=1.0`: factor of the second derivative along axis 1, which gives the anisotropic operator
    ``a\\,\\partial_1^2 + \\sum_{i>1} \\partial_i^2``.
  - `stencil_order::Int=8`: ``q``, 2 or 8.
  - `maxdim::Int=typemax(Int)`, `cutoff::Float64=1e-15`: truncation.

# Examples

```julia
Δ = symmetrized_laplacian(p_field, grid; stencil_order=8)
A = -1.0 * Δ                                  # SPD for the paper's Chorin setup (Sec. 5.2)
b = symmetrize_rhs(p_field, -1.0 * div_u)     # weight the right-hand side with the same W
```

See also the boundary-condition tutorial [Velocity–pressure formulation](@ref bc_velocity_pressure).
"""
function symmetrized_laplacian(
    field::ScalarField{N},
    grid::Grid{N};
    asymmetry::Float64=1.0,
    stencil_order::Int=8,
    maxdim::Int=typemax(Int),
    cutoff::Float64=1e-15,
) where {N}
    laplacian = kronecker_laplacian(
        field, grid; asymmetry=asymmetry, stencil_order=stencil_order, maxdim=maxdim, cutoff=cutoff
    )

    # Symmetrize the whole rows, not each 1D operator: halving only the 1D boundary row of each axis
    # leaves the tangential derivatives of a Neumann row with double weight
    weight = neumann_weight_mpo(field; cutoff=cutoff, maxdim=maxdim)
    return isnothing(weight) ? laplacian : apply(weight, laplacian; cutoff=cutoff, maxdim=maxdim)
end

"""
    kronecker_laplacian(field::ScalarField{N}, grid::Grid{N}; asymmetry::Float64=1.0, stencil_order::Int=8,
                        maxdim::Int=typemax(Int), cutoff::Float64=1e-15) where N -> MPO

Build the Laplacian ``\\Delta = \\sum_i \\partial_i^2`` of `field` as the Kronecker sum of the one-dimensional second
derivatives of [`onedim_differential_operator`](@ref), each closed with the boundary conditions of its axis
(truncated at Dirichlet and open sides, mirrored at Neumann sides, wrapped on periodic axes) and scaled by
``1/h_i^2``. The sum is assembled by [`ndim_operator_sum_mpo`](@ref), whose bond dimension between axes is 2 (Kazeev &
Khoromskij, 2012). Unlike [`symmetrized_laplacian`](@ref), the Neumann rows are not weighted, so it is the consistent
operator for explicit terms (e.g. the diffusion of the vorticity), but it is not symmetric with Neumann sides. Without
Neumann sides, both functions return the same operator. `asymmetry` multiplies the second derivative along axis 1.

See also the boundary-condition tutorial [Velocity–pressure formulation](@ref bc_velocity_pressure).
"""
function kronecker_laplacian(
    field::ScalarField{N},
    grid::Grid{N};
    asymmetry::Float64=1.0,
    stencil_order::Int=8,
    maxdim::Int=typemax(Int),
    cutoff::Float64=1e-15,
) where {N}

    # helper for loop to create raw 1D d2x, d2y, d2z...
    raw_onedim_ops = ntuple(
        i -> begin
            onedim_op = onedim_differential_operator(
                2,
                field.arch,
                i,
                field.bcu[i],
                backend_type(field);
                stencil_order=stencil_order,
                cutoff=cutoff,
                maxdim=maxdim,
            )

            if onedim_op isa MPO
                onedim_op = onedim_op * (1/grid.dx[i]^2)
            else
                onedim_op = onedim_op .* (1/grid.dx[i]^2)
            end

            return i == 1 ? asymmetry * onedim_op : onedim_op
        end,
        N,
    )

    return ndim_operator_sum_mpo(raw_onedim_ops, field.usites, field.arch)
end

"""
    equicorrelated_laplacian(field::ScalarField{N}, grid::Grid{N}; correlation::Float64, stencil_order::Int=8,
                             maxdim::Int=typemax(Int), cutoff::Float64=0.0) where N -> MPO

Build the anisotropic diffusion operator ``\\nabla \\cdot (K \\nabla)`` with the equicorrelated diffusion tensor
``K = (1 - \\rho)\\, \\mathbb{I} + \\rho\\, \\mathbf{1}\\mathbf{1}^\\top``, ``\\rho`` = `correlation`, i.e. minus the
operator ``\\mathcal{L}(\\rho)`` of Sec. 5.1 of Torrente et al., *Factorized approximate inverse preconditioners in the
tensor train format*.

The Laplacian is the ideal case for separable preconditioners, such as the sinc preconditioner, which approximate
``\\mathcal{L}_0^{-1} = (-\\Delta_h)^{-1}``. The equicorrelated tensor of Eq. (27) moves the problem away from that case
in a controlled way, through a rotated anisotropy that is not aligned with any grid axis. Since
``\\mathrm{diag}(K) = \\mathbb{I}``, centered finite differences give Eq. (28),

```math
-\\mathcal{L}(\\rho) = \\sum_{\\ell=1}^N \\partial_\\ell^2 + 2\\rho \\sum_{\\ell<m} \\partial_\\ell\\, \\partial_m,
```

where ``\\partial_\\ell^2`` and ``\\partial_\\ell`` are the second and first derivatives of
[`onedim_differential_operator`](@ref) along axis ``\\ell``, i.e. ``-\\mathcal{A}_\\ell`` and ``\\mathcal{C}_\\ell`` of
the paper. Even though the mixed term couples every pair of axes, we build the MPO directly from the pairwise
finite-state machine of [`ndim_operator_sum_mpo`](@ref) (`pair_ops`), so its bond dimension is exactly 3 for any ``N``
(Eq. (29)), and 2 for ``\\rho = 0``, where it reduces to the Laplacian of [`kronecker_laplacian`](@ref).

``K`` is SPD for ``-1/(N-1) < \\rho < 1``, and for ``\\rho \\ge 0`` the floor of Eq. (30),
``\\kappa(\\mathcal{L}_0^{-1}\\mathcal{L}) \\le F(\\rho, N) = (1 + (N-1)\\rho)/(1 - \\rho)``, holds, as derived in
Appendix A of the paper. That derivation assumes `stencil_order = 2`, homogeneous Dirichlet sides and the spacing
``h = 1/(n+1)`` of [`interior_grid`](@ref); with [`Grid`](@ref), ``h = 1/(n-1)``, which only scales the operator. The
default `stencil_order = 8` builds the same structure with wider stencils, for which the bound is not proven.

Only `TTArchitecture` is supported. Neumann sides are not supported, since the weighting of the mixed term is not
defined, and the Dirichlet lifts of [`scalar_dirichlet_correction`](@ref) only cover the second derivatives, so
nonhomogeneous Dirichlet data are not fully supported either.

# Keywords

  - `correlation::Float64`: ``\\rho``.
  - `stencil_order::Int=8`: ``q``, 2 or 8 (2 in the paper).
  - `maxdim::Int=typemax(Int)`, `cutoff::Float64=0.0`: truncation of the one-dimensional operators.

# Throws

  - `ArgumentError`: if the architecture is not a `TTArchitecture`, or if `field` has a `NeumannBC` side.

# Examples

```julia
arch = TTArchitecture(ntuple(_ -> 16, 3))
bc = BoundaryCond1d(DirichletBC(0.0), DirichletBC(0.0))
field = ScalarField(siteinds(arch), arch, BoundaryConditions(bc, bc, bc))
grid = interior_grid(arch, (1.0, 1.0, 1.0))
F, N = 10, 3
L = -1.0 * equicorrelated_laplacian(field, grid; correlation=(F - 1) / (F + N - 1), stencil_order=2)  # Eq. (32)
maxlinkdim(L)  # 3
```
"""
function equicorrelated_laplacian(
    field::ScalarField{N},
    grid::Grid{N};
    correlation::Float64,
    stencil_order::Int=8,
    maxdim::Int=typemax(Int),
    cutoff::Float64=0.0,
) where {N}
    field.arch isa TTArchitecture || throw(ArgumentError("equicorrelated_laplacian only supports TT architectures"))
    for i in 1:N
        if field.bcu[i].left isa NeumannBC || field.bcu[i].right isa NeumannBC
            throw(ArgumentError("equicorrelated_laplacian does not support Neumann boundary conditions (dimension $i)"))
        end
    end

    d2_ops = ntuple(
        i -> begin
            onedim_op = onedim_differential_operator(
                2,
                field.arch,
                i,
                field.bcu[i],
                backend_type(field);
                stencil_order=stencil_order,
                cutoff=cutoff,
                maxdim=maxdim,
            )
            return onedim_op .* (1/grid.dx[i]^2)
        end,
        N,
    )

    if iszero(correlation)
        return ndim_operator_sum_mpo(d2_ops, field.usites, field.arch)
    end

    d1_ops = ntuple(
        i -> begin
            onedim_op = onedim_differential_operator(
                1,
                field.arch,
                i,
                field.bcu[i],
                backend_type(field);
                stencil_order=stencil_order,
                cutoff=cutoff,
                maxdim=maxdim,
            )
            return onedim_op .* (1/grid.dx[i])
        end,
        N,
    )
    scaled_d1_ops = ntuple(i -> (2 * correlation) .* d1_ops[i], N)

    return ndim_operator_sum_mpo(d2_ops, field.usites, field.arch; pair_ops=(d1_ops, scaled_d1_ops))
end

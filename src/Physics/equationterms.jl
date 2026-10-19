# None of these functions implement Dirichlet boundary nullifiers, they do implemente periodic, neumann or open BCs

"""
    advection(fields::NTuple{N, MPS}, diff_ops::DifferentialOps{N}) where N

Compute the advection term `(u ⋅ ∇) u` for an N-dimensional vector field.

# Arguments

  - `fields::NTuple{N, MPS}`: The vector field components (velocity).
  - `diff_ops::DifferentialOps{N}`: Dimensional differential operators.
"""
function advection(
    fields::NTuple{N,MPS}, diff_ops::DifferentialOps{N}; cutoff::Float64=1e-15, maxdim::Int=typemax(Int64)
) where {N}
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
    diffusion(fields::NTuple{N, MPS}, diff_ops::DifferentialOps{N}) where N

Compute the diffusion term `∇²u` (Laplacian) for an N-dimensional vector field.

# Arguments

  - `fields::NTuple{N, MPS}`: The vector field components.
  - `diff_ops::DifferentialOps{N}`: Dimensional differential operators.
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
    divergence(fields::NTuple{N, MPS}, diff_ops::DifferentialOps{N}) where N

Compute the divergence `∇ ⋅ u` of an N-dimensional vector field.

# Arguments

  - `fields::NTuple{N, MPS}`: The vector field components.
  - `diff_ops::DifferentialOps{N}`: Dimensional differential operators.
"""
function divergence(
    fields::NTuple{N,MPS}, diff_ops::DifferentialOps{N}; cutoff::Float64=1e-15, maxdim::Int=typemax(Int64)
) where {N}
    terms = [apply(diff_ops.ops[i].d_ops[i], fields[i]; cutoff=cutoff, maxdim=maxdim) for i in 1:N]
    return sum(terms; cutoff=cutoff, maxdim=maxdim)
end

"""
    gradient(field::MPS, diff_ops::DifferentialOpsScalar{N}) where N

Compute the gradient `∇p` of a scalar field, returning an N-tuple of MPSs.

# Arguments

  - `field::MPS`: The scalar field.
  - `diff_ops::DifferentialOpsScalar{N}`: Scalar differential operators.
"""
function gradient(
    field::MPS, diff_ops::DifferentialOpsScalar{N}; cutoff::Float64=1e-15, maxdim::Int=typemax(Int64)
) where {N}
    return ntuple(i -> apply(diff_ops.d_ops[i], field; cutoff=cutoff, maxdim=maxdim), N)
end

"""
    symmetrized_laplacian(field::ScalarField{N}, grid::Grid{N}; asymmetry::Float64=1.0, stencil_order::Int=8, maxdim::Int=typemax(Int), cutoff::Float64=1e-15) where N

Construct the full Laplacian operator for a scalar field, symmetrized for `NeumannBC` faces.
Builds the 1D d²/dx² operators per dimension (with the mirrored Neumann closures), scales them by the grid
spacing and sums them into the N-dimensional Kronecker sum `Δ`. If the field has a `NeumannBC` face, the
whole operator is then multiplied by the Neumann weight `W` (see [`neumann_weight_mpo`](@ref)), which gives
the symmetric finite volume half-cell operator `W Δ`. Use it with [`symmetrize_rhs`](@ref) on the right-hand
side. `cutoff` and `maxdim` truncate that product; the default `cutoff` keeps the operator exact (see
`neumann_weight_mpo`).

# Returns

  - `MPO`: The symmetrized Laplacian operator.
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
    kronecker_laplacian(field::ScalarField{N}, grid::Grid{N}; asymmetry::Float64=1.0, stencil_order::Int=8, maxdim::Int=typemax(Int), cutoff::Float64=1e-15) where N

Laplacian with the closures of the boundary conditions of `field` (Dirichlet truncated, Neumann mirrored, periodic
wrapped), built as the Kronecker sum of the 1D second derivatives. Unlike [`symmetrized_laplacian`](@ref), the
Neumann rows are not weighted, so it is the consistent operator for explicit terms, but it is not symmetric with
Neumann faces. Without Neumann faces, both functions return the same operator.

# Returns

  - `MPO`: The Laplacian operator.
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
    equicorrelated_laplacian(field::ScalarField{N}, grid::Grid{N}; correlation::Float64, stencil_order::Int=8, maxdim::Int=typemax(Int), cutoff::Float64=0.0) where N

Construct the anisotropic diffusion operator `∇⋅(K∇)` with the equicorrelated diffusion tensor
`K = (1-ρ) I + ρ 𝟏𝟏ᵀ`, `ρ = correlation`. Since `diag(K) = 1`, the operator is discretized as
`Σ_ℓ d²_ℓ + 2ρ Σ_{ℓ<m} d_ℓ d_m`, with centered first derivatives in the mixed term. The MPO is built
directly with the pairwise finite-state machine of `ndim_operator_sum_mpo`, so its bond dimension is
exactly 3 (2 for `correlation == 0`, where it reduces to the Laplacian).

`K` is SPD for `-1/(N-1) < ρ < 1`. Only TT architectures are supported. Neumann BCs are not supported,
since the symmetrization of the mixed term is not defined, and `scalar_dirichlet_correction` only covers
the second-derivative part, so inhomogeneous Dirichlet BCs are not fully supported either.

# Returns

  - `MPO`: The equicorrelated diffusion operator.
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

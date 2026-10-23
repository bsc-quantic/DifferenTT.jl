"""
    neumann_weight_mpo(field::ScalarField{N}; cutoff::Float64=1e-15,
                       maxdim::Int=typemax(Int)) where N -> Union{MPO,Nothing}

Build the diagonal weight ``W = W_1 \\otimes \\dots \\otimes W_N`` that symmetrizes a linear system built with the
mirrored [`NeumannBC`](@ref) closures of `field`.

The mirror closure, ``u_{-k} = u_k``, places the boundary on the stored node and counts its row of the second derivative
twice, so the operator is no longer symmetric and CG cannot be applied to it. ``W_i`` is the identity with ``1/2`` on
the Neumann boundary rows of axis ``i`` (the first and/or last row), i.e. the identity plus
[`onedim_neumann_symmetrization_correction`](@ref) applied to it, and multiplying the whole system
``A\\mathbf{x} = \\mathbf{b}`` by ``W`` gives the finite-volume half-cell equations: ``WA`` is symmetric, for stencil
orders 2 and 8, and has the same solution. A node shared by two Neumann faces gets the weight ``1/4``. We weight whole
``N``-dimensional rows, rather than the boundary row of each one-dimensional operator, because halving only the latter
leaves the tangential derivatives of a Neumann row with double weight.

In QTT, ``W_i`` has bond dimension 2 (one Neumann side) or 3 (both sides) inside its axis. The lifted ``W`` has bond
dimension 1 between axes in TT and in non-interleaved QTT, while an interleaved QTT threads the bonds of every axis
through the others. The weight is built on the backend of `field`. Returns `nothing` if `field` has no `NeumannBC`
side.

See also [`symmetrized_laplacian`](@ref), [`symmetrize_rhs`](@ref), and the boundary-condition tutorial
[Velocity–pressure formulation](@ref bc_velocity_pressure).
"""
function neumann_weight_mpo(field::ScalarField{N}; cutoff::Float64=1e-15, maxdim::Int=typemax(Int)) where {N}
    array_type = backend_type(field)
    weights = Union{MPO,Nothing}[]

    for i in 1:N
        bcs = field.bcu[i]
        if !(bcs.left isa NeumannBC || bcs.right isa NeumannBC)
            push!(weights, nothing)
            continue
        end

        if field.arch isa QTTArchitecture
            identity_op = identity_mpo(siteinds(2, abs(field.arch.var_nsites[i])), array_type)
        else
            identity_op = adapt(array_type, identity_array(field.arch.var_dims[i]))
        end
        correction = onedim_neumann_symmetrization_correction(
            identity_op, field.arch, i, bcs; cutoff=cutoff, maxdim=maxdim
        )
        onedim_weight = if identity_op isa MPO
            _nothing_sum_handler(identity_op, correction; cutoff=cutoff, maxdim=maxdim)
        else
            identity_op + correction
        end

        push!(weights, ndim_operator_mpo(onedim_weight, i, field.usites, field.arch))
    end

    return _nothing_apply_handler(weights...; cutoff=cutoff, maxdim=maxdim)
end

"""
    symmetrize_rhs(field::ScalarField{N}, rhs::MPS; cutoff::Float64=1e-15, maxdim::Int=typemax(Int)) where N -> MPS

Return ``W\\mathbf{b}``, the right-hand side `rhs` weighted with the Neumann weight ``W`` of `field` (see
[`neumann_weight_mpo`](@ref)), i.e. halved on the `NeumannBC` boundary rows of each axis, and divided by four on the
nodes shared by two Neumann faces. It goes with [`symmetrized_laplacian`](@ref), which applies the same ``W`` to the
operator, so that the weighted system ``WA\\mathbf{x} = W\\mathbf{b}`` has the solution of the original one. Returns
`rhs` itself if `field` has no `NeumannBC` side.

See also the boundary-condition tutorial [Velocity–pressure formulation](@ref bc_velocity_pressure).
"""
function symmetrize_rhs(field::ScalarField{N}, rhs::MPS; cutoff::Float64=1e-15, maxdim::Int=typemax(Int)) where {N}
    weight = neumann_weight_mpo(field; cutoff=cutoff, maxdim=maxdim)
    return isnothing(weight) ? rhs : apply(weight, rhs; cutoff=cutoff, maxdim=maxdim)
end

"""
    face_profile_mps(field::ScalarField{2}, axis::Int, values; cutoff::Float64=1e-15, maxdim::Int=typemax(Int)) -> MPS

Return the MPS on the sites of `field` that is constant along `axis` and equal to `values` along the other axis, i.e.
the profile of a face of `axis` extended to the whole two-dimensional domain. The lift operators of the Dirichlet
corrections are only meaningful on vectors that are constant along their axis, and this is the vector they are
applied to ([`face_dirichlet_lift`](@ref)). `values` is either a `Number`, for a constant profile, or a vector with one
value per stored node of the other axis, in increasing node index. A vector profile is built as a dense diagonal
matrix, converted to an MPO of the other axis and applied to the all-ones MPS, which is cheap up to about ``2^{11}``
nodes per axis in QTT.

# Throws

  - `ArgumentError`: if a vector `values` does not have one value per node of the other axis.

See also the boundary-condition tutorial [Vorticity–streamfunction formulation](@ref bc_vorticity_streamfunction).
"""
function face_profile_mps(field::ScalarField{2}, axis::Int, values; cutoff::Float64=1e-15, maxdim::Int=typemax(Int))
    array_type = backend_type(field)
    values isa Number && return values * ones_mps(field.usites, 1, array_type)

    other = 3 - axis
    length(values) == variable_dims(field.arch)[other] || throw(
        ArgumentError(
            "Face profile of axis $axis must have $(variable_dims(field.arch)[other]) values, got $(length(values))"
        ),
    )
    onedim = _onedim_diagonal_operator(field.arch, other, values, array_type; cutoff=cutoff)
    profile_op = ndim_operator_mpo(onedim, other, field.usites, field.arch)
    return apply(profile_op, ones_mps(field.usites, 1, array_type); cutoff=cutoff, maxdim=maxdim)
end

"""
    _onedim_diagonal_operator(arch::TTArchitecture, axis::Int, values, array_type; cutoff::Float64=1e-15)
    _onedim_diagonal_operator(arch::QTTArchitecture, axis::Int, values, array_type; cutoff::Float64=1e-15)

Return the one-dimensional diagonal operator ``\\mathrm{diag}(\\mathtt{values})`` of axis `axis` on `array_type`: a
dense matrix in TT, and in QTT an MPO on new sites of dimension 2 built from the dense ``2^{N_i} \\times 2^{N_i}``
matrix by `mat2mpo` with `cutoff`.
"""
function _onedim_diagonal_operator(arch::TTArchitecture, axis::Int, values, array_type; cutoff::Float64=1e-15)
    adapt(array_type, Matrix(Diagonal(Vector{Float64}(values))))
end

# TODO(#70): a diagonal built from vec2mps and diagmpo would avoid the dense 2ⁿ × 2ⁿ matrix (fine up to n ≈ 11)
function _onedim_diagonal_operator(arch::QTTArchitecture, axis::Int, values, array_type; cutoff::Float64=1e-15)
    sites = siteinds(2, abs(arch.var_nsites[axis]))
    return adapt(array_type, mat2mpo(Matrix(Diagonal(Vector{Float64}(values))), sites; cutoff=cutoff))
end

"""
    face_dirichlet_lift(field::ScalarField{2}, grid::Grid{2}, derivative_order::Int, axis::Int, side::Symbol, values;
                        stencil_order::Int=2, cutoff::Float64=1e-15, maxdim::Int=typemax(Int)) -> Union{MPS,Nothing}

Return the ghost contribution of one face to the derivative of order ``k`` = `derivative_order` along `axis`, the
lift that completes a truncated stencil with nonzero ghost values.

The derivative operators truncate the stencils at the non-periodic faces, i.e. they take the ghost values as zero.
When the ghost nodes of the `side` (`:left` or `:right`) face of `axis` hold the values ``g``, the full stencil equals
the truncated one plus

```math
\\boldsymbol{\\ell} = \\frac{1}{h^k} (C \\otimes \\mathbb{I})\\, \\mathbf{g},
```

where ``C`` is the one-dimensional block of [`onedim_left_dirichlet_correction`](@ref) or
[`onedim_right_dirichlet_correction`](@ref), lifted to two dimensions along `axis`, ``h`` the spacing of `grid` along
`axis`, and ``\\mathbf{g}`` the face values extended along `axis` by [`face_profile_mps`](@ref). `values` is a `Number`
or a profile along the face. With the default `stencil_order = 2` (the vorticity–streamfunction scheme), the lift is
nonzero only on the row next to the face. Returns `nothing` if `values` is `nothing` or zero.

# Throws

  - `ArgumentError`: if `side` is not `:left` or `:right`, or if a profile has the wrong length.

See also [`scalar_dirichlet_correction`](@ref), and the boundary-condition tutorial [Vorticity–streamfunction
formulation](@ref bc_vorticity_streamfunction).
"""
function face_dirichlet_lift(
    field::ScalarField{2},
    grid::Grid{2},
    derivative_order::Int,
    axis::Int,
    side::Symbol,
    values;
    stencil_order::Int=2,
    cutoff::Float64=1e-15,
    maxdim::Int=typemax(Int),
)
    (isnothing(values) || all(iszero, values)) && return nothing
    side in (:left, :right) || throw(ArgumentError("side must be :left or :right, got $side"))
    array_type = backend_type(field)
    onedim = if side == :left
        onedim_left_dirichlet_correction(derivative_order, field.arch, axis, array_type; stencil_order=stencil_order)
    else
        onedim_right_dirichlet_correction(derivative_order, field.arch, axis, array_type; stencil_order=stencil_order)
    end
    onedim =
        onedim isa MPO ? onedim * (1 / grid.dx[axis]^derivative_order) : onedim .* (1 / grid.dx[axis]^derivative_order)
    correction = ndim_operator_mpo(onedim, axis, field.usites, field.arch)
    return apply(
        correction, face_profile_mps(field, axis, values; cutoff=cutoff, maxdim=maxdim); cutoff=cutoff, maxdim=maxdim
    )
end

"""
    remove_mean(rhs::MPS; cutoff::Float64=1e-15, maxdim::Int=typemax(Int)) -> MPS

Return ``\\mathbf{b} - \\bar{b}\\,\\mathbf{1}``, the right-hand side `rhs` minus its node mean
``\\bar{b} = \\frac{1}{n} \\sum_j b_j`` over the ``n`` grid nodes. A singular system whose null space is the constant
vector (pure Neumann or periodic boundaries) is only consistent when the right-hand side is orthogonal to that
vector, and this projection makes it so. The vorticity–streamfunction scheme uses it; note that the Chorin projection
removes a different constant (see [`poisson_rhs`](@ref) and the known limitations of [`INSExperiment`](@ref),
[#49](https://github.com/bsc-quantic/DifferenTT.jl/issues/49)).

See also the boundary-condition tutorial [Vorticity–streamfunction formulation](@ref bc_vorticity_streamfunction).
"""
function remove_mean(rhs::MPS; cutoff::Float64=1e-15, maxdim::Int=typemax(Int))
    sites = siteinds(rhs)
    mean = integrate(rhs) / prod(dim, sites)
    return +(rhs, -mean * ones_mps(sites, 1, backend_type(rhs)); cutoff=cutoff, maxdim=maxdim)
end

"""
    divergence_dirichlet_correction(u_fields::VectorField{N}; stencil_order::Int=8, cutoff::Float64=1e-15,
                                    maxdim::Int=typemax(Int64)) where N -> Union{MPO,Nothing}

Build the diagonal projector that zeroes the divergence of a velocity field on the rows next to its Dirichlet faces.

The term ``\\partial_i u_i`` of the divergence is computed with the first derivative of component ``i`` along axis
``i``, whose stencil is truncated within ``q/2`` rows of a Dirichlet side of that component. These rows are not
consistent approximations of the derivative, so the Chorin projection removes them from the pressure equation: the
projector is the product over the axes of the one-dimensional projectors of [`onedim_boundary_nullifier`](@ref),
built from the boundary conditions of component ``i`` along axis ``i`` only, and applied to the divergence it sets the
source of the pressure Poisson equation to zero on those rows (see [`poisson_rhs`](@ref)). Returns `nothing` if no
component has a Dirichlet side along its own axis.

# Arguments

  - `u_fields::VectorField{N}`: the velocity field, whose components carry the boundary conditions.

# Keywords

  - `stencil_order::Int=8`: ``q``, which sets the ``q/2`` rows that are zeroed.
  - `cutoff::Float64=1e-15`, `maxdim::Int=typemax(Int64)`: truncation of the product of the projectors.

See also [`BoundaryNull`](@ref), and the boundary-condition tutorial [Velocity–pressure formulation](@ref
bc_velocity_pressure).
"""
function divergence_dirichlet_correction(
    u_fields::VectorField{N}; stencil_order::Int=8, cutoff::Float64=1e-15, maxdim::Int=typemax(Int64)
) where {N}
    array_type = u_fields.components[1].array_type

    deleters = Union{MPO,Nothing}[]

    for i in 1:N
        comp = u_fields.components[i]
        bc_i = comp.bcu[i]

        onedim_deleter = onedim_boundary_nullifier(
            u_fields.arch, i, bc_i, array_type; stencil_order=stencil_order, cutoff=cutoff, maxdim=maxdim
        )

        deleter =
            (isnothing(onedim_deleter)) ? nothing : ndim_operator_mpo(onedim_deleter, i, u_fields.usites, u_fields.arch)
        push!(deleters, deleter)
    end

    return _nothing_apply_handler(deleters...; cutoff=cutoff, maxdim=maxdim)
end

"""
    scalar_dirichlet_correction_dim(field::ScalarField{N}, grid::Grid{N}, derivative_order::Int, dim::Int;
                                    stencil_order::Int=8, cutoff::Float64=1e-15,
                                    maxdim::Int=typemax(Int64)) where N -> Union{MPO,Nothing}

Build the operator whose product with the all-ones MPS is the Dirichlet lift of `field` along the axis `dim`, for the
derivative of order ``k`` = `derivative_order`.

Each `DirichletBC` side of the axis with value ``g`` (``|g| > 10^{-14}``) contributes ``g\\, C / h^k``, where ``C`` is
[`onedim_left_dirichlet_correction`](@ref) or [`onedim_right_dirichlet_correction`](@ref) lifted to ``N`` dimensions,
and ``h`` the spacing of `grid` along `dim`. The ghost nodes beyond each Dirichlet side therefore all hold its value
``g``, and ``(\\sum_\\text{sides} g\\,C/h^k)\\,\\mathbf{1}`` is what the truncated derivative misses on the rows next to
that side. Note that the result is an operator, not the lift itself: the callers apply it to `ones_mps` (see
[`poisson_rhs`](@ref) and `∇p_correction`). Returns `nothing` if the axis has no Dirichlet side with a nonzero value.

See also [`scalar_dirichlet_correction`](@ref), and the boundary-condition tutorial [Velocity–pressure formulation](@ref
bc_velocity_pressure).
"""
function scalar_dirichlet_correction_dim(
    field::ScalarField{N},
    grid::Grid{N},
    derivative_order::Int,
    dim::Int;
    stencil_order::Int=8,
    cutoff::Float64=1e-15,
    maxdim::Int=typemax(Int64),
) where {N}
    bc_dim = field.bcu[dim]

    left_correction, right_correction = nothing, nothing

    if bc_dim.left isa DirichletBC
        val_L = bc_dim.left.value
        if abs(val_L) > 1e-14
            onedim_left_correction = onedim_left_dirichlet_correction(
                derivative_order, field.arch, dim, field.array_type; stencil_order=stencil_order
            )
            left_correction = ndim_operator_mpo(onedim_left_correction, dim, field.usites, field.arch)
            left_correction = left_correction * val_L
        end
    end

    if bc_dim.right isa DirichletBC
        val_R = bc_dim.right.value
        if abs(val_R) > 1e-14
            onedim_right_correction = onedim_right_dirichlet_correction(
                derivative_order, field.arch, dim, field.array_type; stencil_order=stencil_order
            )
            right_correction = ndim_operator_mpo(onedim_right_correction, dim, field.usites, field.arch)
            right_correction = right_correction * val_R
        end
    end

    dim_correction = _nothing_sum_handler(left_correction, right_correction; cutoff=cutoff, maxdim=maxdim)
    if !isnothing(dim_correction)
        if dim_correction isa MPO
            dim_correction = dim_correction * (1 / grid.dx[dim]^derivative_order)
        else
            dim_correction = dim_correction .* (1 / grid.dx[dim]^derivative_order)
        end
    end

    return dim_correction
end

"""
    scalar_dirichlet_correction(field::ScalarField{N}, grid::Grid{N}, derivative_order::Int; stencil_order::Int=8,
                                cutoff::Float64=1e-15, maxdim::Int=typemax(Int64)) where N -> Union{MPO,Nothing}

Build the operator whose product with the all-ones MPS is the Dirichlet lift of `field` for the derivative of order
`derivative_order` summed over all axes, i.e. the sum over the axes of [`scalar_dirichlet_correction_dim`](@ref). For
`derivative_order = 2` it completes the truncated Laplacian with the Dirichlet values of `field`:
``\\Delta_\\text{full}\\,\\mathbf{p} = \\Delta\\,\\mathbf{p} + C\\,\\mathbf{1}``, where ``C`` is the returned
operator, so it moves nonhomogeneous Dirichlet data of the pressure to the right-hand side of the Chorin projection
([`poisson_rhs`](@ref)). Only the second-derivative terms of each axis are covered: an operator with mixed derivatives
(e.g. [`equicorrelated_laplacian`](@ref)) needs further lifts. Returns `nothing` if no axis has a Dirichlet side with a
nonzero value.

# Examples

```julia
C = scalar_dirichlet_correction(p_field, grid, 2; stencil_order=8)
lift = apply(C, ones_mps(p_field.usites, 1))
```

See also the boundary-condition tutorial [Velocity–pressure formulation](@ref bc_velocity_pressure).
"""
function scalar_dirichlet_correction(
    field::ScalarField{N},
    grid::Grid{N},
    derivative_order::Int;
    stencil_order::Int=8,
    cutoff::Float64=1e-15,
    maxdim::Int=typemax(Int64),
) where {N}
    corrections = Union{MPO,Nothing}[]

    for i in 1:N
        dim_correction = scalar_dirichlet_correction_dim(
            field, grid, derivative_order, i; stencil_order=stencil_order, cutoff=cutoff, maxdim=maxdim
        )
        push!(corrections, dim_correction)
    end

    return _nothing_sum_handler(corrections...; cutoff=cutoff, maxdim=maxdim)
end

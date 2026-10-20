"""
    lower_toeplitz_matrix(coefficients::Vector{<:Number}) -> Matrix

Return the ``n \\times n`` lower triangular Toeplitz matrix ``L_{ij} = c_{i-j+1}``, ``i \\ge j``, of the
coefficients ``(c_1, \\dots, c_n)``, i.e. the matrix with first column `coefficients`.

# Examples

```jldoctest
julia> DifferenTT.lower_toeplitz_matrix([1, 2, 3])
3×3 Matrix{Int64}:
 1  0  0
 2  1  0
 3  2  1
```
"""
function lower_toeplitz_matrix(coefficients::Vector{<:Number})
    n = length(coefficients)
    lower_toeplitz = zeros(eltype(coefficients), n, n)
    for j in 1:n
        for i in j:n
            lower_toeplitz[i, j] = coefficients[i-j+1]
        end
    end
    return lower_toeplitz
end

"""
    upper_toeplitz_matrix(coefficients::Vector{<:Number})

Return the upper triangular Toeplitz matrix with first row `coefficients`, the (lazy) transpose of
[`lower_toeplitz_matrix`](@ref).
"""
function upper_toeplitz_matrix(coefficients::Vector{<:Number})
    return transpose(lower_toeplitz_matrix(coefficients))
end

# -------------------------------------------------------------------------------------------------
# 1D Differential Operators
# -------------------------------------------------------------------------------------------------

"""
    onedim_differential_operator(derivative_order::Int, arch::QTTArchitecture, dim::Int, [n_eyepad::Int,]
                                 boundary_conditions::BoundaryCond1d, array_type::Type{<:AbstractArray}=Array;
                                 stencil_order::Int=8, cutoff::Float64=1e-15, maxdim::Int=typemax(Int64)) -> MPO
    onedim_differential_operator(derivative_order::Int, arch::TTArchitecture, dim::Int, [n_eyepad::Int,]
                                 boundary_conditions::BoundaryCond1d, array_type::Type{<:AbstractArray}=Array;
                                 stencil_order::Int=8, kwargs...) -> AbstractMatrix

Build the unscaled (``h = 1``) one-dimensional finite-difference operator of order `derivative_order` along the axis
`dim`, closed with the boundary conditions of that axis.

The operator is the central stencil of order ``q`` = `stencil_order`, with the coefficients ``c_k``, ``|k| \\le q/2``,
of [`derivative_coefficients`](@ref), placed as the band Toeplitz matrix ``D_{ij} = c_{j-i}``. Its rows near the
boundaries are then completed as the boundary conditions prescribe. A periodic axis wraps the band into a circulant. A
Dirichlet or open side truncates it, i.e. takes the ghost values as zero, and the lifts of
[`scalar_dirichlet_correction`](@ref) restore nonzero Dirichlet data on the right-hand side. A homogeneous Neumann side
mirrors the ghost values, ``u_{-k} = u_k``, by subtracting the corner block of [`neumann_correction_matrix`](@ref). In
QTT, the operator is an MPO on new sites of dimension 2 built by [`toeplitz_mpo`](@ref), whose bond dimension does not
depend on the number of sites (Kazeev & Khoromskij, 2012); in TT, it is a dense ``n \\times n`` matrix. Dividing it by
``h^k`` gives the derivative on a grid of spacing ``h``, and [`ndim_operator_mpo`](@ref) lifts it to ``N`` dimensions.

With `n_eyepad` ``= p > 0``, the operator acts on part of the axis only, and the identity on the rest (eye padding).
The two architectures read this differently. In QTT, the operator is built on the ``N_i - p`` most significant bits
and the identity acts on the ``p`` least significant ones, ``D \\otimes \\mathbb{I}_{2^p}``: it is the derivative on
a grid of ``2^{N_i - p}`` nodes, applied to each of the ``2^p`` subsets of nodes whose indices have the same remainder
modulo ``2^p``. In TT, the operator acts on the first ``n - p`` nodes and the identity on the last ``p``,
``\\mathrm{diag}(D, \\mathbb{I}_p)``. In both cases the boundary closures are those of the shorter operator ``D``.
On a periodic TT axis of at most ``q`` nodes, a stencil wraps onto itself, and the coefficients that land on the
same node add up.

# Arguments

  - `derivative_order::Int`: 1, 2 or 4.
  - `arch`: the architecture, which gives the number of sites (QTT) or nodes (TT) of the axis.
  - `dim::Int`: the axis.
  - `n_eyepad::Int`: the number of padded sites (QTT) or nodes (TT); 0 when omitted.
  - `boundary_conditions::BoundaryCond1d`: the conditions of the two sides of the axis.
  - `array_type::Type{<:AbstractArray}=Array`: the array backend.

# Keywords

  - `stencil_order::Int=8`: ``q``, 2 or 8.
  - `cutoff::Float64=1e-15`, `maxdim::Int=typemax(Int64)`: truncation of the QTT sums (not used in TT).

# Returns

  - `MPO` (QTT), on new sites of dimension 2 with site 1 the most significant bit, or a matrix (TT).

# Throws

  - `ArgumentError`: for a derivative order other than 1, 2 or 4, or a stencil order other than 2 or 8.

# Examples

```jldoctest
julia> bc = BoundaryCond1d(NeumannBC(0.0), DirichletBC(0.0));

julia> DifferenTT.onedim_differential_operator(2, TTArchitecture((5,)), 1, bc; stencil_order=2)
5×5 Matrix{Float64}:
 -2.0   2.0   0.0   0.0   0.0
  1.0  -2.0   1.0   0.0   0.0
  0.0   1.0  -2.0   1.0   0.0
  0.0   0.0   1.0  -2.0   1.0
  0.0   0.0   0.0   1.0  -2.0
```

See also [`DifferentialOpsScalar`](@ref), and the boundary-condition tutorial [Velocity–pressure formulation](@ref
bc_velocity_pressure).
"""
function onedim_differential_operator(
    derivative_order::Int,
    arch::QTTArchitecture,
    dim::Int,
    boundary_conditions::BoundaryCond1d,
    array_type::Type{<:AbstractArray}=Array;
    stencil_order::Int=8,
    cutoff::Float64=1e-15,
    maxdim::Int=typemax(Int64),
)
    return onedim_differential_operator(
        derivative_order,
        arch,
        dim,
        0,
        boundary_conditions,
        array_type;
        stencil_order=stencil_order,
        cutoff=cutoff,
        maxdim=maxdim,
    )
end

function onedim_differential_operator(
    derivative_order::Int,
    arch::QTTArchitecture,
    dim::Int,
    n_eyepad::Int,
    boundary_conditions::BoundaryCond1d,
    array_type::Type{<:AbstractArray}=Array;
    stencil_order::Int=8,
    cutoff::Float64=1e-15,
    maxdim::Int=typemax(Int64),
)
    nsite = abs(arch.var_nsites[dim])
    dummy_site_indices = siteinds(2, nsite)
    non_padded_site_indices = siteinds(2, nsite - n_eyepad)

    coeffs = derivative_coefficients(derivative_order; stencil_order=stencil_order)
    onedim_mpo = toeplitz_mpo(non_padded_site_indices, coeffs, boundary_conditions, array_type)

    # Note: we use QTT version of neumann boundary correction explicitly here because we have sites
    neumann_corr = if (boundary_conditions.left isa NeumannBC)
        left_neumann_boundary_correction(
            derivative_order,
            arch,
            non_padded_site_indices,
            array_type;
            stencil_order=stencil_order,
            cutoff=cutoff,
            maxdim=maxdim,
        )
    else
        nothing
    end
    right_neumann_corr = if (boundary_conditions.right isa NeumannBC)
        right_neumann_boundary_correction(
            derivative_order,
            arch,
            non_padded_site_indices,
            array_type;
            stencil_order=stencil_order,
            cutoff=cutoff,
            maxdim=maxdim,
        )
    else
        nothing
    end
    neumann_corr = _nothing_sum_handler(neumann_corr, right_neumann_corr; cutoff=cutoff, maxdim=maxdim)

    if !isnothing(neumann_corr)
        onedim_mpo = _nothing_sum_handler(onedim_mpo, -1.0 * neumann_corr; cutoff=cutoff, maxdim=maxdim)
    end

    if n_eyepad > 0
        return onedim_eye_padding(onedim_mpo, dummy_site_indices, :right)
    else
        return onedim_mpo
    end
end

function onedim_differential_operator(
    derivative_order::Int,
    arch::TTArchitecture,
    dim::Int,
    boundary_conditions::BoundaryCond1d,
    array_type::Type{<:AbstractArray}=Array;
    stencil_order::Int=8,
    kwargs...,
)
    return onedim_differential_operator(
        derivative_order, arch, dim, 0, boundary_conditions, array_type; stencil_order=stencil_order, kwargs...
    )
end

function onedim_differential_operator(
    derivative_order::Int,
    arch::TTArchitecture,
    dim::Int,
    n_eyepad::Int,
    boundary_conditions::BoundaryCond1d,
    array_type::Type{<:AbstractArray}=Array;
    stencil_order::Int=8,
    kwargs...,
)
    noneye_dim = arch.var_dims[dim] - n_eyepad
    coeffs = derivative_coefficients(derivative_order; stencil_order=stencil_order)

    mat = zeros(Float64, noneye_dim, noneye_dim)
    for i in 1:noneye_dim
        for (offset, val) in coeffs
            j = i + offset
            if boundary_conditions.left isa PeriodicBC # If periodic
                # Offsets that wrap onto the same node (an axis of at most q nodes) add up
                mat[i, mod1(j, noneye_dim)] += val
            else
                if 1 <= j <= noneye_dim
                    mat[i, j] = val
                end
            end
        end
    end

    left_neumann_corr = if (boundary_conditions.left isa NeumannBC)
        left_neumann_boundary_correction(derivative_order, arch, noneye_dim, array_type; stencil_order=stencil_order)
    else
        nothing
    end
    right_neumann_corr = if (boundary_conditions.right isa NeumannBC)
        right_neumann_boundary_correction(derivative_order, arch, noneye_dim, array_type; stencil_order=stencil_order)
    else
        nothing
    end

    mat_gpu = adapt(array_type, mat)
    if !isnothing(left_neumann_corr)
        mat_gpu -= left_neumann_corr
    end
    if !isnothing(right_neumann_corr)
        mat_gpu -= right_neumann_corr
    end

    if n_eyepad > 0
        full_dim = arch.var_dims[dim]
        padded_mat_cpu = zeros(eltype(mat), full_dim, full_dim)

        # move unpadded to cpu to embed if it was gpu array
        unpadded_cpu = Array(mat_gpu)
        padded_mat_cpu[1:noneye_dim, 1:noneye_dim] = unpadded_cpu
        for i in (noneye_dim+1):full_dim
            padded_mat_cpu[i, i] = 1.0
        end

        return adapt(array_type, padded_mat_cpu)
    else
        return mat_gpu
    end
end

# -------------------------------------------------------------------------------------------------
# DifferentialOps Structs
# -------------------------------------------------------------------------------------------------

"""
    _eyepad_step_size(grid::Grid, arch::QTTArchitecture, dim::Int, n_eyepad::Int) -> Float64
    _eyepad_step_size(grid::Grid, arch::TTArchitecture, dim::Int, n_eyepad::Int) -> Float64

Return the spacing by which [`DifferentialOpsScalar`](@ref) scales an operator eye-padded by `n_eyepad` sites or
nodes: in QTT, ``L/(2^{N_i - p} - 1)``, the spacing of the coarse grid of ``2^{N_i - p}`` nodes over the same box
(nodes at both ends, as in [`Grid`](@ref)); in TT, the original spacing, since the operator acts on the first
``n - p`` nodes.
"""
function _eyepad_step_size(grid::Grid, arch::QTTArchitecture, dim::Int, n_eyepad::Int)
    return grid.box_size[dim] / (2^(abs(arch.var_nsites[dim]) - n_eyepad) - 1)
end
_eyepad_step_size(grid::Grid, arch::TTArchitecture, dim::Int, n_eyepad::Int) = grid.dx[dim]

"""
    DifferentialOpsScalar(grid::Grid{N}, fields::ScalarField{N}; stencil_order::Int=8, cutoff::Float64=1e-15,
                          maxdim::Int=typemax(Int64)) where N
    DifferentialOpsScalar(grid::Grid{N}, fields::ScalarField{N}, n_eyepad::NTuple{N,Int}; stencil_order::Int=8,
                          cutoff::Float64=1e-15, maxdim::Int=typemax(Int64)) where N

First and second derivative operators of a scalar field along each axis, as MPOs on the sites of the field.

Each operator is the one-dimensional operator of [`onedim_differential_operator`](@ref), closed with the boundary
conditions of `fields` on that axis, scaled by ``1/h_i^k`` with the spacing ``h_i`` of `grid` and lifted to ``N``
dimensions by [`ndim_operator_mpo`](@ref). Therefore `d_ops[i]` approximates ``\\partial_i`` and `d2_ops[i]`
approximates ``\\partial_i^2``, with truncated rows at Dirichlet and open sides (see [`DirichletBC`](@ref)). They are
built on the backend of `fields`. With `n_eyepad`, axis ``i`` is eye-padded by `n_eyepad[i]` sites (QTT) or nodes (TT),
and its operators are scaled by the spacing of the grid that the shorter operator acts on: ``L_i/(2^{N_i - p} - 1)`` in
QTT, for a grid of ``2^{N_i - p}`` nodes over the same box, and ``h_i`` in TT.

# Fields

  - `d_ops::NTuple{N,MPO}`: the first derivatives ``\\partial_1, \\dots, \\partial_N``.
  - `d2_ops::NTuple{N,MPO}`: the second derivatives ``\\partial_1^2, \\dots, \\partial_N^2``.

# Examples

```julia
arch = QTTArchitecture((5, 4))
bcs = BoundaryConditions(BoundaryCond1d(DirichletBC(0.0), NeumannBC(0.0)), BoundaryCond1d(PeriodicBC(), PeriodicBC()))
p = ScalarField(siteinds(arch), arch, bcs)
ops = DifferentialOpsScalar(Grid(arch, (2.0, 1.0)), p; stencil_order=8)
q = ones_mps(p.usites, 1)
∂x_q = apply(ops.d_ops[1], q)   # zero up to round-off, except on the truncated Dirichlet rows
```

See also [`DifferentialOps`](@ref), [`kronecker_laplacian`](@ref), and the boundary-condition tutorial
[Velocity–pressure formulation](@ref bc_velocity_pressure).
"""
struct DifferentialOpsScalar{N}
    d_ops::NTuple{N,MPO}   # First derivatives Dx, Dy, Dz...
    d2_ops::NTuple{N,MPO}  # Second derivatives Dxx, Dyy, Dzz...
end

function DifferentialOpsScalar(
    grid::Grid{N}, fields::ScalarField{N}; stencil_order::Int=8, cutoff::Float64=1e-15, maxdim::Int=typemax(Int64)
) where {N}
    array_type = backend_type(fields)

    function make_ops(order::Int)
        ntuple(
            i -> begin
                onedim_op = onedim_differential_operator(
                    order,
                    fields.arch,
                    i,
                    fields.bcu[i],
                    array_type;
                    stencil_order=stencil_order,
                    cutoff=cutoff,
                    maxdim=maxdim,
                )
                if onedim_op isa MPO
                    onedim_op *= (1 / grid.dx[i]^order)
                else
                    onedim_op = onedim_op .* (1 / grid.dx[i]^order)
                end
                stacked_op = ndim_operator_mpo(onedim_op, i, fields.usites, fields.arch)
                return stacked_op
            end,
            N,
        )
    end

    d_ops = make_ops(1)
    d2_ops = make_ops(2)

    return DifferentialOpsScalar{N}(d_ops, d2_ops)
end

function DifferentialOpsScalar(
    grid::Grid{N},
    fields::ScalarField{N},
    n_eyepad::NTuple{N,Int};
    stencil_order::Int=8,
    cutoff::Float64=1e-15,
    maxdim::Int=typemax(Int64),
) where {N}
    array_type = backend_type(fields)

    function make_ops(order::Int)
        ntuple(
            i -> begin
                effective_step_size = _eyepad_step_size(grid, fields.arch, i, n_eyepad[i])

                onedim_op = onedim_differential_operator(
                    order,
                    fields.arch,
                    i,
                    n_eyepad[i],
                    fields.bcu[i],
                    array_type;
                    stencil_order=stencil_order,
                    cutoff=cutoff,
                    maxdim=maxdim,
                )

                if onedim_op isa MPO
                    onedim_op *= (1 / effective_step_size^order)
                else
                    onedim_op = onedim_op .* (1 / effective_step_size^order)
                end

                stacked_op = ndim_operator_mpo(onedim_op, i, fields.usites, fields.arch)
                return stacked_op
            end,
            N,
        )
    end

    d_ops = make_ops(1)
    d2_ops = make_ops(2)

    return DifferentialOpsScalar{N}(d_ops, d2_ops)
end

"""
    DifferentialOps(grid::Grid{N}, fields::VectorField{N}; stencil_order::Int=8, cutoff::Float64=1e-15,
                    maxdim::Int=typemax(Int64)) where N

The [`DifferentialOpsScalar`](@ref) of every component of a vector field, each built with the boundary conditions of
its own component. `ops[i]` returns those of component `i`, so `ops[i].d_ops[j]` approximates ``\\partial_j`` acting
on component ``i``.

# Fields

  - `ops::NTuple{N,DifferentialOpsScalar{N}}`: the operators of the components.
"""
struct DifferentialOps{N}
    ops::NTuple{N,DifferentialOpsScalar{N}}
end

function DifferentialOps(
    grid::Grid{N}, fields::VectorField{N}; stencil_order::Int=8, cutoff::Float64=1e-15, maxdim::Int=typemax(Int64)
) where {N}
    ops = ntuple(
        i -> DifferentialOpsScalar(
            grid, fields.components[i]; stencil_order=stencil_order, cutoff=cutoff, maxdim=maxdim
        ),
        N,
    )
    return DifferentialOps{N}(ops)
end

"""
    getindex(ops::DifferentialOps, i::Int) -> DifferentialOpsScalar

Return the operators of component `i`.
"""
function Base.getindex(ops::DifferentialOps{N}, i::Int) where {N}
    return ops.ops[i]
end

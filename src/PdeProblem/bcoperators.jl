"""
    stencil_rows(stencil_order::Int) -> Vector{Int}

Return the rows next to the left boundary whose central stencil of order ``q`` = `stencil_order` reaches beyond it,
`1:q/2`: `[1]` for ``q = 2`` and `[1, 2, 3, 4]` for ``q = 8``. The rows next to the right boundary are their mirror
images, `n + 1 .- reverse(stencil_rows(q))`. Throws an `ArgumentError` for any other order.
"""
function stencil_rows(stencil_order::Int)
    if stencil_order == 8
        return [1, 2, 3, 4]
    elseif stencil_order == 2
        return [1]
    else
        throw(ArgumentError("Only stencil orders 2 or 8 are implemented"))
    end
end

"""
    onedim_boundary_nullifier(arch::QTTArchitecture, axis::Int, bcs::BoundaryCond1d,
                              array_type::Type{<:AbstractArray}=Array; stencil_order::Int=8, cutoff::Float64=1e-15,
                              maxdim::Int=typemax(Int64)) -> Union{MPO,Nothing}
    onedim_boundary_nullifier(arch::TTArchitecture, axis::Int, bcs::BoundaryCond1d,
                              array_type::Type{<:AbstractArray}=Array; stencil_order::Int=8, cutoff::Float64=1e-15,
                              maxdim::Int=typemax(Int64)) -> Union{AbstractMatrix,Nothing}

Build the one-dimensional diagonal projector of axis `axis` that zeroes the rows next to its Dirichlet sides,

```math
Z = \\mathbb{I} - \\sum_{r \\in R} \\mathbf{e}_r \\mathbf{e}_r^\\top,
```

where ``R`` holds the ``q/2`` rows next to each `DirichletBC` side ([`stencil_rows`](@ref)), i.e. the rows whose
stencil reaches a ghost node. Applied to the right-hand side of an explicit time step, ``Z`` keeps the field fixed on
these rows ([`BoundaryNullScalar`](@ref)), and applied to the divergence, it removes them from the pressure equation
([`divergence_dirichlet_correction`](@ref)). In QTT, ``Z`` is an MPO on new sites of dimension 2, built from
[`row_pick_mpo`](@ref), and in TT a dense ``n \\times n`` matrix. Returns `nothing` if neither side is a `DirichletBC`.

See also the boundary-condition tutorial [Velocity–pressure formulation](@ref bc_velocity_pressure).
"""
function onedim_boundary_nullifier(
    arch::QTTArchitecture,
    axis::Int,
    bcs::BoundaryCond1d,
    array_type::Type{<:AbstractArray}=Array;
    stencil_order::Int=8,
    cutoff::Float64=1e-15,
    maxdim::Int=typemax(Int64),
)
    nsites = abs(arch.var_nsites[axis])
    sites = siteinds(2, nsites)
    nrows = 2^nsites

    onedim_left_row_selector =
        (bcs.left isa DirichletBC) ? row_pick_mpo(sites, stencil_rows(stencil_order), array_type) : nothing
    onedim_right_row_selector = if (bcs.right isa DirichletBC)
        row_pick_mpo(sites, nrows + 1 .- reverse(stencil_rows(stencil_order)), array_type)
    else
        nothing
    end

    onedim_row_selector = _nothing_sum_handler(
        onedim_left_row_selector, onedim_right_row_selector; cutoff=cutoff, maxdim=maxdim
    )

    if !isnothing(onedim_row_selector)
        id = identity_mpo(sites, array_type)
        onedim_row_deleter = +(
            id, (-1.0) * tt_mul(onedim_row_selector, id; cutoff=cutoff, maxdim=maxdim); cutoff=cutoff, maxdim=maxdim
        )
    else
        onedim_row_deleter = nothing
    end

    return onedim_row_deleter
end

function onedim_boundary_nullifier(
    arch::TTArchitecture,
    axis::Int,
    bcs::BoundaryCond1d,
    array_type::Type{<:AbstractArray}=Array;
    stencil_order::Int=8,
    cutoff::Float64=1e-15,
    maxdim::Int=typemax(Int64),
)
    dim = arch.var_dims[axis]
    mat = identity_array(dim)

    if bcs.left isa DirichletBC
        mat[stencil_rows(stencil_order), stencil_rows(stencil_order)] .= 0.0
    end

    if bcs.right isa DirichletBC
        rows = dim + 1 .- reverse(stencil_rows(stencil_order))
        mat[rows, rows] .= 0.0
    end

    if bcs.left isa DirichletBC || bcs.right isa DirichletBC
        return adapt(array_type, mat)
    else
        return nothing
    end
end

"""
    BoundaryNullScalar(field::ScalarField{N}; stencil_order::Int=8, cutoff::Float64=1e-15,
                       maxdim::Int=typemax(Int64)) where N

Projector that holds the Dirichlet boundary rows of a scalar field fixed in the explicit time steppers.

Since the operators truncate the stencil at a Dirichlet side, the ``q/2`` rows next to it do not approximate the
derivatives consistently. We therefore zero these rows of the right-hand side, so that an explicit step leaves the
field there at its initial values, which act as the boundary data (see [`DirichletBC`](@ref)). The projector is the
product ``Z = \\prod_i Z_i`` of the one-dimensional projectors of [`onedim_boundary_nullifier`](@ref), each lifted to
``N`` dimensions by [`ndim_operator_mpo`](@ref), so it zeroes every row next to a Dirichlet side of any axis. It is
built on the backend of `field`.

# Fields

  - `u_row_deleter::Union{MPO,Nothing}`: the projector ``Z``, or `nothing` if `field` has no `DirichletBC` side.

See also [`BoundaryNull`](@ref), and the boundary-condition tutorial [Velocity–pressure formulation](@ref
bc_velocity_pressure).
"""
struct BoundaryNullScalar{N}
    u_row_deleter::Union{MPO,Nothing}
end

function BoundaryNullScalar(
    field::ScalarField{N}; stencil_order::Int=8, cutoff::Float64=1e-15, maxdim::Int=typemax(Int64)
) where {N}
    array_type = backend_type(field)
    deleters = Union{MPO,Nothing}[]

    for i in 1:N
        bc_i = field.bcu[i]
        onedim_deleter = onedim_boundary_nullifier(
            field.arch, i, bc_i, array_type; stencil_order=stencil_order, cutoff=cutoff, maxdim=maxdim
        )
        deleter = (isnothing(onedim_deleter)) ? nothing : ndim_operator_mpo(onedim_deleter, i, field.usites, field.arch)
        push!(deleters, deleter)
    end

    u_row_deleter = _nothing_apply_handler(deleters...; cutoff=cutoff, maxdim=maxdim)
    return BoundaryNullScalar{N}(u_row_deleter)
end

"""
    BoundaryNull(fields::VectorField{N}; stencil_order::Int=8, cutoff::Float64=1e-15,
                 maxdim::Int=typemax(Int64)) where N

The [`BoundaryNullScalar`](@ref) projectors of the ``N`` components of a vector field, each built from the boundary
conditions of its own component; `bn[i]` returns the projector of component `i`.

# Fields

  - `components::NTuple{N,BoundaryNullScalar{N}}`: the projectors of the components.
"""
struct BoundaryNull{N}
    components::NTuple{N,BoundaryNullScalar{N}}
end

function BoundaryNull(
    fields::VectorField{N}; stencil_order::Int=8, cutoff::Float64=1e-15, maxdim::Int=typemax(Int64)
) where {N}
    components = ntuple(
        i -> BoundaryNullScalar(fields.components[i]; stencil_order=stencil_order, cutoff=cutoff, maxdim=maxdim), N
    )
    return BoundaryNull{N}(components)
end

"""
    getindex(bn::BoundaryNull, i::Int) -> BoundaryNullScalar

Return the projector of component `i`.
"""
function Base.getindex(bn::BoundaryNull{N}, i::Int) where {N}
    return bn.components[i]
end

# -------------------------------------------------------------------------------------------------
# Neumann boundary corrections (QTT and TT): they impose NeumannBC(0.0) in the differential operators themselves.
# They break the symmetry of the second derivative, so CG needs the operator and the right-hand side weighted by
# `neumann_weight_mpo` (`symmetrized_laplacian`, `symmetrize_rhs`)
# -------------------------------------------------------------------------------------------------

"""
    neumann_correction_matrix(derivative_order::Int, side::Symbol; stencil_order::Int=8) -> Matrix{Float64}

Return the ``q \\times q`` corner block ``M`` that the homogeneous Neumann closure subtracts from a one-dimensional
operator of derivative order `derivative_order` and stencil order ``q`` = `stencil_order` (2 or 8).

The boundary lies on the stored node ``u_0`` and the ghost values are mirrored, ``u_{-k} = u_k``. The coefficient
``c_k`` that row ``i`` applies to a ghost node ``u_{i+k}``, ``i + k < 0``, therefore moves to the stored node
``u_{-(i+k)}``. With 0-based row and column indices of the block, `side = :left` gives

```math
M_{i,m} = -c_{-(i+m)}, \\qquad 0 \\le i < q/2, \\quad 1 \\le m \\le q/2 - i,
```

with every other entry zero, for the top-left corner, and `side = :right` gives the reflected block
``M_{q-1-i,\\,q-1-m} = -c_{i+m}`` for the bottom-right corner. The closed operator is ``D - M``, where ``D`` is the
truncated Toeplitz operator and ``c_k`` are the coefficients of [`derivative_coefficients`](@ref). Any other `side`
returns the zero block.

# Throws

  - `ArgumentError`: if `stencil_order` is not 2 or 8.

# Examples

```jldoctest
julia> DifferenTT.neumann_correction_matrix(2, :left; stencil_order=2)
2×2 Matrix{Float64}:
 -0.0  -1.0
  0.0   0.0
```

See also the boundary-condition tutorial [Velocity–pressure formulation](@ref bc_velocity_pressure).
"""
function neumann_correction_matrix(derivative_order::Int, side::Symbol; stencil_order::Int=8)
    coef_dict = derivative_coefficients(derivative_order; stencil_order=stencil_order)

    if stencil_order == 2
        mat = zeros(Float64, 2, 2)
        if side == :left
            mat[1, 1:2] .= -coef_dict[-1] .* [0.0, 1.0]
        elseif side == :right
            mat[2, 1:2] .= -coef_dict[1] .* [1.0, 0.0]
        end
        return mat
    elseif stencil_order == 8
        mat = zeros(Float64, 8, 8)
        if side == :left
            ordered_coef = [coef_dict[i] for i in -4:-1]
            left_rolling_coef = vcat([0.0, 1.0], zeros(6))
            toeplitz_matrix = upper_toeplitz_matrix(ordered_coef)

            rolling_matrix = zeros(eltype(left_rolling_coef), 4, length(left_rolling_coef))
            for (idx, j) in enumerate(3:-1:0)
                rolling_matrix[idx, :] = circshift(left_rolling_coef, j)
            end
            correction_matrix = toeplitz_matrix * rolling_matrix
            mat[1:4, 1:8] .= -correction_matrix
        elseif side == :right
            ordered_coef = [coef_dict[i] for i in 4:-1:1]
            right_rolling_coef = vcat(zeros(6), [1.0, 0.0])
            toeplitz_matrix = lower_toeplitz_matrix(ordered_coef)

            rolling_matrix = zeros(eltype(right_rolling_coef), 4, length(right_rolling_coef))
            for (idx, j) in enumerate(0:3)
                rolling_matrix[idx, :] = circshift(right_rolling_coef, -j)
            end
            correction_matrix = toeplitz_matrix * rolling_matrix
            mat[5:8, 1:8] .= -correction_matrix
        end
        return mat
    else
        throw(ArgumentError("Only stencil orders 2 and 8 are supported."))
    end
end

"""
    left_neumann_boundary_correction(derivative_order::Int, arch::TTArchitecture, mat_dim::Int,
                                     array_type::Type{<:AbstractArray}=Array; stencil_order::Int=8) -> AbstractMatrix
    left_neumann_boundary_correction(derivative_order::Int, arch::QTTArchitecture, sites::Vector{<:Index},
                                     array_type::Type{<:AbstractArray}=Array; stencil_order::Int=8,
                                     cutoff::Float64=1e-15, maxdim::Int=typemax(Int64)) -> MPO

Embed the `:left` block ``M`` of [`neumann_correction_matrix`](@ref) in the top-left corner of an otherwise zero
one-dimensional operator: a `mat_dim × mat_dim` matrix (`mat_dim ≥ q`) in TT, or an MPO on the QTT sites `sites`
(at least ``\\log_2 q`` of them), built on ``\\log_2 q`` sites and padded by [`zeros_padding`](@ref). Subtracting it
from the truncated operator gives the homogeneous Neumann closure of the left side. `arch` only selects the
method, and `cutoff` and `maxdim` are not used.

See also [`right_neumann_boundary_correction`](@ref), and the boundary-condition tutorial [Velocity–pressure
formulation](@ref bc_velocity_pressure).
"""
function left_neumann_boundary_correction(
    derivative_order::Int,
    arch::TTArchitecture,
    mat_dim::Int,
    array_type::Type{<:AbstractArray}=Array;
    stencil_order::Int=8,
)
    mat = zeros(Float64, mat_dim, mat_dim)

    correction_matrix = neumann_correction_matrix(derivative_order, :left; stencil_order=stencil_order)
    sz = size(correction_matrix, 1)
    mat[1:sz, 1:sz] .= correction_matrix
    return adapt(array_type, mat)
end

"""
    right_neumann_boundary_correction(derivative_order::Int, arch::TTArchitecture, mat_dim::Int,
                                      array_type::Type{<:AbstractArray}=Array; stencil_order::Int=8) -> AbstractMatrix
    right_neumann_boundary_correction(derivative_order::Int, arch::QTTArchitecture, sites::Vector{<:Index},
                                      array_type::Type{<:AbstractArray}=Array; stencil_order::Int=8,
                                      cutoff::Float64=1e-15, maxdim::Int=typemax(Int64)) -> MPO

Embed the `:right` block ``M`` of [`neumann_correction_matrix`](@ref) in the bottom-right corner of an otherwise zero
one-dimensional operator: a `mat_dim × mat_dim` matrix (`mat_dim ≥ q`) in TT, or an MPO on the QTT sites `sites`
(at least ``\\log_2 q`` of them), built on ``\\log_2 q`` sites and padded by [`zeros_padding`](@ref). Subtracting it
from the truncated operator gives the homogeneous Neumann closure of the right side. `arch` only selects the
method, and `cutoff` and `maxdim` are not used.

See also [`left_neumann_boundary_correction`](@ref), and the boundary-condition tutorial [Velocity–pressure
formulation](@ref bc_velocity_pressure).
"""
function right_neumann_boundary_correction(
    derivative_order::Int,
    arch::TTArchitecture,
    mat_dim::Int,
    array_type::Type{<:AbstractArray}=Array;
    stencil_order::Int=8,
)
    mat = zeros(Float64, mat_dim, mat_dim)

    correction_matrix = neumann_correction_matrix(derivative_order, :right; stencil_order=stencil_order)
    sz = size(correction_matrix, 1)
    mat[(end-sz+1):end, (end-sz+1):end] .= correction_matrix
    return adapt(array_type, mat)
end

function left_neumann_boundary_correction(
    derivative_order::Int,
    arch::QTTArchitecture,
    sites::Vector{<:Index},
    array_type::Type{<:AbstractArray}=Array;
    stencil_order::Int=8,
    cutoff::Float64=1e-15,
    maxdim::Int=typemax(Int64),
)
    correction_matrix = neumann_correction_matrix(derivative_order, :left; stencil_order=stencil_order)

    n_dummy = Int(log2(size(correction_matrix, 1)))
    dummy_sites = siteinds(2, n_dummy)

    correction_mpo = adapt(array_type, mat2mpo(correction_matrix, dummy_sites))

    return zeros_padding(correction_mpo, sites, :upleft)
end

function right_neumann_boundary_correction(
    derivative_order::Int,
    arch::QTTArchitecture,
    sites::Vector{<:Index},
    array_type::Type{<:AbstractArray}=Array;
    stencil_order::Int=8,
    cutoff::Float64=1e-15,
    maxdim::Int=typemax(Int64),
)
    correction_matrix = neumann_correction_matrix(derivative_order, :right; stencil_order=stencil_order)

    n_dummy = Int(log2(size(correction_matrix, 1)))
    dummy_sites = siteinds(2, n_dummy)

    correction_mpo = adapt(array_type, mat2mpo(correction_matrix, dummy_sites))

    return zeros_padding(correction_mpo, sites, :bottomright)
end

# -------------------------------------------------------------------------------------------------
# Symmetrization of the 1D Neumann rows -> -1/2 times the Neumann boundary rows of a 1D operator. Applied
# to the identity, it gives the 1D factors of the Neumann weight W (see `neumann_weight_mpo`), which
# symmetrizes whole linear systems with Neumann BCs
# -------------------------------------------------------------------------------------------------

"""
    onedim_neumann_symmetrization_correction(operator::Union{MPO,AbstractMatrix}, arch::Architecture, axis::Int,
                                             boundary_conditions::BoundaryCond1d; stencil_order::Int=8,
                                             cutoff::Float64=1e-15, maxdim::Int=typemax(Int64))

Return minus one half of the Neumann boundary rows of the one-dimensional operator ``A`` = `operator`,

```math
C = -\\frac{1}{2} \\sum_{r \\in R} \\mathbf{e}_r \\mathbf{e}_r^\\top A,
```

where ``R`` holds the first row if the left side of `boundary_conditions` is a `NeumannBC`, and the last row if the
right side is. ``A + C`` is therefore ``A`` with its Neumann boundary rows halved, i.e. the rows that the mirror closure
counts twice. Applied to the identity, ``\\mathbb{I} + C`` is the one-dimensional factor ``W_i`` of the weight of
[`neumann_weight_mpo`](@ref). `operator` is an MPO for a `QTTArchitecture` (the rows are picked by an element-wise
product with [`row_pick_mpo`](@ref)) and a matrix for a `TTArchitecture`; `axis` and `stencil_order` are not used.
Returns `nothing` if neither side is a `NeumannBC`.

See also the boundary-condition tutorial [Velocity–pressure formulation](@ref bc_velocity_pressure).
"""
function onedim_neumann_symmetrization_correction(
    operator::Union{MPO,AbstractMatrix},
    arch::Architecture,
    axis::Int,
    boundary_conditions::BoundaryCond1d;
    stencil_order::Int=8,
    cutoff::Float64=1e-15,
    maxdim::Int=typemax(Int64),
)
    left_sym = if (boundary_conditions.left isa NeumannBC)
        left_neumann_symmetrization_correction(operator, arch, axis; cutoff=cutoff, maxdim=maxdim)
    else
        nothing
    end
    right_sym = if (boundary_conditions.right isa NeumannBC)
        right_neumann_symmetrization_correction(operator, arch, axis; cutoff=cutoff, maxdim=maxdim)
    else
        nothing
    end

    if arch isa QTTArchitecture
        return _nothing_sum_handler(left_sym, right_sym; cutoff=cutoff, maxdim=maxdim)
    else
        if isnothing(left_sym) && isnothing(right_sym)
            return nothing
        elseif isnothing(left_sym)
            return right_sym
        elseif isnothing(right_sym)
            return left_sym
        else
            return left_sym + right_sym
        end
    end
end

"""
    left_neumann_symmetrization_correction(operator::MPO, arch::QTTArchitecture, axis::Int; cutoff, maxdim) -> MPO
    left_neumann_symmetrization_correction(operator::AbstractMatrix, arch::TTArchitecture, axis::Int; cutoff, maxdim)

Return ``-\\tfrac{1}{2}\\mathbf{e}_1\\mathbf{e}_1^\\top A``, minus half the first row of the one-dimensional operator
``A`` = `operator`; see [`onedim_neumann_symmetrization_correction`](@ref).
"""
function left_neumann_symmetrization_correction(
    operator::MPO, arch::QTTArchitecture, axis::Int; cutoff::Float64=1e-15, maxdim::Int=typemax(Int64)
)
    site_indices = only.(siteinds(operator; plev=0))
    array_type = backend_type(operator)
    row_selector = row_pick_mpo(site_indices, 1, array_type)
    return -0.5 * tt_mul(row_selector, operator; cutoff=cutoff, maxdim=maxdim)
end

"""
    right_neumann_symmetrization_correction(operator::MPO, arch::QTTArchitecture, axis::Int; cutoff, maxdim) -> MPO
    right_neumann_symmetrization_correction(operator::AbstractMatrix, arch::TTArchitecture, axis::Int; cutoff, maxdim)

Return ``-\\tfrac{1}{2}\\mathbf{e}_n\\mathbf{e}_n^\\top A``, minus half the last row of the one-dimensional operator
``A`` = `operator`; see [`onedim_neumann_symmetrization_correction`](@ref).
"""
function right_neumann_symmetrization_correction(
    operator::MPO, arch::QTTArchitecture, axis::Int; cutoff::Float64=1e-15, maxdim::Int=typemax(Int64)
)
    site_indices = only.(siteinds(operator; plev=0))
    array_type = backend_type(operator)
    N = prod(dim.(only.(siteinds(operator; plev=0))))
    row_selector = row_pick_mpo(site_indices, N, array_type)
    return -0.5 * tt_mul(row_selector, operator; cutoff=cutoff, maxdim=maxdim)
end

function left_neumann_symmetrization_correction(
    operator::AbstractMatrix, arch::TTArchitecture, axis::Int; cutoff::Float64=1e-15, maxdim::Int=typemax(Int64)
)
    mat = zero(operator)
    mat[1, :] .= -0.5 .* operator[1, :]
    return mat
end

function right_neumann_symmetrization_correction(
    operator::AbstractMatrix, arch::TTArchitecture, axis::Int; cutoff::Float64=1e-15, maxdim::Int=typemax(Int64)
)
    mat = zero(operator)
    mat[end, :] .= -0.5 .* operator[end, :]
    return mat
end

# -------------------------------------------------------------------------------------------------
# Dirichlet boundary corrections: they move the ghost values of a Dirichlet side to the right-hand side. They are
# built for a unit boundary value and h = 1, and scaled by the value and by 1/h^k afterwards (see
# `scalar_dirichlet_correction_dim` and `face_dirichlet_lift` in linearsyscorrections.jl)
#
# TODO(v0.2): every ghost node beyond a Dirichlet side holds the boundary value g, which is exact only for q = 2 or a
# flat solution, so with q = 8 the closure (and the solution) is only first-order accurate. Odd reflection about the
# first ghost node, u_{n+1+m} = 2g - u_{n+1-m}, gives second order and keeps W A symmetric and -W A SPD (checked
# densely in 1D during WP3b); it changes the q = 8 pressure systems of the Chorin projection (Sec. 5.2, Fig. 3)
# -------------------------------------------------------------------------------------------------

"""
    onedim_left_dirichlet_correction(derivative_order::Int, arch::QTTArchitecture, axis::Int,
                                     array_type::Type{<:AbstractArray}=Array; stencil_order::Int=8) -> MPO
    onedim_left_dirichlet_correction(derivative_order::Int, arch::TTArchitecture, axis::Int,
                                     array_type::Type{<:AbstractArray}=Array; stencil_order::Int=8) -> AbstractMatrix

Build the unscaled (``h = 1``) one-dimensional operator ``C_L`` of axis `axis` whose product with the all-ones vector
gives, on each row, the stencil coefficients that the truncation at the left side dropped.

A truncated stencil takes the ghost values beyond the left boundary as zero. When they all hold the same value ``g``,
the missing contribution to row ``i`` (1-based) is ``g \\sum_{k=-q/2}^{-i} c_k``, where ``c_k`` are the coefficients of
[`derivative_coefficients`](@ref). ``C_L`` is the upper triangular Toeplitz block

```math
(C_L)_{ij} = c_{j-i-4}, \\qquad 1 \\le i \\le j \\le 4,
```

in the top-left corner of an otherwise zero ``n \\times n`` operator, so that
``(C_L \\mathbf{1})_i = \\sum_{k=-4}^{-i} c_k`` (the coefficients beyond ``q/2`` vanish for ``q = 2``). Scaled by
``g/h^k``, ``C_L \\mathbf{1}`` is the lift that completes the truncated operator: the truncated operator applied to the
stored values, plus this lift, equals the full stencil applied to the values extended with the ghost values ``g``. Note
that ``C_L`` is only meaningful applied to a vector that is constant along the axis, since its columns are not the ghost
nodes. In QTT it is built by [`corner_pick_mpo`](@ref) on new sites of dimension 2.

See also [`onedim_right_dirichlet_correction`](@ref), [`scalar_dirichlet_correction`](@ref),
[`face_dirichlet_lift`](@ref), and the boundary-condition tutorial [Velocity–pressure formulation](@ref
bc_velocity_pressure).
"""
function onedim_left_dirichlet_correction(
    derivative_order::Int,
    arch::QTTArchitecture,
    axis::Int,
    array_type::Type{<:AbstractArray}=Array;
    stencil_order::Int=8,
)
    nsites = abs(arch.var_nsites[axis])
    sites = siteinds(2, nsites)
    coeffs = derivative_coefficients(derivative_order; stencil_order=stencil_order)
    left_coeffs = [coeffs[i] for i in -4:-1]
    return corner_pick_mpo(left_coeffs, sites, :upper, array_type)
end

"""
    onedim_right_dirichlet_correction(derivative_order::Int, arch::QTTArchitecture, axis::Int,
                                      array_type::Type{<:AbstractArray}=Array; stencil_order::Int=8) -> MPO
    onedim_right_dirichlet_correction(derivative_order::Int, arch::TTArchitecture, axis::Int,
                                      array_type::Type{<:AbstractArray}=Array; stencil_order::Int=8) -> AbstractMatrix

Build the unscaled (``h = 1``) one-dimensional operator ``C_R`` of the right side, the mirror image of
[`onedim_left_dirichlet_correction`](@ref): the lower triangular Toeplitz block of ``(c_4, c_3, c_2, c_1)`` in the
bottom-right corner, so that ``(C_R \\mathbf{1})_{n+1-r} = \\sum_{k=r}^{4} c_k`` for ``r = 1, \\dots, 4``, the
coefficients that the truncation dropped from the ``r``-th row counted from the right boundary.

See also the boundary-condition tutorial [Velocity–pressure formulation](@ref bc_velocity_pressure).
"""
function onedim_right_dirichlet_correction(
    derivative_order::Int,
    arch::QTTArchitecture,
    axis::Int,
    array_type::Type{<:AbstractArray}=Array;
    stencil_order::Int=8,
)
    nsites = abs(arch.var_nsites[axis])
    sites = siteinds(2, nsites)
    coeffs = derivative_coefficients(derivative_order; stencil_order=stencil_order)
    right_coeffs = [coeffs[i] for i in 4:-1:1]
    return corner_pick_mpo(right_coeffs, sites, :lower, array_type)
end

function onedim_left_dirichlet_correction(
    derivative_order::Int,
    arch::TTArchitecture,
    axis::Int,
    array_type::Type{<:AbstractArray}=Array;
    stencil_order::Int=8,
)
    dim = arch.var_dims[axis]
    coeffs = derivative_coefficients(derivative_order; stencil_order=stencil_order)
    left_coeffs = [coeffs[i] for i in -4:-1]
    mat = zeros(Float64, dim, dim)
    mat[1:4, 1:4] .= corner_pick_matrix(left_coeffs, :upper)
    return adapt(array_type, mat)
end

function onedim_right_dirichlet_correction(
    derivative_order::Int,
    arch::TTArchitecture,
    axis::Int,
    array_type::Type{<:AbstractArray}=Array;
    stencil_order::Int=8,
)
    dim = arch.var_dims[axis]
    coeffs = derivative_coefficients(derivative_order; stencil_order=stencil_order)
    right_coeffs = [coeffs[i] for i in 4:-1:1]
    mat = zeros(Float64, dim, dim)
    mat[(end-3):end, (end-3):end] .= corner_pick_matrix(right_coeffs, :lower)
    return adapt(array_type, mat)
end

"""
    corner_pick_matrix(coefficients::Vector{<:AbstractFloat}, flag::Symbol=:upper) -> Matrix

Return the dense corner block of [`corner_pick_mpo`](@ref): the upper triangular Toeplitz matrix with first row
`coefficients` for `flag = :upper`, and the lower triangular Toeplitz matrix with first column `coefficients` for any
other `flag`.
"""
function corner_pick_matrix(coefficients::Vector{<:AbstractFloat}, flag::Symbol=:upper)
    return flag == :upper ? upper_toeplitz_matrix(coefficients) : lower_toeplitz_matrix(coefficients)
end

# -------------------------------------------------------------------------------------------------
# Wall vorticity corrections: the ghost contribution of no-slip walls to the Laplacian of the vorticity in the
# vorticity-streamfunction formulation, where the walls are ghost nodes one spacing outside the grid
# -------------------------------------------------------------------------------------------------

"""
    onedim_entry_operator(arch::QTTArchitecture, axis::Int, entries::Vector{Tuple{Int,Int,Float64}},
                          array_type::Type{<:AbstractArray}=Array; cutoff::Float64=1e-15,
                          maxdim::Int=typemax(Int64)) -> Union{MPO,Nothing}
    onedim_entry_operator(arch::TTArchitecture, axis::Int, entries::Vector{Tuple{Int,Int,Float64}},
                          array_type::Type{<:AbstractArray}=Array; cutoff::Float64=1e-15,
                          maxdim::Int=typemax(Int64)) -> Union{AbstractMatrix,Nothing}

Build the one-dimensional operator ``\\sum v\\, \\mathbf{e}_r \\mathbf{e}_c^\\top`` of axis `axis` from its nonzero
entries `(r, c, v)` (1-based, repeated entries add up): an MPO on new sites of dimension 2, as a sum of
[`entry_mpo`](@ref) terms, in QTT, and a dense matrix in TT. Returns `nothing` if `entries` is empty.
"""
function onedim_entry_operator(
    arch::QTTArchitecture,
    axis::Int,
    entries::Vector{Tuple{Int,Int,Float64}},
    array_type::Type{<:AbstractArray}=Array;
    cutoff::Float64=1e-15,
    maxdim::Int=typemax(Int64),
)
    isempty(entries) && return nothing
    sites = siteinds(2, abs(arch.var_nsites[axis]))
    terms = [value * entry_mpo(sites, row, column, array_type) for (row, column, value) in entries]
    return _nothing_sum_handler(terms...; cutoff=cutoff, maxdim=maxdim)
end

function onedim_entry_operator(
    arch::TTArchitecture,
    axis::Int,
    entries::Vector{Tuple{Int,Int,Float64}},
    array_type::Type{<:AbstractArray}=Array;
    cutoff::Float64=1e-15,
    maxdim::Int=typemax(Int64),
)
    isempty(entries) && return nothing
    dim = arch.var_dims[axis]
    mat = zeros(Float64, dim, dim)
    for (row, column, value) in entries
        mat[row, column] += value
    end
    return adapt(array_type, mat)
end

"""
    onedim_wall_vorticity_entries(walls::NTuple{2,Bool}, n::Int; formula::Symbol=:jensen)
    onedim_wall_vorticity_entries(bcs::BoundaryCond1d, n::Int; formula::Symbol=:jensen)

Return the nonzero entries `(row, column, value)` (a `Vector{Tuple{Int,Int,Float64}}`) of the unscaled (``h = 1``)
one-dimensional wall vorticity correction on `n` nodes: for each side flagged as a wall in `walls = (left, right)` (or
each `DirichletBC` side of `bcs`), the row of the node next to the wall holds ``c_1 \\psi_1 + c_2 \\psi_2``, where
``\\psi_1`` and ``\\psi_2`` are the first and second nodes away from the wall and ``(c_1, c_2)`` the coefficients of
[`wall_vorticity_coefficients`](@ref) (`(1, 1, c₁)`, `(1, 2, c₂)` on the left; `(n, n, c₁)`, `(n, n - 1, c₂)` on the
right; the ``c_2`` entry is omitted for Thom's formula).

See also the boundary-condition tutorial [Vorticity–streamfunction formulation](@ref bc_vorticity_streamfunction).
"""
function onedim_wall_vorticity_entries(walls::NTuple{2,Bool}, n::Int; formula::Symbol=:jensen)
    c1, c2, _ = wall_vorticity_coefficients(formula)
    entries = Tuple{Int,Int,Float64}[]
    if walls[1]
        push!(entries, (1, 1, c1))
        iszero(c2) || push!(entries, (1, 2, c2))
    end
    if walls[2]
        push!(entries, (n, n, c1))
        iszero(c2) || push!(entries, (n, n - 1, c2))
    end
    return entries
end

function onedim_wall_vorticity_entries(bcs::BoundaryCond1d, n::Int; formula::Symbol=:jensen)
    onedim_wall_vorticity_entries(dirichlet_flags(bcs), n; formula=formula)
end

"""
    dirichlet_flags(bcs::BoundaryCond1d) -> NTuple{2,Bool}

Return `(left, right)`, whether each side of `bcs` is a `DirichletBC`.
"""
dirichlet_flags(bcs::BoundaryCond1d) = (bcs.left isa DirichletBC, bcs.right isa DirichletBC)

"""
    onedim_wall_vorticity_correction(arch::Architecture, axis::Int, bcs::BoundaryCond1d,
                                     array_type::Type{<:AbstractArray}=Array; formula::Symbol=:jensen,
                                     cutoff::Float64=1e-15, maxdim::Int=typemax(Int64))
    onedim_wall_vorticity_correction(arch::Architecture, axis::Int, walls::NTuple{2,Bool},
                                     array_type::Type{<:AbstractArray}=Array; formula::Symbol=:jensen,
                                     cutoff::Float64=1e-15, maxdim::Int=typemax(Int64))

Build the unscaled (``h = 1``) one-dimensional operator ``S_1`` of axis `axis` such that ``S_1 \\boldsymbol{\\psi}``
holds ``c_1 \\psi_1 + c_2 \\psi_2`` on the node next to each wall and zero elsewhere, where ``\\psi_1`` and
``\\psi_2`` are the first and second nodes away from the wall and the wall itself is a ghost node with ``\\psi = 0``
(see [`onedim_wall_vorticity_entries`](@ref)). Dividing by ``h^2`` gives the wall vorticity of a stationary wall, and
dividing by ``h^4`` its ghost contribution to the Laplacian of the vorticity on that node. The walls are the
`DirichletBC` sides of `bcs`, or the sides flagged in `walls = (left, right)`, which allows only some of the
`DirichletBC` sides (the no-slip walls) to be selected. The result is an MPO for a `QTTArchitecture` and a dense matrix
for a `TTArchitecture`; returns `nothing` if there is no wall.

See also [`wall_vorticity_operator`](@ref), and the boundary-condition tutorial [Vorticity–streamfunction
formulation](@ref bc_vorticity_streamfunction).
"""
function onedim_wall_vorticity_correction(
    arch::Architecture,
    axis::Int,
    bcs::BoundaryCond1d,
    array_type::Type{<:AbstractArray}=Array;
    formula::Symbol=:jensen,
    cutoff::Float64=1e-15,
    maxdim::Int=typemax(Int64),
)
    return onedim_wall_vorticity_correction(
        arch, axis, dirichlet_flags(bcs), array_type; formula=formula, cutoff=cutoff, maxdim=maxdim
    )
end

function onedim_wall_vorticity_correction(
    arch::Architecture,
    axis::Int,
    walls::NTuple{2,Bool},
    array_type::Type{<:AbstractArray}=Array;
    formula::Symbol=:jensen,
    cutoff::Float64=1e-15,
    maxdim::Int=typemax(Int64),
)
    n = variable_dims(arch)[axis]
    entries = onedim_wall_vorticity_entries(walls, n; formula=formula)
    return onedim_entry_operator(arch, axis, entries, array_type; cutoff=cutoff, maxdim=maxdim)
end

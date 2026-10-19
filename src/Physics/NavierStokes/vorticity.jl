# -------------------------------------------------------------------------------------------------
# Boundary faces of the vorticity-streamfunction scheme. Each face fixes the conditions of both ψ and ω,
# and the boundary condition types of the fields follow from it (see `vorticity_boundary_conditions`).
# -------------------------------------------------------------------------------------------------

"""
    abstract type VorticityFace

Boundary face of an [`INSVorticityExperiment`](@ref). Faces are given per axis as `faces[axis] = (left, right)`.

Conventions:

  - A profile is a `Vector` of values at the stored nodes along the face, in increasing node index of the other
    axis, i.e. of length `variable_dims(arch)[3 - axis]`. A `Number` means a constant value.
  - The tangential velocity of a face of axis 2 is `u`, and the one of a face of axis 1 is `v`. It is signed in the
    `+` direction of that other axis.
  - The normal velocity of a face of axis `i` is the velocity component along axis `i`, signed in the `+` direction
    of that axis (positive on a left inflow).
"""
abstract type VorticityFace end

"""
    NoSlipWall(; velocity=0.0, psi=0.0)

No-slip wall, possibly moving tangentially with `velocity` (a constant or a profile, see [`VorticityFace`](@ref)).
`ψ = psi` (constant) on the wall, and the wall vorticity is given by the wall vorticity formula of the experiment.
The wall is a ghost node one grid spacing outside the stored nodes.
"""
struct NoSlipWall <: VorticityFace
    velocity::Union{Float64,Vector{Float64}}
    psi::Float64
end
NoSlipWall(; velocity=0.0, psi=0.0) = NoSlipWall(_face_values(velocity), Float64(psi))

"""
    SlipWall(; psi=0.0)

Free-slip (symmetry) wall: `ψ = psi` (constant) and `ω = 0` on the wall. The wall is a ghost node one grid spacing
outside the stored nodes.
"""
struct SlipWall <: VorticityFace
    psi::Float64
end
SlipWall(; psi=0.0) = SlipWall(Float64(psi))

"""
    InflowFace(; psi, omega=0.0, normal_velocity)

Inflow face with a prescribed velocity profile: `ψ = psi` and `ω = omega` on the face, which carries the advective
flux `normal_velocity * omega` into the domain. Each value is a constant or a profile (see [`VorticityFace`](@ref)).
The face is a ghost node one grid spacing outside the stored nodes.

`psi`, `omega` and `normal_velocity` must belong to the same velocity profile, and `psi` must be consistent with the
`psi` of the adjacent walls so that mass is conserved. For example, the parabolic profile of a left inflow into a
channel of height `H` with walls `ψ = 0` (bottom) and `ψ = 2UH/3` (top) is

    u = 4U y (H - y) / H²,    ψ = U (2y²/H - 4y³/(3H²)),    ω = -4U (H - 2y) / H².
"""
struct InflowFace <: VorticityFace
    psi::Union{Float64,Vector{Float64}}
    omega::Union{Float64,Vector{Float64}}
    normal_velocity::Union{Float64,Vector{Float64}}
end
function InflowFace(; psi, omega=0.0, normal_velocity)
    InflowFace(_face_values(psi), _face_values(omega), _face_values(normal_velocity))
end

"""
    OutflowFace()

Zero-gradient outflow: `∂ψ/∂n = 0` and `∂ω/∂n = 0`, with first-order upwind advection on the face. The face is the
last stored node (`NeumannBC`). It assumes that the flow leaves the domain through the face: backflow is not
supported.
"""
struct OutflowFace <: VorticityFace end

"""
    PeriodicFace()

Periodic face. It must be used on both sides of an axis.
"""
struct PeriodicFace <: VorticityFace end

_face_values(x::Real) = Float64(x)
_face_values(x::AbstractVector{<:Real}) = Vector{Float64}(x)

Base.:(==)(a::F, b::F) where {F<:VorticityFace} = all(getfield(a, k) == getfield(b, k) for k in fieldnames(F))

is_ghost_face(face::VorticityFace) = face isa Union{NoSlipWall,SlipWall,InflowFace}

"""
    vorticity_boundary_conditions(faces)

`BoundaryConditions` of both the vorticity and the stream function fields for the faces `faces[axis] = (left, right)`:
`DirichletBC` for `NoSlipWall`, `SlipWall` and `InflowFace`, `NeumannBC` for `OutflowFace` and `PeriodicBC` for
`PeriodicFace`. The `DirichletBC` values are not read: the faces hold the boundary data.
"""
function vorticity_boundary_conditions(faces)
    dirichlet, neumann, periodic = DirichletBC(0.0), NeumannBC(0.0), PeriodicBC()
    face_bc(face) = is_ghost_face(face) ? dirichlet : (face isa OutflowFace ? neumann : periodic)
    return BoundaryConditions(ntuple(i -> BoundaryCond1d(face_bc(faces[i][1]), face_bc(faces[i][2])), length(faces))...)
end

"""
    cavity_faces(U_lid::Real)

Faces of the lid-driven cavity: four no-slip walls with `ψ = 0`, the top one (axis 2, right side) moving in `+x`
with velocity `U_lid`.
"""
cavity_faces(U_lid::Real) = ((NoSlipWall(), NoSlipWall()), (NoSlipWall(), NoSlipWall(velocity=U_lid)))

"""
    vorticity_grid(arch, domain_size, faces)

`Grid` of the stored nodes of the domain `[0, L₁] × [0, L₂]` for the faces `faces`: every `NoSlipWall`, `SlipWall`
and `InflowFace` is a ghost node, an `OutflowFace` is a stored node and a periodic axis has one ghost side (see the
three-argument [`interior_grid`](@ref)).
"""
function vorticity_grid(arch, domain_size, faces)
    interior_grid(arch, domain_size, ntuple(i -> faces[i][1] isa PeriodicFace ? 1 : count(is_ghost_face, faces[i]), 2))
end

noslip_flags(faces) = ntuple(i -> (faces[i][1] isa NoSlipWall, faces[i][2] isa NoSlipWall), 2)

function _expected_bc_type(face::VorticityFace)
    is_ghost_face(face) ? DirichletBC : (face isa OutflowFace ? NeumannBC : PeriodicBC)
end

_face_profiles(face::NoSlipWall) = (face.velocity,)
_face_profiles(face::InflowFace) = (face.psi, face.omega, face.normal_velocity)
_face_profiles(::VorticityFace) = ()

"""
    struct INSVorticityExperiment{N} <: INSExperimentBase

Defines a 2D Incompressible Navier-Stokes experiment without obstacles in the vorticity-streamfunction formulation
(Kiffner & Jaksch, arXiv:2303.03010), with one [`VorticityFace`](@ref) per side of the domain: no-slip walls
(possibly moving), slip walls, inflow, outflow and periodic faces. It describes the physical problem only: the time
integrator and the Poisson solver are passed to the solving scheme (e.g. `vorticity_evolve!`).

Walls and inflow faces are ghost nodes one grid spacing outside the stored nodes, and outflow faces are the last
stored node; build `grid` with [`vorticity_grid`](@ref) (or [`interior_grid`](@ref) for four ghost faces). The
boundary condition types of both fields must match the faces, see [`vorticity_boundary_conditions`](@ref).

# Fields

  - `grid::Grid{N}`: Grid of the stored nodes.
  - `omega_field::ScalarField{N}`: Vorticity `ω = ∂ₓv - ∂ᵧu`.
  - `psi_field::ScalarField{N}`: Stream function `ψ`, with `u = ∂ᵧψ`, `v = -∂ₓψ`.
  - `Re::Number`: Reynolds number.
  - `faces::NTuple{2, Tuple{VorticityFace, VorticityFace}}`: Faces `faces[axis] = (left, right)`.
  - `vmax::Float64`: Characteristic speed of the flow, used by the CFL check.
  - `wall_formula::Symbol`: Wall vorticity formula, `:jensen` (second order, as in the paper) or `:thom` (first order).
  - `stencil_order::Int`: Order of the finite difference stencils (only 2 is supported).
  - `maxdim::Int`: Maximum bond dimension allowed throughout the simulation.
  - `cutoff::Float64`: SVD cutoff performed throughout the simulation.
"""
struct INSVorticityExperiment{N} <: INSExperimentBase
    grid::Grid{N}
    omega_field::ScalarField{N}
    psi_field::ScalarField{N}
    Re::Number
    faces::NTuple{2,Tuple{VorticityFace,VorticityFace}}
    vmax::Float64
    wall_formula::Symbol
    stencil_order::Int
    maxdim::Int
    cutoff::Float64

    """
        INSVorticityExperiment(grid::Grid{N}, omega_field::ScalarField{N}, psi_field::ScalarField{N}, Re::Number, faces::NTuple{2, Tuple{VorticityFace, VorticityFace}}; wall_formula::Symbol=:jensen, stencil_order::Int=2, maxdim::Int=100, cutoff::Float64=1e-8, vmax::Union{Nothing, Real}=nothing) where N

    Construct an INSVorticityExperiment. Checks that the problem is 2D, that the stencils are second order, that
    both fields share their site indices, that the boundary conditions of both fields match the faces, that
    periodic faces come in pairs and that every face profile has one value per node along the face.

    # Arguments
    - `grid::Grid{N}`: Grid of the stored nodes (see [`vorticity_grid`](@ref)).
    - `omega_field::ScalarField{N}`: Initial vorticity (usually zero, fluid at rest).
    - `psi_field::ScalarField{N}`: Stream function field.
    - `Re::Number`: Reynolds number.
    - `faces`: Faces `faces[axis] = (left, right)`, see [`VorticityFace`](@ref).
    - `wall_formula::Symbol`: `:jensen` or `:thom`, see [`wall_vorticity_coefficients`](@ref).
    - `stencil_order::Int`: Finite difference stencil order. Must be 2: the wall vorticity formulas and the ghost
      faces only reach one node beyond the grid.
    - `maxdim::Int`: Maximum bond dimension for TT arithmetic in the evolution. Not the solver bond dimension.
    - `cutoff::Float64`: SVD cutoff for that same TT arithmetic. Not the solver truncation or stopping tolerance.
    - `vmax`: Characteristic speed for the CFL check. Defaults to the largest wall and inflow speed, or to 1 if
      every face is at rest (e.g. doubly periodic flows).
    """
    function INSVorticityExperiment(
        grid::Grid{N},
        omega_field::ScalarField{N},
        psi_field::ScalarField{N},
        Re::Number,
        faces::NTuple{2,Tuple{VorticityFace,VorticityFace}};
        wall_formula::Symbol=:jensen,
        stencil_order::Int=2,
        maxdim::Int=100,
        cutoff::Float64=1e-8,
        vmax::Union{Nothing,Real}=nothing,
    ) where {N}
        N == 2 || throw(ArgumentError("INSVorticityExperiment is only implemented in 2D, got N = $N"))
        stencil_order == 2 ||
            throw(ArgumentError("INSVorticityExperiment only supports stencil_order = 2, got $stencil_order"))
        wall_vorticity_coefficients(wall_formula) # Throws for an unknown formula

        if omega_field.usites != psi_field.usites
            throw(ArgumentError("Physical sites for vorticity and stream function fields do not match"))
        end

        dims = variable_dims(psi_field.arch)
        for axis in 1:N
            left, right = faces[axis]
            if (left isa PeriodicFace) != (right isa PeriodicFace)
                throw(
                    ArgumentError(
                        "PeriodicFace must be on both sides of axis $axis or on neither, got $(typeof(left)) and $(typeof(right))",
                    ),
                )
            end
            for (side_index, side) in ((1, :left), (2, :right))
                face = faces[axis][side_index]
                expected = _expected_bc_type(face)
                for (name, field) in (("omega_field", omega_field), ("psi_field", psi_field))
                    bc = side_index == 1 ? field.bcu[axis].left : field.bcu[axis].right
                    if !(bc isa expected)
                        throw(
                            ArgumentError(
                                "$name has $(typeof(bc)) on the $side side of axis $axis, but the face is a $(typeof(face)), which needs $expected",
                            ),
                        )
                    end
                end
                for profile in _face_profiles(face)
                    if profile isa AbstractVector && length(profile) != dims[3-axis]
                        throw(
                            ArgumentError(
                                "Profile of the $(typeof(face)) on the $side side of axis $axis must have $(dims[3-axis]) values, got $(length(profile))",
                            ),
                        )
                    end
                end
            end
        end

        if isnothing(vmax)
            speeds = [maximum(abs, value) for pair in faces for face in pair for value in (
                if face isa NoSlipWall
                    (face.velocity,)
                elseif face isa InflowFace
                    (face.normal_velocity,)
                else
                    ()
                end
            )]
            vmax = isempty(speeds) ? 0.0 : maximum(speeds)
            if iszero(vmax)
                @debug "Every face is at rest: using vmax = 1 for the CFL check. Pass an explicit vmax to INSVorticityExperiment."
                vmax = 1.0
            end
        end

        check_kolmogorov_scale(grid, Re)

        return new{N}(
            grid, omega_field, psi_field, Re, faces, Float64(vmax), wall_formula, stencil_order, maxdim, cutoff
        )
    end
end

"""
    INSVorticityExperiment(grid::Grid{N}, omega_field::ScalarField{N}, psi_field::ScalarField{N}, Re::Number, U_lid::Real; kwargs...) where N

Lid-driven cavity: the experiment with the faces [`cavity_faces`](@ref)`(U_lid)` (build `grid` with
[`interior_grid`](@ref) and every face of both fields as `DirichletBC`).
"""
function INSVorticityExperiment(
    grid::Grid{N}, omega_field::ScalarField{N}, psi_field::ScalarField{N}, Re::Number, U_lid::Real; kwargs...
) where {N}
    INSVorticityExperiment(grid, omega_field, psi_field, Re, cavity_faces(U_lid); kwargs...)
end

"""
    wall_vorticity_operator(field::ScalarField{N}, grid::Grid{N}; formula::Symbol=:jensen, cutoff::Float64=1e-15, maxdim::Int=typemax(Int64)) where N

Build the MPO `S` of the ghost contribution of the wall vorticity of stationary walls to the Laplacian of the
vorticity: on the node next to each `DirichletBC` wall, `(S ψ) = ω_wall / h² = (c₁ ψ₁ + c₂ ψ₂) / h⁴`. Built as a
sum of 1D [`onedim_wall_vorticity_correction`](@ref) operators lifted to N dimensions, so a node next to two
walls receives both contributions. Returns `nothing` if `field` has no `DirichletBC` face.
"""
function wall_vorticity_operator(
    field::ScalarField{N}, grid::Grid{N}; formula::Symbol=:jensen, cutoff::Float64=1e-15, maxdim::Int=typemax(Int64)
) where {N}
    walls = ntuple(i -> dirichlet_flags(field.bcu[i]), N)
    return wall_vorticity_operator(field, grid, walls; formula=formula, cutoff=cutoff, maxdim=maxdim)
end

"""
    wall_vorticity_operator(field::ScalarField{N}, grid::Grid{N}, walls::NTuple{N, NTuple{2, Bool}}; formula::Symbol=:jensen, cutoff::Float64=1e-15, maxdim::Int=typemax(Int64)) where N

Same as the method without `walls`, with the wall sides of each axis given explicitly by `walls[axis] = (left, right)`
instead of every `DirichletBC` side. Returns `nothing` if there is no wall.
"""
function wall_vorticity_operator(
    field::ScalarField{N},
    grid::Grid{N},
    walls::NTuple{N,NTuple{2,Bool}};
    formula::Symbol=:jensen,
    cutoff::Float64=1e-15,
    maxdim::Int=typemax(Int64),
) where {N}
    array_type = backend_type(field)
    ops = ntuple(
        i -> begin
            onedim_op = onedim_wall_vorticity_correction(
                field.arch, i, walls[i], array_type; formula=formula, cutoff=cutoff, maxdim=maxdim
            )
            isnothing(onedim_op) && return nothing
            onedim_op = onedim_op isa MPO ? onedim_op * (1 / grid.dx[i]^4) : onedim_op .* (1 / grid.dx[i]^4)
            return ndim_operator_mpo(onedim_op, i, field.usites, field.arch)
        end,
        N,
    )
    return _nothing_sum_handler(ops...; cutoff=cutoff, maxdim=maxdim)
end

# -------------------------------------------------------------------------------------------------
# Static boundary terms. Every one of them is a ghost value on a face, restored on the truncated stencils with
# `face_dirichlet_lift`.
# -------------------------------------------------------------------------------------------------

const FACE_SIDES = ((1, :left), (1, :right), (2, :left), (2, :right))

face_at(faces, axis::Int, side::Symbol) = faces[axis][side == :left ? 1 : 2]

# Sign of the moving-wall term of the wall vorticity, from ω = ∂ₓv - ∂ᵧu: +1 on the top (axis 2, right) and
# left (axis 1, left) walls, -1 on the bottom and right walls
tangential_sign(axis::Int, side::Symbol) = ((axis == 2) == (side == :right)) ? 1.0 : -1.0

# Ghost value of ψ, or `nothing` if the face is not a ghost node
ghost_psi(face::Union{NoSlipWall,SlipWall,InflowFace}) = face.psi
ghost_psi(::VorticityFace) = nothing

# Static part of the ghost value of ω. On a no-slip wall the ghost vorticity is
# (c₁ ψ₁ + c₂ ψ₂)/h² + c_w ψ_w/h² + s c_U U_t/h with c_w = -(c₁ + c₂): the first term depends on ψ and is
# `wall_vorticity_operator`, the other two are static.
function ghost_omega(face::NoSlipWall, axis::Int, side::Symbol, h::Real, formula::Symbol)
    c1, c2, cU = wall_vorticity_coefficients(formula)
    return tangential_sign(axis, side) * cU .* face.velocity ./ h .- (c1 + c2) * face.psi / h^2
end
ghost_omega(face::InflowFace, args...) = face.omega
ghost_omega(::VorticityFace, args...) = nothing # Slip walls (ω = 0), outflow and periodic faces

function lift_sum(
    experiment::INSVorticityExperiment,
    field::ScalarField,
    derivative_order::Int,
    terms;
    cutoff::Float64=1e-15,
    maxdim::Int=typemax(Int64),
)
    lifts = [
        face_dirichlet_lift(
            field,
            experiment.grid,
            derivative_order,
            axis,
            side,
            value;
            stencil_order=experiment.stencil_order,
            cutoff=cutoff,
            maxdim=maxdim,
        ) for (axis, side, value) in terms
    ]
    return _nothing_sum_handler(lifts...; cutoff=cutoff, maxdim=maxdim)
end

"""
    outflow_advection_closure(ops::DifferentialOpsScalar{2}, field::ScalarField{2}, grid::Grid{2}, faces; cutoff::Float64=1e-15, maxdim::Int=typemax(Int64))

Replace the outflow rows of the first derivatives of `ops` with the first-order upwind (one-sided) difference.
The mirrored Neumann closure makes those rows zero, so the advection would never carry the vorticity out of the
domain. Assumes that the flow leaves through every `OutflowFace`. Returns `ops` itself if there is no outflow face.
"""
function outflow_advection_closure(
    ops::DifferentialOpsScalar{2},
    field::ScalarField{2},
    grid::Grid{2},
    faces;
    cutoff::Float64=1e-15,
    maxdim::Int=typemax(Int64),
)
    any(face -> face isa OutflowFace, (face for pair in faces for face in pair)) || return ops
    d_ops = ntuple(
        axis -> begin
            n = variable_dims(field.arch)[axis]
            entries = Tuple{Int,Int,Float64}[]
            faces[axis][1] isa OutflowFace && append!(entries, [(1, 1, -1.0), (1, 2, 1.0)]) # (f₂ - f₁) / h
            faces[axis][2] isa OutflowFace && append!(entries, [(n, n, 1.0), (n, n - 1, -1.0)]) # (fₙ - fₙ₋₁) / h
            onedim = onedim_entry_operator(field.arch, axis, entries, backend_type(field); cutoff=cutoff, maxdim=maxdim)
            isnothing(onedim) && return ops.d_ops[axis]
            onedim = onedim isa MPO ? onedim * (1 / grid.dx[axis]) : onedim .* (1 / grid.dx[axis])
            return +(
                ops.d_ops[axis],
                ndim_operator_mpo(onedim, axis, field.usites, field.arch);
                cutoff=cutoff,
                maxdim=maxdim,
            )
        end,
        2,
    )
    return DifferentialOpsScalar{2}(d_ops, ops.d2_ops)
end

"""
    VorticityCache

Operators and static boundary terms of the vorticity-streamfunction scheme that only depend on the experiment and
are built once per run.

# Fields

  - `diff_ops`: First and second derivative operators of the stream function (`DifferentialOpsScalar`), for the velocity.
  - `omega_diff_ops`: First and second derivative operators of the vorticity, for the advection. Same as `diff_ops`
    but with the upwind outflow rows, see [`outflow_advection_closure`](@ref).
  - `laplacian::MPO`: Laplacian of the stream function, weighted on the Neumann (outflow) rows: `W Δψ`, see
    [`symmetrized_laplacian`](@ref).
  - `omega_laplacian::MPO`: Unweighted Laplacian of the vorticity, see [`kronecker_laplacian`](@ref). Equal to
    `laplacian` without outflow faces.
  - `poisson_op::MPO`: Operator `A = -W Δ` of the stream function Poisson system (plus the pin `c 𝟏𝟏ᵀ` if
    `regularize`). It is symmetric positive definite.
  - `wall_op`: Ghost contribution of the wall vorticity of the no-slip walls to `Δω` that depends on ψ, see
    [`wall_vorticity_operator`](@ref). `nothing` without no-slip walls.
  - `wall_source`: Static ghost contribution of the faces to `Δω`: moving walls, walls with `ψ ≠ 0` and inflow vorticity.
  - `psi_lift`: Ghost contribution of the face values of ψ to `Δψ`.
  - `velocity_lift`: Ghost contribution of the face values of ψ to `(∂ₓψ, ∂ᵧψ)`.
  - `advection_lift`: Ghost advective flux of the inflow faces.
  - `regularize::Bool`: Whether ψ has no ghost face, so that the Poisson system is singular and is pinned to zero mean.

The static terms are `nothing` when they vanish.
"""
struct VorticityCache{D,E}
    diff_ops::D
    omega_diff_ops::E
    laplacian::MPO
    omega_laplacian::MPO
    poisson_op::MPO
    wall_op::Union{MPO,Nothing}
    wall_source::Union{MPS,Nothing}
    psi_lift::Union{MPS,Nothing}
    velocity_lift::NTuple{2,Union{MPS,Nothing}}
    advection_lift::Union{MPS,Nothing}
    regularize::Bool
end

"""
    VorticityCache(experiment::INSVorticityExperiment{N}; cutoff::Float64=1e-15, maxdim::Int=typemax(Int64)) where N

Build the operators and the static boundary terms of the vorticity-streamfunction scheme for `experiment`. They are
built with their own tight `cutoff`, not with `experiment.cutoff`, so that the boundary entries are not truncated away.
"""
function VorticityCache(
    experiment::INSVorticityExperiment{N}; cutoff::Float64=1e-15, maxdim::Int=typemax(Int64)
) where {N}
    psi_field = experiment.psi_field
    omega_field = experiment.omega_field
    grid = experiment.grid
    faces = experiment.faces
    formula = experiment.wall_formula
    stencil_order = experiment.stencil_order

    @timeit_debug timeit_to "psi_diff_ops" begin
        diff_ops = DifferentialOpsScalar(grid, psi_field; stencil_order=stencil_order, cutoff=cutoff, maxdim=maxdim)
    end
    @timeit_debug timeit_to "omega_diff_ops" begin
        omega_diff_ops = DifferentialOpsScalar(
            grid, omega_field; stencil_order=stencil_order, cutoff=cutoff, maxdim=maxdim
        )
        omega_diff_ops = outflow_advection_closure(
            omega_diff_ops, omega_field, grid, faces; cutoff=cutoff, maxdim=maxdim
        )
    end
    @timeit_debug timeit_to "laplacian" begin
        Δ = symmetrized_laplacian(psi_field, grid; stencil_order=stencil_order, cutoff=cutoff, maxdim=maxdim)
        Δω = kronecker_laplacian(omega_field, grid; stencil_order=stencil_order, cutoff=cutoff, maxdim=maxdim)
    end

    poisson_op = -1.0 * Δ
    regularize = !any(is_ghost_face, (face for pair in faces for face in pair))
    if regularize
        # Pin the constant null space: with c = 1/(number of nodes), A 𝟏 = 𝟏. Summed exactly (bond dimension + 1): a
        # relative cutoff on the large Laplacian would blur the small rank-1 pin
        c = 1.0 / prod(variable_dims(psi_field.arch))
        poisson_op = +(poisson_op, c * ones_mpo(psi_field.usites, backend_type(Δ)); alg="directsum")
    end

    @timeit_debug timeit_to "wall vorticity operators" begin
        wall_op = wall_vorticity_operator(
            psi_field, grid, noslip_flags(faces); formula=formula, cutoff=cutoff, maxdim=maxdim
        )
        wall_source = lift_sum(
            experiment,
            omega_field,
            2,
            [
                (axis, side, ghost_omega(face_at(faces, axis, side), axis, side, grid.dx[axis], formula)) for
                (axis, side) in FACE_SIDES
            ];
            cutoff=cutoff,
            maxdim=maxdim,
        )
    end
    @timeit_debug timeit_to "boundary lifts" begin
        psi_lift = lift_sum(
            experiment,
            psi_field,
            2,
            [(axis, side, ghost_psi(face_at(faces, axis, side))) for (axis, side) in FACE_SIDES];
            cutoff=cutoff,
            maxdim=maxdim,
        )
        velocity_lift = ntuple(
            i -> lift_sum(
                experiment,
                psi_field,
                1,
                [(axis, side, ghost_psi(face_at(faces, axis, side))) for (axis, side) in FACE_SIDES if axis == i];
                cutoff=cutoff,
                maxdim=maxdim,
            ),
            2,
        )
        inflow_fluxes = [
            (axis, side, face.normal_velocity .* face.omega) for (axis, side) in FACE_SIDES for
            face in (face_at(faces, axis, side),) if face isa InflowFace
        ]
        advection_lift = lift_sum(experiment, omega_field, 1, inflow_fluxes; cutoff=cutoff, maxdim=maxdim)
    end

    return VorticityCache(
        diff_ops,
        omega_diff_ops,
        Δ,
        Δω,
        poisson_op,
        wall_op,
        wall_source,
        psi_lift,
        velocity_lift,
        advection_lift,
        regularize,
    )
end

"""
    poisson_rhs(experiment::INSVorticityExperiment, cache::VorticityCache, omega::MPS)

Right-hand side of the stream function Poisson system `A ψ = W (ω + ℓ)`, from `-Δψ = ω` with the truncated
Laplacian completed by the ghost values of ψ (`ℓ = cache.psi_lift`) and weighted on the outflow rows (`W`, see
[`symmetrize_rhs`](@ref)). If `cache.regularize`, the mean is removed so that the singular system is consistent.
For four walls with `ψ = 0` (the lid-driven cavity), it is the vorticity itself.
"""
function poisson_rhs(experiment::INSVorticityExperiment, cache::VorticityCache, omega::MPS)
    cutoff, maxdim = experiment.cutoff, experiment.maxdim
    rhs = _nothing_sum_handler(omega, cache.psi_lift; cutoff=cutoff, maxdim=maxdim)
    # The weight keeps its default tight cutoff: experiment.cutoff could drop the halved outflow rows on large grids
    rhs = symmetrize_rhs(experiment.psi_field, rhs; maxdim=maxdim)
    if cache.regularize && !iszero_tt(rhs)
        rhs = remove_mean(rhs; cutoff=cutoff, maxdim=maxdim)
    end
    return rhs
end

"""
    poisson_streamfunction(experiment::INSVorticityExperiment{N}, cache::VorticityCache, omega::MPS, solver::AbstractLinearSolver) where N

Solve the stream function Poisson equation `-Δψ = ω` with the face values of ψ, i.e. the system `A ψ = b` with
`b =` [`poisson_rhs`](@ref)`(experiment, cache, omega)`, using the previous stream function (if any, and nonzero) as
initial guess. If `b` is identically zero (fluid at rest with `ψ = 0` on every face), returns `ψ = 0` without
calling the solver.

# Arguments

  - `experiment::INSVorticityExperiment{N}`: The experimental setup.
  - `cache::VorticityCache`: Precomputed operators.
  - `omega::MPS`: Vorticity.
  - `solver::AbstractLinearSolver`: Linear solver for the Poisson system. The operator is symmetric positive definite,
    so CG-based solvers apply. A `ResidualObserver` without `A`/`b` in its `SweepingArgs` is bound to this system.

# Returns

  - The stream function `ψ` and the l2 norm of the residual `||A ψ - b||`.
"""
function poisson_streamfunction(
    experiment::INSVorticityExperiment{N}, cache::VorticityCache, omega::MPS, solver::AbstractLinearSolver
) where {N}
    rhs = poisson_rhs(experiment, cache, omega)
    if iszero_tt(rhs)
        return zeros_mps(experiment.psi_field.usites, backend_type(rhs)), 0.0
    end
    A = cache.poisson_op

    previous_psi = last(experiment.psi_field)
    psi_guess = (isnothing(previous_psi) || iszero_tt(previous_psi)) ? nothing : previous_psi

    @timeit_debug timeit_to "poisson solver" begin
        psi = linearsolve(solver, A, rhs, psi_guess)
    end

    # l2 error of the linear solver = ||Ax - b||² = <psi|A†A|psi> - 2Re<psi|A†|b> + <b|b>
    term1 = ITensorMPS.inner(A, psi, A, psi)
    term2 = 2 * real(ITensorMPS.inner(rhs', A, psi))
    term3 = ITensorMPS.inner(rhs, rhs)

    res_l2norm = sqrt(max(0.0, term1 - term2 + term3))

    return psi, res_l2norm
end

"""
    streamfunction_velocity(cache::VorticityCache, psi::MPS; cutoff::Float64=1e-15, maxdim::Int=typemax(Int64))

Velocity `(u, v) = (∂ᵧψ, -∂ₓψ)` of the stream function `psi`, on the stored nodes, with the central differences
completed by the face values of ψ (`cache.velocity_lift`). Without face values (e.g. four walls with `ψ = 0`), it
is exactly discretely divergence-free: `∂ₓu + ∂ᵧv = (∂ₓ∂ᵧ - ∂ᵧ∂ₓ)ψ = 0`.
"""
function streamfunction_velocity(cache::VorticityCache, psi::MPS; cutoff::Float64=1e-15, maxdim::Int=typemax(Int64))
    d_ops = cache.diff_ops.d_ops
    u = _nothing_sum_handler(
        apply(d_ops[2], psi; cutoff=cutoff, maxdim=maxdim), cache.velocity_lift[2]; cutoff=cutoff, maxdim=maxdim
    )
    v =
        -1.0 * _nothing_sum_handler(
            apply(d_ops[1], psi; cutoff=cutoff, maxdim=maxdim), cache.velocity_lift[1]; cutoff=cutoff, maxdim=maxdim
        )
    return u, v
end

"""
    wall_vorticity_source(cache::VorticityCache, psi::MPS; cutoff::Float64=1e-15, maxdim::Int=typemax(Int64))

Ghost contribution `s = S ψ + ℓ` of the face vorticity to the Laplacian of the vorticity on the nodes next to the
ghost faces, i.e. the term that completes the truncated Laplacian `Δω` with the wall values given by the wall
vorticity formula (`S ψ`, see [`wall_vorticity_operator`](@ref)) and the static ghost values `ℓ = cache.wall_source`
(moving walls, walls with `ψ ≠ 0`, inflow vorticity). Returns `nothing` if there is no such contribution.
"""
function wall_vorticity_source(cache::VorticityCache, psi::MPS; cutoff::Float64=1e-15, maxdim::Int=typemax(Int64))
    wall_term = isnothing(cache.wall_op) ? nothing : apply(cache.wall_op, psi; cutoff=cutoff, maxdim=maxdim)
    return _nothing_sum_handler(wall_term, cache.wall_source; cutoff=cutoff, maxdim=maxdim)
end

"""
    VorticityTransportRHS{E, C} <: AbstractRHS

Right-hand side of the vorticity transport equation in conservative form,

    ∂ₜω = -∂ₓ(u ω) - ∂ᵧ(v ω) + (1/Re) (Δω + s),

with the velocity `(u, v)` and the wall vorticity source `s` frozen over the stages of the time integrator. The
advective fluxes through the walls vanish because the normal velocity does, so the truncated stencils (ghost value 0)
give them exactly. The inflow fluxes come from `cache.advection_lift`, and the outflow faces use the upwind rows of
`cache.omega_diff_ops` (see [`outflow_advection_closure`](@ref)). The diffusion uses the unweighted
`cache.omega_laplacian`.

# Fields

  - `experiment::E`: The experimental setup (`Re`, `cutoff`, `maxdim`).
  - `cache::C`: Precomputed operators (`VorticityCache`).
  - `u::MPS`, `v::MPS`: Velocity components.
  - `source`: Wall vorticity source `s`, see [`wall_vorticity_source`](@ref). `nothing` if there is none.
"""
struct VorticityTransportRHS{E<:INSExperimentBase,C} <: AbstractRHS
    experiment::E
    cache::C
    u::MPS
    v::MPS
    source::Union{MPS,Nothing}
end

function (rhs::VorticityTransportRHS)(omega::Tuple, t::Real, stage::Int)
    experiment = rhs.experiment
    cutoff = experiment.cutoff
    maxdim = experiment.maxdim
    ω = only(omega)
    d_ops = rhs.cache.omega_diff_ops.d_ops

    # 1. Advection (conservative form)
    @timeit_debug timeit_to "advection" begin
        uω = tt_mul(rhs.u, ω; cutoff=cutoff, maxdim=maxdim)
        vω = tt_mul(rhs.v, ω; cutoff=cutoff, maxdim=maxdim)
        advection_term = _nothing_sum_handler(
            apply(d_ops[1], uω; cutoff=cutoff, maxdim=maxdim),
            apply(d_ops[2], vω; cutoff=cutoff, maxdim=maxdim),
            rhs.cache.advection_lift;
            cutoff=cutoff,
            maxdim=maxdim,
        )
    end

    # 2. Diffusion, completed with the face vorticity
    @timeit_debug timeit_to "diffusion" begin
        diffusion_term = _nothing_sum_handler(
            apply(rhs.cache.omega_laplacian, ω; cutoff=cutoff, maxdim=maxdim), rhs.source; cutoff=cutoff, maxdim=maxdim
        )
    end

    # 3. Combine RHS = - ∇⋅(u ω) + (1/Re) (Δω + s)
    return (+(-1.0 * advection_term, (1 / experiment.Re) * diffusion_term; cutoff=cutoff, maxdim=maxdim),)
end

"""
    vorticity_transport(experiment::INSVorticityExperiment{N}, cache::VorticityCache, omega::MPS, psi::MPS, stepper::AbstractExplicitStepper, t::Real) where N

Advance the vorticity `omega` by one time step `stepper.time_evol_kwargs.dt`, with the velocity and the wall
vorticity of the stream function `psi` frozen over the stages of `stepper`.

# Returns

  - The vorticity at `t + dt`.
"""
function vorticity_transport(
    experiment::INSVorticityExperiment{N},
    cache::VorticityCache,
    omega::MPS,
    psi::MPS,
    stepper::AbstractExplicitStepper,
    t::Real,
) where {N}
    cutoff = experiment.cutoff
    maxdim = experiment.maxdim

    @timeit_debug timeit_to "velocity" begin
        u, v = streamfunction_velocity(cache, psi; cutoff=cutoff, maxdim=maxdim)
    end
    @timeit_debug timeit_to "wall vorticity" begin
        source = wall_vorticity_source(cache, psi; cutoff=cutoff, maxdim=maxdim)
    end

    rhs_func = VorticityTransportRHS(experiment, cache, u, v, source)
    @timeit_debug timeit_to "timestep" begin
        omega_next = only(timestep(stepper, (omega,), rhs_func, t))
    end
    return omega_next
end

"""
    vorticity_step(experiment::INSVorticityExperiment{N}, cache::VorticityCache, omega::MPS, solver::AbstractLinearSolver, stepper::AbstractExplicitStepper, t::Real) where N

Advance the vorticity `omega` by one time step `stepper.time_evol_kwargs.dt` with the vorticity-streamfunction
scheme: one stream function Poisson solve, then the explicit vorticity transport with the resulting velocity and
wall vorticity frozen over the stages of `stepper`. Does not modify `experiment`; storing the results is left to
the caller.

# Arguments

  - `experiment::INSVorticityExperiment{N}`: The experimental setup. Its last stream function (if any) is used as the Poisson initial guess.
  - `cache::VorticityCache`: Precomputed operators.
  - `omega::MPS`: Vorticity at time `t`.
  - `solver::AbstractLinearSolver`: Linear solver for the stream function Poisson equation.
  - `stepper::AbstractExplicitStepper`: Time integrator for the vorticity transport.
  - `t::Real`: Time at the start of the step.

# Returns

  - `(omega_next, psi, linsolve_error)`: The vorticity at `t + dt`, the stream function at `t` and the l2 residual norm of the Poisson solve.
"""
function vorticity_step(
    experiment::INSVorticityExperiment{N},
    cache::VorticityCache,
    omega::MPS,
    solver::AbstractLinearSolver,
    stepper::AbstractExplicitStepper,
    t::Real,
) where {N}
    psi, linsolve_error = poisson_streamfunction(experiment, cache, omega, solver)
    omega_next = vorticity_transport(experiment, cache, omega, psi, stepper, t)
    return omega_next, psi, linsolve_error
end

"""
    vorticity_evolve!(experiment::INSVorticityExperiment{N}, solver::AbstractLinearSolver, stepper::AbstractExplicitStepper, save_every::Int=1, logger=nothing, log_filename::String="vorticity_log") where N

Run the time evolution of the vorticity-streamfunction experiment. Updates the fields in `experiment` in-place
(appends to time series): at each checkpoint, the stream function of the step and the vorticity at its end.

# Arguments

  - `experiment::INSVorticityExperiment{N}`: The experiment object to evolve. Its vorticity must be initialized and its stream function empty.
  - `solver::AbstractLinearSolver`: Linear solver for the stream function Poisson equation (`ALS`, `ResidualALS`, `GlobalCG`, ...).
  - `stepper::AbstractExplicitStepper`: Time integrator for the vorticity transport (`ExplicitEuler`, `RK4`, ...). Its `TimeEvolArgs` set the time step and the number of steps.
  - `save_every::Int`: MPS saving frequency. Defaults to 1 (save every step).
  - `logger`: Optional logger for data persistence.
  - `log_filename::String`: Filename prefix for logged data.
"""
function vorticity_evolve!(
    experiment::INSVorticityExperiment{N},
    solver::AbstractLinearSolver,
    stepper::AbstractExplicitStepper,
    save_every::Int=1,
    logger=nothing,
    log_filename::String="vorticity_log",
) where {N}
    length(experiment.omega_field) == 0 && throw(ArgumentError("Vorticity field in experiment must be initialized."))
    length(experiment.psi_field) > 0 && throw(ArgumentError("Stream function field must not be initialized."))

    time_evol_kwargs = stepper.time_evol_kwargs
    check_cfl(experiment, stepper; vmax=experiment.vmax)

    save_checkpoints = collect(1:save_every:time_evol_kwargs.nsteps)
    save_checkpoints = if save_checkpoints[end] == time_evol_kwargs.nsteps
        save_checkpoints
    else
        push!(save_checkpoints, time_evol_kwargs.nsteps)
    end

    cache = VorticityCache(experiment)
    linsolve_errorlog = Float64[]

    for s in 1:time_evol_kwargs.nsteps
        t = time_evol_kwargs.ti + (s - 1) * time_evol_kwargs.dt
        omega_next, psi, linsolve_error = vorticity_step(
            experiment, cache, last(experiment.omega_field), solver, stepper, t
        )

        if s in save_checkpoints
            push!(experiment.psi_field, psi)
            push!(experiment.omega_field, omega_next)
            push!(linsolve_errorlog, linsolve_error)
        else
            experiment.psi_field[length(experiment.psi_field)] = psi
            experiment.omega_field[length(experiment.omega_field)] = omega_next
        end
    end

    if !isnothing(logger)
        log_data(logger, log_filename, "omega_field", experiment.omega_field)
        log_data(logger, log_filename, "psi_field", experiment.psi_field)
        log_data(logger, log_filename, "linsolve_errorlog", linsolve_errorlog)
    end
end

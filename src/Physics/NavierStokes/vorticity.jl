# -------------------------------------------------------------------------------------------------
# Boundary faces of the vorticity-streamfunction scheme. Each face fixes the conditions of both ψ and ω,
# and the boundary condition types of the fields follow from it (see `vorticity_boundary_conditions`).
# -------------------------------------------------------------------------------------------------

"""
    abstract type VorticityFace

Boundary face of an [`INSVorticityExperiment`](@ref), given per axis as `faces[axis] = (left, right)`.

In the vorticity–streamfunction formulation, every face fixes the conditions of both ``\\psi`` and ``\\omega`` at
once, and the boundary-condition types of the two fields follow from it ([`vorticity_boundary_conditions`](@ref)).
Walls and inflows ([`NoSlipWall`](@ref), [`SlipWall`](@ref), [`InflowFace`](@ref)) are ghost nodes one spacing outside
the stored nodes, an [`OutflowFace`](@ref) is the last stored node, and a [`PeriodicFace`](@ref) wraps the axis.

The face data follow three conventions:

  - A profile is a `Vector` of values at the stored nodes along the face, in increasing node index of the other axis,
    i.e. of length `variable_dims(arch)[3 - axis]`; a `Number` is a constant value.
  - The tangential velocity of a face of axis 2 is ``u``, and that of a face of axis 1 is ``v``, signed in the ``+``
    direction of the other axis.
  - The normal velocity of a face of axis ``i`` is the velocity component along axis ``i``, signed in the ``+``
    direction of that axis (positive on a left inflow).

See also the boundary-condition tutorial [Vorticity–streamfunction formulation](@ref bc_vorticity_streamfunction).
"""
abstract type VorticityFace end

"""
    NoSlipWall(; velocity=0.0, psi=0.0)

No-slip wall, possibly moving tangentially with `velocity` (a constant or a profile, see [`VorticityFace`](@ref)). The
wall is a ghost node one spacing outside the stored nodes, with ``\\psi`` = `psi` (a constant) on it. Its vorticity is
not prescribed: the wall vorticity formula of the experiment ([`wall_vorticity_coefficients`](@ref)) recovers it
from ``\\psi`` near the wall at every step, which splits into a part that depends on ``\\psi``
([`wall_vorticity_operator`](@ref)) and a static part (`ghost_omega`).

See also the boundary-condition tutorial [Vorticity–streamfunction formulation](@ref bc_vorticity_streamfunction).
"""
struct NoSlipWall <: VorticityFace
    velocity::Union{Float64,Vector{Float64}}
    psi::Float64
end
NoSlipWall(; velocity=0.0, psi=0.0) = NoSlipWall(_face_values(velocity), Float64(psi))

"""
    SlipWall(; psi=0.0)

Free-slip (symmetry) wall, a ghost node one spacing outside the stored nodes with ``\\psi`` = `psi` (a constant) and
``\\omega = 0`` on it.

See also the boundary-condition tutorial [Vorticity–streamfunction formulation](@ref bc_vorticity_streamfunction).
"""
struct SlipWall <: VorticityFace
    psi::Float64
end
SlipWall(; psi=0.0) = SlipWall(Float64(psi))

"""
    InflowFace(; psi, omega=0.0, normal_velocity)

Inflow face with a prescribed velocity profile, a ghost node one spacing outside the stored nodes with ``\\psi`` =
`psi` and ``\\omega`` = `omega` on it, which carries the advective flux `normal_velocity * omega` into the domain.
Each value is a constant or a profile (see [`VorticityFace`](@ref)).

`psi`, `omega` and `normal_velocity` must belong to the same velocity profile, and `psi` must match the `psi` of the
adjacent walls, since the difference of ``\\psi`` between two walls is the flow rate between them. For example, the
parabolic profile of a left inflow into a channel of height ``H``, with walls ``\\psi = 0`` (bottom) and
``\\psi = 2UH/3`` (top), is

```math
u = \\frac{4Uy(H - y)}{H^2}, \\qquad \\psi = U \\left( \\frac{2y^2}{H} - \\frac{4y^3}{3H^2} \\right), \\qquad
\\omega = -\\frac{4U(H - 2y)}{H^2}.
```

See also the boundary-condition tutorial [Vorticity–streamfunction formulation](@ref bc_vorticity_streamfunction).
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

Zero-gradient outflow, ``\\partial \\psi / \\partial n = 0`` and ``\\partial \\omega / \\partial n = 0``, on the last
stored node (`NeumannBC`), with first-order upwind advection on the face ([`outflow_advection_closure`](@ref)). It
assumes that the flow leaves the domain through the face; backflow is not supported.

See also the boundary-condition tutorial [Vorticity–streamfunction formulation](@ref bc_vorticity_streamfunction).
"""
struct OutflowFace <: VorticityFace end

"""
    PeriodicFace()

Periodic face (`PeriodicBC`). It must be used on both sides of an axis.

See also the boundary-condition tutorial [Vorticity–streamfunction formulation](@ref bc_vorticity_streamfunction).
"""
struct PeriodicFace <: VorticityFace end

"""
    _face_values(x::Real) -> Float64
    _face_values(x::AbstractVector{<:Real}) -> Vector{Float64}

Convert a face value to `Float64`, or a face profile to `Vector{Float64}`.
"""
_face_values(x::Real) = Float64(x)
_face_values(x::AbstractVector{<:Real}) = Vector{Float64}(x)

"""
    ==(a::F, b::F) where {F<:VorticityFace}

Two faces of the same type are equal when all their fields are.
"""
Base.:(==)(a::F, b::F) where {F<:VorticityFace} = all(getfield(a, k) == getfield(b, k) for k in fieldnames(F))

"""
    is_ghost_face(face::VorticityFace) -> Bool

Return whether `face` is a ghost face ([`NoSlipWall`](@ref), [`SlipWall`](@ref) or [`InflowFace`](@ref)), whose values
sit on a ghost node one spacing outside the stored nodes.
"""
is_ghost_face(face::VorticityFace) = face isa Union{NoSlipWall,SlipWall,InflowFace}

"""
    vorticity_boundary_conditions(faces) -> BoundaryConditions

Return the [`BoundaryConditions`](@ref) of both the vorticity and the stream function for the faces
`faces[axis] = (left, right)`: `DirichletBC` for the ghost faces ([`NoSlipWall`](@ref), [`SlipWall`](@ref),
[`InflowFace`](@ref)), so that the operators truncate their stencils there, `NeumannBC` for an [`OutflowFace`](@ref)
and `PeriodicBC` for a [`PeriodicFace`](@ref). The values of the `DirichletBC` are zero and are not read: the faces hold
the boundary data, which enter through the lifts of [`VorticityCache`](@ref).

# Examples

```jldoctest
julia> bcs = vorticity_boundary_conditions((
           (InflowFace(; psi=0.0, normal_velocity=1.0), OutflowFace()), (NoSlipWall(), NoSlipWall())
       ));

julia> typeof(bcs[1].right), typeof(bcs[2].left)
(NeumannBC{Float64}, DirichletBC{Float64})
```

See also the boundary-condition tutorial [Vorticity–streamfunction formulation](@ref bc_vorticity_streamfunction).
"""
function vorticity_boundary_conditions(faces)
    dirichlet, neumann, periodic = DirichletBC(0.0), NeumannBC(0.0), PeriodicBC()
    face_bc(face) = is_ghost_face(face) ? dirichlet : (face isa OutflowFace ? neumann : periodic)
    return BoundaryConditions(ntuple(i -> BoundaryCond1d(face_bc(faces[i][1]), face_bc(faces[i][2])), length(faces))...)
end

"""
    cavity_faces(U_lid::Real)

Return the faces of the lid-driven cavity (Ghia, Ghia & Shin, 1982): four [`NoSlipWall`](@ref)s with ``\\psi = 0``,
the top one (axis 2, right side) moving in ``+x`` with velocity `U_lid`.
"""
cavity_faces(U_lid::Real) = ((NoSlipWall(), NoSlipWall()), (NoSlipWall(), NoSlipWall(velocity=U_lid)))

"""
    vorticity_grid(arch, domain_size, faces) -> Grid

Return the [`Grid`](@ref) of the stored nodes of the domain ``[0, L_1] \\times [0, L_2]`` for the faces `faces`: every
ghost face (no-slip and slip walls, inflows) adds a ghost side to its axis, an [`OutflowFace`](@ref) is a stored node,
and a periodic axis has one ghost side, the image of its first node. The spacing of axis ``i`` is therefore
``h_i = L_i/(n_i - 1 + g_i)`` (see the three-argument [`interior_grid`](@ref)).

See also the boundary-condition tutorial [Vorticity–streamfunction formulation](@ref bc_vorticity_streamfunction).
"""
function vorticity_grid(arch, domain_size, faces)
    interior_grid(arch, domain_size, ntuple(i -> faces[i][1] isa PeriodicFace ? 1 : count(is_ghost_face, faces[i]), 2))
end

"""
    noslip_flags(faces) -> NTuple{2,NTuple{2,Bool}}

Return, per axis, `(left, right)` flags of the [`NoSlipWall`](@ref) faces, the walls of
[`wall_vorticity_operator`](@ref).
"""
noslip_flags(faces) = ntuple(i -> (faces[i][1] isa NoSlipWall, faces[i][2] isa NoSlipWall), 2)

"""
    _expected_bc_type(face::VorticityFace) -> Type{<:BoundaryCond}

Return the boundary-condition type that both fields must have on the side of `face`: `DirichletBC` for a ghost face,
`NeumannBC` for an outflow and `PeriodicBC` for a periodic face.
"""
function _expected_bc_type(face::VorticityFace)
    is_ghost_face(face) ? DirichletBC : (face isa OutflowFace ? NeumannBC : PeriodicBC)
end

"""
    _face_profiles(face::VorticityFace) -> Tuple

Return the values of `face` that may be profiles, whose lengths the experiment checks: the velocity of a
[`NoSlipWall`](@ref), and `psi`, `omega` and `normal_velocity` of an [`InflowFace`](@ref).
"""
_face_profiles(face::NoSlipWall) = (face.velocity,)
_face_profiles(face::InflowFace) = (face.psi, face.omega, face.normal_velocity)
_face_profiles(::VorticityFace) = ()

"""
    struct INSVorticityExperiment{N} <: INSExperimentBase

Two-dimensional incompressible Navier–Stokes experiment without obstacles in the vorticity–streamfunction
formulation (Kiffner & Jaksch, 2023), with one [`VorticityFace`](@ref) per side of the domain: no-slip walls (possibly
moving), slip walls, inflows, outflows and periodic faces.

The formulation evolves the vorticity ``\\omega = \\partial_x v - \\partial_y u`` by the transport equation of
[`VorticityTransportRHS`](@ref), and recovers the velocity from the stream function, ``u = \\partial_y \\psi``,
``v = -\\partial_x \\psi``, which solves ``-\\Delta \\psi = \\omega``. The velocity is therefore discretely
divergence-free by construction, without a pressure projection. The experiment describes the physical problem only:
the time integrator and the Poisson solver are passed to the scheme, [`vorticity_evolve!`](@ref) or
[`vorticity_step`](@ref).

Walls and inflow faces are ghost nodes one spacing outside the stored nodes, and outflow faces are the last stored
node, so `grid` must be built with [`vorticity_grid`](@ref) (or [`interior_grid`](@ref) for four ghost faces). The
boundary-condition types of both fields must match the faces ([`vorticity_boundary_conditions`](@ref)).

# Fields

  - `grid::Grid{N}`: Grid of the stored nodes.
  - `omega_field::ScalarField{N}`: Vorticity `ω = ∂ₓv - ∂ᵧu`.
  - `psi_field::ScalarField{N}`: Stream function `ψ`, with `u = ∂ᵧψ`, `v = -∂ₓψ`.
  - `Re::Number`: Reynolds number.
  - `faces::NTuple{2, Tuple{VorticityFace, VorticityFace}}`: Faces `faces[axis] = (left, right)`.
  - `vmax::Float64`: Characteristic speed of the flow, used by the CFL check.
  - `wall_formula::Symbol`: Wall vorticity formula, `:jensen` (second order, as in Kiffner & Jaksch, 2023) or `:thom`
    (first order).
  - `stencil_order::Int`: Order of the finite difference stencils (only 2 is supported).
  - `maxdim::Int`: Maximum bond dimension allowed throughout the simulation.
  - `cutoff::Float64`: SVD cutoff performed throughout the simulation.

See also the boundary-condition tutorial [Vorticity–streamfunction formulation](@ref bc_vorticity_streamfunction).
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
        INSVorticityExperiment(grid::Grid{N}, omega_field::ScalarField{N}, psi_field::ScalarField{N}, Re::Number,
                               faces::NTuple{2,Tuple{VorticityFace,VorticityFace}}; wall_formula::Symbol=:jensen,
                               stencil_order::Int=2, maxdim::Int=100, cutoff::Float64=1e-8,
                               vmax::Union{Nothing,Real}=nothing) where N

    Build an [`INSVorticityExperiment`](@ref). The constructor checks that the problem is two-dimensional, that the
    stencils are second order, that both fields share their site indices, that the boundary conditions of both fields
    match the faces, that periodic faces come in pairs and that every face profile has one value per node along the
    face, and it calls [`check_kolmogorov_scale`](@ref). Without `vmax` and with every face at rest, it logs at
    `@info` level that it uses ``v_\\max = 1``.

    # Throws
    - `ArgumentError`: if any of the checks above fails, or for an unknown `wall_formula`.

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
                @info "Every face is at rest: using vmax = 1 for the CFL check. Pass an explicit vmax to INSVorticityExperiment."
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
    INSVorticityExperiment(grid::Grid{N}, omega_field::ScalarField{N}, psi_field::ScalarField{N}, Re::Number,
                           U_lid::Real; kwargs...) where N

Build the lid-driven cavity, the experiment with the faces [`cavity_faces`](@ref)`(U_lid)`: build `grid` with
[`interior_grid`](@ref) and give every side of both fields a `DirichletBC`.

# Examples

```julia
arch = QTTArchitecture((6, 6))
wall = BoundaryCond1d(DirichletBC(0.0), DirichletBC(0.0))
walls = BoundaryConditions(wall, wall)
sites = siteinds(arch)
omega = ScalarField(zeros_mps(sites), arch, walls)
experiment = INSVorticityExperiment(interior_grid(arch, (1.0, 1.0)), omega, ScalarField(sites, arch, walls), 100.0, 1.0)
```
"""
function INSVorticityExperiment(
    grid::Grid{N}, omega_field::ScalarField{N}, psi_field::ScalarField{N}, Re::Number, U_lid::Real; kwargs...
) where {N}
    INSVorticityExperiment(grid, omega_field, psi_field, Re, cavity_faces(U_lid); kwargs...)
end

"""
    wall_vorticity_operator(field::ScalarField{N}, grid::Grid{N}; formula::Symbol=:jensen, cutoff::Float64=1e-15,
                            maxdim::Int=typemax(Int64)) where N -> Union{MPO,Nothing}
    wall_vorticity_operator(field::ScalarField{N}, grid::Grid{N}, walls::NTuple{N,NTuple{2,Bool}};
                            formula::Symbol=:jensen, cutoff::Float64=1e-15, maxdim::Int=typemax(Int64)) where N

Build the MPO ``S`` of the part of the ghost contribution of the wall vorticity to ``\\Delta\\omega`` that depends on
``\\psi``. The truncated Laplacian of the vorticity misses the wall value ``\\omega_\\text{wall}/h^2`` on the row next
to each wall, and the wall vorticity formula gives it in terms of ``\\psi``, so on that row

```math
(S \\boldsymbol{\\psi})_1 = \\frac{\\omega_\\text{wall}}{h^2} = \\frac{c_1 \\psi_1 + c_2 \\psi_2}{h^4},
```

for a stationary wall with ``\\psi_\\text{wall} = 0``, where ``(c_1, c_2)`` are the coefficients of `formula`
([`wall_vorticity_coefficients`](@ref)); the static rest (moving walls, ``\\psi_\\text{wall} \\ne 0``) is `ghost_omega`.
``S`` is the sum of the one-dimensional [`onedim_wall_vorticity_correction`](@ref) operators lifted to ``N``
dimensions, so a node next to two walls receives both contributions. The walls are the `DirichletBC` sides of
`field`, or the sides flagged in `walls[axis] = (left, right)`. Returns `nothing` if there is no wall.

See also the boundary-condition tutorial [Vorticity–streamfunction formulation](@ref bc_vorticity_streamfunction).
"""
function wall_vorticity_operator(
    field::ScalarField{N}, grid::Grid{N}; formula::Symbol=:jensen, cutoff::Float64=1e-15, maxdim::Int=typemax(Int64)
) where {N}
    walls = ntuple(i -> dirichlet_flags(field.bcu[i]), N)
    return wall_vorticity_operator(field, grid, walls; formula=formula, cutoff=cutoff, maxdim=maxdim)
end

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

"""
    FACE_SIDES

The four sides `(axis, side)` of a two-dimensional domain, in the order in which [`VorticityCache`](@ref) sums the
static boundary terms.
"""
const FACE_SIDES = ((1, :left), (1, :right), (2, :left), (2, :right))

"""
    face_at(faces, axis::Int, side::Symbol) -> VorticityFace

Return the face of `faces` on the `side` (`:left` or `:right`) of `axis`.
"""
face_at(faces, axis::Int, side::Symbol) = faces[axis][side == :left ? 1 : 2]

"""
    tangential_sign(axis::Int, side::Symbol) -> Float64

Return the sign of the moving-wall term of the wall vorticity, from ``\\omega = \\partial_x v - \\partial_y u`` with the
tangential velocity signed along the ``+`` direction of the other axis: ``+1`` on the top (axis 2, right) and left
(axis 1, left) walls, and ``-1`` on the bottom and right walls.
"""
tangential_sign(axis::Int, side::Symbol) = ((axis == 2) == (side == :right)) ? 1.0 : -1.0

"""
    ghost_psi(face::VorticityFace)

Return the value of ``\\psi`` on a ghost face (a constant, or a profile for an [`InflowFace`](@ref)), or `nothing` for
an outflow or periodic face.
"""
ghost_psi(face::Union{NoSlipWall,SlipWall,InflowFace}) = face.psi
ghost_psi(::VorticityFace) = nothing

"""
    ghost_omega(face::NoSlipWall, axis::Int, side::Symbol, h::Real, formula::Symbol)
    ghost_omega(face::InflowFace, args...)
    ghost_omega(face::VorticityFace, args...)

Return the static part of the ghost value of ``\\omega`` on `face`, or `nothing`. On a no-slip wall, the ghost
vorticity is

```math
\\omega_\\text{wall} = \\frac{c_1 \\psi_1 + c_2 \\psi_2}{h^2} - (c_1 + c_2) \\frac{\\psi_w}{h^2}
    + s\\, c_U \\frac{U_t}{h},
```

where ``\\psi_w`` is the value of ``\\psi`` on the wall, ``U_t`` its tangential velocity, ``s`` the
[`tangential_sign`](@ref) and ``(c_1, c_2, c_U)`` the coefficients of `formula`. The first term depends on ``\\psi`` and
is [`wall_vorticity_operator`](@ref); this method returns the other two. An inflow returns its prescribed `omega`, and
slip walls (``\\omega = 0``), outflows and periodic faces return `nothing`.
"""
function ghost_omega(face::NoSlipWall, axis::Int, side::Symbol, h::Real, formula::Symbol)
    c1, c2, cU = wall_vorticity_coefficients(formula)
    return tangential_sign(axis, side) * cU .* face.velocity ./ h .- (c1 + c2) * face.psi / h^2
end
ghost_omega(face::InflowFace, args...) = face.omega
ghost_omega(::VorticityFace, args...) = nothing # Slip walls (ω = 0), outflow and periodic faces

"""
    lift_sum(experiment::INSVorticityExperiment, field::ScalarField, derivative_order::Int, terms;
             cutoff::Float64=1e-15, maxdim::Int=typemax(Int64)) -> Union{MPS,Nothing}

Return the sum of the [`face_dirichlet_lift`](@ref)s of `field` for the derivative of order `derivative_order`, one
per `(axis, side, value)` in `terms`, with the grid and stencil order of `experiment`; terms whose value is `nothing`
or zero are skipped, and the result is `nothing` if every term vanishes.
"""
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
    outflow_advection_closure(ops::DifferentialOpsScalar{2}, field::ScalarField{2}, grid::Grid{2}, faces;
                              cutoff::Float64=1e-15, maxdim::Int=typemax(Int64)) -> DifferentialOpsScalar{2}

Return `ops` with the outflow rows of its first derivatives replaced by the first-order upwind difference,
``(f_2 - f_1)/h`` on a left outflow and ``(f_n - f_{n-1})/h`` on a right one. The mirrored Neumann closure makes those
rows zero, so the advection would never carry the vorticity out of the domain. This assumes that the flow leaves
through every [`OutflowFace`](@ref). The second derivatives are unchanged. Returns `ops` itself if there is no outflow
face.

See also the boundary-condition tutorial [Vorticity–streamfunction formulation](@ref bc_vorticity_streamfunction).
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
    VorticityCache(experiment::INSVorticityExperiment{N}; cutoff::Float64=1e-15, maxdim::Int=typemax(Int64)) where N

Operators and static boundary terms of the vorticity–streamfunction scheme that depend only on the experiment, built
once per run.

Every face fixes ``\\psi`` and ``\\omega`` on its ghost nodes, and the truncated operators take those ghost values as
zero. Ghost values that depend on ``\\psi`` (the wall vorticity) become the operator `wall_op`, and static ghost values
become lifts that are added to the right-hand sides: `psi_lift` and `velocity_lift` from the face values of ``\\psi``,
`wall_source` from moving walls, walls with ``\\psi \\ne 0`` and inflow vorticity, and `advection_lift` from the
inflow fluxes. The operators and lifts are built with their own tight `cutoff`, not with `experiment.cutoff`, so that
the boundary entries are not truncated away.

# Fields

  - `diff_ops`: First and second derivative operators of the stream function (`DifferentialOpsScalar`), for the
    velocity.
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
  - `wall_source`: Static ghost contribution of the faces to `Δω`: moving walls, walls with `ψ ≠ 0` and inflow
    vorticity.
  - `psi_lift`: Ghost contribution of the face values of ψ to `Δψ`.
  - `velocity_lift`: Ghost contribution of the face values of ψ to `(∂ₓψ, ∂ᵧψ)`.
  - `advection_lift`: Ghost advective flux of the inflow faces.
  - `regularize::Bool`: Whether ψ has no ghost face, so that the Poisson system is singular and is pinned to zero mean.

The static terms are `nothing` when they vanish.

See also the boundary-condition tutorial [Vorticity–streamfunction formulation](@ref bc_vorticity_streamfunction).
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
    poisson_rhs(experiment::INSVorticityExperiment, cache::VorticityCache, omega::MPS) -> MPS

Return the right-hand side ``W(\\boldsymbol{\\omega} + \\boldsymbol{\\ell})`` of the stream-function Poisson system
``A\\boldsymbol{\\psi} = W(\\boldsymbol{\\omega} + \\boldsymbol{\\ell})``, which discretizes ``-\\Delta\\psi = \\omega``
with the truncated Laplacian completed by the face values of ``\\psi`` (``\\boldsymbol{\\ell}`` = `cache.psi_lift`) and
weighted on the outflow rows (``W``, see [`symmetrize_rhs`](@ref)). If `cache.regularize` (no ghost face), the node
mean is removed with [`remove_mean`](@ref), so that the singular system is consistent. For four walls with
``\\psi = 0`` (the lid-driven cavity), it is the vorticity itself.

See also the boundary-condition tutorial [Vorticity–streamfunction formulation](@ref bc_vorticity_streamfunction).
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
    poisson_streamfunction(experiment::INSVorticityExperiment{N}, cache::VorticityCache, omega::MPS,
                           solver::AbstractLinearSolver) where N -> Tuple{MPS,Float64}

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

See also [`VorticityCache`](@ref), [`streamfunction_velocity`](@ref).
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
    streamfunction_velocity(cache::VorticityCache, psi::MPS; cutoff::Float64=1e-15,
                            maxdim::Int=typemax(Int64)) -> Tuple{MPS,MPS}

Return the velocity ``(u, v) = (\\partial_y \\psi, -\\partial_x \\psi)`` of the stream function `psi` on the stored
nodes, with the central differences completed by the face values of ``\\psi`` (`cache.velocity_lift`). Without face
values (e.g. four walls with ``\\psi = 0``), the velocity is exactly discretely divergence-free, since the
one-dimensional difference operators of different axes commute:
``\\partial_x u + \\partial_y v = (\\partial_x \\partial_y - \\partial_y \\partial_x)\\psi = 0``.

See also the boundary-condition tutorial [Vorticity–streamfunction formulation](@ref bc_vorticity_streamfunction).
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
    wall_vorticity_source(cache::VorticityCache, psi::MPS; cutoff::Float64=1e-15,
                          maxdim::Int=typemax(Int64)) -> Union{MPS,Nothing}

Return the ghost contribution ``\\mathbf{s} = S\\boldsymbol{\\psi} + \\boldsymbol{\\ell}`` of the face vorticity to the
Laplacian of the vorticity on the nodes next to the ghost faces, i.e. the term that completes the truncated Laplacian
``\\Delta\\omega`` with the wall values given by the wall vorticity formula (``S\\boldsymbol{\\psi}``, see
[`wall_vorticity_operator`](@ref)) and with the static ghost values ``\\boldsymbol{\\ell}`` = `cache.wall_source`
(moving walls, walls with ``\\psi \\ne 0``, inflow vorticity). Returns `nothing` if there is no such contribution.

See also the boundary-condition tutorial [Vorticity–streamfunction formulation](@ref bc_vorticity_streamfunction).
"""
function wall_vorticity_source(cache::VorticityCache, psi::MPS; cutoff::Float64=1e-15, maxdim::Int=typemax(Int64))
    wall_term = isnothing(cache.wall_op) ? nothing : apply(cache.wall_op, psi; cutoff=cutoff, maxdim=maxdim)
    return _nothing_sum_handler(wall_term, cache.wall_source; cutoff=cutoff, maxdim=maxdim)
end

"""
    VorticityTransportRHS(experiment::INSExperimentBase, cache, u::MPS, v::MPS, source) <: AbstractRHS

Right-hand side of the vorticity transport equation in conservative form,

```math
\\partial_t \\omega = -\\partial_x(u\\omega) - \\partial_y(v\\omega) + \\frac{1}{Re}(\\Delta\\omega + s),
```

as a callable for the explicit time steppers, `rhs((ω,), t, stage) -> (dω/dt,)`, with the velocity ``(u, v)`` and the
wall vorticity source ``s`` frozen over the stages of the time integrator. The advective fluxes through the walls
vanish because the normal velocity does, so the truncated stencils (ghost value 0) give them exactly. The inflow fluxes
come from `cache.advection_lift`, and the outflow faces use the upwind rows of `cache.omega_diff_ops` (see
[`outflow_advection_closure`](@ref)). The diffusion uses the unweighted `cache.omega_laplacian`, since it is an
explicit term. The products ``u\\omega`` and ``v\\omega`` are computed by [`tt_mul`](@ref).

# Fields

  - `experiment::E`: The experimental setup (`Re`, `cutoff`, `maxdim`).
  - `cache::C`: Precomputed operators (`VorticityCache`).
  - `u::MPS`, `v::MPS`: Velocity components.
  - `source`: Wall vorticity source `s`, see [`wall_vorticity_source`](@ref). `nothing` if there is none.

See also the boundary-condition tutorial [Vorticity–streamfunction formulation](@ref bc_vorticity_streamfunction).
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
    vorticity_transport(experiment::INSVorticityExperiment{N}, cache::VorticityCache, omega::MPS, psi::MPS,
                        stepper::AbstractExplicitStepper, t::Real) where N -> MPS

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
    vorticity_step(experiment::INSVorticityExperiment{N}, cache::VorticityCache, omega::MPS,
                   solver::AbstractLinearSolver, stepper::AbstractExplicitStepper, t::Real) where N -> Tuple

Advance the vorticity `omega` by one time step `stepper.time_evol_kwargs.dt` with the vorticity-streamfunction
scheme: one stream function Poisson solve, then the explicit vorticity transport with the resulting velocity and
wall vorticity frozen over the stages of `stepper`. Does not modify `experiment`; storing the results is left to
the caller.

# Arguments

  - `experiment::INSVorticityExperiment{N}`: The experimental setup. Its last stream function (if any) is used as the
    Poisson initial guess.
  - `cache::VorticityCache`: Precomputed operators.
  - `omega::MPS`: Vorticity at time `t`.
  - `solver::AbstractLinearSolver`: Linear solver for the stream function Poisson equation.
  - `stepper::AbstractExplicitStepper`: Time integrator for the vorticity transport.
  - `t::Real`: Time at the start of the step.

# Returns

  - `(omega_next, psi, linsolve_error)`: The vorticity at `t + dt`, the stream function at `t` and the l2 residual norm
    of the Poisson solve.
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
    vorticity_evolve!(experiment::INSVorticityExperiment{N}, solver::AbstractLinearSolver,
                      stepper::AbstractExplicitStepper, save_every::Int=1, logger=nothing,
                      log_filename::String="vorticity_log") where N

Run the time evolution of `experiment` with the vorticity–streamfunction scheme, storing the vorticity and the stream
function in its fields.

We check the CFL condition ([`check_cfl`](@ref), with `experiment.vmax`), build the [`VorticityCache`](@ref) once, and
advance the last stored vorticity `stepper.time_evol_kwargs.nsteps` times with [`vorticity_step`](@ref). The fields
keep the initial vorticity and, for the steps `save_every`, `2 save_every`, … and the last step, which is always
stored, the vorticity at the end of the step and the stream function solved at its start; the last stored vorticity is
also the input of the next step. If `logger` is given, the fields and the residual norms of the stored stream-function
solves (`linsolve_errorlog`) are written to `log_filename` with [`log_data`](@ref).

# Arguments

  - `experiment::INSVorticityExperiment{N}`: the experiment, with its initial vorticity stored and its stream function
    empty.
  - `solver::AbstractLinearSolver`: the solver of the stream-function system (`ALS`, `ResidualALS`, `GlobalCG`, …).
  - `stepper::AbstractExplicitStepper`: the integrator of the transport (`ExplicitEuler`, `RK4`); its `TimeEvolArgs`
    set the time step and the number of steps.
  - `save_every::Int=1`: the interval, in steps, between stored states.
  - `logger=nothing`: a [`SimpleLogger`](@ref), or `nothing` to write nothing.
  - `log_filename::String="vorticity_log"`: the name of the JLD2 file in the directory of `logger`.

# Throws

  - `ArgumentError`: if the vorticity has no stored state, if the stream function has one, or if `save_every < 1`.

# Examples

```julia
stepper = RK4(; time_evol_kwargs=TimeEvolArgs(0.0, 1.0, 100), cutoff=1e-12, maxdim=32)
solver = …                                          # e.g. ALS or GlobalCG
vorticity_evolve!(experiment, solver, stepper, 10)  # stores the initial state and the steps 10, 20, …, 100
```
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

    save_every >= 1 || throw(ArgumentError("save_every must be at least 1, got $save_every"))
    # A new slot is pushed at the steps 1, 1 + save_every, …, which freezes the previous one (the initial state, then
    # the states of the steps save_every, 2 save_every, …). Between checkpoints the last slot is overwritten, so it
    # always holds the latest step, which is the input of the next one, and the last step is always stored
    save_checkpoints = 1:save_every:time_evol_kwargs.nsteps

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
            linsolve_errorlog[end] = linsolve_error
        end
    end

    if !isnothing(logger)
        log_data(logger, log_filename, "omega_field", experiment.omega_field)
        log_data(logger, log_filename, "psi_field", experiment.psi_field)
        log_data(logger, log_filename, "linsolve_errorlog", linsolve_errorlog)
    end
end

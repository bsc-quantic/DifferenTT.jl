"""
    AbstractPreconditioner

Supertype of the preconditioners accepted by [`linearsolve`](@ref): [`Preconditioner`](@ref), which stores an
approximate inverse ``M^{-1} \\approx A^{-1}`` as one MPO, and [`FactorizedPreconditioner`](@ref), which stores the
factor ``G_L`` of ``M^{-1} = G_L^\\top G_L``.
"""
abstract type AbstractPreconditioner end

"""
    Preconditioner(operator::MPO) <: AbstractPreconditioner

Preconditioner given by an approximate inverse ``M^{-1} \\approx A^{-1}`` stored as one MPO, with the site indices of
`A` at prime levels 0 (input) and 1 (output).

Globally, [`GlobalCG`](@ref) and [`GlobalGMRES`](@ref) apply it to the residual with `ITensorMPS.apply`, truncated as
the iterates. Locally, a DMRG-type solver projects it onto the sites of each local problem with the same basis
``U`` as the operator, so that the local Krylov solver uses ``U^\\top M^{-1} U`` (Sec. 2.1 of Torrente et al.,
*Factorized approximate inverse preconditioners in the tensor train format*). For CG, ``M^{-1}`` must be SPD. Note
that a truncated approximate inverse controls the norm error but not the sign of its eigenvalues, so it can be
indefinite (Fig. 3); [`FactorizedPreconditioner`](@ref) avoids this by construction.

The constructors [`rank1_preconditioner`](@ref), [`newtoniter_preconditioner`](@ref),
[`linsolve_preconditioner`](@ref), [`sincquadrature_preconditioner`](@ref) and
[`inverse_laplacian_preconditioner`](@ref) return a `Preconditioner`.

# Fields

  - `operator::MPO`: the approximate inverse ``M^{-1}``.
"""
struct Preconditioner <: AbstractPreconditioner
    operator::MPO
end

"""
    copy(P::Preconditioner)
    copy(P::FactorizedPreconditioner)

Return a preconditioner of the same type that wraps a copy of the MPO of `P`.
"""
Base.copy(preconditioner::Preconditioner) = Preconditioner(copy(preconditioner.operator))

"""
    FactorizedPreconditioner(G_L::MPO) <: AbstractPreconditioner

Preconditioner ``M^{-1} = G_L^\\top G_L`` stored through its factor ``G_L`` alone, as built by
[`fai_preconditioner`](@ref).

Since ``\\mathbf{v}^\\top G_L^\\top G_L \\mathbf{v} = \\|G_L \\mathbf{v}\\|^2 \\ge 0``, the preconditioner is positive
semidefinite by construction, and definite whenever ``G_L`` is non-singular, whatever the truncation of ``G_L``. We
never form the product ``G_L^\\top G_L``, which would square the bond dimension and whose truncation would lose this
property. Globally, we apply ``G_L`` and then its transpose ([`transpose_mpo`](@ref)), each truncated as the
iterates. Locally, a DMRG-type solver projects ``G_L`` alone and applies the two layers
``(U^\\top G_L^\\top U)(U^\\top G_L U)``, at a cost relative to an operator of bond dimension ``\\chi_A`` of
``1 + 2\\chi_{G_L}/\\chi_A`` (Sec. 4, Fig. 8 of Torrente et al.). The intermediate projection ``U U^\\top`` makes this
local preconditioner differ from ``U^\\top G_L^\\top G_L U``, but it is still positive semidefinite, and definite
whenever ``U^\\top G_L U`` is non-singular.

# Fields

  - `operator::MPO`: the factor ``G_L``, with prime levels 0 (input) and 1 (output).
"""
struct FactorizedPreconditioner <: AbstractPreconditioner
    operator::MPO
end

Base.copy(preconditioner::FactorizedPreconditioner) = FactorizedPreconditioner(copy(preconditioner.operator))

"""
    AbstractReducedPreconditioner

Supertype of the projections of a preconditioner onto the sites of a local problem: [`ReducedPreconditioner`](@ref)
and [`ReducedFactorizedPreconditioner`](@ref). Both wrap an `ITensorMPS.ProjMPO` of the preconditioner's MPO, whose
environments are updated with those of the operator during the sweeps, and act on a local tensor `v` through
`contract(P, v)` (primed output), `product(P, v)` and `P(v)` (unprimed output). `contract(P)` returns the dense local
preconditioner.
"""
abstract type AbstractReducedPreconditioner end

"""
    ReducedPreconditioner(reduced_op::ProjMPO) <: AbstractReducedPreconditioner

Projection ``U^\\top M^{-1} U`` of a [`Preconditioner`](@ref) onto the sites of the current local problem.
"""
struct ReducedPreconditioner <: AbstractReducedPreconditioner
    reduced_op::ProjMPO
end

function Base.copy(reduced_preconditioner::ReducedPreconditioner)
    ReducedPreconditioner(copy(reduced_preconditioner.reduced_op))
end
ITensors.contract(P::ReducedPreconditioner, v::ITensor) = ITensors.contract(P.reduced_op, v)
ITensors.contract(P::ReducedPreconditioner) = dense_projected_operator(P.reduced_op)
ITensorMPS.product(P::ReducedPreconditioner, v::ITensor) = ITensorMPS.product(P.reduced_op, v)
(P::ReducedPreconditioner)(v::ITensor) = ITensorMPS.product(P, v)

"""
    ReducedFactorizedPreconditioner(reduced_op::ProjMPO) <: AbstractReducedPreconditioner

Projection of a [`FactorizedPreconditioner`](@ref) onto the sites of the current local problem. It stores the
projection ``U^\\top G_L U`` of the factor alone, and applies the two layers ``(U^\\top G_L^\\top U)(U^\\top G_L U)``
by contracting the same environment and site tensors a second time on their output indices (Fig. 8 of Torrente et
al.).
"""
struct ReducedFactorizedPreconditioner <: AbstractReducedPreconditioner
    reduced_op::ProjMPO
end

function Base.copy(reduced_preconditioner::ReducedFactorizedPreconditioner)
    ReducedFactorizedPreconditioner(copy(reduced_preconditioner.reduced_op))
end

"""
    _itensor_map(P::AbstractProjMPO) -> Vector

Return the tensors of the projected operator `P` in contraction order: its left environment, its MPO tensors on the
current sites and its right environment (`OneITensor` for an empty environment).
"""
function _itensor_map(P::AbstractProjMPO)
    itensor_map = Union{ITensor,OneITensor}[lproj(P)]
    append!(itensor_map, P.H[site_range(P)])
    push!(itensor_map, rproj(P))
    return itensor_map
end

"""
    _swapprime(tens, args...)

`swapprime` that also accepts the `OneITensor` placeholder of an empty environment, which has no indices and is
returned unchanged.
"""
_swapprime(tens::ITensor, args...) = swapprime(tens, args...)
_swapprime(tens::OneITensor, args...) = tens

function ITensors.contract(P::ReducedFactorizedPreconditioner, v::ITensor)::ITensor
    Glv = contract(P.reduced_op, v)
    transpose_itensor_map = _itensor_map(P.reduced_op)
    if dim(first(transpose_itensor_map)) == 1
        reverse!(transpose_itensor_map)
    end
    GlTGlv = Glv
    for it in transpose_itensor_map
        GlTGlv *= it
    end
    return prime(GlTGlv)
end

function ITensors.contract(P::ReducedFactorizedPreconditioner)::ITensor
    contractedGL = dense_projected_operator(P.reduced_op)
    transpose_itensor_map = _itensor_map(P.reduced_op)

    transpose_itensor_map = [_swapprime(tensor, 0, 2) for tensor in transpose_itensor_map]

    if dim(first(transpose_itensor_map)) == 1
        reverse!(transpose_itensor_map)
    end

    GlTGl = contractedGL
    for it in transpose_itensor_map
        GlTGl *= it
    end
    swapprime!(GlTGl, 2, 1)
    return GlTGl
end

function ITensorMPS.product(P::ReducedFactorizedPreconditioner, v::ITensor)::ITensor
    Pv = contract(P, v)
    if order(Pv) != order(v)
        error("Order mismatch")
    end
    return noprime(Pv)
end

(P::ReducedFactorizedPreconditioner)(v::ITensor) = ITensorMPS.product(P, v)

"""
    ReducedPrecondLinearProblem(linear_problem::ReducedLinearProblem, preconditioner::ReducedPreconditioner)
        <: AbstractProjMPO

Local problem of a DMRG-type solve preconditioned by a [`Preconditioner`](@ref). It moves the projections of the
operator, the right-hand side and ``M^{-1}`` together along the sweeps (`position!`, `makeL!`, `makeR!`,
`set_nsite!`), so that [`KrylovUpdater`](@ref) can solve each local problem with ``U^\\top M^{-1} U``. Build it with
[`initialize_reducedprecondlinearproblem`](@ref).

# Fields

  - `linear_problem::ReducedLinearProblem`: the projected operator and right-hand side.
  - `preconditioner::ReducedPreconditioner`: the projected preconditioner.
"""
mutable struct ReducedPrecondLinearProblem <: AbstractProjMPO
    linear_problem::ReducedLinearProblem
    preconditioner::ReducedPreconditioner
end

"""
    ReducedFAIPrecondLinearProblem(linear_problem::ReducedLinearProblem,
                                   preconditioner::ReducedFactorizedPreconditioner) <: AbstractProjMPO

Local problem of a DMRG-type solve preconditioned by a [`FactorizedPreconditioner`](@ref). It tracks the projection
of the factor ``G_L`` only, which also serves for its transpose, so storing and updating it costs as much as for any
other MPO; each local application then costs ``1 + 2\\chi_{G_L}/\\chi_A`` times that of the operator (Sec. 4, Fig. 8
of Torrente et al.). Build it with [`initialize_reducedprecondlinearproblem`](@ref).

# Fields

  - `linear_problem::ReducedLinearProblem`: the projected operator and right-hand side.
  - `preconditioner::ReducedFactorizedPreconditioner`: the projected factor ``G_L``.
"""
mutable struct ReducedFAIPrecondLinearProblem <: AbstractProjMPO
    linear_problem::ReducedLinearProblem
    preconditioner::ReducedFactorizedPreconditioner
end

"""
    initialize_reducedprecondlinearproblem(operator::MPO, constant_term::MPS, preconditioner::Preconditioner)
        -> ReducedPrecondLinearProblem
    initialize_reducedprecondlinearproblem(operator::MPO, constant_term::MPS, preconditioner::FactorizedPreconditioner)
        -> ReducedFAIPrecondLinearProblem

Return the preconditioned local problem of the system `operator * x = constant_term`, with unpositioned projections,
for the sweeps of a DMRG-type solver.
"""
function initialize_reducedprecondlinearproblem(operator::MPO, constant_term::MPS, preconditioner::Preconditioner)
    linear_problem = ReducedLinearProblem(ProjMPO(operator), [ReducedConstantTerm(constant_term)])
    reducedMinv = ReducedPreconditioner(ProjMPO(preconditioner.operator))
    return ReducedPrecondLinearProblem(linear_problem, reducedMinv)
end

function initialize_reducedprecondlinearproblem(
    operator::MPO, constant_term::MPS, preconditioner::FactorizedPreconditioner
)
    linear_problem = ReducedLinearProblem(ProjMPO(operator), [ReducedConstantTerm(constant_term)])
    reducedGl = ReducedFactorizedPreconditioner(ProjMPO(preconditioner.operator))
    return ReducedFAIPrecondLinearProblem(linear_problem, reducedGl)
end

# The sweep interface of ITensorMPS (copy, nsite, set_nsite!, makeL!, makeR!, position!), forwarded to the projections
# of both the linear problem and the preconditioner, so that they always sit at the same sites
function Base.copy(reduced_problem::ReducedPrecondLinearProblem)
    return ReducedPrecondLinearProblem(copy(reduced_problem.linear_problem), copy(reduced_problem.preconditioner))
end

function ITensorMPS.nsite(reduced_problem::ReducedPrecondLinearProblem)
    return nsite(reduced_problem.linear_problem)
end

function ITensorMPS.set_nsite!(reduced_problem::ReducedPrecondLinearProblem, nsite)
    set_nsite!(reduced_problem.linear_problem, nsite)
    set_nsite!(reduced_problem.preconditioner.reduced_op, nsite)
    return reduced_problem
end

function ITensorMPS.makeL!(reduced_problem::ReducedPrecondLinearProblem, state::MPS, position::Int)
    makeL!(reduced_problem.linear_problem, state, position)
    makeL!(reduced_problem.preconditioner.reduced_op, state, position)
    return reduced_problem
end

function ITensorMPS.makeR!(reduced_problem::ReducedPrecondLinearProblem, state::MPS, position::Int)
    makeR!(reduced_problem.linear_problem, state, position)
    makeR!(reduced_problem.preconditioner.reduced_op, state, position)
    return reduced_problem
end

function ITensorMPS.position!(reduced_problem::ReducedPrecondLinearProblem, state::MPS, position::Int)
    position!(reduced_problem.linear_problem, state, position)
    position!(reduced_problem.preconditioner.reduced_op, state, position)
    return reduced_problem
end

function Base.copy(reduced_problem::ReducedFAIPrecondLinearProblem)
    return ReducedFAIPrecondLinearProblem(copy(reduced_problem.linear_problem), copy(reduced_problem.preconditioner))
end

function ITensorMPS.nsite(reduced_problem::ReducedFAIPrecondLinearProblem)
    return nsite(reduced_problem.linear_problem)
end

function ITensorMPS.set_nsite!(reduced_problem::ReducedFAIPrecondLinearProblem, nsite)
    set_nsite!(reduced_problem.linear_problem, nsite)
    set_nsite!(reduced_problem.preconditioner.reduced_op, nsite)
    return reduced_problem
end

function ITensorMPS.makeL!(reduced_problem::ReducedFAIPrecondLinearProblem, state::MPS, position::Int)
    makeL!(reduced_problem.linear_problem, state, position)
    makeL!(reduced_problem.preconditioner.reduced_op, state, position)
    return reduced_problem
end

function ITensorMPS.makeR!(reduced_problem::ReducedFAIPrecondLinearProblem, state::MPS, position::Int)
    makeR!(reduced_problem.linear_problem, state, position)
    makeR!(reduced_problem.preconditioner.reduced_op, state, position)
    return reduced_problem
end

function ITensorMPS.position!(reduced_problem::ReducedFAIPrecondLinearProblem, state::MPS, position::Int)
    position!(reduced_problem.linear_problem, state, position)
    position!(reduced_problem.preconditioner.reduced_op, state, position)
    return reduced_problem
end

"""
    dense_projected_operator(P::AbstractProjMPO) -> ITensor

Contract the projected operator `P` (left environment, the MPO tensors of its current sites and right environment)
into one ITensor, the dense local operator of the sweep at the current position.
"""
function dense_projected_operator(P::AbstractProjMPO)::ITensor
    itensor_map = Union{ITensor,OneITensor}[lproj(P)]
    append!(itensor_map, P.H[site_range(P)])
    push!(itensor_map, rproj(P))

    if dim(first(itensor_map)) == 1
        reverse!(itensor_map)
    end

    reduced_H = ITensor(true)
    for it in itensor_map
        reduced_H *= it
    end
    return reduced_H
end

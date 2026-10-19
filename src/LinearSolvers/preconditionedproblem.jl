using ITensorMPS: ITensorMPS, AbstractProjMPO, ReducedLinearProblem, ReducedConstantTerm
using ITensorMPS: ITensorMPS, set_nsite!, makeL!, makeR!, nsite, site_range, apply, lproj, rproj, position!
using ITensors: OneITensor

# For dispatch purposes

abstract type AbstractPreconditioner end

struct Preconditioner <: AbstractPreconditioner
    operator::MPO
end

Base.copy(preconditioner::Preconditioner) = Preconditioner(copy(preconditioner.operator))

struct FactorizedPreconditioner <: AbstractPreconditioner
    operator::MPO
end

Base.copy(preconditioner::FactorizedPreconditioner) = FactorizedPreconditioner(copy(preconditioner.operator))

abstract type AbstractReducedPreconditioner end

struct ReducedPreconditioner <: AbstractReducedPreconditioner
    reduced_op::ProjMPO
end

function Base.copy(reduced_preconditioner::ReducedPreconditioner)
    ReducedPreconditioner(copy(reduced_preconditioner.reduced_op))
end
ITensors.contract(P::ReducedPreconditioner, v::ITensor) = ITensors.contract(P.reduced_op, v)
ITensors.contract(P::ReducedPreconditioner) = ITensors.contract(P.reduced_op)
ITensorMPS.product(P::ReducedPreconditioner, v::ITensor) = ITensorMPS.product(P.reduced_op, v)
(P::ReducedPreconditioner)(v::ITensor) = ITensorMPS.product(P, v)

struct ReducedFactorizedPreconditioner <: AbstractReducedPreconditioner
    reduced_op::ProjMPO
end

function Base.copy(reduced_preconditioner::ReducedFactorizedPreconditioner)
    ReducedFactorizedPreconditioner(copy(reduced_preconditioner.reduced_op))
end

function _itensor_map(P::AbstractProjMPO)
    itensor_map = Union{ITensor,OneITensor}[lproj(P)]
    append!(itensor_map, P.H[site_range(P)])
    push!(itensor_map, rproj(P))
    return itensor_map
end

ITensors.swapprime(tens::OneITensor, args::Vararg{Any}) = tens

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
    contractedGL = contract(P.reduced_op)
    transpose_itensor_map = _itensor_map(P.reduced_op)

    transpose_itensor_map = [swapprime(tensor, 0, 2) for tensor in transpose_itensor_map]

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
    ReducedPrecondLinearProblem <: AbstractProjMPO

A projected linear problem that includes a preconditioner MPO alongside the operator and constant term.

Wraps a `ReducedLinearProblem` together with a `ProjMPO` preconditioner,
enabling preconditioned iterative solvers within the `alternating_update` sweep framework.

# Fields

  - `linear_problem::ReducedLinearProblem`: The underlying projected operator and RHS.
  - `preconditioner::ProjMPO`: The projected preconditioner operator.
"""
mutable struct ReducedPrecondLinearProblem <: AbstractProjMPO
    linear_problem::ReducedLinearProblem
    preconditioner::ReducedPreconditioner
end

"""
    ReducedFAIPrecondLinearProblem <: AbstractProjMPO

A projected linear problem that includes a factorized preconditioner M⁻¹ ≈ Gl^T * Gl alongside the operator and constant term.
This struct only tracks the projection of G_L, which can be reused for its transpose too

Wraps a `ReducedLinearProblem` together with a `ProjMPO` for the preconditioner Gl factor,
enabling preconditioned iterative solvers within the `alternating_update` sweep framework.

# Fields

  - `linear_problem::ReducedLinearProblem`: The underlying projected operator and RHS.
  - `preconditioner::ProjMPO`: The projected preconditioner factor Gl operator.
"""
mutable struct ReducedFAIPrecondLinearProblem <: AbstractProjMPO
    linear_problem::ReducedLinearProblem
    preconditioner::ReducedFactorizedPreconditioner
end

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

function initialize_reducedprecondlinearproblem(operator::MPO, constant_terms::Vector{MPS}, Minv::Preconditioner)
    error("Not implemented yet")
    # # The residual in this case would be Σbᵢ which would have an absurd bond dimension, so...
    # linear_problem = ReducedLinearProblem(ProjMPO(operator), ReducedConstantTerm.(constant_terms))
    # projMinv = ProjMPO(Minv)
    # return ReducedPrecondLinearProblem(linear_problem, projMinv)
end

function initialize_reducedprecondlinearproblem(
    operator::MPO, constant_terms::Vector{MPS}, Gl::FactorizedPreconditioner
)
    error("Not implemented yet")
    # # The residual in this case would be Σbᵢ which would have an absurd bond dimension, so...
    # linear_problem = ReducedLinearProblem(ProjMPO(operator), ReducedConstantTerm.(constant_terms))
    # projGl = ProjMPO(Gl)
    # return ReducedPrecondLinearProblem(linear_problem, projGl)
end

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

function ITensors.contract(P::AbstractProjMPO)::ITensor
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

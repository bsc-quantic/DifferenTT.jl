using ITensorMPS: operator, constant_term, contract

ITensors.contract(P::ReducedPreconditioner, v::ITensor) = ITensors.contract(P.reduced_op, v)
ITensors.contract(P::ReducedPreconditioner) = ITensors.contract(P.reduced_op)
ITensorMPS.product(P::ReducedPreconditioner, v::ITensor) = ITensorMPS.product(P.reduced_op, v)
(P::ReducedPreconditioner)(v::ITensor) = ITensorMPS.product(P, v)

function _itensor_map(P::AbstractProjMPO)
    itensor_map = Union{ITensor,OneITensor}[lproj(P)]
    append!(itensor_map, P.H[site_range(P)])
    push!(itensor_map, rproj(P))
    return itensor_map
end

ITensors.swapprime(tens::OneITensor, args::Vararg{Any}) = tens

# This ends up not being used
function _transpose_itensor_map(P::AbstractProjMPO)
    transpose_itensor_map = Union{ITensor,OneITensor}[swapprime(dag(lproj(P)), 0, 1)]
    for j in site_range(P)
        push!(transpose_itensor_map, swapprime(dag(P.H[j]), 0, 1))
    end
    push!(transpose_itensor_map, swapprime(dag(rproj(P)), 0, 1))
    return transpose_itensor_map
end

# TODO If we were to contract the two layers all at once we would need to implement double prime or something like that
# in order for both sides of the 2-layer structure not to contract with each other too

# TODO Optimize contraction order by reordering ITensorMap through ITensor usage of TensorOperations.jl for optimizing contraction paths
# using ITensors
# using TensorOperations 
# result = contract(tensor_vector; sequence = ITensors.optimal_contraction_sequence(tensor_vector))

function ITensors.contract(P::ReducedFactorizedPreconditioner, v::ITensor)::ITensor
    Glv = contract(P.reduced_op, v)
    # Since Glv has primed indices (it is built by contract, not product that unprimes the output)
    # the regular itensormap will already apply GlT, as it will contract via its primed indices and
    # leave unprimed outputs (that is why we prime them by hand at the end, to follow contract vs product convention)
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

    # We move the output to primelevel=2 to avoid computing the trace over the whole thing. otherwise
    # the output unprimed legs of GlT would be contracted with the input ones from Gl, non primed.
    transpose_itensor_map = [swapprime(tensor, 0, 2) for tensor in transpose_itensor_map]

    if dim(first(transpose_itensor_map)) == 1
        reverse!(transpose_itensor_map)
    end

    GlTGl = contractedGL
    for it in transpose_itensor_map
        GlTGl *= it
    end

    # To finish things off, once the joined primelevel=1 legs have been contracted, we 
    # move the primelevel=2 output to primelevel=1 to follow the usual convention for matrices/operators
    swapprime!(GlTGl, 2, 1)
    return GlTGl
end

function product(P::ReducedFactorizedPreconditioner, v::ITensor)::ITensor
    Pv = contract(P, v)
    if order(Pv) != order(v)
        error(
            string(
                "The order of the ProjMPO-ITensor product P*v is not equal to the order of the ITensor v, ",
                "this is probably due to an index mismatch.\nCommon reasons for this error: \n",
                "(1) You are trying to multiply the ProjMPO with the $(nsite(P.reduced_op))-site wave-function at the wrong position.\n",
                "(2) `orthogonalize!` was called, changing the MPS without updating the ProjMPO.\n\n",
                "P*v inds: $(inds(Pv)) \n\n",
                "v inds: $(inds(v))",
            ),
        )
    end
    return noprime(Pv)
end

(P::ReducedFactorizedPreconditioner)(v::ITensor) = product(P, v)

"""
    ITensors.contract(P::AbstractProjMPO) -> ITensor

Contract the projected MPO environments and operator tensors into a single dense ITensor.

Contracts the left projection, the MPO tensors in the active site range, and the right projection
into a single ITensor representing the reduced operator matrix. Useful for computing condition
numbers or converting the local problem to a dense matrix.
"""
function ITensors.contract(P::AbstractProjMPO)::ITensor
    itensor_map = Union{ITensor,OneITensor}[lproj(P)]
    append!(itensor_map, P.H[site_range(P)])
    push!(itensor_map, rproj(P))

    # Reverse the contraction order of the map if
    # the first tensor is a scalar (for example we
    # are at the left edge of the system)
    if dim(first(itensor_map)) == 1
        reverse!(itensor_map)
    end

    # Apply the map
    reduced_H = ITensor(true)
    for it in itensor_map
        reduced_H *= it
    end
    return reduced_H
end

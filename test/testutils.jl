"""
Shared helpers and fixtures for the DifferenTT test suite.

Include it from a test file with

    include(joinpath(@__DIR__, "..", "testutils.jl"))
    using .TestUtils

Every randomized call in the tests receives `rng()`, so that each file is reproducible on its own.
"""
module TestUtils

using DifferenTT
using ITensors
using ITensorMPS
using LinearAlgebra
using Logging
using Random
using StableRNGs

export rng, dense, relres, mps_equal, quietly, tt_laplacian, qtt_laplacian

"""
Return a stable, seeded random number generator for reproducible tests.
"""
rng(seed::Integer=1234) = StableRNG(seed)

"""
Return the dense array of an MPS or MPO, with site 1 as the most significant index.
"""
dense(x::MPS) = DifferenTT.contract_mps(x)
dense(A::MPO) = DifferenTT.contract_mpo(A)

"""
Return the relative residual ``\\|A x - b\\| / \\|b\\|``, computed with dense arrays.
"""
relres(A::MPO, x::MPS, b::MPS) = norm(dense(A) * dense(x) - dense(b)) / norm(dense(b))

"""
    mps_equal(a, b)

Return whether two MPS or MPO have the same length and identical tensors, site by site. Tuples of MPS, as stored by a
`VectorField`, are compared component by component.
"""
function mps_equal(a::ITensorMPS.AbstractMPS, b::ITensorMPS.AbstractMPS)
    length(a) == length(b) && all(a[i] == b[i] for i in eachindex(a))
end
mps_equal(a::Tuple, b::Tuple) = length(a) == length(b) && all(map(mps_equal, a, b))

"""
Run `f()` with all log messages discarded.
"""
quietly(f) = with_logger(f, NullLogger())

function _dirichlet_laplacian(arch, N, stencil_order)
    sites = siteinds(arch)
    bc = quietly(() -> BoundaryConditions(ntuple(_ -> BoundaryCond1d(DirichletBC(0.0), DirichletBC(0.0)), N)...))
    grid = Grid(arch, ntuple(_ -> 1.0, N))
    A = -1.0 * symmetrized_laplacian(ScalarField(sites, arch, bc), grid; stencil_order=stencil_order)
    return A, sites, arch, grid
end

"""
    tt_laplacian(dims::NTuple{N,Int}; stencil_order=2) -> (A, sites, arch, grid)

Return the SPD operator ``-\\Delta`` with homogeneous Dirichlet conditions on an `n₁ × … × n_N` TT grid of unit box.
"""
function tt_laplacian(dims::NTuple{N,Int}; stencil_order::Int=2) where {N}
    _dirichlet_laplacian(TTArchitecture(dims), N, stencil_order)
end

"""
    qtt_laplacian(nsites::NTuple{N,Int}; stencil_order=2) -> (A, sites, arch, grid)

Return the SPD operator ``-\\Delta`` with homogeneous Dirichlet conditions on a QTT grid of `2^nsites[i]` points per
dimension and unit box.
"""
function qtt_laplacian(nsites::NTuple{N,Int}; stencil_order::Int=2) where {N}
    _dirichlet_laplacian(QTTArchitecture(nsites), N, stencil_order)
end

end

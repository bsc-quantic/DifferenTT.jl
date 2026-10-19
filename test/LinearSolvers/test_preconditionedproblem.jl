using DifferenTT
using Test
using ITensors
using ITensorMPS
using LinearAlgebra

@testset "preconditionedproblem.jl" begin
    n = 4
    sites = siteinds("S=1/2", n)
    A = MPO(sites, "Id")
    b = random_mps(sites; linkdims=2)

    @testset "ReducedPreconditioner operations" begin
        M_prec = rank1_preconditioner(A)
        reduced_prob1 = DifferenTT.initialize_reducedprecondlinearproblem(A, b, M_prec)

        # Test basic properties
        @test reduced_prob1 isa DifferenTT.ReducedPrecondLinearProblem
        @test ITensorMPS.nsite(reduced_prob1) == 2

        # Test set_nsite!
        ITensorMPS.set_nsite!(reduced_prob1, 1)
        @test ITensorMPS.nsite(reduced_prob1) == 1
        ITensorMPS.set_nsite!(reduced_prob1, 2)

        # Test positioning and makeL! / makeR!
        ITensorMPS.position!(reduced_prob1, b, 2) # position at site 2, nsite=2 -> sites 2,3

        red_M1 = reduced_prob1.preconditioner
        @test red_M1 isa DifferenTT.ReducedPreconditioner

        # Copy test
        red_M1_copy = copy(red_M1)
        @test red_M1_copy isa DifferenTT.ReducedPreconditioner
        @test red_M1_copy !== red_M1

        # Create a compatible ITensor to apply the reduced preconditioner on
        v1 = b[2] * b[3]

        # Test contract, product, and call methods
        out_contract1 = ITensors.contract(red_M1, v1)
        out_product1 = ITensorMPS.product(red_M1, v1)
        out_call1 = red_M1(v1)

        @test out_contract1 isa ITensor
        @test out_product1 isa ITensor
        @test out_call1 isa ITensor

        # Check that the order is conserved after product / call
        @test order(out_product1) == order(v1)
        @test order(out_call1) == order(v1)

        # Product and call should be mathematically equivalent
        @test norm(out_product1 - out_call1) ≈ 0.0 atol=1e-12

        # Contract retains prime levels, so it shouldn't equal out_product directly but be related by unpriming
        @test norm(out_product1 - noprime(out_contract1)) ≈ 0.0 atol=1e-12
    end

    @testset "ReducedFactorizedPreconditioner operations" begin
        solver = ALS(sweep_kwargs=DifferenTT.SweepingArgs(nsweeps=1, maxdim=2, cutoff=1e-8))
        M_fai = fai_preconditioner(A; μ=0.5, solver, D_maxdim=2, D_cutoff=1e-8)

        reduced_prob2 = DifferenTT.initialize_reducedprecondlinearproblem(A, b, M_fai)

        # Test basic properties
        @test reduced_prob2 isa DifferenTT.ReducedFAIPrecondLinearProblem
        @test ITensorMPS.nsite(reduced_prob2) == 2

        # Test set_nsite!
        ITensorMPS.set_nsite!(reduced_prob2, 1)
        @test ITensorMPS.nsite(reduced_prob2) == 1
        ITensorMPS.set_nsite!(reduced_prob2, 2)

        # Test positioning and makeL! / makeR!
        ITensorMPS.position!(reduced_prob2, b, 2)

        red_M2 = reduced_prob2.preconditioner
        @test red_M2 isa DifferenTT.ReducedFactorizedPreconditioner

        # Copy test
        red_M2_copy = copy(red_M2)
        @test red_M2_copy isa DifferenTT.ReducedFactorizedPreconditioner
        @test red_M2_copy !== red_M2

        # Apply tests
        v2 = b[2] * b[3]

        out_contract2 = ITensors.contract(red_M2, v2)
        out_product2 = ITensorMPS.product(red_M2, v2)
        out_call2 = red_M2(v2)

        @test out_contract2 isa ITensor
        @test out_product2 isa ITensor
        @test out_call2 isa ITensor

        # Check that the order is conserved after product / call
        @test order(out_product2) == order(v2)
        @test order(out_call2) == order(v2)

        # Product and call should be mathematically equivalent
        @test norm(out_product2 - out_call2) ≈ 0.0 atol=1e-12

        # Contract retains prime levels, check equality with unpriming
        @test norm(out_product2 - noprime(out_contract2)) ≈ 0.0 atol=1e-12

        # Test full operator matrix contraction (Dense)
        dense_M2 = ITensors.contract(red_M2)
        @test dense_M2 isa ITensor

        # Applying the dense matrix to the vector should give the same result as the structured contraction
        out_dense_applied = dense_M2 * v2
        @test norm(noprime(out_dense_applied) - out_product2) ≈ 0.0 atol=1e-12
    end
end

include(joinpath(@__DIR__, "..", "testutils.jl"))
using .TestUtils

# The local preconditioners against their definitions (Sec. 4 of Torrente et al.): with U the basis of the projection
# built from the other tensors of the state, a Preconditioner applies Uᵀ M⁻¹ U, and a FactorizedPreconditioner the
# two-layer (UᵀGᵀU)(UᵀGU) of Fig. 8, which is not Uᵀ GᵀG U
@testset "Local preconditioners against the dense projections" begin
    sites = siteinds(2, 4)
    A = MPO(sites, "Id")
    b = random_mps(rng(1), sites; linkdims=2)
    x = random_mps(rng(2), sites; linkdims=3)
    M = random_mpo(rng(3), sites) + random_mpo(rng(4), sites)
    Mfull = prod(M)

    # U v and Uᵀ w for the 2-site projection at sites 2 and 3
    embed(v) = x[1] * v * x[4]
    project(w) = w * x[1] * x[4]
    apply_full(Op, v) = project(noprime(Op * embed(v)))
    v = random_itensor(rng(5), inds(x[2] * x[3]))

    P = DifferenTT.initialize_reducedprecondlinearproblem(A, b, Preconditioner(M))
    ITensorMPS.position!(P, x, 2)
    @test ITensorMPS.product(P.preconditioner, v) ≈ apply_full(Mfull, v) rtol = 1e-12

    F = DifferenTT.initialize_reducedprecondlinearproblem(A, b, FactorizedPreconditioner(M))
    ITensorMPS.position!(F, x, 2)
    two_layer = apply_full(swapprime(Mfull, 0, 1), apply_full(Mfull, v))
    @test ITensorMPS.product(F.preconditioner, v) ≈ two_layer rtol = 1e-12
    # The dense local factorized preconditioner is the same operator
    @test noprime(ITensors.contract(F.preconditioner) * v) ≈ two_layer rtol = 1e-12
    # It differs from the projection of the full GᵀG, since UUᵀ ≠ I in between
    @test !isapprox(two_layer, project(noprime(swapprime(Mfull, 0, 1) * noprime(Mfull * embed(v)))); rtol=1e-3)
end

@testset "makeL!, makeR! and copy of the preconditioned problems" begin
    sites = siteinds(2, 4)
    A = MPO(sites, "Id")
    b = random_mps(rng(1), sites; linkdims=2)
    x = random_mps(rng(2), sites; linkdims=3)
    M = random_mpo(rng(3), sites)
    for P in (Preconditioner(M), FactorizedPreconditioner(M))
        problem = DifferenTT.initialize_reducedprecondlinearproblem(A, b, P)
        ITensorMPS.makeL!(problem, x, 2)
        ITensorMPS.makeR!(problem, x, 3)
        @test problem.linear_problem.reduced_operator.lpos == 2
        @test problem.preconditioner.reduced_op.lpos == 2
        @test problem.linear_problem.reduced_operator.rpos == 3
        @test problem.preconditioner.reduced_op.rpos == 3
        copied = copy(problem)
        @test copied isa typeof(problem)
        @test copied.preconditioner !== problem.preconditioner

        P_copy = copy(P)
        @test P_copy isa typeof(P)
        @test P_copy.operator !== P.operator
        @test todense(P_copy.operator) == todense(P.operator)
    end
end
